"""Enhanced critic -> fixer -> patch -> rerun -> evaluate loop, v2.

Differences from experiments/critic_fixer_loop.py (F10):
  - Critic sees the FULL 30-chunk prediction set (correct AND wrong), not just mismatches --
    so it can see what NOT to break, not only what's failing.
  - Critic sees the round-by-round HISTORY of this loop: each prior patch, the prompt
    snapshot before/after it, and the mechanical before/after score -- not just the current
    prompt in isolation.
  - Critic is explicitly told: prior patches and diagnoses (including ones already baked
    into the current prompt from earlier sessions) are HYPOTHESES, not ground truth --
    re-evaluate them against current evidence and feel free to recommend revisiting one.
  - Orchestrator role is unchanged from F10: apply the fixer's patch by exact match only,
    mechanical revert on non-improvement, no independent content judgment.

Usage:
    .venv/bin/python experiments/critic_fixer_loop_v2.py score <pass>
    .venv/bin/python experiments/critic_fixer_loop_v2.py critic <pass> <round_num>
    .venv/bin/python experiments/critic_fixer_loop_v2.py fixer <pass> <pattern_file> <round_num>
    .venv/bin/python experiments/critic_fixer_loop_v2.py log-round <pass> <round_num> <patch_summary> <kept|reverted>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tag_three_pass import GLOBAL_PASS_SYSTEM_PROMPT, LOCAL_PASS_SYSTEM_PROMPT, SCENE_PASS_SYSTEM_PROMPT
from tag_white_nights import client

from iterate_pass import PASS_FIELDS, run_pass, score_pass
from run_first30_eval import load_gold

LOG_DIR = ROOT / "experiments" / "critic_fixer_logs"
LOG_DIR.mkdir(exist_ok=True)
HISTORY_PATH = LOG_DIR / "v2_history.json"

PASS_PROMPTS = {
    "local": ("LOCAL_PASS_SYSTEM_PROMPT", LOCAL_PASS_SYSTEM_PROMPT),
    "scene": ("SCENE_PASS_SYSTEM_PROMPT", SCENE_PASS_SYSTEM_PROMPT),
    "global": ("GLOBAL_PASS_SYSTEM_PROMPT", GLOBAL_PASS_SYSTEM_PROMPT),
}

CRITIC_SYSTEM = """You are a diagnostic critic reviewing the FULL prediction set for ONE pass of a
multi-pass literary annotation pipeline, against a hand-built gold standard, across the
full 30-chunk gold set at once.

You will be shown:
1. The CURRENT system prompt for this pass (verbatim).
2. EVERY chunk's gold value and current prediction for the fields this pass owns -- both
   correct and wrong. Use the correct ones to see what NOT to break, not just the wrong
   ones to see what to fix.
3. The ROUND-BY-ROUND HISTORY of this loop so far: each prior patch attempted, the prompt
   text it changed, and the MECHANICAL before/after score that patch produced.

IMPORTANT: prior patches and diagnoses -- including anything already baked into the
CURRENT prompt from earlier sessions, not just this loop's own history -- are HYPOTHESES,
not ground truth. Re-evaluate them against the evidence you are looking at right now. A
patch that seemed reasonable when it was made may not be well-supported by the full
current evidence. If you believe a specific prior patch is not earning its keep, or is
actively causing part of the current error pattern, you may propose REVERTING or revising
it as your top-priority pattern this round -- treat that as equally valid to proposing a
brand-new patch.

Your job:
1. Group the CURRENT mismatches into distinct failure patterns based on the behavioral
   mechanism producing the errors. Do not produce one pattern per chunk. Do not group
   mismatches merely because they share the same wrong label.
2. For each pattern, determine whether it is plausibly caused by the CURRENT PROMPT TEXT
   (which may include text from a prior round's patch -- see above).
3. If plausibly prompt-caused: identify and quote the exact responsible language; explain
   the behavioral problem; propose a generalizable correction; provide draft wording.
4. If not caused by the prompt: label it RESIDUAL MODEL ERROR and do not invent a
   speculative prompt change.
5. List patterns in priority order, most impactful first (affected chunks, severity,
   importance). A proposal to revert/revise a prior patch competes for priority on the
   same basis as any other pattern -- do not automatically place it first or last.

CONSTRAINTS (unchanged from before):
- Do not comment on fields this pass does not own.
- Treat the gold set as ground truth for evaluation, but do not merely optimize for
  agreement with these 30 examples -- a fix must express a generalizable behavioral rule.
- Do not create chunk-specific exceptions or encode individual gold answers.
- Do not recommend a change merely because it would raise the score here if it makes the
  rule less general or less faithful to the taxonomy.
- Focus on the smallest meaningful behavioral correction. Do not rewrite the entire prompt.
- Do not perform a new annotation of the passages -- your task is diagnosis.

For each prompt-caused pattern (including a revert/revise-prior-patch proposal), use:

PATTERN:
<short name -- if this is a revert/revise of a prior patch, say so explicitly>

AFFECTED:
<number of chunks>

FAILURE MECHANISM:
<what behavioral mistake is occurring>

PROMPT LANGUAGE:
<exact quotation from the current prompt -- if reverting a prior patch, quote the patch's
own added text>

WHY THIS CAUSES THE FAILURE:
<brief explanation>

PROPOSED FIX:
<general behavioral change -- may be "revert to the pre-patch wording" if that's the
right call given current evidence>

DRAFT INSTRUCTION:
<proposed wording that should be added/replaced/restored>

For residual model/error-pattern issues, use:

PATTERN:
<short name>

AFFECTED:
<number of chunks>

FAILURE MECHANISM:
<what behavioral mistake is occurring>

CLASSIFICATION:
RESIDUAL MODEL ERROR — no clear prompt-level cause

REASON:
<brief explanation>

Do not produce an exact find-and-replace patch. A separate fixer will convert your
proposal into a precise textual patch."""

FIXER_SYSTEM = """You are a patch specialist. You convert a critic's diagnostic proposal into a
precise, minimal, literally applicable text patch to the CURRENT system prompt.

You do NOT diagnose the annotation errors yourself. You do NOT second-guess the critic's
judgment. You do NOT invent additional fixes. Your only job is to operationalize the
critic's stated proposal (which may be "restore the prior wording" / a revert, or a new
addition) into an exact find-and-replace patch.

You will be shown:
1. The CURRENT full prompt text for ONE pass, verbatim.
2. The critic's diagnosis and proposed fix for ONE failure pattern.

Your task:
1. Identify the exact contiguous substring of the CURRENT prompt that should be changed.
2. Copy that substring EXACTLY into OLD, preserving wording, capitalization, punctuation,
   spaces, indentation, line breaks.
3. Produce NEW containing the smallest possible replacement that implements the critic's
   proposed behavioral correction (which may be shorter than OLD, if this is a revert).

RULES:
- OLD must be a verbatim, contiguous substring of the CURRENT prompt. Do not paraphrase or
  normalize whitespace. Do not include surrounding text unless it is part of the substring
  being replaced.
- NEW must implement ONLY the specific behavioral change proposed by the critic.
- Preserve all unrelated instructions. Do not rewrite the entire prompt. Do not improve the
  critic's proposal or add content it did not request. Do not fix other failure patterns.
  Do not modify instructions for fields outside the critic's proposed pattern.
- Do not introduce chunk-specific exceptions or encode individual gold answers.
- If the critic's proposal is vague, take the most literal, minimal interpretation.
- If the critic's proposal does not identify a change that can actually be implemented in
  the current prompt, do not invent one.

If the critic explicitly classifies the pattern as RESIDUAL MODEL ERROR, or if no suitable
exact substring exists, return:

old: NONE
new: NONE

Do not output explanations, commentary, reasoning, markdown fences, or additional fields.
Return ONLY:

old: <exact verbatim substring of the current prompt>

new: <replacement text>"""


def _load_history():
    return json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else {}


def _save_history(h):
    HISTORY_PATH.write_text(json.dumps(h, indent=2, ensure_ascii=False))


def _all_predictions_text(pass_name):
    gold = load_gold()
    cache_path = ROOT / "experiments" / "pass_cache" / f"{pass_name}.json"
    results = json.loads(cache_path.read_text())
    from chunking import load_chunks
    chunks = {c["id"]: c for c in load_chunks()}
    lines = []
    for cid in gold:
        r = results.get(cid)
        lines.append(f"--- {cid} ---")
        lines.append(f"TEXT: {chunks[cid]['text']}")
        for field in PASS_FIELDS[pass_name]:
            g = gold[cid][field]
            p = r[field] if r else "FAIL"
            status = "CORRECT" if g == p else "WRONG"
            lines.append(f"  {field}: gold={g!r} pred={p!r}  [{status}]")
        lines.append("")
    return "\n".join(lines)


def cmd_critic(pass_name, round_num):
    prompt_name, prompt_text = PASS_PROMPTS[pass_name]
    preds_text = _all_predictions_text(pass_name)
    history = _load_history().get(pass_name, [])

    hist_lines = []
    if history:
        hist_lines.append("=== ROUND-BY-ROUND HISTORY OF THIS LOOP SO FAR ===")
        for h in history:
            hist_lines.append(f"Round {h['round']}: {h['pattern_name']}")
            hist_lines.append(f"  patch old: {h['old'][:200]}")
            hist_lines.append(f"  patch new: {h['new'][:200]}")
            hist_lines.append(f"  score before: {h['score_before']}")
            hist_lines.append(f"  score after:  {h['score_after']}")
            hist_lines.append(f"  outcome: {h['outcome']}")
            hist_lines.append("")
    else:
        hist_lines.append("=== ROUND-BY-ROUND HISTORY OF THIS LOOP SO FAR ===")
        hist_lines.append("(none yet -- this is round 1 of this loop. Note: the CURRENT "
                           "prompt below may still contain patches from an EARLIER, separate "
                           "session's loop -- treat those as hypotheses too, per your "
                           "instructions.)")

    user_msg = (f"=== CURRENT {prompt_name} ===\n{prompt_text}\n\n"
                + "\n".join(hist_lines) + "\n\n"
                f"=== FULL PREDICTION SET (all 30 chunks, fields this pass owns: "
                f"{PASS_FIELDS[pass_name]}) ===\n{preds_text}")

    r = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role": "system", "content": CRITIC_SYSTEM},
                  {"role": "user", "content": user_msg}],
    )
    out = r.choices[0].message.content
    print(out)
    out_path = LOG_DIR / f"v2_{pass_name}_r{round_num}_critic.txt"
    out_path.write_text(out)
    print(f"\n[saved -> {out_path}]")


def cmd_fixer(pass_name, pattern_file, round_num):
    prompt_name, prompt_text = PASS_PROMPTS[pass_name]
    pattern_text = Path(pattern_file).read_text()
    user_msg = f"=== CURRENT {prompt_name} ===\n{prompt_text}\n\n=== CRITIC'S PATTERN ===\n{pattern_text}"
    r = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role": "system", "content": FIXER_SYSTEM},
                  {"role": "user", "content": user_msg}],
    )
    out = r.choices[0].message.content
    print(out)
    out_path = LOG_DIR / f"v2_{pass_name}_r{round_num}_fixer.txt"
    out_path.write_text(out)
    print(f"\n[saved -> {out_path}]")


def cmd_log_round(pass_name, round_num, pattern_name, old, new, score_before, score_after, outcome):
    h = _load_history()
    h.setdefault(pass_name, [])
    h[pass_name].append({
        "round": round_num, "pattern_name": pattern_name, "old": old, "new": new,
        "score_before": score_before, "score_after": score_after, "outcome": outcome,
    })
    _save_history(h)
    print(f"logged round {round_num} for {pass_name} -> {HISTORY_PATH}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, pass_name = sys.argv[1], sys.argv[2]
    if cmd == "score":
        gold = load_gold()
        results = run_pass(pass_name, list(gold.keys()))
        score_pass(pass_name, gold, results)
    elif cmd == "critic":
        cmd_critic(pass_name, sys.argv[3])
    elif cmd == "fixer":
        cmd_fixer(pass_name, sys.argv[3], sys.argv[4])
    else:
        print(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
