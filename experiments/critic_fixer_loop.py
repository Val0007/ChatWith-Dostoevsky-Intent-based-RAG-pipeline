"""Strict critic -> fixer -> patch -> rerun -> verify loop, ONE pattern at a time, for ONE pass
of the E (three-pass, routed/adjudicated) pipeline. Orchestrator (the calling agent) applies
patches mechanically via exact string match only -- no independent judgment, no batching.

Subcommands:
    score  <pass>              run the pass fresh over all 30 gold chunks, cache + print
                                mechanical scoring (exact match against gold) for the fields
                                this pass owns
    critic <pass>               call the critic on the CURRENT cached mismatches + CURRENT
                                prompt text; print + save raw response
    fixer  <pass> <pattern_file> call the fixer with the CURRENT prompt text + ONE pattern
                                block (read from pattern_file); print + save old/new patch

The orchestrator (not this script) is responsible for: extracting pattern #1 from the critic's
response, applying the fixer's patch via exact Edit, deciding continue/stop from the mechanical
score comparison, and reverting a patch that doesn't improve the tracked metric.
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

PASS_PROMPTS = {
    "local": ("LOCAL_PASS_SYSTEM_PROMPT", LOCAL_PASS_SYSTEM_PROMPT),
    "scene": ("SCENE_PASS_SYSTEM_PROMPT", SCENE_PASS_SYSTEM_PROMPT),
    "global": ("GLOBAL_PASS_SYSTEM_PROMPT", GLOBAL_PASS_SYSTEM_PROMPT),
}

CRITIC_SYSTEM = """You are a diagnostic critic reviewing mismatches for ONE pass of a
multi-pass literary annotation pipeline, against a hand-built gold
standard, across the full 30-chunk gold set at once.

You will be shown:

1. The CURRENT system prompt for this pass (verbatim).
2. Every mismatch between this pass's predictions and gold, on the fields
   this pass owns: passage text, gold value, predicted value.

Your job:

1. Group the mismatches into distinct failure patterns based on the
   behavioral mechanism producing the errors. Do not produce one pattern
   per chunk. Do not group mismatches merely because they share the same
   wrong label.

2. For each pattern, determine whether the mismatch is plausibly caused by
   the CURRENT PROMPT TEXT.

3. If the pattern is plausibly prompt-caused:
   - identify the exact language in the CURRENT PROMPT that contributes to
     the failure;
   - quote that exact language;
   - explain the behavioral problem it creates;
   - propose a generalizable correction;
   - provide draft wording for the corrected instruction.

4. If the pattern does not appear to be caused by the CURRENT PROMPT:
   explicitly label it as a residual model/error-pattern issue and do NOT
   invent a speculative prompt change.

5. List patterns in priority order, with the most impactful patterns first.
   Prioritize by the number of affected chunks, severity, and importance
   to the field's intended behavior.

IMPORTANT CONSTRAINTS:

- Do not comment on fields this pass does not own.
- Treat the GOLD SET as the ground truth for evaluating the current system.
- However, do not merely optimize for agreement with these 30 examples.
- A proposed prompt fix must express a generalizable behavioral rule
  supported by the observed failure pattern and consistent with the
  taxonomy.
- Do not create chunk-specific exceptions.
- Do not create instructions that encode individual gold answers.
- Do not recommend rules such as "when this particular passage/chunk
  occurs, output X" unless that is genuinely necessary to express a
  general rule.
- Do not assume every mismatch is caused by the prompt.
- Do not recommend changes merely because they would increase the score
  on these examples if they would make the underlying annotation rule
  less general or less faithful to the taxonomy.
- Focus on identifying the smallest meaningful behavioral correction.
- Do not rewrite the entire prompt.
- Do not propose fixes for unrelated fields.
- Do not perform a new annotation of the passages. Your task is diagnosis
  of the observed failure patterns and the current prompt.

For each prompt-caused failure pattern, use this structure:

PATTERN:
<short name>

AFFECTED:
<number of chunks / affected examples>

FAILURE MECHANISM:
<what behavioral mistake is occurring>

PROMPT LANGUAGE:
<exact quotation from the current prompt>

WHY THIS CAUSES THE FAILURE:
<brief explanation>

PROPOSED FIX:
<general behavioral change>

DRAFT INSTRUCTION:
<proposed wording that should be added/replaced>

For residual model/error-pattern issues, use:

PATTERN:
<short name>

AFFECTED:
<number of chunks / affected examples>

FAILURE MECHANISM:
<what behavioral mistake is occurring>

CLASSIFICATION:
RESIDUAL MODEL ERROR — no clear prompt-level cause

REASON:
<brief explanation of why the current prompt does not clearly warrant
a change>

Do not produce an exact find-and-replace patch. A separate fixer will
convert your proposal into a precise textual patch."""

FIXER_SYSTEM = """You are a patch specialist.

You convert a critic's diagnostic proposal into a precise, minimal,
literally applicable text patch to the CURRENT system prompt.

You do NOT diagnose the annotation errors yourself.
You do NOT second-guess the critic's judgment.
You do NOT invent additional fixes.

Your only job is to operationalize the critic's stated proposal into an
exact find-and-replace patch.

You will be shown:

1. The CURRENT full prompt text for ONE pass, verbatim.
2. The critic's diagnosis and proposed fix for ONE failure pattern.

Your task:

1. Identify the exact contiguous substring of the CURRENT prompt that
   should be changed.

2. Copy that substring EXACTLY into OLD, preserving:
   - wording
   - capitalization
   - punctuation
   - spaces
   - indentation
   - line breaks

3. Produce NEW containing the smallest possible replacement that implements
   the critic's proposed behavioral correction.

RULES:

- OLD must be a verbatim, contiguous substring of the CURRENT prompt.
- Do not paraphrase OLD.
- Do not normalize whitespace.
- Do not include surrounding text unless it is part of the substring being
  replaced.
- NEW must implement ONLY the specific behavioral change proposed by the
  critic.
- Preserve all unrelated instructions in OLD.
- Do not rewrite the entire prompt.
- Do not improve the critic's proposal.
- Do not introduce content that the critic did not request.
- Do not add additional rules based on your own judgment.
- Do not fix other failure patterns.
- Do not modify instructions concerning fields outside the critic's
  proposed pattern.
- Do not introduce chunk-specific exceptions.
- Do not encode individual gold answers.
- Do not add specific chunk IDs, passages, or gold labels unless the critic
  explicitly proposes them as part of a genuinely generalizable rule.
- If the critic's proposal is vague, take the most literal and minimal
  interpretation possible.
- If the critic's proposal does not identify a change that can actually be
  implemented in the current prompt, do not invent one.

If the critic explicitly classifies the pattern as:

RESIDUAL MODEL ERROR — no clear prompt-level cause

then return:

old: NONE
new: NONE

If no suitable exact substring exists in the CURRENT prompt, return:

old: NONE
new: NONE

Do not output explanations, commentary, reasoning, markdown fences, or
additional fields.

Return ONLY:

old: <exact verbatim substring of the current prompt>

new: <replacement text>"""


def cmd_score(pass_name):
    gold = load_gold()
    results = run_pass(pass_name, list(gold.keys()))
    score_pass(pass_name, gold, results)


def _mismatches_text(pass_name):
    gold = load_gold()
    cache_path = ROOT / "experiments" / "pass_cache" / f"{pass_name}.json"
    results = json.loads(cache_path.read_text())
    from chunking import load_chunks
    chunks = {c["id"]: c for c in load_chunks()}
    lines = []
    for field in PASS_FIELDS[pass_name]:
        for cid in gold:
            g = gold[cid][field]
            p = results.get(cid, {}).get(field) if results.get(cid) else "FAIL"
            if g != p:
                lines.append(f"--- {cid}  field={field} ---")
                lines.append(f"TEXT: {chunks[cid]['text']}")
                lines.append(f"GOLD: {g!r}")
                lines.append(f"PREDICTED: {p!r}")
                lines.append("")
    return "\n".join(lines)


def cmd_critic(pass_name, round_num):
    prompt_name, prompt_text = PASS_PROMPTS[pass_name]
    mismatch_text = _mismatches_text(pass_name)
    user_msg = (f"=== CURRENT {prompt_name} ===\n{prompt_text}\n\n"
                f"=== MISMATCHES (fields this pass owns: {PASS_FIELDS[pass_name]}) ===\n{mismatch_text}")
    r = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role": "system", "content": CRITIC_SYSTEM},
                  {"role": "user", "content": user_msg}],
    )
    out = r.choices[0].message.content
    print(out)
    out_path = LOG_DIR / f"{pass_name}_r{round_num}_critic.txt"
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
    out_path = LOG_DIR / f"{pass_name}_r{round_num}_fixer.txt"
    out_path.write_text(out)
    print(f"\n[saved -> {out_path}]")


def main():
    if len(sys.argv) < 3:
        print("usage: critic_fixer_loop.py score <pass>")
        print("       critic_fixer_loop.py critic <pass> <round_num>")
        print("       critic_fixer_loop.py fixer <pass> <pattern_file> <round_num>")
        sys.exit(1)
    cmd, pass_name = sys.argv[1], sys.argv[2]
    if cmd == "score":
        cmd_score(pass_name)
    elif cmd == "critic":
        cmd_critic(pass_name, sys.argv[3])
    elif cmd == "fixer":
        cmd_fixer(pass_name, sys.argv[3], sys.argv[4])
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
