"""Tight iteration loop on ONE hard chunk, ONE arm at a time (never both at once).

Usage pattern (semi-automated -- patches are reviewed and applied by hand, never
auto-applied, same discipline as prompt_feedback.py):

    .venv/bin/python experiments/iterate_hard_chunk.py check <chunk_id> <local|context> <field>
        -> tags the chunk fresh with the CURRENT prompt, compares to gold, prints result,
           appends the attempt to experiments/iterate_<chunk_id>_<arm>_<field>.json

    .venv/bin/python experiments/iterate_hard_chunk.py critic <chunk_id> <local|context> <field>
        -> sends the critic the chunk, gold, the LATEST wrong prediction, the relevant prompt
           text (SYSTEM_PROMPT only for local; SYSTEM_PROMPT + CONTEXT_USER_TEMPLATE for
           context), AND the full history of prior attempts/patches on this exact chunk so it
           doesn't re-suggest something already tried. Prints diagnosis + patch, saves it.

Deliberately scoped to ONE chunk so each critic call is maximally focused -- no diluting the
question across multiple failure modes at once. Deliberately never mixes local and context in
the same run: pass one arm, converge it, THEN move to the other.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunking import load_chunks
from ingest import _scene_context
from scenes import render_global_map
from tag_white_nights import (CONTEXT_USER_TEMPLATE, SYSTEM_PROMPT, client,
                               tag_one_chunk, tag_with_context)

GOLD_PATH = ROOT / "experiments" / "gold_first30.jsonl"

CRITIC_SYSTEM = """You are a prompt-engineering diagnostician, iterating on ONE specific
failing example, one round at a time. You will be shown:
  1. The prompt text currently in force for this arm (verbatim).
  2. ONE passage, its gold label + why, and what the tagger just predicted.
  3. The FULL HISTORY of every prior attempt on this exact chunk: what was predicted, what
     patch was tried, and whether it worked. Do NOT repeat a patch that was already tried and
     failed -- propose something genuinely different this round.

Your job:
  1. DIAGNOSE why THIS SPECIFIC passage is still getting mistagged, given everything already
     tried. Quote the exact current prompt language you think is responsible.
  2. PROPOSE A MINIMAL PATCH: the smallest edit to the CURRENT prompt text that would fix this
     one case, without contradicting instructions unrelated to this failure. Give exact OLD
     (verbatim substring) and NEW text, and say which variable (SYSTEM_PROMPT or
     CONTEXT_USER_TEMPLATE).
  3. STATE REGRESSION RISK: what currently-correct behavior might this plausibly break.

Return ONLY:

DIAGNOSIS:
<2-4 sentences>

PATCH:
target: SYSTEM_PROMPT | CONTEXT_USER_TEMPLATE
old: <exact verbatim substring to replace>
new: <replacement text>

REGRESSION RISK:
<1-2 sentences, or "none identified">
"""


def load_gold():
    gold = {}
    for line in GOLD_PATH.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            gold[d["id"]] = d
    return gold


def history_path(cid, arm, field):
    return ROOT / "experiments" / f"iterate_{cid}_{arm}_{field}.json"


def load_history(cid, arm, field):
    p = history_path(cid, arm, field)
    return json.loads(p.read_text()) if p.exists() else []


def save_history(cid, arm, field, history):
    history_path(cid, arm, field).write_text(json.dumps(history, indent=2, ensure_ascii=False))


def tag_chunk(cid, arm):
    chunks = {c["id"]: c for c in load_chunks()}
    c = chunks[cid]
    if arm == "local":
        prev = chunks[c["prev_id"]]["text"] if c["prev_id"] else None
        nxt = chunks[c["next_id"]]["text"] if c["next_id"] else None
        return tag_one_chunk(c["text"], prev, nxt).model_dump()
    else:
        gm_str = render_global_map(json.loads((ROOT / "context" / "global_map.json").read_text()))
        scenes = json.loads((ROOT / "context" / "scenes.json").read_text())
        cards = json.loads((ROOT / "context" / "scene_cards.json").read_text())
        scene_of = {c2id: i for i, s in enumerate(scenes) for c2id in s["chunk_ids"]}
        sc = _scene_context(cards, scene_of[cid])
        return tag_with_context(c["text"], gm_str, sc).model_dump()


def cmd_check(cid, arm, field):
    gold = load_gold()
    g = gold[cid]
    pred_tags = tag_chunk(cid, arm)
    pred = pred_tags[field]
    correct = pred == g[field]
    print(f"{'CORRECT' if correct else 'WRONG':8} gold={g[field]!r:32} pred={pred!r}")
    history = load_history(cid, arm, field)
    history.append({"attempt": len(history) + 1, "pred": pred, "gold": g[field],
                     "correct": correct, "full_pred": pred_tags})
    save_history(cid, arm, field, history)
    print(f"[history -> {history_path(cid, arm, field)}, {len(history)} attempt(s) so far]")
    return correct


def cmd_critic(cid, arm, field, rationale=None):
    gold = load_gold()
    g = gold[cid]
    chunks = {c["id"]: c for c in load_chunks()}
    text = chunks[cid]["text"]
    history = load_history(cid, arm, field)
    if not history:
        print("No history yet -- run 'check' first.")
        sys.exit(1)
    latest = history[-1]
    if latest["correct"]:
        print("Latest attempt was already correct -- nothing to fix.")
        return

    lines = [f"CHUNK: {cid}   ARM: {arm}   FIELD: {field}", ""]
    lines.append("=== PROMPT TEXT CURRENTLY IN FORCE FOR THIS ARM ===")
    lines.append("--- SYSTEM_PROMPT (shared by both arms) ---")
    lines.append(SYSTEM_PROMPT)
    if arm == "context":
        lines.append("")
        lines.append("--- CONTEXT_USER_TEMPLATE (context arm only) ---")
        lines.append(CONTEXT_USER_TEMPLATE)
    lines.append("")
    lines.append("=== THE PASSAGE ===")
    lines.append(text)
    lines.append("")
    lines.append(f"GOLD {field}: {g[field]!r}")
    if rationale:
        lines.append(f"GOLD RATIONALE: {rationale}")
    lines.append(f"LATEST PREDICTION (attempt {latest['attempt']}): {latest['pred']!r}  <- WRONG")
    lines.append("")
    lines.append("=== FULL HISTORY ON THIS EXACT CHUNK (do not repeat a failed patch) ===")
    for h in history:
        status = "CORRECT" if h["correct"] else "WRONG"
        lines.append(f"  attempt {h['attempt']}: predicted {h['pred']!r} -> {status}")
    prior_patches = ROOT / "experiments" / f"iterate_{cid}_{arm}_{field}_patches.json"
    if prior_patches.exists():
        lines.append("")
        lines.append("=== PATCHES ALREADY TRIED ON THIS CHUNK (do not repeat these) ===")
        for p in json.loads(prior_patches.read_text()):
            lines.append(f"  - {p}")

    user_msg = "\n".join(lines)
    r = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role": "system", "content": CRITIC_SYSTEM},
                  {"role": "user", "content": user_msg}],
    )
    out = r.choices[0].message.content
    print(out)
    out_path = ROOT / "experiments" / f"iterate_{cid}_{arm}_{field}_round{len(history)}.txt"
    out_path.write_text(out)
    print(f"\n[saved -> {out_path}]")
    return out


def cmd_log_patch(cid, arm, field, summary):
    p = ROOT / "experiments" / f"iterate_{cid}_{arm}_{field}_patches.json"
    patches = json.loads(p.read_text()) if p.exists() else []
    patches.append(summary)
    p.write_text(json.dumps(patches, indent=2, ensure_ascii=False))
    print(f"logged patch #{len(patches)} -> {p}")


def main():
    if len(sys.argv) < 5:
        print("usage:")
        print("  python experiments/iterate_hard_chunk.py check <chunk_id> <local|context> <field>")
        print("  python experiments/iterate_hard_chunk.py critic <chunk_id> <local|context> <field> [rationale]")
        print("  python experiments/iterate_hard_chunk.py logpatch <chunk_id> <local|context> <field> <summary>")
        sys.exit(1)
    cmd, cid, arm, field = sys.argv[1:5]
    assert arm in ("local", "context")
    if cmd == "check":
        cmd_check(cid, arm, field)
    elif cmd == "critic":
        rationale = sys.argv[5] if len(sys.argv) > 5 else None
        cmd_critic(cid, arm, field, rationale)
    elif cmd == "logpatch":
        cmd_log_patch(cid, arm, field, sys.argv[5] if len(sys.argv) > 5 else "")
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
