"""Tight iteration loop per PASS (not per chunk) for the E (routed, candidate->adjudicate)
3-pass pipeline (src/tag_three_pass.py). Test one pass's owned fields against the FULL
30-chunk gold set, diagnose failures with the critic, patch, and retest -- repeat until
improvement plateaus.

Pass 2 (scene) is independent of pass 1 (no chaining). Pass 3 (global) takes pass 2's
CANDIDATE themes/characters and adjudicates them (final canonical_themes/characters_present
come from pass 3, not pass 2) in addition to producing narrative_relation. To avoid
re-running upstream passes every time you're only iterating on a downstream pass's prompt,
this script caches each pass's raw output separately and lets you re-run just ONE pass while
reusing the others from cache.

Usage:
    .venv/bin/python experiments/iterate_pass.py run local              # (re)run pass 1 over all 30, cache + score
    .venv/bin/python experiments/iterate_pass.py run scene              # independent of local
    .venv/bin/python experiments/iterate_pass.py run global             # uses cached scene as input; owns final themes/characters
    .venv/bin/python experiments/iterate_pass.py critic global          # send current mismatches + prompt to critic
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from chunking import load_chunks
from ingest import _scene_context
from scenes import render_global_map
from tag_three_pass import (GLOBAL_PASS_SYSTEM_PROMPT, LOCAL_PASS_SYSTEM_PROMPT,
                             SCENE_PASS_SYSTEM_PROMPT, ScenePassTags,
                             tag_global_pass, tag_local_pass, tag_scene_pass)

from run_first30_eval import load_gold, pct, score_set_field, score_single_label

CACHE_DIR = ROOT / "experiments" / "pass_cache"
CACHE_DIR.mkdir(exist_ok=True)

PASS_FIELDS = {
    "local": ["narrating_voice", "speaking_voice", "speaker_relation"],
    "scene": ["canonical_themes", "characters_present"],   # PASS 2's own candidate output
    "global": ["narrative_relation"],                       # PASS 3's own fresh field only
}
PASS_PROMPTS = {
    "local": ("LOCAL_PASS_SYSTEM_PROMPT", LOCAL_PASS_SYSTEM_PROMPT),
    "scene": ("SCENE_PASS_SYSTEM_PROMPT", SCENE_PASS_SYSTEM_PROMPT),
    "global": ("GLOBAL_PASS_SYSTEM_PROMPT", GLOBAL_PASS_SYSTEM_PROMPT),
}


def _cache_path(pass_name):
    return CACHE_DIR / f"{pass_name}.json"


def _load_cache(pass_name):
    p = _cache_path(pass_name)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save_cache(pass_name, data):
    _cache_path(pass_name).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def run_pass(pass_name: str, gold_ids: list):
    chunks = {c["id"]: c for c in load_chunks()}
    gm_str = render_global_map(json.loads((ROOT / "context" / "global_map.json").read_text()))
    scenes = json.loads((ROOT / "context" / "scenes.json").read_text())
    cards = json.loads((ROOT / "context" / "scene_cards.json").read_text())
    scene_of = {cid: i for i, s in enumerate(scenes) for cid in s["chunk_ids"]}

    out = {}
    if pass_name == "global":
        scene_cache = _load_cache("scene")
        if not scene_cache:
            print("No cached scene pass output -- run 'run scene' first.")
            sys.exit(1)

    for i, cid in enumerate(gold_ids, 1):
        c = chunks[cid]
        sc = _scene_context(cards, scene_of[cid])
        try:
            if pass_name == "local":
                prev = chunks[c["prev_id"]]["text"] if c["prev_id"] else None
                nxt = chunks[c["next_id"]]["text"] if c["next_id"] else None
                tags = tag_local_pass(c["text"], prev, nxt).model_dump()
            elif pass_name == "scene":
                tags = tag_scene_pass(c["text"], sc).model_dump()
            else:  # global -- adjudicates pass 2's candidates, also emits fresh narrative_relation
                scene = ScenePassTags(**scene_cache[cid])
                tags = tag_global_pass(c["text"], sc, gm_str,
                                        scene.canonical_themes, scene.characters_present).model_dump()
        except Exception as e:
            print(f"  ! {pass_name} pass failed for {cid}: {e}")
            tags = None
        out[cid] = tags
        fields = PASS_FIELDS[pass_name]
        summary = {f: (tags[f] if tags else "FAIL") for f in fields}
        print(f"  {i:>2}/{len(gold_ids)}  {cid:<32} {summary}")

    _save_cache(pass_name, out)
    print(f"\nCached -> {_cache_path(pass_name)}")
    return out


def score_pass(pass_name: str, gold: dict, results: dict):
    fields = PASS_FIELDS[pass_name]
    print(f"\n{'=' * 90}\nPASS: {pass_name.upper()}\n{'=' * 90}")
    all_mismatches = []
    for field in fields:
        preds = {cid: results.get(cid) for cid in gold}
        if field == "canonical_themes":
            s = score_set_field(field, gold, preds)
            print(f"\n-- {field}  micro P={pct(s['micro']['precision'])} "
                  f"R={pct(s['micro']['recall'])} F1={s['micro']['f1']:.2f}")
            if s["over_applied"]:
                print(f"   over-applied: {s['over_applied']}")
            if s["under_applied"]:
                print(f"   under-applied: {s['under_applied']}")
        elif field == "characters_present":
            s = score_set_field(field, gold, preds)
            print(f"\n-- {field}  micro P={pct(s['micro']['precision'])} "
                  f"R={pct(s['micro']['recall'])} F1={s['micro']['f1']:.2f}")
        else:
            s = score_single_label(field, gold, preds)
            print(f"\n-- {field}  accuracy {pct(s['accuracy'])}")
            for cls, m in s["per_class"].items():
                print(f"   {str(cls):<16} support={m['support']:<4} "
                      f"P={pct(m['precision']):<6} R={pct(m['recall']):<6} F1={m['f1']:.2f}")
            for m in s["mismatches"]:
                print(f"     {m['id']:<32} gold={str(m['gold']):<16} pred={m['pred']}")
                all_mismatches.append({"id": m["id"], "field": field, "gold": m["gold"], "pred": m["pred"]})
    return all_mismatches


def cmd_run(pass_name):
    gold = load_gold()
    results = run_pass(pass_name, list(gold.keys()))
    score_pass(pass_name, gold, results)


def cmd_critic(pass_name, extra_note=""):
    gold = load_gold()
    results = _load_cache(pass_name)
    if not results:
        print(f"No cached {pass_name} output -- run 'run {pass_name}' first.")
        sys.exit(1)
    mismatches = score_pass(pass_name, gold, results)
    if not mismatches:
        print("\nNo mismatches on the fields this pass owns -- nothing to send to the critic.")
        return

    chunks = {c["id"]: c for c in load_chunks()}
    prompt_name, prompt_text = PASS_PROMPTS[pass_name]

    lines = [f"PASS: {pass_name}  (prompt variable: {prompt_name})", ""]
    if extra_note:
        lines.append(extra_note)
        lines.append("")
    lines.append(f"=== CURRENT {prompt_name} ===")
    lines.append(prompt_text)
    lines.append("")
    lines.append(f"=== ALL {len(mismatches)} MISMATCHES ON FIELDS THIS PASS OWNS ({PASS_FIELDS[pass_name]}) ===")
    for m in mismatches:
        cid = m["id"]
        lines.append(f"--- {cid}  field={m['field']} ---")
        lines.append(f"TEXT: {chunks[cid]['text']}")
        lines.append(f"GOLD: {m['gold']!r}")
        lines.append(f"PREDICTED: {m['pred']!r}")
        lines.append("")

    user_msg = "\n".join(lines)
    from tag_white_nights import client
    CRITIC_SYSTEM = """You are a prompt-engineering diagnostician reviewing ALL current mismatches
for ONE pass of a multi-pass tagging pipeline, across the full 30-chunk gold set at once (not
one chunk at a time). Look for PATTERNS across the mismatches, not just individual cases.

Your job:
1. DIAGNOSE: group the mismatches into 1-3 distinct failure patterns (e.g. "over-applies X",
   "confuses X with Y under condition Z"). For each pattern, quote the exact current prompt
   language you think causes it.
2. PROPOSE PATCHES: for each pattern, propose the smallest edit to the current prompt that
   would fix it, without contradicting instructions for a DIFFERENT pattern. Give exact OLD
   (verbatim substring) and NEW text for each patch, in priority order (most impactful first).
3. STATE REGRESSION RISK for each patch.

Return, for each distinct pattern found:

PATTERN N: <short name>
DIAGNOSIS: <2-4 sentences>
AFFECTED CHUNKS: <ids>
PATCH:
old: <exact verbatim substring>
new: <replacement text>
REGRESSION RISK: <1-2 sentences>
"""
    r = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role": "system", "content": CRITIC_SYSTEM},
                  {"role": "user", "content": user_msg}],
    )
    out = r.choices[0].message.content
    print("\n\n" + "=" * 90)
    print("CRITIC RESPONSE")
    print("=" * 90)
    print(out)
    out_path = ROOT / "experiments" / f"pass_critic_{pass_name}.txt"
    out_path.write_text(out)
    print(f"\n[saved -> {out_path}]")


def main():
    if len(sys.argv) < 3:
        print("usage: python experiments/iterate_pass.py <run|critic> <local|scene|global> [note]")
        sys.exit(1)
    cmd, pass_name = sys.argv[1], sys.argv[2]
    assert pass_name in ("local", "scene", "global")
    if cmd == "run":
        cmd_run(pass_name)
    elif cmd == "critic":
        cmd_critic(pass_name, sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
