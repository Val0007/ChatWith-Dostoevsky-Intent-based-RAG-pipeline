"""Phase 01 of docs/tagger_study_plan.html: the four-arm context ablation, run for real.

    A - local        = TARGET + prev/next chunk only (tag_one_chunk)
    B - +scenes       = TARGET + adjacent scene cards, NO global map
    C - +global map   = TARGET + global map, NO adjacent scene cards
    D - both          = TARGET + global map + adjacent scene cards (tag_with_context, the
                         production arm -- same as "context" in run_first30_eval.py)

B, C, D all use the SAME tag_with_context() / CONTEXT_USER_TEMPLATE -- only which context
strings are populated varies (the exact pattern experiments/tagging/narrative_context_ablation.py used
for its 7-chunk narrative_relation-only probe). This run covers all 30 gold chunks and ALL
fields, using whatever prompt is currently in src/tag_white_nights.py (post-F7 in this repo's
history) -- earlier ablations (F1, F5) used older, less-developed prompts, so numbers here are
not directly comparable to those without accounting for the prompt having changed.

Run from repo root:
    .venv/bin/python experiments/tagging/run_four_arm_ablation.py                # tag + score
    .venv/bin/python experiments/tagging/run_four_arm_ablation.py --score-only   # reuse cached output
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "tagging"))

from chunking import load_chunks
from ingest import _scene_context
from scenes import render_global_map
from tag_white_nights import tag_one_chunk, tag_with_context

from run_first30_eval import (SET_FIELDS, SINGLE_LABEL_FIELDS, load_gold, pct,
                               score_set_field, score_single_label)

CACHE_PATH = ROOT / "experiments" / "tagging" / "results" / "four_arm_ablation_output.json"
ARMS = ["A_local", "B_scenes", "C_global", "D_both"]
ARM_LABELS = {
    "A_local": "A - local (prev/next only)",
    "B_scenes": "B - +scenes (no global map)",
    "C_global": "C - +global map (no scenes)",
    "D_both": "D - both (production)",
}


def run_tagger(gold_ids: list) -> dict:
    chunks = {c["id"]: c for c in load_chunks()}
    gm_str = render_global_map(json.loads((ROOT / "data" / "context" / "global_map.json").read_text()))
    scenes = json.loads((ROOT / "data" / "context" / "scenes.json").read_text())
    cards = json.loads((ROOT / "data" / "context" / "scene_cards.json").read_text())
    scene_of = {cid: i for i, s in enumerate(scenes) for cid in s["chunk_ids"]}

    out = {}
    for i, cid in enumerate(gold_ids, 1):
        c = chunks[cid]
        prev = chunks[c["prev_id"]]["text"] if c["prev_id"] else None
        nxt = chunks[c["next_id"]]["text"] if c["next_id"] else None
        sc = _scene_context(cards, scene_of[cid])
        row = {}

        def try_tag(fn, *args):
            for attempt in (1, 2):
                try:
                    return fn(*args).model_dump()
                except Exception as e:
                    if attempt == 2:
                        print(f"  ! tag failed for {cid}: {e}")
                        return None

        row["A_local"] = try_tag(tag_one_chunk, c["text"], prev, nxt)
        row["B_scenes"] = try_tag(tag_with_context, c["text"], "(none)", sc)
        row["C_global"] = try_tag(tag_with_context, c["text"], gm_str, "(none)")
        row["D_both"] = try_tag(tag_with_context, c["text"], gm_str, sc)

        out[cid] = row
        nr = {a: (row[a]["narrative_relation"] if row[a] else "FAIL") for a in ARMS}
        print(f"  {i:>2}/{len(gold_ids)}  {cid:<32} "
              f"A={nr['A_local']:<12} B={nr['B_scenes']:<12} C={nr['C_global']:<12} D={nr['D_both']}")
    return out


def print_report(gold: dict, results: dict):
    print("\n" + "=" * 100)
    print("PER-FIELD ACCURACY / F1 ACROSS ALL FOUR ARMS")
    print("=" * 100)

    header = f"{'field':<20}" + "".join(f"{a:<28}" for a in ARMS)
    print("\n" + header)
    for field in SINGLE_LABEL_FIELDS:
        row = f"{field:<20}"
        for arm in ARMS:
            preds = {cid: results[cid].get(arm) for cid in gold}
            s = score_single_label(field, gold, preds)
            row += f"{pct(s['accuracy']):<28}"
        print(row)
    for field in SET_FIELDS:
        row = f"{field + ' (F1)':<20}"
        for arm in ARMS:
            preds = {cid: results[cid].get(arm) for cid in gold}
            s = score_set_field(field, gold, preds)
            row += f"{s['micro']['f1']:.2f}{'':<24}"
        print(row)

    print("\n" + "-" * 100)
    print("narrative_relation detail (the field this ablation is really about)")
    print("-" * 100)
    for arm in ARMS:
        preds = {cid: results[cid].get(arm) for cid in gold}
        s = score_single_label("narrative_relation", gold, preds)
        print(f"\n{ARM_LABELS[arm]}  (accuracy {pct(s['accuracy'])})")
        for cls, m in s["per_class"].items():
            print(f"   {str(cls):<14}{'support='+str(m['support']):<12}"
                  f"P={pct(m['precision']):<8}R={pct(m['recall']):<8}F1={m['f1']:.2f}")

    print("\n" + "-" * 100)
    print("canonical_themes: unrequited-longing over-application (gold has 0/30) per arm")
    print("-" * 100)
    for arm in ARMS:
        leaks = [cid for cid in gold if results[cid].get(arm)
                 and "unrequited-longing" in results[cid][arm]["canonical_themes"]]
        print(f"  {ARM_LABELS[arm]:<32} {len(leaks)}/30")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    gold = load_gold()

    if args.score_only:
        if not CACHE_PATH.exists():
            print(f"No cache at {CACHE_PATH} -- run without --score-only first.")
            sys.exit(1)
        results = json.loads(CACHE_PATH.read_text())
    else:
        print(f"Tagging {len(gold)} chunks x 4 arms...")
        results = run_tagger(list(gold.keys()))
        CACHE_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nCached raw output -> {CACHE_PATH}")

    print_report(gold, results)


if __name__ == "__main__":
    main()
