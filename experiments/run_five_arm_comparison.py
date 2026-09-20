"""Adds arm E (the 3-pass pipeline, src/tag_three_pass.py) to the A-D four-arm ablation and
scores all five against gold.

    A - local        = TARGET + prev/next chunk only
    B - +scenes       = TARGET + scene cards, no global map
    C - +global map   = TARGET + global map, no scene cards
    D - both          = TARGET + scene cards + global map, ONE pass, ONE prompt (production)
    E - three_pass    = 3 SEPARATE passes, each field routed to the context arm F8 showed it
                         needs: local pass (speaker fields), scene pass (themes/characters
                         candidates), global pass (narrative_relation + adjudicated
                         themes/characters)

Reuses the cached A-D output from run_four_arm_ablation.py (no need to re-spend API calls on
those) and only tags the NEW arm E.

Run from repo root:
    .venv/bin/python experiments/run_five_arm_comparison.py                # tag E + score all 5
    .venv/bin/python experiments/run_five_arm_comparison.py --score-only   # reuse cached E too
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from chunking import load_chunks
from ingest import _scene_context
from scenes import render_global_map
from tag_three_pass import tag_three_pass

from run_first30_eval import SET_FIELDS, SINGLE_LABEL_FIELDS, load_gold, pct, score_set_field, score_single_label

FOUR_ARM_CACHE = ROOT / "experiments" / "four_arm_ablation_output.json"
FIVE_ARM_CACHE = ROOT / "experiments" / "five_arm_comparison_output.json"
ARMS = ["A_local", "B_scenes", "C_global", "D_both", "E_three_pass"]
ARM_LABELS = {
    "A_local": "A - local (prev/next only)",
    "B_scenes": "B - +scenes (no global map)",
    "C_global": "C - +global map (no scenes)",
    "D_both": "D - both, one pass (production)",
    "E_three_pass": "E - three separate passes (routed)",
}


def tag_arm_e(gold_ids: list) -> dict:
    chunks = {c["id"]: c for c in load_chunks()}
    gm_str = render_global_map(json.loads((ROOT / "context" / "global_map.json").read_text()))
    scenes = json.loads((ROOT / "context" / "scenes.json").read_text())
    cards = json.loads((ROOT / "context" / "scene_cards.json").read_text())
    scene_of = {cid: i for i, s in enumerate(scenes) for cid in s["chunk_ids"]}

    out = {}
    for i, cid in enumerate(gold_ids, 1):
        c = chunks[cid]
        prev = chunks[c["prev_id"]]["text"] if c["prev_id"] else None
        nxt = chunks[c["next_id"]]["text"] if c["next_id"] else None
        sc = _scene_context(cards, scene_of[cid])
        try:
            tags = tag_three_pass(c["text"], prev, nxt, sc, gm_str)
        except Exception as e:
            print(f"  ! three-pass failed for {cid}: {e}")
            tags = None
        out[cid] = tags
        nr = tags["narrative_relation"] if tags else "FAIL"
        print(f"  {i:>2}/{len(gold_ids)}  {cid:<32} E narrative_relation={nr}")
    return out


def print_report(gold: dict, results: dict):
    print("\n" + "=" * 110)
    print("FIVE-ARM COMPARISON: A local · B +scenes · C +global · D both · E three-pass (routed)")
    print("=" * 110)

    header = f"{'field':<20}" + "".join(f"{a:<24}" for a in ARMS)
    print("\n" + header)
    for field in SINGLE_LABEL_FIELDS:
        row = f"{field:<20}"
        for arm in ARMS:
            preds = {cid: results[cid].get(arm) for cid in gold}
            s = score_single_label(field, gold, preds)
            row += f"{pct(s['accuracy']):<24}"
        print(row)
    for field in SET_FIELDS:
        row = f"{field + ' (F1)':<20}"
        for arm in ARMS:
            preds = {cid: results[cid].get(arm) for cid in gold}
            s = score_set_field(field, gold, preds)
            row += f"{s['micro']['f1']:.2f}{'':<20}"
        print(row)

    print("\n" + "-" * 110)
    print("narrative_relation detail")
    print("-" * 110)
    for arm in ARMS:
        preds = {cid: results[cid].get(arm) for cid in gold}
        s = score_single_label("narrative_relation", gold, preds)
        print(f"\n{ARM_LABELS[arm]}  (accuracy {pct(s['accuracy'])})")
        for cls, m in s["per_class"].items():
            print(f"   {str(cls):<14}{'support='+str(m['support']):<12}"
                  f"P={pct(m['precision']):<8}R={pct(m['recall']):<8}F1={m['f1']:.2f}")

    print("\n" + "-" * 110)
    print("speaker_relation detail (the field every context arm hurt in F8)")
    print("-" * 110)
    for arm in ARMS:
        preds = {cid: results[cid].get(arm) for cid in gold}
        s = score_single_label("speaker_relation", gold, preds)
        print(f"  {ARM_LABELS[arm]:<40} accuracy={pct(s['accuracy'])}")

    print("\n" + "-" * 110)
    print("canonical_themes: unrequited-longing over-application (gold has 0/30) per arm")
    print("-" * 110)
    for arm in ARMS:
        leaks = [cid for cid in gold if results[cid].get(arm)
                 and "unrequited-longing" in results[cid][arm]["canonical_themes"]]
        print(f"  {ARM_LABELS[arm]:<40} {len(leaks)}/30")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    gold = load_gold()
    if not FOUR_ARM_CACHE.exists():
        print(f"No A-D cache at {FOUR_ARM_CACHE} -- run run_four_arm_ablation.py first.")
        sys.exit(1)
    four_arm = json.loads(FOUR_ARM_CACHE.read_text())

    if args.score_only:
        if not FIVE_ARM_CACHE.exists():
            print(f"No E cache at {FIVE_ARM_CACHE} -- run without --score-only first.")
            sys.exit(1)
        arm_e = json.loads(FIVE_ARM_CACHE.read_text())
    else:
        print(f"Tagging {len(gold)} chunks with the three-pass pipeline (arm E)...")
        arm_e = tag_arm_e(list(gold.keys()))
        FIVE_ARM_CACHE.write_text(json.dumps(arm_e, indent=2, ensure_ascii=False))
        print(f"\nCached arm E output -> {FIVE_ARM_CACHE}")

    results = {cid: {**four_arm[cid], "E_three_pass": arm_e.get(cid)} for cid in gold}
    print_report(gold, results)


if __name__ == "__main__":
    main()
