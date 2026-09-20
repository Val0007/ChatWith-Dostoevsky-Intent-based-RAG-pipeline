"""Runs src/union_method.py (direct fusion UNION scene-card candidates -> cheap rerank)
over the 10-question gold set. Used across tuning rounds (experiments/findings_retrieval.md
R6) — pass --round N to label output so each round's numbers are preserved.

Run from repo root:
    .venv/bin/python experiments/run_union_pilot.py --round 0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import union_method as um  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "production_pilot_final.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--n-scenes", type=int, default=6)
    args = ap.parse_args()

    questions = json.loads(GOLD_PATH.read_text())
    prod = json.loads(PROD_RESULTS_PATH.read_text()) if PROD_RESULTS_PATH.exists() else None
    prod_by_id = {row["id"]: row for row in prod["questions"]} if prod else {}

    results = {"k": K, "round": args.round, "n_scenes": args.n_scenes,
               "scene_field_weights": um.SCENE_FIELD_WEIGHTS, "questions": []}

    for q in questions:
        ids, debug = um.union_retrieve(q["question"], k=K, n_scenes=args.n_scenes)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "intent": debug["intent"],
               "top_scenes": debug["top_scenes"], "new_from_scenes": debug["new_from_scenes"],
               "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        prod_row = prod_by_id.get(q["id"], {})
        prod_mark = f"hit@{prod_row['rank']}" if prod_row.get("hit") else "miss"
        flip = ""
        if prod_row and prod_row["hit"] != hit:
            flip = "  <-- FLIPPED vs production_method" if hit else "  <-- REGRESSED vs production_method"
        print(f"{q['id']:<4} {mark:<8} (prod: {prod_mark}) new_from_scenes={debug['new_from_scenes']:<3}{flip}  {q['question'][:55]}")

    out_path = ROOT / "experiments" / f"union_pilot_round{args.round}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    print(f"\nWrote {out_path}")
    print(f"=== round {args.round} union hit@6: {hits}/{len(questions)} ===")
    if prod:
        phits = sum(1 for row in prod["questions"] if row["hit"])
        print(f"=== production_method (R4) hit@6: {phits}/{len(questions)} ===")


if __name__ == "__main__":
    main()
