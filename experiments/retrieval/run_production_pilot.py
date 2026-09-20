"""Runs src/production_method.py's full pipeline (classify -> weighted fusion ->
cheap rerank -> diversify) over the 10-question gold set and scores hit@6.

Used across multiple tuning rounds (experiments/retrieval/FINDINGS.md R4) — pass
--round N to label the output file so each round's numbers are preserved, not
overwritten. No LLM calls except the one embedding call per question (dense scoring) —
this whole pipeline is designed to be cheap to re-run many times while tuning.

Run from repo root:
    .venv/bin/python experiments/retrieval/run_production_pilot.py --round 0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import production_method as pm  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval" / "gold" / "retrieval_gold_pilot10.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()

    questions = json.loads(GOLD_PATH.read_text())
    results = {"k": K, "round": args.round, "weights": pm.WEIGHTS, "questions": []}

    for q in questions:
        ids, debug = pm.production_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "gold_intent": q["intent"],
               "classified_intent": debug["intent"], "weights_used": debug["weights"],
               "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)
        mark = f"hit@{rank}" if hit else "miss"
        print(f"{q['id']:<4} [{debug['intent']:<24}] {mark:<8} {q['question'][:65]}")

    out_path = ROOT / "experiments" / "retrieval" / "results" / f"production_pilot_round{args.round}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    print(f"\nWrote {out_path}")
    print(f"=== round {args.round} hit@6: {hits}/{len(questions)} ===")


if __name__ == "__main__":
    main()
