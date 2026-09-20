"""Runs experiments/retrieval/variants/production_with_dot_product.py (production_method's candidate pool, but
reranked by pure embedding dot-product instead of cheap_rerank's formula) over the
10-question gold set.

Run from repo root:
    .venv/bin/python experiments/retrieval/run_dotproduct_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "retrieval" / "variants"))

from production_with_dot_product import production_dotproduct_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval" / "gold" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "dotproduct_pilot_results.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "production_pilot_final.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    prod = json.loads(PROD_RESULTS_PATH.read_text())
    prod_by_id = {row["id"]: row for row in prod["questions"]}

    results = {"k": K, "questions": []}
    for q in questions:
        ids, debug = production_dotproduct_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "intent": debug["intent"],
               "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        pm = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        note = ""
        if not prod_by_id[q["id"]]["hit"] and hit:
            note = "  <-- NEW HIT vs production_method (cheap_rerank)"
        elif prod_by_id[q["id"]]["hit"] and not hit:
            note = "  <-- REGRESSED vs production_method (cheap_rerank)"
        print(f"{q['id']:<4} dotproduct={mark:<8} (cheap_rerank={pm:<8}){note}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])

    print(f"\nWrote {RESULTS_PATH}")
    print("=== hit@6 summary ===")
    print(f"  production_method (cheap_rerank, R4):  {prod_hits}/10")
    print(f"  production_with_dot_product:            {hits}/10")


if __name__ == "__main__":
    main()
