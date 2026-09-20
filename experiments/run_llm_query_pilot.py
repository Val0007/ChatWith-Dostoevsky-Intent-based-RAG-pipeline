"""Runs src/llm_query_production.py (LLM query rewrite -> production_method's fuse with
externally-supplied intent -> cheap rerank -> diversify) over the 10-question gold set.

Run from repo root:
    .venv/bin/python experiments/run_llm_query_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_query_production import run  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "llm_query_pilot_results.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "production_pilot_final.json"
UNION_RESULTS_PATH = ROOT / "experiments" / "union_pilot_final.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    prod = json.loads(PROD_RESULTS_PATH.read_text())
    union = json.loads(UNION_RESULTS_PATH.read_text())
    prod_by_id = {row["id"]: row for row in prod["questions"]}
    union_by_id = {row["id"]: row for row in union["questions"]}

    results = {"k": K, "questions": []}
    for q in questions:
        ids, debug = run(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "gold_intent": q["intent"],
               "original_intent": debug["original_intent"],
               "rewritten_query": debug["rewrite"]["rewritten_query"],
               "reference_variants": debug["rewrite"]["reference_variants"],
               "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        pm = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        um = "hit@" + str(union_by_id[q["id"]]["rank"]) if union_by_id[q["id"]]["hit"] else "miss"
        flip = ""
        if prod_by_id[q["id"]]["hit"] != hit:
            flip = "  <-- FLIPPED vs production_method" if hit else "  <-- REGRESSED vs production_method"
        print(f"{q['id']:<4} llm_query={mark:<8} (prod={pm:<8} union={um:<8}){flip}")
        print(f"      intent={debug['original_intent']:<12} variants={debug['rewrite']['reference_variants']}")
        print(f"      rewritten: {debug['rewrite']['rewritten_query'][:90]}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])
    union_hits = sum(1 for r in union["questions"] if r["hit"])

    print(f"\nWrote {RESULTS_PATH}")
    print("=== hit@6 summary ===")
    print(f"  production_method (R4): {prod_hits}/10")
    print(f"  union_method (R6):      {union_hits}/10")
    print(f"  llm_query_production (R8/R9): {hits}/10")


if __name__ == "__main__":
    main()
