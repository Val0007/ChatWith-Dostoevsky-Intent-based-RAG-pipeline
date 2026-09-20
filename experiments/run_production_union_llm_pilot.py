"""Runs src/production_union_with_llm.py (original query + up to 4 LLM-generated
alternatives, union by max fused score, cheap rerank) over the 10-question gold set.

Run from repo root:
    .venv/bin/python experiments/run_production_union_llm_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_union_with_llm import production_union_llm_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "production_union_llm_pilot_results.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "production_pilot_final.json"
UNION_RESULTS_PATH = ROOT / "experiments" / "union_pilot_final.json"
LLM_QUERY_RESULTS_PATH = ROOT / "experiments" / "llm_query_pilot_results.json"

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
    llm_query = json.loads(LLM_QUERY_RESULTS_PATH.read_text())
    prod_by_id = {row["id"]: row for row in prod["questions"]}
    union_by_id = {row["id"]: row for row in union["questions"]}
    llmq_by_id = {row["id"]: row for row in llm_query["questions"]}

    results = {"k": K, "questions": []}
    for q in questions:
        ids, debug = production_union_llm_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "intent": debug["intent"],
               "alternatives": debug["alternatives"], "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        pm = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        um = "hit@" + str(union_by_id[q["id"]]["rank"]) if union_by_id[q["id"]]["hit"] else "miss"
        lm = "hit@" + str(llmq_by_id[q["id"]]["rank"]) if llmq_by_id[q["id"]]["hit"] else "miss"
        note = ""
        if not prod_by_id[q["id"]]["hit"] and hit:
            note = "  <-- NEW HIT vs production_method"
        elif prod_by_id[q["id"]]["hit"] and not hit:
            note = "  <-- REGRESSED vs production_method"
        print(f"{q['id']:<4} this={mark:<8} (prod={pm:<8} union={um:<8} llm_query={lm:<8}){note}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])
    union_hits = sum(1 for r in union["questions"] if r["hit"])
    llmq_hits = sum(1 for r in llm_query["questions"] if r["hit"])

    print(f"\nWrote {RESULTS_PATH}")
    print("=== hit@6 summary ===")
    print(f"  production_method (R4):        {prod_hits}/10")
    print(f"  union_method (R6):             {union_hits}/10")
    print(f"  llm_query_production (R8/R9):  {llmq_hits}/10")
    print(f"  production_union_with_llm:     {hits}/10")


if __name__ == "__main__":
    main()
