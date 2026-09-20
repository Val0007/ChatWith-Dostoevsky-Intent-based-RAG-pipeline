"""Runs src/production_llm_retries.py (pure production retrieval -> LLM sufficiency
check -> conditional second retrieval round) over the 10-question gold set.

Run from repo root:
    .venv/bin/python experiments/run_llm_retries_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_llm_retries import production_retries_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "llm_retries_pilot_results.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "production_pilot_final.json"

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
        ids, debug = production_retries_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "intent": debug["intent"],
               "retried": debug["retried"], "attempts": debug["attempts"],
               "union_size": debug["union_size"], "checks": debug["checks"],
               "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        pm = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        note = ""
        if not prod_by_id[q["id"]]["hit"] and hit:
            note = "  <-- NEW HIT vs production_method"
        elif prod_by_id[q["id"]]["hit"] and not hit:
            note = "  <-- REGRESSED vs production_method"
        print(f"{q['id']:<4} retries={mark:<8} (prod={pm:<8}) attempts={debug['attempts']}{note}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])
    n_retried = sum(1 for row in results["questions"] if row["retried"])

    print(f"\nWrote {RESULTS_PATH}")
    print(f"=== hit@6 summary (retried on {n_retried}/10 questions) ===")
    print(f"  production_method (R4):    {prod_hits}/10")
    print(f"  production_llm_retries:    {hits}/10")


if __name__ == "__main__":
    main()
