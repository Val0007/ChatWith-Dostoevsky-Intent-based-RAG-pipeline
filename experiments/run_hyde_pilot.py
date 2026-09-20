"""Runs src/production_with_hyde.py (HyDE hypothetical passage -> dense retrieval ->
union with original query's own fuse() top-N -> cheap_rerank anchored on original query)
over the 10-question gold set.

Run from repo root:
    .venv/bin/python experiments/run_hyde_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_with_hyde import production_hyde_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "hyde_pilot_results.json"
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
        ids, debug = production_hyde_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "intent": debug["intent"],
               "hyde_passage": debug["hyde_passage"], "union_size": debug["union_size"],
               "new_from_hyde": debug["new_from_hyde"], "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        pm = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        note = ""
        if not prod_by_id[q["id"]]["hit"] and hit:
            note = "  <-- NEW HIT vs production_method"
        elif prod_by_id[q["id"]]["hit"] and not hit:
            note = "  <-- REGRESSED vs production_method"
        print(f"{q['id']:<4} hyde={mark:<8} (prod={pm:<8}) new_from_hyde={debug['new_from_hyde']:<3}{note}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])

    print(f"\nWrote {RESULTS_PATH}")
    print("=== hit@6 summary ===")
    print(f"  production_method (R4):     {prod_hits}/10")
    print(f"  production_with_hyde:       {hits}/10")


if __name__ == "__main__":
    main()
