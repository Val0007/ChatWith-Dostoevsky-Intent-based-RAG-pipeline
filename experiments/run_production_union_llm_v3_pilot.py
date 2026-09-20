"""Runs src/production_union_with_llm.py v3 (per-variant candidate ID union, final
ranking anchored on the ORIGINAL query alone) over the 10-question gold set. Compares
against v2 (max-across-variants scoring, R10) and the session's best (production_method /
union_method, 6/10).

Run from repo root:
    .venv/bin/python experiments/run_production_union_llm_v3_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_union_with_llm import production_union_llm_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "production_union_llm_v3_pilot_results.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "production_pilot_final.json"
V2_RESULTS_PATH = ROOT / "experiments" / "production_union_llm_pilot_results.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    prod = json.loads(PROD_RESULTS_PATH.read_text())
    v2 = json.loads(V2_RESULTS_PATH.read_text())
    prod_by_id = {row["id"]: row for row in prod["questions"]}
    v2_by_id = {row["id"]: row for row in v2["questions"]}

    results = {"k": K, "questions": []}
    for q in questions:
        ids, debug = production_union_llm_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "intent": debug["intent"],
               "alternatives": debug["alternatives"], "union_size": debug["union_size"],
               "convergence": debug["convergence"], "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        pm = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        v2m = "hit@" + str(v2_by_id[q["id"]]["rank"]) if v2_by_id[q["id"]]["hit"] else "miss"
        note = ""
        if not prod_by_id[q["id"]]["hit"] and hit:
            note = "  <-- NEW HIT vs production_method"
        elif prod_by_id[q["id"]]["hit"] and not hit:
            note = "  <-- REGRESSED vs production_method"
        conv_gold = {cid: c for cid, c in debug["convergence"].items()}
        print(f"{q['id']:<4} v3={mark:<8} (prod={pm:<8} v2={v2m:<8}){note}")
        print(f"      union_size={debug['union_size']:<3} convergence of retrieved: {conv_gold}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])
    v2_hits = sum(1 for r in v2["questions"] if r["hit"])

    print(f"\nWrote {RESULTS_PATH}")
    print("=== hit@6 summary ===")
    print(f"  production_method (R4):              {prod_hits}/10")
    print(f"  production_union_with_llm v2 (R10):   {v2_hits}/10")
    print(f"  production_union_with_llm v3:         {hits}/10")


if __name__ == "__main__":
    main()
