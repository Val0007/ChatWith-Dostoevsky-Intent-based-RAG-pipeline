"""Runs BOTH cross-encoder pipelines (experiments/retrieval/variants/production_with_encoder.py,
experiments/retrieval/variants/production_with_llm_variants.py) over the 10-question gold set and compares against
the cheap_rerank baselines (production_method R4, production_union_with_llm v3 R11).

Run from repo root:
    .venv/bin/python experiments/retrieval/run_encoder_pilots.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "retrieval" / "variants"))

from production_with_encoder import production_encoder_retrieve  # noqa: E402
from production_with_llm_variants import production_llm_variants_encoder_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval" / "gold" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "encoder_pilot_results.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "production_pilot_final.json"
V3_RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "production_union_llm_v3_pilot_results.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    prod = json.loads(PROD_RESULTS_PATH.read_text())
    v3 = json.loads(V3_RESULTS_PATH.read_text())
    prod_by_id = {row["id"]: row for row in prod["questions"]}
    v3_by_id = {row["id"]: row for row in v3["questions"]}

    results = {"k": K, "questions": []}
    for q in questions:
        e_ids, e_debug = production_encoder_retrieve(q["question"], k=K)
        e_hit, e_rank = _score(e_ids, q["gold_evidence"])

        v_ids, v_debug = production_llm_variants_encoder_retrieve(q["question"], k=K)
        v_hit, v_rank = _score(v_ids, q["gold_evidence"])

        row = {"id": q["id"], "night": q["night"],
               "encoder": {"retrieved": e_ids, "hit": e_hit, "rank": e_rank},
               "encoder_variants": {"retrieved": v_ids, "hit": v_hit, "rank": v_rank,
                                     "alternatives": v_debug["alternatives"]}}
        results["questions"].append(row)

        pm = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        v3m = "hit@" + str(v3_by_id[q["id"]]["rank"]) if v3_by_id[q["id"]]["hit"] else "miss"
        em = "hit@" + str(e_rank) if e_hit else "miss"
        vm = "hit@" + str(v_rank) if v_hit else "miss"
        print(f"{q['id']:<4} cheap_rerank(prod={pm:<8} v3={v3m:<8}) | cross-encoder(plain={em:<8} +variants={vm:<8})")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    e_hits = sum(1 for r in results["questions"] if r["encoder"]["hit"])
    v_hits = sum(1 for r in results["questions"] if r["encoder_variants"]["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])
    v3_hits = sum(1 for r in v3["questions"] if r["hit"])

    print(f"\nWrote {RESULTS_PATH}")
    print("=== hit@6 summary ===")
    print(f"  production_method (cheap_rerank, R4):            {prod_hits}/10")
    print(f"  production_union_with_llm v3 (cheap_rerank, R11): {v3_hits}/10")
    print(f"  production_with_encoder (cross-encoder):          {e_hits}/10")
    print(f"  production_with_llm_variants (cross-encoder):     {v_hits}/10")


if __name__ == "__main__":
    main()
