"""Runs BOTH LLM-judge ablations (src/production_llm_judge.py, experiments/retrieval/variants/scene_card_llm_judge.py)
over the 10-question gold set and scores hit@6 against production_method (R4) and
union_method (R6), the two best fusion+cheap-rerank methods so far.

Run from repo root:
    .venv/bin/python experiments/retrieval/run_judge_ablation_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "retrieval" / "variants"))

from production_llm_judge import production_judge_retrieve  # noqa: E402
from scene_card_llm_judge import scene_card_judge_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval" / "gold" / "retrieval_gold_pilot10.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "production_pilot_final.json"
UNION_RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "union_pilot_final.json"

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
        p_ids, p_debug = production_judge_retrieve(q["question"], k=K)
        p_hit, p_rank = _score(p_ids, q["gold_evidence"])

        s_ids, s_debug = scene_card_judge_retrieve(q["question"], k=K)
        s_hit, s_rank = _score(s_ids, q["gold_evidence"])

        row = {"id": q["id"], "night": q["night"],
               "production_judge": {"retrieved": p_ids, "hit": p_hit, "rank": p_rank, "debug": p_debug},
               "scene_card_judge": {"retrieved": s_ids, "hit": s_hit, "rank": s_rank, "debug": s_debug}}
        results["questions"].append(row)

        prod_mark = "hit@" + str(prod_by_id[q["id"]]["rank"]) if prod_by_id[q["id"]]["hit"] else "miss"
        union_mark = "hit@" + str(union_by_id[q["id"]]["rank"]) if union_by_id[q["id"]]["hit"] else "miss"
        p_mark = "hit@" + str(p_rank) if p_hit else "miss"
        s_mark = "hit@" + str(s_rank) if s_hit else "miss"
        print(f"{q['id']:<4} prod(fusion)={prod_mark:<8} union(fusion)={union_mark:<8} "
              f"prod+judge={p_mark:<8} scene+judge={s_mark:<8}  {q['question'][:45]}")

    out_path = ROOT / "experiments" / "retrieval" / "results" / "judge_ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    p_hits = sum(1 for r in results["questions"] if r["production_judge"]["hit"])
    s_hits = sum(1 for r in results["questions"] if r["scene_card_judge"]["hit"])
    prod_hits = sum(1 for r in prod["questions"] if r["hit"])
    union_hits = sum(1 for r in union["questions"] if r["hit"])

    print(f"\nWrote {out_path}")
    print("=== hit@6 summary ===")
    print(f"  production_method  (fusion + cheap rerank): {prod_hits}/10")
    print(f"  union_method       (fusion + cheap rerank): {union_hits}/10")
    print(f"  production_llm_judge (union + LLM judge):   {p_hits}/10")
    print(f"  scene_card_llm_judge (union + LLM judge):   {s_hits}/10")


if __name__ == "__main__":
    main()
