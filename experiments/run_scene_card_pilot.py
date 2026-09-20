"""Runs src/scene_card_production.py's scene-cards-first pipeline over the 10-question
gold set and scores hit@6. Tests whether routing through scene-card summaries first
(paraphrased prose, not the raw passage) recovers the paraphrase-gap misses documented in
experiments/findings_retrieval.md R1-A / R4 (q01, q02, q10).

Run from repo root:
    .venv/bin/python experiments/run_scene_card_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scene_card_production import scene_card_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "scene_card_pilot_results.json"
PROD_RESULTS_PATH = ROOT / "experiments" / "production_pilot_final.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    prod = json.loads(PROD_RESULTS_PATH.read_text()) if PROD_RESULTS_PATH.exists() else None
    prod_by_id = {row["id"]: row for row in prod["questions"]} if prod else {}

    results = {"k": K, "questions": []}
    for q in questions:
        ids, debug = scene_card_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        gold_scenes = set()
        row = {"id": q["id"], "night": q["night"], "top_scenes": debug["top_scenes"],
               "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        prod_row = prod_by_id.get(q["id"], {})
        prod_mark = f"hit@{prod_row['rank']}" if prod_row.get("hit") else "miss"
        flip = ""
        if prod_row and prod_row["hit"] != hit:
            flip = "  <-- FLIPPED vs production_method" if hit else "  <-- REGRESSED vs production_method"
        print(f"{q['id']:<4} {mark:<8} (production_method: {prod_mark}){flip}  {q['question'][:60]}")
        print(f"      top_scenes={debug['top_scenes']}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    hits = sum(1 for row in results["questions"] if row["hit"])
    print(f"\nWrote {RESULTS_PATH}")
    print(f"=== scene_card hit@6: {hits}/{len(questions)} ===")
    if prod:
        phits = sum(1 for row in prod["questions"] if row["hit"])
        print(f"=== production_method (R4) hit@6: {phits}/{len(questions)} ===")


if __name__ == "__main__":
    main()
