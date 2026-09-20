"""Phase 00 smoke test on real gold data: run all 7 retrieval presets over the 10-question
pilot (experiments/retrieval_gold_pilot10.json) and score whether each config's top-k
surfaces the gold evidence chunk(s) for that question.

Metric: hit@k (did >=1 gold evidence id land in the top k=6?) and the rank of the first
gold hit (1-indexed; None if missed). This is retrieval-only — no answer generation, no
LLM-as-judge grading — so it is cheap to re-run and has no rubric to argue with.

Run from repo root:
    .venv/bin/python experiments/run_retrieval_pilot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import PRESETS, retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "retrieval_pilot_results.json"

K = 6


# example ─ in:  (["white_nights_fourth_night_011", "white_nights_fourth_night_012", ...], ["white_nights_fourth_night_012"])
#           out: (True, 2)   # hit, first gold id found at rank 2 (1-indexed)
def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    results = {"k": K, "questions": []}

    for q in questions:
        print(f"{q['id']} [{q['night']} / {q['intent']}] {q['question'][:70]}")
        row = {"id": q["id"], "night": q["night"], "intent": q["intent"], "presets": {}}
        for name in PRESETS:
            retrieved = retrieve(q["question"], preset=name, k=K)
            hit, rank = _score(retrieved, q["gold_evidence"])
            row["presets"][name] = {"retrieved": retrieved, "hit": hit, "rank": rank}
            mark = f"hit@{rank}" if hit else "miss"
            print(f"    {name:<10} {mark}")
        results["questions"].append(row)
        print()

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {RESULTS_PATH}")

    # ---- summary table ----
    print("\n=== hit@6 rate per preset ===")
    for name in PRESETS:
        hits = sum(1 for row in results["questions"] if row["presets"][name]["hit"])
        print(f"  {name:<10} {hits}/{len(questions)}")


if __name__ == "__main__":
    main()
