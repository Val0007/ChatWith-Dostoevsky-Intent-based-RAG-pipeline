"""8th method: an LLM router. Instead of a fixed preset for every question, ask an LLM
(retrieval.choose_method) to look at the SPECIFIC question + the chunk schema and pick
which of the 7 presets fits — then run that preset. This is a real (if simple) stand-in
for the query-intent routing Phase 01 was supposed to add, one step up from the crude
token-overlap `_metadata_boost` used in R1.

Run from repo root:
    .venv/bin/python experiments/run_router_pilot.py

Compares against the fixed-preset numbers already cached in
experiments/retrieval_pilot_results.json (from run_retrieval_pilot.py) rather than
re-running those — same gold set, same k, so the numbers are directly comparable.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import PRESETS, auto_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
FIXED_RESULTS_PATH = ROOT / "experiments" / "retrieval_pilot_results.json"
RESULTS_PATH = ROOT / "experiments" / "router_pilot_results.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    fixed = json.loads(FIXED_RESULTS_PATH.read_text()) if FIXED_RESULTS_PATH.exists() else None
    fixed_by_id = {row["id"]: row for row in fixed["questions"]} if fixed else {}

    results = {"k": K, "questions": []}
    for q in questions:
        ids, choice = auto_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {
            "id": q["id"], "night": q["night"], "intent": q["intent"],
            "chosen_method": choice["method"], "reason": choice["reason"],
            "retrieved": ids, "hit": hit, "rank": rank,
        }
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        fixed_mark = ""
        if q["id"] in fixed_by_id:
            fp = fixed_by_id[q["id"]]["presets"].get(choice["method"], {})
            fixed_mark = f" (fixed '{choice['method']}' preset alone got: {'hit@' + str(fp['rank']) if fp.get('hit') else 'miss'})"
        print(f"{q['id']} [{q['night']} / {q['intent']}] -> router picked '{choice['method']}': {mark}{fixed_mark}")
        print(f"    reason: {choice['reason']}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {RESULTS_PATH}")

    hits = sum(1 for row in results["questions"] if row["hit"])
    print(f"\n=== router hit@6: {hits}/{len(questions)} ===")

    if fixed:
        print("\n=== for comparison, fixed-preset hit@6 from R1 ===")
        for name in PRESETS:
            fhits = sum(1 for row in fixed["questions"] if row["presets"][name]["hit"])
            print(f"  {name:<10} {fhits}/{len(questions)}")
        print(f"  {'router':<10} {hits}/{len(questions)}")


if __name__ == "__main__":
    main()
