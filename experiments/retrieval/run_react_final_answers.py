"""Final deliverable: run all 10 gold questions through the fixed production_react
(v4 prompt + Lookup-confirmation fix, R23/R24) and pair each generated answer with our
own expected/gold answer for direct comparison.

Run from repo root:
    .venv/bin/python experiments/retrieval/run_react_final_answers.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from production_react import react_answer  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval" / "gold" / "retrieval_gold_pilot10.json"
RESULTS_PATH = ROOT / "experiments" / "retrieval" / "results" / "react_final_answers.json"


def main():
    questions = json.loads(GOLD_PATH.read_text())
    results = []
    for q in questions:
        print(f"Running {q['id']}...", flush=True)
        llm_answer, debug = react_answer(q["question"])
        results.append({
            "id": q["id"], "night": q["night"], "intent": q["intent"],
            "question": q["question"], "expected_answer": q["expected_answer"],
            "gold_evidence": q["gold_evidence"], "llm_answer": llm_answer,
            "steps": debug["steps"], "encountered": debug["encountered"],
            "confirmed": debug["confirmed"],
        })
        print(f"  done ({debug['steps']} steps, {debug['confirmed']} confirmed via Lookup)", flush=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
