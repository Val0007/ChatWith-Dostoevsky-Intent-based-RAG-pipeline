"""9th method: a structured diagnostician instead of a single-label router. Where
choose_method() (R2) picks one of the 7 fixed presets, diagnose_query() analyzes the
question along several independent axes (expected lexical/semantic overlap, whether
metadata would help, multi-hop need, rerank need) and composes its OWN ordered plan out
of primitive operations (bm25/dense/hybrid/metadata_filter/rerank/narrative) via
run_plan(). Not limited to the 7 pre-named combinations.

Run from repo root:
    .venv/bin/python experiments/run_plan_pilot.py

Compares against experiments/retrieval_pilot_results.json (R1, fixed presets) and
experiments/router_pilot_results.json (R2, single-label router) — same gold set, same k.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import PRESETS, plan_retrieve  # noqa: E402

GOLD_PATH = ROOT / "experiments" / "retrieval_gold_pilot10.json"
FIXED_RESULTS_PATH = ROOT / "experiments" / "retrieval_pilot_results.json"
ROUTER_RESULTS_PATH = ROOT / "experiments" / "router_pilot_results.json"
RESULTS_PATH = ROOT / "experiments" / "plan_pilot_results.json"

K = 6


def _score(retrieved: list[str], gold: list[str]) -> tuple[bool, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return True, rank
    return False, None


def main():
    questions = json.loads(GOLD_PATH.read_text())
    fixed = json.loads(FIXED_RESULTS_PATH.read_text()) if FIXED_RESULTS_PATH.exists() else None
    router = json.loads(ROUTER_RESULTS_PATH.read_text()) if ROUTER_RESULTS_PATH.exists() else None
    fixed_by_id = {row["id"]: row for row in fixed["questions"]} if fixed else {}
    router_by_id = {row["id"]: row for row in router["questions"]} if router else {}

    results = {"k": K, "questions": []}
    for q in questions:
        ids, diag = plan_retrieve(q["question"], k=K)
        hit, rank = _score(ids, q["gold_evidence"])
        row = {"id": q["id"], "night": q["night"], "intent": q["intent"],
               "diagnosis": diag, "retrieved": ids, "hit": hit, "rank": rank}
        results["questions"].append(row)

        mark = f"hit@{rank}" if hit else "miss"
        print(f"{q['id']} [{q['night']} / {q['intent']}] plan={diag['recommended_plan']} -> {mark}")
        print(f"    intent={diag['intent']!r} lexical={diag['lexical_overlap_expected']} "
              f"semantic={diag['semantic_similarity_expected']} metadata={diag['metadata_usefulness']} "
              f"multi_hop={diag['needs_multi_hop']} rerank={diag['needs_reranking']}")
        fx = fixed_by_id.get(q["id"], {}).get("presets", {})
        rt = router_by_id.get(q["id"], {})
        best_fixed = max(fx.items(), key=lambda kv: (kv[1]["hit"], -(kv[1]["rank"] or 99)), default=(None, {}))
        print(f"    (R1 best fixed preset: {best_fixed[0]} {'hit@'+str(best_fixed[1]['rank']) if best_fixed[1].get('hit') else 'miss'}"
              f" | R2 router picked '{rt.get('chosen_method')}': {'hit@'+str(rt['rank']) if rt.get('hit') else 'miss'})")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {RESULTS_PATH}")

    hits = sum(1 for row in results["questions"] if row["hit"])
    print(f"\n=== plan hit@6: {hits}/{len(questions)} ===")

    if fixed:
        print("\n=== all methods so far, hit@6 ===")
        for name in PRESETS:
            fhits = sum(1 for row in fixed["questions"] if row["presets"][name]["hit"])
            print(f"  {name:<10} {fhits}/{len(questions)}")
        if router:
            rhits = sum(1 for row in router["questions"] if row["hit"])
            print(f"  {'router':<10} {rhits}/{len(questions)}")
        print(f"  {'plan':<10} {hits}/{len(questions)}")


if __name__ == "__main__":
    main()
