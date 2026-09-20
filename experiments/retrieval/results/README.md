# results/

Per-method output of the `run_*.py` pilots in the parent folder — typically, for each of the 10 questions: the intent chosen, the retrieved chunk ids, whether a gold chunk was in the top 6 (`hit`) and its `rank` (the shape varies a little per method). See the scoreboard in [`../README.md`](../README.md) for what each file belongs to.

| Files | Belong to |
|---|---|
| `retrieval_pilot_results.json`, `router_pilot_results.json`, `plan_pilot_results.json` | R1–R3 (presets, router, diagnostician) |
| `production_pilot_round0.json`, `…_round1.json`, `…_final.json` | R4 weight-tuning rounds; `_final` is the last full run with the locked weights |
| `scene_card_pilot_results.json`, `union_pilot_round0.json`, `…_round2.json`, `…_final.json` | R5–R6 |
| `judge_ablation_results.json` | R7 (both LLM-judge ablations) |
| `llm_query_pilot_results.json`, `production_union_llm_pilot_results.json`, `production_union_llm_v3_pilot_results.json` | R8–R12 |
| `encoder_pilot_results.json`, `hyde_pilot_results.json`, `dotproduct_pilot_results.json` | R13, R14, dot-product |
| `llm_retries_pilot_results.json` | R16–R18 |
| `react_pilot_results.json`, `react_final_answers.json` | R20–R24 (`react_final_answers.json` pairs each generated answer with its expected answer) |
