# experiments/retrieval/

The retrieval study: which strategy finds the right passages for questions about an interpretive text, measured on 10 hand-written questions.

- **Findings log:** [`FINDINGS.md`](FINDINGS.md) (R1–R24). Each entry: what was tested, the result, the takeaway.
- **Metric:** **hit@6** — does any gold evidence chunk land in the top 6 retrieved? Gold set: [`gold/`](gold/).
- **Shared code:** `src/retrieval.py` (candidate generation, judge, diversify) and `src/production_method.py`; everything that was tried and did not beat it is in [`variants/`](variants/).

## Scoreboard

| Method | Module | hit@6 | Round |
|---|---|---|---|
| dense / BM25 / hybrid | `src/retrieval.py` presets | 5/10 | R1 |
| + metadata boost / + judge / + diversify / + expand (the 7-preset ladder) | `src/retrieval.py` presets | 4 / 3 / 4 / 4 | R1 |
| LLM router picks a preset per question | `src/retrieval.py` | 4/10 | R2 |
| LLM diagnostician composes a plan | `src/retrieval.py` | 4/10 | R3 |
| **Production** — intent-weighted fusion + formula rerank | `src/production_method.py` | **6/10** | R4 |
| Scene-card-first routing | `variants/scene_card_production.py` | 5/10 | R5 |
| Union of direct fusion + scene-card candidates | `variants/union_method.py` | 6/10 | R6 |
| Production + LLM judge (no fusion, no formula) | `src/production_llm_judge.py` | 2/10 | R7 |
| Scene-level candidates + LLM judge | `variants/scene_card_llm_judge.py` | 4/10 | R7 |
| LLM query rewriting | `variants/llm_query_production.py` | 4/10 | R8–R9 |
| Original + up to 4 LLM alternative queries | `variants/production_union_with_llm.py` | 5/10 → 6/10 (v3) | R10–R12 |
| Cross-encoder rerank | `variants/production_with_encoder.py`, `production_with_llm_variants.py` | 2/10, 3/10 | R13 |
| HyDE | `variants/production_with_hyde.py` | 6/10 | R14 |
| Embedding dot-product rerank | `variants/production_with_dot_product.py` | 5/10 *(not in FINDINGS; computed from its results file)* | — |
| LLM sufficiency check gates a second round | `variants/production_llm_retries.py` | 6/10 | R16–R18 |
| Token-weighted context annotation | `variants/production_with_weighted_context.py` | answer unchanged | R19 |
| **ReAct loop** (four prompt versions) | `src/production_react.py` | 4 → 5 → 4 → **6**/10 | R20–R24 |

Four questions (q01, q02, q08, q10) are misses for essentially every method — diagnosed against raw per-signal ranks as ceilings of the shared retrieval tool, not tuning gaps. q01 (*"How does the narrator first meet Nastenka?"*) fails because the gold scene never contains the word "Nastenka".

## Runners

Each `run_*.py` runs one method over the 10 gold questions, scores hit@6, and writes to [`results/`](results/). Run from the repo root: `.venv/bin/python experiments/retrieval/<script>.py`.

| Script | Runs | Writes | Cost |
|---|---|---|---|
| `run_retrieval_pilot.py` | the 7 presets | `retrieval_pilot_results.json` | LLM calls (judge presets) |
| `run_router_pilot.py`, `run_plan_pilot.py` | router / diagnostician | `router_…`, `plan_pilot_results.json` | LLM calls |
| `run_production_pilot.py --round N` | `production_method` | `production_pilot_round<N>.json` | 1 embedding call / question |
| `run_scene_card_pilot.py`, `run_union_pilot.py --round N` | scene-card-first, union | `scene_card_…`, `union_pilot_round<N>.json` | embeddings |
| `run_judge_ablation_pilot.py` | both judge ablations | `judge_ablation_results.json` | LLM judge calls |
| `run_llm_query_pilot.py`, `run_production_union_llm_pilot.py`, `…_v3_pilot.py` | query rewriting / alternatives | `llm_query_…`, `production_union_llm_…` | LLM calls |
| `run_encoder_pilots.py` | both cross-encoder pipelines | `encoder_pilot_results.json` | local model + embeddings (+ LLM calls for the multi-query pipeline) |
| `run_hyde_pilot.py`, `run_dotproduct_pilot.py`, `run_llm_retries_pilot.py` | HyDE, dot-product, retries | `hyde_…`, `dotproduct_…`, `llm_retries_pilot_results.json` | LLM calls / embeddings |
| `run_react_pilot.py` | the ReAct agent | `react_pilot_results.json` | up to 6 LLM steps / question |
| `run_react_final_answers.py` | ReAct v4, full answers | `react_final_answers.json` | LLM calls |

Running a script overwrites its results file (except the `--round` scripts, which label theirs).
