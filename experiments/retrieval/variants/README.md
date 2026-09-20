# variants/

Retrieval methods that were built and measured but did **not** beat `src/production_method.py` (6/10). They import from `src/` (`retrieval.py`, `production_method.py`); nothing in `src/` imports them. Each is exercised by a `run_*_pilot.py` in the parent folder, which adds this folder to `sys.path`.

| Module | Idea | hit@6 | Verdict (see `../FINDINGS.md`) |
|---|---|---|---|
| `scene_card_production.py` | Retrieve scenes via their cards first, then chunks inside them | 5/10 | R5 — trades one vocabulary gap for another, and loses the metadata signal |
| `union_method.py` | Union direct-fusion candidates with scene-card candidates | 6/10 | R6 — ties production; useful diagnostic, no gain |
| `scene_card_llm_judge.py` | Scene-level candidates → LLM judge | 4/10 | R7 — the only method to solve q08 *and* q10, at the cost of three easy hits |
| `llm_query_production.py` | LLM rewrites the query before retrieval | 4/10 | R8–R9 — a paraphrase destroys near-exact lexical matches |
| `production_union_with_llm.py` | Original query + up to 4 LLM alternatives, unioned | 5/10 → 6/10 (v3) | R10–R12 — v3 only ties; a stricter alternatives prompt changed nothing |
| `production_with_encoder.py` | Cross-encoder rerank of production's pool | 2/10 | R13 — old verdict holds: scores surface relevance, not question intent |
| `production_with_llm_variants.py` | Cross-encoder over the multi-query pool | 3/10 | R13 |
| `production_with_hyde.py` | Hypothetical-passage embedding, unioned with the original query | 6/10 | R14 — exact tie, no new hits |
| `production_with_dot_product.py` | Rerank by pure embedding dot-product instead of the keyword bonus | 5/10 | not written up; score computed from `results/dotproduct_pilot_results.json` |
| `production_llm_retries.py` | LLM sufficiency check gates a second retrieval round | 6/10 | R16–R18 — safe (never regresses) but the gate over-triggers, retrying on every question |
| `production_with_weighted_context.py` | Confidence-weight each passage in the answer context (RAG-Token style) | — | R19 — mechanically correct, answer unchanged; the gap is upstream of the answer step |
| `rag.py` | Stage-1 naive RAG (embed → top-k → answer), the original baseline | — | superseded by `src/retrieval.py`; no retrieval study result |
