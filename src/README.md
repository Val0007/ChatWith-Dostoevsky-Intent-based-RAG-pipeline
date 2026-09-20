# src/

The working system. Modules import each other by bare name (`from retrieval import ...`), so run scripts from the repo root as `python src/<file>.py`, or put `src/` on `sys.path` as the experiment scripts do.

## Corpus preparation (run in this order, or just `ingest.py`)

| Module | Role |
|---|---|
| `chunking.py` | Split the text at chapter headings, clean PDF artifacts, recursive-chunk to 350 tokens (50 overlap, per chapter). Yields the 84 chunks + `prev_id`/`next_id` links. Local only, no API. |
| `scenes.py` | Segment the chunks into 27 scenes, write a **factual** card per scene, synthesize the global map. Prompts forbid interpretation on purpose. |
| `tag_white_nights.py` | The original single-call tagger (one ~190-line prompt → 7 fields) plus the shared schema (`ChunkTags`, allowed themes, label sets). |
| `tag_three_pass.py` | The current tagger: pass 1 local → speaker fields, pass 2 + scene cards → themes/characters, pass 3 + global map → `narrative_relation`. Every pass's output is final. |
| `ingest.py` | Orchestrates scenes → tags → embeddings → Chroma (`data/db`), and writes `data/tags.jsonl`. |

## Retrieval and chat

| Module | Role |
|---|---|
| `retrieval.py` | Shared primitives: BM25 + dense candidate generation, metadata boost/filter, LLM-judge rerank, `_diversify`, the stance-aware answer dossier and the persona prompt. Also the original 7-preset "ladder" and `search()`, the pipeline the chat currently uses. |
| `conversation.py` | Multi-turn loop: rewrite the latest message into a standalone query → `retrieval.search` → answer with history + deduped scene/chunk context. |
| `chat_cli.py` | `python src/chat_cli.py [--debug]` — interactive chat, persona voice. |

## The three retrieval methods from the retrieval study

| Module | hit@6 (10 gold questions) | Idea |
|---|---|---|
| `production_method.py` | **6/10** | regex intent classifier → intent-weighted BM25 + dense + metadata fusion → deterministic keyword-overlap rerank → ≤2 chunks per scene. The only LLM call is the final answer. |
| `production_llm_judge.py` | 2/10 | Same three candidate sources, no fusion, no formula — one LLM judge ranks the union. The controlled ablation showing the judge, not the pool, is the problem. |
| `production_react.py` | 6/10 (v4) | ReAct agent (Thought/Action/Observation) with `Search` / `Lookup` / `Finish`; first action is forced to be a production search on the literal question. |

> The chat loop currently calls `retrieval.search()`, **not** `production_method`. See the top-level README's *Known gaps*.

## Where the rest went

The retrieval variants that were tried and did not beat `production_method` (query rewriting, HyDE, cross-encoders, scene-card-first, union methods, sufficiency retries, …) live in [`experiments/retrieval/variants/`](../experiments/retrieval/variants/) — they import from here, nothing here imports them.
