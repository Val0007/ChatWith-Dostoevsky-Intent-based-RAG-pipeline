# Chat with Dostoevsky — an intent-based RAG pipeline over *White Nights*

A retrieval-augmented chatbot that lets you argue with Dostoevsky about his novella — the motivating
question was *"Why are your main characters so pathetic?"* — and the experiment log of what it took to
make retrieval and tagging work on a text that is **interpretive**: even humans disagree about what
the author intended, so "what does this passage mean for the book?" is not a lookup problem.

Two write-ups explain the reasoning (source in
[`Val0007/Portfolio-md`](https://github.com/Val0007/Portfolio-md/tree/main/content/articles)):

1. *Either I Am Bad at Reading White Nights, or LLMs Are Bad at Understanding Interpretative Literature* — the **tagging** study
2. *Building an Intent-Based RAG Pipeline* — the **retrieval** study

This README keeps only what matters from them. Everything is backed by raw results in
[`experiments/`](experiments/).

---

## The pipeline

```
data/white_nights.txt
   │  chunking.py      5 chapters → 84 chunks (350 tokens, 50 overlap, ids like white_nights_first_night_001)
   ▼
   │  scenes.py        84 chunks → 27 scenes → 27 factual scene cards → 1 global map   (evidence only, no interpretation)
   ▼
   │  tag_three_pass.py  every chunk gets 7 metadata fields, decided in 3 separate passes
   ▼                     (see "Tagging" below)              → data/tags.jsonl
   │  ingest.py        embed + store in Chroma             → data/db/   (gitignored, rebuildable)
   ▼
   │  retrieval.py / production_method.py      find the right chunks for a question
   ▼
   │  conversation.py  rewrite follow-ups → retrieve → stance-aware dossier → persona answer
   ▼
     chat_cli.py
```

**Why tag chunks at all?** So the answering model never treats a passage as the book's final word.
A chunk where the Dreamer says "now I am happy" is tagged `narrative_relation: undermines` — the book
takes that happiness apart a few scenes later — and the answer prompt is told not to present such
ideas as Dostoevsky's own belief. The tags earn their keep at **answer time**, not as a retrieval signal.

| Tag | Meaning |
|---|---|
| `narrating_voice` / `speaking_voice` | who narrates (always the Dreamer) / who actually speaks in this passage |
| `speaker_relation` | speaker's stance: `asserts` · `doubts` · `rejects` · `explores` |
| `narrative_relation` | how the *whole book* treats the idea: `supports` · `complicates` · `undermines` · `unresolved` · `unclear` |
| `canonical_themes` | 0–3 from a fixed list (isolation, the-dreamer, self-deception, …) |
| `characters_present`, `local_motifs` | who is in the passage; free-text notable details |

---

## What the experiments found

### Tagging — 30-chunk hand-labelled gold set, scored field by field

- **More context ≠ better tagging; the *right* context per field is.** A first "context wins" result
  turned out to be a prompt confound: with the prompt held fixed, accuracy was flat (86%) across every
  arm *including no context* — context data bought **stability** (0.81 → 0.95), not accuracy.
- **Each field wants different context** (four-arm ablation, local / +scenes / +global map / both):

  | Field | Best context | Result |
  |---|---|---|
  | `narrative_relation` | global map | 47% vs 43% local; scene cards alone were *worse* (37%) |
  | `canonical_themes` | scene cards | F1 0.67 vs 0.46 local |
  | `speaker_relation` | none (local only) | every context arm 30% vs 37% local |

- **Three-pass pipeline** (`src/tag_three_pass.py`): pass 1 sees the chunk + neighbours, pass 2 adds
  scene cards, pass 3 adds the global map; each pass **owns its fields outright and nothing downstream
  revises them**. Themes reach the best score of any arm (0.67). An earlier "candidate → adjudicate"
  design was worse for a measurable reason: telling pass 2 its answer was only a draft made it over-tag
  (2.0 themes/chunk vs 1.2), and the adjudicator changed just 2 of 30 chunks.
- **`characters_present` is not solved** (F1 0.54, worst arm). The single-call prompt never carried
  rules for it — it got narrator-vs-others discipline for free from deciding `speaking_voice` in the
  same call. Stripping the isolated pass to that minimal prompt collapsed it to 0.25.
- **Tightening a guard relocates the model's default label, it doesn't teach discrimination.**
  `unresolved` → `undermines` → `complicates`; `unclear` never got above 20% recall in five rewrites.
- **Examples vs rules is field-dependent.** `speaking_voice` (a scan-and-count procedure): 90–93% with
  rules, 83% with rules + examples, **57%** with examples only. `speaker_relation` (closed-set
  judgment) does best with contrastive examples (50%).
- **Automated prompt repair mostly failed.** A critic → fixer → patch loop kept **1 of 5** patches
  (a `speaking_voice` fix, 90 → 93%); two further attempts on `speaker_relation` were reverted.
  Final locked state = "config G" = the verified wins only.
- **Noise caveat:** `speaker_relation` swings ~13 points across *identical* reruns at temperature 0
  (20 / 30 / 33%), and n = 30 — treat deltas of a few points as noise.

### Retrieval — 10 hand-written questions, metric = hit@6 (gold chunk in the top 6)

| Method | hit@6 | Notes |
|---|---|---|
| Plain hybrid (BM25 + dense) | 5/10 | baseline; every added mechanism initially made it *worse* |
| **Production** (`production_method.py`) | **6/10** | regex intent classifier → weighted BM25/dense/metadata fusion → cheap keyword-overlap rerank (no LLM) → ≤2 chunks per scene |
| Production + LLM judge (`production_llm_judge.py`) | 2/10 | the judge demotes correct evidence on interpretive questions — even when the gold chunk is confirmed in its pool |
| ReAct loop (`production_react.py`) | 4 → 5 → 4 → **6**/10 | four prompt versions; v4 only matches production because rule 1 forces its first action to be a production search on the literal question |
| Also tried (all in `experiments/retrieval/variants/`) | 2–6/10 | query rewriting 4, union + LLM alternatives 5→6, HyDE 6, cross-encoder 2–3, scene-card-first 5, sufficiency retries 6 — none beat production |

- A cheap deterministic reranker **beat** every LLM-based one. More instruction on *when to stop*
  made the ReAct agent stop **less** (2/10 calls to `Finish` → 0/10).
- **One question every method misses:** *"How does the narrator first meet Nastenka?"* — the gold scene
  happens before she is named, so the passage never contains the word being searched for. The ReAct
  agent tried six genuinely different rephrasings; all still said "Nastenka".
- **A retrieval miss produces a confident wrong answer, not a vague one** — with no Morning-chapter
  chunks retrieved, the model asserted the book gives no resolution. It does.

---

## Known gaps

- **The chat CLI does not use the best retrieval method.** `conversation.py` calls `retrieval.search()`
  (hybrid → LLM-judge rerank → diversify, no metadata boost — a close cousin of the R1 `narrative` preset, which scored 4/10; the judge-based presets were 3–4/10 vs production's 6/10).
  `production_method.py` (6/10) is implemented and tested but not wired into the chat loop.
- The deterministic three-pass rebuild and the pass-2-in-isolation tests happened after the last
  numbered finding; they live in `src/tag_three_pass.py` and `experiments/tagging/results/pass2_*.json`
  but are not yet written up in `experiments/tagging/FINDINGS.md`.
- `data/tags.jsonl` was written (config G, Aug 26) **before** the deterministic three-pass rebuild, and the rebuild
  was only re-run on the 30 gold chunks. Regenerate all 84 with `python experiments/tagging/tag_full_book_g.py` (API cost).
- `experiments/tagging/analyze_tagger.py` still reads the pre-split `speaker` field and fails on current tags.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY

# The tags and scene context are committed; only the vector DB needs building (embedding calls only):
python experiments/tagging/rebuild_db_from_tags.py

python src/chat_cli.py          # chat (add --debug to see the rewritten query + retrieved chunks)
```

Full re-derivation from the raw text (**spends API credits**: scene cards, global map, 84 × 3 tagging calls):
`python src/ingest.py`.

Reproduce article numbers **for free** from cached results:

```bash
python experiments/tagging/run_five_arm_comparison.py --score-only   # the 5-arm tagging table
python experiments/tagging/run_first30_eval.py --score-only          # per-field scoring vs gold
python experiments/retrieval/run_production_pilot.py --round 100     # retrieval hit@6 (10 embedding calls; writes a NEW results file, doesn't overwrite)
```

## Repo map

| Path | What's in it |
|---|---|
| [`src/`](src/) | the working system: chunking, scenes, tagging, ingest, retrieval, the three article methods, chat |
| [`data/`](data/) | source text, tags, scene cards + global map, and the (gitignored) vector DB |
| [`experiments/tagging/`](experiments/tagging/) | tagging study: scripts, gold set, cached results, `FINDINGS.md` (F1–F14) |
| [`experiments/retrieval/`](experiments/retrieval/) | retrieval study: pilots, gold questions, cached results, `FINDINGS.md` (R1–R24), all variants |
| [`docs/`](docs/) | original study plans and the build guide |

Each folder has its own README.
