# Findings — Retrieval (V2)

Running log for the retrieval ladder (`docs/rag_lab_plan.html`), parallel to
`experiments/tagging/FINDINGS.md` for the tagger. Same rule: every entry is what was tested, the
result, and the takeaway — numbers before conclusions.

---

## R1 — 10-question pilot: hit@6 across the 7-config ladder

**Question.** Now that retrieval is configurable (`src/retrieval.py`'s `retrieve(query,
preset=...)`, 7 presets — see prior session), does adding mechanisms (metadata boost →
LLM judge → diversify) actually improve retrieval on real questions, or does the ladder
need Phase 01 (intent extraction) before it pays off?

**Setup.** 10 hand-written gold questions spanning First through Morning
(`experiments/retrieval/gold/retrieval_gold_pilot10.json`), each with 1-2 gold evidence chunk ids
verified by grep against `white_nights.txt` (not invented — every fact was checked
against the source text before writing the expected answer). `experiments/retrieval/run_retrieval_pilot.py`
runs all 7 presets over all 10 questions and scores **hit@6**: did any gold evidence id
land in the top-6 retrieved ids. Raw output cached in `experiments/retrieval/results/retrieval_pilot_results.json`.
Reproduce: `.venv/bin/python experiments/retrieval/run_retrieval_pilot.py`

**Result.**

| preset | hit@6 (of 10) |
|---|---|
| dense | 5 |
| bm25 | 5 |
| hybrid | 5 |
| metadata | 4 |
| judge | 3 |
| narrative | 4 |
| full_v2 | 4 |

**Headline: on this pilot, every mechanism added on top of plain hybrid retrieval made
the hit rate go DOWN, not up.** That is the opposite of what the plan's ladder assumes
("each config fixes a specific failure"). Four separate mechanisms are entangled in that
number — A through D below pull them apart with the actual retrieved lists, not just the
score.

---

### A. Basic "what happened" questions can rank worse than chance under BOTH retrievers

q01 — *"How does the narrator first meet Nastenka?"* — gold evidence
(`first_night_009`/`010`, the drunken-gentleman-and-a-stick rescue scene) is **missing
from the top-20 candidate pool entirely, under dense, BM25, AND hybrid.** Checked against
the full 84-chunk ranking directly (not just pool=20): dense ranks it **50th and 61st of
84**; BM25 ranks it **74th and 43rd of 84**. For the single most concrete, memorable plot
event in the book's opening.

**Why — isolated precisely, not just "vocabulary gap."** Swapped the query's wording
while holding meaning fixed, to find the actual variable: `"How does the narrator first
meet Nastenka?"` → gold ranks 50th/61st by dense. Drop her name and describe the scene
concretely instead — `"How does the narrator first meet the girl by the canal, with the
drunk gentleman and the stick?"` → gold jumps to **rank 2nd/3rd**. The fix wasn't
rephrasing "meet"; it was removing "Nastenka." **The name is the whole problem**: this
scene happens before she's ever named in the book — both gold chunks call her only "the
girl"/"her"/"my unknown lady," and "Nastenka" appears in them zero times. The query's use
of her name pulls dense search toward whichever passages say "Nastenka" most, e.g. `"...
(I need hardly say, Nastenka) malicious people were!"`, `"sky, Nastenka. Look!"` — passages
about nothing in particular except that they address her by name repeatedly, which
outweighs any signal for "this passage describes an encounter." Not a synonym gap; a
named-entity mismatch between what the question assumes the reader can call her and what
the source text at that point actually calls her.

q02 — *"How does the narrator describe his own life in Petersburg before meeting
Nastenka?"* — same shape of miss: dense ranks the true gold (`first_night_001`) only
17th of pool=20; BM25 and hybrid drop it below 20 entirely; nearly every retrieved
passage is Second Night material (which also discusses loneliness, just not the specific
passage asked about) — retrieved on topic-adjacency, not on being *the* passage.

**Takeaway.** This is exactly the gap Phase 01 (query intent + entity extraction) is
supposed to close: a question that names an *event* ("first meet") or a *subject*
("his own life... before meeting Nastenka") needs to be routed to the right *location*
in the book (opening scene, specific character), which literal similarity — dense or
keyword — doesn't reliably do on its own. This is independent confirmation of the plan's
own stated rationale for Phase 01, now with a concrete failing example in hand.

---

### B. RRF fusion can strand a candidate that only ONE retriever finds

q08 — the "now I am happy" doomed-joy question (`fourth_night_012` — the exact chunk
F1/F5 in the tagger findings used as the doomed-joy example). Dense alone barely keeps it
at **rank 20/20** — the very edge of the pool. BM25 doesn't find it in the top 20 at all.
Because RRF fusion needs it to rank reasonably in *both* lists to score well, and it's
absent from one of them, **the hybrid pool drops it entirely** — worse than dense alone
would have done by keeping it (barely) at the bottom.

**Takeaway.** Hybrid/RRF is usually the safer default (recall@6 ties dense and BM25
here, 5/10 each), but it is not strictly better than either single retriever per-question
— a candidate strong in one signal and absent in the other can fall through the fusion
gap that neither pure-dense nor pure-BM25 has.

---

### C. The LLM judge demotes correct evidence on interpretive/comparative questions —
### and the demotion is partly *defensible*, which is the more interesting finding

q05 and q09 both need the Second Night "the dreamer... is superior to all desire"
monologue (`second_night_013`) as evidence. It's comfortably present in the hybrid
pool (**rank 2 of 20** for both questions) — but after the LLM judge rescoring, it falls
to **rank 7** (q05) and **rank 12** (q09), pushed below the top-6 cutoff both times.

Pulling the judge's raw 1-5 scores for q05 explains why: it gave `second_night_013`
only **3/5**, while scoring `second_night_019` and `second_night_020` **5/5** — and
those two turn out to be the narrator's own *later, explicit self-disgust* at his
fantasy life ("after my fantastic nights I have moments of returning sobriety, which
are awful"; "...so luxuriously deceived him"). That is arguably **better** evidence
that the book undercuts the earlier rapture than the rapture passage itself is — the
rapture states the claim, the later passage IS the undercutting.

**This means q05's single-chunk gold label is too narrow, not that the judge is
malfunctioning.** An interpretive "does the book undercut X" question is well-supported
by *multiple, non-adjacent* chunks (the claim + its later reversal), and hit@6-against-
one-gold-id can't see that a judge swapping one valid piece of evidence for another,
arguably stronger one is a *correct* rerank, not a miss. This is precisely Confound 2
from `docs/rag_lab_plan.html` ("gold evidence is subjective") showing up empirically, and
it is the same mechanism as the tagger study's F1 finding — narrative judgment lives
outside any single chunk — now surfacing in the retrieval judge instead of the tagger.

**Consequence for Phase 02.** Single-chunk `gold_evidence` lists are adequate for factual
questions (q01-q04, q06, q10) but need to become *sets of acceptable chunks* (or a
looser "supports the claim" judge-graded metric) for interpretive/comparative/evaluation
questions before hit@k numbers on those types can be trusted.

---

### D. The metadata boost (this session's crude placeholder) is not yet earning its
### keep — expected, since it isn't real intent routing

`metadata` trails plain `hybrid` (4/10 vs 5/10). Concretely, on q06 hybrid alone finds
the gold evidence (`second_night_034`) at rank 1; the metadata boost — pure query-token
vs. tag-token overlap, no actual query understanding — reorders it out of the top 6
entirely, promoting chunks that happen to share theme/character words with the query
without being the right passage. On q07 the boost helps (ties every preset at hit@1)
because that question's own wording ("compare... feelings for") happens to overlap with
tag vocabulary. That inconsistency is expected: `_metadata_boost()` was built explicitly
as *"a cheap substitute for real intent-routing"* (see its docstring in
`src/retrieval.py`), not a finished mechanism — this pilot is the first real evidence
that the substitute needs Phase 01's actual intent extractor to become net-positive,
rather than being tuned further as a heuristic.

**`full_v2` == `narrative` on every question in this pilot** — expected, not a bug:
`expand` only changes context assembled for answering (not retrieval order) and
`intent` is accepted by `retrieve()`'s config but not yet wired to anything (Phase 01).
Confirms Phase 00's refactor is behaving exactly as documented.

---

## What this changes for the plan

1. **Phase 01 (query intent) is the right next investment, not further tuning of Phase
   00's mechanisms.** Both A and D point the same direction: literal similarity (dense,
   BM25, hybrid, and the crude metadata boost) plateaus around 4-5/10 on this pilot
   specifically because none of them understand *what the question is asking for*
   (an event, a specific passage, a comparison) — they only measure surface overlap.
2. **The judge and diversify mechanisms need to be evaluated on answer quality, not
   hit@6-against-single-gold-chunk**, for interpretive/comparative/evaluation questions
   — C shows the judge can be making a legitimate call that a narrow gold label scores
   as a miss. Phase 02's harness should carry both a strict recall@k AND a looser
   judge-graded "is the retrieved set sufficient to answer" metric per the plan's own
   `evidence precision` / `faithfulness` metrics — recall@k alone is not enough,
   confirmed empirically rather than just assumed from the doc's risk section.
3. **n=10 is a smoke test, not a result.** Every number above should be read as "worth
   investigating," not "the ladder underperforms" — the real Phase 02 pilot (30
   questions, human-adjudicated, closed-book row 0) is what turns these into a defended
   claim. This entry exists to catch exactly the kind of surprise (B, C) that motivates
   building that harness properly rather than skipping straight to a 100-200 question
   run.

**Method note for reruns.** `rerank()`'s LLM judge runs at `temperature=0` but is a
single call per question with no repeat runs — the tagger study (`FINDINGS.md` F10)
found real run-to-run swing on gpt-4o-mini even at temp 0 for nuanced classification
fields. The judge-demotion numbers in C were confirmed by pulling raw scores once, not
across multiple runs; treat the exact ranks (7, 12) as indicative of a real, large
effect (pool rank 2 → outside top-6 is a big swing either way), not as precise to the
rank.

---

## R2 — an 8th method: an LLM router that picks a preset per-question

**Question.** R1 showed a fixed preset either does or doesn't fit a given question. What
if, instead of a human picking one preset for all 10 questions, an LLM looks at each
*specific* question plus the schema of what's stored per chunk, and picks which of the 7
presets should handle it? This is a real (if simple) version of what Phase 01's "intent
extraction" was meant to eventually do — one step up from R1-D's crude token-overlap
metadata boost.

**Setup.** `retrieve.choose_method(query)` — one LLM call, given the question, a
description of all 7 presets ("dense: best when wording differs... bm25: best when the
question reuses specific words... judge: best for 'why' questions needing real reading
comprehension...") and the chunk schema (text, chapter/section, scene_id,
narrating_voice, speaking_voice, speaker_relation, narrative_relation, canonical_themes,
local_motifs, characters_present) — returns `{"method": ..., "reason": ...}`.
`auto_retrieve()` then just calls `retrieve(query, preset=chosen_method)`.
`experiments/retrieval/run_router_pilot.py` runs this over the same 10 gold questions. Reproduce:
`.venv/bin/python experiments/retrieval/run_router_pilot.py`. Raw output:
`experiments/retrieval/results/router_pilot_results.json`.

**Result — router hit@6: 4/10.** Middle of the pack, not a new best:

| method | hit@6 |
|---|---|
| dense / bm25 / hybrid | 5/10 each |
| metadata / narrative / full_v2 | 4/10 each |
| **router** | **4/10** |
| judge | 3/10 |

| question | router picked | hit? | router's stated reason |
|---|---|---|---|
| q01 factual | bm25 | miss | "likely includes specific wording... phrases from the text" |
| q02 character | full_v2 | miss | "best answered by examining passages with surrounding context" |
| q03 motivation | judge | hit@5 (coin flip — see below) | "requires understanding of character motivations... deeper reading comprehension" |
| q04 factual | bm25 | **hit@1** | "likely includes specific names and phrases... verbatim" |
| q05 interpretive | judge | miss | "requires evaluative judgment... nuanced comprehension" |
| q06 evaluation | judge | **hit@2** | "requires understanding of motivations and nuances of argument" |
| q07 comparative | narrative | **hit@1** | "requires evidence from multiple scenes... diversifies across scenes" |
| q08 evaluation | judge | miss | "requires evaluative reading... complexity of emotions" |
| q09 comparative | narrative | miss | "requires evidence from multiple scenes... development over time" |
| q10 development | full_v2 | miss | "involves multiple scenes and contextual nuances" |

**The router's reasoning is genuinely good — it correctly reads what TYPE each question
is.** It called q01/q04 factual-with-specific-wording (bm25), q05/q06/q08 as needing real
reading comprehension over surface overlap (judge), and q07/q09 as needing cross-scene
evidence (narrative). Every one of those classifications is defensible and matches the
`intent` label I'd hand-assigned in the gold set independently. **This is the clearest
sign yet that the "figure out what kind of question this is" step (Phase 01's whole
premise) is buildable and works** — the routing logic itself is not the weak link.

**But correct routing doesn't fix a method that's broken for reasons unrelated to which
method it is.** This is the real finding:
- **q08**: router correctly says "this needs the judge." But R1-B already showed this
  question's gold chunk never survives past hybrid candidate generation (RRF drops it
  because it's absent from BM25's list) — the judge never even gets to see it. Correct
  routing to `judge` cannot rescue a chunk that candidate generation already threw away
  *before* judge runs.
- **q05**: router again correctly says "judge." But R1-C already showed the judge
  actively *demotes* this exact gold chunk (rank 2 → rank 7) in favor of arguably-also-
  valid alternative evidence. Routing to the "right" method here walks straight into a
  known weakness of that method.
- **q01**: router's reasoning ("likely includes specific wording") is a completely
  sensible guess — and wrong. R1-A already showed this question's wording shares almost
  no vocabulary with the actual passage (dense rank 50/84, BM25 rank 74/84). The router
  has no way to know that without actually trying retrieval and checking — it's
  reasoning from the *question's surface phrasing*, not from *how the book is actually
  written*, so a plausible-sounding guess can still be empirically false.

**q03's result is noise, not a router win — worth flagging explicitly.** The router
picked `judge`, and it scored hit@5 in the router run, but the fixed `judge` preset
scored a miss on the exact same question in R1. Re-running `retrieve(..., preset="judge")`
on q03 three times back-to-back: **hit, miss, hit** — the gold chunk sits right at the
rank 6/7 boundary and the judge's temp=0 rerank still isn't fully deterministic (same
nondeterminism `FINDINGS.md` F10 documented for the tagger). Don't read q03 as "the
router found something the fixed preset couldn't" — it's the same coin, flipped twice.

**Takeaway.** An LLM router is a real, working way to classify *what kind of question*
this is — that part of Phase 01 is validated. But routing only helps if the destination
method is itself reliable, and R1 already showed judge/hybrid have their own independent
failure modes (dropped-before-judge-runs, demoted-by-judge, vocabulary-gap) that no
amount of correct classification fixes. The next real lever isn't a smarter router — it's
fixing what R1 already found broken in candidate generation (A, B) and rethinking how
"correct" is scored for interpretive questions (C), which a router built on top of the
same broken layer inherits by construction.

---

## R3 — a 9th method: a structured diagnostician that composes its own plan

**Question.** R2's router picks one label out of 7 fixed bundles. What if instead the
LLM analyzes the question along several independent axes — expected lexical overlap,
expected semantic overlap, whether metadata would help, whether it needs multi-hop
evidence, whether it needs a careful reread — and composes its OWN ordered plan out of
primitive operations, not limited to 7 pre-named combinations? This mirrors a schema a
retrieval-savvy person suggested: `{intent, evidence_requirement,
lexical_overlap_expected, semantic_similarity_expected, metadata_usefulness,
needs_multi_hop, needs_reranking, recommended_plan}`, where `recommended_plan` is an
ordered list like `["metadata_filter", "dense", "rerank"]`.

**Setup.** `retrieve.diagnose_query(query)` — one LLM call, returns exactly that JSON
shape. `retrieve.run_plan(query, plan)` actually executes the ordered list: each
retrieval step (`bm25`/`dense`/`hybrid`) ranks the *whole* corpus, then restricts to
whatever candidates survived the steps before it — so `["metadata_filter", "dense",
"rerank"]` genuinely means "narrow by tags, THEN rank what's left by meaning, THEN
LLM-rereread it," not three independent, unrelated retrievals. Building this also caught
and fixed the R1-D bug: `_metadata_boost`/the new `_metadata_filter` now strip stopwords
before counting tag-word overlap (`"the"`/`"and"`/`"for"` no longer count as matches).
`experiments/retrieval/run_plan_pilot.py` runs this over the same 10 questions. Reproduce:
`.venv/bin/python experiments/retrieval/run_plan_pilot.py`. Raw output:
`experiments/retrieval/results/plan_pilot_results.json`.

**Result — plan hit@6: 4/10.** Same tier as the R2 router, still behind plain
dense/BM25/hybrid (5/10 each):

| method | hit@6 |
|---|---|
| dense / bm25 / hybrid | 5/10 |
| metadata / narrative / full_v2 / router / **plan** | 4/10 |
| judge | 3/10 |

**Headline finding: the diagnostic fields don't diagnose anything.** Pulling every
question's raw values:

| field | value across ALL 10 questions |
|---|---|
| `lexical_overlap_expected` | **"medium" — every single time** |
| `semantic_similarity_expected` | **"high" — every single time** |
| `metadata_usefulness` | **"high" — every single time** |
| `needs_multi_hop` | **false — every single time** |
| `needs_reranking` | true on 9/10 (false only on q01) |

Four of five structured fields are **completely constant** regardless of the actual
question — q01 (empirically proven in R1-A to have almost NO lexical or semantic overlap
with its gold passage: dense rank 50/84, BM25 rank 74/84) got the exact same
`lexical_overlap_expected: "medium"` / `semantic_similarity_expected: "high"` as q04
(empirically the easiest question in the set — BM25 finds it at rank 1 immediately,
maximum real lexical overlap). **The model is producing a plausible-looking, fully-
populated JSON object without actually calibrating most of its fields to the specific
question.** Only two things vary at all: the free-text `intent` label (genuinely good —
`find_specific_narrative_event`, `character_motivation`, `thematic_interpretation`,
`comparison`, `story_development`, matching the hand-assigned gold `intent` field well)
and the exact composition/ordering of `recommended_plan`. This is the same shape of
failure as the tagger study's F12 (`FINDINGS.md`): asking a model for more structured
self-assessment doesn't make its confidence real — absent something that forces genuine
per-input discrimination, it fills the form the same way every time.

**Case study: the near-universal "add rerank" recommendation actively broke the easiest
question in the set.** q04 is the cleanest possible factual lookup — "What was
Nastenka's living situation with her grandmother, and what promise did the lodger make
before leaving for Moscow?" Plain BM25 alone finds it at **rank 1** (R1). Traced the
composed plan (`["metadata_filter", "bm25", "rerank"]`) step by step:

1. `metadata_filter` narrows 84 chunks to 17 — both gold chunks survive (correctly; a
   real filter, not the old stopword-bug boost).
2. `bm25`, restricted to those 17, puts a gold chunk at **rank 1** — exactly as strong as
   unrestricted BM25 was. The filter didn't hurt.
3. `rerank` — the LLM judge — **demotes it to rank 7**, outside the top-6 cutoff. Miss.

Steps 1–2 of the model's own plan would have been a clean hit on their own. The plan's
own third step undid it — the exact same judge-demotion mechanism documented in R1-C and
inherited by R2, now shown breaking a question that had ALREADY been solved by the time
the judge ran, on a plan the diagnostician itself composed. And this isn't a one-off
misjudgment: `needs_reranking` was `true` for 9 of the 10 questions, i.e. the
diagnostician defaults to "add reranking" almost regardless of the question, so this
failure mode fires often, not rarely.

**Takeaway.** Composing a plan from a richer schema didn't produce better decisions than
R2's single-label router (4/10 either way) — because the extra fields that were supposed
to justify each plan turned out to be decorative rather than load-bearing. The one part
of this that IS working (the `intent` classification, and building a real
`metadata_filter` that fixed a genuine known bug) is worth keeping; the low/medium/high
self-assessment fields as currently prompted are not adding real signal and would need
either concrete calibration examples per field (the same fix F11 found worked for the
tagger's `speaker_relation`) or grounding in something measurable — real retrieval
statistics for a labeled sample, not the model's own guess — before they're worth trusting.

**Method note.** `_metadata_filter`'s narrowing (84 → 17 for q04) and BM25's within-
filter rank-1 result were both verified directly (not inferred from the final score) by
running `run_plan`'s internals step-by-step outside the harness — see the case study
above. The judge's exact demotion rank (7) carries the same single-run caveat as R1-C
and R2: gpt-4o-mini's rerank is not perfectly deterministic at temp 0, so treat "pushed
outside top-6" as the reliable part of this finding, not the precise rank number.

---

## R4 — `src/production_method.py`: weighted intent-conditioned fusion + a non-LLM
## reranker. New best: 6/10, tuned across 3 logged rounds

**Question.** R1–R3 all routed to one of 7 fixed *bundles* (or composed a plan out of
discrete steps). What if intent instead controls a continuous **blend** — BM25, dense,
and metadata each contribute a weighted fraction of one fused score, with the weights
set by intent — and the final rerank step is a cheap deterministic formula instead of
another LLM call? This is a genuinely different mechanism from R1–R3, not another preset,
so it lives in its own file: `src/production_method.py`.

**Pipeline** (matches the diagram sketched in chat):
```
QUERY -> classify_intent() [regex, no API call]
      -> fuse() [BM25 + dense + metadata, each min-max normalized over all 84 chunks,
                 blended by WEIGHTS[intent]]
      -> cheap_rerank() [fused score + a literal-keyword-precision bonus — no LLM call]
      -> _diversify() [reused from retrieval.py]
      -> top k -> retrieval.answer() [LLM, unchanged]
```

**The cheap intent classifier.** Pure regex over the question text, checked in priority
order (most specific/confident first): `INTERPRETIVE` ("does the book/narrative
endorse...", "presented as...", "undermin-/endors-") → `COMPARATIVE_MULTI_HOP`
("compare", "differ-", "whereas", "versus") → `MOTIVATION` ("why does/did/would...",
added in round 1, see below) → `EVENT` (a ~30-verb list: meet, leave, marry, promise,
convince, confess, return, decide, resolve...) → `CHARACTER_RELATIONSHIP` (a named
character + "feels about"/"relationship"/"thinks of", or 2+ named characters) →
`TEMPORAL` ("before", "after", "first", "again"...) → `LEXICAL_FACT` (starts with
"who"/"what is"/"which") → `GENERAL` (fallback). Costs nothing but a regex scan — no
API call, unlike R2/R3's LLM router/diagnostician.

**Why the reranker is NOT an LLM.** R1-C, R2, and R3 all independently found the same
failure: an LLM judge, asked to rescore candidates, demotes correct evidence for
interpretive/evaluative questions specifically — the exact question types this pipeline
is supposed to help most. `cheap_rerank()` instead adds a small, deterministic bonus for
literal keyword overlap on top of the fused score — a formula, not a second opinion, so
it can't talk itself out of a good answer the way the judge repeatedly did.

### Tuning log (all rounds: `experiments/retrieval/run_production_pilot.py --round N`, raw output
### in `experiments/retrieval/results/production_pilot_round*.json`)

**Round 0 — initial weights** (the 3 the ladder's designer specified for EVENT/LEXICAL_FACT/
CHARACTER_RELATIONSHIP, plus a reasoned starting guess for TEMPORAL/COMPARATIVE_MULTI_HOP/
INTERPRETIVE/GENERAL): **5/10** — misses on q01, q02, q03, q08, q10. Already ties the best
prior method (dense/bm25/hybrid), and notably **q05 and q09 — the two questions R1-C/R2/R3
all showed the LLM judge actively demoting — are hit@1 here**, first real evidence the
non-LLM reranker avoids that specific, three-times-repeated failure.

**Diagnosing round 0's 5 misses** — pulled the raw per-signal rank (of 84) for each gold
chunk, not just the blended score, to tell "weights are wrong" apart from "no signal
exists to weight":

| miss | bm25 rank | dense rank | metadata rank | diagnosis |
|---|---|---|---|---|
| q01 | 74 / 43 | 50 / 61 | 81 / 2 | weak-to-mixed on ALL signals for one chunk; nothing to reweight toward |
| q02 | 30 | **17** | 74 | dense is real but TEMPORAL's weights (0.4) under-used it |
| q03 | **1** | 53 | 27 | bm25 is a near-perfect match — INTERPRETIVE's 0.1 bm25 weight was drowning an easy answer |
| q08 | 62 | **20** | 68 | dense is the only real signal; INTERPRETIVE's weight (0.5) was leaving value on the table |
| q10 | 76 / 70 | 76 / 45 | 70 / 72 | weak on ALL signals for both gold chunks; same shape as q01 |

**Round 1 — two targeted fixes from that diagnosis:**
1. **Split `MOTIVATION` off from `INTERPRETIVE`.** q03 ("Why does Nastenka make the
   narrator promise...") is a plain character-motivation question that happens to share
   exact vocabulary with its answer passage — nothing like q05/q08's genuine whole-book
   judgment questions. Gave it its own bucket (`bm25 0.5 / dense 0.35 / metadata 0.15`,
   EVENT-like) and re-anchored `INTERPRETIVE`'s patterns to only the "does the book
   endorse/undercut" framing, not "why does X" generally.
2. **Reweighted `TEMPORAL`** (`dense 0.4→0.55`, `metadata 0.3→0.2` — the metadata signal
   was contributing noise, rank 74/84, not signal) and **`INTERPRETIVE`**
   (`dense 0.5→0.6`, `metadata 0.4→0.3`) toward their one real signal (dense) per the
   table above.

**Result: 6/10** — q03 now hit@5, the one round-1 flip. q01, q02, q08, q10 remain misses.
q02's TEMPORAL reweight moved its retrieved set (compare round 0 vs round 1's top-6 —
`second_night_013` and `second_night_019` swap in/out) but its gold chunk
(`first_night_001`) never entered the top 6 in either round — a real but insufficient
shift, not the fix it was aimed at. q01 and q10 were untouched by any round-1 change and
remain misses exactly as predicted from the "weak on all signals" diagnosis — no weight
vector can promote a chunk that isn't well-represented by any of the three signals being
blended.

**Round 2 — testing whether q08 is a ceiling or just needed more push.** Pushed
`INTERPRETIVE` to near-pure dense (`bm25 0.05 / dense 0.7 / metadata 0.25`) specifically
to see if q08 (dense rank 20/84, its only real signal) would cross into the top 6 with
maximum weight on its one working signal. **Still a miss, still 6/10, identical
hit/miss pattern to round 1.** This confirms q08 is a genuine ceiling for this pipeline —
even devoting nearly the whole score to its best signal isn't enough, because the
competing chunks at rank 1-5 also score well on dense (they're topically adjacent, just
not the right passage) and nothing here can out-rank them further. Reverted to round 1's
more moderate weights (no reason to keep an extreme setting that didn't help) and locked
that as final — `WEIGHTS` in `src/production_method.py` is the round-1 state.

### Final result

| method | hit@6 |
|---|---|
| dense / bm25 / hybrid | 5/10 |
| metadata / narrative / full_v2 / router (R2) / plan (R3) | 4/10 |
| judge | 3/10 |
| **production (R4)** | **6/10 — new best** |

**Remaining misses (q01, q02, q08, q10) are honest, diagnosed ceilings, not untried
tuning.** All four were checked against their raw per-signal ranks (not just the blended
score) and confirmed weak or absent in BM25, dense, AND metadata simultaneously (q01,
q10), moved by round 1's reweight but still short of top-6 (q02), or maxed-out on their
one working signal without enough separation from competitors (q08 — round 2 confirmed
this by pushing INTERPRETIVE to near-pure dense with no change in outcome). This matches
R1-A's finding exactly — q01 was already shown to sit at dense rank 50/84 and BM25 rank
74/84 in R1 — confirming that no amount of *reweighting existing signals* fixes a genuine
vocabulary/paraphrase gap. Fixing these would need a different mechanism entirely (query
expansion/rewriting, or a fourth signal), not further tuning of this one.

**Addendum — what these 4 misses actually do to the final ANSWER, not just the score.**
Ran `production_answer()` on all four (not just `production_retrieve()`) to check
whether a retrieval miss degrades gracefully or corrupts the answer outright:

- **q01**: the model answered with a real quote from a *different* scene (Nastenka and
  the narrator anxiously awaiting her lover, Third Night) and presented it as "how they
  first met" — a wrong scene stated as settled fact, fluently, with no hedging.
- **q02**: thematically plausible (isolation, self-deception) but built entirely from
  Second Night material; the concrete facts asked for (eight years, the houses, the old
  man) never appear because the actual opening passage was never retrieved.
- **q08**: substituted a *different* passage (Third Night: "how unbearable a happy
  person is sometimes") and reasoned its way to the SAME correct conclusion the gold
  answer reaches — right theme, wrong citation. The one case where the miss didn't
  visibly damage the answer's correctness, only its grounding.
- **q10** — the serious one: with zero Morning-chapter chunks retrieved, the model
  answered *"The story does not provide a clear resolution for the narrator and
  Nastenka's relationship."* **That's false — the book resolves.** Not a vague or
  hedged answer; a confident, specific denial that an ending exists, produced entirely
  because the ending was never in the context window.

**This directly answers the plan's own Q12** ("do local-vs-context tag differences
change the ANSWER, or only the metadata?") for retrieval, not just tagging: yes, and not
always gracefully — q10 shows a retrieval miss can flip an answer to something actively
wrong rather than merely vaguer, and none of the four wrong answers read as uncertain.
Every prior finding in this file (R1–R4) only measured hit@6; this is the first check of
whether that number's failures are cosmetic or real, and for q10 at least, they're real.

---

## R5 — `experiments/retrieval/variants/scene_card_production.py`: does routing through scene cards first fix the
## paraphrase-gap misses? No — it trades one gap for another, and drops metadata's value

**Question.** R4 (and R1-A before it) found a real, structural failure: q01/q02/q10
share almost no vocabulary with their answer PASSAGE at any signal — q01's gold chunk
sits at dense rank 50/84, BM25 rank 74/84. But scene cards are LLM-written prose
*summaries* of each scene, in different words than the source text. Concretely,
`scene_03`'s card (the exact scene q01 asks about) says *"the narrator intervenes with a
stick, scaring off the gentleman, and offers his arm to the girl"* — much closer to the
question's own phrasing than the raw passage. Hypothesis: search scene cards first to
find the right scene (coarse), then search only within that scene for the actual passage
(fine) — the vocabulary gap should be smaller at the card level.

**Pipeline**: `query -> top_scenes()` (BM25 + dense over the 27 scene-card summaries,
50/50) `-> chunks within those scenes, scored 3 ways` (local BM25 — an index built ONLY
from that chunk set, different IDF than the whole-book index; global BM25 — the
whole-book index restricted to those chunks; dense) `-> fusion -> cheap_rerank()`
(reused from R4, still no LLM call) `-> diversify -> top-k`. No metadata signal — not
part of the pipeline as specified. `experiments/retrieval/run_scene_card_pilot.py` runs it over
the same 10 questions.

**Result: 5/10 (after tuning `n_scenes` 6 → 9) — worse than production_method's 6/10,
and it did NOT fix any of the three questions it was built to fix.**

| n_scenes | hit@6 | notes |
|---|---|---|
| 6 (initial) | 4/10 | regressed q07 and q09, which production_method got right |
| 9 | 5/10 | recovered q07 (its scene had ranked 8th — just outside the cutoff) |
| 12 | 5/10 | no further gain — same diminishing-returns shape as R4's round 2 |

**Finding 1 — the hypothesis was wrong: scene cards have their OWN version of the
paraphrase gap, not a fix for the passage-level one.** Checked `scene_03`'s card rank
directly for q01's query: **22nd of 27 scenes.** Why: the query asks about "Nastenka" by
name, but `scene_03`'s card — describing the FIRST time they meet, before her name is
ever spoken — never says "Nastenka," only "the girl" (`'nastenka' in
scene_cards['scene_03']['summary'].lower()` → `False`). Every scene card that says
"Nastenka" prominently outranked it, pulling the top scenes toward generic
Nastenka-and-the-narrator content (`scene_14`, `scene_11`, `scene_08`) instead of the
specific first-meeting scene. This is the exact same "gold answer needs a name the
passage/card hasn't earned yet" issue `FINDINGS.md` (the tagger's) already flagged for
`characters_present` — now shown to break scene-level retrieval too, for a different
reason each time (the passage lacks the vocabulary; the card lacks the name).

**Finding 2 — even when scene selection DOES find the right scene, within-scene fusion
can still lose the chunk.** q10's gold scene (`scene_27`, the Morning chapter) ranked
**2nd of 27** scene cards — the coarse step worked perfectly here. But the chunk-level
fusion within the ~20-chunk candidate pool still didn't promote either gold chunk into
the final top 6. Coarse-to-fine only helps if BOTH stages work; getting the scene right
is necessary but not sufficient.

**Finding 3 — dropping metadata cost more than restructuring gained.** q07 and q09 both
worked cleanly under production_method (hit@1 each) and both regressed here. Traced q09:
its correct chunk (`fourth_night_012`) IS in the candidate pool at `n_scenes=9` (its
scene ranked 9th, just inside the cutoff) but lands at **rank 23 of 29** candidates after
local-BM25 + global-BM25 + dense fusion — badly outscored by chunks from other included
scenes with more surface-similar "happiness with Nastenka" content. Under
production_method, this exact question was classified `COMPARATIVE_MULTI_HOP` and routed
through 40% metadata weight — the `canonical_themes`/`narrative_relation` tag signal that
this pipeline simply doesn't have access to. Scene cards weren't a bad idea; they're just
not a substitute for the signal that was actually doing the work.

**Takeaway.** The specific mechanism this session's data motivated (query names an entity
before the text/card is "allowed" to) turned out to defeat scene-card matching almost as
readily as it defeats passage matching, so the hoped-for fix didn't materialize on the
questions it targeted. And restructuring into scene-first search meant leaving out
metadata, which cost two working questions to gain zero broken ones back. **Locked
production_method (R4, 6/10) remains the best method found this session.** If scene cards
are revisited, the more promising next step per this data isn't discarding metadata —
it's adding metadata's tag signal INTO the within-scene fusion stage here, since Finding 2
shows the coarse step alone isn't the bottleneck.

---

## R6 — `experiments/retrieval/variants/union_method.py`: UNION direct fusion with scene-card candidates instead of
## replacing one with the other. Ties the best score (6/10); genuinely new diagnostic insight

**Question.** R4 (direct BM25+dense+metadata fusion) scored 6/10. R5 (scene-cards
REPLACING direct search) scored 5/10 and lost ground on 2 questions R4 had solved. What
if scene-card search only ever ADDS candidates on top of R4's, never replaces them — a
scene-card miss then costs nothing, since the direct-fusion candidates are untouched?
Also upgraded the scene scorer to use all 5 scene-card fields (`summary` 0.40,
`developments` 0.25, `characters` 0.15, `consequences` 0.10,
`important_beliefs_or_feelings` 0.10 — the weights specified for this run), not just
`summary` as in R5.

**Pipeline**: `query -> production_method.fuse()` (existing candidates, top 20) `+`
`scene_scores_weighted()` (5-field weighted scene search -> chunks in the top-n scenes,
additional candidates) `-> union (dedupe) -> cheap_rerank -> diversify -> top 6`.

**Round 0 (`n_scenes=6`, straight union): 6/10 — ties R4, changes nothing.** Diagnosed
why by checking directly whether each of R4's 4 misses (q01/q02/q08/q10) even entered the
union: **none of them did.** The weighted 5-field scene scorer still didn't surface their
correct scene inside the top 6 (q01's scene ranked 21st of 27, barely moved from R5's
summary-only 22nd — the "girl not yet named Nastenka" problem from R5 Finding 1 persists
across all 5 fields, since none of them use her name that early in the book either).

**Round 1 (`n_scenes=10`): still 6/10, but now genuinely diagnostic.** q08's scene
(rank 9) and q10's scene (rank 10) both now enter the union — checked directly: their
gold chunks ARE present in the candidate pool this time. But they still don't survive
`cheap_rerank` into the top 6, because **a bug**: `cheap_rerank` was scoring every
candidate — including ones that only got in via the scene path — using `fused[cid]`, the
SAME direct BM25/dense/metadata score that had already ranked them near the bottom of the
corpus. A chunk earning its place through a strong scene match got zero credit for that
in the final ranking. Real bug, not a tuning gap — fixed by blending in the chunk's own
scene's relevance score before reranking (`blended[cid] = fused[cid] + scene_bonus_weight
* scene_scores[chunk's scene]`, `scene_card_production.py` refactored to expose
`scene_scores_weighted()` — the full 27-scene score dict, not just the top-n list — so
this blend is possible at all).

**Round 2 (fix applied, `n_scenes=10`, `scene_bonus_weight=0.3`): still 6/10 — no
regression, no gain.** Traced q08 and q10 directly: the blend DOES move their scores up
(q08's gold chunk: fused 0.648 → blended 0.813; q10's better chunk: fused 0.541 →
blended 0.738) — but not enough. They land at rank 19 and rank 30 of ~36-38 candidates,
still nowhere near top 6, because the chunks that were already winning ALSO belong to
well-matching scenes and get comparably boosted — the blend doesn't change anyone's
*relative* position much when everyone's scene is reasonably strong.

**Round 3 (a real stress test of the bonus, not just tuning): `scene_bonus_weight` swept
0.3 → 0.6 → 1.0 → 1.5.** 0.3 through 1.0 all hold at 6/10 (no change either way) — **1.5
actively regresses to 4/10**, breaking 2 previously-correct questions by letting scene
relevance overwhelm the direct passage-level signal that was correctly identifying them.
Also swept `n_scenes` 10 → 15 → 20 (nearly the whole 27-scene book) at the safe bonus
weight: still 6/10, flat. **Locked final config: `n_scenes=10`, `scene_bonus_weight=0.3`**
— the widest and most-generously-boosted setting that doesn't cost anything, even though
it doesn't gain anything either.

**This is a genuine ceiling, now confirmed from three independent angles.** R1-A found
q01's gold chunk weak on both raw BM25 and dense. R4 confirmed it weak on metadata too.
R5 found its SCENE card also doesn't say her name yet. R6 now shows that even a generous
union of both retrieval paths, with candidate coverage pushed to nearly the entire book
(20 of 27 scenes) and the scene signal weighted enough to start breaking other questions,
still can't promote it into the top 6. The bottleneck by this point is almost certainly
the final ranking/reranking step's ability to discriminate among many plausible-looking
candidates, not candidate recall — recall was never actually the problem for q08/q10
once `n_scenes=10` was tried; a scoring/discrimination problem was.

**Takeaway.** Union-not-replace was the right instinct — it's the only scene-card variant
this session that didn't lose ground (R5 dropped q07/q09; R6 keeps everything R4 had).
But it didn't add ground either. The real, reusable finding is methodological: **adding a
candidate to a pool is necessary but not sufficient — the final scoring function has to
be told WHY a candidate is there, or it re-applies the same judgment that already
rejected it.** That bug (round 1) was worth finding on its own, independent of whether
fixing it moved the final number. `production_method` (R4) and `union_method` (R6) are
now tied at 6/10 as this session's best; `union_method` is the more defensible one to
build on since it strictly dominates R4 (same misses, zero new ones, plus real evidence
about where the remaining ceiling actually lives).

---

## R7 — two ablations: strip fusion + cheap-rerank from both methods, let an LLM judge
## pick directly. One crashes (2/10); the other cracks the two hardest questions all
## session (4/10, but the first to solve q08 AND q10)

**Question.** R4/R6 both work by (1) generating candidates from multiple signals, (2)
combining them into ONE score via weighted fusion, (3) reranking with a cheap
deterministic formula. R1-C/R2/R3 found an LLM judge, used as step 3 on a FUSED pool,
reliably demotes correct evidence on interpretive questions. Does that hold if fusion is
removed entirely too — i.e. the judge sees a wider, unweighted UNION of each signal's own
top candidates, never blended into one ranking first? Built two ablations to test it on
both candidate-generation strategies from this session.

**`src/production_llm_judge.py`**: BM25 top-15 ∪ dense top-15 ∪ metadata top-15 (each
independently, no fusion) → straight to `retrieval.rerank()` (the LLM judge) → diversify
→ top 6. **`experiments/retrieval/variants/scene_card_llm_judge.py`**: BM25-over-scene-cards top-8 ∪
dense-over-scene-cards top-8 scenes (no fusion) → every chunk in those scenes → same LLM
judge → diversify → top 6. Both remove fusion AND the cheap reranker; the LLM judge is
the only ranking mechanism from candidates to top-6 in either.
`experiments/retrieval/run_judge_ablation_pilot.py` runs both over the same 10 questions.

**Result:**

| method | hit@6 |
|---|---|
| production_method (R4, fusion + cheap rerank) | 6/10 |
| union_method (R6, fusion + cheap rerank) | 6/10 |
| **production_llm_judge** (union + LLM judge) | **2/10** |
| **scene_card_llm_judge** (union + LLM judge) | **4/10** |

**`production_llm_judge` crashes — and takes down 3 previously-clean hits with it.**
q03, q05, and q09 all scored hit@1 or hit@5 under fusion+cheap-rerank; all three MISS
here. Checked q05 directly: `second_night_013` (the "superior to all desire" chunk) was
confirmed present in the union pool (it independently ranks top-3 on BM25, dense, AND
metadata — see R4), so the judge saw it and still didn't put it in its top 6. **This is
the R1-C/R2/R3 demotion finding holding up on a completely different, wider, unweighted
pool** — not an artifact of what was being reranked before. The mechanism generalizes:
this judge, prompted this way, systematically doesn't prioritize this specific passage
for "does the book undercut this" framings, regardless of what pool it's chosen from.

**`scene_card_llm_judge` is worse on raw score (4/10) but is the ONLY method all
session — across R1, R4, R5, R6 — to solve q08 AND q10.** Verified both aren't flukes:
- **q08**: gold chunk `fourth_night_012` lands at **rank 2 of 6**. Its scene (`scene_25`)
  was one of 14 scenes in the union.
- **q10**: `morning_002` (one of 2 gold ids) lands at **rank 4 of 6**, alongside
  `morning_001`/`morning_003` also pulled into the pool — the judge correctly identified
  the Morning-chapter cluster as the relevant narrative region and populated its answer
  from it, something no fusion-based method managed all session (R1/R4/R6 all had
  `morning_002` sitting at raw fused rank 47-70 of 84).

**But it lost q03, q06, and q07 — all previously easy hits — the same demotion pattern
as the other ablation.** Checked q06 and q07 directly: both gold chunks' scenes
(`scene_17`, `scene_20`) were confirmed present in the candidate pool; the judge simply
didn't select them.

**What explains the asymmetry.** The judge isn't uniformly worse or better — it's highly
sensitive to what pool it's judging. Given the production-style pool (near-duplicate,
high-surface-similarity chunks clustered around a few scenes, since BM25/dense/metadata
all tend to agree on "generically relevant" content), it keeps finding an equally-valid
alternative passage and preferring it (R1-C's exact mechanism). Given the scene-card
pool — wider, and structurally forced to span more DIFFERENT scenes since it's built from
scene-level search rather than chunk-level similarity — the judge has a fundamentally
different, more diverse set of candidates to choose from, and for q08/q10 specifically
that diversity is what put the actually-correct chunk in front of it at all (fusion
never even considered it a candidate). The judge's own discrimination didn't get better;
what it was given to discriminate among did.

**Takeaway.** The LLM judge is not simply "worse than a cheap reranker" — that was true
for the production pool R4 was built on, but this shows it's pool-dependent, not
judge-dependent. It is the ONLY mechanism this session that reached q08 and q10 at all,
which four different fusion-based methods (R1, R4, R5, R6) could not, at the direct cost
of 3 questions those same fusion methods had solved easily. Neither ablation is a better
overall system than `union_method` (R6, still the session's best at 6/10) — but
`scene_card_llm_judge`'s success on q08/q10 is a genuine, reproducible data point that a
wide, narratively-diverse candidate pool plus LLM judgment can do something structurally
different from what fusion + a formula can, even if this particular combination trades
away more than it gains. The natural next experiment this points to: run the LLM judge
ONLY on the questions where fusion has already failed (a fallback tier, not a full
replacement), rather than substituting it wholesale.

---

## R8 — `experiments/retrieval/variants/llm_query_production.py`: an LLM rewrites the query before retrieval.
## Three bugs found on ONE question (q01); two fixed, one open; still a miss

**Question.** q01 ("How does the narrator first meet Nastenka?") was diagnosed earlier
this session as a pure named-entity mismatch, not a real semantic gap: swapping "Nastenka"
out for "the girl by the canal" in the query text moves the gold chunk from dense rank
50/84 to rank 2/84, because that scene happens before she's ever named in the book (see
the sharpened R1-A entry above). Can an LLM query-rewriting stage — given a detailed,
corpus-agnostic prompt for exactly this kind of entity-aliasing — find that fix on its
own, using only the query text (no corpus access, explicitly forbidden from inventing
aliases)?

**Pipeline**: `original_query -> classify_intent(original_query)` (kept separate from the
rewrite) `+ rewrite_query(original_query)` (LLM call, JSON: `rewritten_query`, per-entity
`entities`, flattened to `reference_variants`) `-> llm_query_retrieve(rewritten_query,
intent, reference_variants)` (reimplements production_method's fuse -> cheap_rerank ->
diversify, but with intent supplied externally) `-> answer(ORIGINAL query, ids)`.

**Round 1 (first working version): MISS, two bugs found.**
1. `classify_intent()` was being called on the REWRITE, not the original. The rewrite
   turned "first **meet**" (verb, matches the `EVENT` pattern) into "first **meeting**"
   (noun) — flipped classification to `TEMPORAL` (caught "first" instead), a different
   weight profile, worse result. A classifier tuned on the original query's grammar
   misfired on the rewrite's different grammar.
2. The rewriter's own `entities` field DID correctly propose `"the girl"` /
   `"the young woman"` as alternatives for Nastenka — exactly the known fix — but that
   field was computed and then never used anywhere; only `rewritten_query` (a single
   string, still saying "Nastenka") reached retrieval.

**Round 2 (fix both): still MISS, third bug found — and it's a genuine schema-design
finding, not a model-capability ceiling.** Fixed #1 by passing `classify_intent(original_query)`
in explicitly. Fixed #2 (attempted) by requesting a simpler flat `reference_variants` list
directly from the LLM and folding it into the search text. Result: **the model stopped
proposing "the girl" entirely** — 4/4 repeat runs, deterministic, only ever returning
`["the narrator", "Nastenka"]` (i.e. the entities' own names echoed back, no real
alternatives) or generic role-swaps for "the narrator" alone. Three escalating prompt
fixes tried directly on this flat-list version (soft guidance → an explicit "before the
name is known" heuristic → a hard "MUST include a role-based alternative" rule) — **none
worked**, still 0/4.

**Direct A/B control test resolved it.** Ran the ORIGINAL, unmodified elaborate prompt
(nested `entities: [{query_reference, retrieval_references}]` structure) 4 times: **3 of
4 runs correctly proposed "the girl"/"the young woman" for Nastenka.** So the capability
was never missing from gpt-4o-mini — my flat-list simplification specifically was what
broke it. **Diagnosis: forcing the model to address each entity in its own object
reliably elicits an alternative for a proper name; asking for one shared flat list lets
it skip the harder case (the name) while still handling the easy one (a generic role
word like "narrator").** Reverted the output schema to the nested per-entity structure,
flattened to `reference_variants` in Python afterward — 3/3 repeat runs now correctly
include "the girl". This is the actual finding worth keeping: **output schema shape
measurably changes what an LLM is willing to generate, independent of the underlying
instructions asking for it** — a flat list and a per-entity list, given equivalent
guidance text, produced categorically different behavior.

**Round 3 (bug #2 truly fixed): STILL a miss, on a fourth, distinct problem.** With "the
girl" reliably in `reference_variants` and folded into the query as `"... (also referred
to as: the girl, the young woman, ...)"`, checked the actual fused rank directly:
**50/84 → 47/84.** Barely moved. `rewritten_query`'s own main text still says "Nastenka"
three times ("first meeting... Nastenka; initial encounter with Nastenka; narrator's
first interaction with Nastenka") — three repetitions of the name dominate the embedding
far more than one alternative appended in a parenthetical at the end can offset. The
isolated experiment that got rank 2/84 used a query that DROPPED the name entirely, not
one that said both. Appending is not the same fix as substituting.

**Result: q01 is still a MISS after all three fixes.** Two of three bugs are genuinely
fixed (intent-passing, schema-driven alternative elicitation); the third (fold-in
strength) is open and diagnosed precisely, not a dead end — candidate next steps are
biasing `rewritten_query` to lead with the alternative rather than the name, or fanning
out into multiple full query variants (name-swapped, not name-augmented) and unioning
their retrieval results, closer to what the original isolated experiment actually did.

**Takeaway.** This is the deepest single-question investigation this session, and it
produced two general, reusable findings beyond q01 itself: (1) intent/classification
components downstream of a query rewriter must receive the ORIGINAL query, or grammar
changes from the rewrite silently corrupt them — a real integration hazard whenever two
independently-tuned components are chained; (2) an LLM's willingness to hedge on a named
entity is highly sensitive to OUTPUT SCHEMA SHAPE, not just instruction content — a
finding worth remembering any time a "list of X" field is quietly under-eliciting
compared to what a more structured per-item version of the same request would get.

---

## R9 — `llm_query_production` on all 10 questions: 4/10, a net regression from
## production_method's 6/10

**Question.** R8 was a deep single-question trace (q01) that ended with two of three
bugs fixed but q01 still missing. Run the same pipeline over the full 10-question set to
see the aggregate picture, not just one hard case.

**Result:** `experiments/retrieval/run_llm_query_pilot.py`, raw output
`experiments/retrieval/results/llm_query_pilot_results.json`.

| method | hit@6 |
|---|---|
| production_method (R4) / union_method (R6) | 6/10 |
| **llm_query_production (R8/R9)** | **4/10** |

q01/q02/q08/q10 remain misses, exactly as R8's diagnosis predicted (the fold-in-strength
problem is real and unresolved, not specific to q01). But **q03 and q06 — both clean
hits under plain production_method — are NEW regressions**, and they reveal a mechanism
R8 didn't: query rewriting can actively destroy an already-good literal match, not just
fail to create one that didn't exist.

**q06** ("What does the narrator convince Nastenka to do... on what grounds") originally
worked because of near-perfect BM25 overlap — the gold chunk contains the narrator
literally saying *"I tell you what, write a **letter**"* and the original query shares
enough vocabulary to rank it 1st of 84 (see the single-question walkthrough earlier this
session). The rewrite — *"narrator persuades Nastenka regarding the lodger's silence;
reasons for convincing Nastenka; arguments made by the narrator about the lodger's
continued lack of communication"* — **never says "letter" at all**, replacing the one
concrete, literal, high-signal noun with abstract paraphrases ("communication",
"silence", "arguments"). Checked the fused rank directly: **28th of 84** — not even in
the candidate pool (pool=20), a straightforward regression from rank 1.

**q03** shows the same shape at smaller scale: rewrite moves the gold chunk from rank 1
(original query, confirmed in R4's diagnosis) to **16th of 84** — still barely inside the
pool, but doesn't survive `cheap_rerank` into the top 6.

**Why this happens, precisely.** The rewrite prompt's whole job is turning "how the user
asks" into "how the source text might say it" — exactly the right instinct for q01/q02
(genuine vocabulary gaps) but actively harmful for q03/q06, where the user's ORIGINAL
phrasing already happened to share strong literal vocabulary with the answer passage.
Paraphrasing a query that was already a near-exact lexical match can only make BM25
overlap worse, not better — there was no gap to close, and the rewrite closed a match
that already existed instead.

**Takeaway.** Query rewriting is not a uniform improvement — it has to be selective, and
this pipeline currently rewrites every query unconditionally. The net result (4/10 vs
6/10) makes it a clear regression as a blanket replacement for the original query. Two
paths forward, neither tried yet: (1) run BOTH the original and rewritten query through
retrieval and take the union/best result per-question (cheap insurance against exactly
this failure mode, similar in spirit to R6's union-not-replace fix for scene cards), or
(2) have the rewriter itself judge whether a rewrite is likely to help (e.g. only rewrite
when it assesses `lexical_overlap_expected` as genuinely low) rather than always
producing one. Given R1-A/R8 already show q01/q02/q08/q10's problems don't come from a
fixable vocabulary gap alone (fold-in strength, ranking discrimination), and R9 now shows
unconditional rewriting actively costs previously-solved questions, **plain
production_method/union_method (6/10) remains this session's best, and the safest to
keep using as-is.**

---

## R10 — `experiments/retrieval/variants/production_union_with_llm.py`: union original + up to 4 LLM-generated
## alternative queries by max score. Fixes R9's regression, but not fully — 5/10

**Question.** R9 showed `llm_query_production`'s unconditional single-query REWRITE
regresses 2 previously-solved questions (q03, q06) by paraphrasing away literal
vocabulary that was doing the work. What if the LLM instead generates several
*additional* candidate queries — never replacing the original — intent is classified
from the original only, and each chunk is credited with its best (max) fused score
across ALL variants? This should behave as "union, not substitution": the original
query's own good matches can only be preserved or reinforced, never displaced.

**Pipeline**: `classify_intent(original)` → `generate_alternatives(original)` (LLM call,
up to 4, using the exact user-specified prompt: "alternatives are additional retrieval
attempts, not replacements") → `all_queries = [original] + alternatives` → `fuse(variant,
intent)` for each → `combined[chunk] = max` across variants → `cheap_rerank(joined text,
pool, combined)` → diversify → top 6.

**Result: 5/10** — better than `llm_query_production`'s 4/10, still below
`production_method`/`union_method`'s 6/10.

| method | hit@6 |
|---|---|
| production_method (R4) / union_method (R6) | 6/10 |
| **production_union_with_llm** | **5/10** |
| llm_query_production (R9) | 4/10 |

**q06 is fixed, exactly as designed.** The original query alone still scores rank 1 for
its gold chunk; since it's always one of the variants in the max, its win survives the
union regardless of what the 4 alternatives do. Confirms the core "union, not
substitution" instinct from R6 generalizes to query-level alternatives, not just
candidate-source alternatives.

**But q03 STILL regresses — for a genuinely different, non-obvious reason.** Traced it
directly: the original query alone ranks the gold chunk **5th of 84**. The
max-across-5-variants COMBINED score ranks it **20th of 84** — worse, despite gold's own
combined score (0.730) being marginally HIGHER than its original-alone score (0.723).
**The mechanism: `max` gives every OTHER candidate 5 chances to score well too, not just
the gold chunk.** Chunks that were mediocre under the original wording specifically but
decent under one of the 4 rephrasings get to cherry-pick their best showing and leapfrog
past a chunk that was sharply, decisively correct under exactly one phrasing. Reranking
then compounds it slightly further (18th of the pool-20 survivors).

**This is the same shape of failure as R1-B (RRF fusion stranding a single-signal
winner), one level up.** There, combining two retrievers per-candidate could bury a chunk
strong in one signal but absent from another. Here, combining five QUERY variants
per-candidate does the analogous thing: generating more attempts systematically favors
chunks that are decently relevant under MANY phrasings over the one chunk that is
decisively correct under exactly one specific phrasing — precisely q03's case, where the
original query's wording happened to share near-perfect vocabulary with the answer.

**Takeaway.** "Union, not substitution" (R6, now R10) is a real, validated principle —
it did what it was built to do (protect q06). But "credit the best score across N
variants" is not a free lunch: it has its own dilution mechanism, structurally similar to
RRF's, that a straightforward union of candidate ID SETS (like R6's original/scene-card
union) doesn't have, because there every source either finds a candidate or doesn't —
there's no per-candidate score-inflation-by-more-attempts effect. A cleaner fix worth
trying next: keep each variant's own TOP-K candidate list separate and union the ID SETS
(as R6 did for scenes) rather than blending every candidate's best score into one ranking
— that would let the original query's rank-5 finding stand on its own merits instead of
competing against a max-inflated field. Not yet tried. **production_method /
union_method (6/10) remain this session's best and most robust methods.**

---

## R11 — `production_union_with_llm` v3: union candidate ID SETS (not scores), anchor
## final ranking on the ORIGINAL query alone. Ties production_method exactly (6/10)

**Question.** R10 (v2) fixed R9's regression on q06 but introduced a new one on q03,
diagnosed precisely: scoring every candidate by its MAX fused score across 5 query
variants gives every competitor 5 chances to score well, letting generically-relevant
chunks leapfrog past one that was decisively correct under the original phrasing alone.
Fix proposed at the end of R10: let each variant contribute candidate IDS only (binary —
recall), and compute final RANKING fresh from the original query alone (never blended
across variants) — so a chunk already winning on the original query's own terms can't be
diluted by how other chunks score on *other* queries.

**Pipeline (v3)**: `classify_intent(original)` → `generate_alternatives(original)` (LLM,
up to 4) → each of the 5 variants gets its OWN top-15 via `fuse(variant, intent)` →
UNION those 5 id lists (tracking `convergence`: how many variants' own top-15 included
each id, as a diagnostic) → re-`fuse(original_query, intent)` over just the union → top-20
by that original-query-only score → `cheap_rerank(original_query, ...)` → diversify →
top 6.

**Result: 6/10 — exact tie with production_method, and the exact same hit/miss pattern**
(same 4 misses: q01, q02, q08, q10; same 6 hits, at the SAME ranks: q03 hit@5, q04 hit@2,
q05 hit@1, q06 hit@1, q07 hit@1, q09 hit@1). q03 and q07, which v2 got wrong/worse
(miss, hit@4), are back to production_method's exact ranks (hit@5, hit@1) — the
anchoring fix worked exactly as diagnosed, with no side effects on anything else.

| method | hit@6 |
|---|---|
| production_method (R4) / union_method (R6) | 6/10 |
| **production_union_with_llm v3** | **6/10 — ties the best** |
| production_union_with_llm v2 (R10) | 5/10 |
| llm_query_production (R9) | 4/10 |

**Why it can tie but not beat production_method — checked directly, not assumed.**
Because final ranking always requires the ORIGINAL query to place a candidate
competitively within the union, the alternates can only ever help recall for chunks that
BOTH (a) some alternate phrasing's own top-15 surfaces, AND (b) the original query itself
still scores decently (just not top-15) so it's competitive once the field narrows to the
union. Checked q08 directly: gold chunk (`fourth_night_012`) is **absent from the union
entirely** — none of the 4 alternates' own top-15 lists caught it, despite its original-
query rank being a moderate 22/84 (not catastrophic — this SHOULD have been recoverable
if even one alternate had ranked it in the top 15). The alternates simply didn't diversify
enough for this question.

**The `convergence` diagnostic ("do chunks that appear across variants score higher and
converge?") answers its own question with a genuine, useful negative result: high
convergence tracks agreement, not correctness, when the variants share a blind spot.**
For q01, the WRONG chunks in the final top-6 (`fourth_night_015`, `third_night_004`,
`second_night_016`) all converged at **5/5** — every single variant, including all 4
LLM-generated alternates, agreed on them. Why: none of the 4 alternates thought to drop
"Nastenka" from the query (this alternates-generation prompt asks for different
*formulations*, not entity-aliasing the way R8's rewrite prompt specifically did) — so
all 5 phrasings share the exact same named-entity bias documented in R1-A, and
"convergence" just measures that they all made the same mistake together. Meanwhile
q03's actual gold chunk converged at only 1/5 (only the original query itself found it)
and still won on rank, because anchoring to the original query means a single strong
signal beats broad-but-shallow agreement — the opposite of what "5/5 convergence" would
naively suggest is more trustworthy.

**Takeaway.** This is the architecturally cleanest LLM-augmented method tried this
session: it provably cannot make an already-good match worse (ranking is always anchored
to the original query, exactly as production_method's alone would score it), and it adds
a free chance to catch additional candidates other phrasings surface — at the cost of one
extra LLM call and no guaranteed benefit if the alternates don't diversify meaningfully.
On this 10-question set they didn't diversify enough to catch anything new, so it lands
exactly at parity with the pure-formula method rather than beating it. **The real lesson
from R9-R11 together: every one of the three query-augmentation designs tried this
session (blind rewrite, max-score union, id-set union) tops out at production_method's
6/10 — none has beaten it, only matched or regressed from it.** The ceiling on q01/q02/
q08/q10 is not a query-phrasing problem this session's LLM-side tools have found a way
past; R1-A/R4's "genuine ranking-discrimination ceiling" diagnosis still stands.

---

## R12 — swapping in a much stricter, anti-hallucination alternative-query prompt:
## ZERO change in outcome. Confirms the ceiling is structural, not a prompt quality issue

**Question.** R11's alternatives prompt was loosely specified ("generate different
formulations"). What if the LLM is instead given a much more rigorous prompt — exactly 4
labeled variants with distinct roles (ORIGINAL / SOURCE-LEXICON / REFERENCE-ROBUST /
SEMANTIC-EVENT), explicit rules against using "pretrained knowledge, world knowledge,
literary knowledge, or assumptions about what the source probably says," and repeated
instructions to never invent entities or guess source-specific terminology? Swapped this
prompt into `production_union_with_llm.py`'s `generate_alternatives()` (same v3
architecture: id-set union, ranking anchored on the original query) and reran all 10
questions.

**Result: 6/10 — identical to the looser prompt (R11), question for question, RANK for
RANK.** Every single one of the 10 questions landed at the exact same hit/miss and the
exact same rank as R11's looser-prompt version. The only observable difference was
`union_size` shrinking slightly across the board (e.g. q01: 29 → 23, q05: 20 → 17) —
expected, since this prompt's constraints produce more conservative, closely-related
alternatives — but that shrinkage never changed which chunks survived into the final top
6.

**q01 confirmed still unsolvable, and for exactly the reason predicted before running
it.** This prompt explicitly forbids the "the girl"-style inference that R8 needed
literary/narrative knowledge to make ("assume you know NOTHING about the underlying
source... do not use literary knowledge... do not infer names, entities... not stated in
the query"). Checked the actual alternatives generated: `"How does the narrator initially
encounter Nastenka?"`, `"...first come across Nastenka?"`, `"What is the first meeting...
like?"` — all three keep "Nastenka" every time, exactly as this stricter prompt requires.
This isn't a prompt failure; it's the prompt correctly doing what it was asked (staying
strictly query-grounded), which happens to be incompatible with the one thing that would
have helped this specific question.

**Takeaway — this closes out the query-augmentation line of investigation for this
session.** Four genuinely different alternative-query-generation designs have now been
tried against the same 4 persistent misses (q01/q02/q08/q10):
1. R8/R9: a single unconditional rewrite (regressed 2 other questions besides)
2. R10 (v2): max-score union across 5 variants (fixed R9's regression, broke a different one)
3. R11 (v3, loose alternatives prompt): id-set union, original-anchored ranking — 6/10
4. R12 (v3, strict alternatives prompt): same architecture, far more careful prompt — 6/10

**None has moved past production_method's 6/10, and the last two — architecturally the
soundest of the four — produced literally identical results despite substantially
different prompts.** That is strong evidence the ceiling on q01/q02/q08/q10 is not a
prompt-engineering problem reachable by asking an LLM to phrase the query differently, no
matter how that's done — it's the structural ranking-discrimination ceiling R1-A/R4/R11
already diagnosed (weak on BM25, dense, AND metadata simultaneously; or present but
outranked by generically-relevant competitors). Further iteration on alternative-query
prompts specifically is very unlikely to be worth pursuing further on this corpus/gold
set — a genuinely different mechanism (a 4th retrieval signal, or multi-hop composition
for q10's two-part question) is the more promising direction if these 4 are worth
chasing further at all.

---

## R13 — re-testing the cross-encoder (previously dropped in V1) on a completely
## different candidate pool: `experiments/retrieval/variants/production_with_encoder.py` and
## `experiments/retrieval/variants/production_with_llm_variants.py`. The old verdict holds, decisively

**Question.** `retrieval.py`'s own docstring (predating this session) says: *"the
cross-encoder was evaluated and dropped — it scored surface relevance, not question
intent, on this literary domain"* — and `requirements.txt` confirms
`sentence-transformers`/`torch` were removed for the same reason. That verdict was
reached on V1's single-query hybrid candidate pool. Does it still hold on
`production_method`'s intent-weighted fusion pool, and on the wider multi-query-variant
union pool from R11 — both meaningfully different, and better-performing (6/10), than
what V1 tested against?

**Setup.** Re-added `sentence-transformers`/`torch` (first non-OpenAI-API dependency
this session) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (the standard, most common
pretrained cross-encoder — trained on MS MARCO web-search query/passage relevance pairs).
Two new files, both swapping `cheap_rerank`'s formula for
`cross_encoder_rerank(query, cand_ids)` — jointly encodes (query, passage) pairs and
scores relevance directly, unlike BM25/dense which score them independently:
- `production_with_encoder.py`: production_method's exact single-query candidate pool
  (`classify_intent` → `fuse` → top 20), cross-encoder reranks instead of `cheap_rerank`.
- `production_with_llm_variants.py`: `production_union_with_llm`'s exact multi-variant
  union (original + up to 4 LLM alternatives, each contributing its own top-15,
  unioned), cross-encoder reranks against the ORIGINAL query only (R11's anchoring fix,
  kept). Isolates two questions at once: does the cross-encoder help at all, and does a
  wider candidate pool help it specifically.

`experiments/retrieval/run_encoder_pilots.py` runs both over the same 10 questions.

**Result: 2/10 (plain) and 3/10 (with variants) — a severe regression from both
cheap_rerank baselines (6/10 each).** The old V1 verdict holds, and holds decisively,
even on a completely different and better-performing candidate pool than what it was
originally tested against:

| method | reranker | hit@6 |
|---|---|---|
| production_method (R4) | cheap_rerank (formula) | 6/10 |
| production_union_with_llm v3 (R11) | cheap_rerank (formula) | 6/10 |
| **production_with_encoder** | cross-encoder | **2/10** |
| **production_with_llm_variants** | cross-encoder | **3/10** |

**It lost q03, q05, q09 — exactly the interpretive/motivation questions cheap_rerank
handles cleanly — and gained nothing new on q01/q02/q08/q10.** Not a wash; a strict
regression on both fronts (no compensating recall win, despite the wider candidate pool
in the `_variants` version).

**Traced q05 specifically** (gold: `second_night_013`, the "superior to all desire"
monologue) since it's the clearest case: fused-pool rank was **1st of 20** before
reranking — cheap_rerank correctly keeps it there (hit@1 in R4). The cross-encoder drops
it to **5th of 20**. What it prefers instead: `second_night_014` — the chunk
immediately AFTER the gold one, which (due to the chunker's 50-token overlap) contains
the tail end of the exact same sentence ("...he desires nothing, because he is superior
to all desire, because he has everything, because he is sati[ated]..."). This is a
genuinely defensible individual call — the cross-encoder found a chunk containing the
literal quoted phrase — but it's symptomatic of the documented failure: the model is
matching on literal phrase overlap (surface relevance), with no apparent sensitivity to
which of two near-duplicate, overlapping chunks is the more complete/canonical one for
answering an interpretive "does the book undercut this" question. A keyword-overlap
formula (`cheap_rerank`) that already has the fused score as a strong prior doesn't get
pulled off course by this; a cross-encoder starting from scratch on each pair does.

**Takeaway — this closes the question definitively, not tentatively.** Two independent
re-tests, on genuinely different (and both better-performing) candidate pools than V1
used, both reproduce the original "surface relevance, not question intent" verdict with
a large, unambiguous margin (6/10 → 2-3/10) rather than a marginal one. This isn't
"the cross-encoder needs a bigger candidate pool" or "needs the original V1 setup" —
it underperforms regardless of what feeds it, on this literary/interpretive-heavy gold
set. Re-confirms (a fourth time this session, after the LLM judge in R1-C/R2/R7) that a
generic relevance-scoring model — whether a big reasoning LLM or a small purpose-built
cross-encoder — is the wrong tool for literary interpretive judgment on this corpus;
`cheap_rerank`'s crude keyword-overlap formula, precisely because it doesn't try to
"understand" the passage at all, keeps winning. `production_method`/`union_method`/
`production_union_with_llm` v3 (all 6/10, all using `cheap_rerank`) remain this
session's best methods. `sentence-transformers`/`torch` can reasonably be removed from
`requirements.txt` again unless a future session wants to try a DIFFERENT cross-encoder
(e.g. one fine-tuned on literary or narrative-QA data specifically, not MS MARCO) —
this result indicts the MS MARCO-trained model's domain mismatch, not the cross-encoder
architecture in principle.

---

## R14 — `experiments/retrieval/variants/production_with_hyde.py`: HyDE, constrained corpus-agnostic, union'd with
## the original query. Exact tie with production_method (6/10), no regressions, no new
## hits — and the reason is a direct, logically necessary consequence of the constraint

**Question.** Gao et al.'s RAG survey (§III, "Query Transformation") names HyDE
specifically: instead of embedding the bare question, prompt an LLM to write a
**hypothetical answer passage** and embed that — answer-to-answer similarity instead of
question-to-passage. Structurally different from every rewrite/alternative-query
technique tried so far (R8-R12), all of which reworded the *question*. But vanilla HyDE
works by deliberately exploiting the model's parametric knowledge to write a fluent,
plausible-sounding answer — and this model plainly has White Nights memorized (a famous
public-domain text), so unconstrained HyDE would really be testing "does the model
already know this book," not the technique itself. Built it with the same
corpus-agnostic GLOBAL RULES as R12's alternative-query prompt (no pretrained/world/
literary knowledge, no inventing entities/events beyond the query, preserve the query's
own concrete nouns) — a genuinely harder test than HyDE is normally given.

**Pipeline**: `classify_intent(original)` → `generate_hyde_passage(original)` (LLM,
constrained) → `_dense_candidates(hyde_passage, pool)` (HyDE is dense-only — answer-to-
answer embedding similarity, not a BM25/metadata substitute) → union with the original
query's own `fuse()` top-20 → `cheap_rerank` anchored on the ORIGINAL query (R11's fix,
kept) → diversify → top 6.

**Result: 6/10 — exact tie with production_method, identical hit/miss pattern.** No
regressions (the union principle held again, fourth time now after R6/R11/R12), but also
no new hits — despite HyDE contributing 4 to 11 genuinely new candidates per question
that weren't in the original query's own top-20.

**Why, checked directly rather than assumed: the constraint that makes this a fair test
also makes it structurally unable to fix q01.** q01's own hypothetical passage:

> *"...my eyes were immediately drawn to a young woman sitting alone at a table by the
> window... That was the moment I first met **Nastenka**..."*

The name is there — correctly, per the rules. The query itself says "Nastenka," and the
prompt explicitly requires *preserving* concrete nouns already given in the query, not
inventing new ones. **The fix that worked back in R8 (drop the name, describe "the girl
by the canal" instead) required knowing, specifically, that this scene happens before
she's named in the book — a fact about the SOURCE TEXT's structure, not something
derivable from the query alone.** A genuinely corpus-agnostic prompt cannot discover
that fix by construction, for exactly the same reason R12's strict alternative-query
prompt couldn't. This isn't a HyDE-specific limitation; it's the second confirmation
that the corpus-agnostic design constraint itself, applied faithfully, rules out the one
class of fix that solves q01 — consistent, not a fluke.

**For q02/q08/q10, the hypothetical passages land in the right TONE but not the right
CONTENT.** q08's hypothetical: *"...watching Nastenka's eyes light up with
recognition... 'Now I am happy,' I declared..."* — genuinely close in mood to the real
scene, correctly guesses the quoted line even. q10's: *"...the weight of Nastenka's
final letter rested heavily in my hands..."* — correctly identifies that a letter is
involved. Both are recognizably in the right emotional register, generated from the
question's own wording alone — a real, working demonstration that the constrained
prompt isn't just refusing to try. But neither passage contains the SPECIFIC narrative
detail (the lodger's absence being *why* q08's joy is precarious; what the letter
actually *says* for q10) that would make its embedding land close enough to the real
passage — because that detail lives in the source text, not the question, and is
exactly what the corpus-agnostic constraint forbids inferring.

**Takeaway.** Three techniques now — R11's variant union, R12's strict rewrite, and this
HyDE test — all land at exactly 6/10 with `cheap_rerank`, all via the same "union, not
substitution" architecture, and all fail on the same 4 questions for a common, principled
reason: a corpus-agnostic LLM component cannot rediscover a fact about the source text's
own structure (an unnamed character, a doomed-in-hindsight declaration, a letter's actual
content) that isn't recoverable from the query text alone. This is a stronger, more
general version of R1-A's original diagnosis: the q01/q02/q08/q10 ceiling isn't just "the
current signals don't rank it well" — it's that fixing it at all requires SOME component
in the pipeline to know something about White Nights specifically, and every method
tried this session that's kept itself honestly corpus-agnostic (correctly, by design)
has been unable to supply that. The one method that DID solve q01 (R8, un-throttled) only
worked by letting the rewriter use exactly the literary knowledge every other method
here was built to avoid — which is itself the finding: on THIS corpus, "stay
corpus-agnostic" and "solve q01" are in direct tension, not simultaneously achievable by
better prompting.

---

## R15 — "why does the dreamer still love the girl": interpretive questions can't find
## chunks for the love of god. A new failure mechanism, verified, and even the fix
## doesn't fully close it

**Question.** An ad-hoc spot check (not part of the 10-question gold set), tested on
`production_method` and `production_with_hyde` after the grandmother/pin question (R14's
follow-up) turned out too easy to stress-test anything. This one has no single clean
textual anchor — genuinely interpretive — but the best candidate gold evidence is the
Fourth Night confession scene, `white_nights_fourth_night_004`: *"It's impossible, but I
love you, Nastenka!"*

**Result: both methods return the IDENTICAL top-6 list, and it's a clean miss for both.**
The confession scene doesn't appear anywhere in either result. Checked its raw fused
rank directly: **21st of 84** — not close to the pool cutoff, let alone top-6.

**Mechanism, verified by the same ablation used for q01 (R1-A): swap the overloaded
term, watch the rank move.**

| query phrasing | confession-scene rank (of 84) |
|---|---|
| "Why does **the dreamer** still love **the girl**?" | 21 |
| "Why does **the narrator** still love **Nastenka**?" | 11 |
| "Why does **he** still love **her** despite everything that happened?" | 8 |

Confirms the mechanism directly: **"the dreamer" is the book's own heavily-repeated
self-label for a completely unrelated theme** — the Second Night philosophical monologue
about isolation and escapist fantasy — so lexical/thematic matching pulls toward "passages
that talk about the dreamer generally" (`second_night_012/014/019`, all isolation/fantasy
content, all present in both methods' actual top-6) instead of "passages where he loves
her." This is q01's exact mechanism (a token that's genuinely in the query is even MORE
common somewhere irrelevant in the book, and wins the tug-of-war) — but triggered by a
recurring ROLE-LABEL instead of a proper name. First confirmation this session that the
"generic-but-frequent term dominates" failure isn't specific to named entities; it's a
general property of any word the query shares with an unrelated, heavily-repeated theme.

**But — reported honestly rather than declared solved — even the best phrasing tested
(rank 8) still doesn't reach top-6.** Stripping the overloaded terms helps
substantially (21 → 8) but doesn't fully close the gap. Some real difficulty remains
beyond the term-choice issue: "why does X still love Y" is a diffuse, interpretive
question with no single concentrated answer passage (the Dreamer's love is shown
throughout the book — his devotion, his sacrifice offer, the ending — not argued in one
place the way "why did the lodger's promise fail" has one scene). This is a different,
harder problem than the named-entity/token-collision mechanism, layered on top of it, not
explained away by it.

**HyDE's failure mode here is notably different from every prior test, and equally
informative.** The hypothetical passage was fluent, on-theme, and entirely empty of
anything concrete: *"...her laughter echoing in his mind... love, once ignited, never
truly fades away..."* — generic romantic sentiment with no scene, no reason, nothing
specific enough to distinguish the real confession from any other tender moment in the
book. Its 9 new candidates entered the union and changed nothing. Earlier HyDE tests
(R14) at least landed on the right *region* even when short on specifics (q08's guessed
line, q10's letter); here, with no concrete anchor in the question itself to build from,
the constrained prompt had nothing to ground a specific-enough passage on, and produced
pure mood instead.

**Takeaway.** Two failure mechanisms are now confirmed distinct and compounding, not
duplicates: (1) token-collision with an unrelated frequent theme (fixable, partially, by
rephrasing — same family as q01, R1-A), and (2) diffuse interpretive questions with no
single concentrated answer chunk (a different, harder problem — related to R1-C's
"gold label too narrow" finding, but sharper here since there may be no single adequate
gold chunk at all, only a spread of supporting evidence across the whole book). Neither
`production_method` nor `production_with_hyde` handles either one on this question; the
combination of both mechanisms firing at once may explain why this one is harder than
any of the 10 gold-set questions, all of which have at least one identifiable answer
passage even when hard to find.

---

## R16 — `experiments/retrieval/variants/production_llm_retries.py`: LLM sufficiency check gates a second
## retrieval round. Safe (never regresses) but the gate is miscalibrated — retries on
## 10/10 questions, including 6 that were already correct

**Question.** Adapted the adaptive/iterative retrieval pattern from Gao et al.'s survey
(Self-RAG/Flare/CRAG — §V) for a closed corpus: run `production_method`'s plain
retrieval, ask an LLM whether the result is actually SUFFICIENT to answer the question
(not just topically related), and only if insufficient, generate targeted
`search_directions` for a second retrieval round, union with the first round, and
re-rank — anchored on the ORIGINAL query throughout (R11's fix, kept). Motivated
directly by the q10 failure documented back in R4: retrieval missed the ending entirely,
and instead of noticing, the system confidently answered *"the story does not provide a
clear resolution"* — a silent failure this design is meant to catch.

**Pipeline**: `production_retrieve(query)` (pure, unmodified) → `check_evidence(query,
initial_ids)` (LLM, strict: judges only the given passages, told explicitly not to use
outside knowledge of the book to rubber-stamp "sufficient") → if insufficient,
`search_directions` each get their own `fuse()` pass over all 84 chunks → union with the
initial candidates → `cheap_rerank` anchored on the original query → diversify → top 6.

**Result: 6/10 — exact tie with production_method, IDENTICAL ranks on every single
question** (q03 hit@5, q04 hit@2, q05/q06/q07/q09 all hit@1, q01/q02/q08/q10 all miss —
every number matches R4 precisely).

**Finding 1 (a real calibration problem): the checker retried on 10 of 10 questions —
including all 6 that were already correct.** This isn't "adaptive" retrieval in
practice; it's "always do a second round," which pays for an extra LLM call and
retrieval pass on every question regardless of whether the first pass already succeeded.
Traced q05 (already hit@1) directly to see why a CORRECT retrieval got flagged
insufficient:

```json
"supported_claims": ["The dreamer desires nothing because he is superior to all
  desire.", "...creates his own life to suit his latest whim.", "...believes his life
  is real and substantial, not a delusion."],
"missing_evidence": ["Whether the book presents the dreamer's perspective as genuine
  wisdom or undercuts the claim."]
```

The checker correctly extracted the passage's content, then declined to call it
sufficient because the passage states the CLAIM but doesn't itself contain the book's
VERDICT on it. But that verdict is properly the ANSWERING step's job — inferring stance
from `narrative_relation` tags and literary judgment, not something retrieval can hand
over as a literal sentence. **This is R1-C's finding recurring in a new place**:
interpretive "does the book undercut X" questions are, by definition, ones where no
single passage contains "the answer" — a strict sufficiency check that expects the
verdict to be textually present will flag EVERY interpretive question as insufficient,
almost tautologically, regardless of retrieval quality.

**Finding 2 (reassuring, and a real architectural validation): even retrying on
everything, not one previously-correct answer got displaced.** Despite the checker's
over-triggering, final ranks for all 6 hits are byte-identical to production_method's.
This is the "anchor final ranking on the original query, never blend/max across
sources" principle (first established in R11 to fix R10's dilution bug) proving robust a
third time now (R14 HyDE, this) — the second retrieval round can only ever ADD
candidates that must still win on the original query's own terms, so an unnecessary
retry costs latency and API spend but never correctness.

**Finding 3: the search_directions inherit R15's exact vocabulary trap.** Ran the
R15 spot-check question ("why does the dreamer still love the girl") through this
pipeline directly — the checker correctly flagged insufficiency, but its own
`search_directions` were `["reasons for the dreamer's love for the girl", "the
dreamer's feelings about the girl", ...]` — **still using "the dreamer" and "the
girl,"** the exact overloaded terms R15 identified as pulling retrieval toward the
wrong theme. The checker diagnoses WHAT's missing accurately but has no visibility into
WHY the first retrieval failed (it never sees the token-collision mechanism, only the
content gap), so its fix-attempt reproduces the same failure. Still a miss.

**Takeaway.** The architecture is sound and provably safe — union-not-substitution with
original-query anchoring means a badly-calibrated gate costs efficiency, never
correctness, which is the right failure mode for a safety mechanism to have. But the
gate itself needs real calibration work before it's actually adaptive rather than
"always retry": the most promising fix, free to implement since intent is already
classified — **route the sufficiency bar by intent.** For EVENT/LEXICAL_FACT questions,
keep the strict standard (the fact should be literally present). For
INTERPRETIVE/MOTIVATION questions, accept "the passage clearly states the relevant claim
or scene" as sufficient without also requiring the book's evaluative verdict spelled
out — that inference is the answering step's job, not retrieval's. Untested, but
directly motivated by Finding 1's traced example, not a guess.

---

## R17 — `production_llm_retries` v2: the intent-aware, ready-to-run-query prompt.
## Found and fixed a real truncation bug; the over-triggering persisted anyway — it's
## verdict-calibration, not visibility

**Question.** R16 diagnosed two problems and proposed fixes: (1) the checker demanded an
explicit textual "verdict" for interpretive questions, when that's the answering step's
job — fix by giving EVENT/LEXICAL_FACT a STRICT bar and INTERPRETIVE/MOTIVATION a
MODERATE one; (2) recovery queries reproduced the exact overloaded vocabulary that
caused the original miss — fix by requiring ready-to-run, anchor-specific queries and
explicit `search_history` awareness. Swapped in a much more detailed, intent-aware
prompt implementing both fixes directly, with literal "Bad: ..." / "Better: ..." examples
for the vocabulary trap specifically.

**Result: still 6/10, still retried on 10 of 10 questions — including every question that
was already correct.** The intent-aware bar did not measurably reduce over-triggering.

**A real bug found and fixed along the way, independent of the prompt swap.**
`check_evidence()` was truncating each passage to 600 characters before showing it to
the checker (copied from `retrieval.py`'s `rerank()`, which only needs a coarse relevance
taste). Chunks run up to ~1750 characters. Checked q06 directly: the gold chunk is 1430
characters, and the actual answer — *"I tell you what, write a letter"* / *"you have a
right to... because he made you a promise"* — falls entirely after character 600. **The
checker's INSUFFICIENT verdict on q06 was correct given what it was shown; it simply
never saw the answer.** Fixed by passing full chunk text, no truncation.

**But after the fix, still 10/10 retried — confirming truncation was not the (sole)
cause.** Re-checked q06's reasoning with full text now visible:

```json
"reason": "The retrieved passages do not explicitly state what the narrator convinces
  Nastenka to do... Key details about the action... are missing."
"new_queries": ["What does the narrator suggest Nastenka WRITE to the lodger...",
  "What reasons does the narrator give... for WRITING to the lodger?", ...]
```

**The checker's own recovery queries reveal it correctly extracted "write a letter" from
the full text — then verdicted INSUFFICIENT anyway, contradicting its own output.** This
points to the prompt itself, not a data problem: its "Intent-specific evidence bar"
section is almost entirely warnings against false positives ("Do not consider evidence
sufficient merely because...", repeated for both bars) with no equivalent counterweight
telling the model when to confidently commit to SUFFICIENT. That asymmetry plausibly
biases gpt-4o-mini toward near-reflexive INSUFFICIENT regardless of the actual evidence
in front of it — the same shape of miscalibration as R16, just relocated from "wants an
explicit verdict" to "defaults to caution regardless of intent."

**The vocabulary trap survives too, despite the prompt now containing the EXACT failure
as a worked "Bad:" example.** Reran R15's question ("why does the dreamer still love the
girl") — the prompt literally lists *`"Dreamer's feelings toward the girl"` — bad, if
those concepts already failed* as a negative example. The checker's actual output:
`["dreamer feelings about the girl", "dreamer emotional state regarding love for the
girl", ...]` — reproducing almost exactly the example it was shown NOT to produce. Not a
subtle miss; the model saw its own mistake spelled out in the instructions and made it
anyway.

**Finding that matters most: the architectural safety property holds a fourth time.**
Despite retrying on all 10 questions and generating recovery queries that frequently
don't escape the failure they're meant to fix, **zero regressions** — every rank is
identical to `production_method`'s. `cheap_rerank` anchored on the original query (R11)
continues to make this class of over-eager augmentation safe by construction: it can
waste API calls and latency, but it structurally cannot make a correct answer worse.

**Takeaway.** Two more confirmations this session that a more elaborate, more explicit
prompt does not reliably fix a model's calibration or instruction-following on this
task, even when the prompt directly anticipates and warns against the exact failure
observed (matches R12's finding that a stricter anti-hallucination prompt didn't change
q01's outcome at all, and F11 in the tagger study's finding that examples alone don't
substitute for a mechanical procedure). The real, verified win from this round is the
truncation bug fix, which is a genuine, permanent improvement independent of whether the
sufficiency gate itself is ever well-calibrated. If this gate is worth pursuing further,
the next lever isn't more prompt detail — it's giving the model a FEW WORKED EXAMPLES of
correctly verdicting SUFFICIENT (not just warnings against insufficient), on the theory
that the current prompt has taught caution but never demonstrated confidence.

---

## R18 — `production_llm_retries` v3: real bounded multi-round loop + attempt-awareness
## prompt section. Partial calibration improvement (2/10 now stop early, was 0/10); still
## 6/10 hit@6, architecture safety holds a fifth time

**Question.** R17 found the checker retried on 10/10 questions regardless of the
intent-aware bar. Added the "Retrieval attempt awareness" prompt section (told the model
`retrieval_attempt`/`max_retrieval_attempts`, instructed it to become "increasingly
selective" near the cap and stop generating queries once no materially different anchor
remains) — and, necessarily, rebuilt the single-retry architecture into a real BOUNDED
LOOP (up to `max_retrieval_attempts=3` rounds), since the prompt now expects that state to
be real and to change round over round, not just be described once. `search_history`
accumulates across ALL rounds, not just round 1, so a later round can see what earlier
rounds already tried.

**Result: still 6/10, identical ranks to production_method on every question.**

**Real, measurable progress on calibration — but partial.** 2 of 10 questions (q02, q04)
now correctly return `SUFFICIENT` on round 1 and stop, versus 0 of 10 before this
change. That's genuine evidence the attempt-awareness framing helped *something* — those
two are the first cases all session where this checker, in any version, recognized
sufficient evidence without being argued out of it by its own strictness.

**But 8 of 10 still burn all 3 rounds, including 5 that were already correct at round
1** (q03, q05, q06, q07, q09 — all hit@1 or hit@5, matching production_method exactly).
Pulled every round's verdict for all 10 questions: **every single round-3 verdict is
INSUFFICIENT, with zero exceptions** — the loop never once got a chance to end early on
"no further gain," because the checker never once decided that after 2 rounds of
additional (unhelpful) evidence. This means the "become increasingly selective near the
cap" instruction had no observable effect on the 8 questions that didn't already resolve
at round 1 — attempt-awareness changed the EASY cases (round-1 sufficiency), not the
HARD ones (persistent insufficiency despite already being answerable).

**This is consistent with R17's diagnosis, not a contradiction of it.** The checker's
verdict-level bias toward INSUFFICIENT (established in R17: it can correctly extract the
right content into its own `new_queries` and still verdict INSUFFICIENT) is a different
axis from attempt-awareness. Telling it "you're running low on attempts, be selective
about whether to keep searching" doesn't fix "you keep deciding correct evidence isn't
enough" — they're separate calibration problems, and this prompt addition only targeted
the first one.

**Cost consequence, worth being explicit about.** 8 of 10 questions now cost 3x the
evidence-check LLM calls (one per round) plus up to 3 rounds of extra retrieval work,
for zero change in the final answer on all 8. The architecture's safety property (R11,
now confirmed a fifth time) means this is wasted spend, not wasted correctness — but it
is real, unnecessary cost at this calibration level.

**Takeaway.** The attempt-awareness addition is a genuine, if narrow, improvement — real
behavior change on 2 questions, zero regressions, same robust safety guarantee. It
doesn't touch R17's deeper finding (the checker's verdict is biased toward caution
independent of actual evidence quality) — that remains open, and remains the more
consequential problem: it's why 5 of the 6 correct answers this session still cost 3
rounds of work to reach the same result 1 round would have given for free. The R17
takeaway stands unrevised: worked examples of confidently correct SUFFICIENT verdicts,
not more prompt structure around WHEN to stop searching, are the more promising next
lever.

---

## R19 — `experiments/retrieval/variants/production_with_weighted_context.py`: explicit RAG-Token-style confidence
## weights per passage. Mechanically correct, answer unchanged — disproves the
## hypothesis it was built to test, and locates the real gap one level upstream

**Question.** A spot check earlier this session ("why does the narrator become so
emotionally attached to Nastenka despite knowing she loves someone else") found
`second_night_014` (the isolation/imagination passage) correctly retrieved into the top
6, but never referenced in the generated answer — which instead built its whole
argument from the other five passages. Hypothesis: nothing in the flat, unweighted
context block told the answering LLM that passage was retrieval-relevant at all, unlike
the original RAG paper (Lewis et al. 2020), where retrieval scores are mathematically
baked into generation as explicit per-passage (RAG-Sequence) or even per-token
(RAG-Token) weights.

**What was built** (approximating RAG-Token's spirit, not its literal mechanism — true
marginalization needs logprob access the chat API doesn't expose): `rerank_scores()`
added to `production_method.py` (a safe refactor exposing `cheap_rerank`'s actual numeric
scores, not just its sort order — `cheap_rerank` itself is unchanged). The kept top-6
scores are softmax-normalized into percentages summing to 100%, and each passage in the
context is explicitly labeled `RETRIEVAL CONFIDENCE: NN%`, with an added instruction
telling the answering LLM to weight higher-confidence passages more heavily but not
ignore lower-confidence ones that are directly relevant.

**Result: the mechanism works exactly as designed, and the answer is unchanged anyway.**
`second_night_014` came out as the LOWEST-confidence passage in the set (12%, tied for
last) — not unlabeled-and-overlooked, but explicitly and correctly flagged as weak. The
regenerated answer, with all six passages now carrying visible confidence numbers, still
never references it — built from the same Third/Fourth Night material as the original,
unweighted run.

**This disproves the hypothesis rather than confirming it, and relocates the real
explanation.** Re-reading what `second_night_014` actually says: *"he desires nothing,
because he is superior to all desire, because he has everything..."* — the RAPTUROUS,
self-sufficient half of the dreamer's monologue. It argues the dreamer needs NOTHING,
which is close to the opposite surface claim from "he's lonely and needs connection."
The passages that would actually support a loneliness-driven-attachment reading are the
monologue's SELF-CRITICAL companions (documented in R1-C: "moments of returning
sobriety, which are awful," "so luxuriously deceived him") — which were never retrieved
at all. **Both the reranker (independently, via the formula) and the answering LLM
(independently, via its own judgment) correctly rated this specific passage as weak
support for this specific question.** The confidence score wasn't hiding useful evidence
under a low label — it was accurately measuring genuinely weak evidence.

**Takeaway.** The gap was never in the answering step's attention/weighting behavior —
it's one level upstream, in candidate RECALL: retrieval found a topically-related
passage but not the specific companion passages that would make the loneliness
interpretation actually supportable. Explicit confidence signaling can't fix a recall
gap; it can only (and, here, did) accurately reflect what's already in the pool. This is
a clean instance of a broader pattern worth remembering: before attributing an answer
gap to the generation step ("it didn't use what it had"), check whether the pool
actually contained what the answer needed — R19 exists because that check wasn't done
before proposing the fix, and doing it after confirms the fix was aimed at the wrong
stage of the pipeline. `rerank_scores()` and the softmax-weighting machinery remain
available in `production_method.py` / `production_with_weighted_context.py` for future
use, since the mechanism itself is verified to work correctly — it just wasn't this
question's actual problem.

---

## R20 — `src/production_react.py`: a genuinely ReAct-faithful agent (Yao et al. 2023).
## 4/10 — a real regression from both production_method and production_llm_retries
## (6/10 each), but with the cleanest evidence all session that q01's ceiling is a
## retrieval-tool limit, not an insufficient-effort one

**Question.** R16-R18's `production_llm_retries` is a two-role design (a stateless
sufficiency-checking LLM + a separate deterministic retrieval formula), not what the
ReAct paper actually describes: one continuous model, one growing Thought/Action/
Observation trace, fine-grained atomic actions (`Search`, `Lookup`, `Finish`), each
preceded by a fresh `Thought` in the SAME context as everything before it. Built a
faithful version: a single Python string is the agent's entire memory, resent in full on
every call; generation is cut off with `stop=["Observation:"]` so the model can never
hallucinate its own observation; `Search[query]` runs real retrieval
(`production_method`'s fuse+rerank, k=5); `Lookup[string]` is a genuine Ctrl+F *within
passages already found this session* — the one action type nothing else this session
has an equivalent of.

**Result: 4/10 — worse than both `production_method` and `production_llm_retries`
(6/10 each).** Lost q03 and q06, both previously trivial hit@1/hit@5. q09 survived but
fell from hit@1 to hit@5.

**Finding 1: severe stopping miscalibration — worse than R16-R18's already-documented
problem.** Only 2 of 10 questions (q05, q07) ever called `Finish[]`. The other 8 ran the
full 6-step budget, wandering through new self-generated search phrasings without ever
deciding they had enough — even on questions where the very first search already
contained the answer (see Finding 2). This is the same shape of miscalibration R16-R18
found in the two-role checker (biased toward "not yet," rarely toward "done"), now
appearing in a single continuously-reasoning agent too — ruling out "the checker role is
the problem" as the explanation, since there's no separate checker role here at all.

**Finding 2: the regression on q06 is precisely diagnosable, and it's R9's exact
failure mode recurring in a new architecture.** Traced q06's trace directly. The
original question — *"What does the narrator convince Nastenka to do about the lodger's
continued silence, and on what grounds does he argue for it?"* — already has
near-perfect literal overlap with the gold passage (confirmed back in R4: rank 1/84 on
raw BM25). But the agent's OWN first move is to compose a search query before ever
trying the original wording: `Search[narrator convinces Nastenka about lodger's
silence]` — dropping "continued," "argue," and "grounds." That rephrase alone was
enough to lose the gold chunk from the top-5 results of that first search. **Unlike
`production_method` (always searches the user's literal original text) or
`production_llm_retries` (round 1 is explicitly, always the original query, only later
rounds rephrase), this agent's very first action is already a self-generated paraphrase
— there's no "try the literal question first" step built into the design at all.** This
is the same paraphrase-destroys-a-good-match mechanism R9 found in
`llm_query_production`, now shown to recur even in a genuinely different architecture,
because nothing in ReAct's own design guarantees the first action preserves the
original wording — the paper's own domain (open web search over named entities) may
simply not have exposed this failure mode the way a small, literal-overlap-sensitive
closed corpus does.

**Finding 3 — genuinely the most interesting, and a real methodological win even inside
a net-negative result: q01's trace is the cleanest evidence all session that its ceiling
is a retrieval-TOOL limit, not an effort/attempts limit.** The agent used all 6 steps as
6 *genuinely different*, self-aware search reformulations — the trace shows real,
correct reasoning at each step ("the search results still do not provide a clear
description... I need to refine my search... I will try searching for the specific
location..."). It tried: "narrator first meets Nastenka" → "...meets Nastenka for the
first time" → "...description first encounter" → "...first meets Nastenka location" →
"first meeting Nastenka" → "...sees Nastenka for the first time." **Every single one
still says "Nastenka" — and the correct chunk (which never uses her name — R1-A) never
appeared in any of the 6 searches.** A genuinely agentic, self-correcting, multi-step
reasoning process — exactly the kind of "more effort, smarter iteration" fix one might
hope would eventually stumble onto dropping the name — never did, across 6 real,
distinct attempts. This is stronger evidence than any single-shot test could give that
q01's ceiling is in the underlying retrieval TOOL's blind spot (the same tool every
method this session shares), not in how many times or how thoughtfully a caller invokes
it.

**Takeaway.** ReAct's architecture (single trace, fine-grained actions, real
Lookup-style drill-down) is a real, distinct paradigm from the two-role
checker-and-retry design — but in this implementation it underperforms both simpler
methods, for two compounding, IN-PRINCIPLE-fixable reasons: no safeguard ensuring the
first action tries the literal original question, and no working signal for when to
stop exploring. Neither is a indictment of agentic retrieval generally — R9 already
showed the "don't lose the original wording" problem is architecture-agnostic (it hit a
totally different two-role design too), and R16-R18 already showed the "when to stop"
problem is a calibration issue this specific model has everywhere, not something unique
to a single-trace agent. What IS new and load-bearing here: q01's trace is now the
strongest evidence in the whole session that its failure lives in the shared retrieval
tool, not in any calling architecture wrapped around it — no amount of reasoning,
self-correction, or additional attempts, however genuine, escapes it.

---

## R21 — `production_react` v2: the detailed search-strategy + stopping-discipline
## prompt. hit@6 improved (4→5/10) but the stopping behavior got WORSE, not better —
## 0/10 ever called Finish now, versus 2/10 before

**Question.** R20 diagnosed two problems in the ReAct agent: (1) its first action is
already a self-generated paraphrase, with nothing enforcing "try the literal question
first," which reproduced R9's exact paraphrase-destroys-a-good-match failure; (2) severe
stopping miscalibration — only 2/10 questions ever called `Finish[]`. Swapped in a much
more detailed prompt directly targeting both: a 5-step search strategy (preserve original
wording first, inspect before reformulating, reformulate only with a genuinely different
anchor, never repeat failed vocabulary, prefer Lookup once close) and an elaborate
stopping section (a "when to finish" rubric per intent type, an explicit "the goal is NOT
to use all available steps" instruction, and a 5-question final self-check before calling
Finish).

**Result: 5/10 — better than v1's 4/10, still below both `production_method` and
`production_llm_retries` (6/10 each).**

**Finding 1 — the search-discipline instructions partially worked, verified directly.**
q06 (v1's regression) is fixed (hit@1 again). Traced why carefully: the agent's actual
first search was *"narrator convince Nastenka lodger's continued silence grounds
argue"* — closer to the original wording than v1's free paraphrase, but still a
compressed keyword list, not the literal sentence. Checked directly:
`production_retrieve` on the TRUE original question hits gold@1 at k=5; on the agent's
compressed version, it still misses. **The instruction says "preserve wording as closely
as possible," and the model complied with the spirit (kept the distinctive words) but not
the letter (didn't keep the actual sentence)** — a real, specific gap between what was
asked and what was produced, even though the overall session recovered the answer via a
later step regardless.

**Finding 2, and the more important, genuinely surprising one: adding MUCH more explicit
"know when to stop" guidance made the model stop LESS often, not more.** v1 (a single,
vague "call this once you believe you have gathered enough evidence" line) got 2/10
Finish calls. v2 — a dedicated stopping section, per-intent-type criteria, an explicit
"the goal is NOT to use all available steps," and a 5-question mandatory self-check —
got **0/10.** Every single question ran the full 6-step budget regardless. This is a
genuinely counter-intuitive result: more elaboration on exactly the behavior that was
missing made that behavior disappear entirely rather than improve.

**Why this plausibly happened, though not confirmed by a controlled ablation:** the
stopping section adds a 5-item mandatory checklist to run through before every `Finish[]`
call, phrased as skeptical, effort-justifying questions ("is another Search likely to
discover something materially different, or am I merely searching harder?"). A checklist
built entirely from doubt-inducing questions, with no equally weighted counter-example of
confidently finishing, may simply make continuing to search the path of least resistance
every time — the same asymmetric-caution pattern R17 found in the sufficiency-checker
prompt (heavy on warnings against false positives, nothing modeling confident
correctness), now recurring in a third, structurally different prompt.

**Consequence: because 0/10 runs ever reached `Finish[]`, the 4→5 hit@6 improvement is
coming entirely from better INTERMEDIATE search/lookup quality, not from the agent ever
actually deciding it had enough.** The final answer set is always built from
`_diversify()` over everything encountered across the full 6-step budget, regardless of
whether steps 4-6 added anything useful — the "when to finish" half of this prompt
change had, empirically, zero effect on actual stopping behavior even though it visibly
changed the model's stated self-checks in the trace text.

**q01 confirms the token-collision ceiling a second time, independently.** All 6
searches still say "Nastenka" (`"how does the narrator first meet Nastenka"` →
`"first meeting narrator Nastenka"` → ... → `"narrator meets Nastenka first time
description"`) — despite Step 4's explicit worked counter-example naming almost this
exact pattern ("the dreamer's feelings about the girl") as what NOT to keep doing. Same
shape as R17's finding: a prompt containing the exact failure as a labeled bad example
does not reliably prevent the model from producing it anyway.

**Takeaway.** Detailed, well-reasoned prompt engineering measurably helped one thing
(search-anchor discipline, partially) and measurably hurt a different thing in the same
prompt (stopping behavior, completely) — a reminder that prompt sections don't act
independently; adding rigor to one part of a system prompt can change behavior in an
unrelated part, plausibly through sheer competition for the model's attention across a
much longer instruction set (echoing F7's "prompt bloat degrades unrelated fields"
finding from the tagger study, now observed in an agentic control-flow prompt rather
than a classification one). `production_method` and `production_llm_retries` (6/10 each,
both simpler than this agent) remain this session's best. If this line is worth pursuing
further, the concrete next experiment the data points to is separating the two prompt
changes and testing them independently — search-strategy discipline alone, and stopping
discipline alone — since bundled together here, one clearly worked and the other clearly
backfired, and there's no way to tell from this result alone whether stopping got worse
*because of* the new search strategy taking attention away from it, or independently.

---

## R22 — `production_react` v3: the "evidence-driven pivoting" prompt. A net regression
## (5→4/10) that also introduced a new problem — `Lookup` nearly abandoned

**Question.** v2 (R21) improved hit@6 (4→5/10) via search-anchor discipline but left
stopping calibration broken (0/10 `Finish[]` calls, worse than v1). Swapped in a
prompt built around a different core idea: treat every search as an *experiment* whose
results — even off-target ones — reveal what the retriever associates with the query;
the next search should be motivated by a concrete clue extracted from what actually came
back ("evidence-driven pivoting"), not another paraphrase of the abstract question.

**Result: 4/10 — a regression from v2's 5/10, back down to v1's level.** Stopping
calibration did not improve either: **0 of 10 questions called `Finish[]`**, identical
to v2.

**A new problem, not present before: `Lookup` was nearly abandoned.** Across all 10
questions combined, only 3 `Lookup` calls total — 8 of 10 questions used all 6 steps as
pure `Search`, zero `Lookup`. v2, by contrast, used `Lookup` heavily (several questions
had 3-5 of their 6 total actions be `Lookup`). The "pivot to a new search using an
extracted clue" framing, whatever its intent, measurably pulled the model away from the
one action type (`Lookup`) that's cheaper and more targeted than a fresh corpus search —
the opposite of what "prefer Lookup once you have something close" (still present in
this prompt's LOOKUP section) was meant to encourage.

**q06 regressed again, and the mechanism is a clean, direct violation of the prompt's
own explicit warning.** Pulled all 6 of its search queries:
`"narrator convinces Nastenka lodger silence"` → `"narrator persuades Nastenka about
lodger"` → `"narrator discusses lodger silence with Nastenka"` → `"narrator feelings
about lodger silence Nastenka"` → `"narrator talks to Nastenka about lodger silence"` →
`"narrator convinces Nastenka to act on lodger silence"`. **This is pure synonym-level
paraphrasing across all 6 steps** (convinces/persuades/discusses/talks to; silence/about
lodger silence) — not evidence-driven pivoting to a new anchor at all, despite the
prompt containing an explicit, concrete warning against precisely this pattern (the
"dreamer's feelings about the girl" bad example). None of the 6 queries ever attempted
the literal original wording, which (confirmed repeatedly across R20-R22 now) hits
gold@1 on its own.

**This is the third separate instance this session of the same specific failure: a
prompt containing a literal worked example of the exact mistake does not reliably
prevent the model from making that mistake.** R17's evidence checker reproduced its own
"dreamer's feelings about the girl" bad example nearly verbatim; R21's v2 agent still
said "Nastenka" in all 6 of q01's searches despite the same warning; now v3 falls into
synonym-swapping on q06 despite an explicit prohibition of exactly that pattern, stated
in the same prompt. Three independent prompts, three independent failure sites, one
consistent limit.

**Takeaway.** Of the three ReAct prompt versions tried (R20 v1: 4/10, R21 v2: 5/10, R22
v3: 4/10), **v2 remains the best-performing**, and its two real wins (search-anchor
discipline improving hit@6) came from a more mechanical, rule-based instruction set
("preserve original wording," "change the retrieval anchor," concrete do/don't examples)
rather than v3's more conceptual "treat search as experimentation" framing — matching a
pattern from the tagger study (F11: a concrete scanning PROCEDURE outperformed abstract
principles for a mechanical task) now recurring in agentic retrieval. None of the three
ReAct variants has matched `production_method`/`production_llm_retries`'s 6/10 — the
architecture remains net-negative on this gold set regardless of prompt version, even
though R20's q01 trace remains this session's best evidence that the underlying
retrieval tool, not the calling architecture, is the real ceiling on the four
persistent misses.

---

## R23 — `production_react` v4: v2 base + three surgical, mechanical rules. First ReAct
## version to match the session's best score (6/10) — but by converging onto
## production_method's own mechanism, not by adding independent value beyond it

**Question.** R21/R22 established v2 (5/10, search-anchor discipline via mechanical
rules) beat v3 (4/10, more conceptual "evidence-driven pivoting" framing) — matching this
session's broader pattern that concrete procedures outperform abstract principles. Rather
than another full rewrite, three narrow, targeted rules on top of v2's exact text: (1)
the first action MUST be `Search[<question verbatim>]`, no rewriting permitted at all —
stricter than v2's "preserve wording as closely as possible," which still let the model
compress q06's question into a keyword list; (2) `Lookup` gets explicit, mandatory
priority over a fresh `Search` whenever any retrieved passage looks promising; (3) a
recovery `Search`, when genuinely needed, must combine ONE concrete anchor from already-
retrieved passages with the missing concept — not swap synonyms into the previous query.
Stopping logic (`WHEN TO FINISH` etc.) left completely untouched from v2, to isolate what
these three changes alone do.

**Result: 6/10 — the first ReAct version to match `production_method`/
`production_llm_retries`'s best score.** (v1: 4/10, v2: 5/10, v3: 4/10, **v4: 6/10**.)

**Both targeted fixes verified working exactly as designed, directly from the traces.**
q06 (the recurring regression across v1 and v3): first action is now
`Search["What does the narrator convince Nastenka to do about the lodger's continued
silence, and on what grounds does he argue for it?"]` — the literal question, verbatim —
and it hits gold at rank 1 immediately. Search/Lookup ratios shifted sharply toward
Lookup across the whole set (e.g. q02 and q04: 1 Search, 5 Lookups each) — Rule 2's
mandatory priority is visibly steering behavior, not just present in the prompt.

**The honest caveat: checked whether the extra machinery is adding value beyond
`production_method`'s own single-shot mechanism, and it is not — yet.** Every one of
the 6 hits lands at the EXACT SAME rank as `production_method` (q03 rank 5/5, q04 rank
2/2, q05/q06/q07/q09 all rank 1/1). This isn't a coincidence: Rule 1 forces the first
`Search` to literally call `production_retrieve()` on the unmodified original question —
the identical mechanism `production_method` itself runs — so when that first call
already succeeds (which per R4 is most of this gold set), the agent's subsequent
Lookup/Search steps just confirm and preserve that result rather than improving on it.
**v4's win is a genuine, verified architectural fix (the two failure modes from R20-R22
are both closed), but the resulting score comes from successfully converging onto the
baseline, not from demonstrating that iterative, multi-step reasoning finds anything the
single-shot formula couldn't.** q01/q02/q08/q10 — the four persistent misses across
every method this entire session — are unchanged: still 0/4, still the same tool-level
ceiling R20's trace already characterized most directly.

**Stopping calibration remains completely unaddressed, as expected — 0 of 10 questions
called `Finish[]`, identical to v2 and v3.** These three rules were deliberately scoped
to search behavior only; the "when to finish" problem identified in R21/R22 (and, in a
different architecture, R16-R18) is untouched and still open.

**Takeaway.** This is a real success, not a wash: two specific, previously-diagnosed
failure modes (paraphrase-before-first-search, Lookup neglect) are now demonstrably
fixed by three short, mechanical, narrowly-scoped rules — confirming both diagnoses were
correct and confirming (again) that mechanical procedures land more reliably than
conceptual framing for this model on this task. But the resulting agent's real
contribution on this gold set is "safely replicate what a single well-executed
`production_method` call already does," not "exceed it." The next genuine test of
whether ReAct's fine-grained, multi-step approach earns its complexity would need
questions where a single-shot call demonstrably fails but a persistent, evidence-
following agent could plausibly still succeed — which, per R20's q01 trace, is not
guaranteed even with unlimited good-faith effort when the underlying retrieval tool has
no path to the answer at all.

**Summary verdict for R20-R23 as a set.** ReAct added adaptive retrieval, but did not
outperform the simpler pipeline because the agent's search policy was underconstrained.
The experiments suggest that tool-using LLMs benefit more from explicit procedural
rules — what action to take under specific conditions — than from conceptual
instructions about good retrieval behavior. v1's vague guidance underperformed; v3's
conceptual "treat search as experimentation" framing underperformed even v1 on hit@6;
v2's numbered, mechanical steps and v4's three narrow, literal rules ("your first action
MUST be...", "use Lookup before Search whenever...") are what actually moved behavior.
The one dimension no version of this prompt fixed — stopping calibration — is also the
most procedurally *under*-specified one across all four versions: every "when to finish"
section stayed at the level of conceptual guidance ("finish when the evidence is
sufficient") rather than a concrete, checkable rule, which is consistent with the same
lesson holding there too, just never yet tested.

---

## R24 — spot check "is the dreamer stupid? why does he still keep loving the girl?"
## surfaces a real bug: `Lookup`-confirmed evidence never reached the final answer
## context. Fixed, verified, logged

**The spot check.** A compound, two-part interpretive question, run through
`production_react` v4. The answer produced was genuinely good — correctly addressed both
halves ("not stupid, but self-deceived"; loves her despite knowing about the lodger),
well-grounded in real quotes. But inspecting the trace showed the agent's `Lookup`
actions were doing real work invisibly: `Lookup["stupid"]` found *"how could I have been
so stupid?"* and a later `Lookup["feelings"]` found *"was nothing, stupid, simple
nullity, there has been nothing but dreams!"* (`second_night_021`) — about as directly
on-point a piece of textual self-criticism as the question could hope for.

**The bug.** `second_night_021` never made it into the final 6 passages. Traced why:
`react_retrieve()`'s final candidate pool was built from `encountered_ids`, which only
ever gets appended to inside the `Search` branch of the loop. `_do_lookup()` finds and
returns which passage a string was found in, but the calling code discarded that
information — it only used the returned text for the trace, never captured *which chunk*
the successful Lookup pointed at. A `Lookup` that explicitly verifies a passage is
arguably STRONGER evidence than a bare `Search` hit (the agent chose to check that exact
passage for that exact detail), and it was structurally invisible to the part of the
pipeline that decides what the answering step gets to see.

**The fix.** `_do_lookup()` now returns `(observation_text, found_chunk_id)` instead of
just text. `react_retrieve()` tracks a new `confirmed_ids` list, appended to whenever a
`Lookup` succeeds. The final pool is built as `confirmed_ids + (everything else
encountered)` — since `_diversify()` is a greedy pass over its input order, Lookup-
confirmed chunks now compete for their scene's slot ahead of chunks that were only ever
returned by a `Search` and never actually checked.

**Verification.** Re-ran the spot check: `confirmed: 2`, and both Lookup-confirmed
chunks (`fourth_night_008`, `third_night_008` on that particular run — stochastic, so
not byte-identical to the run that first surfaced the bug, but the mechanism is what
matters) landed in the final top-6, in front of merely-encountered ones. Re-ran the full
10-question pilot: **still 6/10, zero regressions** — the fix is safe as well as
correct.

**Why this is worth keeping as a general lesson, not just a one-off patch.** This is a
different category of finding than R20-R23's prompt-engineering rounds — not "the model
didn't follow an instruction well enough," but a genuine implementation bug: a signal the
agent itself generated (a verified relevant passage) was being computed and then thrown
away by the surrounding code before it could matter. No amount of prompt tuning on
`REACT_SYSTEM` could ever have fixed this, because the agent was behaving exactly as
intended (`Lookup` correctly confirming relevant passages) — the loss happened one layer
below, in how the orchestrating Python code used that output. Worth the general reminder
it leaves: when an agent's intermediate actions look right in the trace but the final
result still seems to miss what those actions found, check whether the actions' outputs
are actually being consumed by the parts of the code that build the final result — not
just whether the model's behavior needs more coaxing.
