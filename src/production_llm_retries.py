"""An LLM evidence-sufficiency check gates a second retrieval round.

    ORIGINAL QUERY
      │
      ▼
    pure production retrieval (production_method: BM25 + dense + metadata, cheap_rerank)
      │
      ▼
    initial candidates (top k)
      │
      ▼
    LLM evidence check  →  {sufficient, supported_claims, missing_evidence, search_directions}
      │
      ├── sufficient = true  ─────────────────────────────────► answer(original query)
      │
      └── sufficient = false
              │
              ▼
        search_directions (LLM-generated, aimed at the MISSING evidence specifically —
        not paraphrases of the original question, which would just re-find the same miss)
              │
              ▼
        second retrieval, each direction searched over ALL 84 corpus chunks
              │
              ▼
        UNION with the first round's initial candidates
              │
              ▼
        final ranking: cheap_rerank anchored on the ORIGINAL query (R11's fix, kept —
        never blend/max scores across search directions, only widen the candidate set)
              │
              ▼
        diversify → top k → answer(original query)

This is the adaptive/iterative retrieval pattern from Gao et al.'s RAG survey (§V,
"Augmentation Process in RAG") — Self-RAG/Flare/CRAG all gate further work on a
confidence or sufficiency judgment instead of always doing a fixed amount of retrieval.
Unlike CRAG (which falls back to WEB search on low confidence), there's nowhere else to
go but back into the same closed corpus — so insufficiency triggers a second,
differently-aimed retrieval pass rather than a refusal or an external lookup.

v2 (this version): swapped in an intent-aware evidence bar and a ready-to-run recovery-
query generator, directly fixing R16's two diagnosed problems:
1. R16 found the checker over-triggered on interpretive questions specifically — it
   wanted the book's evaluative VERDICT spelled out in a passage, which is the answering
   step's job, not retrieval's. v2 gives EVENT/LEXICAL_FACT a STRICT bar (the concrete
   fact must be explicit) and INTERPRETIVE/MOTIVATION a MODERATE bar (evidence the
   answer can be reasonably inferred from is enough — no verbatim verdict required).
2. R16 found the recovery queries reproduced the exact overloaded vocabulary ("the
   dreamer", "the girl") that caused the original miss. v2 requires each recovery query
   to be a concrete, ready-to-run string (not an abstract "direction"), explicitly
   forbidden from centering on a generic/overloaded term unless the retrieved evidence
   shows it's actually discriminative, and required to differ from `search_history`
   (everything already tried) in a genuine retrieval-relevant way — a real anchor
   (distinctive action, object, name, location) instead of a synonym-level rewrite.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_method import classify_intent, cheap_rerank, fuse, production_retrieve  # noqa: E402
from retrieval import CHAT_MODEL, _diversify, answer, by_id, chunk_to_scene, client, order  # noqa: E402

EVIDENCE_CHECK_SYSTEM = """You are a retrieval sufficiency checker and recovery-query generator for a literary QA system.
Your task is to determine whether the retrieved passages are sufficient to answer the user's question. If they are insufficient, generate a small number of NEW, READY-TO-RUN SEARCH QUERIES.
The search queries must be designed to recover the missing evidence. Do not output abstract search directions, diagnoses, or instructions for another system. The output queries themselves must contain the vocabulary and clues needed to perform the recovery.
Inputs
You receive:

* `question`: the user's original question
* `intent`: the already-classified question intent
* `retrieved_chunks`: passages retrieved so far
* `search_history`: all queries already attempted
* `retrieval_attempt`: the number of retrieval rounds that have already been attempted
* `max_retrieval_attempts`: the maximum number of retrieval rounds available

Retrieval attempt awareness
Use `retrieval_attempt` and `max_retrieval_attempts` to judge the state of the retrieval process.
The attempt count is NOT evidence about whether the answer is present.
Always judge sufficiency from the retrieved evidence itself.
If the required evidence is present, return `SUFFICIENT` regardless of the attempt count.
If the evidence is insufficient and `retrieval_attempt` is well below `max_retrieval_attempts`, generate a small number of genuinely new recovery queries.
As `retrieval_attempt` approaches `max_retrieval_attempts`, become increasingly selective about generating another recovery query.
Do NOT generate another query merely because the evidence is insufficient.
Generate another query only if there is a plausible, materially different retrieval strategy that has not already been tried, i.e. try a different direction only if previous ones made you unsatisfied.
If previous attempts have repeatedly failed and no meaningful new retrieval anchor is available, return `INSUFFICIENT` with an empty `new_queries` list.
Never repeat or lightly paraphrase previous queries simply to consume another retrieval attempt.
The purpose of additional attempts is to recover missing evidence, not to continue searching indefinitely.

Intent-specific evidence bar
EVENT / LEXICAL_FACT
Use a STRICT evidence standard.
The retrieved passages must explicitly state, describe, or unambiguously establish the requested fact, event, action, object, identity, location, or other concrete detail.
Do not consider evidence sufficient merely because it is topically related or makes the answer seem plausible.
When generating recovery queries, target the concrete textual fact itself.
INTERPRETIVE / MOTIVATION
Use a MODERATE evidence standard.
The retrieved passages do not need to state the final answer verbatim. They are sufficient if they provide strong evidence for the relevant motivation, emotional state, prior event, relationship, conflict, consequence, or causal chain from which the answer can reasonably be inferred.
Do not require the source text to use the same interpretive vocabulary as the question.
When generating recovery queries, search for the underlying evidence that supports the interpretation, not just the interpretation itself.
Sufficiency
Return `SUFFICIENT` when the retrieved evidence is adequate for the question's intent.
Return `INSUFFICIENT` when important evidence is missing, ambiguous, contradictory, or too weak.
If sufficient, generate no new queries.
Recovery-query generation
When evidence is insufficient, generate NEW queries that are ready to send directly to the retrieval system.
Every query must actively attempt to overcome the reason the previous retrieval failed.
CRITICAL RULE: DO NOT PARAPHRASE THE FAILED QUERY
Do not simply rewrite the original question.
Do not produce synonym-level variants of previous queries.
Do not repeatedly search the same generic nouns or overloaded labels that already failed.
If previous searches relied on generic terms such as "the dreamer", "the girl", "the man", "their relationship", "his feelings", etc., do not continue centering new queries on those terms unless the retrieved evidence demonstrates that they are actually discriminative.
The query itself must introduce a meaningful retrieval change.
A good recovery query changes one or more of:

* vocabulary,
* entity representation,
* concrete event or action,
* distinctive object,
* location,
* temporal/narrative context,
* relationship,
* emotional or causal evidence,
* distinctive wording or clue from the retrieved passages.

How to construct recovery queries
If the question asks for an EVENT or LEXICAL_FACT
Search for the concrete thing that must appear in the source.
Prefer combinations such as:
`distinctive person/entity + concrete action + distinctive object/event`
or:
`distinctive event + location/context + participant`
or:
`rare textual clue + surrounding action`
Avoid abstract formulations of the question.
If the question asks for MOTIVATION or INTERPRETATION
Search for the evidence that would establish the answer.
Prefer combinations such as:
`character/entity + relevant action + preceding circumstance`
`character/entity + emotional state + triggering event`
`relationship + concrete interaction + consequence`
`character/entity + distinctive event + stated feeling`
Do not merely search for "why" plus the original question.
Using retrieved evidence
Use clues in `retrieved_chunks` to improve the new queries.
If a retrieved passage contains a distinctive name, object, action, location, unusual phrase, emotional state, or narrative event, prefer that clue over generic terminology from the question.
If the retrieved passages reveal that the question's terminology does not match the book's terminology, deliberately switch to the terminology actually suggested by the evidence.
Search-history awareness
Compare every proposed query against `search_history`.
Do not generate a query that is substantially equivalent to an already attempted query.
The new query must have a genuine retrieval purpose.
Bad:

* "Why does the dreamer trust the girl?"
* "Why does the dreamer believe the girl?"
* "Dreamer's feelings toward the girl"

if those concepts already failed.
Better:

* use a distinctive action, object, location, name, prior event, or unusual textual clue that can lead retrieval to the relevant passage.

Query quality
Queries should be concise enough for retrieval, but contain enough distinctive information to escape the previous failure.
Do not make queries artificially long.
Do not include explanations, labels, or meta-instructions inside the query.
The query must look like something a search engine / BM25 / dense retriever could actually search.
Output
Return exactly:
{
"verdict": "SUFFICIENT" | "INSUFFICIENT",
"reason": "brief explanation",
"new_queries": [
"ready-to-run retrieval query",
"ready-to-run retrieval query"
]
}
If `verdict` is `SUFFICIENT`, `new_queries` must be an empty list.
If `verdict` is `INSUFFICIENT`, generate only a small number of genuinely different, high-value queries.
Do not output search directions separately.
Do not output a failure-mode label separately.
Do not tell another system how to construct the queries.
Construct the recovery strategy directly into the queries themselves.
The fundamental rule is:
The next query must not merely ask the same question differently. It must search for the missing evidence using different, more discriminative textual anchors.

Respond with valid JSON only, in the exact shape given above."""


# example ─ in:  ("Why does the dreamer still love the girl?", "MOTIVATION",
#                 ["white_nights_second_night_014", ...], ["Why does the dreamer still love the girl?"], 1, 3)
#           out: {"verdict": "INSUFFICIENT", "reason": "...",
#                 "new_queries": ["confession scene I love you Nastenka fourth night", ...]}
def check_evidence(query: str, intent: str, cand_ids: list, search_history: list,
                    retrieval_attempt: int, max_retrieval_attempts: int) -> dict:
    # full text, not truncated: chunks are capped at 350 tokens (~1400-1750 chars) by the
    # chunker, and truncating to 600 chars (copied from retrieval.py's rerank(), which
    # only needs a taste for coarse relevance) was cutting off the actual answer inside
    # longer chunks -- confirmed directly on q06, where the key sentence fell after char 600
    listing = "\n\n".join(f"[{i}] {by_id[cid]['text']}" for i, cid in enumerate(cand_ids))
    user = (f"question: {query}\nintent: {intent}\n"
            f"retrieval_attempt: {retrieval_attempt}\nmax_retrieval_attempts: {max_retrieval_attempts}\n"
            f"search_history: {json.dumps(search_history)}\n\n"
            f"retrieved_chunks:\n{listing}")
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": EVIDENCE_CHECK_SYSTEM}, {"role": "user", "content": user}],
    )
    data = json.loads(r.choices[0].message.content)
    data.setdefault("verdict", "SUFFICIENT")
    data.setdefault("reason", "")
    data.setdefault("new_queries", [])
    return data


# example ─ in:  "Why does the dreamer still love the girl?"
#           out: (["white_nights_fourth_night_004", ...],
#                 {"intent": "MOTIVATION", "checks": [{...}, {...}], "attempts": 2,
#                  "retried": True, "union_size": 27})
def production_retries_retrieve(query: str, k: int = 6, per_direction_pool: int = 15,
                                 per_scene: int = 2, max_retrieval_attempts: int = 3):
    """A real bounded loop, not a single retry: each round's evidence check sees how many
    attempts have already run and how many remain, and is told explicitly to stop proposing
    new queries once further attempts are unlikely to find a materially different anchor --
    per the prompt's own "Retrieval attempt awareness" section. `search_history` accumulates
    across ALL rounds (not just the first), so a later round can see what earlier rounds
    already tried, not just what round 1 tried."""
    intent = classify_intent(query)  # ORIGINAL query only, every round
    fused_orig = fuse(query, intent)  # scores ALL 84 chunks — anchors every round's ranking

    initial_ids, _ = production_retrieve(query, k=k, per_scene=per_scene)  # pure, unmodified
    all_ids = list(initial_ids)
    search_history = [query]
    checks = []
    attempt = 1
    current_ids = initial_ids

    while True:
        check = check_evidence(query, intent, current_ids, search_history,
                                retrieval_attempt=attempt, max_retrieval_attempts=max_retrieval_attempts)
        checks.append(check)
        if check["verdict"] == "SUFFICIENT" or not check["new_queries"] or attempt >= max_retrieval_attempts:
            break

        search_history.extend(check["new_queries"])
        round_ids = []
        for nq in check["new_queries"]:
            fd = fuse(nq, intent)
            round_ids.extend(sorted(order, key=lambda c: fd[c], reverse=True)[:per_direction_pool])

        all_ids = list(dict.fromkeys(all_ids + round_ids))
        reranked = cheap_rerank(query, all_ids, fused_orig)  # anchored on ORIGINAL query (R11)
        current_ids = reranked[:k]  # what the NEXT check evaluates
        attempt += 1

    final_reranked = cheap_rerank(query, all_ids, fused_orig)
    final_ids = _diversify(final_reranked, k, per_scene)
    return final_ids, {"intent": intent, "checks": checks, "attempts": attempt,
                        "retried": attempt > 1, "union_size": len(all_ids)}


def production_retries_answer(query: str, k: int = 6):
    ids, debug = production_retries_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Why does the dreamer still love the girl?"
    ids, debug = production_retries_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']} attempts={debug['attempts']} "
          f"retried={debug['retried']} union_size={debug['union_size']}\n")
    for i, check in enumerate(debug["checks"], start=1):
        print(f"CHECK (round {i}):")
        print(json.dumps(check, indent=2))
        print()
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
