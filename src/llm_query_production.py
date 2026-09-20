"""Query rewriting in front of the production pipeline — v2.

    original_query
      |
      +--> classify_intent(original_query)  --------------------------+  (intent, from
      |                                                                |   ORIGINAL grammar)
      +--> rewrite_query(original_query)  -> {rewritten_query,        |
      |                                        reference_variants}    |
      v                                                                v
    llm_query_retrieve(query=rewritten_query, intent=..., reference_variants=...)
      -> fuse() [production_method, but with the EXTERNALLY supplied intent]
      -> cheap_rerank -> diversify -> top k -> answer(ORIGINAL query, ids)

Three bugs found and diagnosed this session, in order (see experiments/findings_retrieval.md
R8 for the full trace with numbers):
1. FIXED — the rewrite turned "first MEET" (verb) into "first MEETING" (noun), which
   flipped production_method's own classify_intent() from EVENT to TEMPORAL. Fixed by
   classifying intent from the ORIGINAL query and passing it in explicitly (`run()` below
   never lets llm_query_retrieve() reclassify the rewrite's own, different grammar).
2. FIXED — a flat `reference_variants` output field made the LLM reliably SKIP proposing
   an alternative for a proper name ("Nastenka") while still varying a generic role word
   ("narrator" -> "protagonist"). Direct A/B test (4 runs each) confirmed this exactly:
   asking instead for a per-entity nested `entities: [{query_reference,
   retrieval_references}]` structure — forcing the model to address each entity in its
   own object — reliably produces "the girl" for Nastenka (3/4, then 3/3 after further
   prompt cleanup). `reference_variants` is now just a flattening of that nested output.
3. STILL OPEN — even with "the girl" correctly proposed and folded into the query as a
   parenthetical suffix, the gold chunk's rank barely moves (50/84 -> 47/84). The name
   "Nastenka" is still repeated 3x in `rewritten_query`'s main text, which dominates the
   embedding far more than one appended alternative can offset. Q1 is still a MISS.
   Candidate next steps (not yet tried): bias the rewriter to LEAD with the descriptive
   alternative instead of appending it alongside the name, or fan out into multiple full
   query variants (one per name/alternative swap) and union their retrieval results,
   rather than blending everything into one string.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_method import classify_intent, cheap_rerank, fuse  # noqa: E402
from retrieval import CHAT_MODEL, _diversify, answer, chunk_to_scene, client, order  # noqa: E402

QUERY_REWRITE_SYSTEM = """You are a query rewriting component in a retrieval system.
Your job is to rewrite a user's information-seeking query into a retrieval-optimized query that is more likely to match relevant passages in an arbitrary text corpus.
The corpus may be a novel, book, transcript, article collection, documentation, conversation, legal text, research corpus, or any other text. You must remain completely agnostic about the corpus, its subject matter, vocabulary, structure, and metadata.

Core objective
Transform the user's query from "how the user expresses the information need" into "how the relevant information might be expressed in the source text."
Preserve the user's intended meaning, entities, relationships, events, attributes, and constraints.
Make implicit retrieval concepts explicit when doing so improves recall.
Do NOT answer the query. Do NOT determine which passage is correct. Do NOT invent facts about the corpus.

What to improve
When appropriate, make explicit: synonymous or paraphrased expressions of important concepts; alternative ways an event or action may be described; descriptive references that may be used instead of a named entity; implicit temporal constraints (first, later, before, after, during, again, last); implicit relational constraints (who did what to whom); scene/event/state/topic descriptions implied by the question; alternative grammatical formulations likely to occur in source text.
Prefer concepts and expressions that could plausibly occur in the source text. Do not keyword-stuff; do not generate an exhaustive synonym list.

Entity handling
A query may refer to an entity using a name, nickname, title, role, pronoun, description, or other reference.
When the query itself provides enough information to identify equivalent references, include useful alternatives.
Do not assume that the source text uses the same name or terminology as the user.
For example, a named entity in the query may be referred to in the source as a description, title, role, pronoun, or other expression.
However, NEVER invent an alias or identity merely because it is plausible.
If an entity relationship is uncertain, preserve the uncertainty rather than asserting it as fact.

Temporal and relational constraints
Preserve constraints such as first/earliest, last/latest, before/after, during, eventually, again, subsequent, simultaneous, initial, final. Preserve who is acting, who is affected, and the relationship between entities. Do not remove these constraints merely to increase lexical recall.

Question preservation
The rewritten query must remain a query for the SAME information need. Do not turn a "when" into a "whether", a "who" into a "what", a "why" into a "what happened", a "where" into a "when", a specific relationship into a general topic, or a specific event into a broad thematic query. Do not broaden the query so far that irrelevant passages become more likely than relevant ones.

Output
Return valid JSON with EXACTLY these fields:
{
"rewritten_query": "a concise retrieval-optimized string — may pack multiple plausible phrasings of the SAME event/concept (e.g. 'first meet between the narrator and the girl; initial encounter; first interaction') rather than one natural-language sentence, since this string is embedded and keyword-matched directly, not read as prose",
"entities": [{"query_reference": "the expression the user used", "retrieval_references": ["alternative 1", "alternative 2", ...]}]
}
entities: for EACH important entity in the query, list it separately with ITS OWN alternative references (descriptions, roles, pronouns) that are justified by the query itself or straightforward linguistic equivalence — NOT invented aliases. Do this per-entity, one object per entity — do not merge all entities' alternatives into a single undifferentiated list. Empty "retrieval_references" list if no useful alternatives exist for that specific entity. Include an entity even if it has no alternatives, so the caller knows it was considered.

Final rules
1. Do not answer the user's question. 2. Do not invent corpus facts. 3. Do not assume knowledge of the corpus. 4. Do not assume a particular genre. 5. Do not assume the corpus uses the user's terminology. 6. Preserve entities and semantic relationships. 7. Preserve temporal and other constraints. 8. Resolve aliases only when justified. 9. Prefer concise semantic expansion over synonym stuffing. 10. The rewritten query must represent the same information need as the original query. 11. When uncertain, preserve the user's wording rather than hallucinating an interpretation. 12. Output JSON only."""


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: {"rewritten_query": "first meet between the narrator and the girl; initial
#                 encounter; first interaction", "reference_variants": ["the girl", "her",
#                 "unknown girl", "young woman"], "entities": [{...}, {...}]}
def rewrite_query(query: str) -> dict:
    """Asks the LLM for a PER-ENTITY nested `entities` structure (query_reference ->
    its own retrieval_references) -- confirmed by direct A/B test to reliably elicit
    alternatives like "the girl" for a named character, where a single flat list field
    does not (the model consistently skips proper names in a flat list, but not when
    forced to address each entity in its own object). Flattened into `reference_variants`
    here in Python so callers get the simple 2-field shape without losing that reliability."""
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": QUERY_REWRITE_SYSTEM}, {"role": "user", "content": query}],
    )
    data = json.loads(r.choices[0].message.content)
    data.setdefault("rewritten_query", query)
    entities = data.setdefault("entities", [])
    variants = []
    for e in entities:
        variants.extend(v for v in e.get("retrieval_references", []) if v not in variants)
    data["reference_variants"] = variants
    return data


# example ─ in:  (query="first meet between the narrator and the girl; initial encounter",
#                 intent="EVENT", reference_variants=["the girl", "her"])
#           out: ["white_nights_first_night_009", ...]
def llm_query_retrieve(query: str, intent: str, reference_variants: list = None,
                        k: int = 6, pool: int = 20, per_scene: int = 2) -> list:
    """`intent` is supplied externally (classified from the ORIGINAL query) rather than
    reclassified here, and `reference_variants` are folded into the text actually scored
    -- fixing both bugs v1 hit. Reimplements production_method.production_retrieve()'s
    body rather than calling it, specifically so intent can be passed in instead of
    recomputed from (rewritten, grammatically different) text."""
    variants = reference_variants or []
    expanded = query + (f" (also referred to as: {', '.join(variants)})" if variants else "")

    fused = fuse(expanded, intent)
    pool_ids = sorted(order, key=lambda c: fused[c], reverse=True)[:pool]
    reranked = cheap_rerank(expanded, pool_ids, fused)
    return _diversify(reranked, k, per_scene)


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"original_intent": "EVENT", "rewrite": {...}})
def run(original_query: str, k: int = 6):
    original_intent = classify_intent(original_query)
    rewrite = rewrite_query(original_query)
    ids = llm_query_retrieve(
        query=rewrite["rewritten_query"],
        intent=original_intent,
        reference_variants=rewrite["reference_variants"],
        k=k,
    )
    return ids, {"original_intent": original_intent, "rewrite": rewrite}


def llm_query_answer(original_query: str, k: int = 6):
    ids, debug = run(original_query, k=k)
    return answer(original_query, ids), debug  # answer the ORIGINAL question; rewrite is retrieval-only


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = run(query)
    print(f"ORIGINAL: {query}")
    print(f"original_intent (from ORIGINAL query): {debug['original_intent']}\n")
    print("REWRITE:")
    print(json.dumps(debug["rewrite"], indent=2))
    print()
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
