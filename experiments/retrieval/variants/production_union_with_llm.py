"""production_union_with_llm (v3): ask an LLM for up to 4 ALTERNATIVE queries —
additions, never a replacement — alongside the original. Each variant contributes only
CANDIDATE IDS from its own top-N (never a score); those id sets are unioned; final
ranking is then computed FRESH from the ORIGINAL query alone, over just that union.

    original query
         │
      regex intent               ← classify_intent(original) ONLY (R8's fix, kept:
         │                          never reclassify a variant's different grammar)
         ▼
    LLM generates 4 variants      ← generate_alternatives(), "additional attempts,
         │                          not replacements" per the exact prompt used
         │
    ┌────┼────┬────┬────┐
    ▼    ▼    ▼    ▼    ▼
 retrieve retrieve ...  (each variant's OWN top-N via fuse(variant, intent) — a
 original variant1       candidate LIST, not a score carried forward)
    │    │    │    │    │
    └────┼────┴────┴────┘
         ▼
    UNION IDs                    ← dedup; track convergence (how many variants'
         │                          own top-N included each id) as a diagnostic
         ▼
   production pipeline           ← re-fuse(ORIGINAL query, intent) over just the
   fuse / rerank                    union — NOT a max/blend across variants
         │
     diversify
         │
       top-k
         │
   answer(original)

v2 (previous version) scored every candidate by its MAX fused score across all 5
variants — this fixed R9's regression (q06) but introduced a NEW one (q03, R10): `max`
gives every OTHER candidate 5 chances to score well too, so a chunk that was decisively
correct under the original phrasing alone (rank 5/84) got outranked by chunks that were
merely decent under several alternate phrasings (rank 20/84 combined), even though the
gold chunk's own combined score went up. v3 fixes this by keeping variants' contribution
to SET MEMBERSHIP only (recall) and anchoring all RANKING to the original query alone —
so a chunk already winning on the original query's own terms can't be diluted by
competitors' scores on OTHER queries, while chunks the original query missed entirely
can still enter via an alternate phrasing's own top-N.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from production_method import classify_intent, cheap_rerank, fuse  # noqa: E402
from retrieval import CHAT_MODEL, _diversify, answer, chunk_to_scene, client, order  # noqa: E402

ALT_QUERY_SYSTEM = """Generate exactly 4 alternative retrieval queries.
Each query must have a different purpose:

1. ORIGINAL:
Preserve the user's wording exactly.
2. SOURCE-LEXICON:
Identify concrete terms already present in the user's query and,
where possible, make them more retrieval-friendly without adding
any information.
IMPORTANT:
Assume you know NOTHING about the underlying source, document,
characters, entities, events, terminology, or wording.
Do not use pretrained knowledge, world knowledge, literary knowledge,
or assumptions about what the source probably says.
You may only:
   * preserve words already present in the query;
   * make purely linguistic transformations of those words;
   * expand a term only when the expansion is directly implied by the
wording of the query itself.
You MUST NOT:
   * guess source-specific terminology;
   * invent synonyms because you think the source might use them;
   * infer names, entities, events, objects, or relationships not stated
in the query;
   * substitute a term with a more "likely" source-text term based on
outside knowledge.
If there is no safe lexical transformation, keep the original
terminology rather than guessing.
3. REFERENCE-ROBUST:
Handle potentially ambiguous references by generating alternative
surface forms that are directly supported by the query itself.
Do not introduce an entity, identity, relationship, or description
that is not supported by the query.
When a reference is genuinely ambiguous, prefer alternatives over
committing to a single interpretation.
4. SEMANTIC-EVENT:
Express the underlying action, event, state, or relationship in
retrieval-friendly language while preserving all concrete information
contained in the original query.

GLOBAL RULES:

* Treat the user query as the ONLY source of truth.
* Assume you have no knowledge of the underlying document.
* Assume you have no knowledge of the expected answer.
* Do not use pretrained/world knowledge to fill gaps in the query.
* Do not infer facts merely because they are plausible.
* Do not invent entities, names, events, relationships, terminology,
or source-text wording.
* Preserve distinctive concrete nouns from the original query.
* Never remove a distinctive term merely to make the query sound more
natural.
* Do not replace a concrete term with a guessed synonym unless the
replacement is a purely linguistic transformation supported by the
query itself.
* The variants must be genuinely different retrieval formulations,
not four cosmetic paraphrases.
* However, "different" must never come at the cost of introducing
unsupported information.
* When no safe transformation is available, retain the original wording.
* The goal is to improve retrieval recall, NOT to answer the question
or reconstruct the underlying source."""

_ALT_LABEL_RE = re.compile(r"\d+\.\s*[A-Z][A-Z\-]*:\s*")


# example ─ in:  "1. ORIGINAL:\nHow does X meet Y?\n\n2. SOURCE-LEXICON:\nHow does X encounter Y?\n..."
#           out: ["How does X meet Y?", "How does X encounter Y?", ...]   (labels stripped)
def _parse_alternatives(text: str, cap: int = 4) -> list[str]:
    parts = [p.strip() for p in _ALT_LABEL_RE.split(text) if p.strip()]
    return parts[:cap]


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: ["How does the narrator initially encounter Nastenka?",     (SOURCE-LEXICON)
#                 "How does the narrator first come into contact with Nastenka?",  (REFERENCE-ROBUST)
#                 "What is the first meeting between the narrator and Nastenka like?"]  (SEMANTIC-EVENT)
def generate_alternatives(query: str, cap: int = 4) -> list[str]:
    """This prompt's variant #1 (ORIGINAL) is, by design, the input query restated
    verbatim — dropped here since `all_queries = [query] + alternatives` already puts
    the true original first; keeping both would waste an API call re-fusing near-
    identical text."""
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": ALT_QUERY_SYSTEM},
                  {"role": "user", "content": f"Original query:\n{query}"}],
    )
    parsed = _parse_alternatives(r.choices[0].message.content, cap + 1)  # +1 to survive the drop
    parsed = [p for p in parsed if p.strip().lower() != query.strip().lower()]
    return parsed[:cap]


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"intent": "EVENT", "alternatives": [...], "n_variants": 5,
#                  "convergence": {"white_nights_first_night_009": 2, ...}})
def production_union_llm_retrieve(query: str, k: int = 6, per_variant_pool: int = 15,
                                   final_pool: int = 20, per_scene: int = 2):
    """v3 — fixes R10's dilution bug (experiments/retrieval/FINDINGS.md): R10 scored every candidate
    by its MAX fused score across all 5 query variants, which gave every competitor 5
    chances to score well and let generically-relevant chunks leapfrog past one that was
    decisively correct under the ORIGINAL phrasing alone (q03: rank 5 solo -> rank 20
    combined). Fix: each variant only ever contributes CANDIDATE IDS (binary — "did this
    phrasing's own top-N surface this chunk at all"), never a score. Final ranking is
    computed FRESH from the ORIGINAL query alone, over just the union — so a chunk that
    was already winning under the original query keeps winning on the original query's
    own terms; the union only WIDENS the candidate pool for chunks the original query's
    own top-`final_pool` might have missed but an alternate phrasing's own top-N caught."""
    intent = classify_intent(query)  # ORIGINAL query only — never reclassified on a variant
    alternatives = generate_alternatives(query)
    all_queries = [query] + alternatives

    # ---- retrieve: each variant's OWN top-N, independently ----
    per_variant_ids = []
    for v in all_queries:
        fv = fuse(v, intent)
        per_variant_ids.append(sorted(order, key=lambda c: fv[c], reverse=True)[:per_variant_pool])

    # ---- UNION IDs (dedup, keep first-seen order); track convergence for diagnostics ----
    union_ids = list(dict.fromkeys(cid for ids in per_variant_ids for cid in ids))
    convergence = {cid: sum(1 for ids in per_variant_ids if cid in ids) for cid in union_ids}

    # ---- production pipeline: fuse / rerank / diversify, anchored on the ORIGINAL query ----
    original_fused = fuse(query, intent)  # re-fuse with the original query only
    pool_ids = sorted(union_ids, key=lambda c: original_fused[c], reverse=True)[:final_pool]
    reranked = cheap_rerank(query, pool_ids, original_fused)
    ids = _diversify(reranked, k, per_scene)

    return ids, {"intent": intent, "alternatives": alternatives, "n_variants": len(all_queries),
                 "union_size": len(union_ids), "convergence": {cid: convergence[cid] for cid in ids}}


def production_union_llm_answer(query: str, k: int = 6):
    ids, debug = production_union_llm_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = production_union_llm_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']}\nalternatives ({len(debug['alternatives'])}):")
    for a in debug["alternatives"]:
        print(f"  - {a}")
    print(f"union_size={debug['union_size']} (of {debug['n_variants']} variants x their own top-N)\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}] found by {debug['convergence'][cid]}/{debug['n_variants']} variants")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
