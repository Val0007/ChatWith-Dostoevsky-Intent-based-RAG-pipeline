"""production_union_with_llm's exact candidate generation (original query + up to 4 LLM
alternatives, each contributing its own top-N candidate IDS, unioned) — but the final
rerank is a CROSS-ENCODER instead of cheap_rerank's formula. Ranking is still computed
against the ORIGINAL query only (R11's fix, kept: never blend/max across variants —
variants only ever widen the candidate SET, never distort the ranking of what's in it).

    original query
      │
      ├──→ classify_intent(original)         (ORIGINAL grammar only)
      ├──→ generate_alternatives(original)    (LLM, up to 4)
      ▼
    all_queries = [original] + alternatives
      │
      ▼
    each variant's own top-N via fuse(variant, intent)   ← candidate IDs only, no score
      │
      ▼
    UNION IDs
      │
      ▼
    CROSS-ENCODER reranks (ORIGINAL query, chunk) pairs over the union
      │
      ▼
    diversify → top k

Companion to src/production_with_encoder.py — same cross-encoder, different candidate
source, so the two together isolate: does the cross-encoder help at all (encoder vs
plain production_method), and does a wider multi-variant pool help the cross-encoder
specifically (encoder vs encoder+variants)?
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_method import classify_intent, fuse  # noqa: E402
from production_union_with_llm import generate_alternatives  # noqa: E402
from production_with_encoder import cross_encoder_rerank  # noqa: E402
from retrieval import _diversify, answer, chunk_to_scene, order  # noqa: E402


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"intent": "EVENT", "alternatives": [...], "union_size": 23})
def production_llm_variants_encoder_retrieve(query: str, k: int = 6, per_variant_pool: int = 15,
                                              per_scene: int = 2):
    intent = classify_intent(query)  # ORIGINAL query only
    alternatives = generate_alternatives(query)
    all_queries = [query] + alternatives

    per_variant_ids = []
    for v in all_queries:
        fv = fuse(v, intent)
        per_variant_ids.append(sorted(order, key=lambda c: fv[c], reverse=True)[:per_variant_pool])
    union_ids = list(dict.fromkeys(cid for ids in per_variant_ids for cid in ids))

    reranked = cross_encoder_rerank(query, union_ids)  # ORIGINAL query only, per R11
    ids = _diversify(reranked, k, per_scene)
    return ids, {"intent": intent, "alternatives": alternatives, "n_variants": len(all_queries),
                 "union_size": len(union_ids)}


def production_llm_variants_encoder_answer(query: str, k: int = 6):
    ids, debug = production_llm_variants_encoder_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = production_llm_variants_encoder_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']} union_size={debug['union_size']}")
    print("alternatives:")
    for a in debug["alternatives"]:
        print(f"  - {a}")
    print()
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
