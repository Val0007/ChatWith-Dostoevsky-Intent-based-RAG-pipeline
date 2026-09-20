"""production_method's exact candidate generation (single query: classify_intent -> fuse
over BM25+dense+metadata), but the final rerank is PURE embedding dot-product similarity
against the original query — no formula, no keyword-overlap bonus — mirroring what the
original RAG paper (Lewis et al. 2020) actually does: DPR's bi-encoder retrieval score
IS the final ranking, nothing layered on top.

    query -> classify_intent -> fuse (BM25+dense+metadata, intent-weighted) -> top pool
          -> DOT-PRODUCT rerank: embed query, embed each candidate (already stored in
             chroma), rank purely by similarity
          -> diversify -> top k

Companion ablation to experiments/retrieval/variants/production_with_encoder.py (R13, cross-encoder) and
src/production_llm_variants.py -- same "swap the final reranker, keep candidate
generation fixed" design, this time testing the simplest possible reranker rather than a
more complex one. Key caveat this file exists to test empirically: our embeddings are
off-the-shelf (text-embedding-3-small), never fine-tuned on this corpus the way DPR's
were on QA data -- so "dot product" here measures generic semantic similarity, not a
trained relevance signal. Also note this signal is NOT new information: it's the same
"dense" component already blended inside fuse() one layer up; this only tests whether it
should be the FINAL word instead of cheap_rerank's formula.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from production_method import classify_intent, fuse  # noqa: E402
from retrieval import _diversify, answer, chunk_to_scene, collection, embed, order  # noqa: E402


# example ─ in:  ("How does the narrator first meet Nastenka?", ["white_nights_first_night_009", ...])
#           out: the same ids, re-ordered by pure embedding similarity to the query (highest first)
def dot_product_rerank(query: str, cand_ids: list) -> list:
    qv = embed(query)
    res = collection.query(query_embeddings=[qv], n_results=len(order), include=["distances"])
    dist_by_id = dict(zip(res["ids"][0], res["distances"][0]))  # lower distance = higher similarity
    return sorted(cand_ids, key=lambda cid: dist_by_id[cid])


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_second_night_016", ...], {"intent": "EVENT", "pool_size": 20})
def production_dotproduct_retrieve(query: str, k: int = 6, pool: int = 20, per_scene: int = 2):
    intent = classify_intent(query)
    fused = fuse(query, intent)
    pool_ids = sorted(order, key=lambda c: fused[c], reverse=True)[:pool]

    reranked = dot_product_rerank(query, pool_ids)
    ids = _diversify(reranked, k, per_scene)
    return ids, {"intent": intent, "pool_size": len(pool_ids)}


def production_dotproduct_answer(query: str, k: int = 6):
    ids, debug = production_dotproduct_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = production_dotproduct_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']} pool_size={debug['pool_size']}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
