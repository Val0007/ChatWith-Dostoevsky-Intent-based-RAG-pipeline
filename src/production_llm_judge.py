"""Ablation of production_method.py: same 3 candidate sources (BM25 / dense / metadata),
but BOTH the weighted fusion (fuse()) and the cheap deterministic reranker
(cheap_rerank()) are removed. Each source's top candidates are gathered independently and
UNIONed — no weighting, no blending — then handed directly to an LLM judge (the same
rerank() from retrieval.py that R1-C/R2/R3 found repeatedly demotes correct evidence on
interpretive/evaluative questions, when reranking a WEIGHTED-fusion pool). This tests
whether that finding holds for a different, wider, unweighted candidate pool, or was
specific to what was being reranked before.

    QUERY
      |
      +--- BM25 (top N) -----+
      +--- Dense (top N) ----+---> UNION (no fusion, no weighting)
      +--- Metadata (top N) -+
      |
      v
   LLM judge (rerank(), retrieval.py)      <- replaces BOTH fuse() and cheap_rerank()
      |
      v
   diversify -> top k
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import _bm25, _diversify, _metadata_boost, _tok, answer  # noqa: E402
from retrieval import chunk_to_scene, collection, embed, order, rerank  # noqa: E402


def _dense_candidates(query: str, pool: int) -> list[str]:
    qv = embed(query)
    return list(collection.query(query_embeddings=[qv], n_results=pool)["ids"][0])


def _bm25_candidates(query: str, pool: int) -> list[str]:
    scores = _bm25.get_scores(_tok(query))
    return [order[i] for i in sorted(range(len(order)), key=lambda i: scores[i], reverse=True)[:pool]]


# example ─ in:  ("What does Nastenka want?", 15)
#           out: top 15 chunk ids by tag-overlap alone -- no fusion target to boost, so
#                this ranks the WHOLE corpus directly instead of reordering a candidate list
def _metadata_candidates(query: str, pool: int) -> list[str]:
    return _metadata_boost(query, list(order))[:pool]


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"bm25_n": 15, "dense_n": 15, "metadata_n": 15, "union_size": 34})
def production_judge_retrieve(query: str, k: int = 6, pool: int = 15, per_scene: int = 2):
    bm25_ids = _bm25_candidates(query, pool)
    dense_ids = _dense_candidates(query, pool)
    meta_ids = _metadata_candidates(query, pool)
    union_ids = list(dict.fromkeys(bm25_ids + dense_ids + meta_ids))  # de-dupe, keep first-seen order

    judged = rerank(query, union_ids)
    ids = _diversify(judged, k, per_scene)
    return ids, {"bm25_n": len(bm25_ids), "dense_n": len(dense_ids), "metadata_n": len(meta_ids),
                 "union_size": len(union_ids)}


def production_judge_answer(query: str, k: int = 6):
    ids, debug = production_judge_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = production_judge_retrieve(query)
    print(f"QUESTION: {query}\n{debug}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
