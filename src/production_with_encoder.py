"""production_method's exact candidate generation (single query: classify_intent -> fuse
over BM25+dense+metadata), but reranked by a CROSS-ENCODER instead of cheap_rerank's
formula.

    query -> classify_intent -> fuse (BM25+dense+metadata, intent-weighted)
          -> top pool
          -> CROSS-ENCODER rerank (query, chunk) pairs, jointly encoded
          -> diversify -> top k

Re-tests a specific prior verdict: retrieval.py's own docstring says "the cross-encoder
was evaluated and dropped — it scored surface relevance, not question intent, on this
literary domain," and requirements.txt confirms sentence-transformers/torch were removed
for the same reason. That verdict was reached on V1's single-query hybrid pool. This file
re-tests the SAME model on production_method's intent-weighted fusion pool instead, to
see whether a better candidate pool changes the outcome, or whether the cross-encoder's
surface-relevance bias holds regardless of what pool feeds it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import CrossEncoder  # noqa: E402

from production_method import classify_intent, fuse  # noqa: E402
from retrieval import _diversify, answer, by_id, chunk_to_scene, order  # noqa: E402

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


# example ─ in:  ("How does the narrator first meet Nastenka?", ["white_nights_first_night_009", ...])
#           out: the same ids, re-ordered by cross-encoder relevance score (highest first)
def cross_encoder_rerank(query: str, cand_ids: list) -> list:
    model = _get_model()
    pairs = [(query, by_id[cid]["text"][:800]) for cid in cand_ids]
    scores = model.predict(pairs)
    return [cid for cid, _ in sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)]


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...], {"intent": "EVENT", "pool_size": 20})
def production_encoder_retrieve(query: str, k: int = 6, pool: int = 20, per_scene: int = 2):
    intent = classify_intent(query)
    fused = fuse(query, intent)
    pool_ids = sorted(order, key=lambda c: fused[c], reverse=True)[:pool]

    reranked = cross_encoder_rerank(query, pool_ids)
    ids = _diversify(reranked, k, per_scene)
    return ids, {"intent": intent, "pool_size": len(pool_ids)}


def production_encoder_answer(query: str, k: int = 6):
    ids, debug = production_encoder_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = production_encoder_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']} pool_size={debug['pool_size']}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
