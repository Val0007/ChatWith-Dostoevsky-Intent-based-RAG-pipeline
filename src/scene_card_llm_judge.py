"""Ablation of scene_card_production.py: scenes are still found via BM25 + dense over
scene-card summaries, but the WITHIN-SCENE fusion (local BM25 + global BM25 + dense,
scene_card_production.fuse_within_scenes) and the cheap deterministic reranker are both
removed. BM25-over-cards and dense-over-cards each contribute their own top-N scenes,
UNIONed (no weighting); every chunk inside those scenes becomes a judge candidate
directly — no formula ranks them first.

    QUERY
      |
      +--- BM25 over scene cards (top N scenes) --+
      +--- Dense over scene cards (top N scenes) --+---> UNION of scenes (no fusion)
      |
      v
   chunks within those scenes   (all candidates, unranked)
      |
      v
   LLM judge (rerank(), retrieval.py)
      |
      v
   diversify -> top k
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import _diversify, _tok, answer, chunk_to_scene, rerank  # noqa: E402
from scene_card_production import (_card_bm25, _chunks_in, _cosine,  # noqa: E402
                                    _CARD_EMBEDDINGS, _scene_order, embed)


def _bm25_scenes(query: str, n: int) -> list[str]:
    scores = _card_bm25.get_scores(_tok(query))
    return [_scene_order[i] for i in
            sorted(range(len(_scene_order)), key=lambda i: scores[i], reverse=True)[:n]]


def _dense_scenes(query: str, n: int) -> list[str]:
    qv = embed(query)
    return sorted(_scene_order, key=lambda sid: _cosine(qv, _CARD_EMBEDDINGS[sid]), reverse=True)[:n]


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"bm25_scenes": [...], "dense_scenes": [...], "candidate_count": 27})
def scene_card_judge_retrieve(query: str, k: int = 6, n_scenes: int = 8, per_scene: int = 2):
    bm25_scenes = _bm25_scenes(query, n_scenes)
    dense_scenes = _dense_scenes(query, n_scenes)
    union_scenes = list(dict.fromkeys(bm25_scenes + dense_scenes))
    candidate_ids = _chunks_in(union_scenes)

    judged = rerank(query, candidate_ids)
    ids = _diversify(judged, k, per_scene)
    return ids, {"bm25_scenes": bm25_scenes, "dense_scenes": dense_scenes,
                 "union_scenes": union_scenes, "candidate_count": len(candidate_ids)}


def scene_card_judge_answer(query: str, k: int = 6):
    ids, debug = scene_card_judge_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = scene_card_judge_retrieve(query)
    print(f"QUESTION: {query}\nunion_scenes={debug['union_scenes']} "
          f"candidates={debug['candidate_count']}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
