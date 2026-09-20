"""Final experiment: UNION production_method's direct fusion candidates with
scene_card_production's (weighted multi-field) scene-search candidates, dedupe, then
cheap-rerank -> top 6.

    QUERY
      |
   BM25 + Dense + Metadata  (production_method.fuse(), intent-weighted)     +  scene-card search
      |                                                                          (5 weighted fields)
   existing candidates                                                       additional candidates
      |__________________________________________________________________________|
                                          |
                                        UNION
                                          |
                                     cheap rerank
                                          |
                                        top 6

Rationale: R4 (production_method) showed direct fusion is strong overall (6/10) but has
a real ceiling on paraphrase-gap questions (q01, q02, q10 — weak on BM25, dense, AND
metadata simultaneously for the actual answer chunk). R5 (scene_card_production) showed
scene-card-first RETRIEVAL, used to REPLACE direct search, is worse overall (5/10) —
scene-level matching has its own vocabulary gaps, and the coarse step can bottleneck even
when it finds the right scene. This experiment tests union instead of replacement: let
scene-card search ADD candidates direct fusion might be missing, without removing
anything direct fusion already found — a scene-card miss here costs nothing, since
existing candidates are untouched.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_method import cheap_rerank, classify_intent, fuse  # noqa: E402
from retrieval import _diversify, answer, chunk_to_scene, order  # noqa: E402
from scene_card_production import (DEFAULT_FIELD_WEIGHTS, _chunks_in,  # noqa: E402
                                    scene_scores_weighted)

SCENE_FIELD_WEIGHTS = dict(DEFAULT_FIELD_WEIGHTS)  # exposed here so tuning rounds edit ONE place


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"intent": "EVENT", "top_scenes": ["scene_03", ...], "union_size": 26, ...})
def union_retrieve(query: str, k: int = 6, pool: int = 20, n_scenes: int = 10, per_scene: int = 2,
                    scene_bonus_weight: float = 0.3):
    intent = classify_intent(query)
    fused = fuse(query, intent)  # scores ALL 84 chunks
    existing = sorted(order, key=lambda c: fused[c], reverse=True)[:pool]

    scene_scores = scene_scores_weighted(query, field_weights=SCENE_FIELD_WEIGHTS)  # ALL 27 scenes
    scenes = sorted(scene_scores, key=scene_scores.get, reverse=True)[:n_scenes]
    additional = _chunks_in(scenes)

    union_ids = list(dict.fromkeys(existing + additional))  # de-dupe, keep first-seen order

    # A chunk that only entered via the scene path but gets scored purely by `fused` (the same
    # direct BM25+dense+metadata score that already failed to surface it) gets no real credit
    # for having a strongly-matching scene -- verified empirically: round 0 added such chunks
    # to the pool and they STILL didn't survive reranking, because `fused` alone still ranked
    # them near the bottom. Blend in the chunk's own scene's relevance so scene-sourced evidence
    # can actually compete.
    blended = {cid: fused[cid] + scene_bonus_weight * scene_scores.get(chunk_to_scene.get(cid), 0.0)
               for cid in union_ids}

    reranked = cheap_rerank(query, union_ids, blended)
    ids = _diversify(reranked, k, per_scene)
    return ids, {
        "intent": intent, "top_scenes": scenes,
        "existing_pool_size": len(existing), "additional_pool_size": len(additional),
        "new_from_scenes": len(set(additional) - set(existing)), "union_size": len(union_ids),
    }


def union_answer(query: str, k: int = 6):
    ids, debug = union_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = union_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']} top_scenes={debug['top_scenes']}")
    print(f"existing={debug['existing_pool_size']} additional={debug['additional_pool_size']} "
          f"new_from_scenes={debug['new_from_scenes']} union={debug['union_size']}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
