"""Scene-card-first retrieval:

    QUERY
      |
      v
  scene cards (BM25 + dense over the 27 scene SUMMARIES, not the raw text)
      |
      v
  top scenes
      |
      v
  chunks within those scenes, scored 3 ways:
    local BM25 (index built ONLY from this scene's chunks -> different IDF)
    global BM25 (the whole-book index, restricted to these chunks)
    dense (embedding similarity, restricted to these chunks)
      |
      v
  fusion (min-max normalize each signal, weighted sum)
      |
      v
  cheap rerank (reused from production_method.py — no LLM call, see R1-C/R2/R3/R4)
      |
      v
  TOP-K -> retrieval.answer()

Hypothesis this tests (experiments/retrieval/FINDINGS.md R1-A, R4): several questions
(q01, q02, q10) share almost no vocabulary with their answer PASSAGE at any signal —
q01's gold chunk sits at dense rank 50/84, BM25 rank 74/84. But scene cards are LLM-
written prose SUMMARIES in different words than the source text. scene_03's card says
"the narrator intervenes with a stick, scaring off the gentleman" for the exact scene
q01 asks about ("how does the narrator first meet Nastenka") — much closer to the
question's own phrasing than the source passage is. If cards bridge that gap, searching
them first should recover scenes that direct passage search misses.
"""
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from production_method import _minmax, cheap_rerank  # noqa: E402
from retrieval import EMBED_MODEL, _diversify, _tok, answer, by_id  # noqa: E402
from retrieval import _bm25 as _global_bm25_index  # noqa: E402
from retrieval import _scenes, chunk_to_scene, client, collection, embed, order, scene_cards  # noqa: E402

_scene_order = [s["scene_id"] for s in _scenes]
_scene_chunks = {s["scene_id"]: s["chunk_ids"] for s in _scenes}
_card_texts = [scene_cards[sid].get("summary", "") for sid in _scene_order]

_card_bm25 = BM25Okapi([_tok(t) for t in _card_texts])


# example ─ in:  ["summary of scene_01", "summary of scene_02", ...]  (27 short texts)
#           out: [[0.01, -0.03, ...], ...]   one embedding per scene card, computed once at import
def _embed_many(texts: list[str]) -> list[list[float]]:
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


_CARD_EMBEDDINGS = dict(zip(_scene_order, _embed_many(_card_texts)))


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------- 1. scene cards -> top scenes
# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: ["scene_03", "scene_02", "scene_01", ...]   6 scene ids, best card-match first
def top_scenes(query: str, n: int = 6) -> list[str]:
    bm25_scores = _card_bm25.get_scores(_tok(query))
    bm25_norm = dict(zip(_scene_order, _minmax(list(bm25_scores))))

    qv = embed(query)
    dense_scores = [_cosine(qv, _CARD_EMBEDDINGS[sid]) for sid in _scene_order]
    dense_norm = dict(zip(_scene_order, _minmax(dense_scores)))

    fused = {sid: 0.5 * bm25_norm[sid] + 0.5 * dense_norm[sid] for sid in _scene_order}
    return sorted(_scene_order, key=lambda s: fused[s], reverse=True)[:n]


# ---------------------------------------------------------------- 1b. weighted multi-field scoring
# R6 extension: scene cards carry more than `summary` — `characters`, `developments`,
# `consequences`, `important_beliefs_or_feelings` are all list fields, unused above. Score
# each field independently (BM25 + dense, 50/50, same as top_scenes) and blend by weight.
DEFAULT_FIELD_WEIGHTS = {
    "summary": 0.40,
    "developments": 0.25,
    "characters": 0.15,
    "consequences": 0.10,
    "important_beliefs_or_feelings": 0.10,
}


def _field_text(card: dict, field: str) -> str:
    v = card.get(field, "")
    return ". ".join(v) if isinstance(v, list) else (v or "")


_FIELD_TEXTS = {
    field: [_field_text(scene_cards[sid], field) for sid in _scene_order]
    for field in DEFAULT_FIELD_WEIGHTS
}
_FIELD_BM25 = {field: BM25Okapi([_tok(t) for t in texts]) for field, texts in _FIELD_TEXTS.items()}

# batch-embed every (field, scene) text in ONE call rather than one per field
_FIELD_EMBED_KEYS = [(field, sid) for field in DEFAULT_FIELD_WEIGHTS for sid in _scene_order]
_FIELD_EMBED_TEXTS = [_FIELD_TEXTS[field][_scene_order.index(sid)] or f"({field}: none)"
                       for field, sid in _FIELD_EMBED_KEYS]
_FIELD_EMBEDDINGS = dict(zip(_FIELD_EMBED_KEYS, _embed_many(_FIELD_EMBED_TEXTS)))


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: {"scene_03": 0.41, "scene_14": 0.77, ...}   ALL 27 scenes, not just the top-n
#                (exposed separately from top_scenes_weighted so callers who need the SCORE,
#                not just the ranking, don't have to recompute it — see union_method.py)
def scene_scores_weighted(query: str, field_weights: dict = None) -> dict:
    weights = field_weights or DEFAULT_FIELD_WEIGHTS
    qv = embed(query)
    field_scores = {}
    for field in weights:
        bm25_scores = _FIELD_BM25[field].get_scores(_tok(query))
        bm25_norm = dict(zip(_scene_order, _minmax(list(bm25_scores))))
        dense_scores = [_cosine(qv, _FIELD_EMBEDDINGS[(field, sid)]) for sid in _scene_order]
        dense_norm = dict(zip(_scene_order, _minmax(dense_scores)))
        field_scores[field] = {sid: 0.5 * bm25_norm[sid] + 0.5 * dense_norm[sid] for sid in _scene_order}

    return {sid: sum(weights[f] * field_scores[f][sid] for f in weights) for sid in _scene_order}


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: ["scene_03", ...]   6 scene ids, ranked by the WEIGHTED blend of all 5 fields
#                (vs. top_scenes() above, which only looks at `summary`)
def top_scenes_weighted(query: str, n: int = 6, field_weights: dict = None) -> list[str]:
    fused = scene_scores_weighted(query, field_weights)
    return sorted(_scene_order, key=lambda s: fused[s], reverse=True)[:n]


def _chunks_in(scene_ids: list[str]) -> list[str]:
    out = []
    for sid in scene_ids:
        out.extend(_scene_chunks[sid])
    return out


# ---------------------------------------------------------------- 2. within-scene chunk scoring
# example ─ in:  ("...", ["white_nights_first_night_009", "white_nights_first_night_010", ...])
#           out: {"white_nights_first_night_009": 4.1, ...}   BM25 score from an index built
#                ONLY on these chunks — different term rarity (IDF) than the full-book index
def _local_bm25_scores(query: str, chunk_ids: list[str]) -> dict:
    idx = BM25Okapi([_tok(by_id[cid]["text"]) for cid in chunk_ids])
    return dict(zip(chunk_ids, idx.get_scores(_tok(query))))


# example ─ in:  same as above
#           out: {"white_nights_first_night_009": 2.7, ...}   same chunks, scored by the
#                WHOLE-BOOK BM25 index (retrieval._bm25) instead of a scene-scoped one
def _global_bm25_scores(query: str, chunk_ids: list[str]) -> dict:
    all_scores = _global_bm25_index.get_scores(_tok(query))
    idx_of = {cid: i for i, cid in enumerate(order)}
    return {cid: all_scores[idx_of[cid]] for cid in chunk_ids}


# example ─ in:  same as above
#           out: {"white_nights_first_night_009": 0.31, ...}   higher = closer (raw, un-normalized)
def _dense_scores(query: str, chunk_ids: list[str]) -> dict:
    qv = embed(query)
    res = collection.query(query_embeddings=[qv], n_results=len(order), include=["distances"])
    dist_by_id = dict(zip(res["ids"][0], res["distances"][0]))
    return {cid: -dist_by_id[cid] for cid in chunk_ids}


# ---------------------------------------------------------------- 3. fusion
# example ─ in:  ("...", [10 chunk ids from 6 top scenes])
#           out: {chunk_id: 0.0-1.0 fused score, ...}
def fuse_within_scenes(query: str, chunk_ids: list[str],
                        weights: tuple = (0.35, 0.35, 0.30)) -> dict:
    """weights = (local_bm25, global_bm25, dense)."""
    w_local, w_global, w_dense = weights
    local = _local_bm25_scores(query, chunk_ids)
    glob = _global_bm25_scores(query, chunk_ids)
    dense = _dense_scores(query, chunk_ids)

    local_n = dict(zip(chunk_ids, _minmax([local[c] for c in chunk_ids])))
    global_n = dict(zip(chunk_ids, _minmax([glob[c] for c in chunk_ids])))
    dense_n = dict(zip(chunk_ids, _minmax([dense[c] for c in chunk_ids])))

    return {cid: w_local * local_n[cid] + w_global * global_n[cid] + w_dense * dense_n[cid]
            for cid in chunk_ids}


# ---------------------------------------------------------------- 4. the pipeline
# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...], {"top_scenes": ["scene_03", ...]})
def scene_card_retrieve(query: str, k: int = 6, n_scenes: int = 9, per_scene: int = 2):
    scenes = top_scenes(query, n=n_scenes)
    chunk_ids = _chunks_in(scenes)
    fused = fuse_within_scenes(query, chunk_ids)
    ranked = sorted(chunk_ids, key=lambda c: fused[c], reverse=True)
    reranked = cheap_rerank(query, ranked, fused)
    ids = _diversify(reranked, k, per_scene)
    return ids, {"top_scenes": scenes}


def scene_card_answer(query: str, k: int = 6):
    ids, debug = scene_card_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = scene_card_retrieve(query)
    print(f"QUESTION: {query}\ntop_scenes={debug['top_scenes']}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
