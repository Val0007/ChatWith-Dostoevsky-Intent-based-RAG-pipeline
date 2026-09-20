"""Full pipeline: chunk -> evidence store -> context-aware tag -> embed -> Chroma.

PASS 1-2 (scenes.py): segment scenes, write scene cards, synthesize the global map.
PASS 3-4 (here): for each chunk, assemble context (global map + adjacent scene cards)
                 and tag it, then embed and store everything in ./db. Also writes
                 tags.jsonl for hand spot-checking.
"""
import json

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from chunking import ROOT, load_chunks
from scenes import (build_evidence_store, load_evidence_store,
                    render_global_map, render_scene_card)
from tag_three_pass import tag_three_pass

load_dotenv()

DB_PATH = str(ROOT / "db")
EMBED_MODEL = "text-embedding-3-small"
COLLECTION = "dostoevsky"

client = OpenAI()

_FALLBACK_TAGS = {
    "narrating_voice": "the Dreamer",
    "speaking_voice": "the Dreamer",
    "characters_present": [],
    "speaker_relation": "explores",
    "narrative_relation": "unclear",
    "canonical_themes": [],
    "local_motifs": [],
}


# example ─ in:  ["It was a wonderful night...", "chunk 2 text", ...]   (all 84 chunk texts)
#           out: [[0.013, -0.021, ...], [...], ...]   (one 1536-dim vector per text, in one call)
def embed(texts: list[str]) -> list[list[float]]:
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


# example ─ in:  (cards, si=24)   # target chunk lives in scene index 24
#           out: "[PREVIOUS SCENE: scene_24]...\n\n[CURRENT SCENE: scene_25]...\n\n
#                 [NEXT SCENE: scene_26]...\n\n[NEXT+1 SCENE: scene_27]..."
def _scene_context(cards: list[dict], si: int) -> str:
    """Render the adjacent-scene window around scene index si: prev, current, +1, +2."""
    window = [(si - 1, "PREVIOUS SCENE"), (si, "CURRENT SCENE"),
              (si + 1, "NEXT SCENE"), (si + 2, "NEXT+1 SCENE")]
    blocks = [render_scene_card(cards[j], label)
              for j, label in window if 0 <= j < len(cards)]
    return "\n\n".join(blocks)


# example ─ in:  (chunks, scenes, cards, global_map)
#           effect: mutates each chunk in place -> c["scene_id"]="scene_25",
#                   c["tags"]={narrating_voice, speaking_voice, speaker_relation,
#                   narrative_relation, canonical_themes, characters_present, local_motifs},
#                   c["tag_ok"]=True; prints a progress line per chunk. returns None.
def tag_chunks_with_context(chunks, scenes, cards, global_map) -> None:
    """PASS 3-4: assemble context per chunk and tag it with the locked G three-pass
    pipeline (see src/tag_three_pass.py, experiments/FINDINGS.md F13) -- one retry, then
    fallback."""
    by_id = {c["id"]: c for c in chunks}
    scene_of = {cid: si for si, sc in enumerate(scenes) for cid in sc["chunk_ids"]}
    gm_str = render_global_map(global_map)
    n = len(chunks)
    for i, c in enumerate(chunks):
        si = scene_of.get(c["id"], 0)
        c["scene_id"] = scenes[si]["scene_id"]
        ctx = _scene_context(cards, si)
        prev = by_id[c["prev_id"]]["text"] if c["prev_id"] else None
        nxt = by_id[c["next_id"]]["text"] if c["next_id"] else None
        for attempt in (1, 2):
            try:
                c["tags"] = tag_three_pass(c["text"], prev, nxt, ctx, gm_str)
                c["tag_ok"] = True
                break
            except Exception as e:
                if attempt == 2:
                    c["tags"] = dict(_FALLBACK_TAGS)
                    c["tag_ok"] = False
                    print(f"  ! tag fallback for {c['id']}: {e}")
        t = c["tags"]
        print(f"  {i + 1:>2}/{n} {c['id']:<32} {c['scene_id']:<9} "
              f"{str(t['speaking_voice']):<24} {t['speaker_relation']:<9} {t['narrative_relation']}")


def main():
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}\n")

    # PASS 1-2: evidence store (reuse context/ if already built)
    store = load_evidence_store()
    if store:
        scenes, cards, global_map = store
        print(f"Reusing evidence store: {len(scenes)} scenes from context/")
    else:
        scenes, cards, global_map = build_evidence_store(chunks)

    # PASS 3-4: context-aware tagging
    print("\nPASS 3-4: context-aware tagging...")
    tag_chunks_with_context(chunks, scenes, cards, global_map)
    n_fail = sum(1 for c in chunks if not c.get("tag_ok"))
    print(f"Tagged: {len(chunks) - n_fail}/{len(chunks)} validated, {n_fail} fell back.")

    # side-output for spot-checking
    with open(ROOT / "tags.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c["id"], "scene_id": c["scene_id"], **c["tags"]},
                               ensure_ascii=False) + "\n")

    # embed + store
    print("\nEmbedding + writing Chroma...")
    db = chromadb.PersistentClient(path=DB_PATH)
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = db.create_collection(COLLECTION)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[
            {
                "work": c["work"],
                "chapter": c["chapter"],
                "section": c["section"],
                "scene_id": c["scene_id"],
                "prev_id": c["prev_id"] or "",
                "next_id": c["next_id"] or "",
                "narrating_voice": c["tags"]["narrating_voice"],
                "speaking_voice": c["tags"]["speaking_voice"] or "",
                "characters_present": ",".join(c["tags"]["characters_present"]),
                "speaker_relation": c["tags"]["speaker_relation"],
                "narrative_relation": c["tags"]["narrative_relation"],
                "canonical_themes": ",".join(c["tags"]["canonical_themes"]),
                "local_motifs": ",".join(c["tags"]["local_motifs"]),
            }
            for c in chunks
        ],
    )
    print(f"Ingested {collection.count()} chunks into '{COLLECTION}'.")


if __name__ == "__main__":
    main()
