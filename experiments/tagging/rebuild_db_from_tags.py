"""Rebuild the Chroma DB from the ALREADY-COMPUTED tags.jsonl (no re-tagging).

tags.jsonl was just written fresh by experiments/tagging/tag_full_book_g.py using the locked G
config. Re-invoking the tagger here would burn another ~250 API calls for no benefit and
risks the DB's tags drifting from tags.jsonl's tags (model non-determinism) -- this script
just loads what's already on disk and does the embed + Chroma write step only.

Run from repo root:
    .venv/bin/python experiments/tagging/rebuild_db_from_tags.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import chromadb

from chunking import load_chunks
from ingest import DB_PATH, COLLECTION, embed

TAGS_PATH = ROOT / "data" / "tags.jsonl"


def main():
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}

    tags_by_id = {}
    for line in TAGS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        tags_by_id[d["id"]] = d

    missing = [c["id"] for c in chunks if c["id"] not in tags_by_id]
    if missing:
        print(f"ERROR: {len(missing)} chunks have no entry in tags.jsonl: {missing[:5]}...")
        sys.exit(1)

    for c in chunks:
        t = tags_by_id[c["id"]]
        c["scene_id"] = t["scene_id"]
        c["tags"] = {k: v for k, v in t.items() if k not in ("id", "scene_id")}

    print(f"Loaded tags for {len(chunks)} chunks from {TAGS_PATH}")

    print("Embedding + writing Chroma...")
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
