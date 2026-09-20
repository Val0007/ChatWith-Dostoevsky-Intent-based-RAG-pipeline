"""Tag the full 84-chunk book with the LOCKED G configuration (src/tag_three_pass.py's
tag_three_pass: local pass with speaker_relation gold examples + F10's verified speaking_voice
fix, scene pass at E baseline, global pass at E baseline with adjudication) and write tags.jsonl.

G was validated on the 30-chunk gold set only (see experiments/tagging/FINDINGS.md F13) -- this applies
that same locked prompt state to the other 54 chunks it has never seen, unchanged.

Run from repo root:
    .venv/bin/python experiments/tagging/tag_full_book_g.py
"""
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chunking import ROOT as CHUNK_ROOT, load_chunks
from scenes import load_evidence_store, render_global_map
from ingest import _scene_context, _FALLBACK_TAGS
from tag_three_pass import tag_three_pass

OUT_PATH = CHUNK_ROOT / "data" / "tags.jsonl"


def main():
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}
    n = len(chunks)
    print(f"Total chunks: {n}")

    store = load_evidence_store()
    if not store:
        print("No evidence store at data/context/ -- cannot proceed (need scenes.json, "
              "scene_cards.json, global_map.json).")
        sys.exit(1)
    scenes, cards, global_map = store
    print(f"Loaded evidence store: {len(scenes)} scenes")

    scene_of = {cid: si for si, sc in enumerate(scenes) for cid in sc["chunk_ids"]}
    gm_str = render_global_map(global_map)

    rows = []
    n_fail = 0
    for i, c in enumerate(chunks):
        si = scene_of.get(c["id"], 0)
        scene_id = scenes[si]["scene_id"]
        sc_ctx = _scene_context(cards, si)
        prev = by_id[c["prev_id"]]["text"] if c["prev_id"] else None
        nxt = by_id[c["next_id"]]["text"] if c["next_id"] else None

        tags = None
        for attempt in (1, 2):
            try:
                tags = tag_three_pass(c["text"], prev, nxt, sc_ctx, gm_str)
                break
            except Exception as e:
                if attempt == 2:
                    tags = dict(_FALLBACK_TAGS)
                    n_fail += 1
                    print(f"  ! tag fallback for {c['id']}: {e}")

        rows.append({"id": c["id"], "scene_id": scene_id, **tags})
        print(f"  {i + 1:>2}/{n} {c['id']:<32} {scene_id:<10} "
              f"{str(tags['speaking_voice']):<24} {tags['speaker_relation']:<9} "
              f"{tags['narrative_relation']}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTagged: {n - n_fail}/{n} validated, {n_fail} fell back.")
    print(f"Wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
