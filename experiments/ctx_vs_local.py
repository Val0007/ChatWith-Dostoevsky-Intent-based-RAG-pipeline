"""Ablation probe: local (prev+next chunk) tagging  vs  context-aware (scene + global map).

Same tagger, same chunks; the ONLY variable is how much context the model sees:
  LOCAL         = tag_one_chunk(text, prev_chunk, next_chunk)          # naive neighbor window
  CONTEXT-AWARE = tag_with_context(text, global_map, adjacent scenes)  # the V1 pipeline

Run from the project root:
    .venv/bin/python experiments/ctx_vs_local.py

This is a two-row slice of the eventual V2 ablation (config: local-neighbors vs full-context).
See experiments/FINDINGS.md for the recorded result.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunking import load_chunks
from ingest import _scene_context
from scenes import render_global_map
from tag_white_nights import tag_one_chunk, tag_with_context

# chunks where the local window and the whole-book view should diverge
TARGETS = [
    ("white_nights_fourth_night_012", 'the doomed joy — "now I am happy" (book undoes it next)'),
    ("white_nights_second_night_012", "rapturous fantasy peak (beautiful prose)"),
    ("white_nights_morning_002", "Nastenka's farewell letter"),
    ("white_nights_third_night_009", 'Nastenka: "you are not like other people"'),
    ("white_nights_second_night_034", "Nastenka recounting the lodger's departure"),
]


def _fields(t: dict) -> tuple:
    return (t["speaker"], t["speaker_relation"], t["narrative_relation"],
            ",".join(t["canonical_themes"]))


def main():
    chunks = {c["id"]: c for c in load_chunks()}
    gm_str = render_global_map(json.loads((ROOT / "context" / "global_map.json").read_text()))
    scenes = json.loads((ROOT / "context" / "scenes.json").read_text())
    cards = json.loads((ROOT / "context" / "scene_cards.json").read_text())
    scene_of = {cid: i for i, s in enumerate(scenes) for cid in s["chunk_ids"]}

    for cid, note in TARGETS:
        c = chunks[cid]
        prev = chunks[c["prev_id"]]["text"] if c["prev_id"] else None
        nxt = chunks[c["next_id"]]["text"] if c["next_id"] else None
        loc = tag_one_chunk(c["text"], prev, nxt).model_dump()
        ctx = tag_with_context(c["text"], gm_str, _scene_context(cards, scene_of[cid])).model_dump()
        ls, lr, ln, lt = _fields(loc)
        cs, cr, cn, ct = _fields(ctx)
        print("=" * 94)
        print(f"{cid}  —  {note}")
        print(f"  {'':14}{'LOCAL (prev+next chunk)':<36}{'CONTEXT-AWARE (scene+map)'}")
        print(f"  speaker      {ls:<36}{cs}{'  <<<' if ls != cs else ''}")
        print(f"  speaker_rel  {lr:<36}{cr}{'  <<<' if lr != cr else ''}")
        print(f"  narrative    {ln:<36}{cn}{'  <<< CHANGED' if ln != cn else ''}")
        print(f"  themes       {lt:<36}{ct}")


if __name__ == "__main__":
    main()
