"""Settles the open question from the F9 write-up: is canonical_themes/characters_present's
regression in the three-pass pipeline (E) caused by Pass 3's adjudication, or was Pass 2's
own candidate already short of B's score?

B (scenes only, single call, production prompt) scores canonical_themes F1 = 0.67 -- identical
to D (scenes+map). Pass 2 of the three-pass pipeline gets the SAME context as B (chunk + scene
cards, no map). If context were the whole story, Pass 2's raw candidate -- scored BEFORE Pass 3
ever adjudicates it -- should already sit near 0.67. E's actual (post-adjudication) score is 0.62.

This script calls tag_scene_pass() directly (Pass 2 alone, no Pass 3) over the same 30 gold
chunks used throughout this study and scores its raw candidate output against gold, isolating
Pass 2 from the adjudication step that follows it in the real pipeline.

Run from repo root:
    .venv/bin/python experiments/tagging/score_pass2_raw.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "tagging"))

from chunking import load_chunks
from ingest import _scene_context
from tag_three_pass import tag_scene_pass

from run_first30_eval import load_gold, score_set_field, pct

CACHE_PATH = ROOT / "experiments" / "tagging" / "results" / "pass2_raw_output.json"


def tag_pass2_raw(gold_ids: list) -> dict:
    chunks = {c["id"]: c for c in load_chunks()}
    scenes = json.loads((ROOT / "data" / "context" / "scenes.json").read_text())
    cards = json.loads((ROOT / "data" / "context" / "scene_cards.json").read_text())
    scene_of = {cid: i for i, s in enumerate(scenes) for cid in s["chunk_ids"]}

    out = {}
    for i, cid in enumerate(gold_ids, 1):
        c = chunks[cid]
        sc = _scene_context(cards, scene_of[cid])
        try:
            tags = tag_scene_pass(c["text"], sc)
            result = {"canonical_themes": tags.canonical_themes,
                      "characters_present": tags.characters_present}
        except Exception as e:
            print(f"  ! pass2 failed for {cid}: {e}")
            result = None
        out[cid] = result
        print(f"  {i:>2}/{len(gold_ids)}  {cid:<32} "
              f"themes={result['canonical_themes'] if result else 'FAIL'}")
    return out


def main():
    gold = load_gold()

    if CACHE_PATH.exists():
        print(f"Reusing cached Pass 2 output at {CACHE_PATH} (delete it to re-tag)")
        preds = json.loads(CACHE_PATH.read_text())
    else:
        print(f"Tagging {len(gold)} chunks with Pass 2 (scene pass) ALONE, no adjudication...")
        preds = tag_pass2_raw(list(gold.keys()))
        CACHE_PATH.write_text(json.dumps(preds, indent=2, ensure_ascii=False))
        print(f"\nCached Pass 2 raw output -> {CACHE_PATH}")

    print("\n" + "=" * 90)
    print("PASS 2 RAW CANDIDATE (pre-adjudication) vs. gold -- settles the F9 regression question")
    print("=" * 90)

    for field in ("canonical_themes", "characters_present"):
        s = score_set_field(field, gold, preds)
        print(f"\n{field}")
        print(f"  Pass 2 raw (this script) micro F1 = {s['micro']['f1']:.2f}   "
              f"(P={pct(s['micro']['precision'])} R={pct(s['micro']['recall'])})")
        if s["over_applied"]:
            print(f"  over-applied: {s['over_applied']}")
        if s["under_applied"]:
            print(f"  under-applied: {s['under_applied']}")

    print("\n" + "-" * 90)
    print("Reference points already established:")
    print("  B  (scenes only, single call, no adjudication)         canonical_themes F1 = 0.67")
    print("  D  (scenes+map, single call, no adjudication)          canonical_themes F1 = 0.67")
    print("  E  (three-pass: Pass2 candidate -> Pass3 adjudicated)  canonical_themes F1 = 0.62")
    print("-" * 90)
    print("If Pass 2 raw (above) is ~0.67: the regression is Pass 3's adjudication step.")
    print("If Pass 2 raw (above) is already below 0.67: the 'candidate' framing itself is")
    print("softening Pass 2's answer, before Pass 3 ever touches it.")


if __name__ == "__main__":
    main()
