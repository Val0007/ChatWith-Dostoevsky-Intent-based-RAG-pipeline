"""Free (no-API) analyses of the current tags over tags.jsonl + context/.

Answers, from existing data:
  Q9  — within-scene agreement per field: is a field really a CHUNK-level property,
        or is it constant across a scene (i.e. it should live at scene level)?
  Q4  — within-Night theme granularity: do themes vary scene-to-scene inside one
        chapter, or is the label chapter-homogeneous?
Run from repo root:  .venv/bin/python experiments/analyze_tagger.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tags = [json.loads(l) for l in open(ROOT / "tags.jsonl")]
by_id = {t["id"]: t for t in tags}
scenes = json.loads((ROOT / "context" / "scenes.json").read_text())


def chapter_of(cid: str) -> str:
    return cid[len("white_nights_"):].rsplit("_", 1)[0]


def key(t, f):
    return tuple(sorted(t["canonical_themes"])) if f == "canonical_themes" else t[f]


FIELDS = ["speaker", "speaker_relation", "narrative_relation", "canonical_themes"]

print("=" * 80)
print("Q9 — WITHIN-SCENE AGREEMENT  (is the field constant across a scene's chunks?)")
print("=" * 80)
multi = [s for s in scenes if len(s["chunk_ids"]) >= 2]
print(f"scenes with >=2 chunks: {len(multi)}/{len(scenes)}\n")
for f in FIELDS:
    uniform = sum(1 for s in multi if len({key(by_id[c], f) for c in s["chunk_ids"] if c in by_id}) == 1)
    pct = 100 * uniform / len(multi)
    print(f"  {f:20s} uniform in {uniform:2d}/{len(multi)} scenes  ({pct:3.0f}%)  "
          f"{'<- effectively scene-level' if pct >= 70 else '<- varies within scene (chunk-level)' if pct <= 40 else ''}")

print("\n" + "=" * 80)
print("Q4 — WITHIN-NIGHT THEME GRANULARITY  (do themes change scene-to-scene in a chapter?)")
print("=" * 80)
chap = defaultdict(list)
for s in scenes:
    chap[chapter_of(s["chunk_ids"][0])].append(s)

order = ["first_night", "second_night", "third_night", "fourth_night", "morning"]
for ch in order:
    ss = chap.get(ch, [])
    if not ss:
        continue
    scene_sets, nr_vals, theme_count = [], set(), Counter()
    for s in ss:
        th = set()
        for c in s["chunk_ids"]:
            if c in by_id:
                th |= set(by_id[c]["canonical_themes"])
                nr_vals.add(by_id[c]["narrative_relation"])
                theme_count.update(by_id[c]["canonical_themes"])
        scene_sets.append(frozenset(th))
    print(f"\n  {ch.upper():13s} scenes={len(ss):2d}  distinct theme-sets across scenes={len(set(scene_sets))}"
          f"  distinct narrative_relation={sorted(nr_vals)}")
    print(f"                themes used in chapter: {dict(theme_count.most_common())}")
