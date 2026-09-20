"""Q7 + Q6 — what context helps `narrative_relation`, and is its variation signal or noise?

Same chunk, four context arms, N runs each at temperature 0.7:
    neither  = no context (chunk only)
    map      = global map only
    scenes   = adjacent scene cards only
    both     = map + scenes  (the production setting)

Per (chunk, arm) we record the N labels -> modal label + stability (fraction agreeing).
A small HAND-VERIFIED gold set (acceptable label sets) lets us score each arm's accuracy.

  Q7 answered by: which arm's modal label most often lands in gold.
  Q6 answered by: mean stability per arm (low stability = the variation is noise).

Run from repo root:  .venv/bin/python experiments/narrative_context_ablation.py
Spends API (chunks x 4 arms x N calls). Logs go to experiments/FINDINGS.md by hand after.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunking import load_chunks
from ingest import _scene_context
from scenes import render_global_map
from tag_white_nights import CONTEXT_USER_TEMPLATE, SYSTEM_PROMPT, ChunkTags, client

N = 3                       # runs per (chunk, arm)
TEMP = 0.7                  # non-zero so stability is meaningful
MODEL = "gpt-4o-mini"

# hand-verified acceptable narrative_relation sets (see FINDINGS spot-checks)
GOLD = {
    "white_nights_second_night_012": {"undermines"},                 # fantasy peak
    "white_nights_second_night_014": {"undermines"},                 # self-deception
    "white_nights_second_night_019": {"undermines"},                 # "lost all instinct for the real"
    "white_nights_first_night_002":  {"unclear"},                    # transitional walk
    "white_nights_fourth_night_012": {"complicates", "undermines"},  # doomed joy
    "white_nights_morning_002":      {"undermines", "unresolved"},   # farewell letter
    "white_nights_second_night_034": {"unresolved", "complicates"},  # departure recounting
}


def narr(text, gm, sc, temp):
    user = CONTEXT_USER_TEMPLATE.format(global_map=gm, scene_context=sc, target_chunk_text=text)
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=temp, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}])
        return ChunkTags(**json.loads(r.choices[0].message.content)).narrative_relation
    except Exception:
        return "INVALID"


def main():
    chunks = {c["id"]: c for c in load_chunks()}
    gm_str = render_global_map(json.loads((ROOT / "context" / "global_map.json").read_text()))
    scenes = json.loads((ROOT / "context" / "scenes.json").read_text())
    cards = json.loads((ROOT / "context" / "scene_cards.json").read_text())
    scene_of = {cid: i for i, s in enumerate(scenes) for cid in s["chunk_ids"]}

    arms = ["neither", "map", "scenes", "both"]
    agg = {a: {"hits": 0, "stab": [], "labels": Counter()} for a in arms}

    for cid, gold in GOLD.items():
        text = chunks[cid]["text"]
        sc = _scene_context(cards, scene_of[cid])
        ctx = {"neither": ("(none)", "(none)"), "map": (gm_str, "(none)"),
               "scenes": ("(none)", sc), "both": (gm_str, sc)}
        print("=" * 78)
        print(f"{cid}   gold={sorted(gold)}")
        for a in arms:
            g, s = ctx[a]
            labels = [narr(text, g, s, TEMP) for _ in range(N)]
            mode, cnt = Counter(labels).most_common(1)[0]
            stab = cnt / N
            hit = mode in gold
            agg[a]["hits"] += hit
            agg[a]["stab"].append(stab)
            agg[a]["labels"][mode] += 1
            print(f"  {a:8s} modal={mode:<12} stability={stab:.2f}  {'✓' if hit else '✗'}   runs={labels}")

    n = len(GOLD)
    print("\n" + "=" * 78)
    print(f"SUMMARY over {n} chunks (N={N}, temp={TEMP})")
    print(f"  {'arm':8s} {'accuracy':>10} {'mean_stability':>16}   modal-label distribution")
    for a in arms:
        acc = agg[a]["hits"] / n
        ms = sum(agg[a]["stab"]) / n
        print(f"  {a:8s} {acc:>9.0%} {ms:>16.2f}   {dict(agg[a]['labels'])}")


if __name__ == "__main__":
    main()
