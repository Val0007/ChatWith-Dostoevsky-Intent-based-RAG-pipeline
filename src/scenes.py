"""Context generation (evidence store) for White Nights.

PASS 1: segment the 84 chunks into coherent scenes, then write a factual card per scene.
PASS 2: synthesize all scene cards into one global novel map (arcs/turning points/
        resolutions/open questions).

Scene cards record EVIDENCE only (what happens), never interpretation. The global map
records the whole-book arc. Together they are the context a later tagger uses to judge
narrative_relation. Outputs land in data/context/ as JSON.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from chunking import ROOT, load_chunks

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"
CONTEXT_DIR = ROOT / "data" / "context"

SEGMENTATION_PROMPT = """You are segmenting Dostoevsky's "White Nights" into coherent narrative scenes.

Given the ordered passages below, group adjacent passages that belong to the
same continuous scene, conversation, memory, or narrative event.

Start a new scene when there is a meaningful change in:
- location
- time
- conversation topic or a distinct action beat
- narrative situation (e.g. present narration vs. a character's recounted memory)

Aim for roughly 15-25 scenes across the whole novella. A single "Night" almost
always contains SEVERAL scenes - do NOT merge an entire Night into one scene.
Most scenes are 2-6 passages long; a scene longer than ~8 passages should be
split at the next natural shift.

Do not interpret the literature. Only identify structural scene boundaries.
Every passage number must appear in exactly one scene, and scenes must be
contiguous and in order.

Return JSON:
{
  "scenes": [
    {
      "scene_id": "scene_01",
      "chunk_ids": ["1", "2", "3"],
      "reason": "brief factual reason these chunks form one scene"
    }
  ]
}"""

SCENE_CARD_PROMPT = """You are creating a factual evidence card for Dostoevsky's "White Nights".

Summarize what happens in this scene so that another model can later
understand individual passages in the context of the whole novella.

Do NOT interpret the author's message.
Do NOT assign supports, complicates, undermines, unresolved, or unclear.
Do NOT describe what Dostoevsky "means."

Record only evidence:
- what happens
- who wants what
- important beliefs or feelings expressed
- important changes from earlier in the story
- consequences of events

Be concise but preserve details that could matter for understanding the
meaning of individual passages.

Return JSON:
{
  "summary": "...",
  "characters": ["..."],
  "important_beliefs_or_feelings": ["..."],
  "developments": ["..."],
  "consequences": ["..."]
}"""

GLOBAL_MAP_PROMPT = """You are creating a factual reference map of Dostoevsky's "White Nights".

Your job is to record the major developments of the novella so that a later
classifier can understand an individual passage in its larger context.

Do NOT classify literary meaning.
Do NOT assign supports, complicates, undermines, unresolved, or unclear.

Focus on:
1. The Dreamer's character arc
2. Nastenka's character arc
3. The development of their relationship
4. Major turning points
5. Recurring ideas that are explicitly developed through events
6. Important contradictions between what characters believe and what happens
7. How important questions are resolved or left open by the ending

Return JSON:
{
  "plot_arc": ["..."],
  "dreamer_arc": ["..."],
  "nastenka_arc": ["..."],
  "relationship_arc": ["..."],
  "major_turning_points": ["..."],
  "important_resolutions": ["..."],
  "important_open_questions": ["..."]
}"""


class SceneCard(BaseModel):
    scene_id: str = ""
    summary: str
    characters: List[str] = []
    important_beliefs_or_feelings: List[str] = []
    developments: List[str] = []
    consequences: List[str] = []


class GlobalMap(BaseModel):
    plot_arc: List[str] = []
    dreamer_arc: List[str] = []
    nastenka_arc: List[str] = []
    relationship_arc: List[str] = []
    major_turning_points: List[str] = []
    important_resolutions: List[str] = []
    important_open_questions: List[str] = []


# example ─ in:  (system_prompt, user_text)   out: {"scenes": [...]}  (parsed JSON object from the LLM)
def _chat_json(system: str, user: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return json.loads(r.choices[0].message.content)


# ---------- PASS 1a: segmentation ----------
# example ─ in:  the 84 chunks
#           out: [{"scene_id": "scene_01", "chunk_ids": ["white_nights_first_night_001",
#                  "...002", "...003"], "reason": "the narrator's opening walk"}, ...]   # 27 scenes
def segment_scenes(chunks: list[dict]) -> list[dict]:
    """Group chunks into scenes; guarantee full, contiguous, one-scene-each coverage."""
    n = len(chunks)
    listing = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    raw = _chat_json(SEGMENTATION_PROMPT, f"PASSAGES:\n\n{listing}")
    raw_scenes = raw.get("scenes", [])

    # first occurrence of each ordinal wins
    assign = {}
    for si, sc in enumerate(raw_scenes):
        for cid in sc.get("chunk_ids", []):
            try:
                o = int(str(cid).strip())
            except ValueError:
                continue
            if 1 <= o <= n and o not in assign:
                assign[o] = si
    # fill any missing ordinal with the previous scene's assignment
    last = 0
    for o in range(1, n + 1):
        if o in assign:
            last = assign[o]
        else:
            assign[o] = last

    groups = defaultdict(list)
    for o in range(1, n + 1):
        groups[assign[o]].append(o)

    scenes = []
    for k, si in enumerate(sorted(groups), start=1):
        ords = groups[si]
        scenes.append({
            "scene_id": f"scene_{k:02d}",
            "chunk_ids": [chunks[o - 1]["id"] for o in ords],
            "reason": raw_scenes[si].get("reason", "") if si < len(raw_scenes) else "",
        })
    return scenes


# ---------- PASS 1b: scene cards ----------
# example ─ in:  (scenes, chunks)
#           out: [{"scene_id": "scene_25", "chunk_ids": [...],
#                  "summary": "Nastenka expresses her conflicting feelings...",
#                  "characters": ["Nastenka", "Narrator"],
#                  "important_beliefs_or_feelings": [...], "developments": [...],
#                  "consequences": [...]}, ...]   # one card per scene, facts only
def build_scene_cards(scenes: list[dict], chunks: list[dict]) -> list[dict]:
    by_id = {c["id"]: c for c in chunks}
    cards = []
    for sc in scenes:
        scene_text = "\n\n".join(by_id[cid]["text"] for cid in sc["chunk_ids"])
        data = _chat_json(SCENE_CARD_PROMPT, f"SCENE TEXT:\n\n{scene_text}")
        card = SceneCard(**data).model_dump()
        card["scene_id"] = sc["scene_id"]
        card["chunk_ids"] = sc["chunk_ids"]
        cards.append(card)
        print(f"  card {sc['scene_id']} ({len(sc['chunk_ids'])} chunks): {card['summary'][:80]}")
    return cards


# ---------- PASS 2: global map ----------
# example ─ in:  the 27 scene cards (their summaries)
#           out: {"relationship_arc": ["strangers -> intimacy -> she returns to the lodger"],
#                 "major_turning_points": [...], "important_resolutions":
#                 ["the Dreamer's hope for love is denied; Nastenka marries the lodger"],
#                 "important_open_questions": ["does he ever leave his fantasy life"], ...}
def build_global_map(cards: list[dict]) -> dict:
    digest = "\n".join(
        f"{c['scene_id']}: {c['summary']}" for c in cards
    )
    data = _chat_json(GLOBAL_MAP_PROMPT, f"SCENE SUMMARIES (in order):\n\n{digest}")
    return GlobalMap(**data).model_dump()


# example ─ in:  ()   out: (scenes, cards, global_map) read from data/context/*.json, or None if missing
def load_evidence_store():
    """Return (scenes, cards, global_map) from data/context/ if all three exist, else None."""
    files = [CONTEXT_DIR / f for f in ("scenes.json", "scene_cards.json", "global_map.json")]
    if not all(f.exists() for f in files):
        return None
    return tuple(json.loads(f.read_text(encoding="utf-8")) for f in files)


# example ─ in:  the 84 chunks
#           out: (scenes, cards, global_map)  AND writes data/context/scenes.json,
#                scene_cards.json, global_map.json  (runs PASS 1 + PASS 2)
def build_evidence_store(chunks: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Run PASS 1 + PASS 2, save to data/context/, return (scenes, cards, global_map)."""
    CONTEXT_DIR.mkdir(exist_ok=True)
    print("PASS 1a: segmenting scenes...")
    scenes = segment_scenes(chunks)
    print(f"  -> {len(scenes)} scenes")
    print("PASS 1b: writing scene cards...")
    cards = build_scene_cards(scenes, chunks)
    print("PASS 2: building global novel map...")
    global_map = build_global_map(cards)

    (CONTEXT_DIR / "scenes.json").write_text(json.dumps(scenes, indent=2, ensure_ascii=False))
    (CONTEXT_DIR / "scene_cards.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    (CONTEXT_DIR / "global_map.json").write_text(json.dumps(global_map, indent=2, ensure_ascii=False))
    print(f"Evidence store written to {CONTEXT_DIR}/")
    return scenes, cards, global_map


# ---------- rendering (used by the tagger) ----------
# example ─ in:  (scene_25_card, "CURRENT SCENE")
#           out: "[CURRENT SCENE: scene_25]\n  summary: Nastenka expresses...\n
#                 beliefs/feelings: ...\n  developments: ...\n  consequences: ..."
def render_scene_card(card: dict, label: str) -> str:
    def line(k):
        vals = card.get(k) or []
        return "; ".join(vals) if vals else "(none)"
    return (
        f"[{label}: {card['scene_id']}]\n"
        f"  summary: {card['summary']}\n"
        f"  beliefs/feelings: {line('important_beliefs_or_feelings')}\n"
        f"  developments: {line('developments')}\n"
        f"  consequences: {line('consequences')}"
    )


# example ─ in:  the global_map dict
#           out: "[GLOBAL NOVEL MAP]\n  relationship_arc:\n    - strangers -> intimacy -> ...\n
#                 important_resolutions:\n    - the Dreamer's hope is denied...\n  ..."
def render_global_map(gm: dict) -> str:
    def block(k):
        vals = gm.get(k) or []
        return "\n".join(f"    - {v}" for v in vals) if vals else "    (none)"
    parts = ["[GLOBAL NOVEL MAP]"]
    for k in ["relationship_arc", "major_turning_points",
              "important_resolutions", "important_open_questions"]:
        parts.append(f"  {k}:\n{block(k)}")
    return "\n".join(parts)


def main():
    chunks = load_chunks()
    print(f"Chunks: {len(chunks)}")
    build_evidence_store(chunks)


if __name__ == "__main__":
    main()
