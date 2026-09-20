"""Token-weighted context annotation: approximates RAG-Token's per-token marginalization
in SPIRIT (not literally — true marginalization needs logprob access the chat API doesn't
expose) by giving the answering LLM an EXPLICIT retrieval-confidence weight per passage,
instead of six equal-looking, unweighted chunks of text.

    query -> production_method's pipeline (fuse + rerank_scores) -> top k, WITH scores kept
          -> softmax-normalize the kept scores into weights (like RAG's p_eta(z|x))
          -> build context with each chunk's confidence stated explicitly, and an
             instruction telling the LLM how to use that signal
          -> answer()

Motivated directly by a spot check earlier this session: asked "why does the narrator
become so emotionally attached to Nastenka despite knowing she loves someone else,"
retrieval correctly surfaced `second_night_014` (the isolation/imagination passage —
direct support for the "loneliness" thread of the real answer) in the top 6, but the
generated answer never referenced it at all, building its whole answer from the other
five passages instead. One hypothesis: nothing in the prompt told the answering LLM that
passage was retrieval-relevant — it just looked like one more paragraph in an unweighted
list. This tests whether making that signal explicit changes what the answer draws on.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from production_method import classify_intent, fuse, rerank_scores  # noqa: E402
from retrieval import ANSWER_SYSTEM, CHAT_MODEL, _diversify, by_id  # noqa: E402
from retrieval import chunk_to_scene, client, order, scene_cards, tags  # noqa: E402

WEIGHTED_ANSWER_SYSTEM = ANSWER_SYSTEM + """

Each passage below is labeled with a RETRIEVAL CONFIDENCE percentage — how strongly the
retrieval system judged it relevant to this specific question, relative to the other
passages shown (the percentages sum to 100% across all passages here). Treat this as a
signal of how much to lean on each passage: higher-confidence passages should generally
anchor your answer more heavily. But do NOT ignore a lower-confidence passage if it
directly supports a fact or theme the question actually asks about — confidence reflects
retrieval's own certainty, not a guarantee of relevance, and it is not a substitute for
reading what each passage actually says."""


# example ─ in:  ({"a": 0.9, "b": 0.6, "c": 0.4}, temperature=0.15)
#           out: {"a": 0.83, "b": 0.12, "c": 0.05}   # softmax, sharpened by temperature
def softmax_weights(scores: dict, cand_ids: list, temperature: float = 0.15) -> dict:
    vals = [scores[cid] for cid in cand_ids]
    m = max(vals)  # numerical stability
    exps = [math.exp((v - m) / temperature) for v in vals]
    total = sum(exps)
    return {cid: e / total for cid, e in zip(cand_ids, exps)}


# example ─ in:  ("white_nights_second_night_014", 0.83)
#           out: "- RETRIEVAL CONFIDENCE: 83%\n  section 14 | ...\n  PASSAGE: ...\n  following: ..."
def _weighted_chunk_block(cid: str, weight: float) -> str:
    c = by_id[cid]
    t = tags.get(cid, {})
    prev = by_id.get(c["prev_id"]) if c["prev_id"] else None
    nxt = by_id.get(c["next_id"]) if c["next_id"] else None
    return (
        f"- RETRIEVAL CONFIDENCE: {round(weight * 100)}%\n"
        f"  section {c['section']} | speaking_voice={t.get('speaking_voice')}; "
        f"speaker_relation={t.get('speaker_relation')}; "
        f"narrative_relation={t.get('narrative_relation')}; themes={t.get('canonical_themes')}\n"
        f"  preceding: ...{prev['text'][-160:] if prev else '(none)'}\n"
        f"  PASSAGE: {c['text']}\n"
        f"  following: {nxt['text'][:160] if nxt else '(none)'}..."
    )


# example ─ in:  (["white_nights_second_night_014", ...], {"white_nights_second_night_014": 0.42, ...})
#           out: "## Second Night | scene_10\nscene summary: ...\n- RETRIEVAL CONFIDENCE: 42%\n..."
def build_weighted_context(top_ids: list, weights: dict) -> str:
    order_scenes, by_scene = [], {}
    for cid in top_ids:
        s = chunk_to_scene.get(cid, "")
        if s not in by_scene:
            by_scene[s] = []
            order_scenes.append(s)
        by_scene[s].append(cid)

    blocks = []
    for s in order_scenes:
        card = scene_cards.get(s, {})
        chapter = by_id[by_scene[s][0]]["chapter"]
        header = f"## {chapter} | {s}\nscene summary: {card.get('summary', '(none)')}"
        chunk_blocks = "\n".join(_weighted_chunk_block(cid, weights[cid]) for cid in by_scene[s])
        blocks.append(header + "\n" + chunk_blocks)
    return "\n\n".join(blocks)


# example ─ in:  "Why does the narrator become so emotionally attached to Nastenka..."
#           out: (["white_nights_third_night_002", ...], {"white_nights_third_night_002": 0.31, ...},
#                 {"intent": "MOTIVATION"})
def production_weighted_retrieve(query: str, k: int = 6, pool: int = 20, per_scene: int = 2,
                                  temperature: float = 0.15):
    intent = classify_intent(query)
    fused = fuse(query, intent)
    pool_ids = sorted(order, key=lambda c: fused[c], reverse=True)[:pool]

    scores = rerank_scores(query, pool_ids, fused)
    reranked = sorted(pool_ids, key=lambda c: scores[c], reverse=True)
    ids = _diversify(reranked, k, per_scene)

    weights = softmax_weights(scores, ids, temperature=temperature)
    return ids, weights, {"intent": intent}


def production_weighted_answer(query: str, k: int = 6):
    ids, weights, debug = production_weighted_retrieve(query, k=k)
    context = build_weighted_context(ids, weights)
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": WEIGHTED_ANSWER_SYSTEM + "\n\nPASSAGES:\n\n" + context},
                  {"role": "user", "content": query}],
    )
    return r.choices[0].message.content, {"ids": ids, "weights": weights, **debug}


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ans, debug = production_weighted_answer(query)
    print(f"QUESTION: {query}\nintent={debug['intent']}\n")
    for cid in debug["ids"]:
        print(f"  {cid} [{chunk_to_scene.get(cid)}] confidence={round(debug['weights'][cid]*100)}%")
    print("\nANSWER:\n" + ans)


if __name__ == "__main__":
    main()
