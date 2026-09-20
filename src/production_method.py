"""Production retrieval: cheap intent classifier -> intent-adjusted weighted fusion
(BM25 + dense + metadata) -> cheap (non-LLM) reranker -> diversify -> top-k -> LLM answer.

    QUERY
      |
      v
  cheap intent classifier          <- classify_intent() : regex rules, no API call
      |
      v
  intent-adjusted retrieval
    /      |       \\
  BM25   dense   metadata           <- each scored + normalized over the WHOLE corpus,
    \\      |       /                  then blended by the weights for the classified intent
      fusion                        <- fuse()
      |
      v
  cheap reranker                    <- cheap_rerank() : a literal-keyword-precision bonus
      |                                on top of the fused score. NOT an LLM call, on
      v                                purpose: R1/R2/R3 in findings_retrieval.md all found
    TOP-K                              the LLM judge demotes correct evidence for exactly
      |                                the interpretive/evaluative questions this pipeline
      v                                is supposed to help with.
  LLM answer                        <- reuses retrieval.answer()

Why a separate file rather than another retrieval.py preset: this isn't a fixed bundle of
on/off toggles like PRESETS, it's a genuinely different retrieval mechanism (continuous
weighted score fusion instead of rank fusion / a boost / a single LLM pick), driven by a
classifier with its own tunable parameters (weight tables, regex rules) that get tuned
through the experiment rounds logged in experiments/findings_retrieval.md (R4).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import (_bm25, _content_overlap, _diversify, _tok, answer, by_id,  # noqa: E402
                        chunk_to_scene, collection, embed, order)

# ---------------------------------------------------------------- 1. cheap intent classifier
# Pure regex, no API call — this is what "cheap" means in the diagram. Order matters: more
# specific/confident patterns are checked first so a question matching several categories
# (e.g. "how does the narrator FIRST meet Nastenka" has both an event verb AND a temporal
# word) lands on the more informative one.
INTENTS = ["INTERPRETIVE", "COMPARATIVE_MULTI_HOP", "MOTIVATION", "EVENT",
           "CHARACTER_RELATIONSHIP", "TEMPORAL", "LEXICAL_FACT", "GENERAL"]

# Narrowed after round 0 (see experiments/findings_retrieval.md R4): originally "why does"
# lived here too, but that swept in plain character-motivation questions ("why does
# Nastenka make him promise X") that behave like EVENT (strong lexical overlap with their
# answer passage) — not like a whole-book judgment question. INTERPRETIVE is now reserved
# for questions asking whether the BOOK endorses/undercuts an idea.
_INTERPRETIVE_PATTERNS = [
    r"\bdoes (the book|this|it|the narrative|the story) (endorse|support|undercut|undermine|suggest|present|signal)\b",
    r"\bhow should we understand\b", r"\bhow (are we|is the reader) meant to\b",
    r"\bis (this|that|it) (genuine|real|wisdom|sincere|secure)\b",
    r"\bpresented as\b",  # "is X presented as secure/genuine/..." — the framing word matters more than what follows
    r"\bwhat does (it|this|the book) (mean|say about|suggest)\b",
    r"\b(undermin\w*|endors\w*|symboliz\w*|foreshadow\w*)\b",
]
# Character-motivation questions: "why does X do Y" — kept separate from INTERPRETIVE
# above because round 0 showed these often have STRONG lexical overlap with their answer
# (the passage literally contains the action + the reason), so they want EVENT-like
# weights, not INTERPRETIVE's low-bm25 weights. Anchored to the start of the question so
# it doesn't accidentally catch "on what grounds" appearing mid-question elsewhere (that
# stays EVENT, e.g. "what does the narrator convince Nastenka to do... on what grounds").
_MOTIVATION_PATTERNS = [
    r"^why does\b", r"^why did\b", r"^why would\b",
    r"\bwhat (is|was) \w+('s)? motivation\b",
]
_COMPARATIVE_PATTERNS = [
    r"\bcompare[ds]?\b", r"\bcomparison\b", r"\bdiffer(s|ing|ent|ence)?\b", r"\bmore than\b",
    r"\bless than\b", r"\bwhereas\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bbetter\b.*\b(than|or)\b",
]
_EVENT_VERBS = [
    "meet", "meets", "met", "leave", "leaves", "left", "arrive", "arrives", "arrived",
    "marry", "marries", "married", "die", "dies", "died", "kiss", "kisses", "kissed",
    "cry", "cries", "cried", "write", "writes", "wrote", "promise", "promises", "promised",
    "confess", "confesses", "confessed", "return", "returns", "returned",
    "rescue", "rescues", "rescued", "save", "saves", "saved",
    "decide", "decides", "decided", "choose", "chooses", "chose", "happen", "happens", "happened",
    "propose", "proposes", "proposed", "part", "parts", "parted", "convince", "convinces", "convinced",
    "reveal", "reveals", "revealed", "resolve", "resolves", "resolved",
]
_CHARACTER_NAMES = {"nastenka", "lodger", "grandmother", "fyokla", "fekla", "matrona"}
_RELATIONAL_WORDS = [r"\bfeel(s)? about\b", r"\brelationship\b", r"\bthink(s)? of\b",
                      r"\bfeelings for\b", r"\bcare(s)? about\b", r"\btrust\b"]
_TEMPORAL_PATTERNS = [
    r"\bbefore\b", r"\bafter\b", r"\blater\b", r"\bfirst\b", r"\bagain\b",
    r"\beventually\b", r"\buntil\b", r"\bsince\b", r"\bthen\b", r"\bfinally\b",
]
_LEXICAL_STARTS = [r"^who\b", r"^what is\b", r"^what was\b", r"^which\b", r"^where\b", r"^when was\b"]


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: "EVENT"   (event verb "meet" outranks the incidental temporal word "first")
def classify_intent(query: str) -> str:
    q = query.lower()
    if any(re.search(p, q) for p in _INTERPRETIVE_PATTERNS):
        return "INTERPRETIVE"
    if any(re.search(p, q) for p in _COMPARATIVE_PATTERNS):
        return "COMPARATIVE_MULTI_HOP"
    if any(re.search(p, q) for p in _MOTIVATION_PATTERNS):
        return "MOTIVATION"
    if any(re.search(rf"\b{v}\b", q) for v in _EVENT_VERBS):
        return "EVENT"
    names = sum(1 for n in _CHARACTER_NAMES if n in q)
    if names >= 1 and any(re.search(p, q) for p in _RELATIONAL_WORDS):
        return "CHARACTER_RELATIONSHIP"
    if names >= 2:
        return "CHARACTER_RELATIONSHIP"
    if any(re.search(p, q) for p in _TEMPORAL_PATTERNS):
        return "TEMPORAL"
    if any(re.search(p, q) for p in _LEXICAL_STARTS):
        return "LEXICAL_FACT"
    return "GENERAL"


# ---------------------------------------------------------------- 2. intent -> weights
# Starting points: the three the user specified (event, lexical, character_relationship),
# plus a reasoned starting guess for the rest -- ALL tuned against the 10-question gold set
# across the rounds logged in experiments/findings_retrieval.md R4. This dict IS the tuning
# record's final state; experiments/tune_production_weights.py is what produced it.
WEIGHTS = {
    "EVENT":                  {"bm25": 0.2, "dense": 0.6, "metadata": 0.2},
    "LEXICAL_FACT":           {"bm25": 0.7, "dense": 0.2, "metadata": 0.1},
    "CHARACTER_RELATIONSHIP": {"bm25": 0.2, "dense": 0.4, "metadata": 0.4},
    # round 1 (findings_retrieval.md R4): q02's gold had strong dense signal (rank 17/84)
    # but near-zero bm25/metadata — raised dense, cut metadata which was contributing noise.
    "TEMPORAL":               {"bm25": 0.25, "dense": 0.55, "metadata": 0.20},
    "COMPARATIVE_MULTI_HOP":  {"bm25": 0.15, "dense": 0.45, "metadata": 0.40},
    # round 1: q08's gold was strong in dense (rank 20/84) and weak elsewhere — pushed
    # further toward dense. q05 (also INTERPRETIVE) is strong on ALL three signals
    # (ranks 1-3), so this is safe to adjust without risking that case.
    "INTERPRETIVE":           {"bm25": 0.1, "dense": 0.6, "metadata": 0.3},
    # round 1, new: split off from INTERPRETIVE. "why does X do Y" questions had strong
    # LEXICAL overlap with their answer in round 0 (q03's gold was bm25 rank 1/84!) —
    # INTERPRETIVE's low bm25 weight was actively burying an easy, exact-match answer.
    "MOTIVATION":             {"bm25": 0.5, "dense": 0.35, "metadata": 0.15},
    "GENERAL":                {"bm25": 0.35, "dense": 0.45, "metadata": 0.20},
}


# ---------------------------------------------------------------- 3. weighted fusion
# example ─ in:  [0.3, 5.1, 0.0, 2.2]
#           out: [0.06, 1.0, 0.0, 0.43]   # min-max scaled to [0, 1]
def _minmax(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


# example ─ in:  ("How does the narrator first meet Nastenka?", "EVENT")
#           out: {"white_nights_first_night_009": 0.71, ...}   # one score per chunk, all 84
def fuse(query: str, intent: str) -> dict:
    """Score every chunk in the corpus on BM25 / dense / metadata, min-max normalize each
    signal independently, then blend by the classified intent's weights. Continuous score
    fusion (not rank fusion like RRF) so the weights actually control the blend directly."""
    w = WEIGHTS.get(intent, WEIGHTS["GENERAL"])

    bm25_raw = _bm25.get_scores(_tok(query))
    bm25_norm = dict(zip(order, _minmax(list(bm25_raw))))

    qv = embed(query)
    dense_res = collection.query(query_embeddings=[qv], n_results=len(order), include=["distances"])
    dist_by_id = dict(zip(dense_res["ids"][0], dense_res["distances"][0]))
    dists = [dist_by_id[cid] for cid in order]
    dense_sim_norm = dict(zip(order, _minmax([-d for d in dists])))  # lower distance -> higher score

    meta_raw = [_content_overlap(query, cid) for cid in order]
    meta_norm = dict(zip(order, _minmax(meta_raw)))

    return {
        cid: w["bm25"] * bm25_norm[cid] + w["dense"] * dense_sim_norm[cid] + w["metadata"] * meta_norm[cid]
        for cid in order
    }


# ---------------------------------------------------------------- 4. cheap (non-LLM) reranker
_RERANK_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "by", "with",
                "is", "are", "was", "were", "does", "do", "did", "he", "she", "it", "this", "that",
                "what", "who", "which", "how", "why", "when", "where"}


# example ─ in:  ("What does Nastenka want?", ["white_nights_fourth_night_011", ...], fused)
#           out: {"white_nights_fourth_night_011": 0.812, ...}   the actual numeric score per
#                candidate — exposed separately from cheap_rerank() so callers that need the
#                real number (not just the sorted order), e.g. to build confidence-weighted
#                context, can get it without recomputing the formula themselves
def rerank_scores(query: str, cand_ids: list, fused: dict, bonus_weight: float = 0.15) -> dict:
    qwords = set(_tok(query)) - _RERANK_STOP
    if not qwords:
        return {cid: fused[cid] for cid in cand_ids}

    def score(cid: str) -> float:
        text_words = set(_tok(by_id[cid]["text"]))
        precision = len(qwords & text_words) / len(qwords)
        return fused[cid] + bonus_weight * precision

    return {cid: score(cid) for cid in cand_ids}


# example ─ in:  ("What does Nastenka want?", ["white_nights_fourth_night_011", ...], fused)
#           out: the same ids re-ordered by fused_score + a literal-keyword-precision bonus
def cheap_rerank(query: str, cand_ids: list, fused: dict, bonus_weight: float = 0.15) -> list:
    """No LLM call — deterministic. Adds a small bonus for candidates that literally
    contain the question's own content words, on top of the fused score. This exists
    specifically to NOT repeat the LLM-judge demotion bug documented in
    experiments/findings_retrieval.md R1-C / R2 / R3: a formula can't second-guess itself
    into preferring a topically-similar-but-wrong passage the way the judge did."""
    scores = rerank_scores(query, cand_ids, fused, bonus_weight)
    return sorted(cand_ids, key=lambda cid: scores[cid], reverse=True)


# ---------------------------------------------------------------- 5. the pipeline
# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"intent": "EVENT", "weights": {"bm25": 0.2, "dense": 0.6, "metadata": 0.2}})
def production_retrieve(query: str, k: int = 6, pool: int = 20, per_scene: int = 2):
    """QUERY -> classify_intent -> fuse (weighted BM25+dense+metadata) -> top pool ->
    cheap_rerank -> diversify -> top k."""
    intent = classify_intent(query)
    fused = fuse(query, intent)
    top_pool = sorted(order, key=lambda cid: fused[cid], reverse=True)[:pool]
    reranked = cheap_rerank(query, top_pool, fused)
    ids = _diversify(reranked, k, per_scene)
    return ids, {"intent": intent, "weights": WEIGHTS.get(intent, WEIGHTS["GENERAL"])}


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (grounded prose answer, {"intent": "EVENT", ...})
def production_answer(query: str, k: int = 6):
    ids, debug = production_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = production_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']} weights={debug['weights']}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
