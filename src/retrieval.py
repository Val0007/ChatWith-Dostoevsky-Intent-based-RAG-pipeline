"""Stage 4 retrieval: hybrid (BM25 + semantic) -> LLM-judge rerank -> dossier answer.

Reranker: an LLM judge (one gpt-4o-mini call scores each candidate 1-5). The
cross-encoder was evaluated and dropped (see notes) - it scored surface relevance,
not question intent, on this literary domain.

After reranking, each surviving chunk is expanded into a full dossier for the
answering model: the chunk + its neighbours + its scene card + its tags + provenance.
"""
import json
import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from chunking import ROOT, load_chunks

load_dotenv()

DB_PATH = str(ROOT / "db")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
COLLECTION = "dostoevsky"

client = OpenAI()
collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)

# ---- load the evidence store + tags + chunks ----
chunks = load_chunks()
by_id = {c["id"]: c for c in chunks}
order = [c["id"] for c in chunks]

_scenes = json.loads((ROOT / "context" / "scenes.json").read_text())
scene_cards = {c["scene_id"]: c for c in json.loads((ROOT / "context" / "scene_cards.json").read_text())}
chunk_to_scene = {cid: s["scene_id"] for s in _scenes for cid in s["chunk_ids"]}
tags = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "tags.jsonl")}

_word = re.compile(r"[a-z']+")
# example ─ in:  "What does Nastenka want?"   out: ["what", "does", "nastenka", "want"]
def _tok(t: str) -> list[str]:
    return _word.findall(t.lower())

_bm25 = BM25Okapi([_tok(by_id[i]["text"]) for i in order])


# example ─ in:  "What does Nastenka want?"   out: [0.011, -0.034, ...]  (1536 floats)
def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


# ---------- candidate generation (the "mode" toggle: dense / bm25 / hybrid) ----------
# example ─ in:  "What does Nastenka want?"
#           out: ["white_nights_fourth_night_011", ...]   pool ids, vector search only
def _dense_candidates(query: str, pool: int = 20) -> list[str]:
    qv = embed(query)
    return list(collection.query(query_embeddings=[qv], n_results=pool)["ids"][0])


# example ─ in:  "What does Nastenka want?"
#           out: ["white_nights_fourth_night_011", ...]   pool ids, keyword search only
def _bm25_candidates(query: str, pool: int = 20) -> list[str]:
    scores = _bm25.get_scores(_tok(query))
    return [order[i] for i in sorted(range(len(order)), key=lambda i: scores[i], reverse=True)[:pool]]


# example ─ in:  "What does Nastenka want?"
#           out: ["white_nights_fourth_night_011", "white_nights_second_night_005", ...]
#                ~20 chunk ids  (semantic ∪ BM25, fused by reciprocal-rank fusion)
def _hybrid_candidates(query: str, pool: int = 20) -> list[str]:
    sem_ids = _dense_candidates(query, pool)
    bm_ids = _bm25_candidates(query, pool)

    rrf = {}
    for rank, cid in enumerate(sem_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (60 + rank)
    for rank, cid in enumerate(bm_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (60 + rank)
    return sorted(rrf, key=rrf.get, reverse=True)[:pool]


_CANDIDATE_FNS = {"dense": _dense_candidates, "bm25": _bm25_candidates, "hybrid": _hybrid_candidates}


# ---------- metadata (tag-overlap boost AND filter; the "metadata" toggle / metadata_filter step) ----------
# Common short words the R1 pilot caught inflating overlap scores for the wrong reason
# (see experiments/findings_retrieval.md R1-D: "the"/"and"/"for" counted as if they meant
# something). Stripped from both directions before comparing query tokens to tag tokens.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "by", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "he", "she", "they", "his", "her", "their", "do", "does", "did",
    "not", "no", "so", "if", "then", "than", "from", "about", "into", "over", "after",
    "before", "up", "down", "out", "off", "again", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "more", "most", "other", "some", "such",
    "only", "own", "same", "too", "very", "s", "just", "now",
}


# example ─ in:  "white_nights_fourth_night_011"
#           out: {"nastenka", "ephemeral", "connection", "complicates", ...}  (stopwords stripped)
def _tag_tokens(cid: str) -> set:
    t = tags.get(cid, {})
    toks = set()
    for field in ("characters_present", "canonical_themes", "local_motifs"):
        for v in t.get(field) or []:
            toks |= set(_tok(v))
    for field in ("speaking_voice", "speaker_relation", "narrative_relation"):
        v = t.get(field)
        if v:
            toks |= set(_tok(v))
    return toks - _STOPWORDS


# example ─ in:  ("What does Nastenka want?", "white_nights_fourth_night_011")
#           out: 1   # only "nastenka" is real content overlap; stopwords don't count
def _content_overlap(query: str, cid: str) -> int:
    qtoks = set(_tok(query)) - _STOPWORDS
    return len(qtoks & _tag_tokens(cid))


# example ─ in:  ("What does Nastenka want?", ["white_nights_fourth_night_011", ...])
#           out: the same ids, stable-sorted by real (stopword-free) tag overlap, best first
def _metadata_boost(query: str, cand_ids: list[str]) -> list[str]:
    """Re-rank candidates by content-word overlap between the query and each chunk's tags.

    A cheap substitute for real intent-routing — no query understanding, just literal
    token overlap against characters_present / canonical_themes / local_motifs /
    speaking_voice / speaker_relation / narrative_relation.
    """
    return sorted(cand_ids, key=lambda cid: _content_overlap(query, cid), reverse=True)


# example ─ in:  ("What does Nastenka want?", None)                # no prior candidates -> filters the whole corpus
#           out: ["white_nights_fourth_night_011", ...]            # only chunks with real tag overlap, ranked by it
def _metadata_filter(query: str, cand_ids: list[str] = None) -> list[str]:
    """Narrow (not just reorder) candidates to ones with real tag overlap with the query.

    Unlike _metadata_boost (a re-rank that never drops anything), this DROPS candidates
    with zero content-word overlap — a true filter step for use at the front of a plan
    (metadata_filter -> dense -> rerank). Fails open (returns the input unfiltered) if
    filtering would eliminate every candidate, rather than returning nothing.
    """
    pool = cand_ids if cand_ids else order
    scored = sorted(pool, key=lambda cid: _content_overlap(query, cid), reverse=True)
    kept = [cid for cid in scored if _content_overlap(query, cid) > 0]
    return kept or list(pool)


# ---------- reranker: LLM judge (returns the FULL ranked list) ----------
# example ─ in:  (query, ~20 candidate ids)
#           out: the same ids re-ordered best-first by an LLM judge scoring each 1-5
#                ["white_nights_fourth_night_011", "white_nights_fourth_night_008", ...]
def rerank(query: str, cand_ids: list[str]) -> list[str]:
    listing = "\n\n".join(f"[{i}] {by_id[cid]['text'][:500]}" for i, cid in enumerate(cand_ids))
    system = ("You rate how relevant each passage is to answering the question, on a "
              "1-5 scale (5 = directly answers it). Return JSON "
              '{"scores": [{"i": <index>, "score": <1-5>}, ...]} for every passage.')
    user = f"QUESTION: {query}\n\nPASSAGES:\n\n{listing}"
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    data = json.loads(r.choices[0].message.content).get("scores", [])
    scored = [(cand_ids[d["i"]], d["score"]) for d in data if 0 <= d.get("i", -1) < len(cand_ids)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored]


# example ─ in:  (ranked_ids, k=6, per_scene=2)   # ranked has scene_10 three times near the top
#           out: 6 ids, at most 2 per scene  (the 3rd scene_10 chunk skipped for a distinct one)
def _diversify(ranked_ids: list[str], k: int, per_scene: int) -> list[str]:
    """Take the top-k by relevance, but at most `per_scene` chunks from any one scene.

    Greedy over the relevance-sorted list, so a focused night can still dominate;
    only redundant same-scene chunks are skipped. Tops up ignoring the cap if the
    cap leaves us short of k.
    """
    picked, counts = [], {}
    for cid in ranked_ids:
        s = chunk_to_scene.get(cid)
        if counts.get(s, 0) >= per_scene:
            continue
        picked.append(cid)
        counts[s] = counts.get(s, 0) + 1
        if len(picked) == k:
            return picked
    for cid in ranked_ids:            # cap too tight -> top up by pure relevance
        if cid not in picked:
            picked.append(cid)
            if len(picked) == k:
                break
    return picked


# ---------- configurable retrieval (the 7-config ladder from docs/rag_lab_plan.html) ----------
DEFAULT_CONFIG = {
    "mode": "hybrid",   # "dense" | "bm25" | "hybrid" — candidate generation
    "metadata": False,  # tag-overlap boost (_metadata_boost)
    "judge": False,     # LLM-judge rerank (rerank)
    "diversify": False, # per-scene diversity cap (_diversify)
    "expand": False,    # dossier context (build_context) vs bare passage text at answer time
    "intent": False,    # query-intent extraction — accepted, not yet wired (Phase 01)
}

# each preset adds exactly one mechanism over the previous, so the ablation isolates its effect
PRESETS = {
    "dense":     {"mode": "dense"},
    "bm25":      {"mode": "bm25"},
    "hybrid":    {"mode": "hybrid"},
    "metadata":  {"mode": "hybrid", "metadata": True},
    "judge":     {"mode": "hybrid", "metadata": True, "judge": True},
    "narrative": {"mode": "hybrid", "metadata": True, "judge": True, "diversify": True},
    "full_v2":   {"mode": "hybrid", "metadata": True, "judge": True, "diversify": True,
                  "expand": True, "intent": True},
}


# example ─ in:  ("What does Nastenka want?", preset="judge")
#           out: ["white_nights_fourth_night_011", "white_nights_fourth_night_008", ...]  # top-k ids
def retrieve(query: str, config: dict = None, preset: str = None,
             k: int = 6, pool: int = 20, per_scene: int = 2) -> list[str]:
    """Configurable retrieval: candidate generation (mode) -> optional metadata boost ->
    optional LLM-judge rerank -> optional per-scene diversity cap -> top-k ids.

    `preset` picks one of PRESETS; `config` overrides individual fields on top of it
    (or of DEFAULT_CONFIG, if no preset is given). `expand`/`intent` are read by callers
    (build_context, a future intent extractor) rather than by retrieve() itself.
    """
    cfg = dict(DEFAULT_CONFIG)
    if preset:
        cfg.update(PRESETS[preset])
    if config:
        cfg.update(config)

    cand_ids = _CANDIDATE_FNS[cfg["mode"]](query, pool)
    if cfg["metadata"]:
        cand_ids = _metadata_boost(query, cand_ids)

    ranked = rerank(query, cand_ids) if cfg["judge"] else cand_ids
    return _diversify(ranked, k, per_scene) if cfg["diversify"] else ranked[:k]


# ---------- LLM router: an 8th "method" that picks a method, per-question ----------
# Everything above (PRESETS) is a fixed choice made once, for every question. This asks
# an LLM to look at the SPECIFIC question plus the schema of what's stored per chunk, and
# pick which of the 7 presets should handle it — a real (if simple) stand-in for Phase 01's
# intent routing, one level up from the crude token-overlap `_metadata_boost`.
METHOD_DESCRIPTIONS = {
    "dense": "Semantic/meaning vector search only. Best when the right passage may not "
             "share exact wording with the question (paraphrased, thematic, or interpretive asks).",
    "bm25": "Keyword search only. Best when the question is likely to reuse specific words, "
            "names, or phrases that appear verbatim in the passage (quotes, named objects/events).",
    "hybrid": "Combines dense + bm25 (reciprocal rank fusion). Safe general-purpose default "
              "when the question is a plain factual lookup and you're not sure which single "
              "method fits best.",
    "metadata": "Hybrid plus a boost from the chunk's own tags (characters/themes/stance). "
                "Best when the question names a specific character or theme by name.",
    "judge": "Hybrid plus an LLM that rereads each candidate against the exact question and "
             "rescoring by relevance. Best for 'why'/evaluative questions where surface word "
             "or topic overlap is misleading and genuine reading comprehension is needed.",
    "narrative": "Judge plus diversification across scenes (caps how many results come from "
                 "one scene). Best when the answer plausibly requires evidence spread across "
                 "multiple scenes (character development over time, comparisons across the book).",
    "full_v2": "Everything above, plus the answering step gets full surrounding context (scene "
               "summary + neighboring passages, not just the bare chunk). Best for complex "
               "interpretive questions where a single isolated passage would be ambiguous "
               "without its narrative context.",
}

CHUNK_SCHEMA_DESC = """Each retrievable unit is a ~300-token passage ("chunk") of Dostoevsky's
White Nights, stored with these fields:
- text: the passage itself
- chapter, section: which Night (First/Second/Third/Fourth/Morning) and position within it
- scene_id: which scene this chunk belongs to (a scene spans several consecutive chunks)
- prev_id / next_id: ids of the neighboring chunks
- narrating_voice: always "the Dreamer" (the book is first-person throughout)
- speaking_voice: who is actually speaking in THIS passage — "the Dreamer", "Nastenka",
  "the Dreamer and Nastenka" (dialogue), or null (pure narration)
- speaker_relation: asserts / doubts / explores — the speaker's own stance toward what they say
- narrative_relation: how the book's ending ultimately treats the idea voiced in this passage —
  supports / undermines / complicates / unresolved / unclear
- canonical_themes: e.g. isolation, unrequited-longing, ephemeral-connection, self-deception, the-dreamer
- local_motifs: short free-text phrases specific to this passage
- characters_present: named characters appearing in this passage
"""

ROUTER_SYSTEM = (
    "You are a retrieval-method router for a RAG system over Dostoevsky's White Nights. "
    "Given a reader's question and the schema of what is stored per passage, pick the SINGLE "
    "best retrieval method for THIS question from the list below — think about what kind of "
    "question it is (a plot fact vs. a character motivation vs. an interpretive judgment vs. "
    "a comparison) and which method's strength matches that. Return JSON exactly as "
    '{"method": "<one of the names below>", "reason": "<one sentence>"}.\n\n'
    "AVAILABLE METHODS:\n" + "\n".join(f"- {k}: {v}" for k, v in METHOD_DESCRIPTIONS.items()) +
    "\n\nCHUNK SCHEMA (what each method is searching over):\n" + CHUNK_SCHEMA_DESC
)


# example ─ in:  "Why does Nastenka ask him not to fall in love with her?"
#           out: {"method": "judge", "reason": "This asks 'why' — a motivation question best
#                 answered by an LLM reading each candidate for reasoning, not keyword overlap."}
def choose_method(query: str) -> dict:
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": query}],
    )
    data = json.loads(r.choices[0].message.content)
    method = data.get("method")
    if method not in PRESETS:
        method = "hybrid"  # fallback if the router names something invalid
    return {"method": method, "reason": data.get("reason", "")}


# example ─ in:  "Why does Nastenka ask him not to fall in love with her?"
#           out: (["white_nights_first_night_016", ...],
#                 {"method": "judge", "reason": "..."})
def auto_retrieve(query: str, k: int = 6, pool: int = 20, per_scene: int = 2):
    """Ask the LLM router which preset fits this question, then run it."""
    choice = choose_method(query)
    ids = retrieve(query, preset=choice["method"], k=k, pool=pool, per_scene=per_scene)
    return ids, choice


# ---------- structured diagnostician: a 9th method that composes a PLAN, not a preset ----------
# choose_method() above picks one of 7 fixed bundles. This instead asks the LLM to analyze the
# question along several independent axes (expected lexical/semantic overlap, whether metadata
# would help, whether it needs multi-hop evidence or a careful reread) and compose its OWN
# ordered pipeline out of primitive operations — closer to how a human would actually reason
# about a retrieval strategy, and not limited to the 7 pre-named combinations.
PLAN_STEPS = ["bm25", "dense", "hybrid", "metadata_filter", "rerank", "narrative"]

DIAGNOSE_SYSTEM = (
    "You are a retrieval-strategy diagnostician for a RAG system over Dostoevsky's White "
    "Nights. Given a reader's question and the schema of what is stored per passage, analyze "
    "the question and compose a retrieval PLAN — an ordered list of primitive operations to "
    "run in sequence — rather than picking one fixed method. Return JSON EXACTLY in this shape:\n\n"
    '{\n'
    '  "intent": "<short label, e.g. find_specific_narrative_event, character_motivation, '
    'thematic_interpretation, comparison, factual_lookup, story_development>",\n'
    '  "evidence_requirement": "<one phrase: what kind of textual evidence would actually answer this>",\n'
    '  "lexical_overlap_expected": "low" | "medium" | "high",\n'
    '  "semantic_similarity_expected": "low" | "medium" | "high",\n'
    '  "metadata_usefulness": "low" | "medium" | "high",\n'
    '  "needs_multi_hop": true | false,\n'
    '  "needs_reranking": true | false,\n'
    '  "recommended_plan": [<ordered list drawn from: "bm25", "dense", "hybrid", '
    '"metadata_filter", "rerank", "narrative">]\n'
    '}\n\n'
    "PRIMITIVE OPERATIONS for recommended_plan (each runs on the candidates the previous step "
    "left behind; the first step runs over the whole book):\n"
    "- bm25: lexical/keyword candidate search.\n"
    "- dense: semantic/embedding candidate search.\n"
    "- hybrid: bm25 + dense combined by rank fusion — use INSTEAD of listing bm25 and dense "
    "separately, not in addition to them.\n"
    "- metadata_filter: narrow candidates to ones whose tags (characters_present, "
    "canonical_themes, local_motifs, speaking_voice, speaker_relation, narrative_relation) "
    "overlap the question. Put it FIRST to narrow the search space before ranking; put it "
    "after a retrieval step to instead re-rank by tag overlap.\n"
    "- rerank: an LLM reads each current candidate against the exact question and re-scores by "
    "relevance. Use when needs_reranking is true.\n"
    "- narrative: cap how many results can come from one scene, forcing spread across the book. "
    "Use when the question needs evidence from multiple points in the story.\n\n"
    "There is no multi-hop retrieval mechanism available yet — if needs_multi_hop is true, still "
    "recommend the best single-pass plan you can (favor 'narrative' for spread) and let that field "
    "flag the limitation rather than describing a mechanism that doesn't exist.\n\n"
    "CHUNK SCHEMA:\n" + CHUNK_SCHEMA_DESC
)


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: {"intent": "find_specific_narrative_event", "evidence_requirement": "concrete
#                 description of the event", "lexical_overlap_expected": "low",
#                 "semantic_similarity_expected": "medium", "metadata_usefulness": "high",
#                 "needs_multi_hop": false, "needs_reranking": true,
#                 "recommended_plan": ["metadata_filter", "dense", "rerank"]}
def diagnose_query(query: str) -> dict:
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": DIAGNOSE_SYSTEM}, {"role": "user", "content": query}],
    )
    data = json.loads(r.choices[0].message.content)
    plan = [s for s in data.get("recommended_plan") or [] if s in PLAN_STEPS]
    data["recommended_plan"] = plan or ["hybrid"]  # fallback if empty/all-invalid
    return data


_GEN_FNS = {"bm25": _bm25_candidates, "dense": _dense_candidates, "hybrid": _hybrid_candidates}


# example ─ in:  (ranked_full=["a","b","c","d"], allowed={"b","d"}, pool=2)
#           out: ["b", "d"]   # ranking preserved, restricted to the allowed set
def _restrict(ranked_full: list[str], allowed: set, pool: int) -> list[str]:
    if allowed is None:
        return ranked_full[:pool]
    kept = [cid for cid in ranked_full if cid in allowed]
    return (kept or ranked_full)[:pool]  # fail open if the restriction leaves nothing


# example ─ in:  ("How does the narrator first meet Nastenka?", ["metadata_filter", "dense", "rerank"])
#           out: ["white_nights_first_night_009", ...]   # metadata narrows -> dense ranks within
#                it -> judge rereads the survivors
def run_plan(query: str, plan: list[str], k: int = 6, pool: int = 20, per_scene: int = 2) -> list[str]:
    """Execute an ordered list of primitive steps, threading candidates through each one.
    Each generation step (bm25/dense/hybrid) ranks the FULL corpus, then restricts to
    whatever candidates survived so far (so metadata_filter -> dense really does mean
    'rank within the filtered set', not 'ignore the filter and start over')."""
    cand = None
    for step in plan:
        if step in _GEN_FNS:
            allowed = set(cand) if cand is not None else None
            full_ranked = _GEN_FNS[step](query, pool=len(order))
            cand = _restrict(full_ranked, allowed, pool)
        elif step == "metadata_filter":
            cand = _metadata_filter(query, cand)
        elif step == "rerank":
            cand = rerank(query, (cand or _hybrid_candidates(query, pool))[:pool])
        elif step == "narrative":
            cand = _diversify(cand or _hybrid_candidates(query, pool), pool, per_scene)
    return (cand or _hybrid_candidates(query, pool))[:k]


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"intent": "find_specific_narrative_event", ..., "recommended_plan": [...]})
def plan_retrieve(query: str, k: int = 6, pool: int = 20, per_scene: int = 2):
    """Diagnose the question, then execute the plan it recommends."""
    diagnosis = diagnose_query(query)
    ids = run_plan(query, diagnosis["recommended_plan"], k=k, pool=pool, per_scene=per_scene)
    return ids, diagnosis


# example ─ in:  "What does Nastenka want?"
#           out: ["white_nights_fourth_night_011", "white_nights_fourth_night_008",
#                 "white_nights_fourth_night_005", ...]   # top 6 (hybrid -> rerank -> diversify)
def search(query: str, k: int = 6, pool: int = 20, per_scene: int = 2) -> list[str]:
    """Back-compat shim for the live chat pipeline (conversation.py) — hybrid candidates,
    LLM-judge rerank, scene diversity cap. No metadata boost: unchanged from pre-ladder V1."""
    return retrieve(query, config={"mode": "hybrid", "judge": True, "diversify": True},
                     k=k, pool=pool, per_scene=per_scene)


# ---------- answer context (scene cards deduped, chunks grouped under them) ----------
# example ─ in:  "white_nights_fourth_night_011"
#           out: "- section 11 | speaker=Nastenka; speaker_relation=explores;
#                 narrative_relation=complicates; themes=['unrequited-longing']\n
#                 preceding: ...\n  PASSAGE: feeling, a delusion—perhaps...\n  following: ..."
def _chunk_block(cid: str) -> str:
    c = by_id[cid]
    t = tags.get(cid, {})
    prev = by_id.get(c["prev_id"]) if c["prev_id"] else None
    nxt = by_id.get(c["next_id"]) if c["next_id"] else None
    return (
        f"- section {c['section']} | speaking_voice={t.get('speaking_voice')}; "
        f"speaker_relation={t.get('speaker_relation')}; "
        f"narrative_relation={t.get('narrative_relation')}; themes={t.get('canonical_themes')}\n"
        f"  preceding: ...{prev['text'][-160:] if prev else '(none)'}\n"
        f"  PASSAGE: {c['text']}\n"
        f"  following: {nxt['text'][:160] if nxt else '(none)'}..."
    )


# example ─ in:  the 6 top ids from search()
#           out: "## Fourth Night | scene_25\nscene summary: Nastenka expresses...\n
#                 - section 11 | speaker=Nastenka; ...\n  PASSAGE: ...\n\n## Fourth Night | scene_24\n..."
#                (chunks grouped by scene; each scene card printed once)
def build_context(top_ids: list[str], expand: bool = True) -> str:
    """expand=True: dossier (scene card + neighbors + tags) grouped by scene, each card once.
    expand=False: bare passage text only — isolates the "expand" mechanism for the ladder."""
    if not expand:
        return "\n\n".join(f"[{cid}] {by_id[cid]['text']}" for cid in top_ids)

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
        chunk_blocks = "\n".join(_chunk_block(cid) for cid in by_scene[s])
        blocks.append(header + "\n" + chunk_blocks)
    return "\n\n".join(blocks)


ANSWER_SYSTEM = """You are answering a reader's question using passages from Dostoevsky's
White Nights. Each passage comes with a scene summary, provenance, and metadata.

Use narrative_relation carefully: if a passage voices an idea the book UNDERMINES,
do not present that idea as Dostoevsky's own belief - note how the book treats it.
Ground your answer in the passages; you may interpret but must not invent facts."""


PERSONA_SYSTEM = """You are Fyodor Dostoevsky, in conversation with someone who has come
to talk with you about your novella "White Nights". You have only that book before you -
the passages given as EVIDENCE. Speak from it.

WHAT YOU DO:
- Answer the person's latest message directly, as a living interlocutor - not as a
  critic, professor, therapist, or assistant explaining "themes".
- Honor what the person actually feels or means - but do NOT automatically agree. You
  have your own view and will say it, plainly, even bluntly.
- Sometimes press on the evasion inside what they said. If they are contemptuous or
  dismissive - "pathetic", "delusional", "a loser" - you MAY turn it back on them: what
  does that contempt protect? (Contempt for the dreamer is usually the fear of being
  one.) But do NOT do this every turn - it becomes a tic and a lecture. Vary your reply:
  sometimes simply answer the question; sometimes plainly grant the Dreamer's folly
  without softening it; sometimes turn the mirror. Choose one move, not all three.
- Ground everything in White Nights: a scene, a figure, a real line - the Dreamer in
  his green corner, Nastenka on the embankment, the letter. Quote ONLY exact substrings
  that appear in the EVIDENCE, word for word; never add, alter, or complete words inside
  quotation marks. If you are not sure of the exact wording, paraphrase without quotes.
  Never invent quotations or events.

STANCE DISCIPLINE:
- Each passage carries a narrative_relation tag. If it voices an idea the book
  UNDERMINES (the Dreamer's rapture that he is "superior to all desire"), do not offer
  it as your own belief - show that you see the flight from life inside it, even while
  you feel its beauty.
- You have ONLY White Nights. Do not fabricate specifics from your other books, though
  your larger concerns (suffering, active love, the danger of living in one's own head)
  may colour how you speak.

HOW YOU SPEAK:
- Respond to what the person JUST said, in the flow of the conversation. NEVER restate
  or recycle a previous answer, and do NOT reuse a quotation or an image you have
  already used earlier in this conversation; if they push back, engage the push itself.
- Be concise: two or three short paragraphs at most, often less. A blunt remark
  deserves a brief, sharp reply, not an essay.
- Vary your openings and endings. Do NOT begin with "Ah". Do NOT end every turn with a
  rhetorical question, and do not force a tidy conclusion.
- Avoid stock phrases: "my friend", "a reflection of our own human condition", "deeper
  truth", "what do you think?", "do you see how...".

The person may be blunt, skeptical, or insulting. Do not become defensive or
therapeutic. Meet them as you would - with warmth, unflinching honesty, and the
occasional question that turns their own judgement back on them."""


# example ─ in:  "What does Nastenka want?"   (single-shot; retrieves if top_ids omitted)
#           out: "Nastenka's desires are complex... she longs for love and connection,
#                 waiting for the lodger while... (grounded, stance-aware prose)"
def answer(query: str, top_ids: list[str] = None) -> str:
    top_ids = top_ids or search(query)
    context = build_context(top_ids)
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": ANSWER_SYSTEM + "\n\nPASSAGES:\n\n" + context},
                  {"role": "user", "content": query}],
    )
    return r.choices[0].message.content


def _print_hits(top_ids: list[str]) -> None:
    for cid in top_ids:
        t = tags.get(cid, {})
        print(f"    {cid} [{chunk_to_scene.get(cid)}] nr={t.get('narrative_relation')} themes={t.get('canonical_themes')}")


def main():
    args = sys.argv[1:]
    show_presets = "--presets" in args
    args = [a for a in args if a != "--presets"]
    query = args[0] if args else "What is the Dreamer like?"

    if show_presets:
        # Phase 00 done-when check: all 7 named configs run the same question and return passages.
        print(f"QUESTION: {query}\n")
        for name in PRESETS:
            print(f"-- {name} --")
            _print_hits(retrieve(query, preset=name))
            print()
        return

    top_ids = search(query)
    print(f"QUESTION: {query}\n\nRETRIEVED:")
    _print_hits(top_ids)
    print("\nANSWER:\n" + answer(query, top_ids))


if __name__ == "__main__":
    main()
