"""HyDE (Hypothetical Document Embeddings) → union IDs, constrained to be corpus-agnostic.

    original query
      │
      ├──→ classify_intent(original)                          (ORIGINAL grammar only)
      ├──→ generate_hyde_passage(original)   ← LLM writes a hypothetical PASSAGE (not a
      │        │                                reworded question) that could plausibly
      │        │                                answer the query, embedded for answer-to-
      │        │                                answer similarity instead of question-to-
      │        │                                passage — HyDE's actual mechanism, per
      │        │                                Gao et al.'s RAG survey (2312.10997 §III).
      │        ▼
      │    dense-only retrieval on the HYPOTHETICAL passage's embedding  ← top-N
      │
      └──→ fuse(original, intent)  ← ORIGINAL query's own top-N (BM25+dense+metadata)
      │
      ▼
    UNION IDs (dedup — same "union, not substitution" principle as R6/R11)
      │
      ▼
    cheap_rerank(ORIGINAL query, union, fuse(original)-anchored score)   ← R11's fix,
      │                                                                    kept: never
      ▼                                                                    blend/max
    diversify → top k                                                     across sources

Why the HyDE prompt is constrained, unlike the technique's usual form: vanilla HyDE
deliberately EXPLOITS an LLM's parametric knowledge — a fluent, plausible-sounding
hypothetical answer embeds closer to real relevant passages than a bare question does,
even when its specific facts are wrong. But this model plainly HAS memorized White
Nights (a famous public-domain text), so letting it write "what Nastenka probably says"
would test "does the model already know this book," not "does HyDE work as a general,
corpus-agnostic technique." The prompt below borrows the same anti-hallucination GLOBAL
RULES from production_union_with_llm.py's alternative-query prompt (R12) — assume no
knowledge of the source, never invent entities/events/wording beyond what the query
itself states or implies — adapted for writing a passage instead of a query. This is a
genuinely harder test for HyDE than the technique is normally given: it can only lean on
the query's own content and generic narrative FORM, never confirmed content it happens
to already know.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_method import classify_intent, cheap_rerank, fuse  # noqa: E402
from retrieval import CHAT_MODEL, _dense_candidates, _diversify  # noqa: E402
from retrieval import answer, chunk_to_scene, client, order  # noqa: E402

HYDE_SYSTEM = """You are generating a HYPOTHETICAL PASSAGE for a retrieval system, using
the HyDE (Hypothetical Document Embeddings) technique.

Given a user's question, write a SHORT passage that could plausibly appear in the source
document and would answer that question — not because you know what the source document
actually says, but as a plausible reconstruction grounded ONLY in the question itself.
This passage is never shown to anyone and never asserted as fact — it exists purely so
its embedding lands closer, in meaning-space, to what a real answering passage would look
like than the bare question does.

GLOBAL RULES (same standard as this system's query-rewriting component):
* Treat the user's question as the ONLY source of truth.
* Assume you have NO knowledge of the underlying document, its characters, entities,
  events, terminology, or wording — even if you recognize the source or believe you know
  the answer. Do not use pretrained knowledge, world knowledge, literary knowledge, or any
  assumption about what the source text probably says.
* Do not invent specific facts, named entities, dialogue, settings, or events beyond what
  the question already names or clearly, directly implies.
* Preserve every concrete noun, named entity, and distinctive term from the question
  exactly as given — never substitute a guessed alternative.
* You MAY use generic, plausible narrative FORM (e.g. if asked "how does X meet Y", a
  meeting scene plausibly involves description and brief dialogue) as long as the CONTENT
  stays strictly grounded in the question's own wording — form, not invented substance.
* When the question is ambiguous or underspecified, keep the passage general rather than
  guessing specifics to fill the gap.
* Write it AS IF IT WERE the source passage itself — first-person or third-person prose
  matching the question's own implied voice, not a summary, analysis, or answer to the
  question.
* Do not answer the question analytically. Do not explain. Output ONLY the hypothetical
  passage text, nothing else."""


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: "I saw her weeping by the water and felt I could not simply pass by..."
def generate_hyde_passage(query: str) -> str:
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": HYDE_SYSTEM}, {"role": "user", "content": query}],
    )
    return r.choices[0].message.content.strip()


# example ─ in:  "How does the narrator first meet Nastenka?"
#           out: (["white_nights_first_night_009", ...],
#                 {"intent": "EVENT", "hyde_passage": "...", "union_size": 27})
def production_hyde_retrieve(query: str, k: int = 6, pool: int = 20, per_scene: int = 2):
    intent = classify_intent(query)  # ORIGINAL query only
    hyde_passage = generate_hyde_passage(query)

    fused_orig = fuse(query, intent)  # scores ALL 84 chunks; reused below for final ranking
    orig_ids = sorted(order, key=lambda c: fused_orig[c], reverse=True)[:pool]

    hyde_ids = _dense_candidates(hyde_passage, pool)  # HyDE is a DENSE-only technique —
                                                       # answer-to-answer embedding similarity,
                                                       # not a BM25/metadata substitute
    union_ids = list(dict.fromkeys(orig_ids + hyde_ids))

    reranked = cheap_rerank(query, union_ids, fused_orig)  # anchored on ORIGINAL query (R11)
    ids = _diversify(reranked, k, per_scene)
    return ids, {"intent": intent, "hyde_passage": hyde_passage, "union_size": len(union_ids),
                 "new_from_hyde": len(set(hyde_ids) - set(orig_ids))}


def production_hyde_answer(query: str, k: int = 6):
    ids, debug = production_hyde_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = production_hyde_retrieve(query)
    print(f"QUESTION: {query}\nintent={debug['intent']}")
    print(f"HYDE PASSAGE:\n{debug['hyde_passage']}\n")
    print(f"union_size={debug['union_size']} new_from_hyde={debug['new_from_hyde']}\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
