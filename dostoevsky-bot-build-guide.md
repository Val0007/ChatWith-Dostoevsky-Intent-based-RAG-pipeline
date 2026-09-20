# Building the Dostoevsky Bot — A Beginner's Build Guide

You know Python and FastAPI. This guide assumes you know **nothing** about RAG, embeddings, vector databases, or the OpenAI API, and explains each as you meet it.

## The one rule that matters most

**Build the dumbest version that works end-to-end first. Then improve it one layer at a time, testing after each layer.**

Do NOT start with metadata, hybrid search, and reranking. Start with: one book, chop it up, retrieve naively, generate. Get Dostoevsky to actually reply to you. That first "it works!" moment is worth more than any architecture. Then you add quality, and every time you add a piece you can tell whether it helped, because you have something working to compare against.

The stages below are in dependency order. Each stage ends with **"You'll know it worked when…"** — don't move on until that's true.

---

## Concepts you need (5-minute version)

- **LLM API call** — you send text to OpenAI's servers, you get text back. That's it. It's just an HTTP request with your API key.
- **Chunking** — splitting a book into small passages (a few hundred words each) so you can retrieve the *relevant* bits instead of stuffing a whole novel into the model.
- **Embedding** — a function that turns a piece of text into a list of ~1500 numbers (a "vector"). Texts with similar *meaning* get similar numbers. This is how the computer measures "these two passages are about the same thing" without understanding words.
- **Vector store / vector database** — a place that holds all your chunk-vectors and can answer "give me the 8 chunks whose vectors are closest to *this* query vector." That's semantic search.
- **RAG (Retrieval-Augmented Generation)** — the whole pattern: **retrieve** relevant chunks, then hand them to the LLM and ask it to **generate** an answer using them. That's all RAG is. Don't be intimidated by the acronym.

That's the entire conceptual toolkit. Everything below is just doing these carefully.

---

## Stage 0 — Setup (½ day)

1. New FastAPI project, virtualenv.
2. `pip install openai chromadb rank-bm25 tiktoken python-dotenv pydantic`
3. Get an OpenAI API key, put it in a `.env` file, load it. **Never commit this file.**
4. Write ONE tiny script that sends "Say hello" to the API and prints the reply.

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from environment

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(resp.choices[0].message.content)
```

**You'll know it worked when:** that script prints a greeting. You now know how to call the LLM. Everything else builds on this.

---

## Stage 1 — The "hello world" pipeline (1 day)

Goal: **one novel**, naive retrieval, working end to end. Ugly is fine.

1. **Get the text.** Download *White Nights* (it's short — start here, not Karamazov) from Project Gutenberg as plain text. Delete the Gutenberg header/footer by hand.

2. **Chunk it naively.** Split into ~300-word pieces. For now, literally split on paragraphs and glue paragraphs together until you hit ~300 words. Store as a list of dicts: `{"id": "wn_0001", "text": "..."}`. Don't overthink it yet.

3. **Embed every chunk** and put it in Chroma:

```python
import chromadb
client_db = chromadb.PersistentClient(path="./db")
collection = client_db.create_collection("dostoevsky")

# Chroma can embed for you, but let's be explicit so you understand it:
def embed(texts):
    r = OpenAI().embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in r.data]

collection.add(
    ids=[c["id"] for c in chunks],
    documents=[c["text"] for c in chunks],
    embeddings=embed([c["text"] for c in chunks]),
)
```

4. **Retrieve** for a question:

```python
def retrieve(query, k=6):
    qvec = embed([query])[0]
    res = collection.query(query_embeddings=[qvec], n_results=k)
    return res["documents"][0]  # list of chunk texts
```

5. **Generate.** Stuff the retrieved chunks into a prompt and ask:

```python
def answer(query):
    passages = retrieve(query)
    context = "\n\n---\n\n".join(passages)
    messages = [
        {"role": "system", "content": "You are answering using the passages below from Dostoevsky's White Nights. Passages:\n\n" + context},
        {"role": "user", "content": query},
    ]
    r = OpenAI().chat.completions.create(model="gpt-4o-mini", messages=messages)
    return r.choices[0].message.content
```

**You'll know it worked when:** you ask "What is the Dreamer like?" and get an answer that clearly used the book. It won't sound like Dostoevsky yet, and that's correct — you haven't done the persona work. You have a working RAG pipeline. This is a real milestone; most people never get here.

---

## Stage 2 — Better chunking + provenance (1 day)

Now improve the foundation. Naive 300-word splits cut mid-scene and give you useless citations.

1. **Chunk on structure**, not word count alone. Split on chapter/scene/dialogue breaks first, then within those, group into ~200–500 word pieces **without breaking sentences**.

2. **Store rich metadata** on every chunk (this is nearly free now and saves you pain later):

```python
{
  "id": "white_nights_night_1_003",
  "work": "White Nights",
  "chapter": "First Night",
  "section": 3,
  "text": "...",
  "prev_id": "white_nights_night_1_002",
  "next_id": "white_nights_night_1_004",
}
```

Chroma lets you attach this as `metadatas=[...]` in `collection.add`. The `prev_id`/`next_id` let you fetch surrounding context later. The chapter info lets you show real citations like "*White Nights*, First Night" instead of "chunk wn_003".

**You'll know it worked when:** your retrieved chunks come back with clean chapter/section info attached, and no chunk ends mid-sentence.

---

## Stage 3 — Metadata tagging (1–2 days) — the part that makes this special

This is the step that separates your project from every generic "chat with a book" bot. It solves a real problem: **Dostoevsky puts ideas in characters' mouths that he is *attacking*.** If you don't track who's speaking and whether the book endorses them, your bot will confidently say "Dostoevsky believes rational self-interest governs behavior" — the exact opposite of the truth.

1. **Write a fixed theme list by hand first** — ~15–20 canonical themes: `isolation`, `the-dreamer`, `rational-egoism`, `suffering-as-redemption`, `faith-vs-doubt`, `guilt`, `active-love`, etc. This is a design decision; make it deliberately.

2. **Run one LLM pass per chunk** that fills in this schema (force JSON output, validate with Pydantic):

```python
{
  "speaker": "narrator | <character name> | author",
  "speaker_relation": "asserts | doubts | rejects | explores",
  "narrative_relation": "supports | complicates | undermines | unresolved | unclear",
  "canonical_themes": ["isolation", "the-dreamer"],   # from your fixed list
  "local_motifs": ["wet streets", "imagined love"]     # freeform, anything notable
}
```

Two fields for stance is the key move: `speaker_relation` = what the *character* is doing with the idea; `narrative_relation` = what the *book* does to that character's idea. **Always include `unclear`** as an option — forcing certainty produces garbage metadata.

3. **Spot-check ~30 tagged chunks by hand.** If speaker or stance is often wrong, fix your tagging prompt and re-run. Everything downstream inherits this metadata, so it must be decent before you continue.

**You'll know it worked when:** you can look at a chunk of the Underground Man ranting and see `speaker_relation: asserts`, `narrative_relation: undermines` — the system now *knows* the book is criticizing him.

---

## Stage 4 — Better retrieval: hybrid + rerank + diversity (1 day)

Pure semantic search misses exact terms (character names, specific words). Fix it.

1. **Add BM25** (keyword search) alongside your vector search using the `rank-bm25` library over the same chunk texts. Run both for a query, merge the results.

2. **Rerank.** Take the top ~20 merged results and score each for real relevance. Simplest version for now: one LLM call that returns a 1–5 relevance score per chunk. Keep the top 6–8.

3. **Diversify.** Don't let all 8 chunks come from one chapter just because they cluster. For a question spanning two novels, force at least a couple of passages from each.

4. **Filter by theme when useful** — a question about "living in my head" should boost chunks tagged `the-dreamer` and `isolation`.

**You'll know it worked when:** a query mentioning a character by name now reliably surfaces that character's passages (BM25 catching the name), and results aren't all clustered in one chapter.

---

## Stage 5 — Conversation handling + the dialogue loop (1 day)

A chat bot has to handle follow-ups. "How is that different from the Dreamer?" is meaningless on its own — retrieval needs the context.

1. **Query rewriting.** Before retrieving, make one small LLM call that rewrites the latest message into a standalone query using the conversation history. "How is that different from the Dreamer?" → "How does the Underground Man's isolation differ from the Dreamer's in White Nights?" **Retrieve using the rewritten query.**

2. **The loop.** Keep conversation history in a list. Each turn: rewrite → retrieve fresh → call the LLM with (persona prompt + history + retrieved passages) → append reply to history.

**You'll know it worked when:** you can ask a follow-up with a pronoun ("what about him?") and the bot retrieves the right passages anyway.

---

## Stage 6 — The persona (1 day) — where the quality lives

Now make it sound and *think* like him. This is a prompt, not code.

1. **Write the persona system prompt.** Encode the thing that makes him worth talking to: **compassion plus unmasking** — honor what the person feels, then press gently on the evasion hiding inside it. Answer with a scene or a character, not a lecture. Tell it explicitly to use the `narrative_relation` metadata so it never voices a character's view (Ivan, the Underground Man) as Dostoevsky's own belief.

2. **Add ~10–15 hand-written "rebuttal moves"** to the prompt — how he counters *kinds* of positions: rational egoism → the Underground Man; despair → Zosima's active love; clever abstraction → Ivan's own image of the child's tear, turned back on the speaker. This handmade list is what makes his pushback feel like *him* instead of a generic clever LLM.

3. **Tell it what's grounded vs. invented.** The passages are evidence; it may interpret but must not invent facts about the books.

**You'll know it worked when:** you tell it "I relate to the Dreamer, I live in my head," and it responds with genuine tenderness *and* an uncomfortable nudge about the life you're avoiding — not bland validation.

---

## Stage 7 — Wrap it in FastAPI (½ day)

You already know this part.

1. `POST /chat` — body: `{session_id, message}` → returns `{reply}`.
2. Store conversation history per `session_id` in an in-memory dict. No database needed for V1.
3. Test with curl or Postman. Don't build a UI until the conversation is genuinely good.

**You'll know it worked when:** you can hold a multi-turn conversation through the API.

---

## Stage 8 — Evaluation (ongoing — do NOT skip)

Without this you'll polish random things and never know if you're improving.

1. Write **15–20 test exchanges**: personal questions, a White Nights discussion, someone arguing back.
2. Score each on two axes, 1–5: **voice fidelity** (does it sound like him) and **substantive fidelity** (does it reason like him, including refusing tidy answers).
3. Re-run whenever you change the prompt or retrieval. This is your dashboard.

**You'll know it worked when:** you can say "changing the persona prompt raised voice from 3.4 to 4.1" — a real number, not a vibe. This sentence is also what makes the project impressive to employers.

---

## Order of attack (realistic timeline)

| Day | Stage |
|-----|-------|
| 1 | Stage 0 + Stage 1 — working naive pipeline, one book |
| 2 | Stage 2 + start Stage 3 — chunking, provenance, tagging |
| 3 | Finish Stage 3 + Stage 4 — tagging spot-check, hybrid retrieval |
| 4 | Stage 5 + Stage 6 — dialogue loop, persona |
| 5 | Stage 7 + Stage 8 — API, eval set |
| after | iterate using eval scores; add more novels |

## What to deliberately NOT build in V1

You'll be tempted by these — they sound advanced but they're scope creep for a first version. Skip them until evals prove you need them:

- A knowledge graph (we established this across the whole plan — no).
- A separate "neutral analysis then persona" two-stage pipeline — it turns a sparring partner into an essay in a costume.
- Pre-built operation functions (`compare_characters()`, etc.) — you don't know which you need yet.
- Fine-tuning — that's a V2 experiment you A/B against this baseline, not part of V1.
- Claim-arrays with confidence scores — the two-field stance already gets you most of the value.

## The two files that decide whether this is good

The **persona prompt** (Stage 6) and the **theme list + tagging prompt** (Stage 3). Everything else is plumbing you now know how to build. Spend disproportionate care on those two.


Say the conversation so far was about the Underground Man, and the user now types: "how's he different from the Dreamer?"

Step 1 — Query understanding (LLM call #1, small/cheap).
The raw message is broken on its own — "he" is meaningless to a search engine. So one fast LLM call, given the conversation history, does two jobs:

Rewrites to standalone: "how's he different from the Dreamer?" → "How does the Underground Man's isolation differ from the Dreamer's in White Nights?"
Extracts structured filters from the question: characters = [Underground Man, Dreamer]; works = [Notes from Underground, White Nights]; themes = [isolation].

That extraction is the bridge you were missing. The rewrite step doesn't just clean the query — it reads out which tags to filter/boost on. This is where rewriting and tagging connect: the LLM turns the messy human sentence into "search for this meaning, and among chunks tagged these ways."

Step 2 — Retrieval (mostly no LLM).
Now you search, using the rewritten query as what you search with and the extracted filters as what you constrain against:

Vector search with the rewritten query's embedding — finds passages about isolation/difference by meaning.
BM25 on the rewritten query — catches exact terms.
Tag filter/boost from the extracted filters — restrict to the two works, boost chunks tagged isolation, ensure coverage of both characters.

The tags do work here that the rewrite cannot: the rewrite made the question good, but only the tags let you say "and make sure I pull the Dreamer's passages and the Underground Man's, from both books, not five chunks about one of them." The rewrite shaped the query; the tags shape the candidate set.

Step 3 — Rerank + diversify (optional LLM call #2).
Score the ~20 candidates for real relevance, keep the best 6–8, enforce that both characters/books are represented (using the tags again). Expand each keeper with its prev/next neighbors so passages arrive as whole scenes.

Step 4 — Persona generation (LLM call #3, the answer).
Now the final call. It receives: the persona prompt, the conversation history, and the retrieved passages with their tags still attached. Here the tags do their second job — guardrails. Each passage arrives labeled "speaker: the Underground Man, narrative_relation: undermines," so the persona voices his spite as his spite the book is critical of, not as Dostoevsky's endorsed view. The rewrite had nothing to do with this; only the tags carry it.

That's the whole flow: rewrite → retrieve (vector + BM25 + tag filter) → rerank → generate. Three LLM calls at query time (understanding, optional rerank, generation), plus the tagging calls that already ran at build-time.