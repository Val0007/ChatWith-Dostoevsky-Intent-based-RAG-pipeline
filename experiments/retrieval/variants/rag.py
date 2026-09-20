"""Stage 1 retrieve + generate: ask a question, answer using White Nights passages.

Naive RAG on purpose — no persona, no reranking yet (those are later stages).
Usage:  python experiments/retrieval/variants/rag.py "What is the Dreamer like?"
"""
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = str(ROOT / "data" / "db")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
COLLECTION = "dostoevsky"

client = OpenAI()
collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)


def embed(texts: list[str]) -> list[list[float]]:
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


def retrieve(query: str, k: int = 6) -> list[dict]:
    """Embed the query, return the k nearest chunks with their metadata."""
    qvec = embed([query])[0]
    res = collection.query(query_embeddings=[qvec], n_results=k)
    out = []
    for doc, meta, cid in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
        out.append({"text": doc, "chapter": meta["chapter"], "id": cid})
    return out


def answer(query: str, k: int = 6) -> tuple[str, list[dict]]:
    passages = retrieve(query, k)
    context = "\n\n---\n\n".join(
        f"[{p['chapter']}] {p['text']}" for p in passages
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are answering using the passages below from Dostoevsky's "
                "White Nights. Ground your answer in them.\n\nPassages:\n\n" + context
            ),
        },
        {"role": "user", "content": query},
    ]
    r = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return r.choices[0].message.content, passages


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the Dreamer like?"
    reply, passages = answer(query)

    print(f"QUESTION: {query}\n")
    print("RETRIEVED SOURCES:")
    for p in passages:
        print(f"  - {p['id']} ({p['chapter']})")
    print("\nANSWER:\n" + reply)


if __name__ == "__main__":
    main()
