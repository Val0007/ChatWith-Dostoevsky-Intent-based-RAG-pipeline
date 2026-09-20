"""Stage 5: conversation loop + query rewriting.

Wraps Stage-4 retrieval so multi-turn follow-ups work. Each turn:
  1. rewrite the latest message into a standalone query using the history
  2. retrieve fresh on the rewritten query (Stage 4: hybrid -> rerank -> diversify)
  3. answer with (history + deduped scene/chunk context)
  4. append the turn to history

Voice is still plain/grounded here; persona is Stage 6.
"""
import sys

import retrieval as R

REWRITE_SYSTEM = """You rewrite the user's latest message into a single standalone
search query for retrieving passages from Dostoevsky's White Nights.

Resolve pronouns and ellipsis using the conversation ("he"/"she"/"that" -> the actual
person or idea). Keep it a concise query, not a question to answer. If the latest
message is already self-contained, return it essentially unchanged.
Return ONLY the rewritten query text, nothing else."""


# example ─ in:  (history=[...about the Dreamer...], message="how is he different from Nastenka?")
#           out: "Differences between the Dreamer and Nastenka in White Nights"  (pronoun resolved)
def rewrite_query(history: list[dict], message: str) -> str:
    """Turn a possibly-context-dependent message into a standalone retrieval query."""
    if not history:
        return message
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    user = f"CONVERSATION SO FAR:\n{convo}\n\nLATEST MESSAGE: {message}\n\nStandalone query:"
    r = R.client.chat.completions.create(
        model=R.CHAT_MODEL, temperature=0,
        messages=[{"role": "system", "content": REWRITE_SYSTEM},
                  {"role": "user", "content": user}],
    )
    return r.choices[0].message.content.strip()


# example ─ in:  [{"role":"user","content":"Tell me about the Dreamer"},
#                 {"role":"assistant","content":"He is..."}]
#           out: "User: Tell me about the Dreamer\nYou (Dostoevsky): He is..."
def _render_history(history: list[dict]) -> str:
    if not history:
        return "(this is the first exchange)"
    label = {"user": "User", "assistant": "You (Dostoevsky)"}
    return "\n".join(f"{label.get(m['role'], m['role'])}: {m['content']}" for m in history)


# example ─ in:  (history, message="no he's just delusional")
#           out: {"reply": "Calling him delusional overlooks...",   # Dostoevsky's answer
#                 "rewritten": "the Dreamer's delusional love for Nastenka",  # used for retrieval
#                 "retrieved": ["white_nights_second_night_012", ...]}   # and appends both to history
def chat(history: list[dict], message: str, system: str = None) -> dict:
    """One turn: rewrite (for retrieval only) -> retrieve -> answer from a structured
    payload (original message + conversation + evidence). Mutates history in place.

    Defaults to the Stage-6 Dreamer persona; pass system=R.ANSWER_SYSTEM for plain voice.
    """
    system = system or R.PERSONA_SYSTEM
    standalone = rewrite_query(history, message)      # used ONLY to retrieve
    top_ids = R.search(standalone)
    evidence = R.build_context(top_ids)

    # Answer the ORIGINAL message (not the sanitized query). USER is the thing to
    # reply to; CONVERSATION is flow; EVIDENCE is only material, not the topic.
    payload = (
        f"Reply to the USER message below, in the flow of the CONVERSATION, grounded in "
        f"the EVIDENCE. Do not repeat your previous answer.\n\n"
        f"USER (reply to this):\n{message}\n\n"
        f"CONVERSATION SO FAR:\n{_render_history(history)}\n\n"
        f"EVIDENCE (retrieved passages from White Nights, with scene summary, "
        f"provenance, and tags - material only, not the question):\n{evidence}"
    )
    r = R.client.chat.completions.create(
        model=R.CHAT_MODEL, temperature=0.6,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": payload}],
    )
    reply = r.choices[0].message.content

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return {"reply": reply, "rewritten": standalone, "retrieved": top_ids}


def _demo():
    """Scripted multi-turn proving pronoun/ellipsis follow-ups retrieve correctly."""
    history = []
    turns = [
        "Tell me about the Dreamer.",
        "How is he different from Nastenka?",   # 'he' -> the Dreamer
        "What does she want?",                  # 'she' -> Nastenka
    ]
    for msg in turns:
        out = chat(history, msg)
        print("=" * 90)
        print(f"USER: {msg}")
        print(f"REWRITTEN QUERY: {out['rewritten']}")
        print(f"RETRIEVED: {[c.split('white_nights_')[-1] for c in out['retrieved']]}")
        print(f"REPLY:\n{out['reply']}\n")


if __name__ == "__main__":
    _demo()
