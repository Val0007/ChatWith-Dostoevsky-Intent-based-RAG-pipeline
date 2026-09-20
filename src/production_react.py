"""A genuinely ReAct-faithful retrieval agent (Yao et al. 2023, arXiv:2210.03629),
adapted for this corpus — one continuous interleaved Thought/Action/Observation trace,
not the two-role checker+formula design of production_llm_retries.

    question
      │
      ▼
    trace = "Question: ...\\n"          <- a single growing string; THIS is the agent's
      │                                    memory. There is no other state. Every step
      │                                    resends the whole trace so far.
      ▼
    ┌─────────────────────────────────────────────────┐
    │ LOOP (up to max_steps):                          │
    │   ask the model to continue the trace by exactly │
    │   ONE "Thought: ...\\nAction: ...", stopping      │
    │   generation before it can hallucinate its own   │
    │   Observation (stop=["Observation:"])            │
    │                                                    │
    │   parse the action:                               │
    │     Search[query]  -> real retrieval (production_ │
    │                        method's fuse+rerank), a   │
    │                        short summary comes back    │
    │     Lookup[string] -> Ctrl+F within passages       │
    │                        ALREADY found via Search    │
    │                        this session, not a fresh   │
    │                        corpus search                │
    │     Finish[]       -> break the loop                │
    │                                                    │
    │   append "Observation: <real result>" to trace    │
    └─────────────────────────────────────────────────┘
      │
      ▼
    all chunk ids the trace ever Searched up, diversified -> top k -> answer(question, ids)
      (the trace's own reasoning is retrieval-only; the FINAL answer still goes through
      retrieval.answer()'s stance-discipline/quotation rules, not the trace's own prose)

Companion/contrast to production_llm_retries (R16-R18): that design has TWO roles (a
sufficiency-checking LLM + a separate deterministic retrieval formula) invoked in
independent, stateless rounds — no shared reasoning carries over between rounds, only a
flat log of past query strings. This has ONE role, one continuously growing context, and
a finer-grained action space (a real "drill into what I already found" action, which
production_llm_retries has no equivalent of at all).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_method import production_retrieve  # noqa: E402
from retrieval import CHAT_MODEL, _diversify, answer, by_id, chunk_to_scene, client  # noqa: E402

REACT_SYSTEM = """You are a research agent answering a question about Dostoevsky's White Nights by interleaving reasoning with actions that search the book.
You do not answer the question directly. You gather evidence, and a separate step writes the final answer from the evidence you find.
You have exactly three actions:

* `Search[query]`: runs a real search over the book for `query`. Returns a short summary of the best-matching passages found, each with an id.
* `Lookup[string]`: searches ONLY within passages already found via Search during this session. It is equivalent to Ctrl+F within the passages already retrieved. It is NOT a fresh search of the whole book.
* `Finish[]`: use this once you have enough evidence to answer the question well.

At every step output EXACTLY one Thought and one Action, then stop.
Do not write an Observation yourself. It will be provided after the action runs.
SEARCH STRATEGY
Use the following search strategy in order.
STEP 1: THE FIRST QUERY IS IMMUTABLE
Your FIRST action must be:

Search[<the user's question verbatim>]

Do not rewrite, summarize, compress, or paraphrase it. Copy the question's exact wording as the entire query, unchanged -- not a keyword extraction of it, not a shortened version, the literal text. The original wording is valuable retrieval evidence in its own right.
STEP 2: LOOKUP GETS EXPLICIT PRIORITY
After every Search:

If any returned passage appears potentially relevant, use Lookup before performing another Search.

Only Search again when the existing passages do not contain a promising place to inspect.

When examining results, ask:
1. Did I find the answer or direct evidence?
2. Did I find a passage that is probably close? -- if yes, Lookup it, do not Search again.
3. Did I discover distinctive vocabulary, names, events, objects, locations, or wording?
4. Did the search fail because the question's vocabulary does not match the book's vocabulary?
STEP 3: A RECOVERY SEARCH MUST USE A CONCRETE ANCHOR FROM RESULTS
If another Search is genuinely necessary (only once Step 2 has ruled out Lookup):

Take one distinctive term, event, object, name, location, or phrase from the passages you have already retrieved, and combine it with the missing concept.

Do not replace words from the previous query with synonyms. A new Search must change the retrieval anchor, not just the grammar.
For interpretive questions, search for the concrete evidence underlying the interpretation rather than repeatedly searching for the interpretation itself.
For example, if the question asks why a character behaves a certain way, search for the character's relevant actions, prior events, feelings, conflicts, or circumstances.
STEP 4: DO NOT REPEAT FAILED VOCABULARY
Track the searches already attempted.
Do not issue another Search that is merely a grammatical or synonym-level variation of a failed Search.
If several searches have failed using the same generic terms, deliberately move to a different retrieval anchor.
For example, if searches repeatedly center on:
"the dreamer", "the girl", "feelings"
do not keep producing:
"the dreamer's feelings about the girl"
Instead, use a concrete event, action, object, location, relationship, or distinctive phrase discovered from the question or retrieved evidence.
STEP 5: USE LOOKUP TO EXTRACT WHAT YOU NEED FROM A PASSAGE YOU ALREADY HAVE
Use Lookup to:

* find a specific word or phrase,
* retrieve the exact sentence containing it,
* inspect surrounding context,
* verify who did what,
* confirm a detail needed for the final answer.

Do not use Lookup as a substitute for Search when the necessary concept has never appeared in a retrieved passage.
WHEN TO FINISH
Do not continue searching merely because more evidence could theoretically exist.
Finish when the evidence gathered is sufficient to construct a reliable answer.
For a factual question, finish when you have direct or unambiguous textual evidence for the requested fact.
For an interpretive or motivation question, finish when you have enough concrete evidence to support the interpretation. The book does not need to state the final interpretation verbatim.
A good interpretive evidence set may combine multiple passages establishing:

* the character's circumstances,
* relevant actions,
* feelings,
* prior events,
* relationships,
* and consequences.

If those pieces together support the answer, finish.
RETRIEVAL BUDGET
You have a limited number of actions.
Treat each Search as an attempt to recover evidence, not as a requirement to keep searching.
If you have already made several genuinely different Searches and they continue returning the same type of irrelevant or insufficient evidence, recognize that the retrieval system may not be exposing the needed passage.
Do not keep generating increasingly speculative queries simply because actions remain available.
When no plausible, materially different retrieval strategy remains, call `Finish[]` with the best evidence gathered.
Likewise, if you already have enough evidence, call `Finish[]` immediately.
The goal is NOT to use all available steps.
The goal is to obtain sufficient evidence with as few actions as reasonably possible.
IMPORTANT DISTINCTION
Do not confuse:
"the evidence is insufficient"
with:
"I have not searched enough."
More reasoning does not guarantee better retrieval.
If repeated, materially different Searches fail to recover the required evidence, additional reformulations may have diminishing value.
FINAL SELF-CHECK BEFORE FINISH
Before calling `Finish[]`, ask:

1. Do I have direct evidence for a factual question?
2. Or, for an interpretive question, do I have enough concrete evidence to support the interpretation?
3. Have I already searched using the most obvious wording?
4. If I performed additional Searches, did they actually use different retrieval anchors?
5. Is another Search likely to discover something materially different, or am I merely searching harder?

If the evidence is sufficient, call `Finish[]`.
If the evidence remains insufficient but no meaningful new retrieval path remains, call `Finish[]` rather than wandering.
OUTPUT FORMAT
At every step output exactly:
Thought:
Action: <Search[...] or Lookup[...] or Finish[]>
Then stop and wait for the Observation.
Never output an Observation yourself.
Never output the final answer.
Never output more than one Action at a time."""


# (v1/v2/v3 prompt history: see experiments/retrieval/FINDINGS.md R20/R21/R22.
#  v4 = v2 base + 3 surgical rules: immutable first query (Step 1), mandatory
#  Lookup-before-Search priority (Step 2), anchor+missing-concept recovery
#  queries instead of a general "useful changes include" list (Step 3).)


# example ─ in:  "Question: Why does...?\nThought: ...\nAction: Search[...]\nObservation: ...\n"
#           out: "Thought: I now know X. I should look up Y.\nAction: Lookup[Y]"
def _next_step(trace: str) -> str:
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2, stop=["Observation:"],
        messages=[{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": trace}],
    )
    return r.choices[0].message.content.strip()


_ACTION_RE = re.compile(r"Action:\s*(Search|Lookup|Finish)\[(.*?)\]", re.DOTALL)


def _parse_action(step_text: str):
    m = _ACTION_RE.search(step_text)
    if not m:
        return None, None
    return m.group(1), m.group(2).strip()


# example ─ in:  "the narrator's feelings after the letter"
#           out: "Found passages: [white_nights_morning_002] \"...my dear! Next week I am
#                 to be married to him...\" [white_nights_morning_004] \"My God, a whole
#                 moment of happiness!...\""
def _do_search(query: str, k: int = 5) -> tuple:
    ids, _ = production_retrieve(query, k=k)
    parts = [f"[{cid}] {by_id[cid]['text'][:180]!r}" for cid in ids]
    return ids, "Found passages: " + " ".join(parts)


# example ─ in:  ("moment of happiness", ["white_nights_morning_004", ...])
#           out: ("Found in white_nights_morning_004: \"...May your sky be clear...may you
#                 be blessed for that moment of blissful happiness...\"", "white_nights_morning_004")
def _do_lookup(string: str, encountered_ids: list) -> tuple:
    needle = string.lower()
    for cid in encountered_ids:
        text = by_id[cid]["text"]
        idx = text.lower().find(needle)
        if idx != -1:
            start, end = max(0, idx - 100), min(len(text), idx + len(string) + 100)
            return f"Found in {cid}: ...{text[start:end]}...", cid
    return f"'{string}' not found in any passage found so far via Search.", None


# example ─ in:  "Why does the narrator become so emotionally attached to Nastenka..."
#           out: (["white_nights_third_night_002", ...],
#                 {"trace": "Question: ...\\nThought: ...", "steps": 4, "encountered": 9, "confirmed": 2})
def react_retrieve(query: str, k: int = 6, max_steps: int = 6, per_scene: int = 2):
    trace = f"Question: {query}\n"
    encountered_ids = []  # every id ANY Search returned — dedup, first-seen order
    confirmed_ids = []    # ids a Lookup actually verified contain something relevant — a
                           # STRONGER signal than a bare Search hit, since the agent chose
                           # to check that specific passage for that specific detail
    steps = 0

    for steps in range(1, max_steps + 1):
        step_text = _next_step(trace)
        trace += step_text + "\n"
        action, arg = _parse_action(step_text)

        if action == "Search":
            ids, obs = _do_search(arg)
            encountered_ids.extend(cid for cid in ids if cid not in encountered_ids)
        elif action == "Lookup":
            obs, found_cid = _do_lookup(arg, encountered_ids)
            if found_cid and found_cid not in confirmed_ids:
                confirmed_ids.append(found_cid)
        elif action == "Finish":
            break
        else:
            obs = "Invalid action format. Use Search[...], Lookup[...], or Finish[]."

        trace += f"Observation: {obs}\n"

    # confirmed_ids go FIRST: _diversify is a greedy pass over its input order, so a
    # Lookup-verified chunk now competes for its scene's slot ahead of chunks that only
    # ever showed up in a Search result and were never actually checked
    pool = confirmed_ids + [cid for cid in encountered_ids if cid not in confirmed_ids]
    pool = pool or production_retrieve(query, k=k)[0]  # fail open if Search never ran
    final_ids = _diversify(pool, k, per_scene)
    return final_ids, {"trace": trace, "steps": steps, "encountered": len(encountered_ids),
                        "confirmed": len(confirmed_ids)}


def react_answer(query: str, k: int = 6):
    ids, debug = react_retrieve(query, k=k)
    return answer(query, ids), debug


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the narrator first meet Nastenka?"
    ids, debug = react_retrieve(query)
    print(debug["trace"])
    print(f"\n[stopped after {debug['steps']} steps, {debug['encountered']} encountered, "
          f"{debug['confirmed']} confirmed via Lookup]\n")
    for cid in ids:
        print(f"  {cid} [{chunk_to_scene.get(cid)}]")
    print("\nANSWER:\n" + answer(query, ids))


if __name__ == "__main__":
    main()
