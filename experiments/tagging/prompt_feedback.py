"""Round-based prompt-fix feedback loop.

For one failure cluster (a field + a set of mismatched chunks from run_first30_eval.py's
cached output), send the CURRENT prompt text + the failing passages to a critic LLM and get
back: a diagnosis of what in the prompt is causing the pattern, and a minimal proposed patch
(exact old/new text, one of SYSTEM_PROMPT or CONTEXT_USER_TEMPLATE).

The patch is NEVER auto-applied -- it's printed and saved to experiments/tagging/results/prompt_feedback/feedback_<round>.txt
for manual review against the full current prompt (conflict-checking) before editing
src/tag_white_nights.py by hand. See experiments/tagging/FINDINGS.md for the round log.

Run from repo root:
    .venv/bin/python experiments/tagging/prompt_feedback.py <round_key>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chunking import load_chunks
from tag_white_nights import CONTEXT_USER_TEMPLATE, SYSTEM_PROMPT, client

GOLD_PATH = ROOT / "experiments" / "tagging" / "gold" / "gold_first30.jsonl"
CACHE_PATH = ROOT / "experiments" / "tagging" / "results" / "first30_tagger_output.json"

CRITIC_SYSTEM = """You are a prompt-engineering diagnostician for a literary metadata tagger.
You will be shown:
  1. The CURRENT system prompt and user template the tagger uses (verbatim).
  2. A cluster of tagging failures: for each, the passage text, the gold label + why, and what
     the tagger predicted instead.

Your job, in this order:
  1. DIAGNOSE: read the failing passages and explain, in plain terms, the actual mechanism behind
     the pattern of errors -- not "the model is bad at this field" but what in the CURRENT PROMPT
     TEXT is causing this specific bias. Quote the exact prompt language you think is responsible.
  2. PROPOSE A MINIMAL PATCH: propose the smallest possible edit to the CURRENT prompt text that
     would fix this failure mode. Rules:
       - Change as little as possible. Prefer clarifying/tightening existing language over adding
         new paragraphs.
       - Do NOT remove or contradict any existing instruction unless it is the direct, demonstrated
         cause of this failure -- and if you do, say explicitly which existing sentence you are
         overriding and why it's necessary.
       - Do NOT touch parts of the prompt unrelated to this failure cluster.
       - Give the exact OLD text (a verbatim substring of the current prompt) and exact NEW text,
         so it can be applied as a literal find-and-replace. State which variable it's in
         (SYSTEM_PROMPT or CONTEXT_USER_TEMPLATE).
  3. STATE ANY REGRESSION RISK: name any currently-correct behavior your patch could plausibly
     break, if any.

Return ONLY this structure, no other prose:

DIAGNOSIS:
<2-5 sentences>

PATCH:
target: SYSTEM_PROMPT | CONTEXT_USER_TEMPLATE
old: <exact verbatim substring to replace>
new: <replacement text>

REGRESSION RISK:
<1-3 sentences, or "none identified">
"""


def load_gold():
    gold = {}
    for line in GOLD_PATH.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            gold[d["id"]] = d
    return gold


def build_user_message(round_name, field, failures, preamble=None):
    chunks = {c["id"]: c for c in load_chunks()}
    gold = load_gold()
    lines = [f"FAILURE CLUSTER: {round_name}  (field: {field})", ""]
    if preamble:
        lines.append(preamble)
        lines.append("")
    lines.append("=== CURRENT SYSTEM_PROMPT ===")
    lines.append(SYSTEM_PROMPT)
    lines.append("")
    lines.append("=== CURRENT CONTEXT_USER_TEMPLATE ===")
    lines.append(CONTEXT_USER_TEMPLATE)
    lines.append("")
    lines.append("=== FAILING EXAMPLES ===")
    for f in failures:
        cid = f["id"]
        c = chunks[cid]
        g = gold[cid]
        lines.append(f"--- {cid} ---")
        lines.append(f"TEXT: {c['text']}")
        lines.append(f"GOLD {field}: {g[field]}")
        lines.append(f"GOLD RATIONALE: {f.get('rationale', '(see gold_first30_notes.md)')}")
        if f.get("local_pred"):
            lines.append(f"LOCAL ARM predicted (same SYSTEM_PROMPT, simpler prev/next-only "
                          f"template, NO CONTEXT_USER_TEMPLATE): {f['local_pred']}  <- correct")
        lines.append(f"CONTEXT ARM predicted (CONTEXT_USER_TEMPLATE): {f['pred']}  <- wrong")
        lines.append("")
    return "\n".join(lines)


def run_round(round_name, field, failures, model="gpt-4o", preamble=None):
    user_msg = build_user_message(round_name, field, failures, preamble=preamble)
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": CRITIC_SYSTEM},
                  {"role": "user", "content": user_msg}],
    )
    out = r.choices[0].message.content
    print(out)
    out_path = ROOT / "experiments" / "tagging" / "results" / "prompt_feedback" / f"feedback_{round_name}.txt"
    out_path.write_text(out)
    print(f"\n[saved -> {out_path}]")
    return out


# ---------------------------------------------------------------- round definitions

ROUNDS = {
    "r1_undermines_softened": dict(
        field="narrative_relation",
        failures=[
            dict(id="white_nights_first_night_002", pred="unclear",
                 rationale="The house/old-man personification isn't cute scene color -- it's the "
                            "Dreamer substituting imagined relationships with buildings and "
                            "strangers for real ones. Mild, comic-register instance of the same "
                            "self-deception pattern that peaks in Second Night."),
            dict(id="white_nights_first_night_003", pred="unclear",
                 rationale="Continuation of 002: the pink house 'painted yellow' is staged as a "
                            "literal betrayal -- same self-deceptive attachment to an imagined "
                            "relationship with an inanimate object."),
            dict(id="white_nights_first_night_012", pred="unresolved",
                 rationale="The imagined-approach-to-a-stranger-lady fantasy is a textbook "
                            "instance of substituting fantasy for real connection -- same "
                            "self-deception pattern, delivered as a hypothetical."),
            dict(id="white_nights_second_night_005", pred="complicates",
                 rationale="The dreamer-archetype self-portrait: settles into his corner 'like a "
                            "snail', embarrassed by real visitors, compares himself to a forger "
                            "or a secret poet. Near-verbatim match to the system prompt's own "
                            "'burrowing fully into his shell' calibration example."),
            dict(id="white_nights_second_night_006", pred="complicates",
                 rationale="Continuation of 005's self-portrait: his isolation actively repels "
                            "even a willing visitor (tongue-tied, forced small talk, friend "
                            "inventing an excuse to leave)."),
            dict(id="white_nights_second_night_008", pred="complicates",
                 rationale="Peak grandiose self-mythologizing -- compares their meeting to King "
                            "Solomon's seal being lifted after a thousand years, 'it was ordained "
                            "that we should meet'. Textbook 'rapturous, self-glorifying fantasy'."),
            dict(id="white_nights_second_night_010", pred="unclear",
                 rationale="He literally loses track of where he is walking mid-fantasy and lies "
                            "to a stranger to save face -- fantasy substituting for lived reality, "
                            "stated almost literally."),
        ],
    ),
    "r3_asserts_vs_explores": dict(
        field="speaker_relation",
        failures=[
            dict(id="white_nights_first_night_003", pred="explores",
                 rationale="Plain, stated description of his habitual attachment to the houses "
                            "and the old man -- presented as established fact about himself, not "
                            "hedged or wondering."),
            dict(id="white_nights_first_night_008", pred="explores",
                 rationale="'when I am happy I am always humming to myself...who has no friend or "
                            "acquaintance with whom to share his joy' -- a stated fact about his "
                            "own habit, not a hypothesis."),
            dict(id="white_nights_first_night_009", pred="explores",
                 rationale="'I bless my luck for the excellent knotted stick' -- a plain, decisive "
                            "declaration accompanying action, not wondering."),
            dict(id="white_nights_first_night_013", pred="explores",
                 rationale="Nastenka's reply is a stated, confident judgment ('I was judging by "
                            "myself; I know a good deal about other people's lives') -- not "
                            "hedged uncertainty."),
            dict(id="white_nights_first_night_016", pred="explores",
                 rationale="Both speakers state their terms/feelings plainly and with conviction "
                            "-- she lays down a firm condition, he swears agreement outright."),
            dict(id="white_nights_first_night_017", pred="explores",
                 rationale="'You have made me happy forever' -- a plain, unhedged declaration of "
                            "present emotional state."),
            dict(id="white_nights_second_night_002", pred="explores",
                 rationale="'I have lived...utterly alone -- alone, entirely alone' -- stated "
                            "plainly and almost defiantly as fact, not a hypothesis."),
            dict(id="white_nights_second_night_003", pred="explores",
                 rationale="'Very well, I am a type!' -- a bold, unhedged self-declaration."),
            dict(id="white_nights_second_night_005", pred="explores",
                 rationale="The dreamer-archetype description is delivered as a confident, "
                            "near-lecture definition ('let me tell you...he is...'), not tentative "
                            "musing."),
            dict(id="white_nights_second_night_006", pred="explores",
                 rationale="Continues the same confident lecture register as 005."),
            dict(id="white_nights_second_night_007", pred="explores",
                 rationale="His confirmation ('Doubtless,' delivered 'with the gravest face') is "
                            "a plain, decisive assertion, not a hedge."),
            dict(id="white_nights_second_night_009", pred="explores",
                 rationale="Plain narration of an ordinary, grounded routine ('he was pleased "
                            "because...happy as a schoolboy') -- stated as fact, not wondering."),
            dict(id="white_nights_second_night_011", pred="explores",
                 rationale="'his soul is sad and empty' -- a plainly stated description of his "
                            "felt emotional state, not a hedge or a hypothesis."),
        ],
    ),
    "r2_context_template_regression": dict(
        field="narrative_relation",
        preamble=(
            "IMPORTANT: every example below is a REGRESSION -- the LOCAL arm (same "
            "SYSTEM_PROMPT, but a plain prev/next-chunk template with NO context data and NO "
            "restated narrative_relation instructions) got the correct label. The CONTEXT arm, "
            "using CONTEXT_USER_TEMPLATE below, got the SAME chunk wrong. Since SYSTEM_PROMPT is "
            "identical in both arms, the cause must be something in CONTEXT_USER_TEMPLATE's own "
            "text -- either it restates/dilutes SYSTEM_PROMPT's narrative_relation calibration in "
            "a way that conflicts with it, or the map/scene context itself is misleading the "
            "model on these specific chunks. Diagnose which, and patch CONTEXT_USER_TEMPLATE "
            "(not SYSTEM_PROMPT) unless you find clear evidence the problem is actually in the "
            "context DATA being fed in (global_map.json / scene_cards.json), not the template."
        ),
        failures=[
            dict(id="white_nights_first_night_018", pred="supports", local_pred="unresolved",
                 rationale="Classic parting-chapter close: 'Till to-morrow!'...'I was so happy'. "
                            "A parting the story leaves open and aching -- textbook unresolved. "
                            "'supports' is a large miss, not just a boundary call."),
            dict(id="white_nights_second_night_002", pred="complicates", local_pred="unclear",
                 rationale="Plain backstory exposition (his 'utterly alone' declaration, "
                            "Nastenka's blind-grandmother backstory) -- establishes facts, takes "
                            "no evaluative stance on them within this chunk."),
            dict(id="white_nights_second_night_008", pred="complicates", local_pred="undermines",
                 rationale="Peak grandiose self-mythologizing (King Solomon's seal metaphor, "
                            "'it was ordained that we should meet') -- textbook rapturous "
                            "self-glorifying fantasy, explicitly called out in SYSTEM_PROMPT's "
                            "own calibration language."),
            dict(id="white_nights_second_night_010", pred="unclear", local_pred="undermines",
                 rationale="He literally loses track of where he's walking mid-fantasy and lies "
                            "to a stranger to save face -- fantasy substituting for lived reality."),
        ],
    ),
    "r4_dialogue_collapsed_to_one_voice": dict(
        field="speaker",
        preamble=(
            "In every example below, gold is 'the Dreamer and Nastenka' (genuine back-and-forth "
            "dialogue) but the tagger picked a single name. Look for a pattern: in each case one "
            "person speaks at much greater LENGTH than the other, but the other still has a real, "
            "substantive turn (not just a one-word interjection). Check whether the current rule "
            "for this field is implicitly steering the model toward whichever voice has more "
            "words, rather than whether BOTH voices carry real content."
        ),
        failures=[
            dict(id="white_nights_first_night_010", pred="the Dreamer",
                 rationale="Alternating exchange right after the rescue: he offers his arm, she "
                            "replies with a full turn ('why did you drive me away?...'), he "
                            "replies, she replies again. Both speak substantively."),
            dict(id="white_nights_first_night_017", pred="the Dreamer",
                 rationale="Alternating trust-building dialogue: his question, her longer reply "
                            "('Sleep soundly...that I might confide in you?'), his reply, her "
                            "reply. Both speak substantively even though his lines run longer."),
            dict(id="white_nights_second_night_004", pred="Nastenka",
                 rationale="The name-reveal exchange (both speak in short alternating turns: "
                            "'At last!'/'Oh, my goodness!'/'My name is Nastenka.'/'Nastenka! And "
                            "nothing else?'/...) followed by the Dreamer beginning his monologue. "
                            "Both voices are substantially present in this chunk."),
            dict(id="white_nights_second_night_008", pred="the Dreamer",
                 rationale="Nastenka interrupts his monologue with a real critique ('you talk as "
                            "though you were reading it out of a book') before he resumes at "
                            "length. Her turn is short but substantive, not a filler word."),
        ],
    ),
    "r5_theme_underapplication": dict(
        field="canonical_themes",
        preamble=(
            "In every example below, gold includes 'self-deception' and/or 'the-dreamer' as a "
            "theme, but the tagger omitted it (often returning a shorter or empty theme list "
            "instead). Note that CONTEXT_USER_TEMPLATE's canonical_themes guidance gives an "
            "explicit one-line definition for 'isolation', 'alienation-from-society', "
            "'ephemeral-connection', and 'unrequited-longing' -- but gives NO definition at all "
            "for 'self-deception' or 'the-dreamer', even though both are in the allowed theme "
            "list. Check whether that absence is the cause, and if so patch it the same way the "
            "other themes are already defined (one clarifying clause each, matching the existing "
            "style) rather than restructuring the section."
        ),
        failures=[
            dict(id="white_nights_first_night_002", pred="isolation",
                 rationale="Personifying the old man and the houses as friends -- an imagined "
                            "relationship substituting for a real one. gold=[isolation, "
                            "self-deception]; pred dropped self-deception."),
            dict(id="white_nights_first_night_003", pred="[]",
                 rationale="The pink-house 'betrayal' bit -- same self-deceptive attachment to an "
                            "imagined relationship. gold=[isolation, self-deception]; pred=[]."),
            dict(id="white_nights_second_night_007", pred="[]",
                 rationale="Nastenka guesses the 'friend visit' story is really about him; he "
                            "confirms gravely. gold=[the-dreamer, self-deception]; pred=[]."),
            dict(id="white_nights_second_night_010", pred="[]",
                 rationale="He loses track of where he's walking mid-fantasy, lies to a stranger "
                            "to save face. gold=[the-dreamer, self-deception]; pred=[]."),
            dict(id="white_nights_second_night_011", pred="[]",
                 rationale="The fantasy dissolves, leaves him sad and empty. gold=[the-dreamer, "
                            "self-deception, isolation]; pred=[]."),
            dict(id="white_nights_second_night_009", pred="[]",
                 rationale="Ordinary contentment on his walk home, part of the extended dreamer "
                            "self-portrait. gold=[the-dreamer]; pred=[] -- the-dreamer theme "
                            "dropped even with no self-deception content to distract from it."),
        ],
    ),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ROUNDS:
        print(f"usage: python experiments/tagging/prompt_feedback.py <round_key>")
        print(f"available rounds: {list(ROUNDS.keys())}")
        sys.exit(1)
    key = sys.argv[1]
    spec = ROUNDS[key]
    run_round(key, spec["field"], spec["failures"], preamble=spec.get("preamble"))


if __name__ == "__main__":
    main()
