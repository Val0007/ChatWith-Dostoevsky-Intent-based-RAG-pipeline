"""Three-pass tagging: route each field through the context arm F8 showed it actually needs,
instead of one blended context bundle for everything.

    PASS 1 (local)  chunk + prev/next        -> narrating_voice, speaking_voice,
                                                 speaker_relation, local_motifs
    PASS 2 (scene)  chunk + scene cards       -> canonical_themes, characters_present (candidates)
    PASS 3 (global) chunk + scene + map       -> narrative_relation (fresh), and adjudicates
                    + pass-2 candidates          (confirms or corrects) pass 2's themes/characters

Per F8 (experiments/FINDINGS.md): narrative_relation is carried by the global map, NOT scenes
(scenes alone were worse than local); canonical_themes/characters_present are carried by scenes,
NOT the map; speaker_relation was best under LOCAL alone, every context arm made it worse.

Each pass's prompt reuses the exact tuned instruction text from tag_white_nights.py's
SYSTEM_PROMPT / CONTEXT_USER_TEMPLATE verbatim (7+ rounds of hand-fixing already went into that
wording -- this file re-routes which fields see it, it does not rewrite it) -- except
speaker_relation and characters_present now see NO global map at all, which is the actual point.
"""
import json
from typing import List, Optional

from pydantic import BaseModel, field_validator

from tag_white_nights import ALLOWED_THEMES, NARRATIVE_REL, SPEAKER_REL, client

# ---------------------------------------------------------------- schemas

class LocalPassTags(BaseModel):
    narrating_voice: str
    speaking_voice: Optional[str] = None
    speaker_relation: str
    local_motifs: List[str] = []

    @field_validator("narrating_voice")
    @classmethod
    def _nv(cls, v):
        assert v == "the Dreamer", f"narrating_voice must be 'the Dreamer', got: {v}"
        return v

    @field_validator("speaker_relation")
    @classmethod
    def _sr(cls, v):
        assert v in SPEAKER_REL, f"bad speaker_relation: {v}"
        return v


class ScenePassTags(BaseModel):
    canonical_themes: List[str]
    characters_present: List[str]

    @field_validator("canonical_themes")
    @classmethod
    def _themes(cls, v):
        assert len(v) <= 3, "at most 3 themes"
        bad = set(v) - ALLOWED_THEMES
        assert not bad, f"themes not in vocabulary: {bad}"
        return v


class GlobalPassTags(BaseModel):
    narrative_relation: str
    canonical_themes: List[str]
    characters_present: List[str]

    @field_validator("narrative_relation")
    @classmethod
    def _nr(cls, v):
        assert v in NARRATIVE_REL, f"bad narrative_relation: {v}"
        return v

    @field_validator("canonical_themes")
    @classmethod
    def _themes(cls, v):
        assert len(v) <= 3, "at most 3 themes"
        bad = set(v) - ALLOWED_THEMES
        assert not bad, f"themes not in vocabulary: {bad}"
        return v


# ---------------------------------------------------------------- PASS 1: local

LOCAL_PASS_SYSTEM_PROMPT = """You are a literary annotation engine for Dostoevsky's novella "White Nights"
(Constance Garnett translation). This is PASS 1 of 3 -- the LOCAL pass. You see
only the TARGET passage plus its immediate previous/next neighbor chunk. You
have NO scene summaries and NO whole-book context in this pass -- do not guess
at anything beyond what these three passages show you. You are precise,
cautious, and you never invent content that is not in the passage.

WHAT YOU RETURN
Return ONLY a single JSON object, no prose, no markdown fences. It must match
this schema exactly:

{
  "narrating_voice": string,         // who is narrating the TARGET passage (see NARRATOR RULES)
  "speaking_voice": string | null,   // who is actually SPEAKING/expressing themselves in it, or null
  "speaker_relation": string,        // one of: asserts | doubts | rejects | explores
  "local_motifs": string[]           // 0-4 freeform short phrases for notable details
}

NARRATOR vs. SPEAKER (two different questions — do not collapse them)
"White Nights" is narrated throughout, first-person, by an unnamed young man
called the Dreamer. That is a fact about the BOOK, not about any one passage —
so:
- narrating_voice is ALWAYS "the Dreamer" for this book. Every passage, without
  exception. Do not vary this field; it is not asking who is talking.
- speaking_voice asks a completely different question: within the TARGET
  passage, whose voice/stance is actually being expressed?
    - When Nastenka is telling her own story or speaking in dialogue, her
      speaking_voice is "Nastenka" — even though the Dreamer (the narrator) is
      the one relaying it on the page. Attribute speech to who is actually
      talking, not to the narrator relaying it.
    - When the Dreamer is speaking in dialogue, OR reflecting/expressing his
      own feelings and opinions in his narration ("I was lonely", "I felt..."),
      speaking_voice is "the Dreamer".
    - For a passage where BOTH the Dreamer and Nastenka speak with real,
      substantive content — a question, challenge, judgment, commitment, or
      vow, not a bare "Yes!" or "Well?" (a short exclamation like "I swear" or
      "I agree to anything" IS substantive, even tagged with "I cried" rather
      than "I said") — set speaking_voice to "the Dreamer and Nastenka". This
      holds no matter how unequal their lengths are and no matter which one
      dominates or opens the passage, in EITHER direction: a long speech from
      either one that gets even a short-but-real reply from the other still
      counts as both. PROCEDURE: before answering, look for one real line from
      the Dreamer AND one real line from Nastenka, independently — do not let
      whichever voice is longer or louder in this particular passage become
      your default answer. This book frequently alternates quoted lines with
      no "he said" / "she said" tag; a new quotation mark right after the
      previous one closes is usually a hand-off to the OTHER person, not the
      same speaker continuing — do not collapse an untagged line into whoever
      spoke last. Nested speech still belongs to whoever is actually saying
      it: if the Dreamer narrates a long imagined or hypothetical speech to
      someone else, that is still just his own voice and doesn't by itself
      add a second real speaker. Ensure that even brief but substantive exchanges between the Dreamer and Nastenka are recognized as qualifying for 'the Dreamer and Nastenka' in the speaking_voice field. Look for any real line from both speakers, regardless of length, to determine dual attribution. Only pick a single name when the other person is truly silent in this passage, or reduced to a bare one-word interjection with no content of its own.
      A common pattern in this book: a RUN of several short quoted lines in a
      row, back-to-back, with NO "he said"/"she said" tag anywhere in the run
      at all. Treat this as a strict back-and-forth script — line 1 is one
      speaker, line 2 is the other, line 3 is back to the first, and so on,
      alternating every single time a new quotation mark opens — do not read
      a whole untagged run as one voice just because there's no tag telling
      you otherwise partway through. In a run like this, check: does the
      alternation reach at least one real line on EACH side by the end of the
      passage? If yes, it's both of them, even though most individual lines in
      the run are short. WORKED EXAMPLE of exactly this pattern: '..."Though
      sometimes it is a good thing to dream!"... added the girl. "Excellent!
      ...Come, listen.... But one minute, I don't know your name yet." "At
      last! You have been in no hurry to think of it!" "Oh, my goodness! It
      never entered my head..." "My name is Nastenka." "Nastenka! And nothing
      else?" "Nothing else!..."' — only the FIRST line has an explicit tag
      ("added the girl"); every line after that is untagged, but it keeps
      alternating: Dreamer ("Excellent!...I don't know your name yet"),
      Nastenka ("At last!..."), Dreamer ("Oh, my goodness!..."), Nastenka ("My
      name is Nastenka"), Dreamer ("Nastenka! And nothing else?"), and so on.
      The single early tag establishing Nastenka as "the girl" does NOT mean
      the rest of the untagged run stays her voice — it still alternates. This
      whole exchange is "the Dreamer and Nastenka", not just "Nastenka" even
      though her line carries the only explicit speech tag in the passage.
      SECOND WORKED EXAMPLE, the reverse imbalance: '..."Only a compact
      beforehand...." "A compact! Speak, tell me, tell me all beforehand; I
      agree to anything, I am ready for anything," I cried delighted. "I
      answer for myself, I will be obedient, respectful ... you know me...."
      "It's just because I do know you that I ask you to come to-morrow," said
      the girl, laughing. ... "But you mustn't fall in love with me, I beg
      you!" "I swear," I cried, gripping her hand.... "Hush, don't swear...."'
      — here Nastenka's speech is much longer overall, but the Dreamer has two
      real lines of his own ("A compact! Speak, tell me...I am ready for
      anything...I answer for myself..." and "I swear," gripping her hand) —
      both tagged with "I cried", not "I said", but still real, substantive
      speech, not filler. This is "the Dreamer and Nastenka" too, even though
      Nastenka's words vastly outnumber his and even though the only "said"-
      style tags in the passage ("I cried", "said the girl") both happen to
      fall on short lines — the tag style is not a signal of who "really"
      spoke; both of them did.
    - null is EXTREMELY RARE and a last resort: only for a passage with no "I"
      at all and no one's feeling, opinion, or dialogue anywhere in it. Any
      passage using "I" ("I walked", "I felt") already has a speaking_voice —
      "the Dreamer" — even with no quoted dialogue and no explicit emotion
      word. When in doubt between null and "the Dreamer", answer "the
      Dreamer".
    - Allowed speaking_voice values for this book: "the Dreamer", "Nastenka",
      "the Dreamer and Nastenka", or null. Only use another name if a
      different character (e.g. the lodger, the grandmother) is unmistakably
      the one speaking.

SPEAKER_RELATION
- speaker_relation = what the SPEAKER is doing with the idea/feeling in the passage:
    asserts  - states or embraces it with conviction
    doubts   - voices it but with uncertainty or ambivalence
    rejects  - argues against or recoils from it
    explores - turns it over, imagines, wonders, without committing
  IMPORTANT: "explores" is NOT the default. Use it only for genuine wondering,
  imagining, or hypothesizing ("what if...", "perhaps one day..."). Judge from
  what the speaker actually does in THIS passage:
    - An emotional declaration or a stated conviction is "asserts"
      (e.g. "now I am happy", "I am a type!", "I love you", "I cannot stay here").
    - Hedged uncertainty or ambivalence ("perhaps", "I don't know", "maybe I was
      mistaken") is "doubts".
    - Recoiling from or arguing against the idea is "rejects".
  Most dialogue in which a speaker states a feeling with conviction is "asserts".
  Do NOT judge this field from the passage's SUBJECT MATTER. A passage about
  fantasy, dreaming, or imagination is not automatically "explores" — if the
  speaker states it as settled fact, with conviction, in a confident narrating
  or lecturing voice ("let me tell you...", "he is...", "I am a type!"), that
  is still "asserts", even though the content itself concerns dreams or
  fantasy. Likewise, plain narration of an event or a routine ("I bless my
  luck...", "he was pleased because...") is "asserts", not "explores" —
  narrating a fact is not the same as wondering about one. Playful tone or
  laughter in the surrounding dialogue does not make a specific stated line
  exploratory either; judge the speaker's actual sentence, not the mood of
  the exchange around it.

DISCIPLINE
- Before you finalize speaking_voice: re-check the passage one more time for
  any quoted line you may have attributed to the wrong person, or missed
  entirely, purely because it had no "he said" / "she said" tag. This is a
  known failure mode — untagged quotation marks get silently folded into
  whoever spoke last. Do not let that happen here.
- Never output commentary, explanation, or anything outside the JSON object."""

LOCAL_PASS_USER_TEMPLATE = """[PREVIOUS]:
{prev_chunk_text}

[TARGET — tag only this]:
{target_chunk_text}

[NEXT]:
{next_chunk_text}

Return the JSON object for the TARGET passage."""


def tag_local_pass(target: str, prev_text: str = None, next_text: str = None) -> LocalPassTags:
    user = LOCAL_PASS_USER_TEMPLATE.format(
        prev_chunk_text=prev_text or "(none)",
        target_chunk_text=target,
        next_chunk_text=next_text or "(none)",
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": LOCAL_PASS_SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
    )
    return LocalPassTags(**json.loads(r.choices[0].message.content))


# ---------------------------------------------------------------- PASS 2: scene

SCENE_PASS_SYSTEM_PROMPT = """You are a literary annotation engine for Dostoevsky's novella "White Nights"
(Constance Garnett translation). This is PASS 2 of 3 -- the SCENE pass. You see
the TARGET passage plus the scene(s) immediately around it (a short factual
summary, not the full text). You do NOT have the whole-book global map in this
pass -- do not reach for whole-book conclusions here; that happens in pass 3.
You are precise, cautious, and you never invent content that is not supported
by the passage or the scene context.

WHAT YOU RETURN
Return ONLY a single JSON object, no prose, no markdown fences. It must match
this schema exactly:

{
  "canonical_themes": string[],      // 0-3 items, ONLY from the allowed theme list
  "characters_present": string[]     // all named/identifiable people present or being spoken to
}

These are CANDIDATE values -- a later pass with the full-book map will confirm
or correct them, so do your best honest read of the passage and scene, but
don't agonize over edge cases only whole-book context could resolve.

CHARACTERS_PRESENT
List every OTHER named or clearly identifiable person present in the TARGET
passage or being spoken to — NEVER include the Dreamer/narrator himself in
this list, even though he is present in every passage as the narrator; that
is tracked separately and is not what this field is for. This field is empty
whenever the Dreamer is alone or the passage is pure narration with no one
else present. Use the scene context to resolve pronouns ("she"/"he") to
proper names, and use the book's own naming: the young woman is "Nastenka"
(not "the girl"), even in passages before her name is textually revealed —
treat her consistently by that name. Name other characters as the book does
(e.g. "the drunken gentleman", not "the gentleman" alone, if that is how the
scene context identifies him). Do not include people only mentioned in
passing with no real presence in the scene.

CANONICAL_THEMES
Tag what THIS passage is specifically about (0-3 themes), and prefer the MOST
specific theme(s). Do NOT tag the book's overall mood. In particular: do NOT
tag "unrequited-longing" unless the passage is specifically about longing for
a particular unattainable person. General loneliness or solitude is
"isolation"; a passage about withdrawing from society is
"alienation-from-society"; the fleeting bond itself is "ephemeral-connection".
"self-deception" is for a passage where the Dreamer substitutes an imagined or
fantasized relationship or idea for a real one, and the passage shows this
happening — small-scale counts too (a personified house or an imagined
stranger, not only grand self-crowning). "the-dreamer" is for a passage
specifically characterizing the dreamer archetype itself — his habits, his
corner, his kind of person — even in a calm, descriptive register, not only at
moments of crisis. A thin transitional passage may have 0 themes rather than
an invented one. Prefer fewer, correct tags — one solid theme beats three
loose ones. Themes must come from the allowed list, spelled exactly; anything
else does not belong in this field.

Never output commentary, explanation, or anything outside the JSON object."""

SCENE_PASS_USER_TEMPLATE = """ALLOWED THEMES: isolation, the-dreamer, unrequited-longing, ephemeral-connection,
self-deception, alienation-from-society, the-city, pride-and-humiliation,
active-love, faith-vs-doubt, suffering-as-redemption, guilt, rational-egoism,
free-will-vs-determinism, confession, innocent-suffering, moral-transgression

[NEARBY SCENES]
{scene_context}

[TARGET — tag only this]
{target_chunk_text}

Return the JSON object for the TARGET passage."""


def tag_scene_pass(target: str, scene_context_str: str) -> ScenePassTags:
    user = SCENE_PASS_USER_TEMPLATE.format(
        scene_context=scene_context_str, target_chunk_text=target,
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SCENE_PASS_SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
    )
    return ScenePassTags(**json.loads(r.choices[0].message.content))


# ---------------------------------------------------------------- PASS 3: global

GLOBAL_PASS_SYSTEM_PROMPT = """You are a literary annotation engine for Dostoevsky's novella "White Nights"
(Constance Garnett translation). This is PASS 3 of 3 -- the GLOBAL pass. You
see the TARGET passage, the scene(s) around it, AND a factual map of the whole
novella's arc (plot, character arcs, turning points, resolutions, open
questions). You are precise, cautious, and you never invent content that is
not in the passage.

WHAT YOU RETURN
Return ONLY a single JSON object, no prose, no markdown fences. It must match
this schema exactly:

{
  "narrative_relation": string,      // one of: supports | complicates | undermines | unresolved | unclear
  "canonical_themes": string[],      // FINAL 0-3 themes, adjudicating pass 2's candidate below
  "characters_present": string[]     // FINAL character list, adjudicating pass 2's candidate below
}

NARRATIVE_RELATION
narrative_relation = how the WORK ITSELF treats the idea/feeling in the TARGET
passage. Match the passage to the triggers below. Do NOT default to any single
value. In particular, "complicates" is NOT the safe fallback — pick it only
when BOTH halves below are truly present.
    supports    - the novella plainly endorses or affirms the idea. Rare here.
    complicates - the passage honors a real feeling AND the book quietly questions
                  it in the same breath (e.g. loneliness that is genuine yet
                  self-imposed). BOTH halves must be present. If only one is, this
                  is the wrong tag.
    undermines  - the passage shows the Dreamer at his most self-deceived or
                  escapist: crowning himself the unrecognized poet, declaring
                  himself "superior to all desire," burrowing fully into his shell,
                  mistaking fantasy for real passion. Here the book is EXPOSING the
                  fantasy as flight from life. Rapturous, self-glorifying fantasy
                  usually belongs HERE, not in "complicates". This includes SUBTLER,
                  quieter self-deceptions too, not only grand ones — but "undermines"
                  is NOT a default for every passage that merely mentions imagination,
                  fantasy, or a stranger; most such passages are "unclear" (neutral
                  scene-setting) or "complicates" (a real bond, gently questioned).
                  Reserve "undermines" for passages where the SUBSTITUTION itself is
                  the point: the passage stages an imagined relationship AS IF it
                  were real (a house treated as a "friend", a stranger's face read as
                  a greeting) or shows the Dreamer explicitly retreating from real
                  people into fantasy. Two real people genuinely getting to know each
                  other, or a passage that simply describes something dreamlike
                  without staging it as a substitute for real connection, is not
                  "undermines" just because feelings or imagination are involved.
    unresolved  - a parting or an emotional question the story deliberately leaves
                  open and aching: farewells, "will he come?", loss without closure.
    unclear     - transitional, scene-setting, or plot-mechanics passages that do
                  not advance an evaluated idea. Do NOT force a stance and do NOT
                  invent a theme to justify one. Use "unclear".

Do NOT pick "supports" merely because the prose is beautiful. When a passage
carries no clear evaluated idea, "unclear" is correct — not a guess.

BASE RATES IN THIS BOOK: across a typical stretch of chunks, "unclear" and
"complicates" are each about as common as "undermines" — roughly a third
each, not "undermines" dominating. If you notice yourself reaching for
"undermines" on more than about a third of the passages you tag in a row,
that is a signal you have drifted into treating it as a default, which it is
NOT — stop and re-apply the SUBSTITUTION test above before answering; most
passages that merely involve the Dreamer's imagination, mood, or interior
life at all are "unclear" or "complicates", not "undermines".

CALIBRATION
- Peak self-crowning fantasy (the Dreamer as the central figure of his own
  glory, "superior to all desire") => undermines.
- A plain scene transition ("we walked out onto the embankment; it was ten
  o'clock") with no idea under evaluation => unclear.
- A farewell that leaves the Dreamer's longing open and unanswered => unresolved.

Decide narrative_relation using the SAME five categories and calibration
above. "undermines" is NOT limited to cases where a later scene goes on to
overturn the feeling — it applies just as much WITHIN a single passage
whenever the Dreamer is shown self-deceived or escapist right there. Use the
map's important_resolutions / important_open_questions and the following
scenes as EVIDENCE for the call, not as the only route to it. Do NOT pick
"unresolved" merely because the mood is wistful; pick it only when the book
truly leaves THIS idea open. If a later scene DOES overturn the feeling in the
target, that is also "undermines"; if the book honors a real feeling while
quietly questioning it in the same breath, that is "complicates".

ADJUDICATION (canonical_themes, characters_present)
You are given a CANDIDATE canonical_themes and characters_present, produced by
an earlier pass that saw the passage and nearby scenes but NOT this global
map. Your job is to adjudicate, not to rewrite from scratch:
- Usually the candidates are correct — CONFIRM them (return them unchanged)
  rather than second-guessing a reasonable, well-supported call.
- Only CORRECT a candidate when the whole-book map reveals something the
  scene-only pass genuinely could not have seen: a character who the wider
  arc makes clear is present/implicated even if not named in this passage's
  immediate scene window, or a theme that the map shows is actually about a
  different, more specific idea once you know how this moment fits the arc
  (e.g. what looked like generic "isolation" is actually the specific
  "unrequited-longing" once the map confirms this is about a particular
  unattainable person, or vice versa — what looked like unrequited-longing is
  actually generic isolation because the map shows no specific person is
  involved yet).
- A correction should be a clear improvement backed by the map, never a
  stylistic preference. When in doubt, keep the candidate as given.
- The same theme-vocabulary discipline applies: 0-3 themes, spelled exactly
  from the allowed list, prefer fewer correct tags over more loose ones.

Never output commentary, explanation, or anything outside the JSON object."""

GLOBAL_PASS_USER_TEMPLATE = """ALLOWED THEMES: isolation, the-dreamer, unrequited-longing, ephemeral-connection,
self-deception, alienation-from-society, the-city, pride-and-humiliation,
active-love, faith-vs-doubt, suffering-as-redemption, guilt, rational-egoism,
free-will-vs-determinism, confession, innocent-suffering, moral-transgression

{global_map}

[NEARBY SCENES]
{scene_context}

[CANDIDATE canonical_themes from the scene pass]: {candidate_themes}
[CANDIDATE characters_present from the scene pass]: {candidate_characters}

[TARGET — tag only this]
{target_chunk_text}

Return the JSON object for the TARGET passage."""


def tag_global_pass(target: str, scene_context_str: str, global_map_str: str,
                     candidate_themes: list, candidate_characters: list) -> GlobalPassTags:
    user = GLOBAL_PASS_USER_TEMPLATE.format(
        global_map=global_map_str, scene_context=scene_context_str,
        candidate_themes=json.dumps(candidate_themes),
        candidate_characters=json.dumps(candidate_characters),
        target_chunk_text=target,
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": GLOBAL_PASS_SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
    )
    return GlobalPassTags(**json.loads(r.choices[0].message.content))


# ---------------------------------------------------------------- merge

# example ─ in:  (chunk_text, prev_text, next_text, scene_context_str, global_map_str)
#           out: {"narrating_voice": "the Dreamer", "speaking_voice": "the Dreamer and Nastenka",
#                 "speaker_relation": "asserts", "local_motifs": [...], "narrative_relation": "complicates",
#                 "canonical_themes": ["ephemeral-connection"], "characters_present": ["Nastenka"]}
def tag_three_pass(target: str, prev_text: str, next_text: str,
                    scene_context_str: str, global_map_str: str) -> dict:
    """Run all 3 passes and merge into a ChunkTags-compatible dict: pass 1 owns the speaker
    fields, pass 3's (adjudicated) values win for themes/characters, pass 3 alone owns
    narrative_relation."""
    local = tag_local_pass(target, prev_text, next_text)
    scene = tag_scene_pass(target, scene_context_str)
    glob = tag_global_pass(target, scene_context_str, global_map_str,
                            scene.canonical_themes, scene.characters_present)
    return {
        "narrating_voice": local.narrating_voice,
        "speaking_voice": local.speaking_voice,
        "speaker_relation": local.speaker_relation,
        "local_motifs": local.local_motifs,
        "narrative_relation": glob.narrative_relation,
        "canonical_themes": glob.canonical_themes,
        "characters_present": glob.characters_present,
    }
