# Gold set — first 30 chunks — rationale

`gold_first30.jsonl` is a hand-built gold standard for `white_nights_first_night_001..018` +
`white_nights_second_night_001..012` (all of First Night, first 12 chunks of Second Night — the
scene where the Dreamer meets the not-yet-named girl through his "I am a type!" dreamer monologue).
Same schema as `tags.jsonl` / `ChunkTags` in `src/tag_white_nights.py`, validated against it.

Purpose: run the tagger (local and context-aware) over these same 30 chunks and diff against this
file, field by field — per the plan at the bottom of `FINDINGS.md`. This is meant to be a reference
someone can disagree with, not a ground truth handed down from on high — every non-obvious call is
explained below so a disagreement can be resolved by re-reading the passage, not by trusting the file.

## Schema change — `speaker` split into `narrating_voice` + `speaking_voice`

Per Phase 02 of `docs/tagger_study_plan.html`. `narrating_voice` is a constant — "the Dreamer" on
every one of these 30 rows, since the whole book is narrated first-person by him; nothing to judge
here. `speaking_voice` absorbs everything the old `speaker` field did (who's actually talking/
expressing themselves in the passage) and is mechanically identical to the old `speaker` value for
all 30 rows: none of these 30 chunks turned out to be pure external description with zero
interiority (the book's narration is consistently first-person/reflective), so `speaking_voice` was
never set to `null` here. That itself is worth knowing — the null case may simply not occur in
*White Nights*, or may only show up later in the book (plain scene-transition sentences with no
"I felt/I said" at all). Don't be surprised if the tagger never predicts `null` either.

## Modeling decisions (apply to every row)

- **`characters_present` / `speaker` name the girl "Nastenka" from chunk 1 onward**, even though the
  text itself doesn't reveal her name until `second_night_004` ("My name is Nastenka"). Two reasons:
  (1) `speaker`'s allowed vocabulary in the schema is a closed set — `"the Dreamer"`, `"Nastenka"`,
  `"the Dreamer and Nastenka"` — there is no "the girl" option, so the field forces this choice
  regardless; (2) gold is meant to be the *correct* answer for a reader who has read the whole book,
  not a simulation of what's inferable chunk-by-chunk. This is a deliberate, known departure from
  strict textual-order faithfulness — worth remembering if you see the tagger use "Nastenka" early
  and are tempted to call that a miss. It isn't, under this convention. (It's also worth separately
  checking whether the *context-aware* tagger's early "Nastenka" is coming from the global map, vs.
  the *local* tagger's, which could only be leaking it from parametric/pretraining knowledge of the
  book — that's Q11 in FINDINGS.md, and this gold set can't distinguish the two on its own.)

- **`unrequited-longing` is withheld from all 30 rows on purpose**, even where the Dreamer is
  clearly lonely or fantasizing about connection (e.g. 004, 012). Per the tagger's own discipline
  note, that theme is reserved for longing after "a particular unattainable person" — no such person
  exists yet in this stretch (the rival lodger hasn't been introduced). Generic loneliness is
  `isolation`; social exclusion is `alienation-from-society`; the fragile Dreamer/Nastenka bond
  itself is `ephemeral-connection`. If the tagger applies `unrequited-longing` anywhere in these 30,
  treat it as a live instance of the exact over-application failure F3 already flagged (theme
  collapsing to the mood-word under context pressure) — that's the single most useful signal this
  gold set can give the study.

- **`speaker_relation` is not defaulted to `asserts`.** First pass over these 30 leaned asserts
  almost everywhere; on a re-read several chunks are genuinely imaginative/hypothetical registers
  ("I fancied…", "who knows, maybe…", cataloguing a daydream) rather than stated convictions, and got
  moved to `explores` (005, 007, 012, 015, 028, 030). Two chunks (014, 019) get `doubts` because the
  text itself hedges ("surely…surely…", "I…felt doubtful"). `rejects` doesn't occur anywhere in this
  range — no one recoils from or argues against an idea until the rival/jealousy material later in
  the book, so its absence here is expected, not a gap.

## Field-by-field notes on the non-obvious `narrative_relation` calls

Distribution across the 30: `unclear` ×10, `complicates` ×10, `undermines` ×8, `unresolved` ×2,
`supports` ×0. No `supports` is expected — the calibration note in the system prompt says it's rare,
and nothing in this stretch is a plain, uncomplicated endorsement.

- **002, 003 → `undermines`.** The house/old-man personification isn't cute scene color — it's the
  Dreamer substituting imagined relationships with buildings and strangers for real ones, and the
  "pink house painted yellow" bit is staged as a literal betrayal. This is the mild, comic register
  of the same self-deception pattern that peaks in Second Night; tag it the same way even though it's
  gentler.
- **004, 005 → `unclear`.** Comic character quirks (chair-position fixation, classifying strangers by
  summer villa) — establishing/scene-setting, no clear book-level judgment landed on it yet within
  the chunk itself. Close call against `undermines`; kept `unclear` because nothing here is exposed
  as *self-deceptive* the way 002/003's personification or Second Night's monologue explicitly is.
- **006 → `complicates`.** Real pain of exclusion ("no one invited me… they had forgotten me") sits
  right next to a fast, self-generated swing to lightness once he's alone outside the city gates —
  both halves (genuine feeling + the book quietly showing it's partly self-authored/labile) present
  in the same chunk.
- **009 → `unclear`.** This is the global map's "major turning point" (the rescue), but it's pure
  plot mechanics — no evaluated idea in the passage itself. A structurally important scene can still
  be `unclear`; importance to the plot arc and narrative_relation are different axes.
- **012 → `undermines`.** The imagined-approach-to-a-stranger-lady fantasy is a textbook instance of
  substituting fantasy for real connection — same self-deception pattern as 002/003/Second Night,
  just delivered as a hypothetical (hence `explores` on speaker_relation, `undermines` on
  narrative_relation — these two fields diverge here, which is the point of keeping them separate).
- **013–017 → `complicates` (except 014, 018 → `unresolved`).** This whole stretch is the real bond
  forming between the two, and nearly every chunk explicitly pairs the warmth with something that
  undercuts or hedges it in the same breath (her "not an appointment" qualifier, the "don't fall in
  love with me" condition placed inside a growing-trust scene, the deferred secret). 014 and 018 are
  the two chunks that instead land squarely on a parting/open question ("surely this is not to be the
  end?"; "Till to-morrow!") with no complicating counter-note — that's what earns `unresolved`
  specifically, per the calibration ("a parting… deliberately left open and aching").
- **019, 021, 025, 029 → `complicates`.** Same pattern as above transposed into Second Night: real
  affection or recognition (Nastenka's return, "I am a dreamer myself", her guessing his secret, the
  post-fantasy emptiness) each paired with an explicit qualifying or questioning note in the same
  chunk.
- **005, 010, 011, 020, 022, 027 → `unclear`.** Backstory/exposition/banter that doesn't yet land an
  evaluated idea — grouped here rather than force-fit into complicates/undermines.

  Correction: 010 is tagged `complicates` in the file, not `unclear` — the rescue's immediate
  aftermath already has a real, fragile bond forming (playful banter, taking his arm) shadowed by the
  book-length awareness that this connection is inherently unequal/fleeting. Listed here only to flag
  it as a genuinely close call against `unclear`; a reasonable second annotator could go either way.
- **023, 024, 026, 028, 030 → `undermines`.** The core of the "I am a type!" dreamer monologue —
  self-portrait as escapist/self-enclosed (023, 024), grandiose self-mythologizing (King Solomon's
  seal, 026), literally losing track of physical reality mid-fantasy (028), and the clearest instance
  in the whole set — crowning himself "the most prominent figure… in his precious person" and pitying
  real people, including Nastenka, as dull by comparison (030). If the tagger gets any chunk in this
  set right, it should be 030 — it's the calibration example ("superior to all desire" energy) almost
  verbatim.

## Known soft spots in this gold set

- `local_motifs` are illustrative, not exhaustive or uniquely-correct — don't diff those field-for-
  field; use them only to sanity-check the tagger noticed the same salient details.
- `speaker`, `speaker_relation`, `characters_present`, and `canonical_themes` are still single-
  annotator only (see cross-validation note below — only `narrative_relation` has been independently
  checked so far). Treat those fields with the same "strong signal, not proof" caution FINDINGS.md
  applies elsewhere, especially the `speaker_relation` calls that got moved off `asserts` (005, 007,
  012, 015, 028, 030) and the two `doubts` calls (014, 019) — those were a single re-read, not a
  second opinion.

## Cross-validation — narrative_relation (second annotator, independent LLM pass)

All 30 `narrative_relation` labels, including all 15 flagged as close calls (see above — 001-013
first-night range, 003/009/010 second-night range), were independently reviewed by a second model
given the same passages and the same schema. **Result: 30/30 confirmed, 0 flips.** Confidence on the
review ran high-to-very-high on all but a handful (FN_002, FN_006, FN_010, FN_013, FN_015, FN_016,
FN_017, SN_002, SN_007, SN_011 came back "high" rather than "very high" — the same chunks this file
already flagged as closer calls, for the same reasons).

This doesn't retroactively make the individual close calls *not* close calls — a second reader who
agrees on the label can still be weighing the same genuine ambiguity the same way, not resolving it.
But two independent reads landing on the same 30/30 is a meaningfully stronger basis than one pass,
and is enough to treat `narrative_relation` in this file as validated for the tagger-comparison run.
The other fields (`speaker`, `speaker_relation`, `characters_present`, `canonical_themes`) have not
had this pass yet.
