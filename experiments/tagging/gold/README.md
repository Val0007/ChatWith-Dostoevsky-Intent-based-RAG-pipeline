# gold/

The reference the tagger is scored against.

| File | Contents |
|---|---|
| `gold_first30.jsonl` | 30 hand-labelled chunks — all of First Night (`001`–`018`) plus the first 12 of Second Night. Same schema as `data/tags.jsonl`: the seven tags per chunk. `narrative_relation` was cross-checked by a second independent annotator pass (30/30 agreement). |
| `gold_first30_notes.md` | Why each non-obvious call was made, so a disagreement can be settled by re-reading the passage rather than trusting the file. Includes the convention that `characters_present` uses "Nastenka" from chunk 1 even though the text doesn't name her until Second Night. |

Caveats: 30 chunks is small, several calls are genuinely arguable (e.g. who speaks in `second_night_001/002`), and one convention (the Nastenka backfill) makes some `characters_present` "misses" arguably correct. Treat this as a reference someone can disagree with.
