# data/

Everything the pipeline reads or produces. Paths are resolved from the repo root in `src/` (`ROOT / "data" / ...`).

| Path | What it is | Made by | Tracked? |
|---|---|---|---|
| `white_nights.txt` | The source novella (Garnett translation, PDF-extracted text with a front-matter preamble) | given | yes |
| `context/scenes.json` | 27 scenes: which of the 84 chunk ids belong to each | `src/scenes.py` | yes |
| `context/scene_cards.json` | One **factual** evidence card per scene (summary, characters, beliefs/feelings, developments, consequences) — no interpretation | `src/scenes.py` | yes |
| `context/global_map.json` | The whole-book map: `plot_arc`, `dreamer_arc`, `nastenka_arc`, `relationship_arc`, `major_turning_points`, `important_resolutions`, `important_open_questions` | `src/scenes.py` | yes |
| `tags.jsonl` | One row per chunk (84): `id`, `scene_id`, and the seven tags — `narrating_voice`, `speaking_voice`, `speaker_relation`, `local_motifs`, `narrative_relation`, `canonical_themes`, `characters_present` | `src/tag_three_pass.py` via `src/ingest.py` or `experiments/tagging/tag_full_book_g.py` — last generated Aug 26 (config G), before the deterministic three-pass rebuild | yes |
| `db/` | Chroma vector store, collection `dostoevsky` (chunk text + embeddings + tags as metadata) | `src/ingest.py` or `experiments/tagging/rebuild_db_from_tags.py` | **no** (gitignored, rebuildable) |

## Rebuilding

- **Only the vector DB is missing** (fresh clone): `python experiments/tagging/rebuild_db_from_tags.py` — embeds the 84 chunks, no tagging calls.
- **Everything, from the raw text** (costs API credits): `python src/ingest.py` — regenerates scenes, scene cards, global map, tags, and the DB.

## Chunk ids

`white_nights_<chapter>_<NNN>` — chapters `first_night`, `second_night`, `third_night`, `fourth_night`, `morning`; numbering restarts per chapter. The 84 chunks are ~350 tokens with a 50-token overlap that never crosses a chapter boundary.
