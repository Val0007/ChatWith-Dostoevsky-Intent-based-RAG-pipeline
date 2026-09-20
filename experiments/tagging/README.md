# experiments/tagging/

The tagging study: how to get an LLM to label *White Nights* chunks with `narrative_relation`, `canonical_themes`, `speaker_relation`, etc. when the right label is a matter of interpretation.

- **Findings log:** [`FINDINGS.md`](FINDINGS.md) (F1–F14, plus the open questions Q1–Q12 that structured the study).
- **Evaluation:** every arm/config is scored field by field against the 30-chunk gold set in [`gold/`](gold/). Single-label fields → accuracy + per-class P/R; set fields (`canonical_themes`, `characters_present`) → micro P/R/F1.
- **Code under test:** `src/tag_white_nights.py` (single call) and `src/tag_three_pass.py` (three passes).

## Scripts

| Script | Finding | What it does |
|---|---|---|
| `ctx_vs_local.py` | F1 | 5-chunk probe: local (prev+next) vs context-aware (scene cards + global map) |
| `analyze_tagger.py` | F2, F3 | Free analyses of `data/tags.jsonl`: within-scene field agreement, within-Night theme diversity. **Stale:** still reads the pre-split `speaker` field, so it fails on current tags |
| `narrative_context_ablation.py` | F5 | Holds the prompt fixed, varies only context data (none / map / scenes / both) on 7 gold chunks, N runs each |
| `run_first30_eval.py` | F6 | Local vs context-aware on all 30 gold chunks. **Also the shared scoring library** (`load_gold`, `score_single_label`, `score_set_field`) imported by most scripts below |
| `prompt_feedback.py` | F6 | Round-based critic feedback on the single-call prompt (proposals are printed, never auto-applied) |
| `iterate_hard_chunk.py` | F7 | Tight loop on ONE persistently-wrong chunk, one arm at a time, with full patch history |
| `run_four_arm_ablation.py` | F8 | The core ablation: local / +scenes / +global map / both, every field |
| `run_five_arm_comparison.py` | F9 | Adds arm E (three-pass pipeline) to the four arms and scores all five |
| `iterate_pass.py` | F10 | Re-run and score ONE pass of the three-pass pipeline, reusing cached outputs for the others |
| `critic_fixer_loop.py` | F10 | Critic → fixer → patch → rerun → verify, one pattern at a time |
| `critic_fixer_loop_v2.py` | F13 | Same loop, but the critic sees the full prediction set and the round history |
| `apply_patch.py` | F10 | Applies a fixer's old→new patch with whitespace-insensitive matching |
| `tag_full_book_g.py` | F13 | Tags all 84 chunks with the locked config G and writes `data/tags.jsonl` (**API cost**) |
| `rebuild_db_from_tags.py` | — | Rebuilds the Chroma DB from the existing `tags.jsonl` (embedding calls only) |
| `score_pass2_raw.py` | after F14 | Scores pass 2 alone, before any adjudication — how the "candidate" framing was diagnosed (see below) |

Free to run (cached results, no API): `run_first30_eval.py --score-only`, `run_five_arm_comparison.py --score-only`, `score_pass2_raw.py` (uses `results/pass2_raw_output.json` if present).

## Folders

| Folder | Contents |
|---|---|
| [`gold/`](gold/) | The hand-built 30-chunk gold set and the rationale for every non-obvious label |
| [`results/`](results/) | Cached model outputs, ablation tables, configs and per-round logs |
| [`prompt_snapshots/`](prompt_snapshots/) | Copies of `tag_three_pass.py` at key points in the prompt-fixing history, kept so diffs can be reproduced |

## Not yet in FINDINGS.md

`FINDINGS.md` ends at F14 and describes the *adjudicating* three-pass design. After it, three things were measured and the pipeline was rebuilt so **no pass reads or revises another pass's fields**:

1. `score_pass2_raw.py` showed pass 2's raw themes already scored F1 0.62 (below the single-call 0.67) — pass 3's adjudication changed only 2 of 30 chunks, so it wasn't the cause. The "you are only a candidate, don't agonize" framing made pass 2 over-tag (2.0 themes/chunk vs 1.2).
2. After removing the candidate framing and the adjudication step, themes recovered to 0.67; `characters_present` did not (0.54).
3. Stripping pass 2's `characters_present` rules to match the single-call prompt's one-line description collapsed it to 0.25 (`results/pass2_minimal_charprompt_output.json`) — the field's discipline had been coming from being decided in the same call as `speaking_voice`.

`results/pass2_raw_output.json` is the pass-2 output under the **old** candidate framing; `results/five_arm_comparison_output.json` is the rebuilt pipeline.
