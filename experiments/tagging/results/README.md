# results/

Cached model outputs and logs from the tagging experiments. Machine-written outputs. Chunk-keyed JSON maps `chunk id → tags` for the 30 gold chunks unless noted.

| File | From | What it is |
|---|---|---|
| `first30_tagger_output.json` | `run_first30_eval.py` | Local vs context-aware (single-call) tags for the 30 gold chunks — the latest cached run |
| `first30_tagger_output_baseline_f6.json` | F6 | Same, snapshot **before** any prompt fixes (the pre-fix baseline) |
| `first30_tagger_output_round5_overcorrected.json` | F6 | Snapshot after round 5, labelled 'overcorrected' — `undermines` was being predicted far too often |
| `first30_tagger_output_round6_narrative_fixed.json` | F6 | Snapshot after round 6's `narrative_relation` correction |
| `four_arm_ablation_output.json` | `run_four_arm_ablation.py` (F8) | Arms A–D: local / +scenes / +global map / both |
| `five_arm_comparison_output.json` | `run_five_arm_comparison.py` (F9) | Arm E: the rebuilt three-pass pipeline (no adjudication) |
| `pass2_raw_output.json` | `score_pass2_raw.py` | Pass 2 alone under the **old** "candidate" framing (themes F1 0.62) |
| `pass2_minimal_charprompt_output.json` | ad-hoc test | Pass 2 with `characters_present` rules stripped to the single-call prompt's one-line description (F1 0.25) |
| `determinism_run1–3.json` | F10 | Three reruns of the same frozen pipeline at temperature 0 — the measured noise band |
| `final_v2_config.json`, `locked_final_config.json` | F13 | Per-chunk tags from the pipeline as frozen at the end of the v2 loop / as locked ("config G"). Despite the names these are outputs, not settings |
| `pass_critic_global.txt` | `iterate_pass.py` | Raw critic output for the global pass (F9) |

| Subfolder | Contents |
|---|---|
| [`iterate_hard_chunk/`](iterate_hard_chunk/) | Per-chunk iteration histories from `iterate_hard_chunk.py` (F7) |
| [`prompt_feedback/`](prompt_feedback/) | Raw critic proposals from `prompt_feedback.py` rounds 1–5 (F6) |
| [`critic_fixer_logs/`](critic_fixer_logs/) | Critic / fixer / pattern text per round of the F10 and F13 loops |
| [`pass_cache/`](pass_cache/) | Per-pass output caches used by `iterate_pass.py` |
