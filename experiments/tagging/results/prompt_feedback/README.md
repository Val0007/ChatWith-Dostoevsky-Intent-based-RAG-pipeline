# prompt_feedback/

Raw critic proposals from `experiments/tagging/prompt_feedback.py` (finding F6, rounds 1–5): the critic model's diagnosis of the tagger's mismatches against gold and the prompt change it suggested.

**These are proposals, not the shipped prompt.** Several were judged too weak or mis-targeted and rewritten before being applied; `src/tag_white_nights.py` and `src/tag_three_pass.py` hold what actually shipped. The files are evidence of the critic's raw diagnostic quality.

| File | Round's target |
|---|---|
| `feedback_r1_undermines_softened.txt` | context arm under-recalling `undermines` |
| `feedback_r2_context_template_regression.txt` | the context template regressing after round 1 |
| `feedback_r3_asserts_vs_explores.txt` | `speaker_relation` over-predicting `explores` |
| `feedback_r4_dialogue_collapsed_to_one_voice.txt` | `speaking_voice` collapsing dialogue to one speaker |
| `feedback_r5_theme_underapplication.txt` | `self-deception` / `the-dreamer` under-applied |
