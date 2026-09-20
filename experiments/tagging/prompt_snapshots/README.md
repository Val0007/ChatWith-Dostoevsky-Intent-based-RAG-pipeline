# prompt_snapshots/

Frozen copies of `src/tag_three_pass.py` at key moments in the prompt-fixing history, taken so the revert-safety net didn't live in `/tmp`. They exist so real diffs can be reproduced (see F14 in `../FINDINGS.md`).

| File | State |
|---|---|
| `tag_three_pass_backup_r1.py` → `..._r2.py` | before / after the one *kept* F10 patch (`speaking_voice` dual-voice fix) |
| `tag_three_pass_backup_scene_r1.py`, `..._global_r1.py` | before the rejected scene / global pass patches |
| `tag_three_pass_backup_before_strip.py` | after F11 added gold examples for both `speaking_voice` and `speaker_relation` |
| `tag_three_pass_v2_r1_before.py` | the locked "config G" state (only `speaker_relation` examples kept) |
| `tag_three_pass_v2_r2_before.py` | before v2 round 2 |

**Historical snapshots — diff them, don't run or import them:** `diff prompt_snapshots/tag_three_pass_backup_r1.py prompt_snapshots/tag_three_pass_backup_r2.py`.
