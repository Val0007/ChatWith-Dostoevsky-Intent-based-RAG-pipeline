# critic_fixer_logs/

Per-round logs of the automated critic → fixer → patch loops (findings F10 and F13), written by `critic_fixer_loop.py` and `critic_fixer_loop_v2.py`.

File names are `<pass>_r<round>_<stage>.txt`, with a `v2_` prefix for the F13 loop:

| Stage | Contents |
|---|---|
| `_critic` | the critic's grouped failure patterns (prompt-caused vs residual model error) |
| `_pattern1` | the single top-priority pattern handed to the fixer |
| `_fixer` | the fixer's exact old→new patch — including the rejected ones, e.g. `v2_local_r2_fixer.txt`, which deleted a load-bearing guard and dropped `asserts` recall 73% → 50% |

`v2_history.json` is the running history the v2 critic is shown. Of five patches across passes, one was kept (a `speaking_voice` dual-voice fix, 90 → 93%).
