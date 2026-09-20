# iterate_hard_chunk/

Output of `experiments/tagging/iterate_hard_chunk.py` (finding F7): tight, single-chunk prompt iteration on the hardest `speaking_voice` cases, one arm (local) at a time.

Files are named `iterate_<chunk id>_<arm>_<field>` with a suffix:

| Suffix | Meaning |
|---|---|
| *(none)* `.json` | the tagger's output for that chunk at the current prompt state |
| `_patches.json` | every patch tried on that chunk so far, so the critic can't re-propose a rejected one |
| `_roundN.txt` | the critic's output for round N |

`first_night_016` took 8 attempts and never converged — logged as a model capability ceiling, not a prompt gap.
