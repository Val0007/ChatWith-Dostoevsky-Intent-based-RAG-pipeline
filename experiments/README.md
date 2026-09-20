# experiments/

Two studies, each with its own gold set, scripts, cached results and running findings log.

| Study | Question | Log | Gold set |
|---|---|---|---|
| [`tagging/`](tagging/) | What context does an LLM need to tag an interpretive text, and how should the tagging prompt be structured? | [`FINDINGS.md`](tagging/FINDINGS.md) — F1–F14 | 30 hand-labelled chunks |
| [`retrieval/`](retrieval/) | Which retrieval strategy finds the right passages for questions about an interpretive text? | [`FINDINGS.md`](retrieval/FINDINGS.md) — R1–R24 | 10 hand-written questions |

Retrieval builds on tagging (the tags feed the metadata signal and the answer prompt), so read tagging first.

## Conventions

- **Run from the repo root** with the project venv: `.venv/bin/python experiments/<study>/<script>.py`. Scripts set up their own `sys.path` and locate data through `ROOT / "data" / ...`.
- **Every script caches its raw model output as JSON under that study's `results/`** so scoring can be re-run without spending API credits. Look for a `--score-only` flag or a "Reusing cached ..." message.
- **Scripts that call the API say so in their docstring.** Tagging scripts are the expensive ones (chunks × arms × calls); the retrieval pilots cost one embedding call per question unless they use an LLM stage.
- **Numbers before conclusions.** Each `FINDINGS.md` entry states what was tested, the result, and the takeaway — and later entries correct earlier ones (F5 shows F1's "context wins" was a prompt confound).
- **Small-n warning.** 30 tagged chunks and 10 questions: a difference of one or two items is noise. Several entries measure that noise directly (determinism runs, repeated configs).
