# docs/

Planning documents written before and during the experiments. They describe *intent*; the results are in [`experiments/`](../experiments/).

| File | What it is |
|---|---|
| `dostoevsky-bot-build-guide.md` | The original beginner's build guide: build the dumbest end-to-end version first (chunk → embed → retrieve → generate), then add one layer at a time, testing after each. The stage numbers in `src/` docstrings ("Stage 3 tagging", "Stage 4 retrieval", "Stage 5 conversation") refer to this guide. |
| `tagger_study_plan.html` | *White Nights Tagger Study* — the phased plan for the tagging experiments (context ablation, schema split, gold-set scoring). Findings: [`experiments/tagging/FINDINGS.md`](../experiments/tagging/FINDINGS.md). |
| `rag_lab_plan.html` | *White Nights RAG Lab* — the plan for the retrieval ladder (configurable retrieval presets, each adding one mechanism). Findings: [`experiments/retrieval/FINDINGS.md`](../experiments/retrieval/FINDINGS.md). |

Open the `.html` files in a browser (they pull Google Fonts; offline they just fall back to system fonts).
