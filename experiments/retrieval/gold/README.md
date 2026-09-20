# gold/

`retrieval_gold_pilot10.json` — 10 hand-written questions spanning First Night to Morning. Each entry:

| Field | Meaning |
|---|---|
| `id` | `q01`–`q10` |
| `night` | which part of the book the answer lives in |
| `intent` | the question type as labelled by hand: `factual`, `character`, `motivation`, `interpretive`, `evaluation`, `comparative`, `development` (separate from the regex classifier's labels in `production_method.py`) |
| `question` | the reader's question |
| `expected_answer` | the answer, checked against the source text |
| `gold_evidence` | 1–2 chunk ids that contain the answer — verified by grepping `data/white_nights.txt`, not assumed |

A question is a **hit@6** when any `gold_evidence` id is in the top 6 retrieved. Ten questions is a small set: one question is 10 points, so treat a one-question difference as noise.
