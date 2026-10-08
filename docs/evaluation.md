# Scoring guide

NoRA compares predicted actions and their supporting facts and reasons with
reference annotations. Scores measure agreement with these references, not
universal moral correctness.

## Metrics

Each example is scored with both soft and Hungarian alignment. Soft alignment
allows multiple items to match the same counterpart; Hungarian alignment uses
one-to-one matching.

| Metric | What it measures |
| --- | --- |
| `action_f1` | Alignment between predicted and reference actions. |
| `fact_f1` | Alignment between their shared fact pools. |
| `reasoning_f1` | How well facts and reasons support aligned actions. |
| `summary_score` | Arithmetic mean of the available components above. |
| `reasonableness_score` | Geometric mean of those components. |

Facts or support in either prediction or reference make that component
available. Missing predicted support therefore does not remove the component
when the reference contains it. Dataset-level scores are means of per-example
scores.

Every submitted candidate action is retained, even when its supporting-reason
list is empty. Annotations contain supporting links only. Direct predictions and
text reconstruction use the same annotation format and validation. The advanced
graph format accepts saved action-rooted graphs; comparisons require matching
formats.

For annotation-format references, the fact pool contains facts linked through
reasons to actions, not every top-level fact. Reason `text`, action
`description`, the first reason tag, and graph links enter scoring. Separate
reason `justification` and action-link `explanation` fields are not scored as
additional text.

The detailed output also includes chosen-action metrics. If the model does not
specify a chosen action, the evaluator can fall back to the first
action. Do not interpret that fallback as an explicit model choice.

## Validity and coverage

Format validity is separate from answer quality. A valid but poor prediction
counts, including an empty action set. Missing predictions and invalid graphs
are excluded and reported separately. Graphs are not automatically repaired.

Duplicate or unknown clip IDs, malformed JSONL, and invalid reference data stop
the run. Scorer failures also stop the run rather than selectively removing
examples. If no valid examples remain, scores are null, not zero.

The CLI returns exit code 1 when no cases are scoreable. It still saves coverage
and failure details. A valid subset succeeds with `status: partial`; missing
clips are not assigned zeros. A full valid run has `status: complete`.

Common reasons for zero valid cases are raw `response` text submitted before
`nora reconstruct`, failed generations, malformed annotations, or an empty
prediction file. See `failures.jsonl` for the specific reason; do not turn
invalid outputs into valid ones by inventing missing content.

`nora compare` recalculates each model's means on the intersection of valid
clip IDs. It checks that references, scoring code, formats, declared prompts,
and reconstruction settings match. Report the shared sample count and each
model's coverage. Validity-conditioned comparisons may be selection-biased.

## Output files

| File | Contents |
| --- | --- |
| `summary.json` | Mean scores and sample counts for both alignments. |
| `coverage.json` | Expected, submitted, valid, missing, and invalid counts. |
| `instance_metrics.csv` | The five main metrics and availability flags per example. |
| `instance_metrics.jsonl` | Full results, including chosen-action metrics and diagnostics. |
| `failures.jsonl` | Missing or invalid predictions and their failure reasons. |
| `cohort_ids.json` | The exact clip IDs scored. |
| `run.json` | Input and code hashes, scorer version, prompts, and environment metadata. |

Choose a new output directory for each run; existing results are not overwritten.

To evaluate saved action graphs against a specific reference file:

```bash
uv run --extra scorer nora evaluate \
  --predictions predictions.jsonl --prediction-format graph \
  --references references.jsonl --offline --progress --output runs/my-model
```

`--progress` writes to stderr; the final JSON remains on stdout. Scoring batches
and caches repeated text pairs within each clip. Minor floating-point differences
can occur across devices and batch sizes.

## Reproducibility

The default test reference is the test split on [Hugging Face](https://huggingface.co/datasets/MINTLABJHUANU/NoRA)
and loads without network access.

The training loader retrieves the corresponding 1,230-clip training split from:

```python
from nora import load_references

train = load_references("train")
```

The scorer uses `cross-encoder/stsb-roberta-base` at revision
`d576534b67143e2c70ee9966d7fdbf5835728d13`.
If the scorer cannot load, evaluation fails rather than using the demo backend.

Record the model and checkpoint, exact prompt text, visual modality, decoding
parameters, native thinking setting, and reconstruction model. The API runner
defaults to temperature `0` and a `4096`-token output limit; these are runner
defaults, not a claim that every paper experiment used those settings.

The [bundled prompts](../src/nora/assets/prediction_prompts.json) are the original
paper prompts, also [distributed with the dataset](https://huggingface.co/datasets/MINTLABJHUANU/NoRA/blob/main/prompts/prediction_prompts.json).
Their IDs are `direct`, `deliberate`, and `structured`.
For model comparisons, keep prompts, candidate inclusion, reconstruction, and
scoring settings fixed. `run.json` records the scoring protocol and available
prediction metadata; it does not recover unreported provider settings.

For the paper's saved action graphs, use `--prediction-format graph`.
To reproduce a reported experiment, match its reference annotations, visual
inputs, prompts, predictions, reconstruction, and inclusion policy.
Direct graph generation and text reconstruction are different procedures.

## Dataset limitations

The annotations are finite reference sets. Appropriate actions can depend on
unseen context, and agreement with an annotation is not a deployment safety
guarantee.
