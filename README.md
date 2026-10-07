# NoRA: Evaluating Grounded Reasonableness in Visual First-person Normative Action Reasoning

Sichao Li, Sai Ma, Daniel Kilov, Secil Yanik Guyot, Zhuang Li, Seth Lazar*

[Website](https://mint-philosophy.github.io/NoRA-benchmark/) |
[Paper](https://arxiv.org/abs/2606.04806) |
[Dataset](https://huggingface.co/datasets/MINTLABJHUANU/NoRA) |
[Model guide](docs/models.md) |
[Scoring guide](docs/evaluation.md) | 
[MINT Lab](https://mintresearch.org/)


NoRA evaluates the actions a model proposes, the facts it observes, and the
reasons connecting them. It includes 190 human-annotated examples
(HumanGold) and 1,230 model-annotated training examples (LLMSilver).

Use this package to evaluate your own model. Run it on the provided visual inputs
and prompts, then score its predictions against the test annotations.
If you already have predictions, skip to [scoring](#4-score-predictions).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Sichao-Li/NoRA-benchmark.git
cd NoRA-benchmark
uv sync --locked
uv run nora demo
```

This creates a local `.venv/` and runs a small synthetic example.
The demo needs no GPU or API key; its scores are **not benchmark results**.

## 1. Download the visual inputs

Download the original **pre-action frame montages** for the 190 test clips:

```bash
uv run nora download-media --media frames --output data/media
```

For a video-capable model, use `--media video` here and in the prediction command.
The files come from [EgoNormia](https://huggingface.co/datasets/open-social-world/EgoNormia)
and remain subject to its access and usage terms. No during-action inputs or
reference answers are sent to the model.

For a two-clip trial, add `--limit 2` to both downloading and prediction.

## 2. Generate predictions

Use an image-capable model through a compatible Chat Completions endpoint.
Replace `YOUR_MODEL` with the model name served by your endpoint. The local
example assumes your model server is already running; for a hosted provider,
replace `--base-url` with its HTTPS endpoint and set `NORA_API_KEY`.

```bash
export NORA_API_KEY="YOUR_INFERENCE_API_KEY"  # Omit for an unauthenticated local server.
uv run nora predict \
  --base-url http://localhost:8000/v1 --model YOUR_MODEL \
  --media frames --media-root data/media \
  --prompt structured --max-tokens 4096 \
  --output runs/my-model-structured/raw.jsonl
```

The runner supplies the selected prompt automatically. Choose one mode per run:

| `--prompt` | What the model is asked to produce |
| --- | --- |
| `direct` | One concise next action, without an explanation. |
| `deliberate` | Several candidate actions with supporting facts and reasons, then a selected action. |
| `structured` (default) | Facts, tagged reasons citing fact IDs, candidate actions citing reason IDs, and a selected action ID. |

All modes use the camera wearer's first-person perspective. The exact system
and user prompts are in [prediction_prompts.json](src/nora/assets/prediction_prompts.json).

**API runner settings:** temperature `0`, maximum output `4096` tokens by default,
one request per clip, and a `180`-second request timeout. No seed or `top_p` is sent.
Native thinking is not enabled or disabled by the runner; configure and record
it in your endpoint or [Python callback](docs/models.md#model-callbacks).
The prompt mode and native thinking mode are separate settings.

Responses are saved after each clip with the model, prompt, and modality labels.
Use a different output path for each model and prompt; existing files are not
overwritten. Failed or truncated responses are recorded as failures.

See the [model guide](docs/models.md) for Python models, video endpoints, custom
settings, and prediction formats. For reproducing a paper experiment, match that
experiment's prompts and settings; see [reproducibility](docs/evaluation.md#reproducibility).

## 3. Convert responses to annotations

The provided prompts produce text. Before scoring, reconstruct its facts,
reasons, actions, and links with a text model. Set your API key and replace
`YOUR_EXTRACTOR_MODEL` with a model supported by the OpenAI Responses API:

```bash
export OPENAI_API_KEY="YOUR_RECONSTRUCTION_API_KEY"
uv run nora reconstruct \
  --input runs/my-model-structured/raw.jsonl \
  --model YOUR_EXTRACTOR_MODEL \
  --output runs/my-model-structured/annotations.jsonl
```

Reconstruction uses only the model response, never reference annotations or
images. It is instructed to preserve the answer rather than improve it.
Inspect the extracted annotations and keep the same extractor settings across
models. Failed candidates and safe error codes are saved separately in
`annotations.jsonl.failures.jsonl`.

If your model already produces [annotation dictionaries](docs/models.md#annotations),
skip reconstruction and score that JSONL file directly.

**Costs:** hosted inference and text reconstruction may incur API fees.
Validation, scoring, tests, and the demo make no paid API calls.

## 4. Score predictions

```bash
uv sync --locked --extra scorer
uv run nora validate runs/my-model-structured/annotations.jsonl
uv run --extra scorer nora evaluate \
  --predictions runs/my-model-structured/annotations.jsonl \
  --output runs/my-model-structured/scores --progress
```

The package includes the human test references. The semantic scorer
downloads on first use; once cached, add `--offline`. CPU scoring is supported.
Use `--references path/to/reference.jsonl` for a different reference set.

Read `summary.json` for aggregate action, fact, support, and reasonableness
scores; `instance_metrics.csv` for per-clip scores; and `coverage.json` for
valid, invalid, and missing counts. Both soft and Hungarian alignment are reported.

All submitted actions are scored, including actions without support. Invalid
and missing predictions are recorded separately, not assigned zeros. Incorrect
predictions that meet the format requirements still count.

An evaluation with no valid predictions saves diagnostics and exits with failure.
Valid subsets still succeed, with missing and invalid cases reported separately.

## Compare models

```bash
uv run nora compare runs/model-a/scores runs/model-b/scores
```

This compares models on the same valid clips and checks that evaluation settings
match. Always report coverage alongside scores: a valid-subset comparison is not
a full-test result. See the [scoring guide](docs/evaluation.md) for definitions,
output files, and comparison requirements.

For issues or contributions, see [Contributing](CONTRIBUTING.md).

## Citation and license

Please cite the [NoRA paper](https://arxiv.org/abs/2606.04806);
citation metadata is in [CITATION.cff](CITATION.cff).

Code is licensed under [MIT](LICENSE). Dataset annotations are
[CC BY-NC 4.0](LICENSE-DATA). Source images and videos have separate access and usage terms.

