# Use your model

You can evaluate saved predictions, connect a Python model, or call an
image- or video-capable API. NoRA does not require a particular inference framework.

## Prediction formats

Save one JSON object per line. The examples below are expanded for readability;
replace their clip IDs and content with your model's predictions.

### Annotations

Facts, reasons, and actions are the default prediction format:

```json
{
  "clip_id": "CLIP_ID_FROM_TEST",
  "facts": [{"fact_id": "F1", "text": "I see a person approaching the doorway."}],
  "reasons": [{"reason_id": "R1", "text": "I can leave room for the approaching person.", "facts": ["F1"], "tags": ["Proxemics"]}],
  "actions": [{"action_id": "A1", "description": "I step aside to leave the doorway clear.", "reasons_to_do": [{"reason_id": "R1"}]}]
}
```

IDs must be unique within each list, and references must point to existing IDs.
Every action must have a `reasons_to_do` list; it can be empty. All actions count.
Reasons may also include `tier` and `justification`, and support links may include
`explanation`.
An optional `chosen_action_id` must point to an existing candidate action.

### Action graphs

Use `--prediction-format graph` for an explicit support graph for each action:

```json
{
  "instance_id": "CLIP_ID_FROM_TEST",
  "chosen_action_id": "A1",
  "action_graphs": [{
    "action_id": "A1",
    "action_text": "I step aside to leave the doorway clear.",
    "facts": [{"id": "F1", "text": "I see a person approaching the doorway."}],
    "reasons": [{"id": "R1", "text": "I can leave room for the approaching person.", "normative_label": "proxemics"}],
    "edges": [
      {"src": "F1", "dst": "R1", "type": "supports"},
      {"src": "R1", "dst": "A1", "type": "motivates"}
    ]
  }]
}
```

You can wrap either format in a record with `clip_id`, `model_id`,
`prompt_id`, and a `prediction` object. The outer and inner IDs must agree.
Keep reference answers out of prediction files.

## Download inputs

Choose the upstream pre-action frame montage or video. Download the original
files into one local directory:

```bash
uv run nora download-media --media frames --output data/media
# Or, for video-capable models:
uv run nora download-media --media video --output data/media
```

Both use the same layout:

```text
data/media/CLIP_ID/frame_all_prev.jpg
data/media/CLIP_ID/video_prev.mp4
```

`download-media` downloads the selected file from `open-social-world/EgoNormia`. It does
not resample videos, create new montages, or download during-action content.
Existing files are reused, so the command can be rerun after a download failure.
Use `--limit 2` for a small trial or `--references subset.jsonl` for selected clips.
Follow the upstream media access and usage terms; use `HF_TOKEN` if access
requires authentication. Files already obtained under those terms can be placed
in the same layout without using the downloader.

Use the same `--media` selection when predicting. The runner checks every path
before inference and records the selected modality in each output row. Frame
and video inputs are different experimental settings; report which you used.
The paper uses frame montages. With the image-only paper prompts, native video
emits a warning and sends the original prompt unchanged; it is not a reproduction
of the paper setting. Use a custom prompt dictionary in Python if you want to
adapt the wording for video, and record that prompt as a separate experiment.

## Python interface

Import the supported interface directly from `nora`. Functions and CLI commands
use the same workflow names:

| Python function | CLI | Returns |
| --- | --- | --- |
| `load_references(split="test")` | `nora data --split test` | List of reference dictionaries; the CLI reports the file path instead. |
| `load_prompts()` | `nora predict --prompt ...` | List of bundled prompt dictionaries. |
| `download_media(references, output=..., media="frames")` | `nora download-media` | Download/reuse counts and per-clip failures. |
| `predict(references, model, ...)` | `nora predict` | Path to the saved prediction JSONL file. |
| `reconstruct(source, output=..., model=...)` | `nora reconstruct` | Output paths and row/failure counts. |
| `validate(predictions, prediction_format="annotation")` | `nora validate` | Valid/invalid counts and failure details; no scores. |
| `evaluate(predictions, output=...)` | `nora evaluate` | Run status, coverage, scores, and output directory. |
| `compare(run_dirs)` | `nora compare` | Shared clip IDs, sample count, and per-run scores. |

`download_media` and `predict` accept reference dictionaries and use only their
clip IDs. `validate`, `evaluate`, and `reconstruct` accept JSON/JSONL file paths;
`compare` accepts evaluation output directories. `validate` checks structure and
fact-reason-action links; `evaluate` additionally checks membership in the reference set.
Neither treats format validity as evidence that an answer is correct.

Use `prediction_format` / `--prediction-format` for prediction files and
`reference_format` / `--reference-format` for reference files. Both default to
`annotation`; select `graph` explicitly for action graphs. Output paths must be
new, except downloaded media files, which are reused.

### Model callbacks

Provide a function that accepts a `ModelInput` and returns text or a prediction
dictionary. Replace `your_inference_function` below with your model's inference
call.

```python
from nora import ModelInput, load_prompts, load_references, predict

def my_model(item: ModelInput) -> str:
    return your_inference_function(
        item.media_path, item.system_prompt, item.user_prompt, item.media
    )

prompt = next(p for p in load_prompts() if p["mode"] == "structured")
predict(
    load_references(), my_model,
    media_root="data/media",
    media="frames",  # Paper setting; native video is a separate experiment.
    prompt=prompt,
    output="runs/custom.jsonl",
    model_id="my-model",
)
```

The function receives the clip ID, media path, modality, and prompts, not
reference facts, reasons, or actions. Returned dictionaries still need validation.
`ChatCompletionsModel`, also imported from `nora`, is the provided callback for
compatible image/video endpoints; custom callbacks can use any model framework.

The runner checks media paths before inference, processes examples sequentially,
and saves each result immediately. It records model failures and refuses to
overwrite files. Automatic resume is not supported. For large or distributed
runs, use your own inference pipeline and submit its JSONL predictions.

## API integration

For a compatible Chat Completions endpoint:

```bash
uv run nora predict --base-url http://localhost:8000/v1 --model YOUR_MODEL \
  --media frames --media-root data/media --prompt structured \
  --max-tokens 4096 --limit 2 --output runs/raw.jsonl
```

The endpoint must already be running and support the selected visual modality.
Replace the local URL with your provider's HTTPS endpoint for hosted inference.
Set `NORA_API_KEY` if your endpoint requires authentication. Remote endpoints
must use HTTPS; localhost HTTP is allowed. Only complete, nonempty text
responses are accepted. Truncated responses are recorded as failures.

The adapter sends temperature `0` and `max_tokens=4096` by default. Change the
output limit with `--max-tokens`; timeout defaults to `180` seconds and is
configurable through `ChatCompletionsModel(timeout=...)` in Python. Requests are
sequential, with one response requested per clip. No seed, `top_p`, or native
thinking setting is sent. Use a custom callback for other decoding parameters.

Select `--prompt direct`, `deliberate`, or `structured`. The
[bundled templates](../src/nora/assets/prediction_prompts.json) contain the paper's
original system and user messages. `load_prompts()` returns those same
dictionaries for Python integrations. Direct prompting requests only a chosen
action; deliberate prompting requests 2 to 4 action analyses and a choice;
structured prompting requests facts, action-local reasons, and a choice.
Save separate files for each setting.

For native video input, choose `--media video` and a video-capable model and
endpoint. The runner sends the local MP4 as a base64 `video_url` content part,
as documented by [OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/videos).
Not every Chat Completions endpoint supports this format. Use a Python callback
for providers with a different video API; videos are never silently converted
to frames.

The adapter does not start a model server or configure GPUs. Use a Python
callback for other API formats. Configure native thinking settings in your
server or callback and record them: a deliberative prompt does not itself
enable a model's native thinking mode.

## Convert free-text responses

The bundled prompts produce text describing proposed actions and their support.
Convert these responses to annotations before scoring:

```bash
# Requires OPENAI_API_KEY and makes paid requests.
export OPENAI_API_KEY="your-api-key"
uv run nora reconstruct --input runs/raw.jsonl --model YOUR_EXTRACTOR_MODEL \
  --output runs/annotations.jsonl
uv run --extra scorer nora evaluate --predictions runs/annotations.jsonl \
  --output runs/scores
```

Reconstruction has two steps: an LLM extracts facts and action-local reasons;
Python builds the public annotation. This keeps repeated local labels such as
`A1/R1` and `A2/R1` separate and assigns unique reason IDs. Explicit foundations
are mapped to the public tags, including `coordination / proactivity` to
`Coordination` and `communication / legibility` to `Communication`. Other
explicit foundations map to `Other`; missing foundations remain untagged.

Opposing reasons never become supporting links. All final candidate actions
remain, even those with only objections or no support, and an explicit chosen
action must resolve to a retained action. Direct answers do not acquire invented
facts or reasons. Deliberate answers may express fact-reason links in prose,
without numbered references. The scoring schema differs from the raw response
and intermediate extraction formats.

Reconstruction uses only the model's response, not reference annotations.
It records the extractor model and prompt in a receipt beside the output.
Failed generations remain failures. Reconstructed annotations pass through the
same validator as direct predictions; invalid output is not repaired.

Reconstruction saves three files:

| File | Contents |
| --- | --- |
| `annotations.jsonl` | Valid predictions and failed-row markers. |
| `annotations.jsonl.failures.jsonl` | Failed candidates, or unparseable extractor text, with a safe error code and stage. |
| `annotations.jsonl.receipt.json` | Reconstruction model, prompt, and input/output record. |

Evaluate only `annotations.jsonl`; the failure file is for inspection, not scoring.
For example, `stage: conversion` with `error_code: dangling_fact_ref` means
the extractor cited a fact ID that does not exist. Request failures have no
candidate unless one was received, and provider exception bodies are not saved.

Keep raw responses and inspect reconstructed annotations: an LLM may omit or invent
support. Use the same reconstruction settings when comparing models. Saved
annotations can be scored again without further API calls.
