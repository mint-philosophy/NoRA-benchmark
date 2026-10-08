"""Model callbacks and an explicit opt-in Chat Completions adapter."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
import urllib.request
import warnings

from nora.data import index_rows
from nora.media import media_path


@dataclass(frozen=True)
class ModelInput:
    """Only pre-action observation and prediction prompts; no reference graph."""

    clip_id: str
    media_path: Path
    system_prompt: str
    user_prompt: str
    prompt_id: str
    media: str = "frames"


def predict(references, model: Callable[[ModelInput], str | dict], *, media_root,
            prompt, output, model_id, media="frames"):
    """Run a model callback for an iterable of reference dictionaries.

    Only clip IDs are used from references; the callback receives ModelInput.
    Use a prompt dictionary from load_prompts and local frames or video. Return
    the saved JSONL Path; failed calls are recorded and existing files refused.
    """
    if media == "video" and prompt.get("input_modality") == ["image"]:
        warnings.warn(
            "The paper prompts use image frames. Native video is a different experimental "
            "setting; the image-worded prompt is sent unchanged. Report media=video separately.",
            UserWarning, stacklevel=2,
        )
    rows_by_id = index_rows(references)
    inputs = []
    for clip_id in rows_by_id:
        path = media_path(media_root, clip_id, media)
        if not path.is_file():
            raise ValueError(f"missing_pre_action_media: {clip_id}; run nora download-media --media {media}")
        inputs.append(ModelInput(clip_id, path, prompt["system_prompt"],
                                 prompt["user_task_template"], prompt["prompt_id"], media))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for item in inputs:
            record = {
                "clip_id": item.clip_id, "model_id": model_id,
                "prompt_id": item.prompt_id,
                "media": item.media,
            }
            try:
                prediction = model(item)
                if isinstance(prediction, str) and prediction.strip():
                    record.update(response=prediction, status="ok")
                elif isinstance(prediction, dict):
                    record.update(prediction=prediction, status="ok")
                else:
                    raise ValueError("model_returned_empty_or_unsupported_output")
                line = json.dumps(record, allow_nan=False)
            except Exception as exc:
                # Provider error bodies can contain credentials or prompts.
                record.pop("prediction", None)
                record.pop("response", None)
                record.update(status="failed", error_type=type(exc).__name__)
                line = json.dumps(record, allow_nan=False)
            handle.write(line + "\n")
            handle.flush()
    return path


class ChatCompletionsModel:
    """Use a compatible image or video endpoint, including local model servers.

    Constructing this class sends no requests. Calling it may incur provider fees.
    Video requires an endpoint supporting video_url content parts.
    """

    def __init__(self, *, base_url, model, api_key_env="NORA_API_KEY",
                 max_tokens=4096, timeout=180):
        url = urlsplit(base_url)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url_must_not_contain_credentials_or_query")
        local = url.hostname in {"localhost", "127.0.0.1", "::1"}
        if url.scheme != "https" and not (url.scheme == "http" and local):
            raise ValueError("HTTPS required except for localhost")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = os.environ.get(api_key_env)
        self.max_tokens = max_tokens
        self.timeout = timeout

    def __call__(self, item):
        if item.media not in {"frames", "video"}:
            raise ValueError("media must be frames or video")
        part_type = "video_url" if item.media == "video" else "image_url"
        mime = mimetypes.guess_type(item.media_path)[0] or (
            "video/mp4" if item.media == "video" else "image/jpeg")
        encoded = base64.b64encode(item.media_path.read_bytes()).decode()
        body = {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": item.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": item.user_prompt},
                    {"type": part_type, part_type: {"url": f"data:{mime};base64,{encoded}"}},
                ]},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        # Never forward authorization or private image content through redirects.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        with urllib.request.build_opener(NoRedirect).open(request, timeout=self.timeout) as response:
            result = json.load(response)
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("incomplete_model_response")
        answer = choice["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty_visible_response")
        return answer
