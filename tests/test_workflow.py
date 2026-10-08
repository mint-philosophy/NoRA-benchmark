from contextlib import nullcontext
import json

import pytest

from nora.cli import main
from nora.data import read_rows
from nora.media import MEDIA_FILES, media_path
from nora import download_media, predict, reconstruct


def annotation(key="x"):
    return {"clip_id": key, "facts": [], "reasons": [], "actions": []}


def extraction():
    return {"clip_id": "x", "facts": [], "actions": [
        {"action_id": "A1", "description": "I wait.", "reasons": []}],
        "chosen_action_id": "A1"}


def save(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize("media", ["frames", "video"])
def test_download_and_predict_share_media_layout(tmp_path, monkeypatch, media):
    cached = tmp_path / "cached"
    cached.write_bytes(b"synthetic media fixture")
    requests = []

    def download(repo, filename, **kwargs):
        requests.append((repo, filename, kwargs))
        return str(cached)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    refs = save(tmp_path / "refs.jsonl", [annotation()])
    root = tmp_path / "media"
    assert main(["download-media", "--references", str(refs), "--media", media,
                 "--output", str(root)]) == 0
    assert requests == [("open-social-world/EgoNormia", f"video/x/{MEDIA_FILES[media]}",
                         {"repo_type": "dataset"})]
    assert media_path(root, "x", media).read_bytes() == cached.read_bytes()
    assert download_media([annotation()], output=root, media=media)["reused"] == 1
    assert len(requests) == 1

    def model(item):
        assert item.media == media
        assert item.media_path.name == MEDIA_FILES[media]
        assert not hasattr(item, "facts")
        return "I wait."

    output = predict([annotation()], model, media_root=root, media=media,
                     prompt={"system_prompt": "system", "user_task_template": "task", "prompt_id": "test"},
                     output=tmp_path / "raw.jsonl", model_id="test")
    row = read_rows(output)[0]
    assert row["media"] == media and row["status"] == "ok"
    assert "image_sha256" not in row and "prompt_sha256" not in row


def test_download_reports_failures_without_exposing_provider_errors(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("private provider response")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    result = download_media([annotation()], output=tmp_path / "media")
    assert result["invalid"] == 1
    assert "private provider response" not in json.dumps(result)
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("media", ["frames", "video"])
def test_predict_cli_passes_selected_media(tmp_path, monkeypatch, media):
    path = media_path(tmp_path / "media", "x", media)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic media fixture")
    refs = save(tmp_path / "refs.jsonl", [annotation()])
    def model(item):
        assert item.media == media and item.media_path == path
        return "I wait."
    monkeypatch.setattr("nora.models.ChatCompletionsModel", lambda **kwargs: model)
    output = tmp_path / "raw.jsonl"
    warning = pytest.warns(UserWarning, match="different experimental setting") if media == "video" else nullcontext()
    with warning:
        assert main(["predict", "--references", str(refs), "--media-root", str(tmp_path / "media"),
                     "--media", media, "--model", "fake", "--base-url", "http://localhost:1/v1",
                     "--output", str(output)]) == 0
    assert read_rows(output)[0]["media"] == media


def test_reconstruction_cli_uses_users_api_key(tmp_path, monkeypatch):
    from nora.reconstruction import ResponsesClient
    def complete(self, instructions, content):
        assert self.key == "test-only-api-key"
        assert self.model == "user-selected-model"
        return json.dumps(extraction())
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-api-key")
    monkeypatch.setattr(ResponsesClient, "__call__", complete)
    raw = save(tmp_path / "raw.jsonl", [{"clip_id": "x", "response": "I wait."}])
    output = tmp_path / "annotations.jsonl"
    assert main(["reconstruct", "--input", str(raw), "--output", str(output),
                 "--model", "user-selected-model"]) == 0
    assert read_rows(output)[0]["status"] == "ok"
    assert "test-only-api-key" not in output.read_text()


@pytest.mark.parametrize("clip_id", ["../x", "/tmp/x", "x/y", "..", "x\\y"])
def test_media_paths_cannot_escape_root(tmp_path, clip_id):
    with pytest.raises(ValueError, match="unsafe"):
        media_path(tmp_path, clip_id, "video")


@pytest.mark.parametrize("rows", [[], [{"clip_id": "x", "response": "I wait."}],
                                   [{"clip_id": "x", "status": "failed"}]])
def test_zero_valid_cases_return_failure_but_save_diagnostics(tmp_path, capsys, rows):
    refs = save(tmp_path / "refs.jsonl", [annotation()])
    predictions = save(tmp_path / "pred.jsonl", rows)
    output = tmp_path / "scores"
    assert main(["evaluate", "--predictions", str(predictions), "--references", str(refs),
                 "--output", str(output)]) == 1
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "failed" and result["coverage"]["valid"] == 0
    assert result["scores"]["soft"]["reasonableness_score"] is None
    assert "no valid predictions" in captured.err
    assert (output / "failures.jsonl").is_file()
    if rows and "response" in rows[0]:
        assert "raw_response_requires_reconstruction" in (output / "failures.jsonl").read_text()


@pytest.mark.parametrize("include_failed", [False, True])
def test_valid_subsets_still_succeed(tmp_path, monkeypatch, capsys, include_failed):
    import nora.evaluation as evaluation
    real_evaluate = evaluation.evaluate

    def demo_evaluate(*args, **kwargs):
        return real_evaluate(*args, **kwargs, _demo=True)

    monkeypatch.setattr(evaluation, "evaluate", demo_evaluate)
    refs = save(tmp_path / "refs.jsonl", [annotation(), annotation("y")])
    rows = [annotation()]
    if include_failed:
        rows.append({"clip_id": "y", "status": "failed"})
    predictions = save(tmp_path / "pred.jsonl", rows)
    assert main(["evaluate", "--predictions", str(predictions), "--references", str(refs),
                 "--output", str(tmp_path / "scores")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "partial" and result["coverage"]["valid"] == 1


def test_failed_reconstruction_is_separate_and_inspectable(tmp_path):
    raw = save(tmp_path / "raw.jsonl", [{"clip_id": "x", "response": "I wait.", "media": "video"}])
    output = tmp_path / "pred.jsonl"
    candidate = extraction()
    candidate["actions"][0]["reasons"] = [
        {"text": "I can avoid bumping into them.", "facts": ["missing"], "stance": "support"}]
    result = reconstruct(raw, output=output, model="fake", client=lambda *a: json.dumps(candidate))
    record = read_rows(output)[0]
    assert record["status"] == "failed" and "prediction" not in record
    assert record["media"] == "video"
    audit = read_rows(result["failures_path"])[0]
    assert audit["candidate"] == candidate
    assert audit["stage"] == "conversion" and audit["error_code"] == "dangling_fact_ref"


@pytest.mark.parametrize("text", ["not json", '{"x": NaN}', "null"])
def test_unparseable_reconstruction_preserves_response(tmp_path, text):
    raw = save(tmp_path / "raw.jsonl", [{"clip_id": "x", "response": "I wait."}])
    result = reconstruct(raw, output=tmp_path / "pred.jsonl", model="fake", client=lambda *a: text)
    audit = read_rows(result["failures_path"])[0]
    assert audit["extractor_response"] == text
    assert result["invalid"] == 1


def test_request_failure_never_saves_exception_body(tmp_path):
    raw = save(tmp_path / "raw.jsonl", [{"clip_id": "x", "response": "I wait."}])
    output = tmp_path / "pred.jsonl"
    def fail(*args):
        raise RuntimeError("a-private-api-key")
    result = reconstruct(raw, output=output, model="fake", client=fail)
    audit = read_rows(result["failures_path"])[0]
    assert audit["error_code"] == "request_failed" and audit["stage"] == "request"
    assert "candidate" not in audit
    assert "a-private-api-key" not in output.read_text() + json.dumps(audit)


def test_missing_response_does_not_block_valid_rows(tmp_path):
    raw = save(tmp_path / "raw.jsonl", [{"clip_id": "bad"}, {"clip_id": "x", "response": "I wait."}])
    result = reconstruct(raw, output=tmp_path / "pred.jsonl", model="fake",
                         client=lambda *a: json.dumps(extraction()))
    assert [row["status"] for row in read_rows(result["output"])] == ["failed", "ok"]
    assert read_rows(result["failures_path"])[0]["error_code"] == "missing_response"


def test_failure_audit_is_not_overwritten(tmp_path):
    raw = save(tmp_path / "raw.jsonl", [])
    output = tmp_path / "pred.jsonl"
    failure = tmp_path / "pred.jsonl.failures.jsonl"
    failure.write_text("existing audit")
    with pytest.raises(FileExistsError):
        reconstruct(raw, output=output, model="fake", client=lambda *a: "{}")
    assert failure.read_text() == "existing audit" and not output.exists()
