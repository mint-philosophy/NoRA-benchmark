from dataclasses import fields
from importlib.resources import files
import json
from pathlib import Path

import pytest

from nora.adapters import to_instance
from nora.data import index_rows, read_rows
from nora import ChatCompletionsModel, ModelInput, compare, evaluate, predict, validate


def save(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def graph(key="x"):
    return {"instance_id": key, "action_graphs": []}


def annotation():
    return {"clip_id": "x", "facts": [{"fact_id": "f", "text": "I see a door."}],
            "reasons": [{"reason_id": "r", "text": "I can pass through.", "facts": ["f"]}],
            "actions": [{"action_id": "a", "description": "I open the door.",
                         "reasons_to_do": [{"reason_id": "r"}]}]}


def test_golden_demo_regression(tmp_path):
    assets = files("nora") / "assets"
    result = evaluate(assets / "demo_pred.json", assets / "demo_gold.json",
                      prediction_format="graph", reference_format="graph", output=tmp_path / "scores", _demo=True)
    assert result["scores"]["soft"]["reasonableness_score"] == pytest.approx(0.6171216391579601)
    assert result["scores"]["hungarian"]["reasonableness_score"] == pytest.approx(0.5942066120307935)
    manifest = json.loads((tmp_path / "scores/run.json").read_text())
    assert manifest["benchmark_score"] is False
    assert (tmp_path / "scores/instance_metrics.jsonl").is_file()


def test_missing_invalid_and_valid_empty_are_distinct(tmp_path):
    refs = save(tmp_path, "gold.jsonl", [graph("x"), graph("y"), graph("z")])
    preds = save(tmp_path, "pred.jsonl", [graph("x"), {"instance_id": "y"}])
    result = evaluate(preds, refs, prediction_format="graph", reference_format="graph", output=tmp_path / "scores", _demo=True)
    assert result["coverage"] == dict(expected=3, submitted=2, valid=1, missing=1,
                                      invalid=1, valid_fraction=1/3)
    assert result["scores"]["soft"]["n"] == 1


def test_no_valid_is_null_and_does_not_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr("nora.evaluation.build_scorer", lambda **kwargs: pytest.fail("model loaded"))
    refs = save(tmp_path, "gold.jsonl", [graph()])
    preds = save(tmp_path, "pred.jsonl", [])
    result = evaluate(preds, refs, prediction_format="graph", reference_format="graph", output=tmp_path / "scores")
    assert result["scores"]["soft"]["reasonableness_score"] is None


def test_model_failure_never_becomes_valid_subset(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr("nora.evaluation.build_scorer", fail)
    refs = save(tmp_path, "gold.jsonl", [graph()])
    with pytest.raises(RuntimeError, match="unavailable"):
        evaluate(refs, refs, prediction_format="graph", reference_format="graph", output=tmp_path / "scores")
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize("mutate", [
    lambda p: p["reasons"][0].update(facts=["absent"]),
    lambda p: p["actions"][0].update(reasons_to_do=[{"reason_id": "absent"}]),
    lambda p: p["actions"][0].update(reasons_to_do=["r"]),
    lambda p: p["facts"].append(p["facts"][0]),
    lambda p: p["reasons"][0].update(tags="Safety"),
    lambda p: p["reasons"][0].update(tier="D"),
    lambda p: p["reasons"][0].update(justification=42),
])
def test_bad_source_references_rejected(mutate):
    payload = annotation()
    mutate(payload)
    with pytest.raises(ValueError):
        to_instance(payload, "annotation")


@pytest.mark.parametrize("metadata", [
    {"tier": None, "justification": None},
    {"tier": "A", "justification": "I have room to pass."},
])
def test_optional_reason_metadata_does_not_change_scoring_graph(metadata):
    payload = annotation()
    expected = to_instance(payload, "annotation")
    payload["reasons"][0].update(metadata)
    assert to_instance(payload, "annotation") == expected


def test_annotation_keeps_actions_without_support():
    payload = annotation()
    payload["actions"].append({"action_id": "b", "description": "Wait.", "reasons_to_do": []})
    instance = to_instance(payload, "annotation")
    assert [g.action_id for g in instance.action_graphs] == ["a", "b"]
    assert instance.action_graphs[1].reasons == ()


def test_ids_cannot_be_repaired():
    with pytest.raises(ValueError, match="mismatch"):
        to_instance({"clip_id": "x", "prediction": graph("y")}, "graph")
    with pytest.raises(ValueError, match="duplicate_clip_id"):
        index_rows([graph(), graph()])


def test_malformed_file_not_silently_dropped(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"clip_id":"a"}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        read_rows(path)


def test_comparison_uses_shared_ids(tmp_path):
    refs = save(tmp_path, "gold.jsonl", [graph("x"), graph("y")])
    a = save(tmp_path, "a.jsonl", [graph("x"), graph("y")])
    b = save(tmp_path, "b.jsonl", [graph("y")])
    for name, path in [("run-a", a), ("run-b", b)]:
        evaluate(path, refs, output=tmp_path / name, prediction_format="graph", reference_format="graph", _demo=True)
    comparison = compare([tmp_path / "run-a", tmp_path / "run-b"])
    assert comparison["cohort_ids"] == ["y"]
    assert all(row["scores"]["soft"]["n"] == 1 for row in comparison["systems"])


def test_empty_shared_cohort_is_null(tmp_path):
    refs = save(tmp_path, "gold.jsonl", [graph("x"), graph("y")])
    for name, key in [("a", "x"), ("b", "y")]:
        pred = save(tmp_path, name + ".jsonl", [graph(key)])
        evaluate(pred, refs, output=tmp_path / name, prediction_format="graph", reference_format="graph", _demo=True)
    result = compare([tmp_path / "a", tmp_path / "b"])
    assert result["n"] == 0
    assert result["systems"][0]["scores"]["soft"]["action_f1"] is None


def test_comparison_rejects_incomplete_modes(tmp_path):
    refs = save(tmp_path, "gold.jsonl", [graph()])
    run = tmp_path / "run"
    evaluate(refs, refs, output=run, prediction_format="graph", reference_format="graph", _demo=True)
    scores = run / "instance_metrics.csv"
    scores.write_text("\n".join(scores.read_text().splitlines()[:2]) + "\n")
    with pytest.raises(ValueError, match="inconsistent_per_case_scores"):
        compare([run, run])


def test_reconstruction_receipt_rejects_modified_predictions(tmp_path):
    refs = save(tmp_path, "gold.jsonl", [graph()])
    Path(str(refs) + ".receipt.json").write_text(json.dumps({"output_sha256": "incorrect"}))
    with pytest.raises(ValueError, match="receipt_hash_mismatch"):
        evaluate(refs, refs, output=tmp_path / "run", prediction_format="graph", reference_format="graph", _demo=True)


def test_chosen_action_must_exist():
    with pytest.raises(ValueError, match="dangling_chosen_action_id"):
        to_instance(dict(graph(), chosen_action_id="absent"), "graph")


def test_reconstruction_preserves_upstream_failure(tmp_path):
    from nora.reconstruction import reconstruct
    client = lambda *a: pytest.fail("failed generation was reconstructed")
    path = save(tmp_path, "raw.jsonl", [{"clip_id": "x", "status": "failed", "model_id": "test-model"}])
    output = tmp_path / "graphs.jsonl"
    result = reconstruct(path, output=output, model="not-called", client=client)
    assert result["invalid"] == 1
    assert read_rows(output)[0]["model_id"] == "test-model"


def test_callback_gets_only_input_view(tmp_path):
    folder = tmp_path / "images/x"
    folder.mkdir(parents=True)
    (folder / "frame_all_prev.jpg").write_bytes(b"synthetic-test-image")
    assert {f.name for f in fields(ModelInput)} == {
        "clip_id", "media_path", "system_prompt", "user_prompt", "prompt_id", "media"}
    def model(item):
        assert not hasattr(item, "facts")
        assert "secret" not in item.user_prompt
        return graph(item.clip_id)
    prompt = {"prompt_id": "test", "system_prompt": "system", "user_task_template": "task"}
    out = predict([dict(annotation(), hidden="secret")], model,
                  media_root=tmp_path / "images", prompt=prompt,
                  output=tmp_path / "pred.jsonl", model_id="custom")
    assert read_rows(out)[0]["status"] == "ok"
    with pytest.raises(FileExistsError):
        predict([annotation()], model, media_root=tmp_path / "images", prompt=prompt,
                output=out, model_id="custom")


@pytest.mark.parametrize("url", ["http://remote.example/v1", "https://user:password@host/v1",
                                  "https://host/v1?token=private"])
def test_endpoint_security(url):
    with pytest.raises(ValueError):
        ChatCompletionsModel(base_url=url, model="test")


def test_callback_failures_do_not_log_secret(tmp_path):
    folder = tmp_path / "x"
    folder.mkdir()
    (folder / "frame_all_prev.jpg").write_bytes(b"test")
    def fail(item):
        raise RuntimeError("sensitive-key")
    prompt = {"prompt_id": "test", "system_prompt": "system", "user_task_template": "task"}
    out = predict([annotation()], fail, media_root=tmp_path, prompt=prompt,
                  output=tmp_path / "out.jsonl", model_id="fake")
    assert "sensitive-key" not in out.read_text()
    assert read_rows(out)[0]["status"] == "failed"


@pytest.mark.parametrize("invalid_output", [False, True])
def test_predict_accepts_iterator_and_records_serialization_failure(tmp_path, invalid_output):
    folder = tmp_path / "x"
    folder.mkdir()
    (folder / "frame_all_prev.jpg").write_bytes(b"test")
    prompt = {"prompt_id": "test", "system_prompt": "system", "user_task_template": "task"}
    def model(item):
        return {"unsupported": {1, 2}} if invalid_output else graph(item.clip_id)
    out = predict(iter([annotation()]), model, media_root=tmp_path, prompt=prompt,
                  output=tmp_path / "out.jsonl", model_id="fake")
    rows = read_rows(out)
    assert len(rows) == 1
    assert rows[0]["status"] == ("failed" if invalid_output else "ok")
    if invalid_output:
        assert "prediction" not in rows[0]


def test_public_workflow_is_discoverable_without_loading_models():
    import inspect
    import subprocess
    import sys
    import nora

    assert set(nora.__all__) == {
        "ChatCompletionsModel", "ModelInput", "compare", "download_media", "evaluate",
        "load_prompts", "load_references", "predict", "reconstruct", "validate",
    }
    for name in nora.__all__:
        public_object = getattr(nora, name)
        assert callable(public_object) and inspect.getdoc(public_object)
    for function in (nora.evaluate, nora.validate):
        parameters = inspect.signature(function).parameters
        assert parameters["prediction_format"].default == "annotation"
        assert "format" not in parameters
    subprocess.run([sys.executable, "-c", "import nora, sys; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'sentence_transformers' not in sys.modules"], check=True)


@pytest.mark.parametrize("prediction_format", ["annotation", "graph"])
def test_python_and_cli_validation_agree(tmp_path, capsys, prediction_format):
    from nora.cli import main

    good = annotation() if prediction_format == "annotation" else graph()
    path = save(tmp_path, "predictions.jsonl", [good, {"clip_id": "y", "status": "failed"}])
    result = validate(path, prediction_format=prediction_format)
    assert result["rows"] == 2 and result["valid"] == result["invalid"] == 1
    assert main(["validate", str(path), "--prediction-format", prediction_format]) == 1
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("parameter", ["prediction_format", "reference_format"])
def test_format_configuration_errors_fail_before_scoring(tmp_path, parameter):
    refs = save(tmp_path, "refs.jsonl", [annotation()])
    predictions = save(tmp_path, "predictions.jsonl", [])
    with pytest.raises(ValueError, match=parameter):
        evaluate(predictions, refs, output=tmp_path / "scores", **{parameter: "typo"})
    assert not (tmp_path / "scores").exists()
    if parameter == "prediction_format":
        with pytest.raises(ValueError, match=parameter):
            validate(predictions, prediction_format="typo")


@pytest.mark.parametrize("command, expected", [
    (None, "download-media"),
    ("download-media", "--media {frames,video}"),
    ("predict", "--media-root"),
    ("reconstruct", "--input"),
    ("validate", "--prediction-format"),
    ("evaluate", "--reference-format"),
    ("compare", "At least two evaluation output directories"),
])
def test_cli_help_describes_the_public_workflow(capsys, command, expected):
    from nora.cli import main

    with pytest.raises(SystemExit) as result:
        main(([command] if command else []) + ["--help"])
    assert result.value.code == 0
    assert expected in capsys.readouterr().out
