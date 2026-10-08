import json

import pytest

from nora import evaluate, load_prompts, load_references, reconstruct
from nora.adapters import to_instance
from nora.cli import build_parser
from nora.data import digest, read_rows, reference_path


def test_bundled_test_annotations_are_exact_complete_and_offline(monkeypatch):
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda *a, **kw: pytest.fail("test references must be bundled"))
    assert digest(reference_path(offline=True)) == (
        "d8ed78b40c38645af5f83634cab8093a1d47699b8f5e6a1e6d39177a25137b68")
    rows = load_references(offline=True)
    assert len(rows) == len({row["clip_id"] for row in rows}) == 190
    assert [sum(len(row[field]) for row in rows) for field in
            ("facts", "reasons", "actions")] == [1324, 1007, 635]
    for row in rows:
        assert set(row) == {"clip_id", "split", "prompt_variant", "facts", "reasons",
                            "actions", "video_source"}
        assert len(to_instance(row, "annotation").action_graphs) == len(row["actions"])
        assert "actions_observed" not in row
        assert row["prompt_variant"] == [prompt["prompt_id"] for prompt in load_prompts()]
        assert all(set(action) == {"action_id", "description", "reasons_to_do"}
                   for action in row["actions"])


def test_annotation_format_and_bundled_test_are_defaults(tmp_path):
    args = build_parser().parse_args(["evaluate", "--predictions", "input", "--output", "output"])
    assert args.prediction_format == "annotation" and args.references is None
    row = load_references()[0]
    path = tmp_path / "pred.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = evaluate(path, output=tmp_path / "run", _demo=True)
    assert result["coverage"]["expected"] == 190
    assert result["coverage"]["valid"] == 1
    assert result["scores"]["soft"]["reasonableness_score"] == pytest.approx(1)


def test_original_paper_prompts_are_bundled():
    prompts = {row["mode"]: row for row in load_prompts()}
    assert {mode: row["prompt_id"] for mode, row in prompts.items()} == {
        "direct": "direct",
        "deliberate": "deliberate",
        "structured": "structured",
    }
    assert {mode: row["expected_output_sections"] for mode, row in prompts.items()} == {
        "direct": ["Chosen action"],
        "deliberate": ["Action analyses", "Chosen action"],
        "structured": ["Facts", "Available actions", "Chosen action"],
    }
    assert "Do not include facts" in prompts["direct"]["user_task_template"]
    assert "2 to 4 plausible next actions" in prompts["deliberate"]["system_prompt"]
    assert "Do not use fact ids" in prompts["deliberate"]["user_task_template"]
    structured = prompts["structured"]
    assert "Restart the reason index within each action" in structured["system_prompt"]
    for field in ("system_prompt", "user_task_template"):
        assert all(text in structured[field] for text in (
            "Reasons to do:", "Reasons not to do:", "Tier: A", "Tier: B",
            "Tier: C", "Foundation:", "Fact refs:"))


def test_reconstruction_uses_same_annotation_path(tmp_path):
    prediction = {"clip_id": "x", "facts": [], "reasons": [], "actions": [
        {"action_id": "A1", "description": "I wait.", "reasons_to_do": []},
        {"action_id": "A2", "description": "I step aside.", "reasons_to_do": []}]}
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"clip_id": "x", "response": "I wait or step aside.",
                                "secret_gold": "must not enter extraction"}) + "\n")
    def client(instructions, content):
        assert set(json.loads(content)) == {"clip_id", "response"}
        assert "secret_gold" not in instructions + content
        return json.dumps(prediction)
    output = tmp_path / "annotations.jsonl"
    assert reconstruct(path, output=output, model="fake", client=client)["invalid"] == 0
    saved = read_rows(output)[0]
    assert saved["prediction"] == prediction
    assert to_instance(saved, "annotation").to_dict() == to_instance(prediction, "annotation").to_dict()
    assert to_instance(saved, "annotation").chosen_action_id is None
    with pytest.raises(FileExistsError):
        reconstruct(path, output=output, model="fake", client=client)


@pytest.mark.parametrize("prediction", [
    {"clip_id": "wrong", "facts": [], "reasons": [], "actions": []},
    {"clip_id": "x", "facts": [], "reasons": [], "actions": [], "chosen_action_id": "A1"},
    {"clip_id": "x", "facts": [], "reasons": [], "actions": [{
        "action_id": "A1", "description": "I wait.", "reasons_to_do": [{"reason_id": "missing"}]}]},
    {"clip_id": "x", "facts": [], "reasons": [], "actions": [{
        "action_id": "A1", "description": "I wait.", "reasons_to_do": [], "reasons_not_to_do": []}]},
])
def test_reconstruction_does_not_repair_invalid_annotations(tmp_path, prediction):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"clip_id":"x","response":"I wait."}\n')
    output = tmp_path / "pred.jsonl"
    result = reconstruct(path, output=output, model="fake", client=lambda *a: json.dumps(prediction))
    assert result["invalid"] == 1
    assert read_rows(output)[0]["status"] == "failed"
    assert "prediction" not in read_rows(output)[0]
