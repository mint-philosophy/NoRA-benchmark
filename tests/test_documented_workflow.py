import json
from pathlib import Path
import re
import shlex

import pytest

from nora import load_prompts
from nora.cli import build_parser, main
from nora.data import read_rows


README = Path(__file__).resolve().parents[1] / "README.md"


def readme_commands():
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", README.read_text(), re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line, comments=True)
            if words[:2] == ["uv", "run"] and "nora" in words:
                commands.append(words[words.index("nora") + 1:])
    return commands


def test_readme_commands_match_cli():
    commands = readme_commands()
    assert [command[0] for command in commands] == [
        "demo", "download-media", "predict", "reconstruct", "validate", "evaluate", "compare"]
    for command in commands:
        build_parser().parse_args(command)


@pytest.mark.parametrize("mode", ["direct", "deliberate", "structured"])
def test_documented_pipeline(tmp_path, monkeypatch, capsys, mode):
    import nora.evaluation as evaluation

    expected = {"clip_id": "example", "facts": [], "reasons": [], "actions": [
        {"action_id": "A1", "description": "I wait.", "reasons_to_do": []}]}
    references = tmp_path / "references.jsonl"
    references.write_text(json.dumps(expected) + "\n")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"synthetic media fixture")
    monkeypatch.setattr("nora.cli.reference_path", lambda **kwargs: references)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(image))
    prompt = next(prompt for prompt in load_prompts() if prompt["mode"] == mode)
    assert prompt["prompt_id"] == mode

    def make_model(**settings):
        assert settings["model"] == "YOUR_MODEL" and settings["max_tokens"] == 4096

        def respond(item):
            assert item.prompt_id == prompt["prompt_id"]
            assert item.system_prompt == prompt["system_prompt"]
            assert item.user_prompt == prompt["user_task_template"]
            assert item.media == "frames" and item.media_path.is_file()
            assert not hasattr(item, "facts")
            return "I wait."

        return respond

    def extract(self, instructions, content):
        assert json.loads(content) == {"clip_id": "example", "response": "I wait."}
        return json.dumps(expected)

    monkeypatch.setattr("nora.models.ChatCompletionsModel", make_model)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr("nora.reconstruction.ResponsesClient.__call__", extract)
    real_evaluate = evaluation.evaluate
    monkeypatch.setattr(evaluation, "evaluate",
                        lambda *args, **kwargs: real_evaluate(*args, **kwargs, _demo=True))
    monkeypatch.chdir(tmp_path)
    for command in readme_commands():
        if command[0] in {"demo", "compare"}:
            continue
        if command[0] == "predict":
            command[command.index("--prompt") + 1] = mode
        assert main(command) == 0
        capsys.readouterr()

    output = tmp_path / "runs/my-model-structured"
    assert read_rows(output / "raw.jsonl")[0]["prompt_id"] == prompt["prompt_id"]
    assert read_rows(output / "annotations.jsonl")[0]["status"] == "ok"
    coverage = json.loads((output / "scores/coverage.json").read_text())
    assert coverage["valid"] == coverage["expected"] == 1
    scores = json.loads((output / "scores/summary.json").read_text())
    assert scores["soft"]["action_f1"] == pytest.approx(1)
    assert json.loads((output / "scores/run.json").read_text())["benchmark_score"] is False
