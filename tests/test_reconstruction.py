import json

import pytest

from nora import reconstruct
from nora.adapters import to_instance
from nora.data import read_rows
from nora.reconstruction import _to_annotation


def local_extraction():
    return {
        "clip_id": "x",
        "facts": [{"fact_id": "F1", "text": "Someone is passing beside me."}],
        "actions": [{"action_id": "A1", "description": "I step aside.", "reasons": [
            {"text": "I can give them room.", "facts": ["F1"], "stance": "support"}]}],
        "chosen_action_id": "A1",
    }


@pytest.mark.parametrize("foundation,tag", [
    ("safety", "Safety"), ("privacy", "Privacy"),
    ("proxemics (personal space)", "Proxemics"), ("proxemics", "Proxemics"),
    ("politeness", "Politeness"), ("cooperation", "Cooperation"),
    ("coordination / proactivity", "Coordination"), ("Coordination", "Coordination"),
    ("communication / legibility", "Communication"), ("Communication", "Communication"),
    ("  COMMUNICATION/LEGIBILITY  ", "Communication"),
    ("Safety and care", "Safety"), ("Privacy: personal information", "Privacy"),
    ("personal space", "Proxemics"), ("proactivity", "Coordination"),
    ("legibility", "Communication"), ("  PERSONAL   SPACE (distance)", "Proxemics"),
    ("safetyish", "Other"),
    ("fairness / equity", "Other"), ("other", "Other"),
])
def test_explicit_foundation_mapping(foundation, tag):
    extracted = local_extraction()
    extracted["actions"][0]["reasons"][0]["foundation"] = foundation
    assert _to_annotation(extracted)["reasons"][0]["tags"] == [tag]


def test_absent_foundation_is_not_inferred():
    extracted = local_extraction()
    extracted["actions"][0]["reasons"][0]["text"] = "I can keep a safe distance."
    assert "tags" not in _to_annotation(extracted)["reasons"][0]


@pytest.mark.parametrize("fields", [
    ("foundation",), ("tier",), ("justification",),
    ("foundation", "tier", "justification"),
])
def test_null_optional_fields_are_absent(fields):
    extracted = local_extraction()
    expected = _to_annotation(extracted)
    extracted["actions"][0]["reasons"][0].update(dict.fromkeys(fields))
    assert _to_annotation(extracted) == expected


def test_structured_local_reasons_and_opposed_choice(tmp_path):
    raw = """Facts:
- F1: Someone is passing beside me.
- F2: Someone is holding a tray toward me.

Available actions:
- A1: I step aside.
  Reasons to do:
  - R1 | Tier: A | Foundation: proxemics (personal space) | Fact refs: F1 | Reason: I can give them room.
- A2: I take the tray.
  Reasons to do:
  - R1 | Tier: B | Foundation: coordination / proactivity | Fact refs: F2 | Reason: I can coordinate the handoff.
- A3: I rush past.
  Reasons not to do:
  - R1 | Tier: A | Foundation: safety | Fact refs: F1 | Reason: I could collide with them.

Chosen action:
A3. I rush past.
"""
    extracted = local_extraction()
    extracted["facts"].append({"fact_id": "F2", "text": "Someone is holding a tray toward me."})
    extracted["actions"][0]["reasons"][0].update(
        foundation="proxemics (personal space)", tier="A")
    extracted["actions"].extend([
        {"action_id": "A2", "description": "I take the tray.", "reasons": [{
            "text": "I can coordinate the handoff.", "facts": ["F2"], "stance": "support",
            "foundation": "coordination / proactivity", "tier": "B"}]},
        {"action_id": "A3", "description": "I rush past.", "reasons": [{
            "text": "I could collide with them.", "facts": ["F1"], "stance": "oppose",
            "foundation": "safety", "tier": "A"}]},
    ])
    extracted["chosen_action_id"] = "A3"
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"clip_id": "x", "response": raw,
                                  "model_id": "hidden-model", "gold": "hidden-reference"}) + "\n")

    def extract(instructions, content):
        assert json.loads(content) == {"clip_id": "x", "response": raw}
        assert "Never merge two reasons just because both are called R1" in instructions
        assert "Reasons not to do are oppose" in instructions
        assert "Keep actions with only objections or no reasons" in instructions
        return json.dumps(extracted)

    output = tmp_path / "annotations.jsonl"
    assert reconstruct(source, output=output, model="fake", client=extract)["invalid"] == 0
    saved = read_rows(output)[0]["prediction"]
    assert [r["reason_id"] for r in saved["reasons"]] == ["R1", "R2"]
    assert [r["facts"] for r in saved["reasons"]] == [["F1"], ["F2"]]
    assert [r["tags"] for r in saved["reasons"]] == [["Proxemics"], ["Coordination"]]
    assert [a["reasons_to_do"] for a in saved["actions"]] == [
        [{"reason_id": "R1"}], [{"reason_id": "R2"}]]
    assert [a["action_id"] for a in saved["actions"]] == ["A1", "A2"]
    assert "chosen_action_id" not in saved
    assert "collide" not in json.dumps(saved)
    instance = to_instance(saved, "annotation")
    assert instance.chosen_action_id is None
    receipt = json.loads(output.with_suffix(".jsonl.receipt.json").read_text())
    assert receipt["path"] == "action-local-reconstruction"


def test_objection_only_response_keeps_no_candidates_and_does_not_mutate_extraction():
    extracted = local_extraction()
    extracted["actions"][0]["reasons"][0]["stance"] = "oppose"
    original = json.dumps(extracted)
    converted = _to_annotation(extracted)
    assert converted["actions"] == converted["reasons"] == []
    assert "chosen_action_id" not in converted
    assert json.dumps(extracted) == original
    to_instance(converted, "annotation")


@pytest.mark.parametrize("position", [0, 1])
def test_mixed_support_and_objections_keeps_candidate_and_choice(position):
    extracted = local_extraction()
    supporting_reason = extracted["actions"][0]["reasons"][0]
    extracted["actions"][0]["reasons"].insert(position, {
        "text": "I might block another person's path.", "facts": ["F1"], "stance": "oppose"})
    converted = _to_annotation(extracted)
    assert len(converted["reasons"]) == len(converted["actions"]) == 1
    assert converted["reasons"][0]["text"] == supporting_reason["text"]
    assert converted["actions"][0]["reasons_to_do"] == [
        {"reason_id": converted["reasons"][0]["reason_id"]}]
    assert to_instance(converted, "annotation").chosen_action_id == "A1"


@pytest.mark.parametrize("chosen", [True, False])
def test_action_only_answer_and_absent_choice(chosen):
    extracted = local_extraction()
    extracted["facts"] = []
    extracted["actions"][0]["reasons"] = []
    if not chosen:
        del extracted["chosen_action_id"]
    converted = _to_annotation(extracted)
    assert converted["facts"] == converted["reasons"] == []
    assert converted["actions"][0]["reasons_to_do"] == []
    assert ("chosen_action_id" in converted) is chosen


def test_deliberate_prose_uses_explicit_links_without_inventing_metadata(tmp_path):
    raw = ("Action analyses:\n- A1:\n"
           "  Analysis: Someone is passing beside me, so I can give them room.\n"
           "  Brief action: I step aside.\nChosen action:\nA1. I step aside.")
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"clip_id": "x", "response": raw}) + "\n")

    def extract(instructions, content):
        assert json.loads(content)["response"] == raw
        assert "numbered\nfact references are not required" in instructions
        return json.dumps(local_extraction())

    result = reconstruct(source, output=tmp_path / "out.jsonl", model="fake", client=extract)
    prediction = read_rows(result["output"])[0]["prediction"]
    assert prediction["reasons"] == [{"reason_id": "R1", "text": "I can give them room.",
                                      "facts": ["F1"]}]


@pytest.mark.parametrize("stance", ["support", "oppose"])
@pytest.mark.parametrize("change,error", [
    ({"stance": "unclear"}, "invalid_reason_stance"),
    ({"text": ""}, "missing_text"),
    ({"text": None}, "missing_text"),
    ({"facts": None}, "invalid_fact_refs"),
    ({"facts": [42]}, "invalid_fact_refs"),
    ({"facts": ["missing"]}, "dangling_fact_ref"),
    ({"facts": ["F1", "F1"]}, "duplicate_fact_refs"),
    ({"foundation": ""}, "invalid_foundation"),
    ({"foundation": 42}, "invalid_foundation"),
    ({"tier": "D"}, "invalid_tier"),
    ({"justification": 42}, "invalid_justification"),
    ({"unexpected": "ignored?"}, "invalid_extraction_fields"),
])
def test_invalid_extractions_are_rejected_not_repaired(change, error, stance):
    extracted = local_extraction()
    reason = extracted["actions"][0]["reasons"][0]
    reason["stance"] = stance
    reason.update(change)
    with pytest.raises(ValueError, match=error):
        _to_annotation(extracted)


@pytest.mark.parametrize("stance", ["support", "oppose"])
def test_duplicate_actions_are_not_merged(stance):
    extracted = local_extraction()
    extracted["actions"][0]["reasons"][0]["stance"] = stance
    extracted["actions"].append(extracted["actions"][0])
    with pytest.raises(ValueError, match="duplicate_node_id"):
        _to_annotation(extracted)


@pytest.mark.parametrize("field,value,error", [
    ("chosen_action_id", "missing", "dangling_chosen_action_id"),
    ("description", "", "missing_text"),
    ("action_id", " A1 ", "node_id_has_whitespace"),
])
def test_invalid_objection_only_actions_fail_before_filtering(field, value, error):
    extracted = local_extraction()
    action = extracted["actions"][0]
    action["reasons"][0]["stance"] = "oppose"
    target = extracted if field == "chosen_action_id" else action
    target[field] = value
    with pytest.raises(ValueError, match=error):
        _to_annotation(extracted)


@pytest.mark.parametrize("stance", ["support", "oppose"])
@pytest.mark.parametrize("field", ["text", "facts", "stance"])
def test_missing_required_reason_fields_fail(field, stance):
    extracted = local_extraction()
    reason = extracted["actions"][0]["reasons"][0]
    reason["stance"] = stance
    del reason[field]
    with pytest.raises(ValueError, match="invalid_extraction_fields"):
        _to_annotation(extracted)
