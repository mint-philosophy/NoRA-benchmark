"""Validate public annotations and build their action-rooted support graphs."""

from __future__ import annotations

from nora.data import index_rows, read_rows, row_id


def validate(predictions, *, prediction_format="annotation"):
    """Check a JSON/JSONL prediction file without scoring or API calls.

    Return row, valid, and invalid counts plus per-clip failures. This checks
    structure and internal links, not answer quality or reference-set membership.
    Malformed files and duplicate clip IDs raise ValueError.
    """
    if prediction_format not in {"annotation", "graph"}:
        raise ValueError("prediction_format must be annotation or graph")
    rows = index_rows(read_rows(predictions))
    failures = []
    for clip_id, row in rows.items():
        try:
            to_instance(row, prediction_format)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            failures.append({"clip_id": clip_id, "error": str(exc)})
    return {"rows": len(rows), "valid": len(rows) - len(failures),
            "invalid": len(failures), "failures": failures}


def _nodes(payload, field, id_field, text_field):
    nodes = payload.get(field)
    if not isinstance(nodes, list):
        raise ValueError(f"expected_array: {field}")
    result = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError(f"expected_object: {field}")
        for key in (id_field, text_field):
            if not isinstance(node.get(key), str) or not node[key].strip():
                raise ValueError(f"missing_text: {field}.{key}")
        key = node[id_field]
        if key != key.strip():
            raise ValueError(f"node_id_has_whitespace: {field}")
        if key in result:
            raise ValueError(f"duplicate_node_id: {field}.{key}")
        result[key] = node
    return result


def validate_annotation(payload):
    row_id(payload)
    facts = _nodes(payload, "facts", "fact_id", "text")
    reasons = _nodes(payload, "reasons", "reason_id", "text")
    actions = _nodes(payload, "actions", "action_id", "description")
    for field, nodes, allowed in (
        ("facts", facts, {"fact_id", "text"}),
        ("reasons", reasons, {"reason_id", "text", "facts", "tags", "tier", "justification"}),
        ("actions", actions, {"action_id", "description", "reasons_to_do"}),
    ):
        for node in nodes.values():
            if set(node) - allowed:
                raise ValueError(f"unexpected_fields: {field}.{sorted(set(node) - allowed)}")
    for reason in reasons.values():
        refs = reason.get("facts")
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
            raise ValueError("invalid_fact_refs")
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate_fact_refs")
        if any(ref not in facts for ref in refs):
            raise ValueError("dangling_fact_ref")
        tags = reason.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("invalid_tags")
        if reason.get("tier") is not None and reason["tier"] not in ("A", "B", "C"):
            raise ValueError("invalid_tier")
        if reason.get("justification") is not None and not isinstance(reason["justification"], str):
            raise ValueError("invalid_justification")
    for action in actions.values():
        refs = action.get("reasons_to_do")
        if not isinstance(refs, list):
            raise ValueError("invalid_reason_refs: reasons_to_do")
        ids = []
        for ref in refs:
            if not isinstance(ref, dict) or not isinstance(ref.get("reason_id"), str):
                raise ValueError("reason_ref_requires_object")
            ids.append(ref["reason_id"])
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_reason_refs")
        if any(key not in reasons for key in ids):
            raise ValueError("dangling_reason_ref")
    chosen = payload.get("chosen_action_id")
    if chosen is not None and (not isinstance(chosen, str) or chosen not in actions):
        raise ValueError("dangling_chosen_action_id")


def _annotation_instance(payload):
    from nora._scoring import ActionGraph, Edge, FactNode, ReasonNode, ReasoningInstance

    facts = {node["fact_id"]: node for node in payload["facts"]}
    reasons = {node["reason_id"]: node for node in payload["reasons"]}
    graphs = []
    for action in payload["actions"]:
        local_reasons = [reasons[ref["reason_id"]] for ref in action["reasons_to_do"]]
        fact_ids = dict.fromkeys(key for reason in local_reasons for key in reason["facts"])
        edges = []
        for reason in local_reasons:
            edges.extend(Edge(key, reason["reason_id"], "supports") for key in reason["facts"])
            edges.append(Edge(reason["reason_id"], action["action_id"], "motivates"))
        graphs.append(ActionGraph(
            action_id=action["action_id"], action_text=action["description"],
            facts=tuple(FactNode(key, facts[key]["text"]) for key in fact_ids),
            reasons=tuple(ReasonNode(reason["reason_id"], reason["text"],
                                    (reason.get("tags") or [""])[0].strip().lower() or None)
                          for reason in local_reasons),
            edges=tuple(edges),
        ))
    return ReasoningInstance(instance_id=row_id(payload), action_graphs=tuple(graphs),
                             chosen_action_id=payload.get("chosen_action_id"))


def to_instance(row, format):
    from nora._scoring import load_instance

    clip_id = row_id(row)
    if row.get("status") == "failed":
        raise ValueError("generation_failed")
    if "response" in row and "prediction" not in row:
        raise ValueError("raw_response_requires_reconstruction: run nora reconstruct before evaluation")
    payload = row.get("prediction", row)
    if not isinstance(payload, dict):
        raise ValueError("prediction_not_object")
    if row_id(payload) != clip_id:
        raise ValueError("prediction_clip_id_mismatch")
    if format == "annotation":
        validate_annotation(payload)
        return _annotation_instance(payload)
    if format == "graph":
        # Canonical evaluator shape, not the separate reconstruction wire format.
        if "action_graphs" not in payload or not isinstance(payload["action_graphs"], list):
            raise ValueError("missing_action_graphs")
        actions = _nodes(payload, "action_graphs", "action_id", "action_text")
        chosen = payload.get("chosen_action_id")
        if chosen is not None and (not isinstance(chosen, str) or chosen not in actions):
            raise ValueError("dangling_chosen_action_id")
        for graph in payload["action_graphs"]:
            for field in ("facts", "reasons"):
                _nodes(dict(graph, **{field: graph.get(field, [])}), field, "id", "text")
            for reason in graph.get("reasons", []):
                label = reason.get("normative_label")
                if label is not None and not isinstance(label, str):
                    raise ValueError("invalid_normative_label")
            edges = graph.get("edges", [])
            if not isinstance(edges, list) or any(
                not isinstance(edge, dict) or any(
                    not isinstance(edge.get(key), str) or not edge[key].strip()
                    for key in ("src", "dst", "type")) for edge in edges
            ):
                raise ValueError("invalid_edges")
        return load_instance(dict(payload, instance_id=clip_id))
    raise ValueError("format must be graph or annotation")
