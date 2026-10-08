"""Validate and score predictions, report coverage, and compare evaluation runs."""

from __future__ import annotations

import csv
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import subprocess

from nora.adapters import to_instance
from nora.data import digest, index_rows, read_rows, reference_path, write_json

SCORER_ID = "cross-encoder/stsb-roberta-base"
SCORER_REVISION = "d576534b67143e2c70ee9966d7fdbf5835728d13"
METRICS = ("action_f1", "fact_f1", "reasoning_f1", "summary_score", "reasonableness_score")
PROTOCOL = "nora-v2-support-graph"


def build_scorer(*, offline=False, demo=False):
    from nora._scoring import HybridSimilarityScorer
    from nora._scoring.embedding_backend import _HashingFallbackBackend
    from nora.scoring import CachedCrossEncoder

    if demo:
        return HybridSimilarityScorer(backend=_HashingFallbackBackend())
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder

    snapshot = snapshot_download(
        SCORER_ID, revision=SCORER_REVISION, local_files_only=offline,
        allow_patterns=["*.json", "merges.txt", "model.safetensors"],
    )
    backend = CachedCrossEncoder(CrossEncoder(snapshot, local_files_only=True), SCORER_ID)
    return HybridSimilarityScorer(backend=backend)


def summarize(rows):
    summary = {}
    for mode in ("soft", "hungarian"):
        selected = [row for row in rows if row["mode"] == mode]
        count = len(selected)
        summary[mode] = {
            key: sum(row[key] for row in selected) / count if count else None
            for key in METRICS
        }
        summary[mode]["n"] = count
    return summary


def evaluate(predictions, references=None, *, output, prediction_format="annotation",
             reference_format="annotation", offline=False, progress=None, _demo=False):
    """Score a JSON/JSONL prediction file against references (bundled test by default).

    ``prediction_format`` and ``reference_format`` each accept ``annotation``
    (clip_id/facts/reasons/actions) or ``graph`` (instance_id/action_graphs).
    Raw prose must be reconstructed first. Write results to a new output
    directory and return status, coverage, scores, and its path. Missing/invalid
    cases are excluded; zero valid cases return status ``failed`` and null scores.
    ``_demo`` is internal and marks outputs as non-benchmark.
    ``progress``, if supplied, receives (completed, total, clip_id).
    """
    from nora._scoring import evaluate_instance

    for name, value in (("prediction_format", prediction_format),
                        ("reference_format", reference_format)):
        if value not in {"annotation", "graph"}:
            raise ValueError(f"{name} must be annotation or graph")
    references = references or reference_path(offline=offline)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output_exists: {destination}")
    gold_rows = index_rows(read_rows(references))
    if not gold_rows:
        raise ValueError("empty_reference_file")
    # Invalid reference data is a dataset error, not a prediction failure.
    gold = {key: to_instance(row, reference_format) for key, row in gold_rows.items()}
    pred_rows = index_rows(read_rows(predictions))
    reconstruction_receipt = None
    receipt_path = Path(str(predictions) + ".receipt.json")
    if receipt_path.exists():
        reconstruction_receipt = json.loads(receipt_path.read_text())
        if reconstruction_receipt.get("output_sha256") != digest(predictions):
            raise ValueError("reconstruction_receipt_hash_mismatch")
    unknown = sorted(set(pred_rows) - set(gold))
    if unknown:
        raise ValueError(f"unknown_clip_ids: {unknown[:5]}")
    valid = {}
    failures = []
    for key in sorted(gold):
        if key not in pred_rows:
            failures.append({"clip_id": key, "stage": "input", "code": "missing_prediction"})
            continue
        row = pred_rows[key]
        try:
            valid[key] = to_instance(row, prediction_format)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            failures.append({"clip_id": key, "stage": "validation",
                             "code": type(exc).__name__, "detail": str(exc)})
    metrics = []
    details = []
    if valid:
        scorer = build_scorer(offline=offline, demo=_demo)
        for completed, (key, pred) in enumerate(valid.items(), start=1):
            if hasattr(scorer.backend, "prepare"):
                scorer.backend.prepare(pred, gold[key])
            for mode in ("soft", "hungarian"):
                result = evaluate_instance(pred, gold[key], scorer=scorer, mode=mode)
                details.append(result.to_dict())
                metrics.append({"clip_id": key, "mode": mode,
                                **{name: float(getattr(result, name)) for name in METRICS},
                                "fact_available": result.fact_available,
                                "graph_available": result.graph_available})
            if progress is not None:
                progress(completed, len(valid), key)
    # Scoring failures abort the run rather than changing its valid-case cohort.
    destination.mkdir(parents=True, exist_ok=False)
    coverage = {"expected": len(gold), "submitted": len(pred_rows), "valid": len(valid),
                "missing": len(set(gold)-set(pred_rows)),
                "invalid": len(pred_rows)-len(valid),
                "valid_fraction": len(valid)/len(gold)}
    status = "failed" if not valid else ("partial" if failures else "complete")
    manifest = {
        "status": status,
        "version": "0.4.0", "protocol": PROTOCOL,
        "benchmark_score": not _demo,
        "reconstruction": "provided" if reconstruction_receipt else "none",
        "prediction_format": prediction_format, "reference_format": reference_format,
        "reference_sha256": digest(references), "prediction_sha256": digest(predictions),
        "scorer": {"id": "synthetic-hashing-demo" if _demo else SCORER_ID,
                   "revision": None if _demo else SCORER_REVISION},
        "cohort_policy": "valid-only; missing/invalid recorded separately",
        "action_policy": "all submitted candidate actions; supporting links only",
        "prompt_ids": sorted({str(row.get("prompt_id", "unspecified")) for row in pred_rows.values()}),
        "model_ids": sorted({str(row.get("model_id", "unspecified")) for row in pred_rows.values()}),
        "reconstruction_receipt": reconstruction_receipt,
        "source_hashes": {
            f"nora/{path.relative_to(Path(__file__).parent)}": digest(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "dependencies": {dist.metadata["Name"]: dist.version for dist in distributions()
                         if dist.metadata["Name"]},
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        manifest["git_commit"] = None
    write_json(destination / "run.json", manifest)
    write_json(destination / "summary.json", summarize(metrics))
    write_json(destination / "coverage.json", coverage)
    write_json(destination / "cohort_ids.json", sorted(valid))
    with (destination / "instance_metrics.jsonl").open("w") as handle:
        for detail in details:
            handle.write(json.dumps(detail, allow_nan=False) + "\n")
    with (destination / "instance_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "mode", *METRICS,
                                                    "fact_available", "graph_available"])
        writer.writeheader()
        writer.writerows(metrics)
    with (destination / "failures.jsonl").open("w") as handle:
        for failure in failures:
            handle.write(json.dumps(failure) + "\n")
    return {"status": status, "coverage": coverage, "scores": summarize(metrics),
            "output": str(destination)}


def compare(run_dirs):
    """Compare at least two saved run directories on their shared valid clip IDs.

    Return shared IDs, sample count, and each run's mean scores without changing
    saved results. Incompatible scoring settings raise ValueError. No API calls
    or rescoring are needed; an empty intersection returns null scores.
    """
    manifests, systems = [], []
    for directory in run_dirs:
        root = Path(directory)
        manifests.append(json.loads((root / "run.json").read_text()))
        with (root / "instance_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        cohort = json.loads((root / "cohort_ids.json").read_text())
        if not isinstance(cohort, list) or any(not isinstance(key, str) for key in cohort):
            raise ValueError("invalid_cohort_ids")
        expected = {(key, mode) for key in cohort for mode in ("soft", "hungarian")}
        actual = [(row["clip_id"], row["mode"]) for row in rows]
        if len(cohort) != len(set(cohort)) or len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("inconsistent_per_case_scores")
        for row in rows:
            for key in METRICS:
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError("nonfinite_per_case_score")
        systems.append(rows)
    if len(systems) < 2:
        raise ValueError("compare_requires_at_least_two_runs")
    for name in ("protocol", "reference_sha256", "scorer", "prediction_format",
                 "reference_format", "benchmark_score", "prompt_ids", "reconstruction", "source_hashes"):
        if any(manifest[name] != manifests[0][name] for manifest in manifests):
            raise ValueError(f"incompatible_runs: {name}")
    reconstruction_settings = [
        {key: (m.get("reconstruction_receipt") or {}).get(key)
         for key in ("model", "prompt_sha256", "path")} for m in manifests
    ]
    if any(settings != reconstruction_settings[0] for settings in reconstruction_settings):
        raise ValueError("incompatible_runs: reconstruction_settings")
    shared = set.intersection(*({row["clip_id"] for row in rows} for rows in systems))
    return {"cohort_ids": sorted(shared), "n": len(shared),
            "interpretation": "Conditional on validity in all systems; not full-set capability.",
            "systems": [{"run": str(path), "scores": summarize(
                [row for row in rows if row["clip_id"] in shared])}
                for path, rows in zip(run_dirs, systems)]}
