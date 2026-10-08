"""Small user-facing command surface; API calls occur only in explicit commands."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path
import sys
import tempfile

from nora.data import read_rows, reference_path


def build_parser():
    p = argparse.ArgumentParser(prog="nora", description="Run models and evaluate predictions on NoRA.")
    commands = p.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Free synthetic demo; not a benchmark score.")
    demo.add_argument("--output")
    data = commands.add_parser("data", help="Show the reference file path (test: bundled; train: HF).")
    data.add_argument("--split", choices=["train", "test"], default="test")
    data.add_argument("--offline", action="store_true")
    val = commands.add_parser("validate", help="Check prediction structure and links without scoring.")
    val.add_argument("predictions", help="Prediction JSON/JSONL file.")
    val.add_argument("--prediction-format", choices=["graph", "annotation"], default="annotation",
                     help="Prediction schema (default: annotation).")
    ev = commands.add_parser("evaluate", help="Score predictions and report coverage.")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--references", help="Local JSON/JSONL; defaults to bundled test annotations.")
    ev.add_argument("--prediction-format", choices=["graph", "annotation"], default="annotation",
                    help="Prediction schema (default: annotation).")
    ev.add_argument("--reference-format", choices=["graph", "annotation"], default="annotation",
                    help="Reference schema (default: annotation).")
    ev.add_argument("--output", required=True, help="New directory for scores and coverage.")
    ev.add_argument("--offline", action="store_true")
    ev.add_argument("--progress", action="store_true", help="Print per-clip progress to stderr.")
    comp = commands.add_parser("compare", help="Compare runs on the same valid examples.")
    comp.add_argument("runs", nargs="+", help="At least two evaluation output directories.")
    prep = commands.add_parser("download-media", help="Download original pre-action frame montages or videos.")
    prep.add_argument("--references", help="Local JSONL; defaults to bundled test annotations.")
    prep.add_argument("--media", choices=["frames", "video"], default="frames")
    prep.add_argument("--output", required=True, help="Root directory for per-clip media files.")
    prep.add_argument("--limit", type=int)
    pred = commands.add_parser("predict", help="Generate model responses from frames or video; API fees may apply.")
    pred.add_argument("--references", help="Local JSONL; defaults to bundled test annotations.")
    pred.add_argument("--media-root", required=True, help="Directory produced by nora download-media.")
    pred.add_argument("--media", choices=["frames", "video"], default="frames",
                      help="Paper setting: frames. Native video is a separate experimental setting.")
    pred.add_argument("--model", required=True)
    pred.add_argument("--base-url", required=True, help="Compatible endpoint ending in /v1.")
    pred.add_argument("--api-key-env", default="NORA_API_KEY")
    pred.add_argument("--prompt", choices=["direct", "deliberate", "structured"], default="structured")
    pred.add_argument("--output", required=True, help="New JSONL file for model responses.")
    pred.add_argument("--limit", type=int)
    pred.add_argument("--max-tokens", type=int, default=4096)
    rec = commands.add_parser("reconstruct", help="Convert model responses to annotations using a paid API.")
    rec.add_argument("--input", required=True, help="JSON/JSONL file containing raw response records.")
    rec.add_argument("--output", required=True, help="New JSONL file for reconstructed annotations.")
    rec.add_argument("--model", required=True, help="OpenAI Responses model; OPENAI_API_KEY required.")
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    try:
        result = _run_command(args)
    except Exception as exc:
        # Do not print provider response bodies, credentials, or traceback locals.
        if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError, ImportError)):
            p.exit(2, f"nora: {exc}\n")
        p.exit(2, f"nora: {type(exc).__name__}; operation failed. Check setup and connectivity.\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if isinstance(result, dict) and result.get("status") == "failed":
        print("nora: no valid predictions were scored; see coverage.json and failures.jsonl.",
              file=sys.stderr)
        return 1
    return 1 if isinstance(result, dict) and result.get("invalid", 0) else 0


def _run_command(args):
    if args.command == "data":
        return {"path": str(reference_path(args.split, offline=args.offline))}
    if args.command == "validate":
        from nora.adapters import validate
        return validate(args.predictions, prediction_format=args.prediction_format)
    if args.command == "demo":
        from nora.evaluation import evaluate
        assets = files("nora") / "assets"
        output = args.output or str(Path(tempfile.mkdtemp(prefix="nora-demo-")) / "scores")
        return {"notice": "SYNTHETIC DEMO ONLY: these are not benchmark scores.",
                **evaluate(assets / "demo_pred.json", assets / "demo_gold.json",
                           output=output, prediction_format="graph", reference_format="graph", _demo=True)}
    if args.command == "evaluate":
        from nora.evaluation import evaluate
        refs = args.references or reference_path(offline=args.offline)
        progress = (lambda done, total, key: print(f"[{done}/{total}] {key}",
                                                   file=sys.stderr, flush=True)) if args.progress else None
        return evaluate(args.predictions, refs, output=args.output, prediction_format=args.prediction_format,
                        reference_format=args.reference_format, offline=args.offline, progress=progress)
    if args.command == "compare":
        from nora.evaluation import compare
        return compare(args.runs)
    if args.command in {"download-media", "predict"}:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = read_rows(args.references or reference_path())
        if args.limit:
            rows = rows[:args.limit]
    if args.command == "download-media":
        from nora.media import download_media
        return download_media(rows, output=args.output, media=args.media)
    if args.command == "predict":
        from nora.data import load_prompts
        from nora.models import ChatCompletionsModel, predict
        if args.max_tokens <= 0:
            raise ValueError("max-tokens must be positive")
        prompt = next(p for p in load_prompts() if p["mode"] == args.prompt)
        model = ChatCompletionsModel(base_url=args.base_url, model=args.model,
                                    api_key_env=args.api_key_env, max_tokens=args.max_tokens)
        output = predict(rows, model, media_root=args.media_root, media=args.media,
                         prompt=prompt, output=args.output, model_id=args.model)
        saved = read_rows(output)
        return {"output": str(output), "rows": len(saved),
                "invalid": sum(row["status"] != "ok" for row in saved)}
    from nora.reconstruction import reconstruct
    return reconstruct(args.input, output=args.output, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
