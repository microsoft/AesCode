from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verifier.pipeline.batch import validate_batch


# Default output root — outside the repo, organised by model.
_DEFAULT_OUTPUT_ROOT = Path(os.environ.get(
    "AESCODE_OUTPUT_ROOT",
    str(Path.home() / "AesCode-output"),
))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a refined sample JSONL in batch and compare artifact backends.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Inference run directory (contains run_metadata.json). Auto-discovers --jsonl and --html-dir from it.",
    )
    parser.add_argument("--jsonl", default=None, help="Path to spec/refined sample JSONL. Inferred from --run-dir when not provided.")
    parser.add_argument("--html-dir", help="Directory containing per-sample HTML files. Inferred from --run-dir when not provided.")
    parser.add_argument(
        "--html-pattern",
        default="{sample_id}.html",
        help="Path pattern relative to --html-dir, for example '{sample_id}.html'. Supports {sample_id} and {application}.",
    )
    parser.add_argument(
        "--html-manifest",
        help="Optional JSON or JSONL mapping sample ids to HTML paths. Overrides --html-pattern when provided.",
    )
    parser.add_argument(
        "--artifact-backends",
        default="static,playwright",
        help="Comma-separated artifact backends to run, for example 'static,playwright' or 'auto'.",
    )
    parser.add_argument(
        "--sample-ids",
        help="Optional comma-separated sample ids to run from the JSONL, for example 'image_0001,image_0003'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of samples to run after filtering.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Shuffle samples before applying --limit.",
    )
    parser.add_argument(
        "--output-jsonl",
        help="Optional path to write one JSON object per sample row.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full batch payload as JSON instead of a text summary.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory for validation outputs. Defaults to ~/AesCode-output "
            "(overridable via AESCODE_OUTPUT_ROOT env var). When --run-dir is used, "
            "outputs go to {output-root}/{run_dir_name}/."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --run-dir: auto-discover jsonl + html-dir from run_metadata.json
    if args.run_dir is not None:
        metadata_path = args.run_dir / "run_metadata.json"
        if not metadata_path.exists():
            parser.error(f"--run-dir does not contain run_metadata.json: {args.run_dir}")
        run_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.jsonl is None:
            args.jsonl = run_meta.get("input_jsonl")
        if args.html_dir is None:
            args.html_dir = run_meta.get("output_dir")

    if not args.jsonl:
        parser.error("Provide --jsonl or --run-dir (to auto-discover from run_metadata.json).")
    if not args.html_dir:
        parser.error("Provide --html-dir or --run-dir (to auto-discover from run_metadata.json).")

    backends = [item.strip() for item in args.artifact_backends.split(",") if item.strip()]
    sample_ids = [item.strip() for item in (args.sample_ids or "").split(",") if item.strip()] or None

    # ── Output directory: always outside the repo, organised by model ──
    output_root = args.output_root or _DEFAULT_OUTPUT_ROOT
    if args.run_dir is not None:
        run_dir_name = args.run_dir.resolve().name
        model_output_dir = output_root / run_dir_name
    else:
        model_output_dir = output_root / "manual"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    artifact_root = model_output_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    if not getattr(args, "output_jsonl", None) or not args.output_jsonl:
        args.output_jsonl = str(model_output_dir / "validation_report.jsonl")

    print(f"Output directory: {model_output_dir}")
    print(f"Artifact root:    {artifact_root}")
    print(f"Report JSONL:     {args.output_jsonl}")

    batch_payload = validate_batch(
        Path(args.jsonl),
        html_dir=Path(args.html_dir) if args.html_dir else None,
        html_pattern=args.html_pattern,
        html_manifest_path=Path(args.html_manifest) if args.html_manifest else None,
        artifact_backends=backends,
        artifact_root=artifact_root,
        sample_ids=sample_ids,
        limit=args.limit,
        shuffle=args.random,
    )

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in batch_payload["rows"]) + "\n",
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(batch_payload, ensure_ascii=False, indent=2))
        return

    print(f"processed_samples={batch_payload['processed_samples']}")
    print(f"selected_sample_ids={batch_payload['selected_sample_ids']}")
    print(f"missing_html_samples={len(batch_payload['missing_html_samples'])}")
    print(f"artifact_backends={batch_payload['artifact_backends']}")
    print(f"inconsistent_backend_pairs={batch_payload['inconsistent_backend_pairs']}")
    for row in batch_payload["rows"]:
        if "error" in row:
            print(f"{row['sample_id']}\terror\t{row['error']}\t{row['html_path']}")
            continue
        comparison_summaries = [
            f"{comparison['baseline_backend']}->{comparison['candidate_backend']}:{comparison['difference_count']}"
            for comparison in row["comparisons"]
        ]
        backend_summary_parts = []
        for backend in batch_payload["artifact_backends"]:
            backend_summary_parts.append(f"{backend}:{row['reports'][backend]['summary']}")
        print(
            f"{row['sample_id']}\t"
            f"artifact_dir={row['artifact_dir']}\t"
            f"{','.join(backend_summary_parts)}\t"
            f"comparisons={comparison_summaries}"
        )


if __name__ == "__main__":
    main()