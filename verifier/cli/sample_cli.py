from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifier.artifacts.paths import get_render_artifact_path, get_sample_artifact_dir
from verifier.pipeline.batch import _resolve_html_path
from verifier.pipeline.runner import load_sample_from_jsonl, validate_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one refined sample against an HTML file.")
    parser.add_argument("--jsonl", required=True, help="Path to refined sample JSONL.")
    parser.add_argument("--sample-id", required=True, help="Sample id to validate.")
    parser.add_argument("--html", help="Path to HTML file.")
    parser.add_argument("--html-dir", help="Directory containing per-sample HTML files.")
    parser.add_argument(
        "--html-pattern",
        default="{sample_id}.html",
        help="Path pattern relative to --html-dir, for example '{sample_id}.html'. Supports {sample_id} and {application}.",
    )
    parser.add_argument(
        "--artifact-backend",
        choices=("auto", "static", "playwright"),
        default="auto",
        help="Artifact geometry backend. 'auto' tries Playwright and falls back to static inference.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation report as JSON instead of a text summary.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.html and not args.html_dir:
        parser.error("one of --html or --html-dir is required")

    sample = load_sample_from_jsonl(Path(args.jsonl), args.sample_id)
    html_path = Path(args.html) if args.html else _resolve_html_path(
        sample,
        html_dir=Path(args.html_dir) if args.html_dir else None,
        html_pattern=args.html_pattern,
        html_manifest={},
        manifest_base_dir=None,
    )
    report = validate_sample(sample, html_path, artifact_backend=args.artifact_backend)

    if args.json:
        payload = {
            "sample_id": report.sample_id,
            "artifact_dir": str(get_sample_artifact_dir(report.sample_id)),
            "source_html": str(get_sample_artifact_dir(report.sample_id) / "source.html"),
            "semantic_html": str(get_sample_artifact_dir(report.sample_id) / "semantic.html"),
            "render_image": str(get_render_artifact_path(sample_id=report.sample_id, backend="playwright")),
            "report_json": str(get_sample_artifact_dir(report.sample_id) / f"report_{args.artifact_backend}.json"),
            "execution_pass": report.execution_pass,
            "quality_pass": report.quality_pass,
            "overall_pass": report.overall_pass,
            "hard_failures": report.hard_failures,
            "summary": report.summary,
            "results": [
                {
                    "requirement_id": result.requirement_id,
                    "validator_name": result.validator_name,
                    "status": result.status.value,
                    "score": result.score,
                    "expected": result.expected,
                    "measured": result.measured,
                    "message": result.message,
                    "evidence": result.evidence,
                    "debug": result.debug,
                }
                for result in report.results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"sample_id={report.sample_id}")
    print(f"artifact_dir={get_sample_artifact_dir(report.sample_id)}")
    print(f"source_html={get_sample_artifact_dir(report.sample_id) / 'source.html'}")
    print(f"semantic_html={get_sample_artifact_dir(report.sample_id) / 'semantic.html'}")
    print(f"render_image={get_render_artifact_path(sample_id=report.sample_id, backend='playwright')}")
    print(f"report_json={get_sample_artifact_dir(report.sample_id) / f'report_{args.artifact_backend}.json'}")
    print(f"execution_pass={report.execution_pass}")
    print(f"quality_pass={report.quality_pass}")
    print(f"overall_pass={report.overall_pass}")
    if isinstance(report.summary, dict):
        scores = report.summary.get("scores", {})
        if isinstance(scores, dict) and "execution_score" in scores:
            print(f"execution_score={scores['execution_score']}")
    print(f"summary={report.summary}")
    for result in report.results:
        print(f"{result.requirement_id}\t{result.status.value}\t{result.score}\t{result.message}")


if __name__ == "__main__":
    main()