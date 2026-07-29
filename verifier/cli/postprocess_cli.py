from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifier.artifacts.semantic import apply_semantic_attribute_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python verifier/scripts/postprocess_semantic_html.py",
        description="Apply stable semantic data attributes to an HTML file.",
    )
    parser.add_argument("--html", required=True, help="Input HTML path.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument(
        "--contract-json",
        default="",
        help="Optional JSON file containing semantic_attribute_contract. If omitted, defaults are used.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the semantic enrichment summary as JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    html_path = Path(args.html).resolve()
    output_path = Path(args.output).resolve()
    contract = {}
    if args.contract_json.strip():
        contract = json.loads(Path(args.contract_json).resolve().read_text(encoding="utf-8"))
    enriched_html, summary = apply_semantic_attribute_contract(
        html_path.read_text(encoding="utf-8"),
        contract,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(enriched_html, encoding="utf-8")
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
