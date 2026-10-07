#!/usr/bin/env python3
"""Pack spec + target HTML into SFT training JSONL.

Usage
-----
    python data_prep/scripts/pack_sft_data.py \
        --spec-jsonl  data/sft_pipeline/specs.jsonl \
        --draft-dir   data/sft_pipeline/draft_html \
        --clean-html-dir data/sft_pipeline/clean_html \
        --output      data/sft_pipeline/sft_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401


def _strip_outer_fences(html: str) -> str:
    """Remove markdown code fences if present."""
    match = re.search(r"```(?:html)?\s*([\s\S]+?)```", html, re.IGNORECASE)
    return match.group(1).strip() if match else html.strip()


def _is_valid_html(html: str) -> bool:
    """Basic check that the string looks like an HTML document."""
    lower = html.lower()
    return "<!doctype html>" in lower or "<html" in lower


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pack SFT training data from pipeline outputs.")
    p.add_argument("--spec-jsonl", type=Path, required=True, help="Spec JSONL from Stage 2")
    p.add_argument("--draft-dir", type=Path, default=None, help="Directory with draft HTML (optional, for buggy_html)")
    p.add_argument("--clean-html-dir", type=Path, default=None, help="Directory with the target/clean HTML")
    p.add_argument("--html-dir", type=Path, default=None,
                   help="Single HTML dir (when only draft is available, used as clean_html)")
    p.add_argument("--output", type=Path, required=True, help="Output SFT JSONL path")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="Override directory for reference images (fallback: processed_data_v4)")
    p.add_argument("--min-html-length", type=int, default=200,
                   help="Discard HTML shorter than this (default: 200 chars)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Determine the source of clean_html
    if args.clean_html_dir:
        clean_dir = args.clean_html_dir.resolve()
        buggy_dir = args.draft_dir.resolve() if args.draft_dir else None
    elif args.html_dir:
        clean_dir = args.html_dir.resolve()
        buggy_dir = None
    else:
        print("ERROR: provide either --clean-html-dir or --html-dir", file=sys.stderr)
        sys.exit(1)

    # Load specs
    specs: List[Dict[str, Any]] = []
    with args.spec_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))
    print(f"Specs loaded: {len(specs)}")

    written = 0
    skipped_no_html = 0
    skipped_invalid = 0
    skipped_short = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out_f:
        for spec in specs:
            sid = str(spec.get("id", ""))

            # Read clean HTML
            clean_path = clean_dir / f"{sid}.html"
            if not clean_path.exists():
                skipped_no_html += 1
                continue
            clean_html = _strip_outer_fences(clean_path.read_text(encoding="utf-8"))

            if not _is_valid_html(clean_html):
                skipped_invalid += 1
                continue
            if len(clean_html) < args.min_html_length:
                skipped_short += 1
                continue

            # Read buggy HTML (optional)
            buggy_html = None
            if buggy_dir:
                buggy_path = buggy_dir / f"{sid}.html"
                if buggy_path.exists():
                    buggy_html = _strip_outer_fences(buggy_path.read_text(encoding="utf-8"))
                    if not _is_valid_html(buggy_html):
                        buggy_html = None

            # Resolve image path from spec
            ref_meta = spec.get("reference_metadata") or {}
            target_image = ref_meta.get("reference_image_path", "")
            if args.image_dir:
                # Use explicit image dir
                target_image = str(args.image_dir / f"{sid}.png")
            elif not target_image or not Path(target_image).exists():
                # Fallback: try processed_data_v4
                fallback = f"${AESCODE_OUTPUT_ROOT}/processed_data_v4/{sid}.png"
                if Path(fallback).exists():
                    target_image = fallback

            record = {
                "id": sid,
                "target_image": target_image,
                "clean_html": clean_html,
                "generation_prompt": spec.get("generation_prompt", ""),
                "reference_metadata": ref_meta,
                "verifiable_requirements": spec.get("verifiable_requirements", []),
                "difficulty": 2,
            }
            if buggy_html:
                record["buggy_html"] = buggy_html

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print()
    print("=" * 60)
    print(f"Total specs:       {len(specs)}")
    print(f"Written:           {written}")
    print(f"Skipped (no HTML): {skipped_no_html}")
    print(f"Skipped (invalid): {skipped_invalid}")
    print(f"Skipped (short):   {skipped_short}")
    print(f"Output:            {args.output}")
    print("=" * 60)

    # Quick stats
    if written > 0:
        with args.output.open("r", encoding="utf-8") as f:
            has_buggy = sum(1 for line in f if '"buggy_html"' in line)
        print(f"  with buggy_html (repair-capable): {has_buggy}")
        print(f"  generate-only:                    {written - has_buggy}")


if __name__ == "__main__":
    main()
