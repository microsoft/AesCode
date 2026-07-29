#!/usr/bin/env python3
"""Pre-warm verifier caches and models for all reference images.

Three things are pre-warmed:
  1. **OCR disk cache** (biggest win) — PaddleOCR results are cached per-image
     on disk.  This eliminates the 2-13s/image OCR bottleneck for subsequent
     verify runs.
  2. **LPIPS model** — AlexNet backbone loaded onto GPU (one-time ~1s).
  3. **SentenceTransformer** — all-MiniLM-L6-v2 loaded into memory (~90MB).

Usage
-----
    # Full pre-warm (all images, GPU OCR)
    python -m verifier.cli.prewarm_ocr_cache \
        --image-dir <data_root>/images \
        --workers 8

    # Only pre-warm images referenced in a spec JSONL
    python -m verifier.cli.prewarm_ocr_cache \
        --image-dir <data_root>/images \
        --spec-jsonl data/sft_pipeline/test50_specs.jsonl \
        --workers 4

    # Custom cache directory
    VERIFIER_OCR_CACHE_DIR=/scratch/ocr_cache python -m verifier.cli.prewarm_ocr_cache \
        --image-dir <data_root>/images
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set


def _collect_image_paths(
    image_dir: Path,
    spec_jsonl: Optional[Path] = None,
) -> List[Path]:
    """Collect reference image paths to pre-warm.

    If spec_jsonl is given, only collect images for samples in the spec.
    Otherwise, collect all image files in image_dir.
    """
    extensions = {".png", ".jpg", ".jpeg", ".webp"}

    if spec_jsonl is not None:
        # Collect sample IDs from spec
        sample_ids: Set[str] = set()
        with spec_jsonl.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"WARN: skipping malformed line {ln} in {spec_jsonl}: {exc}")
                    continue
                sample_ids.add(str(data.get("id", "")))

        # Find matching images
        paths: List[Path] = []
        for sid in sorted(sample_ids):
            for ext in extensions:
                candidate = image_dir / f"{sid}{ext}"
                if candidate.exists():
                    paths.append(candidate)
                    break
        return paths

    # Collect all images in dir
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def _prewarm_models() -> None:
    """Pre-load LPIPS and SentenceTransformer models into memory/GPU."""
    # ── LPIPS (AlexNet → GPU) ──
    print("[model] Loading LPIPS (AlexNet)...", end=" ", flush=True)
    t0 = time.time()
    try:
        from verifier.validators.visual import _get_lpips_model
        model = _get_lpips_model()
        device = "unknown"
        try:
            device = str(next(model.parameters()).device)
        except Exception:
            pass
        print(f"OK ({time.time()-t0:.1f}s, device={device})")
    except Exception as exc:
        print(f"FAILED ({exc})")

    # ── SentenceTransformer (CPU) ──
    print("[model] Loading SentenceTransformer (all-MiniLM-L6-v2)...", end=" ", flush=True)
    t0 = time.time()
    try:
        from verifier.validators.contentlayout import _get_sbert_model
        model = _get_sbert_model()
        status = "OK" if model is not None else "unavailable"
        print(f"{status} ({time.time()-t0:.1f}s)")
    except Exception as exc:
        print(f"FAILED ({exc})")


def _prewarm_one(image_path: Path) -> dict:
    """Run OCR on one image, populating the cache."""
    from verifier.validators.contentlayout import (
        _extract_reference_text_blocks,
        _ocr_cache_get,
    )

    str_path = str(image_path)

    # Check if already cached
    cached = _ocr_cache_get(str_path)
    if cached is not None:
        blocks, _ = cached
        return {
            "path": str_path,
            "status": "cached",
            "block_count": len(blocks),
        }

    t0 = time.time()
    try:
        blocks, debug = _extract_reference_text_blocks(str_path)
        elapsed = time.time() - t0
        return {
            "path": str_path,
            "status": "computed",
            "block_count": len(blocks),
            "elapsed": round(elapsed, 2),
            "error": debug.get("error"),
        }
    except Exception as exc:
        return {
            "path": str_path,
            "status": "error",
            "error": str(exc)[:200],
            "elapsed": round(time.time() - t0, 2),
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-warm verifier caches (OCR disk cache, LPIPS, SBERT).",
    )
    p.add_argument("--image-dir", type=Path, required=True,
                   help="Directory containing reference images")
    p.add_argument("--spec-jsonl", type=Path, default=None,
                   help="Only pre-warm images for samples in this spec JSONL")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for OCR")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit to first N images (0 = no limit)")
    p.add_argument("--shard", type=str, default="",
                   help="Shard spec 'K/N': take every N-th image starting at K "
                        "(0-indexed). Lets you run N parallel prewarm processes "
                        "with no overlap.")
    p.add_argument("--skip-models", action="store_true",
                   help="Skip LPIPS/SBERT model pre-loading (OCR only)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        print(f"ERROR: --image-dir {image_dir} does not exist or is not a directory")
        sys.exit(1)

    # ── Phase 1: Model pre-loading ──
    if not args.skip_models:
        print("=" * 60)
        print("Phase 1: Pre-loading verifier models")
        print("=" * 60)
        _prewarm_models()
        print()

    # ── Phase 2: OCR disk cache ──
    print("=" * 60)
    print("Phase 2: OCR disk cache pre-warm")
    print("=" * 60)

    paths = _collect_image_paths(image_dir, args.spec_jsonl)
    if args.limit > 0:
        paths = paths[:args.limit]
    if args.shard:
        try:
            k_str, n_str = args.shard.split("/")
            k, n = int(k_str), int(n_str)
            if not (0 <= k < n):
                raise ValueError("K must be in [0, N)")
        except (ValueError, IndexError) as exc:
            print(f"ERROR: --shard must be 'K/N' with 0 <= K < N (got '{args.shard}'): {exc}")
            sys.exit(1)
        paths = paths[k::n]
        print(f"Shard {k}/{n}: processing {len(paths)} images")

    print(f"Images to process: {len(paths)}")
    print(f"Workers: {args.workers}")

    # Show cache dir
    from verifier.validators.contentlayout import _get_ocr_cache_dir
    cache_dir = _get_ocr_cache_dir()
    print(f"Cache directory: {cache_dir}")

    if not paths:
        print("Nothing to do.")
        return

    cached_count = 0
    computed_count = 0
    error_count = 0
    total_ocr_time = 0.0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_prewarm_one, p): p for p in paths}
        for fut in as_completed(futures):
            result = fut.result()

            if result["status"] == "cached":
                cached_count += 1
            elif result["status"] == "computed":
                computed_count += 1
                total_ocr_time += result.get("elapsed", 0)
            else:
                error_count += 1
                print(f"  ERROR {Path(result['path']).name}: {result.get('error', '?')[:80]}")

            total = cached_count + computed_count + error_count
            if total % 50 == 0 or total == len(paths):
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(paths) - total) / rate / 60 if rate > 0 else 0
                print(
                    f"  [{total}/{len(paths)}] cached={cached_count} "
                    f"computed={computed_count} errors={error_count} "
                    f"rate={rate:.1f}/s ETA={eta:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f}min")
    print(f"  Cached (skipped): {cached_count}")
    print(f"  Computed (new):   {computed_count}")
    print(f"  Errors:           {error_count}")
    if computed_count > 0:
        print(f"  Avg OCR time:     {total_ocr_time/computed_count:.2f}s/image")


if __name__ == "__main__":
    main()
