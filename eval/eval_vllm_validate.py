#!/usr/bin/env python3
"""Stage 2: render every generated page and run the rule validators over it.

Each rollout is rendered headless with Playwright at the fixed canvas size, the
screenshot and DOM artifacts are written under results/artifacts/<sid>_k<k>/, and
the rule validators report per-dimension scores into results/validation_report.jsonl.
Rendering is CPU-bound, so the work is spread with multiprocessing.Pool.

The screenshot this stage produces is the ONLY thing the checklist judge in stage 3
ever sees, so a rollout that fails to render has no judgment and is scored as a
total failure under the fixed denominator. Keep --workers modest: rendering at high
concurrency drops a few percent of rollouts that render fine at --workers 8, and
eval/render_parity.py exists to find and retry exactly those.

Usage:
    # The benchmark setting: n=3 rollouts per sample
    DISABLED_VALIDATORS=DesignValidator python3 eval/eval_vllm_validate.py \
        --spec-jsonl data/eval_spec.jsonl \
        --html-dir runs/my_model_final300/html \
        --output-dir runs/my_model_final300/results \
        --ref-image-dir data/images \
        --pattern sample --n 3 --workers 8 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Bootstrap
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
if (_REPO_ROOT / ".env").exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Worker GPU assignment – round-robin across available GPUs
# ---------------------------------------------------------------------------
_worker_gpu_counter: Optional[Any] = None  # multiprocessing.Value


def _worker_init(gpu_counter, num_gpus):
    """Assign each pool worker a distinct GPU via CUDA_VISIBLE_DEVICES."""
    with gpu_counter.get_lock():
        gpu_id = gpu_counter.value % num_gpus
        gpu_counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker PID={os.getpid()}] CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)


# ---------------------------------------------------------------------------
# Single-sample validation (runs in subprocess)
# ---------------------------------------------------------------------------

def _validate_one(args_tuple: Tuple) -> Dict[str, Any]:
    """Validate a single HTML file.  Designed to run in a multiprocessing.Pool worker.

    args_tuple: (spec_dict, html_path_str, artifact_root_str, backend)
    Returns a dict with sample_id, scores, overall_score, and per-validator breakdown.
    """
    spec_dict, html_path_str, artifact_root_str, backend = args_tuple
    html_path = Path(html_path_str)
    artifact_root = Path(artifact_root_str) if artifact_root_str else None

    try:
        from verifier.pipeline.parser import _parse_sample
        from verifier.pipeline.runner import validate_sample
        from verifier.pipeline.serialization import serialize_validation_report

        sample = _parse_sample(spec_dict)
        report = validate_sample(
            sample,
            html_path,
            artifact_backend=backend,
            artifact_root=artifact_root,
        )
        serialized = serialize_validation_report(report)

        # Extract per-validator scores from summary
        scores_summary = serialized.get("summary", {}).get("scores", {})
        per_validator = scores_summary.get("per_validator", {})
        overall_score = scores_summary.get("overall_score", 0.0)

        return {
            "sample_id": sample.id,
            "html_path": html_path_str,
            "status": "success",
            "overall_score": overall_score,
            "per_validator": per_validator,
            "execution_pass": report.execution_pass,
            "quality_pass": report.quality_pass,
            "full_report": serialized,
        }
    except Exception as e:
        sample_id = spec_dict.get("id", "unknown")
        return {
            "sample_id": sample_id,
            "html_path": html_path_str,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "overall_score": 0.0,
            "per_validator": {},
        }


# ---------------------------------------------------------------------------
# Pass@K computation
# ---------------------------------------------------------------------------

def _extract_k_index(result: Dict[str, Any]) -> int:
    """Extract k index from html_path (e.g. '...sample_id_k5.html' -> 5)."""
    import re
    html_path = result.get("html_path", "")
    m = re.search(r"_k(\d+)\.html$", html_path)
    return int(m.group(1)) if m else 0


def pass_at_k_scores(
    results_by_sample: Dict[str, List[Dict[str, Any]]],
    k_values: List[int],
    n_trials: int = 200,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute pass@K scores using ordered k indices.

    For each sample with N generated outputs, pass@K means:
      - Take the first K outputs (k0 ~ k_{K-1}), ordered by k index
      - Pick the one with the highest *overall_score* as the representative
      - Report that representative's overall_score and its per-validator scores
      - Average across all samples

    Returns: {f"pass@{k}": {"overall_score": mean, per_validator means...}} for each k.
    """
    all_k_scores: Dict[str, Dict[str, List[float]]] = {}

    for k in k_values:
        k_label = f"pass@{k}"
        all_k_scores[k_label] = defaultdict(list)

        for sample_id, results in results_by_sample.items():
            if not results:
                continue

            # Sort by k index and take first K
            sorted_results = sorted(results, key=_extract_k_index)
            first_k = sorted_results[:k]

            best_result = max(first_k, key=lambda r: r.get("overall_score", 0.0))
            all_k_scores[k_label]["overall_score"].append(best_result.get("overall_score", 0.0))
            for vname, vscore in best_result.get("per_validator", {}).items():
                all_k_scores[k_label][vname].append(vscore)

    # Average across samples
    output: Dict[str, Dict[str, float]] = {}
    for k_label, metric_lists in all_k_scores.items():
        output[k_label] = {}
        for metric, values in metric_lists.items():
            output[k_label][metric] = round(sum(values) / len(values), 4) if values else 0.0

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel validation for vLLM eval outputs")
    parser.add_argument("--spec-jsonl", type=str, required=True,
                        help="Evaluation spec JSONL")
    parser.add_argument("--html-dir", type=str, required=True,
                        help="Directory with generated HTML files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for validation results")
    parser.add_argument("--pattern", choices=["greedy", "sample"], default="greedy",
                        help="File naming pattern: greedy={id}.html, sample={id}_k{0..n-1}.html")
    parser.add_argument("--n", type=int, default=64,
                        help="Number of samples per spec (for sample pattern)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel validation workers")
    parser.add_argument("--backend", default="playwright",
                        help="Artifact backend (default: playwright)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max specs to validate (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples with existing validation results")
    parser.add_argument("--ref-image-dir", type=str, default="",
                        help="Override reference image dir for all samples")
    args = parser.parse_args()

    # Set env var so verifier's resolve_reference_image_path picks it up
    if args.ref_image_dir:
        import os
        os.environ["REFERENCE_IMAGE_DIR"] = args.ref_image_dir

    # Load specs
    specs: List[Dict[str, Any]] = []
    with open(args.spec_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                specs.append(json.loads(line))

    # Override reference image paths in specs
    if args.ref_image_dir:
        from pathlib import Path as _P
        ref_dir = _P(args.ref_image_dir)
        for s in specs:
            sid = str(s.get("id", ""))
            ref_path = ref_dir / f"{sid}.png"
            if not ref_path.exists():
                ref_path = ref_dir / f"{sid}.jpg"
            if ref_path.exists():
                meta = s.setdefault("reference_metadata", {})
                meta["reference_image_path"] = str(ref_path)

    if args.limit > 0:
        specs = specs[:args.limit]

    html_dir = Path(args.html_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Build work items
    work_items: List[Tuple] = []
    spec_index: Dict[str, Dict[str, Any]] = {}

    # Load existing results for resume
    report_path = output_dir / "validation_report.jsonl"
    existing_results: Dict[str, bool] = {}
    if args.resume and report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        key = row.get("html_path", row.get("sample_id", ""))
                        existing_results[key] = True
                    except json.JSONDecodeError:
                        pass

    for spec in specs:
        sample_id = str(spec.get("id", ""))
        spec_index[sample_id] = spec

        if args.pattern == "greedy":
            html_path = html_dir / f"{sample_id}.html"
            if not html_path.exists():
                continue
            if args.resume and str(html_path) in existing_results:
                continue
            work_items.append((spec, str(html_path), str(artifact_root), args.backend))

        elif args.pattern == "sample":
            for k in range(args.n):
                html_path = html_dir / f"{sample_id}_k{k}.html"
                if not html_path.exists():
                    continue
                if args.resume and str(html_path) in existing_results:
                    continue
                # For sampling, use separate artifact dirs per k
                k_artifact_root = artifact_root / f"{sample_id}_k{k}"
                work_items.append((spec, str(html_path), str(k_artifact_root), args.backend))

    if not work_items:
        print("Nothing to validate (all samples already processed or no HTML files found).")
        # Still compute pass@K if we have existing results
        if args.pattern == "sample" and report_path.exists():
            _compute_and_print_passk(report_path, output_dir)
        return

    print(f"Pattern:   {args.pattern}")
    print(f"Specs:     {len(specs)}")
    print(f"HTML dir:  {html_dir}")
    print(f"Work items: {len(work_items)}")
    print(f"Workers:   {args.workers}")
    print(f"Output:    {output_dir}")
    print(f"Starting at {time.strftime('%H:%M:%S')}")

    t0 = time.time()

    # Run validation in parallel
    import multiprocessing
    import subprocess
    # Count GPUs without initializing CUDA (which would be inherited by forked workers)
    try:
        _nv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        num_gpus = len(_nv.strip().splitlines())
    except Exception:
        num_gpus = 1
    print(f"GPU count: {num_gpus}")
    ctx = multiprocessing.get_context("fork")
    gpu_counter = ctx.Value("i", 0)
    results: List[Dict[str, Any]] = []

    # APPEND-PER-ITEM streaming write: open compact + full report files NOW
    # so each completed work_item is flushed to disk immediately. Protects
    # against SIGKILL / OOM / PaddleOCR deadlock in the pool — any partial
    # progress survives and validate --resume picks up cleanly from the
    # already-written sample_id × k_idx rows.
    compact_mode = "a" if args.resume and report_path.exists() else "w"
    full_report_path = output_dir / "validation_full_reports.jsonl"
    full_mode = "a" if args.resume and full_report_path.exists() else "w"
    compact_fp = open(report_path, compact_mode, encoding="utf-8")
    full_fp = open(full_report_path, full_mode, encoding="utf-8")

    def _stream_write(r: Dict[str, Any]) -> None:
        """Write one result's compact + full report rows and fsync."""
        compact = {
            "sample_id": r["sample_id"],
            "html_path": r.get("html_path", ""),
            "status": r["status"],
            "overall_score": r["overall_score"],
            "per_validator": r.get("per_validator", {}),
            "execution_pass": r.get("execution_pass", False),
            "quality_pass": r.get("quality_pass", False),
        }
        if r["status"] == "error":
            compact["error"] = r.get("error", "")
        compact_fp.write(json.dumps(compact, ensure_ascii=False) + "\n")
        compact_fp.flush()
        if "full_report" in r:
            full_fp.write(json.dumps(r["full_report"], ensure_ascii=False) + "\n")
            full_fp.flush()

    try:
        with ctx.Pool(
            processes=min(args.workers, len(work_items)),
            initializer=_worker_init,
            initargs=(gpu_counter, num_gpus),
            maxtasksperchild=int(os.environ.get("BENCH_MAXTASKSPERCHILD", "50")),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(_validate_one, work_items), 1):
                results.append(result)
                _stream_write(result)
                if i % 50 == 0 or i == len(work_items):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    ok = sum(1 for r in results if r["status"] == "success")
                    err = sum(1 for r in results if r["status"] == "error")
                    eta = (len(work_items) - i) / rate if rate > 0 else 0
                    mean_score = (
                        sum(r["overall_score"] for r in results if r["status"] == "success") / ok
                        if ok > 0 else 0.0
                    )
                    print(
                        f"  [{i}/{len(work_items)}] ok={ok} err={err} "
                        f"mean_score={mean_score:.4f} rate={rate:.1f}/s ETA={eta/60:.1f}min",
                        flush=True,
                    )
    finally:
        compact_fp.close()
        full_fp.close()

    elapsed = time.time() - t0

    # NOTE: jsonl writes already happened streaming inside the pool loop above
    # (append-per-item + flush). No need to re-write here — re-writing would
    # either double the rows (mode='a') or wipe the streaming progress
    # (mode='w'). We only do summary stats from the in-memory `results` now.

    # Summary stats
    ok_results = [r for r in results if r["status"] == "success"]
    err_results = [r for r in results if r["status"] == "error"]
    mean_overall = sum(r["overall_score"] for r in ok_results) / len(ok_results) if ok_results else 0.0

    print(f"\nDone in {elapsed/60:.1f}min.")
    print(f"  Success: {len(ok_results)}")
    print(f"  Errors:  {len(err_results)}")
    print(f"  Mean overall_score: {mean_overall:.4f}")

    # Per-validator mean scores
    validator_sums: Dict[str, List[float]] = defaultdict(list)
    for r in ok_results:
        for vname, vscore in r.get("per_validator", {}).items():
            validator_sums[vname].append(vscore)

    if validator_sums:
        print("\n  Per-validator mean scores:")
        for vname in sorted(validator_sums.keys()):
            vals = validator_sums[vname]
            mean_v = sum(vals) / len(vals) if vals else 0.0
            print(f"    {vname}: {mean_v:.4f} (n={len(vals)})")

    # Pass@K for sampling pattern
    if args.pattern == "sample":
        _compute_and_print_passk(report_path, output_dir)

    print(f"\nReport: {report_path}")


def _compute_and_print_passk(report_path: Path, output_dir: Path) -> None:
    """Load all results from report, group by sample_id, compute pass@K."""
    all_results: List[Dict[str, Any]] = []
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Group by sample_id (include error results with score 0 for fair pass@K)
    results_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        results_by_sample[r["sample_id"]].append(r)

    if not results_by_sample:
        print("\nNo successful results for pass@K computation.")
        return

    k_values = [1, 8, 16, 32, 64]
    # Filter to k_values that are meaningful (≤ max n per sample)
    max_n = max(len(v) for v in results_by_sample.values())
    k_values = [k for k in k_values if k <= max_n]

    passk = pass_at_k_scores(results_by_sample, k_values)

    print(f"\n{'='*60}")
    print(f"Pass@K Results ({len(results_by_sample)} samples)")
    print(f"{'='*60}")

    # Header
    validators = set()
    for k_data in passk.values():
        validators.update(k for k in k_data.keys() if k != "overall_score")
    validators = sorted(validators)

    header = f"{'K':>6} | {'overall':>8}"
    for v in validators:
        short_name = v.replace("Validator", "")
        header += f" | {short_name:>10}"
    print(header)
    print("-" * len(header))

    for k_label in sorted(passk.keys(), key=lambda x: int(x.split("@")[1])):
        k_data = passk[k_label]
        k_val = k_label.split("@")[1]
        row = f"{k_val:>6} | {k_data.get('overall_score', 0):.4f}"
        for v in validators:
            row += f" | {k_data.get(v, 0):.4f}"
        print(row)

    # Save pass@K results
    passk_path = output_dir / "pass_at_k.json"
    passk_path.write_text(json.dumps(passk, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPass@K saved: {passk_path}")


if __name__ == "__main__":
    main()
