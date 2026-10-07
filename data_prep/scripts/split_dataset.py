"""Split the scored spec dataset into SFT / RL / Eval subsets.

Reads verify scores JSONL (from verify_train_split.py) + original spec JSONL,
produces three output JSONL files in the same spec format.

Usage:
    python data_prep/scripts/split_dataset.py \
        --spec-jsonl ${AESCODE_OUTPUT_ROOT}/spec_train15k/final/spec_train15k.jsonl \
        --scores-jsonl ${AESCODE_OUTPUT_ROOT}/spec_train15k/final/verifier_scores.jsonl \
        --output-dir ${AESCODE_OUTPUT_ROOT}/spec_train15k/final/ \
        --sft-count 15000 --eval-count 500
"""
import _bootstrap  # noqa: F401

import argparse
import json
import logging
import random
import statistics
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _bin_index(score: float, n_bins: int = 10) -> int:
    """Map a [0, 1] score to a bin index."""
    idx = int(score * n_bins)
    return min(idx, n_bins - 1)


def main():
    parser = argparse.ArgumentParser(description="Split dataset by Verifier scores into SFT/RL/Eval.")
    parser.add_argument("--spec-jsonl", required=True, help="Original spec JSONL (19k)")
    parser.add_argument("--scores-jsonl", required=True, nargs="+",
                        help="Verify scores JSONL(s) — can pass multiple shard files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--sft-count", type=int, default=15000, help="Target SFT subset size")
    parser.add_argument("--eval-count", type=int, default=500, help="Target Eval subset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-score-sft", type=float, default=None,
                        help="Minimum score threshold for SFT (auto-computed if not set)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scores from one or more JSONL files
    logger.info("Loading scores from %d file(s)...", len(args.scores_jsonl))
    scores: dict[str, dict] = {}
    for scores_path in args.scores_jsonl:
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = obj["id"]
                # Keep the latest if duplicate
                scores[sid] = obj
    logger.info("Loaded scores for %d samples", len(scores))

    # Load spec JSONL
    logger.info("Loading spec JSONL: %s", args.spec_jsonl)
    spec_by_id: dict[str, dict] = {}
    spec_order: list[str] = []
    with open(args.spec_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj["id"]
            spec_by_id[sid] = obj
            spec_order.append(sid)
    logger.info("Loaded %d specs", len(spec_by_id))

    # Intersect: only samples with both spec and score
    valid_ids = [sid for sid in spec_order if sid in scores]
    logger.info("Samples with both spec and scores: %d", len(valid_ids))

    # Attach score to each sample and sort
    scored = [(sid, scores[sid].get("overall_score", 0.0)) for sid in valid_ids]

    # --- Step 1: Eval — stratified sampling across score bins ---
    n_bins = 10
    bins: dict[int, list[str]] = defaultdict(list)
    for sid, score in scored:
        bins[_bin_index(score, n_bins)].append(sid)

    eval_ids: set[str] = set()
    per_bin_target = args.eval_count // n_bins  # 50 per bin by default

    # First pass: sample up to per_bin_target from each bin
    for bin_idx in range(n_bins):
        pool = bins[bin_idx]
        rng.shuffle(pool)
        take = min(per_bin_target, len(pool))
        eval_ids.update(pool[:take])

    # Second pass: if we haven't reached eval_count, sample more from larger bins
    remaining_needed = args.eval_count - len(eval_ids)
    if remaining_needed > 0:
        leftover = [sid for sid, _ in scored if sid not in eval_ids]
        rng.shuffle(leftover)
        eval_ids.update(leftover[:remaining_needed])

    logger.info("Eval set: %d samples", len(eval_ids))

    # --- Step 2: SFT + RL from remainder, sorted by score descending ---
    remainder = [(sid, sc) for sid, sc in scored if sid not in eval_ids]
    remainder.sort(key=lambda x: x[1], reverse=True)

    sft_count = min(args.sft_count, len(remainder))
    sft_ids = set(sid for sid, _ in remainder[:sft_count])
    rl_ids = set(sid for sid, _ in remainder[sft_count:])

    logger.info("SFT set: %d samples", len(sft_ids))
    logger.info("RL set: %d samples", len(rl_ids))

    # --- Write output JSONL files ---
    splits = {
        "split_sft.jsonl": sft_ids,
        "split_rl.jsonl": rl_ids,
        "split_eval.jsonl": eval_ids,
    }

    for filename, id_set in splits.items():
        out_path = output_dir / filename
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for sid in spec_order:
                if sid in id_set:
                    # Write original spec + attach score metadata
                    spec = spec_by_id[sid]
                    spec["_verify_score"] = scores.get(sid, {}).get("overall_score", 0.0)
                    spec["_verify_per_validator"] = scores.get(sid, {}).get("per_validator", {})
                    f.write(json.dumps(spec, ensure_ascii=False) + "\n")
                    count += 1
        logger.info("Wrote %s: %d samples", out_path, count)

    # --- Stats ---
    def _stats(id_set: set[str]) -> dict:
        sc = [scores.get(sid, {}).get("overall_score", 0.0) for sid in id_set]
        if not sc:
            return {"count": 0}
        return {
            "count": len(sc),
            "mean": round(statistics.mean(sc), 4),
            "median": round(statistics.median(sc), 4),
            "stdev": round(statistics.stdev(sc), 4) if len(sc) > 1 else 0.0,
            "min": round(min(sc), 4),
            "max": round(max(sc), 4),
            "p10": round(sorted(sc)[len(sc) // 10], 4),
            "p90": round(sorted(sc)[9 * len(sc) // 10], 4),
        }

    # Score distribution per bin
    all_scores = [scores.get(sid, {}).get("overall_score", 0.0) for sid in valid_ids]
    bin_dist = defaultdict(int)
    for sc in all_scores:
        bin_dist[f"{_bin_index(sc, n_bins) / n_bins:.1f}-{(_bin_index(sc, n_bins) + 1) / n_bins:.1f}"] += 1

    stats = {
        "total_scored": len(valid_ids),
        "sft": _stats(sft_ids),
        "rl": _stats(rl_ids),
        "eval": _stats(eval_ids),
        "score_distribution_bins": dict(sorted(bin_dist.items())),
        "sft_score_threshold": round(remainder[sft_count - 1][1], 4) if sft_count > 0 and sft_count <= len(remainder) else None,
    }

    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Stats written to %s", stats_path)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    for name in ["sft", "rl", "eval"]:
        s = stats[name]
        print(f"  {name.upper():>5}: {s['count']:>6} samples | "
              f"score: {s['mean']:.3f} ± {s['stdev']:.3f} "
              f"[{s['min']:.3f}, {s['max']:.3f}] "
              f"(p10={s['p10']:.3f}, p90={s['p90']:.3f})")
    if stats["sft_score_threshold"] is not None:
        print(f"\n  SFT score threshold: >= {stats['sft_score_threshold']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
