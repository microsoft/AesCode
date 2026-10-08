"""Convert AesCode JSONL dataset to verl parquet format.

verl expects parquet files with specific columns:
  - data_source: str — identifies which reward function to dispatch
  - prompt: list[dict] — chat messages (system + user)
  - ability: str — task category
  - reward_model: dict — {"style": "rule", "ground_truth": ""}
  - extra_info: dict — metadata for reward function (sample_id, requirements, etc.)
  - images: list[str] — image paths for VLM multimodal input

Usage:
    # Recommended: separate train/val JSONL files (avoids data contamination)
    # Set SPEC_DIR (default ${AESCODE_OUTPUT_ROOT}/spec_train15k/final) to override the prefix.
    python -m rl_verl.data_prep \
        --input ${SPEC_DIR}/split_rl.jsonl \
        --val-input ${SPEC_DIR}/split_eval.jsonl \
        --val-samples 100 \
        --output-dir dataset/ \
        --image-dir ${SPEC_DIR}/images

    # Legacy: single JSONL with random split (NOT recommended — risks eval contamination)
    python -m rl_verl.data_prep \
        --input dataset/sft_train.jsonl \
        --output-dir dataset/ \
        --val-ratio 0.15 \
        --image-dir data/testset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    no_image_mode,
    DEFAULT_USER_TEMPLATE,
    GENERATE_OUTPUT_INSTRUCTION,
    OUTPUT_SECTION_MARKER,
    build_user_text,
)

logger = logging.getLogger(__name__)

DATA_SOURCE = "aescode"

# Match SFT training: IMAGE_MAX_PIXELS = 1280 * 28 * 28 = 1,003,520
# (see sft/train_sft.sh — defaults to
# 1280*28*28 for the reference image). This controls image token count in
# Qwen3-VL: tokens ≈ pixels / (patch_size * merge_size)^2 = 1003520 / 1024 ≈ 980.
IMAGE_MIN_PIXELS = 4 * 28 * 28       # 3,136     (Qwen3-VL default minimum)
IMAGE_MAX_PIXELS = 1280 * 28 * 28    # 1,003,520 (matches SFT training)


def _build_chat_prompt(sample: dict, image_path: str | None) -> list[dict]:
    """Build chat-format prompt matching the inference pipeline.

    Returns a list of message dicts: [system, user].
    The user message text is constructed identically to the GDPO pipeline
    via build_user_text() from inference.prompts, with the [Output] section
    explicitly set to GENERATE_OUTPUT_INSTRUCTION for train/inference consistency.
    """
    has_image = image_path is not None and os.path.exists(image_path)

    user_text = build_user_text(
        sample,
        DEFAULT_USER_TEMPLATE,
        has_image=has_image,
    )

    # Replace [Output] with the canonical generate instruction (COT + fence),
    # matching what SFT trained the model to produce.
    idx = user_text.find(OUTPUT_SECTION_MARKER)
    if idx >= 0:
        user_text = user_text[:idx]
    user_text += GENERATE_OUTPUT_INSTRUCTION

    # For verl's multimodal pipeline, we embed the image placeholder
    # into the user content. verl's data loader replaces <image> tokens
    # with actual pixel values using the processor.
    #
    # Qwen-VL models expect: <|vision_start|><|image_pad|><|vision_end|>
    # The processor handles this when images are provided via data.image_key.
    #
    # We prepend a simple image reference to the user text so the model
    # knows an image is attached. The actual image is loaded separately
    # by verl's multimodal data pipeline from the 'images' column.
    if has_image and not no_image_mode():
        user_content = f"<image>\n{user_text}"
    else:
        user_content = user_text

    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _resolve_image_path(
    sample: dict,
    image_dir: Path | None,
) -> str | None:
    """Resolve the absolute image path for a sample.

    When --image-dir is provided, always prefer looking up the image there
    (by sample_id or by the basename from reference_metadata). This ensures
    all paths point to the canonical image directory and avoids stale paths
    from the original JSONL (e.g. data/testset/ or /scratch/ paths).
    """
    sample_id = sample.get("id", "")

    # Priority 1: Look in image_dir by sample_id (canonical location)
    if image_dir and sample_id:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = image_dir / f"{sample_id}{ext}"
            if candidate.exists():
                return str(candidate)

    # Priority 2: Look in image_dir by basename from reference_metadata
    ref_meta = sample.get("reference_metadata") or {}
    declared = ref_meta.get("reference_image_path")
    if image_dir and declared:
        basename = Path(declared).name
        candidate = image_dir / basename
        if candidate.exists():
            return str(candidate)

    # Priority 3: target_image field (absolute path from SFT JSONL)
    target_image = sample.get("target_image", "")
    if target_image and os.path.exists(target_image):
        return str(target_image)

    # Priority 4: reference_metadata.reference_image_path as-is
    if declared:
        p = Path(declared)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            return str(p)

    return None


def convert_sample(sample: dict, image_dir: Path | None) -> dict:
    """Convert a single JSONL sample to verl parquet row format."""
    sample_id = sample.get("id", "")
    image_path = _resolve_image_path(sample, image_dir)

    # Build chat prompt (identical to GDPO pipeline)
    prompt = _build_chat_prompt(sample, image_path)

    # Extra info carries everything the reward function needs.
    # Fix reference_image_path: use the resolved image_path (from --image-dir)
    # so the verifier can find the reference image for visual comparison.
    ref_meta = dict(sample.get("reference_metadata", {}))
    if image_path:
        ref_meta["reference_image_path"] = image_path

    verifiable_requirements = sample.get("verifiable_requirements", [])

    extra_info = {
        "sample_id": sample_id,
        "generation_prompt": sample.get("generation_prompt", ""),
        "verifiable_requirements": verifiable_requirements,
        "reference_metadata": ref_meta,
        "difficulty": sample.get("difficulty", 0),
        # interaction_kwargs: verl strips non-standard extra_info keys on some
        # paths but forwards this dict verbatim, so per-sample context rides here.
        "interaction_kwargs": {
            "name": "aescode",  # must match the name in your verl interaction config
            "sample_id": sample_id,
            "verifiable_requirements": verifiable_requirements,
            "reference_metadata": ref_meta,
            "generation_prompt": sample.get("generation_prompt", ""),
        },
    }

    # rubric_questions: optional, attached to a sample when rubric data
    # (JSONL or per-sample JSON directory) is merged in by --rubric-input.
    # Consumed by RubricDesignScorer via
    # extra_info["rubric_questions"]. If absent, rubric scorer marks the
    # sample applicable=0 and skips it.
    if sample.get("rubric_questions"):
        extra_info["rubric_questions"] = sample["rubric_questions"]
        # Also embed inside interaction_kwargs: verl strips non-standard
        # top-level extra_info keys before agent.run(), but forwards
        # interaction_kwargs verbatim. RubricDesignScorer reads top-level then
        # falls back to ikw; embedding here guarantees the judge sees it
        # (top-level alone => applicable=0 => design collapses to 0.0).
        extra_info["interaction_kwargs"]["rubric_questions"] = sample["rubric_questions"]

    row = {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "html_generation",
        "reward_model": {
            "style": "rule",
            "ground_truth": "",  # Not used; our reward uses verifier pipeline
        },
        "extra_info": extra_info,
    }

    # Images for VLM multimodal input
    # verl expects image paths in a list under data.image_key
    # Track whether a reference image actually resolved (reward needs it),
    # independent of whether it is fed to the model input.
    row["_has_ref_image"] = bool(image_path)
    if image_path and not no_image_mode():
        row["images"] = [image_path]
    else:
        row["images"] = []

    return row


def _load_jsonl(path: str) -> list[dict]:
    """Load JSONL file, skipping malformed lines."""
    samples = []
    parse_errors = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors += 1
                logger.warning("Skipping malformed JSON on line %d in %s: %s", line_num, path, e)
    if parse_errors > 0:
        logger.warning("%d/%d lines had JSON errors in %s", parse_errors, parse_errors + len(samples), path)
    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


def _load_rubric_index(path: str) -> dict[str, list]:
    """Load rubric index from JSONL file OR rubric directory.

    Supported inputs:
      1. JSONL rows: {"id": "...", "rubric_questions": [...]} (legacy)
      2. JSONL rows: {"id": "...", "rubric": [...]} (v5.2 compatible)
      3. Directory of JSON files: <sid>.json with either
         {"rubric": [...]} or {"rubric_questions": [...]}.

    Returns:
      {sample_id -> normalized rubric questions}
    where each question keeps at least {id, question, pass_answer}.
    """

    def _normalize_questions(raw: object) -> list[dict]:
        out_q: list[dict] = []
        if not isinstance(raw, list):
            return out_q
        for i, q in enumerate(raw, start=1):
            if not isinstance(q, dict):
                continue
            qtext = str(q.get("question") or "").strip()
            if not qtext:
                continue
            qid = str(q.get("id") or f"q{i}").strip()
            pa = str(q.get("pass_answer", q.get("expected", "yes"))).strip().lower()
            if pa not in {"yes", "no"}:
                pa = "yes"
            out_q.append({
                "id": qid,
                "question": qtext,
                "pass_answer": pa,
            })
        return out_q

    p = Path(path)
    out: dict[str, list] = {}
    missing = 0

    if p.is_dir():
        files = sorted(p.glob("*.json"))
        for fp in files:
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Skipping malformed rubric file %s: %s", fp, e)
                missing += 1
                continue
            sid = str(obj.get("id") or fp.stem).strip()
            rq = _normalize_questions(obj.get("rubric") or obj.get("rubric_questions") or [])
            if not sid or not rq:
                missing += 1
                continue
            out[sid] = rq
        logger.info(
            "Indexed rubric dir: %d samples from %s (skipped %d empty/malformed)",
            len(out), path, missing,
        )
        return out

    samples = _load_jsonl(path)
    for s in samples:
        sid = s.get("id")
        rq = _normalize_questions(s.get("rubric_questions") or s.get("rubric") or [])
        if not sid or not rq:
            missing += 1
            continue
        out[str(sid)] = rq
    logger.info(
        "Indexed rubric: %d samples with rubric_questions/rubric (skipped %d empty/no-id)",
        len(out), missing,
    )
    return out


def _attach_rubrics(samples: list[dict], rubric_idx: dict[str, list], split_name: str) -> int:
    """Mutate samples to add rubric_questions when present in rubric_idx.

    Returns the count of samples that received a rubric. Logs a warning if
    the coverage is partial so the operator notices unaligned rubric data.
    """
    attached = 0
    for s in samples:
        sid = s.get("id")
        rq = rubric_idx.get(sid) if sid else None
        if rq:
            s["rubric_questions"] = rq
            attached += 1
    if samples:
        coverage = attached / len(samples)
        if coverage < 1.0:
            logger.warning(
                "[%s] rubric coverage %d/%d (%.1f%%) — some samples have no rubric and "
                "will be marked applicable=0 by RubricDesignScorer (excluded from design reward).",
                split_name, attached, len(samples), 100 * coverage,
            )
        else:
            logger.info("[%s] rubric coverage %d/%d (100%%)", split_name, attached, len(samples))
    return attached


def _convert_samples(samples: list[dict], image_dir_path: Path | None) -> list[dict]:
    """Convert JSONL samples to verl rows, skipping those without images."""
    rows = []
    skipped_no_image = []
    for sample in samples:
        row = convert_sample(sample, image_dir_path)
        has_ref = row.pop("_has_ref_image", bool(row["images"]))
        if not has_ref:
            skipped_no_image.append(sample.get("id", "unknown"))
            continue
        rows.append(row)
    if skipped_no_image:
        logger.warning(
            "Skipped %d/%d samples with no reference image: %s",
            len(skipped_no_image), len(skipped_no_image) + len(rows), skipped_no_image[:20],
        )
    return rows


def prepare_dataset(
    input_path: str,
    output_dir: str,
    val_ratio: float = 0.0,
    image_dir: str | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    val_input_path: str | None = None,
    val_samples: int | None = None,
    rubric_input_path: str | None = None,
    val_rubric_input_path: str | None = None,
) -> tuple[str, str]:
    """Convert JSONL dataset to verl parquet train/val split.

    Two modes:
      1. Separate train/val JSONL (recommended): --input + --val-input
         Uses all samples from --input for training. Randomly samples
         --val-samples from --val-input for validation (default: all).
         This avoids eval data contamination in training.

      2. Single JSONL with random split (legacy): --input + --val-ratio
         Randomly splits the input into train/val. NOT recommended because
         eval samples may leak into training data.

    Args:
        input_path: Path to training JSONL file
        output_dir: Directory to write parquet files
        val_ratio: Fraction for validation (only used when val_input_path is None)
        image_dir: Directory to search for images by sample_id
        seed: Random seed for split / sampling
        max_samples: If set, only process the first N train samples (debugging)
        val_input_path: Separate JSONL file for validation (recommended)
        val_samples: Number of val samples to randomly draw (None = use all)

    Returns:
        Tuple of (train_parquet_path, val_parquet_path)
    """
    try:
        import datasets
        import pandas as pd
    except ImportError:
        raise ImportError(
            "datasets and pandas are required for data_prep. "
            "Install with: pip install datasets pandas"
        )

    image_dir_path = Path(image_dir) if image_dir else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # ── Load & convert training data ──
    train_samples = _load_jsonl(input_path)
    if max_samples is not None and max_samples > 0:
        train_samples = train_samples[:max_samples]
        logger.info("Truncated train to first %d samples (--max-samples)", max_samples)

    # Attach rubric_questions BEFORE convert_sample, so they get folded
    # into extra_info via the convert_sample path.
    if rubric_input_path:
        train_rubric_idx = _load_rubric_index(rubric_input_path)
        _attach_rubrics(train_samples, train_rubric_idx, split_name="train")

    train_rows = _convert_samples(train_samples, image_dir_path)

    # ── Load & convert validation data ──
    if val_input_path:
        # Mode 1: Separate val JSONL (recommended)
        val_all_samples = _load_jsonl(val_input_path)

        # Val rubric: prefer a dedicated --val-rubric-input; otherwise reuse
        # --rubric-input (which may contain both train + val ids).
        val_rubric_src = val_rubric_input_path or rubric_input_path
        if val_rubric_src:
            val_rubric_idx = _load_rubric_index(val_rubric_src)
            _attach_rubrics(val_all_samples, val_rubric_idx, split_name="val")

        val_all_rows = _convert_samples(val_all_samples, image_dir_path)

        if val_samples is not None and val_samples < len(val_all_rows):
            rng.shuffle(val_all_rows)
            val_rows = val_all_rows[:val_samples]
            logger.info("Sampled %d/%d val samples from %s", val_samples, len(val_all_rows), val_input_path)
        else:
            val_rows = val_all_rows

        # Sanity check: no overlap between train and val sample_ids
        def _get_sid(row):
            ei = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
            return ei.get("interaction_kwargs", {}).get("sample_id", "")
        train_ids = {_get_sid(r) for r in train_rows}
        val_ids = {_get_sid(r) for r in val_rows}
        overlap = train_ids & val_ids
        if overlap:
            logger.error(
                "DATA CONTAMINATION: %d eval samples found in training set! "
                "Examples: %s. Fix your data splits.",
                len(overlap), sorted(overlap)[:10],
            )
            raise ValueError(f"Data contamination: {len(overlap)} eval samples in training set")
    elif val_ratio > 0:
        # Mode 2: Random split from single JSONL (legacy)
        logger.warning(
            "Using random val split from single JSONL. This risks eval contamination. "
            "Prefer --val-input for clean train/val separation."
        )
        all_rows = train_rows
        indices = list(range(len(all_rows)))
        rng.shuffle(indices)
        val_size = max(1, int(len(all_rows) * val_ratio))
        val_indices = set(indices[:val_size])
        train_rows = [all_rows[i] for i in range(len(all_rows)) if i not in val_indices]
        val_rows = [all_rows[i] for i in range(len(all_rows)) if i in val_indices]
    else:
        # Mode 3: No validation requested — use ALL of --input for training.
        # verl still requires val_files to point at a non-empty parquet (even
        # with test_freq=-1, the dataloader is constructed at startup), so we
        # reuse the first training row as a 1-sample placeholder. Total
        # training data: 100% of split_rl.jsonl, no contamination risk.
        logger.info(
            "val_ratio=0 and no --val-input: using all %d samples for training; "
            "writing 1-row placeholder val parquet (verl requires it to exist).",
            len(train_rows),
        )
        val_rows = train_rows[:1] if train_rows else []

    logger.info("Final: %d train, %d val", len(train_rows), len(val_rows))

    # Convert to HuggingFace datasets and save as parquet
    # verl expects:
    #   - prompt: list[dict] — NOT a JSON string, must be native list of message dicts
    #   - images: list[dict] — each image as {"image": "file:///abs/path"} for qwen_vl_utils.fetch_image
    #   - extra_info: JSON string — pyarrow can't handle varying-schema nested dicts;
    #     we patch verl's __getitem__ to deserialize it (see below)
    #   - reward_model: JSON string is OK (only used by NaiveRewardManager which handles strings)
    for row_list in (train_rows, val_rows):
        for row in row_list:
            # extra_info and reward_model: serialize as JSON strings for pyarrow compatibility
            if isinstance(row.get("extra_info"), dict):
                row["extra_info"] = json.dumps(row["extra_info"], ensure_ascii=False)
            if isinstance(row.get("reward_model"), dict):
                row["reward_model"] = json.dumps(row["reward_model"], ensure_ascii=False)
            # prompt: keep as native list[dict] — verl iterates over messages directly

            # images: convert path strings to qwen_vl_utils-compatible dicts
            # verl's _build_messages passes each image to process_image() which calls
            # fetch_image(). fetch_image accepts {"image": "file:///path"} format.
            # We add min_pixels/max_pixels to match SFT training resolution.
            raw_images = row.get("images", [])
            converted_images = []
            for img in raw_images:
                if isinstance(img, str) and img:
                    # Convert absolute path to file:// URI dict for qwen_vl_utils
                    uri = f"file://{img}" if img.startswith("/") else img
                    converted_images.append({
                        "image": uri,
                        "min_pixels": IMAGE_MIN_PIXELS,
                        "max_pixels": IMAGE_MAX_PIXELS,
                    })
                elif isinstance(img, dict):
                    # Ensure pixel limits are set even for pre-existing dicts
                    img.setdefault("min_pixels", IMAGE_MIN_PIXELS)
                    img.setdefault("max_pixels", IMAGE_MAX_PIXELS)
                    converted_images.append(img)
            row["images"] = converted_images if converted_images else []

    train_dataset = datasets.Dataset.from_list(train_rows)
    val_dataset = datasets.Dataset.from_list(val_rows)

    train_path = str(output_path / "verl_train.parquet")
    val_path = str(output_path / "verl_val.parquet")

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    logger.info("Saved train parquet: %s (%d rows)", train_path, len(train_rows))
    logger.info("Saved val parquet: %s (%d rows)", val_path, len(val_rows))

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert AesCode JSONL dataset to verl parquet format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to training JSONL file (e.g. ${SPEC_DIR}/split_rl.jsonl)"
    )
    parser.add_argument(
        "--val-input", default=None,
        help="Separate JSONL file for validation (recommended). "
             "If not set, falls back to random split from --input (legacy, not recommended)."
    )
    parser.add_argument(
        "--val-samples", type=int, default=None,
        help="Number of val samples to randomly draw from --val-input (default: use all)"
    )
    parser.add_argument(
        "--output-dir", default="dataset/",
        help="Directory to write verl_train.parquet and verl_val.parquet"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.0,
        help="Fraction of data for validation split (legacy mode; only used when --val-input is not set). "
             "Default 0 = use ALL of --input for training (no split). Set >0 to enable random split."
    )
    parser.add_argument(
        "--image-dir", default=None,
        help="Directory to search for images by sample_id (fallback if target_image not found)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="If set, only process the first N samples (useful for debugging)"
    )
    parser.add_argument(
        "--rubric-input", default=None,
        help="Path to rubric data. Supports JSONL ({id, rubric_questions|rubric, ...}) "
             "or a directory of per-sample JSON files (<sid>.json with rubric|rubric_questions). "
             "Matched rubrics are attached to extra_info for RubricDesignScorer."
    )
    parser.add_argument(
        "--val-rubric-input", default=None,
        help="Optional dedicated rubric JSONL for the val split. If omitted, "
             "--rubric-input is reused for val."
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    train_path, val_path = prepare_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        image_dir=args.image_dir,
        seed=args.seed,
        max_samples=args.max_samples,
        val_input_path=args.val_input,
        val_samples=args.val_samples,
        rubric_input_path=args.rubric_input,
        val_rubric_input_path=args.val_rubric_input,
    )

    print(f"\nDone! Created:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"\nTo use with verl:")
    print(f"  data.train_files={train_path}")
    print(f"  data.val_files={val_path}")


if __name__ == "__main__":
    main()
