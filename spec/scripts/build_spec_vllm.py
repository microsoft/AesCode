#!/usr/bin/env python3
"""Generate specs using a locally-deployed vLLM model (OpenAI-compatible API).

Reuses the schema and parsing logic from spec/ and routes requests through
one or more vLLM endpoints, with an enhanced system prompt that produces more
detailed generation_prompt output (2000+ chars).

Usage
-----
    # Single endpoint
    python spec/scripts/build_spec_vllm.py \
        --input-jsonl data/spec_train_input.jsonl \
        --output-jsonl data/spec_vllm.jsonl \
        --image-dir ${AESCODE_OUTPUT_ROOT}/processed_data_v4 \
        --api-base http://localhost:8000/v1

    # Multiple endpoints (load balanced)
    python spec/scripts/build_spec_vllm.py \
        --input-jsonl data/spec_train_input.jsonl \
        --output-jsonl data/spec_vllm.jsonl \
        --image-dir ${AESCODE_OUTPUT_ROOT}/processed_data_v4 \
        --api-base http://node0:8000/v1,http://node0:8001/v1 \
        --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401

from spec.io import (
    get_reference_aspect_ratio,
    get_reference_orientation,
    image_to_data_url,
    load_existing_ids,
    load_jsonl,
    resolve_image_path,
)
from spec.prompting import (
    MIN_PROMPT_LENGTH,
    MIN_REQUIREMENTS,
    build_detailed_spec_prompts,
    parse_spec_response,
)
from spec.schema import build_output_schema, build_schema_prompt_contract


# ---------------------------------------------------------------------------
# Client pool
# ---------------------------------------------------------------------------

class ClientPool:
    """Thread-safe round-robin pool of OpenAI clients for load balancing."""

    def __init__(self, base_urls: List[str], api_key: str = "EMPTY"):
        from openai import OpenAI

        self._clients = [
            OpenAI(api_key=api_key, base_url=url) for url in base_urls
        ]
        self._cycle = itertools.cycle(range(len(self._clients)))
        self._lock = threading.Lock()

    def next(self) -> Any:
        with self._lock:
            idx = next(self._cycle)
        return self._clients[idx]


# ---------------------------------------------------------------------------
# API call via vLLM
# ---------------------------------------------------------------------------

def call_spec_api_vllm(
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    image_data_url: str,
    max_tokens: int,
) -> str:
    """Call vLLM OpenAI-compatible API with vision input."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_one_vllm(
    idx: int,
    item: Dict,
    root_dir: Path,
    image_dir: Path,
    client_pool: ClientPool,
    model: str,
    system_prompt: str,
    user_template: str,
    output_schema: Dict[str, Any],
    schema_contract: str,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
    image_max_long_side: int = 1120,
) -> Dict:
    """Process one sample: build prompt, call vLLM, parse response."""
    image_path = resolve_image_path(item, root_dir=root_dir, image_dir=image_dir)
    image_data_url = image_to_data_url(image_path, max_long_side=image_max_long_side)

    reference_image_wh = item.get("reference_image_wh", "")
    reference_aspect_ratio = get_reference_aspect_ratio(reference_image_wh)
    reference_orientation = get_reference_orientation(reference_image_wh)
    user_text = user_template.format(
        id=item.get("id", ""),
        reference_image_wh=reference_image_wh,
        reference_aspect_ratio=reference_aspect_ratio,
        reference_orientation=reference_orientation,
        schema_contract=schema_contract,
    )

    last_err = None
    last_raw = ""
    # Total attempts = retries (normal) + 1 (quality retry if prompt too short)
    max_attempts = retries + 1
    quality_retried = False

    for attempt in range(1, max_attempts + 1):
        try:
            client = client_pool.next()
            raw = call_spec_api_vllm(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                image_data_url=image_data_url,
                max_tokens=max_tokens,
            )
            last_raw = raw or ""
            if not raw:
                raise RuntimeError("Empty response from model.")
            result = parse_spec_response(
                raw,
                item=item,
                reference_aspect_ratio=reference_aspect_ratio,
                reference_orientation=reference_orientation,
                schema=output_schema,
            )

            # ── Quality gate: check prompt length and requirement count ──
            prompt_len = len(result.get("generation_prompt", ""))
            req_count = len(result.get("verifiable_requirements", []))

            if prompt_len < MIN_PROMPT_LENGTH and not quality_retried:
                quality_retried = True
                print(
                    f"  [quality] {item.get('id', '?')}: prompt too short "
                    f"({prompt_len} < {MIN_PROMPT_LENGTH}), retrying...",
                    flush=True,
                )
                time.sleep(retry_sleep)
                continue

            if req_count < MIN_REQUIREMENTS and not quality_retried:
                quality_retried = True
                print(
                    f"  [quality] {item.get('id', '?')}: too few requirements "
                    f"({req_count} < {MIN_REQUIREMENTS}), retrying...",
                    flush=True,
                )
                time.sleep(retry_sleep)
                continue

            # Tag the result with quality metadata
            result["_spec_model"] = model
            result["_prompt_len"] = prompt_len
            result["_req_count"] = req_count
            result["_quality_retried"] = quality_retried
            return result

        except Exception as err:
            last_err = err
            if attempt < max_attempts:
                time.sleep(retry_sleep)

    failed = dict(item)
    failed["generation_prompt"] = ""
    failed["spec_error"] = str(last_err)
    failed["spec_model"] = model
    if last_raw:
        failed["spec_raw_response"] = last_raw[:3000]
    return failed


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate specs using a vLLM-deployed model.",
    )
    parser.add_argument("--input-jsonl", type=Path, required=True,
                        help="Input JSONL with image references")
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/spec_vllm.jsonl"),
                        help="Output spec JSONL")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Directory containing reference images")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="Comma-separated vLLM API base URLs")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key (default: EMPTY)")
    parser.add_argument("--model", default="${AESCODE_OUTPUT_ROOT}/models/Qwen3.5-122B-A10B-AWQ-4bit",
                        help="Model name as registered in vLLM")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=4000,
                        help="Max completion tokens (default: 4000)")
    parser.add_argument("--image-max-long-side", type=int, default=1120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--failed-log", type=Path, default=None,
                        help="Path for logging failed samples")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root_dir = Path.cwd()
    input_path = (root_dir / args.input_jsonl).resolve()
    output_path = (root_dir / args.output_jsonl).resolve()
    image_dir = (root_dir / args.image_dir).resolve()

    system_prompt, user_template = build_detailed_spec_prompts()
    output_schema = build_output_schema()
    schema_contract = build_schema_prompt_contract(output_schema)

    if "{schema_contract}" not in user_template:
        raise ValueError("User template must contain placeholder: {schema_contract}")

    # Build client pool
    base_urls = [u.strip() for u in args.api_base.split(",") if u.strip()]
    if not base_urls:
        print("ERROR: No valid API base URLs provided.", file=sys.stderr)
        sys.exit(1)
    client_pool = ClientPool(base_urls, api_key=args.api_key)
    print(f"Client pool: {len(base_urls)} endpoint(s)")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_completion_tokens}")
    print(f"Quality gates: min_prompt_len={MIN_PROMPT_LENGTH}, min_reqs={MIN_REQUIREMENTS}")

    # Load input
    rows = load_jsonl(input_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    existing_ids = load_existing_ids(output_path) if args.resume else set()
    pending = [r for r in rows if r.get("id") not in existing_ids]

    if not pending:
        print("No pending items.")
        return

    print(f"Total: {len(rows)}, Pending: {len(pending)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output_path.exists() else "w"

    total = len(pending)
    done = 0
    success = 0
    failed = 0
    prompt_lengths: list[int] = []
    req_counts: list[int] = []
    quality_retried_count = 0
    t0 = time.time()

    write_lock = threading.Lock()

    with output_path.open(mode, encoding="utf-8") as out_f:
        failed_f = None
        if args.failed_log:
            failed_log_path = (root_dir / args.failed_log).resolve()
            failed_log_path.parent.mkdir(parents=True, exist_ok=True)
            failed_mode = "a" if args.resume and failed_log_path.exists() else "w"
            failed_f = failed_log_path.open(failed_mode, encoding="utf-8")

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    process_one_vllm,
                    idx,
                    item,
                    root_dir,
                    image_dir,
                    client_pool,
                    args.model,
                    system_prompt,
                    user_template,
                    output_schema,
                    schema_contract,
                    args.max_completion_tokens,
                    args.retries,
                    args.retry_sleep,
                    args.image_max_long_side,
                ): item.get("id", f"idx_{idx}")
                for idx, item in enumerate(pending, start=1)
            }

            for fut in as_completed(futures):
                sample_id = futures[fut]
                result = fut.result()
                done += 1

                with write_lock:
                    if result.get("generation_prompt"):
                        plen = result.pop("_prompt_len", 0)
                        rcount = result.pop("_req_count", 0)
                        retried = result.pop("_quality_retried", False)
                        result.pop("_spec_model", None)

                        prompt_lengths.append(plen)
                        req_counts.append(rcount)
                        if retried:
                            quality_retried_count += 1

                        line = json.dumps(result, ensure_ascii=False) + "\n"
                        out_f.write(line)
                        out_f.flush()
                        success += 1
                        elapsed = time.time() - t0
                        retry_tag = " [retried]" if retried else ""
                        print(
                            f"[{done}/{total}] \u2713 {sample_id}  "
                            f"prompt={plen}  reqs={rcount}{retry_tag}  "
                            f"({elapsed:.1f}s elapsed)",
                            flush=True,
                        )
                    else:
                        failed += 1
                        err_text = str(result.get("spec_error", "Unknown error"))[:200]
                        print(
                            f"[{done}/{total}] \u2717 {sample_id}  error: {err_text}",
                            flush=True,
                        )
                        if failed_f is not None:
                            failed_f.write(
                                json.dumps(
                                    {
                                        "id": result.get("id", sample_id),
                                        "spec_error": err_text,
                                        "spec_raw_response": result.get(
                                            "spec_raw_response", ""
                                        )[:2000],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            failed_f.flush()

        if failed_f is not None:
            failed_f.close()

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"Success: {success}/{total}  Failed: {failed}/{total}")
    if prompt_lengths:
        avg_plen = sum(prompt_lengths) / len(prompt_lengths)
        print(
            f"Prompt length: avg={avg_plen:.0f}  "
            f"min={min(prompt_lengths)}  max={max(prompt_lengths)}"
        )
        below_threshold = sum(1 for p in prompt_lengths if p < MIN_PROMPT_LENGTH)
        print(f"  Below {MIN_PROMPT_LENGTH}: {below_threshold}/{len(prompt_lengths)}")
    if req_counts:
        avg_rc = sum(req_counts) / len(req_counts)
        print(f"Requirements: avg={avg_rc:.1f}  min={min(req_counts)}  max={max(req_counts)}")
    print(f"Quality retries: {quality_retried_count}")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
