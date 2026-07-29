#!/usr/bin/env python3
"""vLLM-based batch inference for evaluation: greedy, sampling (pass@K), and refine.

Usage:
    # Greedy (base/p1/p2)
    python eval/eval_vllm_infer.py \
        --spec-jsonl <data_root>/eval.jsonl \
        --image-dir <data_root>/images \
        --endpoints http://ip:8000/v1,http://ip:8001/v1,... \
        --mode greedy --output-dir data/inference_runs/eval500_base_8b \
        --max-tokens 12000 --workers 64 --resume

    # Sampling pass@64
    python eval/eval_vllm_infer.py \
        --spec-jsonl ... --image-dir ... --endpoints ... \
        --mode sample --n 64 --temperature 0.8 --top-p 0.95 \
        --output-dir data/inference_runs/eval500_sft_cot_p2_sample64 \
        --max-tokens 12000 --workers 64 --resume

    # Refine (uses greedy output as draft)
    python eval/eval_vllm_infer.py \
        --spec-jsonl ... --image-dir ... --endpoints ... \
        --mode refine --draft-dir data/inference_runs/eval500_sft_cot_p2 \
        --output-dir data/inference_runs/eval500_sft_cot_p2_refine \
        --max-tokens 12000 --workers 64 --resume

    # Azure GPT (fair comparison — same prompts, image handling, max_tokens)
    python eval/eval_vllm_infer.py \
        --spec-jsonl ... --image-dir ... \
        --azure --azure-deployment gpt-5.4 \
        --mode greedy --output-dir data/inference_runs/eval500_gpt54_fair \
        --max-tokens 12000 --workers 4 --resume
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bootstrap: add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
if (_REPO_ROOT / ".env").exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
    except ImportError:
        pass

from inference.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_TEMPLATE,
    build_user_text,
    build_refine_user_text,
    refine_use_prev_render,
)

# Pixel budgets for eval/inference — larger than SFT training values in
# prompts.py because eval wants maximum visual fidelity from the model.
REFINE_REF_IMAGE_MAX_PIXELS = 2048 * 28 * 28    # 1_605_632
REFINE_REF_IMAGE_MIN_PIXELS = 256 * 28 * 28     # 200_704
REFINE_PREV_RENDER_MAX_PIXELS = 768 * 28 * 28   # 602_112
REFINE_PREV_RENDER_MIN_PIXELS = 256 * 28 * 28   # 200_704


# ---------------------------------------------------------------------------
# HTML extraction (inline, avoids heavy imports)
# ---------------------------------------------------------------------------
def extract_html(text: str) -> str:
    """Extract HTML document from model output (handles ```html fences)."""
    stripped = text.strip()
    # Try to find ```html ... ``` block first
    m = re.search(r"```html\s*(.*?)```", stripped, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner:
            stripped = inner
    elif stripped.startswith("```"):
        stripped = re.sub(r"^```(?:html)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    doctype_index = stripped.lower().find("<!doctype html>")
    if doctype_index >= 0:
        return stripped[doctype_index:].strip()
    html_index = stripped.lower().find("<html")
    if html_index >= 0:
        return "<!DOCTYPE html>\n" + stripped[html_index:].strip()
    # Fallback: return as-is if it contains any HTML tags
    if "<" in stripped:
        return stripped
    raise ValueError("No HTML document in model output")


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------
def image_to_data_url(path: Path) -> str:
    """Convert image file to base64 data URL."""
    suffix = path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "gif": "image/gif"}.get(suffix.lstrip("."), "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_generate_messages(
    sample: Dict[str, Any], image_dir: Path, system_prompt: str,
) -> List[Dict[str, Any]]:
    """Build OpenAI-format messages for generate mode."""
    sample_id = str(sample.get("id", ""))
    img_path = image_dir / f"{sample_id}.png"
    has_image = img_path.exists()
    user_text = build_user_text(sample, DEFAULT_USER_TEMPLATE, has_image=has_image)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    user_content: List[Dict[str, Any]] = []
    if has_image:
        data_url = image_to_data_url(img_path)
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    user_content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_refine_messages(
    sample: Dict[str, Any], draft_html: str, image_dir: Path,
    system_prompt: str, max_prev_html_chars: int = 20000,
    prev_render_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build OpenAI-format messages for refine mode.

    When `prev_render_dir` is provided AND REFINE_USE_PREV_RENDER is on AND
    a matching {sample_id}.png exists, append it as image #2 and switch
    the suffix template to mention "Image 2 = previous render".
    """
    sample_id = str(sample.get("id", ""))
    img_path = image_dir / f"{sample_id}.png"
    has_image = img_path.exists()

    # Resolve prev_render path (only honored if env flag is on)
    prev_render_path: Optional[Path] = None
    if prev_render_dir is not None and refine_use_prev_render():
        cand = prev_render_dir / f"{sample_id}.png"
        if cand.exists():
            prev_render_path = cand

    prev_html = draft_html[:max_prev_html_chars] if len(draft_html) > max_prev_html_chars else draft_html
    user_text = build_refine_user_text(
        sample, prev_html,
        has_image=has_image,
        with_prev_render=prev_render_path is not None,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    user_content: List[Dict[str, Any]] = []
    if has_image:
        data_url = image_to_data_url(img_path)
        user_content.append({"type": "image_url", "image_url": {
            "url": data_url,
            "min_pixels": REFINE_REF_IMAGE_MIN_PIXELS,
            "max_pixels": REFINE_REF_IMAGE_MAX_PIXELS,
        }})
    if prev_render_path is not None:
        prev_url = image_to_data_url(prev_render_path)
        user_content.append({"type": "image_url", "image_url": {
            "url": prev_url,
            "min_pixels": REFINE_PREV_RENDER_MIN_PIXELS,
            "max_pixels": REFINE_PREV_RENDER_MAX_PIXELS,
        }})
    user_content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Async inference
# ---------------------------------------------------------------------------
class EndpointPool:
    """Round-robin pool of vLLM endpoints."""
    def __init__(self, endpoints: List[str]):
        self._endpoints = endpoints
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> str:
        async with self._lock:
            ep = self._endpoints[self._idx % len(self._endpoints)]
            self._idx += 1
            return ep


async def call_vllm(
    pool: EndpointPool,
    messages: List[Dict[str, Any]],
    *,
    model: str = "model",
    max_tokens: int = 12000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n: int = 1,
    retries: int = 3,
) -> List[str]:
    """Call vLLM endpoint via OpenAI API, return list of generated texts."""
    from openai import AsyncOpenAI

    for attempt in range(1, retries + 1):
        endpoint = await pool.next()
        client = AsyncOpenAI(base_url=endpoint, api_key="dummy")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )
            return [choice.message.content or "" for choice in response.choices]
        except Exception as e:
            if attempt == retries:
                raise
            await asyncio.sleep(2 * attempt)
    return []  # unreachable


async def call_azure(
    client: Any,  # AsyncAzureOpenAI
    messages: List[Dict[str, Any]],
    *,
    model: str = "gpt-5.4",
    max_tokens: int = 12000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n: int = 1,
    retries: int = 3,
) -> List[str]:
    """Call Azure OpenAI endpoint, same interface as call_vllm."""
    for attempt in range(1, retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )
            return [choice.message.content or "" for choice in response.choices]
        except Exception as e:
            if attempt == retries:
                raise
            await asyncio.sleep(2 * attempt)
    return []  # unreachable


# Generic call function type: (messages, *, max_tokens, temperature, ...) -> List[str]
# We use closures to bind pool/client so process_* functions stay uniform.


async def process_greedy(
    sample: Dict[str, Any],
    call_fn: Any,  # async callable(messages, *, max_tokens, temperature, n) -> List[str]
    image_dir: Path,
    output_dir: Path,
    system_prompt: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    output_path = output_dir / f"{sample_id}.html"
    async with sem:
        try:
            messages = build_generate_messages(sample, image_dir, system_prompt)
            texts = await call_fn(messages, max_tokens=max_tokens, temperature=0.0, n=1)
            raw_text = texts[0]
            # Save raw model output (including CoT) alongside extracted HTML
            raw_path = output_dir / f"{sample_id}.raw.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
            html = extract_html(raw_text)
            output_path.write_text(html + "\n", encoding="utf-8")
            return {"sample_id": sample_id, "status": "success"}
        except Exception as e:
            return {"sample_id": sample_id, "status": "error", "error": str(e)}


async def process_sample(
    sample: Dict[str, Any],
    call_fn: Any,
    image_dir: Path,
    output_dir: Path,
    system_prompt: str,
    max_tokens: int,
    n: int,
    temperature: float,
    top_p: float,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    async with sem:
        try:
            messages = build_generate_messages(sample, image_dir, system_prompt)
            tasks = []
            for _ in range(n):
                tasks.append(call_fn(
                    messages, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, n=1,
                ))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = 0
            for k, res in enumerate(results):
                out_path = output_dir / f"{sample_id}_k{k}.html"
                if isinstance(res, Exception):
                    continue
                try:
                    html = extract_html(res[0])
                    out_path.write_text(html + "\n", encoding="utf-8")
                    success_count += 1
                except Exception:
                    pass

            return {"sample_id": sample_id, "status": "success", "n_ok": success_count, "n_total": n}
        except Exception as e:
            return {"sample_id": sample_id, "status": "error", "error": str(e)}


async def process_refine(
    sample: Dict[str, Any],
    call_fn: Any,
    image_dir: Path,
    draft_dir: Path,
    output_dir: Path,
    system_prompt: str,
    max_tokens: int,
    max_prev_html_chars: int,
    sem: asyncio.Semaphore,
    prev_render_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    output_path = output_dir / f"{sample_id}.html"
    draft_path = draft_dir / f"{sample_id}.html"
    if not draft_path.exists():
        return {"sample_id": sample_id, "status": "skip", "reason": "no_draft"}
    async with sem:
        try:
            draft_html = draft_path.read_text(encoding="utf-8")
            messages = build_refine_messages(
                sample, draft_html, image_dir, system_prompt,
                max_prev_html_chars=max_prev_html_chars,
                prev_render_dir=prev_render_dir,
            )
            texts = await call_fn(messages, max_tokens=max_tokens, temperature=0.0, n=1)
            raw_text = texts[0]
            raw_path = output_dir / f"{sample_id}.raw.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
            html = extract_html(raw_text)
            output_path.write_text(html + "\n", encoding="utf-8")
            return {"sample_id": sample_id, "status": "success"}
        except Exception as e:
            return {"sample_id": sample_id, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> None:
    # Load specs
    specs: List[Dict[str, Any]] = []
    with open(args.spec_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                specs.append(json.loads(line))

    if args.limit > 0:
        specs = specs[:args.limit]

    # Override reference image metadata at runtime
    if args.ref_image_dir:
        from PIL import Image as _PILImage
        ref_dir = Path(args.ref_image_dir)
        print(f"Overriding ref images from: {ref_dir}")
        _wh_cache: Dict[str, str] = {}
        for s in specs:
            sid = str(s.get("id", ""))
            ref_path = ref_dir / f"{sid}.png"
            if not ref_path.exists():
                ref_path = ref_dir / f"{sid}.jpg"
            if ref_path.exists():
                meta = s.setdefault("reference_metadata", {})
                meta["reference_image_path"] = str(ref_path)
                ref_key = str(ref_path)
                if ref_key not in _wh_cache:
                    with _PILImage.open(ref_path) as img:
                        w, h = img.size
                    _wh_cache[ref_key] = f"{w}x{h}"
                meta["reference_image_wh"] = _wh_cache[ref_key]
                w_str, h_str = meta["reference_image_wh"].split("x")
                meta["reference_orientation"] = "landscape" if int(w_str) >= int(h_str) else "portrait"
                meta["reference_aspect_ratio"] = f"{int(w_str)/int(h_str):.4f}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    # Filter for resume
    if args.resume:
        if args.mode in ("greedy", "refine"):
            specs = [s for s in specs if not (output_dir / f"{s['id']}.html").exists()]
        elif args.mode == "sample":
            # Resume: skip samples that have all K files
            filtered = []
            for s in specs:
                existing = sum(1 for k in range(args.n) if (output_dir / f"{s['id']}_k{k}.html").exists())
                if existing < args.n:
                    filtered.append(s)
            specs = filtered

    if not specs:
        print("Nothing to do (all samples already processed).")
        return

    # Build the call function: either vLLM endpoint pool or Azure client
    use_azure = getattr(args, "azure", False)
    if use_azure:
        from openai import AsyncAzureOpenAI
        azure_client = AsyncAzureOpenAI(
            api_version=args.azure_api_version,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", os.environ.get("ENDPOINT", "")),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", os.environ.get("API_KEY", "")),
        )
        deployment = args.azure_deployment

        async def call_fn(messages, *, max_tokens=12000, temperature=0.0, top_p=1.0, n=1):
            return await call_azure(
                azure_client, messages,
                model=deployment, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, n=n,
            )

        print(f"Backend:   Azure ({deployment})")
    else:
        endpoints = [ep.strip() for ep in args.endpoints.split(",") if ep.strip()]
        if not endpoints:
            print("ERROR: --endpoints required (or use --azure)")
            sys.exit(1)
        pool = EndpointPool(endpoints)

        async def call_fn(messages, *, max_tokens=12000, temperature=0.0, top_p=1.0, n=1):
            return await call_vllm(
                pool, messages,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n,
            )

        print(f"Endpoints: {len(endpoints)}")

    sem = asyncio.Semaphore(args.workers)
    system_prompt = DEFAULT_SYSTEM_PROMPT

    print(f"Mode:      {args.mode}")
    print(f"Samples:   {len(specs)}")
    print(f"Workers:   {args.workers}")
    print(f"Output:    {output_dir}")
    if args.mode == "sample":
        print(f"n={args.n}, temp={args.temperature}, top_p={args.top_p}")
    print(f"Starting at {time.strftime('%H:%M:%S')}")

    t0 = time.time()

    if args.mode == "greedy":
        tasks = [
            process_greedy(s, call_fn, image_dir, output_dir, system_prompt, args.max_tokens, sem)
            for s in specs
        ]
    elif args.mode == "sample":
        tasks = [
            process_sample(
                s, call_fn, image_dir, output_dir, system_prompt,
                args.max_tokens, args.n, args.temperature, args.top_p, sem,
            )
            for s in specs
        ]
    elif args.mode == "refine":
        draft_dir = Path(args.draft_dir)
        if not draft_dir.exists():
            print(f"ERROR: --draft-dir not found: {draft_dir}")
            sys.exit(1)
        prev_render_dir: Optional[Path] = None
        if refine_use_prev_render() and getattr(args, "prev_render_dir", ""):
            prev_render_dir = Path(args.prev_render_dir)
            print(f"Using prev_render_dir: {prev_render_dir}")
        tasks = [
            process_refine(
                s, call_fn, image_dir, draft_dir, output_dir, system_prompt,
                args.max_tokens, args.max_prev_html_chars, sem,
                prev_render_dir=prev_render_dir,
            )
            for s in specs
        ]
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        if done % 50 == 0 or done == len(tasks):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            ok = sum(1 for r in results if r["status"] == "success")
            err = sum(1 for r in results if r["status"] == "error")
            eta = (len(tasks) - done) / rate if rate > 0 else 0
            print(
                f"  [{done}/{len(tasks)}] ok={ok} err={err} "
                f"rate={rate:.1f}/s ETA={eta/60:.1f}min",
                flush=True,
            )

    # Save manifest
    manifest_path = output_dir / "inference_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "success")
    err = sum(1 for r in results if r["status"] == "error")
    skip = sum(1 for r in results if r["status"] == "skip")
    print(f"\nDone in {elapsed/60:.1f}min. ok={ok} err={err} skip={skip}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM batch inference for eval")
    parser.add_argument("--spec-jsonl", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--endpoints", type=str, default="",
                        help="Comma-separated vLLM endpoint URLs (required unless --azure)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", choices=["greedy", "sample", "refine"], required=True)
    parser.add_argument("--azure", action="store_true",
                        help="Use Azure OpenAI instead of vLLM endpoints")
    parser.add_argument("--azure-deployment", type=str, default="gpt-5.4",
                        help="Azure deployment name (default: gpt-5.4)")
    parser.add_argument("--azure-api-version", type=str, default="2025-03-01-preview",
                        help="Azure API version")
    parser.add_argument("--n", type=int, default=64, help="Number of samples for pass@K")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--max-prev-html-chars", type=int, default=20000,
                        help="Truncate draft HTML for refine mode")
    parser.add_argument("--draft-dir", type=str, default="",
                        help="Directory with draft HTML for refine mode")
    parser.add_argument("--workers", type=int, default=64,
                        help="Max concurrent requests")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ref-image-dir", type=str, default="",
                        help="Override reference image dir. Auto-updates reference_image_path "
                             "and reference_image_wh in each sample at runtime.")
    parser.add_argument("--prev-render-dir", type=str, default="",
                        help="Directory of pre-rendered draft PNGs ({id}.png) used as image #2 "
                             "in refine mode. Only honored when REFINE_USE_PREV_RENDER=1.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
