#!/usr/bin/env python3
"""Batch inference for evaluation: greedy and sampling (pass@K).

Every backend is reached through the OpenAI chat-completions schema, so a local
vLLM or SGLang server and a hosted API differ only in the base URL and the model
name. Set OPENAI_API_KEY when the endpoint needs one; local servers ignore it.

Stage 1 of the benchmark is the `sample` mode below, at the locked knobs.

    # The benchmark setting: n=3 rollouts at T=0.8, top_p=0.95
    python3 eval/eval_vllm_infer.py \
        --spec-jsonl data/eval_spec.jsonl --image-dir data/images \
        --endpoints http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
        --mode sample --n 3 --temperature 0.8 --top-p 0.95 \
        --output-dir runs/my_model_final300/html \
        --max-tokens 12000 --workers 64 --resume

    # A hosted API as a baseline, same prompts and the same token budget
    OPENAI_API_KEY=sk-... python3 eval/eval_vllm_infer.py \
        --spec-jsonl data/eval_spec.jsonl --image-dir data/images \
        --endpoints https://api.openai.com/v1 --model gpt-5.5 \
        --mode sample --n 3 --temperature 0.8 --top-p 0.95 \
        --output-dir runs/gpt55_final300/html --workers 4 --resume

    # Greedy -- not used by the main table
    python3 eval/eval_vllm_infer.py ... --mode greedy
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
)


# ---------------------------------------------------------------------------
# HTML extraction (inline, avoids heavy imports)
# ---------------------------------------------------------------------------
def normalize_escaped_document(text: str) -> str:
    """Decode whole-document line escapes emitted by some chat templates."""
    if text.count("\\n") >= 5 and text.count("\n") <= 3:
        return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return text


def extract_html(text: str) -> str:
    """Extract HTML document from model output (handles ```html fences)."""
    stripped = normalize_escaped_document(text.strip())
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
# The passages gen_noref_arm.py deletes to build its no-image prompt. The API
# no-image arms (gpt55plain, gpt51plain, …) were generated with that prompt, so
# an open-source no-image arm has to be generated with the identical one or the
# two blocks of the paper table are not comparable. Kept as a literal copy of
# gen_noref_arm._TEMPLATE_CUTS and asserted at call time, so a template edit
# fails loudly here instead of silently producing a differently-prompted arm.
_CLEAN_NOIMAGE_CUTS = [
    " and uses the reference image only as visual design guidance.",
    "[Reference Image]\n{reference_note}\n\n",
]


def clean_noimage_template() -> str:
    out = DEFAULT_USER_TEMPLATE
    for cut in _CLEAN_NOIMAGE_CUTS:
        if cut not in out:
            raise RuntimeError(
                "DEFAULT_USER_TEMPLATE no longer contains a passage the clean "
                f"no-image mode cuts; update _CLEAN_NOIMAGE_CUTS.\nmissing: {cut!r}"
            )
        out = out.replace(cut, "." if cut.startswith(" and") else "")
    return out


def build_generate_messages(
    sample: Dict[str, Any], image_dir: Path, system_prompt: str,
    no_image_eval: bool = False, clean_noimage: bool = False,
) -> List[Dict[str, Any]]:
    """Build OpenAI-format messages for generate mode.

    Three mutually exclusive modes:

    * default: attach the reference image.
    * ``no_image_eval``: drop the image and rewrite the [Reference Image] block
      to the long "[non-image training mode]…" note, matching what our own
      noimage-trained model saw during training. Only fair to that model.
    * ``clean_noimage``: drop the image and delete the [Reference Image] block
      outright, exactly as gen_noref_arm.py does. This is the mode to use for
      third-party models, because the training-parity note is a prompt they
      never saw and would be scored on.
    """
    if no_image_eval and clean_noimage:
        raise ValueError("no_image_eval and clean_noimage are mutually exclusive")
    sample_id = str(sample.get("id", ""))
    img_path = image_dir / f"{sample_id}.png"
    has_image = img_path.exists() and not (no_image_eval or clean_noimage)
    template = clean_noimage_template() if clean_noimage else DEFAULT_USER_TEMPLATE
    user_text = build_user_text(sample, template, has_image=has_image)

    # Training-test parity: in no-image mode, replace the [Reference Image] block
    # with the same explicit note that dataset_loader.py injects during training.
    if no_image_eval:
        import re as _re
        new_note = (
            "[non-image training mode] No reference image is provided in this example. "
            "This is intentional — the task here is to design the page from the generation prompt alone, "
            "inferring palette, hierarchy, layout, and typography purely from the text spec. "
            "Treat the generation prompt as the single source of truth for both content and visual direction."
        )
        user_text = _re.sub(
            r"\[Reference Image\]\n.*?\n\n\[Canvas\]",
            f"[Reference Image]\n{new_note}\n\n[Canvas]",
            user_text,
            count=1,
            flags=_re.DOTALL,
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    user_content: List[Dict[str, Any]] = []
    if has_image:
        data_url = image_to_data_url(img_path)
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    user_content.append({"type": "text", "text": user_text})
    # Some chat templates (InternVL3) do `str + content` and reject a list even
    # when it is text-only. Flattening is opt-in and only legal with no image,
    # so arms generated before this switch existed are bit-for-bit unaffected.
    if not has_image and os.environ.get("FLATTEN_TEXT_ONLY", "") == "1":
        messages.append({"role": "user", "content": user_text})
        return messages
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
        client = AsyncOpenAI(base_url=endpoint,
                             api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY")
        try:
            kwargs = {}
            # Reasoning checkpoints (Qwen3.5/3.8) default enable_thinking=true
            # in their chat template and then spend the entire token budget
            # deliberating, so the response holds a monologue instead of HTML.
            if os.environ.get("DISABLE_THINKING", "").strip() == "1":
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}}
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                **kwargs,
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
    no_image_eval: bool = False,
    clean_noimage: bool = False,
) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    output_path = output_dir / f"{sample_id}.html"
    async with sem:
        try:
            messages = build_generate_messages(sample, image_dir, system_prompt, no_image_eval=no_image_eval, clean_noimage=clean_noimage)
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
    no_image_eval: bool = False,
    clean_noimage: bool = False,
) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    async with sem:
        try:
            messages = build_generate_messages(sample, image_dir, system_prompt, no_image_eval=no_image_eval, clean_noimage=clean_noimage)
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
                    # dump raw so extraction failures are diagnosable (CoT-truncation vs format drift)
                    try:
                        raw = res[0] if (res and not isinstance(res, Exception)) else ""
                        (output_dir / f"{sample_id}_k{k}.rawfail.txt").write_text(raw or "", encoding="utf-8")
                    except Exception:
                        pass

            return {"sample_id": sample_id, "status": "success", "n_ok": success_count, "n_total": n}
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
        if args.mode == "greedy":
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

    endpoints = [ep.strip() for ep in args.endpoints.split(",") if ep.strip()]
    if not endpoints:
        print("ERROR: --endpoints is required: one or more OpenAI-compatible "
              "chat-completions base URLs, comma-separated.")
        sys.exit(1)
    pool = EndpointPool(endpoints)

    async def call_fn(messages, *, max_tokens=12000, temperature=0.0, top_p=1.0, n=1):
        return await call_vllm(
            pool, messages, model=args.model,
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
            process_greedy(s, call_fn, image_dir, output_dir, system_prompt, args.max_tokens, sem,
                           no_image_eval=args.no_image_eval,
                           clean_noimage=args.clean_noimage)
            for s in specs
        ]
    elif args.mode == "sample":
        tasks = [
            process_sample(
                s, call_fn, image_dir, output_dir, system_prompt,
                args.max_tokens, args.n, args.temperature, args.top_p, sem,
                no_image_eval=args.no_image_eval,
                clean_noimage=args.clean_noimage,
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
                        help="Comma-separated OpenAI-compatible base URLs. A local "
                             "vLLM/SGLang server and a hosted API are the same thing "
                             "here; requests are spread round-robin over the list.")
    parser.add_argument("--model", type=str, default="model",
                        help="Model name sent to the endpoints. Leave as 'model' when "
                             "the server was started with --served-model-name model.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", choices=["greedy", "sample"], required=True)
    parser.add_argument("--n", type=int, default=64, help="Number of samples for pass@K")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=64,
                        help="Max concurrent requests")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-image-eval", action="store_true",
                        help="Train/test parity: do NOT attach the reference image AND "
                             "rewrite [Reference Image] section with the '[non-image training mode]…' "
                             "note, mirroring the noimage SFT ablation. Used to evaluate noimage ckpts.")
    parser.add_argument("--clean-noimage", action="store_true",
                        help="Do NOT attach the reference image and DELETE the [Reference Image] "
                             "block, leaving a prompt that stands on its own. This is the setting "
                             "for the reference-free arms. Use it rather than --no-image-eval for "
                             "any model you did not train: that flag's training-parity note is a "
                             "prompt such a model has never seen.")
    parser.add_argument("--ref-image-dir", type=str, default="",
                        help="Override reference image dir. Auto-updates reference_image_path "
                             "and reference_image_wh in each sample at runtime.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
