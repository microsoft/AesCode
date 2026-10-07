"""Generate one HTML infographic from a prompt and a reference image.

Three backends:

    --provider openai     OpenAI, or any OpenAI-compatible server (vLLM, SGLang)
    --provider anthropic  Anthropic Messages API
    --provider local      downloaded weights through transformers

Examples:

    # AesCode served locally with vLLM
    vllm serve microsoft/AesCode-32B --port 8000 --limit-mm-per-prompt image=2
    python -m inference.generate --provider openai --model microsoft/AesCode-32B \
        --base-url http://127.0.0.1:8000/v1 \
        --prompt-file brief.txt --image reference.png --out page.html

    # downloaded weights, no server
    python -m inference.generate --provider local --model microsoft/AesCode-32B \
        --prompt-file brief.txt --image reference.png --out page.html

    # a closed API as a baseline
    python -m inference.generate --provider anthropic --model claude-sonnet-4-5 \
        --prompt-file brief.txt --image reference.png --out page.html
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.prompts import DEFAULT_SYSTEM_PROMPT


def extract_html_document(text: str) -> str:
    """Pull the HTML document out of a model reply, tolerating code fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:html)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    doctype_index = stripped.lower().find("<!doctype html>")
    if doctype_index >= 0:
        return stripped[doctype_index:].strip()
    html_index = stripped.lower().find("<html")
    if html_index >= 0:
        return "<!DOCTYPE html>\n" + stripped[html_index:].strip()
    raise ValueError("Model output did not contain an HTML document.")


def to_data_url(path: str | Path) -> str:
    path = Path(path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode()
    return f"data:{media_type};base64,{payload}"


def generate(
    *,
    provider: str,
    model: str,
    user_text: str,
    image: str | Path | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 12000,
    temperature: float = 0.8,
    top_p: float = 0.95,
    base_url: str | None = None,
) -> str:
    if provider == "local":
        from inference.providers.local import (
            build_qwen3_vl_messages,
            call_qwen3_vl_inference,
            load_qwen3_vl_model,
        )

        bundle = load_qwen3_vl_model(model_name=model)
        messages = build_qwen3_vl_messages(
            image=str(image) if image else "", prompt=user_text, system_prompt=system_prompt
        )
        return call_qwen3_vl_inference(bundle, messages, max_new_tokens=max_tokens)[0]

    image_urls = [to_data_url(image)] if image else []

    if provider == "openai":
        from inference.providers.openai_api import call_html_generation, load_client

        client = load_client(base_url=base_url)
    elif provider == "anthropic":
        from inference.providers.anthropic_api import call_html_generation, load_client

        client = load_client()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return call_html_generation(
        client,
        model,
        system_prompt,
        user_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        image_data_urls=image_urls,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate one HTML infographic.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "local"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $AESCODE_BASE_URL)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-file", type=Path,
                        help="File holding the generation prompt")
    source.add_argument("--spec-file", type=Path,
                        help="Spec JSON from spec/ or inference.from_idea. Wraps the prompt in "
                             "the canvas and reference template used for training and evaluation")
    parser.add_argument("--image", type=Path, help="Reference image")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the HTML")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    if args.spec_file:
        from inference.prompts import DEFAULT_USER_TEMPLATE, build_user_text

        sample = json.loads(args.spec_file.read_text(encoding="utf-8"))
        user_text = build_user_text(sample, DEFAULT_USER_TEMPLATE, has_image=args.image is not None)
    else:
        user_text = args.prompt_file.read_text(encoding="utf-8")

    raw = generate(
        provider=args.provider,
        model=args.model,
        user_text=user_text,
        image=args.image,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        base_url=args.base_url,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(extract_html_document(raw), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    cli()
