#!/usr/bin/env python3
"""DesignCoder — try-it demo.

Turn a free-form description of what you want into a rendered design, end to end:

    1. your idea        → a synthetic *reference image* (image model)
    2. reference image  → a detailed *generation prompt* (VLM)
    3. prompt + image   → standalone *HTML code* (the DesignCoder model / any VLM)
    4. HTML placeholders → filled with real images (generated or user-supplied)

Why go through a reference image first? A user's own description is usually a
short paragraph — not the long, structured, distribution-matched prompt the
DesignCoder model was trained on. Rendering the idea into a reference image
first turns the user's intent into something concrete and visual; the detailed
generation prompt is then read back off that image, so the downstream code model
receives an input that looks like its training distribution while still faithfully
reflecting what the user asked for.

This is a demonstration entry point for trying the system on an ad-hoc idea. It
reads all credentials from `.env` (see `.env.example`) and does NOT touch the
training pipeline.

Usage
-----
    python eval/demo_generate.py \
        --idea "A dark-mode dashboard for a weather app. Show a big current-
                temperature panel on the left, a 5-day forecast strip along the
                bottom, and an hourly line chart on the right (hours 00-12 with
                values 8, 7, 7, 9, 12, 15, 17, 18, 16, 14, 11, 9 C). Cool blue
                palette." \
        --out  demo_out/

    # For charts/tables, put the real data in --idea (numeric series or table
    # rows). The model was trained on prompts that carry concrete data, so this
    # keeps the input in-distribution and the render reflects your actual numbers.

    # skip image generation and bring your own reference image:
    python eval/demo_generate.py --idea "..." --reference-image my_ref.png --out demo_out/

    # fill code placeholders with a specific user image instead of generating:
    python eval/demo_generate.py --idea "..." --placeholder-image logo.png --out demo_out/

Outputs (under --out):
    reference.png      the synthetic reference image (stage 1)
    prompt.txt         the detailed generation prompt (stage 2)
    design.html        the generated HTML (stage 3)
    design.filled.html HTML with placeholders replaced (stage 4)

Requirements: openai (+ google-genai if using the Gemini image provider),
Pillow, and the credentials in `.env`.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env (simple, dependency-free).
_env = REPO_ROOT / ".env"
if _env.exists():
    for _raw in _env.read_text(encoding="utf-8").splitlines():
        _l = _raw.strip()
        if not _l or _l.startswith("#") or "=" not in _l:
            continue
        if _l.startswith("export "):
            _l = _l[len("export "):].strip()
        _k, _v = _l.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v


# --------------------------------------------------------------------------- #
# Stage 1: idea -> reference image (concretizes the user's intent)
# --------------------------------------------------------------------------- #
def idea_to_reference_image(idea: str, out_path: Path, provider: str) -> Path:
    """Generate a reference image from the user's description using an image model.

    provider: 'gemini' (Google Nano Banana Pro) or 'gpt' (OpenAI gpt-image).
    """
    prompt = (
        "Generate a single, clean, self-contained design reference image that "
        f"visually realizes this request:\n\n{idea}\n\n"
        "Make it look like a polished UI / infographic mockup."
    )
    if provider == "gemini":
        from google import genai  # type: ignore
        client = genai.Client(api_key=os.environ["GOOGLE_NANO_BANANA"])
        model = os.environ.get("IMAGE_MODEL", "gemini-3-pro-image-preview")
        resp = client.models.generate_content(model=model, contents=prompt)
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None):
                out_path.write_bytes(part.inline_data.data)
                return out_path
        raise RuntimeError("Gemini returned no image data.")
    elif provider == "gpt":
        from openai import OpenAI
        # Works with OpenAI or any OpenAI-compatible image endpoint. Set
        # OPENAI_API_KEY (+ optional OPENAI_BASE_URL) in .env.
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("AZURE_OPENAI_API_KEY", "")),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        model = os.environ.get("IMAGE_MODEL", "gpt-image-1")
        result = client.images.generate(model=model, prompt=prompt, size="1024x1024")
        out_path.write_bytes(base64.b64decode(result.data[0].b64_json))
        return out_path
    raise ValueError(f"Unknown image provider: {provider!r} (use 'gemini' or 'gpt')")


def _image_data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# Stage 2: reference image -> detailed generation prompt
# --------------------------------------------------------------------------- #
def image_to_detailed_prompt(image_path: Path, idea: str) -> str:
    """Ask a VLM to expand the reference image into a detailed generation prompt."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    system = (
        "You analyze one reference design image and write a single, detailed, "
        "self-contained generation prompt that fully specifies the layout, all "
        "visible text, structure, and coarse visual style, so a code model can "
        "reproduce the design as standalone HTML. Use https://placehold.co/ URLs "
        "for any photographic/logo regions. Return only the prompt text."
    )
    user_text = f"Original idea: {idea}\n\nWrite the detailed generation prompt for this reference image."
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
            ]},
        ],
        max_completion_tokens=2000,
    )
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Stage 3: prompt + image -> HTML code
# --------------------------------------------------------------------------- #
def prompt_to_html(generation_prompt: str, image_path: Path, provider: str = "azure") -> str:
    """Generate standalone HTML from the prompt + reference image.

    provider:
      'azure' — use the Azure OpenAI VLM (default; good for trying the pipeline
                without serving your own model).
      'vllm'  — point an OpenAI-compatible client at your own served DesignCoder
                model (e.g. `vllm serve <ckpt> --served-model-name designcoder`).
                Set VLLM_BASE_URL (e.g. http://localhost:8000/v1),
                VLLM_MODEL (served model name), and optional VLLM_API_KEY in .env.
    """
    from inference.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE, build_user_text
    from inference.providers import call_html_generation, extract_html_document

    if provider == "vllm":
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
        deployment = os.environ.get("VLLM_MODEL", "designcoder")
    else:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        )
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    sample = {"generation_prompt": generation_prompt, "verifiable_requirements": []}
    user_text = build_user_text(sample, DEFAULT_USER_TEMPLATE, has_image=True)
    raw = call_html_generation(
        client=client,
        deployment=deployment,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_text=user_text,
        max_tokens=8192,
        image_data_url=_image_data_url(image_path),
    )
    return extract_html_document(raw)


# --------------------------------------------------------------------------- #
# Stage 4: fill placeholders with real images
# --------------------------------------------------------------------------- #
_PLACEHOLDER_RE = re.compile(r"https?://placehold\.co/[^\s\"')]+")


def fill_placeholders(html: str, image_provider: str, user_image: Path | None, out_dir: Path) -> str:
    """Replace placehold.co URLs with real images (user-supplied or generated)."""
    urls = list(dict.fromkeys(_PLACEHOLDER_RE.findall(html)))
    if not urls:
        return html
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls):
        target = assets / f"placeholder_{i}.png"
        if user_image is not None:
            target.write_bytes(Path(user_image).read_bytes())
        else:
            # Derive a caption from the placehold.co ?text= hint, if any.
            m = re.search(r"text=([^&]+)", url)
            caption = (m.group(1).replace("+", " ") if m else "image asset")
            idea_to_reference_image(caption, target, image_provider)
        html = html.replace(url, f"assets/{target.name}")
    return html


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="DesignCoder try-it demo.")
    ap.add_argument("--idea", required=True,
                    help="A paragraph describing what you want to design (free-form). "
                         "If it includes a chart or table, put the real data in the text "
                         "(numeric series / table rows) — DesignCoder was trained on prompts "
                         "that carry concrete data, so this keeps the input in-distribution.")
    ap.add_argument("--out", type=Path, default=Path("demo_out"), help="Output directory.")
    ap.add_argument("--image-provider", choices=["gemini", "gpt"], default="gemini",
                    help="Image model for reference/placeholder generation (gemini | gpt).")
    ap.add_argument("--provider", choices=["azure", "vllm"], default="azure",
                    help="Stage-3 HTML generator: 'azure' (default) or 'vllm' to use "
                         "your own served DesignCoder model (see VLLM_* in .env.example).")
    ap.add_argument("--reference-image", type=Path, default=None,
                    help="Skip stage 1 and use this reference image instead.")
    ap.add_argument("--placeholder-image", type=Path, default=None,
                    help="Fill all code placeholders with this user image (skip generation).")
    ap.add_argument("--skip-fill", action="store_true", help="Leave placehold.co URLs as-is.")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1
    if args.reference_image:
        ref = out / "reference.png"
        ref.write_bytes(Path(args.reference_image).read_bytes())
        print(f"[1/4] using provided reference image → {ref}")
    else:
        ref = idea_to_reference_image(args.idea, out / "reference.png", args.image_provider)
        print(f"[1/4] reference image generated → {ref}")

    # Stage 2
    prompt = image_to_detailed_prompt(ref, args.idea)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"[2/4] detailed generation prompt → {out/'prompt.txt'} ({len(prompt)} chars)")

    # Stage 3
    html = prompt_to_html(prompt, ref, args.provider)
    (out / "design.html").write_text(html, encoding="utf-8")
    print(f"[3/4] HTML generated → {out/'design.html'} ({len(html)} chars)")

    # Stage 4
    if args.skip_fill:
        print("[4/4] skipped placeholder fill.")
    else:
        filled = fill_placeholders(html, args.image_provider, args.placeholder_image, out)
        (out / "design.filled.html").write_text(filled, encoding="utf-8")
        print(f"[4/4] placeholders filled → {out/'design.filled.html'}")

    print("\nDone. Open the .html in a browser to view the result.")


if __name__ == "__main__":
    main()
