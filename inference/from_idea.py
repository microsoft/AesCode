"""Turn a one-line idea into the two inputs AesCode expects, then generate.

AesCode is trained on a detailed generation prompt paired with a reference
image. A user who only has a sentence has neither, so this module reconstructs
both before calling the model:

    idea  ──(image API)──>  reference image
                               │
                               └──(vision API, spec/)──>  generation prompt
                                                          + verifiable requirements
                               │
    reference image + generation prompt ──(AesCode)──>  HTML

The reference image is design guidance, not ground truth. Its text and numbers
are whatever the image model invented, so ``spec.json`` is written out as an
editable artifact: change the wording in ``generation_prompt`` and the figures
in each requirement's ``expected.data``, then resume. Those two fields are
coupled, and ``load_edited_spec`` re-checks the coupling after every edit.

Usage
-----
    # draft, then stop so the facts can be corrected
    python -m inference.from_idea \\
        --idea "A quarterly sales dashboard for a European EV brand, with \\
                revenue and units for the last four quarters, a regional \\
                breakdown, and a table of the top five models." \\
        --outdir out/ --skip-generate

    # ... edit out/spec.json ...

    python -m inference.from_idea \\
        --spec-file out/spec.json --outdir out/ \\
        --model microsoft/AesCode-32B \\
        --base-url http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.generate import extract_html_document, generate
from inference.prompts import DEFAULT_USER_TEMPLATE, build_user_text
from spec.io import get_reference_aspect_ratio, get_reference_orientation, image_to_data_url
from spec.prompting import (
    MIN_PROMPT_LENGTH,
    MIN_REQUIREMENTS,
    build_detailed_spec_prompts,
    parse_spec_response,
    text_exact_is_supported,
)
from spec.schema import build_output_schema, build_schema_prompt_contract


def image_size(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as handle:
        width, height = handle.size
    return f"{width}x{height}"


def load_spec_client(base_url: Optional[str] = None) -> Any:
    """Client for the hosted vision model that recovers the generation prompt.

    Deliberately does not fall back to ``AESCODE_BASE_URL``/``AESCODE_API_KEY``:
    those address the local server hosting the AesCode checkpoint, which is a
    different endpoint from the model used for prompt recovery.
    """
    from openai import OpenAI

    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=600.0,
    )


def call_spec_model(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_data_url: str,
    base_url: Optional[str],
    max_tokens: int,
) -> str:
    """One vision call returning the raw spec JSON text."""
    if provider == "anthropic":
        from inference.providers.anthropic_api import _image_block, load_client

        client = load_client()
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": user_text}, _image_block(image_data_url)],
            }],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

    client = load_spec_client(base_url)
    response = client.chat.completions.create(
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
        max_completion_tokens=max_tokens,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def build_spec(
    image_path: Path,
    *,
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    sample_id: str = "idea_0001",
    max_tokens: int = 4000,
    retries: int = 3,
    retry_sleep: float = 2.0,
    image_max_long_side: int = 1120,
) -> Dict[str, Any]:
    """Read a reference image and return a validated spec object."""
    system_prompt, user_template = build_detailed_spec_prompts()
    schema = build_output_schema()

    reference_image_wh = image_size(image_path)
    item = {"id": sample_id, "reference_image_wh": reference_image_wh}
    aspect_ratio = get_reference_aspect_ratio(reference_image_wh)
    orientation = get_reference_orientation(reference_image_wh)

    user_text = user_template.format(
        id=sample_id,
        reference_image_wh=reference_image_wh,
        reference_aspect_ratio=aspect_ratio,
        reference_orientation=orientation,
        schema_contract=build_schema_prompt_contract(schema),
    )
    image_data_url = image_to_data_url(image_path, max_long_side=image_max_long_side)

    last_error: Optional[Exception] = None
    thin_retried = False
    for attempt in range(1, retries + 2):
        try:
            raw = call_spec_model(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                image_data_url=image_data_url,
                base_url=base_url,
                max_tokens=max_tokens,
            )
            if not raw:
                raise RuntimeError("Empty response from the spec model.")
            spec = parse_spec_response(
                raw,
                item=item,
                reference_aspect_ratio=aspect_ratio,
                reference_orientation=orientation,
                schema=schema,
            )

            prompt_length = len(spec.get("generation_prompt", ""))
            requirement_count = len(spec.get("verifiable_requirements", []))
            if not thin_retried and (
                prompt_length < MIN_PROMPT_LENGTH or requirement_count < MIN_REQUIREMENTS
            ):
                thin_retried = True
                print(
                    f"  spec too thin ({prompt_length} chars, {requirement_count} "
                    f"requirements), retrying",
                    file=sys.stderr,
                )
                time.sleep(retry_sleep)
                continue

            spec.setdefault("reference_metadata", {})
            spec["reference_metadata"]["reference_image_path"] = str(image_path)
            return spec
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
            last_error = error
            if attempt <= retries:
                time.sleep(retry_sleep)

    raise RuntimeError(f"Could not build a spec from {image_path}: {last_error}")


def _rejection_reason(requirement: Dict[str, Any], generation_prompt: str) -> str:
    category = requirement.get("category")
    if category == "text_exact":
        text = (requirement.get("expected") or {}).get("text") or ""
        if not text_exact_is_supported(text, generation_prompt):
            return (f"its text is not in generation_prompt verbatim; "
                    f"add {text!r} to the prompt or delete the requirement")
        return "the text looks corrupted or carries no readable characters"
    if category == "chart_data":
        return 'needs target.chart_type "echarts" and a valid target.chart_subtype'
    if category == "table_data":
        return "expected.data must be a 2D array whose first row is the headers"
    return "it does not satisfy the spec contract"


def load_edited_spec(path: Path) -> Dict[str, Any]:
    """Re-validate a hand-edited spec against the same contract that produced it.

    Editing is the point of writing ``spec.json`` out, but two fields are
    coupled: every ``text_exact`` string must still appear verbatim in
    ``generation_prompt``, and chart and table payloads must keep their shape.
    Anything that no longer holds is dropped here rather than silently
    weakening the page, and the reason is printed.
    """
    try:
        edited = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}") from error

    metadata = edited.get("reference_metadata") or {}
    reference_image_wh = metadata.get("reference_image_wh", "")
    before = {
        str(r.get("id")): r
        for r in edited.get("verifiable_requirements") or []
        if isinstance(r, dict)
    }

    spec = parse_spec_response(
        json.dumps(edited),
        item={"id": edited.get("id", "idea_0001"), "reference_image_wh": reference_image_wh},
        reference_aspect_ratio=get_reference_aspect_ratio(reference_image_wh),
        reference_orientation=get_reference_orientation(reference_image_wh),
        schema=build_output_schema(),
    )
    spec["reference_metadata"]["reference_image_path"] = metadata.get(
        "reference_image_path", ""
    )

    kept = {str(r.get("id")) for r in spec.get("verifiable_requirements", [])}
    prompt = spec.get("generation_prompt", "")
    for req_id, requirement in before.items():
        if req_id not in kept:
            print(f"  dropped {req_id}: {_rejection_reason(requirement, prompt)}",
                  file=sys.stderr)

    if len(prompt) < MIN_PROMPT_LENGTH:
        print(f"  note: generation_prompt is {len(prompt)} characters, below the "
              f"{MIN_PROMPT_LENGTH} used during spec construction", file=sys.stderr)
    return spec


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    spec_path = outdir / "spec.json"

    if args.spec_file:
        print(f"[1/3] Reusing the edited spec {args.spec_file}")
        spec = load_edited_spec(args.spec_file)
        reference_path = args.reference_image or Path(
            spec["reference_metadata"].get("reference_image_path") or ""
        )
        if not reference_path or not reference_path.exists():
            raise SystemExit(
                "The edited spec does not point at a readable reference image. "
                "Pass --reference-image."
            )
    else:
        if args.reference_image:
            reference_path = args.reference_image
            print(f"[1/3] Using reference image {reference_path}")
        else:
            from inference.providers.image_api import (
                build_image_prompt,
                generate_reference_image,
                load_client,
            )

            reference_path = outdir / "reference.png"
            generate_reference_image(
                load_client(base_url=args.image_base_url),
                args.image_model,
                build_image_prompt(args.idea, args.orientation),
                reference_path,
                orientation=args.orientation,
                size=getattr(args, "image_size", None),
            )
            print(f"[1/3] Reference image -> {reference_path}")

        spec = build_spec(
            reference_path,
            provider=args.spec_provider,
            model=args.spec_model,
            base_url=args.spec_base_url,
            max_tokens=args.spec_max_tokens,
            retries=args.retries,
        )

    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    user_text = build_user_text(spec, DEFAULT_USER_TEMPLATE, has_image=True)
    preview_path = outdir / "preview.txt"
    preview_path.write_text(user_text, encoding="utf-8")
    print(
        f"[2/3] Spec -> {spec_path} "
        f"({len(spec['generation_prompt'])} chars, "
        f"{len(spec.get('verifiable_requirements', []))} verifiable requirements)\n"
        f"      Model input preview -> {preview_path}"
    )

    if args.skip_generate:
        print(
            "[3/3] Stopped before generation. Edit the facts in spec.json — the page text\n"
            "      in generation_prompt, the numbers in each requirement's expected.data —\n"
            f"      then rerun with --spec-file {spec_path}"
        )
        return

    raw = generate(
        provider=args.provider,
        model=args.model,
        user_text=user_text,
        image=reference_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        base_url=args.base_url,
    )
    page = outdir / "page.html"
    page.write_text(extract_html_document(raw), encoding="utf-8")
    print(f"[3/3] Page -> {page}")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reference image and generation prompt from one line, then generate.",
    )
    parser.add_argument("--idea",
                        help="Short description of the page: what it is, what it should show,\n and any style preference. A few sentences work better than a few words.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Directory for reference.png, spec.json, preview.txt and page.html")
    parser.add_argument("--orientation", choices=["landscape", "portrait", "square"],
                        default="landscape")
    parser.add_argument("--reference-image", type=Path,
                        help="Use this image instead of generating one; --idea is then unused")
    parser.add_argument("--spec-file", type=Path,
                        help="Resume from an edited spec.json, skipping both API calls")

    parser.add_argument("--image-model",
                        default=os.environ.get("IMAGE_MODEL", "gpt-image-2"),
                        help="Text-to-image model (default: $IMAGE_MODEL or gpt-image-2)")
    parser.add_argument("--image-base-url", help="Image endpoint (default: $OPENAI_BASE_URL)")
    parser.add_argument("--image-size", help="Override the canvas implied by --orientation")

    parser.add_argument("--spec-provider", choices=["openai", "anthropic"], default="openai",
                        help="Vision model that reads the reference image")
    parser.add_argument("--spec-model",
                        default=os.environ.get("VLM_MODEL", "gpt-5.5"),
                        help="Vision model (default: $VLM_MODEL or gpt-5.5)")
    parser.add_argument("--spec-base-url",
                        help="Vision endpoint (default: $OPENAI_BASE_URL)")
    parser.add_argument("--spec-max-tokens", type=int, default=4000)
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--provider", choices=["openai", "anthropic", "local"], default="openai")
    parser.add_argument("--model", default="microsoft/AesCode-32B")
    parser.add_argument("--base-url", help="AesCode endpoint (default: $AESCODE_BASE_URL)")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--skip-generate", action="store_true",
                        help="Stop after writing spec.json so its facts can be edited")

    args = parser.parse_args()
    if not (args.idea or args.reference_image or args.spec_file):
        parser.error(
            "Provide --idea, or --reference-image to skip image generation, "
            "or --spec-file to resume from an edited spec."
        )
    run(args)


if __name__ == "__main__":
    cli()
