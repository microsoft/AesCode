"""Text-to-image provider used to draft a reference image.

Works with the OpenAI Images API and with any server exposing the same route:

    export OPENAI_BASE_URL=https://api.openai.com/v1
    export OPENAI_API_KEY=...

The reference is design guidance only. Its text, numbers and labels are
expected to be imperfect; the generation prompt remains the source of truth
for content.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional

# Canvas presets accepted by the OpenAI Images API.
SIZE_BY_ORIENTATION = {
    "landscape": "1536x1024",
    "portrait": "1024x1536",
    "square": "1024x1024",
}

REFERENCE_IMAGE_TEMPLATE = (
    "A professional, information-rich infographic {orientation} poster about: {idea}. "
    "Flat vector editorial design with a clear visual hierarchy: a headline, several "
    "labelled sections, at least one chart, and short blocks of body text. "
    "Clean margins, consistent alignment, legible typography, no photographic imagery."
)


def load_client(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
) -> Any:
    from openai import OpenAI

    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        timeout=timeout,
    )


def build_image_prompt(idea: str, orientation: str) -> str:
    return REFERENCE_IMAGE_TEMPLATE.format(idea=idea.strip(), orientation=orientation)


def generate_reference_image(
    client: Any,
    model: str,
    prompt: str,
    out_path: Path,
    *,
    orientation: str = "landscape",
    size: Optional[str] = None,
) -> Path:
    """Render one reference image and write it to ``out_path``."""
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size or SIZE_BY_ORIENTATION.get(orientation, "1536x1024"),
        n=1,
    )
    item = response.data[0]

    payload = getattr(item, "b64_json", None)
    if payload:
        data = base64.b64decode(payload)
    else:
        url = getattr(item, "url", None)
        if not url:
            raise RuntimeError("Image response contained neither b64_json nor url.")
        import urllib.request

        with urllib.request.urlopen(url) as handle:
            data = handle.read()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path
