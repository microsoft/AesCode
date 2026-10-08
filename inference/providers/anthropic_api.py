"""Anthropic Messages API HTML generation provider.

    export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence


def load_client(*, api_key: Optional[str] = None, timeout: float = 600.0) -> Any:
    from anthropic import Anthropic

    return Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout)


def _image_block(data_url: str) -> dict:
    """Convert a data: URL into an Anthropic base64 image block."""
    header, _, payload = data_url.partition(",")
    media_type = "image/png"
    if header.startswith("data:") and ";" in header:
        media_type = header[len("data:") : header.index(";")] or media_type
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": payload},
    }


def call_html_generation(
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    *,
    max_tokens: int = 12000,
    temperature: float = 0.8,
    top_p: float = 0.95,
    image_data_urls: Sequence[str] = (),
) -> str:
    """Generate one HTML document. Images come before the text, in order."""
    content: List[Any] = [_image_block(url) for url in image_data_urls if url]
    content.append({"type": "text", "text": user_text})

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
