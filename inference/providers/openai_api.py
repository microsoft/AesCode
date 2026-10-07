"""OpenAI-compatible HTML generation provider.

Works with the OpenAI API and with any server that speaks the same chat
completions schema, including a local vLLM server hosting AesCode:

    vllm serve microsoft/AesCode-32B --port 8000 --limit-mm-per-prompt image=2

    export AESCODE_BASE_URL=http://127.0.0.1:8000/v1
    export AESCODE_API_KEY=EMPTY
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence


def load_client(*, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 600.0) -> Any:
    from openai import OpenAI

    return OpenAI(
        base_url=base_url or os.environ.get("AESCODE_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        api_key=api_key or os.environ.get("AESCODE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        timeout=timeout,
    )


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
    content: List[Any] = [
        {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls if url
    ]
    content.append({"type": "text", "text": user_text})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return (response.choices[0].message.content or "").strip()
