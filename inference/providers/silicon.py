"""Silicon Flow (硅基流动) OpenAI-compatible HTML generation provider."""

from __future__ import annotations

import os
from typing import Any, List, Optional


def _read_dotenv_key(env_key: str) -> str:
    """Parse a specific key from .env in the current working directory."""
    from pathlib import Path
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.exists():
        return ""
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        sep = "=" if "=" in line else (":" if ":" in line else "")
        if not sep:
            continue
        k, v = line.split(sep, 1)
        if k.strip() == env_key:
            return v.strip().strip('"').strip("'")
    return ""


def load_silicon_api_key() -> str:
    key = os.getenv("SILICON_API_KEY", "").strip() or _read_dotenv_key("SILICON_API_KEY")
    if key:
        return key
    raise ValueError(
        "Silicon Flow API key not found. Set SILICON_API_KEY in environment or .env file."
    )


def call_silicon_generation(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int,
    image_data_url: Optional[str] = None,
) -> str:
    """Call Silicon Flow via OpenAI-compatible chat completions endpoint."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    if image_data_url:
        user_content: Any = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Silicon Flow API returned empty content.")
    return content
