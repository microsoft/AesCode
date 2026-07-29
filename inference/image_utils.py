"""Image + API-key I/O utilities for the inference pipeline.

Self-contained helpers used by inference/batch.py: load an Azure OpenAI key
from env/.env, encode a reference image as a size-bounded data URL, and resolve
a sample's reference-image path from a dataset row.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional


def _parse_dotenv_key(dotenv_path: Path) -> str:
    if not dotenv_path.exists():
        return ""
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        separator = "=" if "=" in line else ":" if ":" in line else ""
        if not separator:
            continue
        key, value = line.split(separator, 1)
        name = key.strip()
        if name not in {"API_KEY", "AZURE_OPENAI_API_KEY"}:
            continue
        text = value.strip().strip('"').strip("'")
        if text:
            return text
    return ""


def load_api_key(api_key_path: Optional[Path]) -> str:
    env_key = os.getenv("AZURE_OPENAI_API_KEY")
    if env_key:
        return env_key
    generic_env_key = os.getenv("API_KEY")
    if generic_env_key:
        return generic_env_key

    dotenv_key = _parse_dotenv_key(Path.cwd() / ".env")
    if dotenv_key:
        return dotenv_key

    if api_key_path and api_key_path.exists():
        data = json.loads(api_key_path.read_text(encoding="utf-8"))
        key = data.get("api_key", "").strip()
        if key:
            return key
    raise ValueError(
        "API key not found. Please set AZURE_OPENAI_API_KEY, API_KEY, add API_KEY to .env, or provide --api-key-path."
    )


def image_to_data_url(image_path: Path, max_long_side: int = 1120) -> str:
    """Encode image as data URL, resizing to max_long_side to reduce payload size."""
    mime_type = "image/png"
    raw: bytes
    if max_long_side and max_long_side > 0:
        try:
            import io as _io
            from PIL import Image as _PIL
            img = _PIL.open(image_path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_long_side:
                if w >= h:
                    new_w, new_h = max_long_side, max(1, round(h * max_long_side / w))
                else:
                    new_w, new_h = max(1, round(w * max_long_side / h)), max_long_side
                img = img.resize((new_w, new_h), _PIL.LANCZOS)
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=90, optimize=True)
            raw = buf.getvalue()
            mime_type = "image/jpeg"
        except Exception:  # PIL unavailable or corrupt image — fall back
            raw = image_path.read_bytes()
    else:
        raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def resolve_image_path(item: Dict, root_dir: Path, image_dir: Path) -> Path:
    ref = item.get("reference_image", "")
    image_id = item.get("id", "")
    candidates = []
    if ref:
        ref_path = Path(ref)
        candidates.append(root_dir / ref_path)
        candidates.append(image_dir / ref_path.name)
    if image_id:
        candidates.append(image_dir / f"{image_id}.png")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot locate image for id={image_id}, reference_image={ref}. "
        f"Tried: {[str(p) for p in candidates]}"
    )
