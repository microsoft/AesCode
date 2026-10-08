"""I/O utilities: image encoding, reference-dimension helpers, and JSONL loading."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

def image_to_data_url(image_path: Path, max_long_side: int = 1120) -> str:
    """Encode image as data URL, resizing to max_long_side before encoding to reduce payload size."""
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


def parse_reference_dimensions(reference_image_wh: str) -> Optional[tuple]:
    if not isinstance(reference_image_wh, str):
        return None
    text = reference_image_wh.strip().lower()
    if not text or "x" not in text:
        return None
    width_str, height_str = text.split("x", 1)
    try:
        width = int(width_str.strip())
        height = int(height_str.strip())
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def get_reference_aspect_ratio(reference_image_wh: str) -> str:
    dims = parse_reference_dimensions(reference_image_wh)
    if not dims:
        return ""
    width, height = dims
    ratio = width / height
    return f"{ratio:.4f}".rstrip("0").rstrip(".")


def get_reference_orientation(reference_image_wh: str) -> str:
    dims = parse_reference_dimensions(reference_image_wh)
    if not dims:
        return ""
    width, height = dims
    if width > height:
        return "landscape"
    if width < height:
        return "portrait"
    return "square"


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSONL at line {line_num}: {err}") from err
    return rows


def load_existing_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" in obj:
                ids.add(obj["id"])
    return ids
