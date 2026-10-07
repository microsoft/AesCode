"""Compatibility helpers kept intentionally minimal.

This module used to rewrite requirement payloads aggressively. That behavior has
been removed so SFT data and refined samples preserve the model's original
structure as much as possible.
"""

from __future__ import annotations

from typing import Any, List


def normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        normalized: List[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    normalized.append(text)
        return normalized
    return []
