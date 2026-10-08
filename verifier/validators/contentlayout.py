"""Content/Layout validators — TableChartValidator, TextValidator, and SemanticLayoutValidator.

Content/Layout layer. Runs after the execution gate.
These validators answer: Is the right content present in the right place
in the right quantity, and is the text layout correct?
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from verifier.base import BaseValidator
from verifier.types import ArtifactBundle, Requirement, ValidationResult, ValidationStatus
from verifier.artifacts.metrics import _clip_crop_bounds


def _safe_imread(path: str | Path, flag: int = cv2.IMREAD_COLOR):
    """Read image via PIL first (thread-safe libpng16), fallback to cv2.

    Uses LOAD_TRUNCATED_IMAGES to handle partially-written Playwright screenshots.
    """
    from PIL import Image, ImageFile
    try:
        _orig = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img = Image.open(str(path))
            img.load()
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = _orig
        if flag == cv2.IMREAD_GRAYSCALE:
            return np.asarray(img.convert("L"))
        else:
            rgb = np.asarray(img.convert("RGB"))
            return rgb[:, :, ::-1].copy()  # RGB→BGR
    except Exception:
        return cv2.imread(str(path), flag)





def _extract_balanced_segment(text: str, start_index: int, open_char: str, close_char: str) -> Optional[str]:
    if start_index < 0 or start_index >= len(text) or text[start_index] != open_char:
        return None
    depth = 0
    quote_char = ""
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
        if quote_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                quote_char = ""
            continue
        if char in {"\"", "'"}:
            quote_char = char
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    return None


def _parse_js_like_json(text: str) -> Any:
    import json
    import re

    normalized = text.strip()
    normalized = re.sub(r",(\s*[}\]])", r"\1", normalized)
    normalized = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', normalized)
    normalized = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", lambda match: json.dumps(match.group(1)), normalized)
    return json.loads(normalized)


def _normalize_chart_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_chart_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_chart_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_table_cell(value: Any) -> str:
    normalized = _normalize_chart_value(value)
    if normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    return " ".join(str(normalized).split())


def _normalize_table_rows(rows: Any) -> list[list[str]]:
    if not isinstance(rows, list):
        return []
    normalized_rows: list[list[str]] = []
    for row in rows:
        if not isinstance(row, list):
            return []
        normalized_rows.append([_normalize_table_cell(cell) for cell in row])
    return normalized_rows


def _extract_chart_data_candidates(html_source: str) -> list[Dict[str, Any]]:
    import re
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_source or "", "html.parser")
    candidates: list[Dict[str, Any]] = []
    for script in soup.select("script"):
        script_text = script.string or script.get_text("\n", strip=False)
        if not script_text or "data" not in script_text:
            continue
        # Pattern 1: object property  data: [...]
        for match in re.finditer(r"\bdata\s*:\s*\[", script_text):
            array_start = script_text.find("[", match.start())
            if array_start == -1:
                continue
            array_text = _extract_balanced_segment(script_text, array_start, "[", "]")
            if not array_text:
                continue
            try:
                parsed = _parse_js_like_json(array_text)
            except Exception:  # noqa: BLE001
                continue
            series_name = None
            prefix = script_text[max(0, match.start() - 200) : match.start()]
            name_matches = re.findall(r"\bname\s*:\s*['\"]([^'\"]+)['\"]", prefix)
            if name_matches:
                series_name = name_matches[-1].strip()
            candidates.append(
                {
                    "series_name": series_name,
                    "data": _normalize_chart_value(parsed),
                    "script_preview": script_text[max(0, match.start() - 80) : min(len(script_text), match.start() + 160)].strip(),
                }
            )
        # Pattern 2: variable assignment  const/let/var name = [...]
        for match in re.finditer(r"(?:const|let|var)\s+(\w+)\s*=\s*\[", script_text):
            array_start = script_text.find("[", match.start())
            if array_start == -1:
                continue
            array_text = _extract_balanced_segment(script_text, array_start, "[", "]")
            if not array_text:
                continue
            try:
                parsed = _parse_js_like_json(array_text)
            except Exception:  # noqa: BLE001
                continue
            var_name = match.group(1)
            candidates.append(
                {
                    "series_name": var_name,
                    "data": _normalize_chart_value(parsed),
                    "script_preview": script_text[max(0, match.start() - 20) : min(len(script_text), match.start() + 120)].strip(),
                }
            )
        # Pattern 3: ECharts dataset.source  source: [[...]]
        for match in re.finditer(r"\bsource\s*:\s*\[", script_text):
            array_start = script_text.find("[", match.start())
            if array_start == -1:
                continue
            array_text = _extract_balanced_segment(script_text, array_start, "[", "]")
            if not array_text:
                continue
            try:
                parsed = _parse_js_like_json(array_text)
            except Exception:  # noqa: BLE001
                continue
            # Flatten outer list if it's a 2-D source (list of row arrays)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                for row in parsed:
                    candidates.append(
                        {
                            "series_name": None,
                            "data": _normalize_chart_value(row),
                            "script_preview": f"dataset.source row: {str(row)[:80]}",
                        }
                    )
            else:
                candidates.append(
                    {
                        "series_name": None,
                        "data": _normalize_chart_value(parsed),
                        "script_preview": script_text[max(0, match.start() - 20) : min(len(script_text), match.start() + 120)].strip(),
                    }
                )
    return candidates


def _extract_chart_scalar_values(data: list) -> list | None:
    """Extract scalar values from enriched ECharts data items.

    ECharts data items are often enriched with extra keys::

        [{value: 14, itemStyle: {...}, name: '...'}, ...]

    When the spec expected list is flat scalars, this helper strips the extra
    metadata so the comparison can still succeed.  Returns ``None`` if the data
    does not look like enriched objects (nothing to extract).
    """
    if not data or not isinstance(data, list):
        return None
    result = []
    any_extracted = False
    for item in data:
        if isinstance(item, dict) and "value" in item:
            v = item["value"]
            if isinstance(v, (int, float, str)):
                result.append(v)
                any_extracted = True
                continue
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float, str)):
                result.append(v[0])
                any_extracted = True
                continue
        result.append(item)
    return result if any_extracted else None


def _extract_table_data_candidates(html_source: str) -> list[Dict[str, Any]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_source or "", "html.parser")
    candidates: list[Dict[str, Any]] = []
    for table in soup.select("table"):
        rows: list[list[str]] = []
        for row in table.select("tr"):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                cells = row.find_all(["th", "td"])
            row_values = [" ".join(cell.get_text(" ", strip=True).split()) for cell in cells]
            if row_values:
                rows.append(row_values)
        if rows:
            candidates.append(
                {
                    "data": rows,
                    "text_preview": " | ".join(" / ".join(row) for row in rows[:3]),
                }
            )
    return candidates


_DECORATIVE_CHARS_RE = None


def _normalize_cell_for_comparison(cell: str) -> str:
    """Normalize a table cell value for fuzzy comparison.

    Strips decorative Unicode symbols (▼▲●○★☆►◄ etc.), collapses whitespace,
    lowercases, and strips leading/trailing space.
    """
    import re
    global _DECORATIVE_CHARS_RE
    if _DECORATIVE_CHARS_RE is None:
        _DECORATIVE_CHARS_RE = re.compile(
            r"[\u25a0-\u25ff\u2600-\u26ff\u2700-\u27bf\u2190-\u21ff"
            r"\uf000-\uf0ff\u2b00-\u2bff]+"
        )
    text = _DECORATIVE_CHARS_RE.sub("", cell)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _cell_value_matches(expected_cell: str, candidate_cell: str) -> bool:
    """Check if an expected cell value matches a candidate cell value."""
    import re as _re
    e = _normalize_cell_for_comparison(expected_cell)
    c = _normalize_cell_for_comparison(candidate_cell)
    if not e:
        return True  # empty expected cell always matches
    if not c:
        return False
    if e == c:
        return True
    # For numeric-looking values, require exact match — substring containment
    # causes false positives ("1" in "18", "3" in "30", etc.)
    e_alnum = _re.sub(r"[^a-z0-9]", "", e)
    if e_alnum.isdigit():
        return False
    # Containment on alphanumeric-only form (strips punctuation differences
    # like "Licensing Trans." vs "Licensing Transactions")
    c_alnum = _re.sub(r"[^a-z0-9]", "", c)
    if e_alnum and c_alnum and (e_alnum in c_alnum or c_alnum in e_alnum):
        return True
    return False


def _table_row_value_match(
    expected_row: list[str],
    candidate_row: list[str],
) -> tuple[bool, int, int]:
    """Check if all non-empty expected cell values appear in candidate_row.

    Order-independent: each expected cell is matched greedily to the best
    unmatched candidate cell.  Returns (matched, found_count, expected_count).
    """
    non_empty = [cell for cell in expected_row if cell.strip()]
    if not non_empty:
        return True, 0, 0
    used: set[int] = set()
    found = 0
    for exp_cell in non_empty:
        for idx, cand_cell in enumerate(candidate_row):
            if idx in used:
                continue
            if _cell_value_matches(exp_cell, cand_cell):
                found += 1
                used.add(idx)
                break
    return found >= len(non_empty), found, len(non_empty)


def _find_best_table_match(
    expected_rows: list[list[str]],
    candidates: list[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
    """Find the best candidate table and compute row coverage score."""
    best_candidate = None
    best_score = 0.0
    best_details: Dict[str, Any] = {}

    for candidate in candidates:
        candidate_rows = candidate.get("data")
        if not isinstance(candidate_rows, list) or not candidate_rows:
            continue

        matched_row_count = 0
        row_details: list[Dict[str, Any]] = []
        used_candidate_rows: set[int] = set()

        for exp_row in expected_rows:
            best_row_idx = -1
            best_found = 0
            best_total = 0
            for c_idx, c_row in enumerate(candidate_rows):
                if c_idx in used_candidate_rows:
                    continue
                ok, found_ct, total_ct = _table_row_value_match(exp_row, c_row)
                if ok and found_ct > best_found:
                    best_found = found_ct
                    best_total = total_ct
                    best_row_idx = c_idx
            if best_row_idx >= 0:
                matched_row_count += 1
                used_candidate_rows.add(best_row_idx)
                row_details.append({"matched": True, "candidate_row_idx": best_row_idx, "cells_found": best_found, "cells_expected": best_total})
            else:
                row_details.append({"matched": False})

        score = matched_row_count / max(len(expected_rows), 1)
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_details = {
                "matched_row_count": matched_row_count,
                "expected_row_count": len(expected_rows),
                "candidate_row_count": len(candidate_rows),
                "row_details": row_details,
            }

    return best_candidate, best_score, best_details


# ---------------------------------------------------------------------------
# TextValidator
# ---------------------------------------------------------------------------


def _normalize_cjk_basic(text: str) -> str:
    """Normalize full-width punctuation and quotes to ASCII equivalents.
    Preserves internal whitespace — only punctuation/symbol-level normalisation
    is applied so that exact sentence matching remains strict."""
    import unicodedata

    # NFKC normalizes full-width letters/digits, some punctuation
    text = unicodedata.normalize("NFKC", text)
    # Normalize CJK quotes, brackets, and common typographic variants
    _CJK_PUNCT_MAP = str.maketrans({
        # Quotes – all curly/smart/CJK variants → ASCII straight
        "\u2018": "'", "\u2019": "'", "\u201a": "'",
        "\u201b": "'", "\u2039": "'", "\u203a": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"',
        "\u201f": '"', "\u00ab": '"', "\u00bb": '"',
        # CJK brackets
        "\u3008": "<", "\u3009": ">", "\u300a": "<", "\u300b": ">",
        "\u300c": "[", "\u300d": "]", "\u300e": "[", "\u300f": "]",
        "\u3010": "[", "\u3011": "]",
        "\uff08": "(", "\uff09": ")",
        # Dashes / hyphens – all variants → ASCII hyphen
        "\u2014": "-", "\u2013": "-", "\u2012": "-",
        "\u2010": "-", "\u2011": "-", "\u2015": "-",
        "\ufe58": "-", "\ufe63": "-", "\uff0d": "-",
        # Dots
        "\u00b7": ".", "\u2022": ".", "\u2026": "...",
        # CJK punctuation
        "\uff0c": ",", "\u3001": ",",
        "\uff1a": ":", "\uff1b": ";",
        "\uff01": "!", "\uff1f": "?",
        "\u3002": ".",
    })
    text = text.translate(_CJK_PUNCT_MAP)
    return text.strip()


def _normalize_text_for_matching(text: str, normalization: str = "cjk_basic") -> str:
    """Apply the declared normalization mode to text."""
    if normalization == "none":
        return text
    if normalization == "trim":
        return text.strip()
    # Default: cjk_basic
    return _normalize_cjk_basic(text)


def _visible_text_for_matching(artifacts) -> str:
    """Visible page text, preferring the rendered DOM over the raw markup.

    Client-rendered pages keep all of their copy inside <script type="text/babel">
    (or a bundler entry point), and the static parse below strips <script>, so a
    markup-only reading of such a page is empty even when it paints correctly.
    ``bundle.text_blocks`` is captured from the live DOM after Playwright settles,
    so it is the faithful source whenever the page rendered at all.
    """
    blocks = getattr(artifacts, "text_blocks", None) or []
    parts = []
    for tb in blocks:
        if isinstance(tb, dict):
            t = tb.get("text") or tb.get("content") or ""
        else:
            t = getattr(tb, "text", "") or ""
        t = str(t).strip()
        if t:
            parts.append(t)
    rendered = " ".join(parts).strip()
    if rendered:
        return rendered

    html_source = getattr(artifacts, "html_source", "") or ""
    if not html_source.strip():
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_source, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


class TextValidator(BaseValidator):
    """Verify that a specific required text string appears in the HTML output.

    Algorithm: extract all visible text nodes from the generated HTML DOM.
    Check each text_exact string declared in verifiable_requirements one by one.
    Pure content-existence check — completely agnostic to where the text is
    rendered on the page.

    Each text_exact requirement declares an expected.text that must appear
    verbatim (after normalization) somewhere in the rendered HTML text content.
    """

    name = "TextValidator"
    supported_categories = ("text_exact",)

    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        expected = requirement.expected or {}
        expected_text = str(expected.get("text") or "").strip()
        if not expected_text:
            return self.build_result(
                requirement,
                status=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
                message="No expected text declared.",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
            )

        normalization = str(expected.get("normalization") or "cjk_basic")
        case_sensitive = bool(expected.get("case_sensitive", True))

        html_source = artifacts.html_source or ""
        if not html_source.strip():
            return self.build_result(
                requirement,
                status=ValidationStatus.FAIL,
                score=0.0,
                message=f"Text exact check failed: empty HTML source. Expected: {expected_text[:60]}",
                evidence={"evidence_basis": "explicit", "evidence_sources": ["html_source"]},
            )

        # Prefer the rendered DOM; client-rendered pages carry no copy in markup.
        visible_text = _visible_text_for_matching(artifacts)

        # Normalize both sides
        norm_expected = _normalize_text_for_matching(expected_text, normalization)
        norm_visible = _normalize_text_for_matching(visible_text, normalization)

        if not case_sensitive:
            norm_expected = norm_expected.lower()
            norm_visible = norm_visible.lower()

        found = norm_expected in norm_visible

        # Text overlap penalty: penalize if text elements overlap on canvas
        overlap_penalty = _text_overlap_penalty(artifacts)
        # Text visibility: penalize if text is outside canvas
        text_vis = _text_visibility(artifacts)

        status = ValidationStatus.PASS if found else ValidationStatus.FAIL
        score = (1.0 if found else 0.0) * overlap_penalty * text_vis

        return self.build_result(
            requirement,
            status=status,
            score=score,
            measured={
                "expected_text": expected_text[:120],
                "found_exact": found,
                "normalization": normalization,
                "case_sensitive": case_sensitive,
                "text_overlap_penalty": round(overlap_penalty, 4),
                "text_visibility": round(text_vis, 4),
            },
            message=(
                f"Text exact: {'found' if found else 'NOT found'} — \"{expected_text[:60]}\""
            ),
            evidence={
                "evidence_basis": "explicit",
                "evidence_sources": ["html_source"],
            },
        )


def _text_overlap_penalty(artifacts) -> float:
    """Compute a penalty for overlapping text elements on the canvas.

    Uses the sum-vs-union approach: if no boxes overlap, union == sum_of_areas.
    overlap_ratio = 1 - union / sum_of_areas.  penalty = 1 - overlap_ratio.

    Returns a multiplier in [0, 1]:  1.0 = no overlap, 0.0 = total overlap.
    Returns 1.0 if no text_bboxes data (backward compatible).
    """
    snap = getattr(artifacts, "dom_snapshot", None) or {}
    boxes = snap.get("text_bboxes")
    if not boxes or len(boxes) < 2:
        return 1.0

    # Filter to boxes with positive area
    rects = []
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        if w > 0 and h > 0:
            rects.append((x, y, x + w, y + h))
    if len(rects) < 2:
        return 1.0

    sum_of_areas = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects)
    if sum_of_areas < 1e-6:
        return 1.0

    # Compute union area via coordinate compression (sweep line)
    # Collect all y-coordinates
    ys = sorted(set(y for r in rects for y in (r[1], r[3])))
    union_area = 0.0
    for i in range(len(ys) - 1):
        y_lo, y_hi = ys[i], ys[i + 1]
        band_h = y_hi - y_lo
        if band_h <= 0:
            continue
        # Collect x-intervals that overlap this y-band
        intervals = []
        for x1, y1, x2, y2 in rects:
            if y1 < y_hi and y2 > y_lo:  # rect overlaps this band
                intervals.append((x1, x2))
        # Merge x-intervals
        intervals.sort()
        merged_width = 0.0
        cur_start, cur_end = -1e18, -1e18
        for a, b in intervals:
            if a > cur_end:
                merged_width += max(0, cur_end - cur_start)
                cur_start, cur_end = a, b
            else:
                cur_end = max(cur_end, b)
        merged_width += max(0, cur_end - cur_start)
        union_area += merged_width * band_h

    overlap_ratio = 1.0 - min(union_area / sum_of_areas, 1.0)
    return 1.0 - overlap_ratio  # = union / sum_of_areas


def _text_visibility(artifacts) -> float:
    """Compute average visibility of text elements within the canvas.

    Same logic as _chart_table_visibility but for text_bboxes.
    Returns a multiplier in [0, 1]:  1.0 = all text inside canvas, 0.0 = all outside.
    Returns 1.0 if no data (backward compatible).
    """
    snap = getattr(artifacts, "dom_snapshot", None) or {}
    boxes = snap.get("text_bboxes")
    pw = snap.get("poster_width", 0)
    ph = snap.get("poster_height", 0)
    if not boxes or not pw or not ph:
        return 1.0

    total_vis = 0.0
    count = 0
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        area = w * h
        if area < 1e-6:
            continue
        ix1 = max(x, 0)
        iy1 = max(y, 0)
        ix2 = min(x + w, pw)
        iy2 = min(y + h, ph)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        total_vis += inter / area
        count += 1

    return total_vis / count if count else 1.0


def _chart_table_visibility(artifacts, element_type: str) -> float:
    """Compute average visibility score for chart or table elements within the canvas.

    Args:
        artifacts: ArtifactBundle with dom_snapshot containing element_bboxes
        element_type: "chart" (matches "chart:" and "chart_canvas:" keys) or "table"

    Returns:
        Average fraction of element area visible inside the poster canvas [0, 1].
        1.0 = all elements fully inside canvas, 0.0 = all fully outside.
        Returns 1.0 if no bbox data available (backward compatible — don't penalize
        when we can't measure, e.g. static backend without Playwright).
    """
    snap = getattr(artifacts, "dom_snapshot", None) or {}
    bboxes = snap.get("element_bboxes", {})
    pw = float(snap.get("poster_width") or 0)
    ph = float(snap.get("poster_height") or 0)
    if not bboxes or pw <= 0 or ph <= 0:
        return 1.0

    vis_scores = []
    for key, bbox in bboxes.items():
        if not key.startswith(element_type):
            continue
        bw = float(bbox.get("width") or 0)
        bh = float(bbox.get("height") or 0)
        # A collapsed element (0 width/height) is not visible: the chart/table
        # exists in the DOM — the echarts runtime may even hold its data — but
        # renders to nothing on screen. Score it 0 rather than skipping it;
        # skipping left vis_scores empty, which fell through to the "no elements
        # → 1.0" return and handed collapsed charts a perfect visibility score.
        if bw <= 0 or bh <= 0:
            vis_scores.append(0.0)
            continue
        bx = float(bbox.get("x") or 0)
        by = float(bbox.get("y") or 0)
        # Intersection with canvas [0, 0, pw, ph]
        ix1 = max(bx, 0.0)
        iy1 = max(by, 0.0)
        ix2 = min(bx + bw, pw)
        iy2 = min(by + bh, ph)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = bw * bh
        vis_scores.append(inter / max(area, 1e-6))

    if not vis_scores:
        return 1.0  # no matching elements found → don't penalize
    return sum(vis_scores) / len(vis_scores)


class TableChartValidator(BaseValidator):
    """Unified validator for HTML table data and ECharts chart data/type.

    Integrates the former TableDataValidator and ChartDataValidator into a
    single validator.  For ECharts charts, the validator not only checks the
    data subset via DOM / runtime probes, but also strictly validates that
    ``series.type`` matches the spec-required ``chart_subtype`` (e.g. pie,
    bar, line) to prevent hallucinated chart types.
    """

    name = "TableChartValidator"
    supported_categories = ("table_data", "chart_data")

    # ------------------------------------------------------------------
    # chart_subtype extraction helper
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_chart_series_types(html_source: str) -> list[str]:
        """Extract all ECharts series.type values from inline scripts."""
        import re as _re
        types: list[str] = []
        for match in _re.finditer(r"""\btype\s*:\s*['"]([a-zA-Z]+)['"]""", html_source):
            t = match.group(1).strip().lower()
            if t in {"pie", "bar", "line", "scatter", "radar", "heatmap",
                      "treemap", "sunburst", "funnel", "gauge", "sankey",
                      "boxplot", "candlestick", "parallel", "graph",
                      "themeRiver", "map", "effectScatter"}:
                types.append(t)
        return types

    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        html_source = artifacts.html_source or ""
        expected = requirement.expected or {}
        category = str(getattr(requirement.category, "value", requirement.category) or "")

        if not html_source.strip():
            return self.build_result(
                requirement,
                status=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
                message="Table/chart data validation skipped: html_source unavailable.",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
            )

        # ── Table path ────────────────────────────────────────────────
        if category == "table_data":
            # Tables may be built at runtime, so read the rendered DOM when the
            # render captured one; html_source alone misses client-rendered rows.
            table_html = (getattr(artifacts, "metadata", None) or {}).get("rendered_html") or html_source
            return self._validate_table(table_html, expected, requirement, artifacts)

        # ── Chart path ────────────────────────────────────────────────
        return self._validate_chart(html_source, expected, requirement, artifacts)

    # ------------------------------------------------------------------
    # Table validation (from former TableDataValidator)
    # ------------------------------------------------------------------
    def _validate_table(
        self, html_source: str, expected: Dict[str, Any], requirement: Requirement,
        artifacts=None,
    ) -> ValidationResult:
        expected_rows = _normalize_table_rows(expected.get("data"))

        if not expected_rows:
            return self.build_result(
                requirement,
                status=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
                message="Table data validation skipped: expected.data must be a non-empty row-array matrix.",
                evidence={"evidence_basis": "unknown", "evidence_sources": ["html_source"]},
            )

        candidates = _extract_table_data_candidates(html_source)
        best_candidate, score, details = _find_best_table_match(expected_rows, candidates)

        # Visibility penalty: table must be inside the canvas area
        visibility = _chart_table_visibility(artifacts, "table") if artifacts else 1.0
        score = score * visibility

        measured: Dict[str, Any] = {
            "candidate_count": len(candidates),
            "score": round(score, 4),
            **details,
        }

        if score >= 1.0:
            message = "All expected table rows matched in the generated HTML table."
        elif best_candidate is not None:
            message = f"Table row coverage: {details.get('matched_row_count', 0)}/{details.get('expected_row_count', 0)} expected rows matched."
        else:
            message = "No visible HTML table matched any expected table rows."
            measured["candidate_previews"] = [c.get("text_preview") for c in candidates[:3]]

        return self.build_result(
            requirement,
            status=ValidationStatus.PASS,
            score=round(score, 4),
            measured=measured,
            message=message,
            evidence={"evidence_basis": "explicit", "evidence_sources": ["html_source"]},
        )

    # ------------------------------------------------------------------
    # Chart validation (from former ChartDataValidator + subtype check)
    # ------------------------------------------------------------------
    def _validate_chart(
        self, html_source: str, expected: Dict[str, Any], requirement: Requirement,
        artifacts=None,
    ) -> ValidationResult:
        expected_data = expected.get("data")
        expected_series_name = str(expected.get("series_name") or "").strip() or None
        expected_subtype = str(expected.get("chart_subtype") or "").strip().lower() or None

        if not isinstance(expected_data, list):
            return self.build_result(
                requirement,
                status=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
                message="Chart data validation skipped: expected.data must be a list.",
                evidence={"evidence_basis": "unknown", "evidence_sources": ["html_source"]},
            )

        # ── Try runtime data first (from Playwright ECharts extraction) ──
        runtime_series = []
        if artifacts and artifacts.dom_snapshot:
            runtime_series = artifacts.dom_snapshot.get("echarts_runtime_data") or []

        # ── subtype enforcement ───────────────────────────────────────
        subtype_ok = True
        detected_types: list[str] = []
        if expected_subtype:
            if runtime_series:
                detected_types = [s["type"] for s in runtime_series if s.get("type")]
            else:
                detected_types = self._extract_chart_series_types(html_source)
            if detected_types:
                subtype_ok = expected_subtype in detected_types

        # ── data matching ─────────────────────────────────────────────
        normalized_expected = _normalize_chart_value(expected_data)
        matched_candidate: Optional[Dict[str, Any]] = None
        match_note: str = ""
        data_source: str = ""

        # Priority 1: runtime data from Playwright
        if runtime_series:
            data_source = "echarts_runtime"
            for series in runtime_series:
                candidate_name = series.get("name")
                if expected_series_name and candidate_name and candidate_name != expected_series_name:
                    continue
                candidate_data = series.get("data")
                if candidate_data is None:
                    continue
                # Exact array match
                if _normalize_chart_value(candidate_data) == normalized_expected:
                    matched_candidate = series
                    break
                # Fallback: enriched items flattened to scalar 'value'
                if isinstance(candidate_data, list):
                    flat = _extract_chart_scalar_values(candidate_data)
                    if flat is not None and _normalize_chart_value(flat) == normalized_expected:
                        matched_candidate = series
                        match_note = "enriched_value_extracted"
                        break

        # Priority 2: regex extraction from HTML source (fallback)
        if matched_candidate is None:
            data_source = "html_regex" if not data_source else data_source + "+html_regex"
            candidates = _extract_chart_data_candidates(html_source)
            for candidate in candidates:
                candidate_name = candidate.get("series_name")
                if expected_series_name and candidate_name and candidate_name != expected_series_name:
                    continue
                candidate_data = candidate.get("data")
                if candidate_data == normalized_expected:
                    matched_candidate = candidate
                    break
                if isinstance(candidate_data, list):
                    flat = _extract_chart_scalar_values(candidate_data)
                    if flat is not None and _normalize_chart_value(flat) == normalized_expected:
                        matched_candidate = candidate
                        match_note = "enriched_value_extracted"
                        break

        # ── combined scoring ──────────────────────────────────────────
        data_score = 1.0 if matched_candidate is not None else 0.0
        subtype_score = 1.0 if subtype_ok else 0.0

        if expected_subtype:
            final_score = round(0.6 * data_score + 0.4 * subtype_score, 4)
        else:
            final_score = data_score

        # Visibility penalty: chart must be inside the canvas area
        visibility = _chart_table_visibility(artifacts, "chart") if artifacts else 1.0
        final_score = final_score * visibility

        measured: Dict[str, Any] = {
            "data_score": data_score,
            "subtype_score": subtype_score,
            "visibility": round(visibility, 4),
            "data_source": data_source,
            "runtime_series_count": len(runtime_series),
        }
        if matched_candidate is not None:
            measured["matched_series_name"] = matched_candidate.get("series_name") or matched_candidate.get("name")
            measured["matched_data_length"] = len(matched_candidate.get("data") or [])
            if match_note:
                measured["match_note"] = match_note
        else:
            measured["expected_series_name"] = expected_series_name
            measured["expected_row_count"] = len(normalized_expected)
            candidates_for_preview = _extract_chart_data_candidates(html_source) if not runtime_series else []
            measured["candidate_previews"] = [c.get("script_preview") for c in candidates_for_preview[:3]]

        if expected_subtype:
            measured["expected_chart_subtype"] = expected_subtype
            measured["detected_chart_types"] = detected_types
            measured["subtype_match"] = subtype_ok

        message = (
            f"Chart validation: data={'matched' if data_score else 'NOT matched'}"
            f" (source={data_source})"
            + (f", subtype={'OK' if subtype_ok else 'MISMATCH (' + ','.join(detected_types[:3]) + ')'}"
               if expected_subtype else "")
            + "."
        )

        return self.build_result(
            requirement,
            status=ValidationStatus.PASS,
            score=final_score,
            measured=measured,
            message=message,
            evidence={"evidence_basis": "explicit", "evidence_sources": ["html_source", "echarts_runtime"]},
        )


import math
try:
    from rapidfuzz.distance import Indel as _Indel
    from rapidfuzz import process as _rfprocess, fuzz as _rfuzz
    def _ratio(a: str, b: str) -> float:
        return _rfuzz.ratio(a, b) / 100.0
except ImportError:
    from difflib import SequenceMatcher
    _rfprocess = None
    def _ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return SequenceMatcher(None, a, b).ratio()
import hashlib
import json as _json
import logging
import os
import threading
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

from bs4 import BeautifulSoup

from verifier.base import BaseValidator
from verifier.artifacts.paths import get_layout_overlay_artifact_path
from verifier.types import ArtifactBundle, Requirement, ValidationResult, ValidationStatus

_logger = logging.getLogger(__name__)

_PADDLE_OCR_CACHE: Dict[str, Any] = {}
_PADDLE_OCR_LOCK = threading.Lock()  # PaddlePaddle C++ predictor is NOT thread-safe

# ---------------------------------------------------------------------------
# Reference OCR disk cache — avoids re-running PaddleOCR on the same
# reference image across verify runs.  OCR is 100% deterministic for a given
# image, so caching is safe.
# ---------------------------------------------------------------------------

_OCR_DISK_CACHE_DIR: Optional[Path] = None


def _get_ocr_cache_dir() -> Optional[Path]:
    """Return the OCR disk-cache directory, or None if caching is disabled."""
    global _OCR_DISK_CACHE_DIR
    if _OCR_DISK_CACHE_DIR is not None:
        return _OCR_DISK_CACHE_DIR if _OCR_DISK_CACHE_DIR != Path("/dev/null") else None

    env_val = os.getenv("VERIFIER_OCR_CACHE_DIR", "").strip()
    if env_val.lower() in ("0", "false", "off", "none", ""):
        # Default: use ~/.cache/aescode_ocr
        _OCR_DISK_CACHE_DIR = Path.home() / ".cache" / "aescode_ocr"
    elif env_val.lower() == "disable":
        _OCR_DISK_CACHE_DIR = Path("/dev/null")  # sentinel for disabled
        return None
    else:
        _OCR_DISK_CACHE_DIR = Path(env_val)

    _OCR_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _OCR_DISK_CACHE_DIR


def _ocr_cache_key(image_path: str) -> str:
    """Compute a stable cache key from the resolved image file path."""
    resolved = str(Path(image_path).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def _ocr_cache_path(image_path: str) -> Optional[Path]:
    """Return the cache file path for a given image, or None if caching disabled."""
    cache_dir = _get_ocr_cache_dir()
    if cache_dir is None:
        return None
    key = _ocr_cache_key(image_path)
    # Use 2-char prefix subdirectory to avoid too many files in one dir
    subdir = cache_dir / key[:2]
    return subdir / f"{key}.json"


def _ocr_cache_get(image_path: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Try to load OCR results from disk cache. Returns (blocks, debug) or None."""
    cache_file = _ocr_cache_path(image_path)
    if cache_file is None or not cache_file.exists():
        return None
    try:
        data = _json.loads(cache_file.read_text(encoding="utf-8"))
        blocks = data.get("blocks", [])
        debug = data.get("debug", {})
        debug["cache_hit"] = True
        return blocks, debug
    except Exception:
        # Corrupted cache file — delete and re-run
        cache_file.unlink(missing_ok=True)
        return None


def _ocr_cache_put(image_path: str, blocks: List[Dict[str, Any]], debug: Dict[str, Any]) -> None:
    """Save OCR results to disk cache. Thread-safe (atomic write).

    NOTE: Empty results (blocks=[]) are intentionally NOT cached, because they
    are usually caused by transient PaddleOCR primitive failures rather than
    a true "no text in image" outcome. Caching empty would permanently mask
    layout signal for those samples on subsequent runs.
    """
    if not blocks:
        return
    cache_file = _ocr_cache_path(image_path)
    if cache_file is None:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp file + rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(cache_file.parent), suffix=".tmp", prefix=".ocr_cache_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                _json.dump({"blocks": blocks, "debug": debug}, f, ensure_ascii=False)
            os.replace(tmp_path, str(cache_file))
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    except Exception as exc:
        _logger.debug("OCR cache write failed for %s: %s", image_path, exc)


# ---------------------------------------------------------------------------
# Sentence-embedding text similarity (CPU, ~90MB model)
# ---------------------------------------------------------------------------
_SBERT_MODEL: Any = None
_SBERT_MODEL_LOADED: bool = False
_sbert_lock = threading.Lock()


def _get_sbert_model() -> Any:
    """Return the cached SentenceTransformer model, or *None* if unavailable."""
    global _SBERT_MODEL, _SBERT_MODEL_LOADED
    if _SBERT_MODEL_LOADED:
        return _SBERT_MODEL
    with _sbert_lock:
        if _SBERT_MODEL_LOADED:
            return _SBERT_MODEL
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _SBERT_MODEL = None
        _SBERT_MODEL_LOADED = True
    return _SBERT_MODEL


def _compute_text_similarity_matrix(
    ref_blocks: List[Dict[str, Any]],
    pred_blocks: List[Dict[str, Any]],
) -> "np.ndarray":
    """Return blended text similarity: 50% embedding + 50% fuzzy char ratio.

    The fuzzy char branch helps recover from OCR-rendered typos in the
    reference image, where semantic embeddings alone can drift too far.
    """
    import numpy as _np

    ref_texts = [str(b.get("normalized_text") or b.get("text") or "") for b in ref_blocks]
    pred_texts = [str(b.get("normalized_text") or b.get("text") or "") for b in pred_blocks]
    ref_compact_texts = [str(b.get("compact_text") or _compact_text_key(text)) for b, text in zip(ref_blocks, ref_texts)]
    pred_compact_texts = [str(b.get("compact_text") or _compact_text_key(text)) for b, text in zip(pred_blocks, pred_texts)]

    n, m = len(ref_blocks), len(pred_blocks)
    fuzzy_matrix = _np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            phrase_similarity = _ratio(ref_texts[i], pred_texts[j])
            compact_similarity = _ratio(ref_compact_texts[i], pred_compact_texts[j])
            fuzzy_matrix[i, j] = max(phrase_similarity, compact_similarity)

    model = _get_sbert_model()
    if model is not None:
        all_texts = ref_texts + pred_texts
        embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        ref_emb = embeddings[: len(ref_texts)]
        pred_emb = embeddings[len(ref_texts) :]
        embedding_matrix = _np.clip(ref_emb @ pred_emb.T, 0.0, 1.0).astype(float)
        return _np.clip(0.5 * embedding_matrix + 0.5 * fuzzy_matrix, 0.0, 1.0).astype(float)

    return fuzzy_matrix


def _ocr_preprocessing_enabled() -> bool:
    # Preprocessing (CLAHE + unsharp + adaptive threshold) is on by default.
    # Disable with DATASET_PIPELINE_OCR_PREPROCESS=off
    value = str(os.getenv("DATASET_PIPELINE_OCR_PREPROCESS", "on") or "on").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _prepare_image_for_ocr(image_path: str) -> tuple[str, Dict[str, Any], Optional[str]]:
    if not _ocr_preprocessing_enabled():
        return image_path, {"enabled": False, "reason": "disabled_by_env"}, None
    try:
        import cv2
    except ImportError as exc:
        return image_path, {"enabled": False, "error": f"opencv unavailable: {exc}"}, None

    image = _safe_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return image_path, {"enabled": False, "error": f"unable to read image: {image_path}"}, None

    original_height, original_width = image.shape[:2]
    scale_factor = 1.35 if max(original_width, original_height) < 1400 else 1.15
    resized = cv2.resize(
        image,
        (max(1, int(round(original_width * scale_factor))), max(1, int(round(original_height * scale_factor)))),
        interpolation=cv2.INTER_CUBIC,
    )

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    contrast_enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=1.1)
    sharpened = cv2.addWeighted(contrast_enhanced, 1.45, blurred, -0.45, 0)

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    hybrid = cv2.addWeighted(normalized, 0.72, binary, 0.28, 0)
    prepared_image = cv2.cvtColor(hybrid, cv2.COLOR_GRAY2BGR)

    temp_file = tempfile.NamedTemporaryFile(prefix="dataset_pipeline_ocr_", suffix=".png", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    cv2.imwrite(temp_path, prepared_image)
    return temp_path, {
        "enabled": True,
        "scale_factor": round(scale_factor, 2),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "prepared_width": int(prepared_image.shape[1]),
        "prepared_height": int(prepared_image.shape[0]),
        "pipeline": ["resize", "clahe_contrast", "unsharp_mask", "adaptive_threshold_blend"],
    }, temp_path


def _box_area(box: Dict[str, Any]) -> float:
    return max(float(box.get("width", 0.0) or 0.0), 0.0) * max(float(box.get("height", 0.0) or 0.0), 0.0)


def _box_center(box: Dict[str, Any]) -> Tuple[float, float]:
    return (
        float(box.get("cx", float(box.get("x", 0.0)) + float(box.get("width", 0.0)) / 2.0)),
        float(box.get("cy", float(box.get("y", 0.0)) + float(box.get("height", 0.0)) / 2.0)),
    )


def _size_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_w = max(float(left.get("width", 0.0)), 1e-6)
    left_h = max(float(left.get("height", 0.0)), 1e-6)
    right_w = max(float(right.get("width", 0.0)), 1e-6)
    right_h = max(float(right.get("height", 0.0)), 1e-6)
    width_ratio = min(left_w, right_w) / max(left_w, right_w)
    height_ratio = min(left_h, right_h) / max(left_h, right_h)
    left_area = max(_box_area(left), 1e-6)
    right_area = max(_box_area(right), 1e-6)
    area_ratio = min(left_area, right_area) / max(left_area, right_area)
    return max(0.0, min(1.0, (width_ratio + height_ratio + area_ratio) / 3.0))


def _normalized_box(name: str, x: float, y: float, width: float, height: float, page_width: float, page_height: float) -> Dict[str, Any]:
    normalized_x = x / page_width if page_width else 0.0
    normalized_y = y / page_height if page_height else 0.0
    normalized_w = width / page_width if page_width else 0.0
    normalized_h = height / page_height if page_height else 0.0
    normalized_x = max(0.0, min(1.0, normalized_x))
    normalized_y = max(0.0, min(1.0, normalized_y))
    normalized_right = max(normalized_x, min(1.0, normalized_x + normalized_w))
    normalized_bottom = max(normalized_y, min(1.0, normalized_y + normalized_h))
    normalized_w = max(0.0, normalized_right - normalized_x)
    normalized_h = max(0.0, normalized_bottom - normalized_y)
    return {
        "name": name,
        "x": round(normalized_x, 4),
        "y": round(normalized_y, 4),
        "width": round(normalized_w, 4),
        "height": round(normalized_h, 4),
        "right": round(normalized_right, 4),
        "bottom": round(normalized_bottom, 4),
        "cx": round(normalized_x + normalized_w / 2.0, 4),
        "cy": round(normalized_y + normalized_h / 2.0, 4),
        "bbox": [round(normalized_x, 4), round(normalized_y, 4), round(normalized_w, 4), round(normalized_h, 4)],
    }


def _normalized_box_from_unit(name: str, x: float, y: float, width: float, height: float) -> Dict[str, Any]:
    return {
        "name": name,
        "x": round(x, 4),
        "y": round(y, 4),
        "width": round(width, 4),
        "height": round(height, 4),
        "right": round(x + width, 4),
        "bottom": round(y + height, 4),
        "cx": round(x + width / 2.0, 4),
        "cy": round(y + height / 2.0, 4),
        "bbox": [round(x, 4), round(y, 4), round(width, 4), round(height, 4)],
    }


def _extract_blocks_from_image(image_path: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return [], {"error": f"opencv unavailable: {exc}"}

    image = _safe_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return [], {"error": f"unable to read image: {image_path}"}

    page_height, page_width = image.shape[:2]
    margin = max(2, int(min(page_width, page_height) * 0.04))
    border_pixels = np.concatenate(
        [
            image[:margin, :, :].reshape(-1, 3),
            image[-margin:, :, :].reshape(-1, 3),
            image[:, :margin, :].reshape(-1, 3),
            image[:, -margin:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels.astype(np.float32), axis=0)
    color_distance = np.linalg.norm(image.astype(np.float32) - background_color.reshape(1, 1, 3), axis=2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 45, 135)

    distance_threshold = max(22.0, float(np.percentile(color_distance, 78)))
    active_mask = ((color_distance >= distance_threshold) | (edges > 0)).astype("uint8") * 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, page_width // 55), max(3, page_height // 55)),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(2, page_width // 180), max(2, page_height // 180)),
    )
    active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_CLOSE, close_kernel)
    active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_OPEN, open_kernel)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(active_mask, connectivity=8)
    min_area = max(64, int(page_width * page_height * 0.0012))
    boxes: List[Dict[str, Any]] = []
    for component_index in range(1, component_count):
        x, y, width, height, area = stats[component_index]
        if area < min_area:
            continue
        if width >= page_width * 0.985 and height >= page_height * 0.985:
            continue
        boxes.append(
            _normalized_box(
                f"image_block_{len(boxes) + 1}",
                float(x),
                float(y),
                float(width),
                float(height),
                float(page_width),
                float(page_height),
            )
        )
    boxes.sort(key=lambda item: (_box_area(item), item["y"], item["x"]), reverse=True)
    return boxes[:48], {
        "page_width": float(page_width),
        "page_height": float(page_height),
        "background_color_bgr": [round(float(value), 2) for value in background_color.tolist()],
        "distance_threshold": round(distance_threshold, 4),
        "raw_component_count": max(component_count - 1, 0),
        "selected_block_count": len(boxes[:48]),
    }


def _extract_blocks_from_artifacts(artifacts: ArtifactBundle) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    snapshot = artifacts.dom_snapshot or {}
    page_width = float(snapshot.get("poster_width") or 0.0)
    page_height = float(snapshot.get("poster_height") or 0.0)
    candidates = list(artifacts.detected_elements or []) or list(artifacts.layout_boxes or []) or list(artifacts.text_blocks or [])
    if not page_width or not page_height:
        for candidate in candidates:
            page_width = max(page_width, float(candidate.get("right", 0.0) or 0.0), float(candidate.get("x", 0.0) or 0.0) + float(candidate.get("width", 0.0) or 0.0))
            page_height = max(page_height, float(candidate.get("bottom", 0.0) or 0.0), float(candidate.get("y", 0.0) or 0.0) + float(candidate.get("height", 0.0) or 0.0))
    if not page_width or not page_height:
        return [], {"error": "missing page geometry for artifact fallback"}

    boxes: List[Dict[str, Any]] = []
    seen = set()
    min_area = max(0.0008, 0.0012)
    for index, candidate in enumerate(candidates, start=1):
        width = float(candidate.get("width", 0.0) or 0.0)
        height = float(candidate.get("height", 0.0) or 0.0)
        x = float(candidate.get("x", 0.0) or 0.0)
        y = float(candidate.get("y", 0.0) or 0.0)
        if width <= 0.0 or height <= 0.0:
            continue
        area_ratio = (width * height) / max(page_width * page_height, 1.0)
        if area_ratio < min_area or area_ratio > 0.96:
            continue
        role = str(candidate.get("role") or "")
        if role in {"page", "page_content", "main_content_cluster", "columns_content_cluster"}:
            continue
        normalized = _normalized_box(
            str(candidate.get("name") or candidate.get("text") or f"artifact_block_{index}"),
            x,
            y,
            width,
            height,
            page_width,
            page_height,
        )
        key = tuple(normalized["bbox"])
        if key in seen:
            continue
        seen.add(key)
        boxes.append(normalized)
    boxes.sort(key=lambda item: (_box_area(item), item["y"], item["x"]), reverse=True)
    return boxes[:48], {
        "page_width": round(page_width, 2),
        "page_height": round(page_height, 2),
        "selected_block_count": len(boxes[:48]),
    }


def _intersection_area(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    inter_x1 = max(float(left.get("x", 0.0)), float(right.get("x", 0.0)))
    inter_y1 = max(float(left.get("y", 0.0)), float(right.get("y", 0.0)))
    inter_x2 = min(float(left.get("right", 0.0)), float(right.get("right", 0.0)))
    inter_y2 = min(float(left.get("bottom", 0.0)), float(right.get("bottom", 0.0)))
    return max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)


def _union_box(name: str, boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    min_x = min(float(box.get("x", 0.0)) for box in boxes)
    min_y = min(float(box.get("y", 0.0)) for box in boxes)
    max_right = max(float(box.get("right", 0.0)) for box in boxes)
    max_bottom = max(float(box.get("bottom", 0.0)) for box in boxes)
    merged = _normalized_box_from_unit(name, min_x, min_y, max_right - min_x, max_bottom - min_y)
    merged["source"] = "html_backbone_aggregated"
    merged["member_names"] = [str(box.get("name") or "") for box in boxes]
    merged["member_count"] = len(boxes)
    return merged


def _axis_overlap_ratio(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    overlap = max(min(left_end, right_end) - max(left_start, right_start), 0.0)
    shorter = max(min(left_end - left_start, right_end - right_start), 1e-6)
    return overlap / shorter


def _box_union_efficiency(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    min_x = min(float(left.get("x", 0.0)), float(right.get("x", 0.0)))
    min_y = min(float(left.get("y", 0.0)), float(right.get("y", 0.0)))
    max_right = max(float(left.get("right", 0.0)), float(right.get("right", 0.0)))
    max_bottom = max(float(left.get("bottom", 0.0)), float(right.get("bottom", 0.0)))
    union_bbox_area = max(max_right - min_x, 0.0) * max(max_bottom - min_y, 0.0)
    if union_bbox_area <= 0.0:
        return 0.0
    return (_box_area(left) + _box_area(right)) / union_bbox_area


def _union_area_ratio(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    min_x = min(float(left.get("x", 0.0)), float(right.get("x", 0.0)))
    min_y = min(float(left.get("y", 0.0)), float(right.get("y", 0.0)))
    max_right = max(float(left.get("right", 0.0)), float(right.get("right", 0.0)))
    max_bottom = max(float(left.get("bottom", 0.0)), float(right.get("bottom", 0.0)))
    return max(max_right - min_x, 0.0) * max(max_bottom - min_y, 0.0)


_TEXT_SEPARATOR_RE = re.compile(r"[\-_:;,.!?\"'()\[\]{}<>/\\&+|，。！？：；、‘’“”《》【】（）]+")
_TEXT_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text_key(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = _TEXT_SEPARATOR_RE.sub(" ", lowered)
    return _TEXT_WHITESPACE_RE.sub(" ", lowered).strip()


def _compact_text_key(value: str) -> str:
    return _normalize_text_key(value).replace(" ", "")


def _parent_tag_name(block: Dict[str, Any]) -> str:
    signature = str(block.get("parent_signature") or "")
    return signature.split("|", 1)[0].strip().lower() if signature else ""


def _looks_heading_text(text: str) -> bool:
    raw_text = str(text or "").strip()
    if not raw_text:
        return False
    letters = [char for char in raw_text if char.isalpha()]
    uppercase_ratio = (sum(1 for char in letters if char.isupper()) / len(letters)) if letters else 0.0
    word_count = len([token for token in raw_text.split() if token])
    if any(char in raw_text for char in ".!?"):
        return False
    return word_count <= 8 and (uppercase_ratio >= 0.6 or raw_text == raw_text.upper())


def _classify_text_block_level(block: Dict[str, Any]) -> str:
    semantic_role = str(block.get("semantic_role") or block.get("role") or "").strip().lower()
    semantic_name = str(block.get("semantic_name") or block.get("name") or "").strip().lower()
    semantic_scope = str(block.get("semantic_scope") or block.get("scope") or "").strip().lower()
    parent_tag = _parent_tag_name(block)
    text = str(block.get("text") or "")

    if semantic_role == "title" or semantic_name == "title" or parent_tag == "h1":
        return "title"

    if (
        semantic_name.startswith("column_") and semantic_name.endswith("_section_heading")
        or (semantic_role == "section_heading" and semantic_scope in {"column_1", "column_2", "column_3"} and float(block.get("y", 0.0)) <= 0.35)
    ):
        return "column_heading"

    if semantic_role == "section_heading" or "heading" in semantic_name or parent_tag in {"h2", "h3", "h4", "h5", "h6"} or _looks_heading_text(text):
        return "section_heading"

    return "body_text"


def _is_heading_level(level: str) -> bool:
    return str(level or "") in {"title", "column_heading", "section_heading"}


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = [token for token in _normalize_text_key(left).split(" ") if token]
    right_tokens = [token for token in _normalize_text_key(right).split(" ") if token]
    if not left_tokens or not right_tokens:
        return 0.0
    left_token_set = set(left_tokens)
    right_token_set = set(right_tokens)
    intersection = len(left_token_set & right_token_set)
    if intersection <= 0:
        return 0.0
    precision = intersection / max(len(right_token_set), 1)
    recall = intersection / max(len(left_token_set), 1)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _char_ngram_similarity(left: str, right: str, *, n: int = 2) -> float:
    left_text = str(left or "")
    right_text = str(right or "")
    if not left_text or not right_text:
        return 0.0
    if len(left_text) < n or len(right_text) < n:
        return 1.0 if left_text == right_text else _ratio(left_text, right_text)

    left_ngrams = {left_text[index : index + n] for index in range(len(left_text) - n + 1)}
    right_ngrams = {right_text[index : index + n] for index in range(len(right_text) - n + 1)}
    if not left_ngrams or not right_ngrams:
        return 0.0
    overlap = len(left_ngrams & right_ngrams)
    union = len(left_ngrams | right_ngrams)
    if union <= 0:
        return 0.0
    return overlap / union


def _soft_token_similarity(left: str, right: str) -> float:
    left_tokens = [token for token in _normalize_text_key(left).split(" ") if token]
    right_tokens = [token for token in _normalize_text_key(right).split(" ") if token]
    if not left_tokens or not right_tokens:
        return 0.0

    if _rfprocess is not None:
        # cdist computes the full token-pair matrix in one vectorised C call
        import numpy as _np
        matrix = _rfprocess.cdist(left_tokens, right_tokens, scorer=_rfuzz.ratio, score_cutoff=0) / 100.0
        forward = float(_np.max(matrix, axis=1).mean())
        backward = float(_np.max(matrix, axis=0).mean())
    else:
        left_to_right = sum(
            max(_ratio(lt, rt) for rt in right_tokens) for lt in left_tokens
        )
        right_to_left = sum(
            max(_ratio(rt, lt) for lt in left_tokens) for rt in right_tokens
        )
        forward = left_to_right / max(len(left_tokens), 1)
        backward = right_to_left / max(len(right_tokens), 1)

    return max(0.0, min(1.0, (forward + backward) / 2.0))


def _is_text_match_candidate(left: Dict[str, Any], right: Dict[str, Any], *, minimum_similarity: float = 0.42, text_similarity: Optional[float] = None) -> bool:
    if text_similarity is None:
        text_similarity = _text_similarity(left, right)
    if text_similarity >= minimum_similarity:
        return True

    left_compact = str(left.get("compact_text") or "")
    right_compact = str(right.get("compact_text") or "")
    if left_compact and right_compact:
        if left_compact in right_compact or right_compact in left_compact:
            return True
    return False


def _same_column_or_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    x_overlap = _axis_overlap_ratio(
        float(left.get("x", 0.0)),
        float(left.get("right", 0.0)),
        float(right.get("x", 0.0)),
        float(right.get("right", 0.0)),
    )
    left_cx = float(left.get("cx", 0.0))
    right_cx = float(right.get("cx", 0.0))
    column_gap = abs(left_cx - right_cx)
    return x_overlap >= 0.45 or column_gap <= 0.08





def _get_paddle_ocr_instance(lang: str = "ch") -> Any:
    cache_key = lang
    if cache_key in _PADDLE_OCR_CACHE:
        return _PADDLE_OCR_CACHE[cache_key]
    # PaddlePaddle C++ predictor + ir_optim init is NOT thread-safe;
    # asyncio AgentLoopWorker (max_concurrency=1000) concurrent first-touch
    # triggers SIGSEGV (stack top: thread_run -> pythread_wrapper, .out shows
    # two ppocr Namespaces printed in same millisecond).
    # Double-checked locking: 99.99% of calls hit the cached path above and
    # never enter the lock; only the first few racing workers serialise on
    # init, after which the cache is shared with everyone.
    with _PADDLE_OCR_LOCK:
        if cache_key in _PADDLE_OCR_CACHE:
            return _PADDLE_OCR_CACHE[cache_key]
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        # Force CPU for OCR: during RL training all GPU memory is occupied by
        # vLLM / FSDP, so PaddlePaddle GPU inference segfaults.  CPU OCR is
        # only ~0.3s slower per image and avoids the crash entirely.
        # Override with PADDLE_OCR_USE_GPU=1 for offline prewarming on idle GPUs.
        use_gpu = os.getenv("PADDLE_OCR_USE_GPU", "0").lower() in ("1", "true", "yes")

        constructor_variants = [
            {
                "lang": lang,
                "use_gpu": use_gpu,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            {"lang": lang, "use_gpu": use_gpu},
            {"lang": lang},
        ]
        last_error: Optional[Exception] = None
        for kwargs in constructor_variants:
            try:
                ocr = PaddleOCR(**kwargs)
                _PADDLE_OCR_CACHE[cache_key] = ocr
                _logger.info("PaddleOCR initialized (use_gpu=%s, lang=%s)", use_gpu, lang)
                return ocr
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(str(last_error) if last_error else "unable to initialize PaddleOCR")


def _extract_reference_text_blocks_with_paddle(
    image_path: str,
    *,
    source: str = "reference_ocr",
    name_prefix: str = "ocr_text",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        return [], {"error": f"opencv unavailable: {exc}", "ocr_engine": "paddleocr"}

    image = _safe_imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return [], {"error": f"unable to read image: {image_path}", "ocr_engine": "paddleocr"}

    original_page_height, original_page_width = image.shape[:2]
    prepared_image_path, preprocessing_debug, cleanup_path = _prepare_image_for_ocr(image_path)
    prepared_image = _safe_imread(prepared_image_path, cv2.IMREAD_COLOR)
    if prepared_image is not None:
        page_height, page_width = prepared_image.shape[:2]
    else:
        page_height, page_width = image.shape[:2]
    try:
        ocr = _get_paddle_ocr_instance("ch")
    except Exception as exc:  # noqa: BLE001
        return [], {"error": f"paddleocr unavailable: {exc}", "ocr_engine": "paddleocr"}

    def _run_ocr(ocr_instance: Any, img_path: str) -> Any:
        if hasattr(ocr_instance, "predict"):
            try:
                return ocr_instance.predict(
                    img_path,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception as predict_exc:
                _logger.warning(
                    "PaddleOCR .predict() failed, falling back to .ocr(): %s",
                    predict_exc,
                )
                # Fall through to .ocr()
        return ocr_instance.ocr(img_path)

    try:
        with _PADDLE_OCR_LOCK:
            raw_result = _run_ocr(ocr, str(prepared_image_path))
    except (IndexError, Exception) as exc:
        # PaddleOCR has a known internal bug: "index N is out of bounds for
        # axis 0 with size N" triggered by certain preprocessed images.
        # Retry once with the original (unprocessed) image before giving up.
        if "out of bounds" in str(exc) and str(prepared_image_path) != str(image_path):
            _logger.warning(
                "PaddleOCR index error on preprocessed image, retrying with original: %s",
                image_path,
            )
            try:
                with _PADDLE_OCR_LOCK:
                    raw_result = _run_ocr(ocr, str(image_path))
                # On retry success, use original image dimensions for coord mapping
                page_height, page_width = original_page_height, original_page_width
                preprocessing_debug["retry_with_original"] = True
            except Exception as retry_exc:
                if cleanup_path:
                    Path(cleanup_path).unlink(missing_ok=True)
                return [], {"error": f"paddleocr failed after retry: {retry_exc}", "ocr_engine": "paddleocr"}
        else:
            if cleanup_path:
                Path(cleanup_path).unlink(missing_ok=True)
            return [], {"error": f"paddleocr failed: {exc}", "ocr_engine": "paddleocr"}

    blocks: List[Dict[str, Any]] = []
    raw_line_count = 0
    if raw_result and hasattr(raw_result[0], "get") and raw_result[0].get("dt_polys") is not None:
        page_result = raw_result[0]
        dt_polys = list(page_result.get("dt_polys") or [])
        rec_texts = list(page_result.get("rec_texts") or [])
        rec_scores = list(page_result.get("rec_scores") or [])
        for index, (points, text, confidence) in enumerate(zip(dt_polys, rec_texts, rec_scores), start=1):
            text = str(text or "").strip()
            if not text:
                continue
            raw_line_count += 1
            normalized_text = _normalize_text_key(text)
            compact_text = _compact_text_key(text)
            if len(compact_text) < 2:
                continue
            try:
                confidence_value = float(confidence or 0.0)
            except (TypeError, ValueError):
                confidence_value = 0.0
            if confidence_value < 0.35:
                continue
            x_coords = [float(point[0]) for point in points]
            y_coords = [float(point[1]) for point in points]
            min_x = min(x_coords)
            min_y = min(y_coords)
            max_x = max(x_coords)
            max_y = max(y_coords)
            if max_x <= min_x or max_y <= min_y:
                continue
            scale_x = float(original_page_width) / float(page_width or original_page_width)
            scale_y = float(original_page_height) / float(page_height or original_page_height)
            pixel_x = min_x * scale_x
            pixel_y = min_y * scale_y
            pixel_w = (max_x - min_x) * scale_x
            pixel_h = (max_y - min_y) * scale_y
            block = _normalized_box(
                f"{name_prefix}_{index}",
                pixel_x,
                pixel_y,
                pixel_w,
                pixel_h,
                float(original_page_width),
                float(original_page_height),
            )
            block["pixel_bbox"] = [round(pixel_x, 2), round(pixel_y, 2), round(pixel_w, 2), round(pixel_h, 2)]
            block["pixel_bbox_source_size"] = [int(original_page_width), int(original_page_height)]
            block["text"] = text
            block["normalized_text"] = normalized_text
            block["compact_text"] = compact_text
            block["source"] = source
            block["confidence"] = round(confidence_value, 4)
            block["word_count"] = len([token for token in normalized_text.split(" ") if token])
            blocks.append(block)
    else:
        line_items = raw_result[0] if raw_result and isinstance(raw_result[0], list) else (raw_result or [])
        for index, item in enumerate(line_items, start=1):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            points = item[0]
            text_payload = item[1]
            if not isinstance(points, (list, tuple)) or len(points) < 4:
                continue
            if not isinstance(text_payload, (list, tuple)) or len(text_payload) < 2:
                continue
            text = str(text_payload[0] or "").strip()
            if not text:
                continue
            raw_line_count += 1
            normalized_text = _normalize_text_key(text)
            compact_text = _compact_text_key(text)
            if len(compact_text) < 2:
                continue
            try:
                confidence = float(text_payload[1] or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.35:
                continue
            x_coords = [float(point[0]) for point in points]
            y_coords = [float(point[1]) for point in points]
            min_x = min(x_coords)
            min_y = min(y_coords)
            max_x = max(x_coords)
            max_y = max(y_coords)
            if max_x <= min_x or max_y <= min_y:
                continue
            scale_x = float(original_page_width) / float(page_width or original_page_width)
            scale_y = float(original_page_height) / float(page_height or original_page_height)
            pixel_x = min_x * scale_x
            pixel_y = min_y * scale_y
            pixel_w = (max_x - min_x) * scale_x
            pixel_h = (max_y - min_y) * scale_y
            block = _normalized_box(
                f"{name_prefix}_{index}",
                pixel_x,
                pixel_y,
                pixel_w,
                pixel_h,
                float(original_page_width),
                float(original_page_height),
            )
            block["pixel_bbox"] = [round(pixel_x, 2), round(pixel_y, 2), round(pixel_w, 2), round(pixel_h, 2)]
            block["pixel_bbox_source_size"] = [int(original_page_width), int(original_page_height)]
            block["text"] = text
            block["normalized_text"] = normalized_text
            block["compact_text"] = compact_text
            block["source"] = source
            block["confidence"] = round(confidence, 4)
            block["word_count"] = len([token for token in normalized_text.split(" ") if token])
            blocks.append(block)

    blocks.sort(key=lambda item: (item["y"], item["x"]))
    if cleanup_path:
        Path(cleanup_path).unlink(missing_ok=True)
    return blocks[:200], {
        "page_width": float(original_page_width),
        "page_height": float(original_page_height),
        "prepared_page_width": float(page_width),
        "prepared_page_height": float(page_height),
        "raw_line_count": raw_line_count,
        "selected_block_count": len(blocks[:200]),
        "ocr_engine": "paddleocr",
        "ocr_languages": ["ch"],
        "ocr_preprocessing": preprocessing_debug,
    }


def _extract_reference_text_blocks(
    image_path: str,
    *,
    source: str = "reference_ocr",
    name_prefix: str = "ocr_text",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract reference text blocks using PaddleOCR (mandatory).

    Results are cached to disk per reference image path because OCR is
    100% deterministic for a given image file.  This avoids re-running
    PaddleOCR (2-13s/image) on subsequent verify calls.
    """
    # --- Check disk cache first ---
    cached = _ocr_cache_get(image_path)
    if cached is not None:
        return cached

    # --- Cache miss: run PaddleOCR ---
    blocks, debug = _extract_reference_text_blocks_with_paddle(
        image_path,
        source=source,
        name_prefix=name_prefix,
    )
    if blocks:
        blocks, postprocess_debug = _postprocess_reference_ocr_blocks(blocks)
        debug["selected_ocr_engine"] = "paddle"
        debug["ocr_postprocess"] = postprocess_debug
        debug["selected_block_count"] = len(blocks)
        # Save to disk cache
        _ocr_cache_put(image_path, blocks, debug)
        return blocks, debug

    debug.setdefault("error", "no_ocr_blocks_extracted")
    debug["selected_ocr_engine"] = None
    # Do NOT cache empty results — see _ocr_cache_put for rationale.
    return [], debug


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def _is_short_text_block(block: Dict[str, Any]) -> bool:
    """Return True for blocks too short to be meaningful layout anchors.

    Filters chart tick-labels, axis numbers, single-letter debris etc.
    Keeps CJK ≥ 2 chars because two Chinese characters already carry meaning.
    """
    compact_text = str(block.get("compact_text") or _compact_text_key(str(block.get("text") or "")))
    if len(compact_text) >= 4:
        return False
    # CJK: 2+ characters are meaningful (e.g. "留白", "冷暖")
    if _contains_cjk(str(block.get("text") or "")) and len(compact_text) >= 2:
        return False
    return True


_is_short_reference_ocr_block = _is_short_text_block



def _postprocess_reference_ocr_blocks(blocks: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter short noisy OCR blocks.  No same-line merging — OCR output
    granularity is kept as-is so each block maps 1:1 to a real text region."""
    if not blocks:
        return [], {
            "raw_block_count": 0,
            "short_block_drop_count": 0,
            "selected_block_count": 0,
        }

    prepared_blocks = sorted(
        (dict(block) for block in blocks),
        key=lambda item: (float(item.get("y", 0.0)), float(item.get("x", 0.0))),
    )
    # Ensure member_count is set for downstream consumers.
    for block in prepared_blocks:
        block.setdefault("member_count", 1)

    filtered_blocks = [b for b in prepared_blocks if not _is_short_reference_ocr_block(b)]
    limited_blocks = filtered_blocks[:200]
    return limited_blocks, {
        "raw_block_count": len(blocks),
        "short_block_drop_count": len(prepared_blocks) - len(filtered_blocks),
        "selected_block_count": len(limited_blocks),
    }


def _extract_html_source_text_candidates(html_source: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_source or "", "html.parser")
    for node in soup.select("script, style, noscript, svg"):
        node.decompose()

    candidates: List[Dict[str, str]] = []
    candidate_tags = "h1, h2, h3, h4, h5, h6, p, li, span, div, a, button, label, td, th"
    for node in soup.select(candidate_tags):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        normalized_text = _normalize_text_key(text)
        compact_text = _compact_text_key(text)
        if len(compact_text) < 2:
            continue
        if node.name in {"div", "span"}:
            child_text_count = 0
            for child in node.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "span", "div", "a", "button", "label"], recursive=False):
                child_text = _compact_text_key(child.get_text(" ", strip=True))
                if len(child_text) >= 2:
                    child_text_count += 1
            if child_text_count:
                continue
        if candidates and candidates[-1]["compact_text"] == compact_text:
            continue
        candidates.append({
            "text": text,
            "normalized_text": normalized_text,
            "compact_text": compact_text,
            "tag": str(node.name or "div"),
        })
    return candidates


def _extract_predicted_text_blocks_from_html_source(artifacts: ArtifactBundle) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = _extract_html_source_text_candidates(artifacts.html_source)
    if not candidates:
        return [], {"source": "html_source", "selected_block_count": 0, "error": "no_text_candidates"}

    base_heights = [0.055 if candidate["tag"].startswith("h") else 0.04 if candidate["tag"] in {"p", "li", "td", "th"} else 0.032 for candidate in candidates[:200]]
    total_height = sum(base_heights) + max(len(base_heights) - 1, 0) * 0.012
    scale = min(1.0, 0.84 / max(total_height, 1e-6))
    y_cursor = 0.06
    blocks: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:200], start=1):
        height = base_heights[index - 1] * scale
        gap = 0.012 * scale
        if y_cursor + height > 0.97:
            break
        text = candidate["text"]
        normalized_text = candidate["normalized_text"]
        compact_text = candidate["compact_text"]
        width = min(0.9, max(0.2, 0.12 + min(len(compact_text), 72) * 0.009))
        x = 0.12 if candidate["tag"] == "li" else 0.08
        block = _normalized_box_from_unit(f"html_text_{index}", x, y_cursor, width, height)
        block["text"] = text
        block["normalized_text"] = normalized_text
        block["compact_text"] = compact_text
        block["source"] = "html_source_text_fallback"
        block["tag"] = candidate["tag"]
        blocks.append(block)
        y_cursor += height + gap

    return blocks, {
        "source": "html_source",
        "selected_block_count": len(blocks),
        "candidate_count": len(candidates),
    }


def _extract_predicted_text_blocks(artifacts: ArtifactBundle) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    snapshot = artifacts.dom_snapshot or {}
    page_width = float(snapshot.get("poster_width") or 0.0)
    page_height = float(snapshot.get("poster_height") or 0.0)
    blocks: List[Dict[str, Any]] = []
    for index, block in enumerate(list(artifacts.text_blocks or []), start=1):
        text = str(block.get("text") or "").strip()
        normalized_text = _normalize_text_key(text)
        compact_text = _compact_text_key(text)
        if len(compact_text) < 2:
            continue
        width = float(block.get("width", block.get("w", 0.0)) or 0.0)
        height = float(block.get("height", block.get("h", 0.0)) or 0.0)
        x = float(block.get("x", 0.0) or 0.0)
        y = float(block.get("y", 0.0) or 0.0)
        normalized_bbox = block.get("normalized_bbox") or []
        if isinstance(normalized_bbox, (list, tuple)) and len(normalized_bbox) == 4:
            normalized = _normalized_box_from_unit(
                str(block.get("name") or f"pred_text_{index}"),
                float(normalized_bbox[0] or 0.0),
                float(normalized_bbox[1] or 0.0),
                float(normalized_bbox[2] or 0.0),
                float(normalized_bbox[3] or 0.0),
            )
        else:
            if width <= 0.0 or height <= 0.0:
                continue
            if not page_width:
                page_width = max(page_width, x + width)
            if not page_height:
                page_height = max(page_height, y + height)
            normalized = _normalized_box(
                str(block.get("name") or f"pred_text_{index}"),
                x,
                y,
                width,
                height,
                float(page_width),
                float(page_height),
            )
        normalized["text"] = text
        normalized["normalized_text"] = normalized_text
        normalized["compact_text"] = compact_text
        normalized["parent_signature"] = block.get("parent_signature")
        normalized["semantic_name"] = block.get("semantic_name")
        normalized["semantic_role"] = block.get("semantic_role")
        normalized["semantic_scope"] = block.get("semantic_scope")
        normalized["tag"] = block.get("tag") or _parent_tag_name(block)
        normalized["source"] = "predicted_text_block"
        normalized["text_level"] = _classify_text_block_level(normalized)
        if width > 0.0 and height > 0.0 and page_width > 0.0 and page_height > 0.0:
            normalized["pixel_bbox"] = [round(x, 2), round(y, 2), round(width, 2), round(height, 2)]
            normalized["pixel_bbox_source_size"] = [int(round(page_width)), int(round(page_height))]
        blocks.append(normalized)

    # Filter short blocks (chart labels, axis ticks, etc.) — same as reference side.
    blocks = [b for b in blocks if not _is_short_text_block(b)]

    if not blocks:
        fallback_blocks, fallback_debug = _extract_predicted_text_blocks_from_html_source(artifacts)
        return fallback_blocks[:200], {
            "page_width": round(page_width, 2),
            "page_height": round(page_height, 2),
            "selected_block_count": len(fallback_blocks[:200]),
            "source": "html_source_fallback",
            "fallback": fallback_debug,
        }

    return blocks[:200], {
        "page_width": round(page_width, 2),
        "page_height": round(page_height, 2),
        "selected_block_count": len(blocks[:200]),
        "source": "artifacts.text_blocks",
    }


def _text_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    """Single-pair text similarity using 50/50 embedding and fuzzy char score.

    For bulk comparisons prefer ``_compute_text_similarity_matrix`` which
    amortises the model-encode overhead across all pairs at once.
    """
    left_text = str(left.get("normalized_text") or left.get("text") or "")
    right_text = str(right.get("normalized_text") or right.get("text") or "")
    if not left_text or not right_text:
        return 0.0

    left_compact = str(left.get("compact_text") or _compact_text_key(left_text))
    right_compact = str(right.get("compact_text") or _compact_text_key(right_text))
    fuzzy_similarity = max(_ratio(left_text, right_text), _ratio(left_compact, right_compact))

    model = _get_sbert_model()
    if model is not None:
        emb = model.encode([left_text, right_text], normalize_embeddings=True, show_progress_bar=False)
        sim = float(emb[0] @ emb[1])
        return max(0.0, min(1.0, 0.5 * sim + 0.5 * fuzzy_similarity))

    return fuzzy_similarity


def _text_pair_metrics(reference_box: Dict[str, Any], predicted_box: Dict[str, Any], *, text_similarity: Optional[float] = None) -> Dict[str, float]:
    geometry = _pair_metrics(reference_box, predicted_box)
    if text_similarity is None:
        text_similarity = _text_similarity(reference_box, predicted_box)
    text_match_candidate = _is_text_match_candidate(reference_box, predicted_box, text_similarity=text_similarity)
    correspondence_score = max(
        0.0,
        min(1.0, (text_similarity + geometry["center_similarity"] + geometry["size_similarity"]) / 3.0),
    )
    alignment_score = max(
        0.0,
        min(1.0, (geometry["center_similarity"] + geometry["size_similarity"]) / 2.0),
    )
    score = alignment_score if text_match_candidate else 0.0
    return {
        **geometry,
        "text_similarity": round(text_similarity, 4),
        "text_match_candidate": text_match_candidate,
        "correspondence_score": round(correspondence_score, 4),
        "alignment_score": round(alignment_score, 4),
        "score": round(score, 4),
    }


def _hungarian_match_text(
    reference_boxes: List[Dict[str, Any]],
    predicted_boxes: List[Dict[str, Any]],
    *,
    min_score: float = 0.42,
    text_sim_matrix: Optional["np.ndarray"] = None,
) -> Dict[str, Any]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _build_match_payload(reference_boxes, predicted_boxes, [], algorithm="hungarian_text_unavailable")

    if not reference_boxes or not predicted_boxes:
        return _build_match_payload(reference_boxes, predicted_boxes, [], algorithm="hungarian_text")

    # Pre-compute the full text similarity matrix once (batch encode)
    if text_sim_matrix is None:
        text_sim_matrix = _compute_text_similarity_matrix(reference_boxes, predicted_boxes)

    score_matrix = np.zeros((len(reference_boxes), len(predicted_boxes)), dtype=float)
    metric_matrix: List[List[Dict[str, float]]] = []
    for reference_index, reference_box in enumerate(reference_boxes):
        metric_row: List[Dict[str, float]] = []
        for predicted_index, predicted_box in enumerate(predicted_boxes):
            ts = float(text_sim_matrix[reference_index, predicted_index])
            metrics = _text_pair_metrics(reference_box, predicted_box, text_similarity=ts)
            metric_row.append(metrics)
        metric_matrix.append(metric_row)
    for reference_index in range(len(reference_boxes)):
        for predicted_index in range(len(predicted_boxes)):
            score_matrix[reference_index, predicted_index] = metric_matrix[reference_index][predicted_index]["correspondence_score"]

    row_indices, col_indices = linear_sum_assignment(1.0 - score_matrix)
    matches: List[Dict[str, Any]] = []
    for reference_index, predicted_index in zip(row_indices.tolist(), col_indices.tolist()):
        metrics = metric_matrix[reference_index][predicted_index]
        if metrics["correspondence_score"] < 0.5 or metrics["text_similarity"] < 0.45 or metrics["score"] < min_score:
            continue
        matches.append(
            {
                "reference_index": reference_index,
                "predicted_index": predicted_index,
                "reference_name": reference_boxes[reference_index].get("name"),
                "predicted_name": predicted_boxes[predicted_index].get("name"),
                "reference_box": reference_boxes[reference_index],
                "predicted_box": predicted_boxes[predicted_index],
                **metrics,
            }
        )
    return _build_match_payload(reference_boxes, predicted_boxes, matches, algorithm="hungarian_text")


def _score_match_payload(matches: List[Dict[str, Any]], reference_blocks: List[Dict[str, Any]], predicted_blocks: List[Dict[str, Any]]) -> Dict[str, float]:
    mean_pair_score = sum(item["score"] for item in matches) / len(matches) if matches else 0.0
    mean_alignment_score = sum(item.get("alignment_score", item.get("score", 0.0)) for item in matches) / len(matches) if matches else 0.0
    mean_correspondence_score = sum(item.get("correspondence_score", item.get("score", 0.0)) for item in matches) / len(matches) if matches else 0.0
    mean_center_distance = sum(item.get("center_distance", 1.0) for item in matches) / len(matches) if matches else 1.0
    mean_size_similarity = sum(item.get("size_similarity", 0.0) for item in matches) / len(matches) if matches else 0.0
    mean_text_similarity = sum(item.get("text_similarity", 0.0) for item in matches) / len(matches) if matches else 0.0
    reference_coverage = len(matches) / max(len(reference_blocks), 1)
    predicted_precision = len(matches) / max(len(predicted_blocks), 1)
    score = max(0.0, min(1.0, 0.40 * reference_coverage + 0.15 * predicted_precision + 0.45 * mean_alignment_score))
    return {
        "reference_coverage": round(reference_coverage, 4),
        "predicted_precision": round(predicted_precision, 4),
        "mean_pair_score": round(mean_pair_score, 4),
        "mean_alignment_score": round(mean_alignment_score, 4),
        "mean_correspondence_score": round(mean_correspondence_score, 4),
        "mean_center_distance": round(mean_center_distance, 4),
        "mean_size_similarity": round(mean_size_similarity, 4),
        "mean_text_similarity": round(mean_text_similarity, 4),
        "score": round(score, 4),
    }


def _status_from_score(score: float, reference_coverage: float, has_matches: bool) -> ValidationStatus:
    return ValidationStatus.PASS


def _merge_candidate_score(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    union_area_ratio = _union_area_ratio(left, right)
    inter_area = _intersection_area(left, right)
    min_area = min(_box_area(left), _box_area(right))
    overlap_ratio = inter_area / max(min_area, 1e-6)
    if overlap_ratio >= 0.15:
        return 0.95 + min(overlap_ratio, 0.3)

    horizontal_gap = max(
        float(right.get("x", 0.0)) - float(left.get("right", 0.0)),
        float(left.get("x", 0.0)) - float(right.get("right", 0.0)),
        0.0,
    )
    vertical_gap = max(
        float(right.get("y", 0.0)) - float(left.get("bottom", 0.0)),
        float(left.get("y", 0.0)) - float(right.get("bottom", 0.0)),
        0.0,
    )
    vertical_overlap_ratio = _axis_overlap_ratio(
        float(left.get("y", 0.0)),
        float(left.get("bottom", 0.0)),
        float(right.get("y", 0.0)),
        float(right.get("bottom", 0.0)),
    )
    horizontal_overlap_ratio = _axis_overlap_ratio(
        float(left.get("x", 0.0)),
        float(left.get("right", 0.0)),
        float(right.get("x", 0.0)),
        float(right.get("right", 0.0)),
    )
    union_efficiency = _box_union_efficiency(left, right)
    left_member_count = int(left.get("member_count") or 1)
    right_member_count = int(right.get("member_count") or 1)

    allow_large_stacked_pair = (
        left_member_count == 1
        and right_member_count == 1
        and vertical_gap <= 0.014
        and horizontal_overlap_ratio >= 0.9
        and union_efficiency >= 0.9
    )
    if union_area_ratio >= 0.12 and not allow_large_stacked_pair:
        return 0.0

    inter_area = _intersection_area(left, right)
    containment_ratio = inter_area / max(min(_box_area(left), _box_area(right)), 1e-6)
    if containment_ratio >= 0.82:
        return 0.9 + min(containment_ratio, 0.15)

    if horizontal_gap <= 0.012 and vertical_overlap_ratio >= 0.72 and union_efficiency >= 0.72:
        return 0.76 + vertical_overlap_ratio * 0.08 - horizontal_gap * 5.0
    if vertical_gap <= 0.014 and horizontal_overlap_ratio >= 0.82 and union_efficiency >= 0.76:
        return 0.74 + horizontal_overlap_ratio * 0.1 - vertical_gap * 5.5

    center_left = _box_center(left)
    center_right = _box_center(right)
    center_distance = math.sqrt((center_left[0] - center_right[0]) ** 2 + (center_left[1] - center_right[1]) ** 2)
    if center_distance <= 0.045 and union_efficiency >= 0.84 and union_area_ratio <= 0.045:
        return 0.66 + union_efficiency * 0.08
    return 0.0


def _aggregate_html_backbone_blocks(html_blocks: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    active_blocks = [dict(box) for box in html_blocks]
    merge_steps = 0
    while len(active_blocks) > 1:
        best_score = 0.0
        best_pair: tuple[int, int] | None = None
        for left_index in range(len(active_blocks)):
            for right_index in range(left_index + 1, len(active_blocks)):
                score = _merge_candidate_score(active_blocks[left_index], active_blocks[right_index])
                if score > best_score:
                    best_score = score
                    best_pair = (left_index, right_index)
        if best_pair is None or best_score < 0.78:
            break

        left_index, right_index = best_pair
        left_box = active_blocks[left_index]
        right_box = active_blocks[right_index]
        merged_box = _union_box(
            f"html_cluster_{merge_steps + 1}",
            [left_box, right_box],
        )
        merged_box["merge_score"] = round(best_score, 4)
        survivors = [
            box
            for index, box in enumerate(active_blocks)
            if index not in {left_index, right_index}
        ]
        survivors.append(merged_box)
        active_blocks = sorted(survivors, key=lambda item: (_box_area(item), item["y"], item["x"]), reverse=True)
        merge_steps += 1

    aggregated_blocks = active_blocks[:48]
    return aggregated_blocks, {
        "input_block_count": len(html_blocks),
        "aggregated_block_count": len(aggregated_blocks),
        "merge_steps": merge_steps,
    }


def _blend_boxes(html_box: Dict[str, Any], image_box: Dict[str, Any], *, name: str) -> Dict[str, Any]:
    metrics = _pair_metrics(html_box, image_box)
    image_weight = 0.35 if metrics["center_distance"] <= 0.05 else 0.3
    html_weight = 1.0 - image_weight
    x = html_weight * float(html_box["x"]) + image_weight * float(image_box["x"])
    y = html_weight * float(html_box["y"]) + image_weight * float(image_box["y"])
    width = html_weight * float(html_box["width"]) + image_weight * float(image_box["width"])
    height = html_weight * float(html_box["height"]) + image_weight * float(image_box["height"])
    blended = _normalized_box_from_unit(name, x, y, width, height)
    blended["source"] = "html_image_fused"
    blended["fusion_metrics"] = metrics
    return blended


def _max_overlap_with_boxes(box: Dict[str, Any], others: List[Dict[str, Any]]) -> float:
    if not others:
        return 0.0
    box_area = max(_box_area(box), 1e-6)
    return max(_intersection_area(box, other) / box_area for other in others)


def _extract_predicted_blocks(artifacts: ArtifactBundle) -> tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
    html_blocks, html_debug = _extract_blocks_from_artifacts(artifacts)
    aggregated_html_blocks, aggregation_debug = _aggregate_html_backbone_blocks(html_blocks)
    render_image_path = artifacts.render_image_path
    if not render_image_path:
        return aggregated_html_blocks, {"html_debug": html_debug, "aggregation_debug": aggregation_debug, "image_debug": None, "fusion": {"mode": "html_only"}}, "artifact_boxes", "hybrid"

    image_blocks, image_debug = _extract_blocks_from_image(str(render_image_path))
    if not aggregated_html_blocks and not image_blocks:
        return [], {"html_debug": html_debug, "aggregation_debug": aggregation_debug, "image_debug": image_debug, "fusion": {"mode": "empty"}}, "render_image", "explicit"
    if not aggregated_html_blocks:
        return image_blocks, {"html_debug": html_debug, "aggregation_debug": aggregation_debug, "image_debug": image_debug, "fusion": {"mode": "image_only"}}, "render_image", "explicit"
    if not image_blocks:
        return aggregated_html_blocks, {"html_debug": html_debug, "aggregation_debug": aggregation_debug, "image_debug": image_debug, "fusion": {"mode": "html_only_no_image_blocks"}}, "artifact_boxes", "hybrid"

    html_image_matches = _hungarian_match(aggregated_html_blocks, image_blocks, min_score=0.08)
    fused_blocks: List[Dict[str, Any]] = []
    matched_html = set()
    matched_image = set()

    for match in html_image_matches["matches"]:
        html_index = int(match["reference_index"])
        image_index = int(match["predicted_index"])
        html_box = aggregated_html_blocks[html_index]
        image_box = image_blocks[image_index]
        if match["center_distance"] <= 0.14 or match["score"] >= 0.4:
            fused_blocks.append(_blend_boxes(html_box, image_box, name=str(html_box.get("name") or f"pred_block_{len(fused_blocks) + 1}")))
            matched_html.add(html_index)
            matched_image.add(image_index)

    for index, html_box in enumerate(aggregated_html_blocks):
        if index in matched_html:
            continue
        retained = dict(html_box)
        retained["source"] = "html_backbone"
        fused_blocks.append(retained)

    for index, image_box in enumerate(image_blocks):
        if index in matched_image:
            continue
        if _box_area(image_box) < 0.008:
            continue
        if _max_overlap_with_boxes(image_box, fused_blocks) >= 0.2:
            continue
        retained = dict(image_box)
        retained["source"] = "image_visual_only"
        fused_blocks.append(retained)

    fused_blocks.sort(key=lambda item: (_box_area(item), item["y"], item["x"]), reverse=True)
    fused_blocks = fused_blocks[:64]
    return fused_blocks, {
        "html_debug": html_debug,
        "aggregation_debug": aggregation_debug,
        "image_debug": image_debug,
        "fusion": {
            "mode": "html_image_hybrid",
            "html_block_count": len(aggregated_html_blocks),
            "raw_html_block_count": len(html_blocks),
            "image_block_count": len(image_blocks),
            "fused_block_count": len(fused_blocks),
            "matched_html_block_count": len(matched_html),
            "matched_image_block_count": len(matched_image),
        },
    }, "html_image_hybrid", "hybrid"


def _pair_metrics(reference_box: Dict[str, Any], predicted_box: Dict[str, Any]) -> Dict[str, float]:
    reference_center = _box_center(reference_box)
    predicted_center = _box_center(predicted_box)
    center_distance = math.sqrt(
        (reference_center[0] - predicted_center[0]) ** 2 + (reference_center[1] - predicted_center[1]) ** 2
    ) / math.sqrt(2.0)
    center_similarity = max(0.0, 1.0 - center_distance)
    size_similarity = _size_similarity(reference_box, predicted_box)
    score = max(0.0, min(1.0, 0.55 * center_similarity + 0.45 * size_similarity))
    return {
        "center_distance": round(center_distance, 4),
        "center_similarity": round(center_similarity, 4),
        "size_similarity": round(size_similarity, 4),
        "score": round(score, 4),
    }


def _greedy_match(reference_boxes: List[Dict[str, Any]], predicted_boxes: List[Dict[str, Any]], *, min_score: float = 0.18) -> Dict[str, Any]:
    candidates: List[tuple[float, int, int, Dict[str, float]]] = []
    for reference_index, reference_box in enumerate(reference_boxes):
        for predicted_index, predicted_box in enumerate(predicted_boxes):
            metrics = _pair_metrics(reference_box, predicted_box)
            if metrics["score"] < min_score:
                continue
            candidates.append((metrics["score"], reference_index, predicted_index, metrics))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_reference = set()
    matched_predicted = set()
    matches: List[Dict[str, Any]] = []
    for _, reference_index, predicted_index, metrics in candidates:
        if reference_index in matched_reference or predicted_index in matched_predicted:
            continue
        matched_reference.add(reference_index)
        matched_predicted.add(predicted_index)
        matches.append(
            {
                "reference_index": reference_index,
                "predicted_index": predicted_index,
                "reference_name": reference_boxes[reference_index].get("name"),
                "predicted_name": predicted_boxes[predicted_index].get("name"),
                "reference_box": reference_boxes[reference_index],
                "predicted_box": predicted_boxes[predicted_index],
                **metrics,
            }
        )

    return _build_match_payload(reference_boxes, predicted_boxes, matches, algorithm="greedy")


def _build_match_payload(
    reference_boxes: List[Dict[str, Any]],
    predicted_boxes: List[Dict[str, Any]],
    matches: List[Dict[str, Any]],
    *,
    algorithm: str,
) -> Dict[str, Any]:
    matched_reference = {item["reference_index"] for item in matches}
    matched_predicted = {item["predicted_index"] for item in matches}
    unmatched_reference = [
        {"reference_index": index, "reference_box": box}
        for index, box in enumerate(reference_boxes)
        if index not in matched_reference
    ]
    unmatched_predicted = [
        {"predicted_index": index, "predicted_box": box}
        for index, box in enumerate(predicted_boxes)
        if index not in matched_predicted
    ]
    return {
        "matches": matches,
        "unmatched_reference": unmatched_reference,
        "unmatched_predicted": unmatched_predicted,
        "matching_algorithm": algorithm,
    }


def _hungarian_match(reference_boxes: List[Dict[str, Any]], predicted_boxes: List[Dict[str, Any]], *, min_score: float = 0.18) -> Dict[str, Any]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_match(reference_boxes, predicted_boxes, min_score=min_score)

    if not reference_boxes or not predicted_boxes:
        return _build_match_payload(reference_boxes, predicted_boxes, [], algorithm="hungarian")

    score_matrix = np.zeros((len(reference_boxes), len(predicted_boxes)), dtype=float)
    metric_matrix: List[List[Dict[str, float]]] = []
    for reference_box in reference_boxes:
        metric_row: List[Dict[str, float]] = []
        for predicted_box in predicted_boxes:
            metrics = _pair_metrics(reference_box, predicted_box)
            metric_row.append(metrics)
        metric_matrix.append(metric_row)
    for reference_index in range(len(reference_boxes)):
        for predicted_index in range(len(predicted_boxes)):
            score_matrix[reference_index, predicted_index] = metric_matrix[reference_index][predicted_index]["score"]

    row_indices, col_indices = linear_sum_assignment(1.0 - score_matrix)
    matches: List[Dict[str, Any]] = []
    for reference_index, predicted_index in zip(row_indices.tolist(), col_indices.tolist()):
        metrics = metric_matrix[reference_index][predicted_index]
        if metrics["score"] < min_score:
            continue
        matches.append(
            {
                "reference_index": reference_index,
                "predicted_index": predicted_index,
                "reference_name": reference_boxes[reference_index].get("name"),
                "predicted_name": predicted_boxes[predicted_index].get("name"),
                "reference_box": reference_boxes[reference_index],
                "predicted_box": predicted_boxes[predicted_index],
                **metrics,
            }
        )
    return _build_match_payload(reference_boxes, predicted_boxes, matches, algorithm="hungarian")


def _box_to_pixel_rect(box: Dict[str, Any], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    pixel_bbox = box.get("pixel_bbox") or []
    pixel_bbox_source_size = box.get("pixel_bbox_source_size") or []
    if isinstance(pixel_bbox, (list, tuple)) and len(pixel_bbox) == 4:
        x = float(pixel_bbox[0] or 0.0)
        y = float(pixel_bbox[1] or 0.0)
        width = float(pixel_bbox[2] or 0.0)
        height = float(pixel_bbox[3] or 0.0)
        if isinstance(pixel_bbox_source_size, (list, tuple)) and len(pixel_bbox_source_size) == 2:
            source_width = max(float(pixel_bbox_source_size[0] or image_width), 1.0)
            source_height = max(float(pixel_bbox_source_size[1] or image_height), 1.0)
            scale_x = float(image_width) / source_width
            scale_y = float(image_height) / source_height
            x *= scale_x
            y *= scale_y
            width *= scale_x
            height *= scale_y
        return int(round(x)), int(round(y)), int(round(width)), int(round(height))
    x = int(round(float(box.get("x", 0.0)) * image_width))
    y = int(round(float(box.get("y", 0.0)) * image_height))
    width = int(round(float(box.get("width", 0.0)) * image_width))
    height = int(round(float(box.get("height", 0.0)) * image_height))
    return x, y, width, height


def _overlay_display_box(box: Dict[str, Any], predicted_ocr_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    supporting_indices = box.get("supporting_ocr_indices") or []
    if not isinstance(supporting_indices, list) or not supporting_indices:
        return box

    supporting_boxes = []
    for index in supporting_indices:
        if not isinstance(index, int):
            continue
        if index < 0 or index >= len(predicted_ocr_blocks):
            continue
        candidate = predicted_ocr_blocks[index]
        pixel_bbox = candidate.get("pixel_bbox") or []
        if not isinstance(pixel_bbox, (list, tuple)) or len(pixel_bbox) != 4:
            continue
        supporting_boxes.append(candidate)

    if not supporting_boxes:
        return box

    first_size = supporting_boxes[0].get("pixel_bbox_source_size") or []
    if not isinstance(first_size, (list, tuple)) or len(first_size) != 2:
        return box

    min_x = min(float(candidate["pixel_bbox"][0]) for candidate in supporting_boxes)
    min_y = min(float(candidate["pixel_bbox"][1]) for candidate in supporting_boxes)
    max_right = max(float(candidate["pixel_bbox"][0]) + float(candidate["pixel_bbox"][2]) for candidate in supporting_boxes)
    max_bottom = max(float(candidate["pixel_bbox"][1]) + float(candidate["pixel_bbox"][3]) for candidate in supporting_boxes)
    source_width = max(float(first_size[0]), 1.0)
    source_height = max(float(first_size[1]), 1.0)

    display_box = dict(box)
    display_box["pixel_bbox"] = [round(min_x, 2), round(min_y, 2), round(max_right - min_x, 2), round(max_bottom - min_y, 2)]
    display_box["pixel_bbox_source_size"] = [int(source_width), int(source_height)]
    display_box["x"] = round(min_x / source_width, 4)
    display_box["y"] = round(min_y / source_height, 4)
    display_box["width"] = round((max_right - min_x) / source_width, 4)
    display_box["height"] = round((max_bottom - min_y) / source_height, 4)
    display_box["right"] = round(max_right / source_width, 4)
    display_box["bottom"] = round(max_bottom / source_height, 4)
    display_box["cx"] = round((min_x + max_right) / 2.0 / source_width, 4)
    display_box["cy"] = round((min_y + max_bottom) / 2.0 / source_height, 4)
    display_box["bbox"] = [display_box["x"], display_box["y"], display_box["width"], display_box["height"]]
    return display_box


def _draw_labeled_box(image: Any, box: Dict[str, Any], label: str, color: tuple[int, int, int]) -> None:
    try:
        import cv2
    except ImportError:
        return

    image_height, image_width = image.shape[:2]
    x, y, width, height = _box_to_pixel_rect(box, image_width, image_height)
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    cv2.putText(
        image,
        label[:28],
        (x + 4, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_match_score(image: Any, point: tuple[int, int], label: str, color: tuple[int, int, int]) -> None:
    try:
        import cv2
    except ImportError:
        return

    x, y = point
    cv2.putText(
        image,
        label,
        (x + 4, max(18, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def _write_overlay_images(artifacts: ArtifactBundle, match_payload: Dict[str, Any]) -> Dict[str, str]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {}

    sample_artifact_dir = artifacts.metadata.get("sample_artifact_dir")
    if not isinstance(sample_artifact_dir, str) or not sample_artifact_dir:
        return {}

    output_dir = Path(sample_artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_image_path = artifacts.reference_image_path or artifacts.metadata.get("reference_image_path")
    predicted_image_path = artifacts.render_image_path
    if not reference_image_path or not predicted_image_path:
        return {}

    reference_image = _safe_imread(reference_image_path, cv2.IMREAD_COLOR)
    predicted_image = _safe_imread(predicted_image_path, cv2.IMREAD_COLOR)
    if reference_image is None or predicted_image is None:
        return {}

    reference_overlay = reference_image.copy()
    predicted_overlay = predicted_image.copy()
    predicted_ocr_blocks = list(match_payload.get("predicted_ocr_blocks") or [])
    predicted_display_blocks = [
        _overlay_display_box(box, predicted_ocr_blocks)
        for box in match_payload.get("predicted_blocks", [])
    ]

    matched_reference_indices = {item["reference_index"] for item in match_payload.get("matches", [])}
    matched_predicted_indices = {item["predicted_index"] for item in match_payload.get("matches", [])}
    reference_match_labels = {
        item["reference_index"]: f"M{match_index}"
        for match_index, item in enumerate(match_payload.get("matches", []), start=1)
    }
    predicted_match_labels = {
        item["predicted_index"]: f"M{match_index}"
        for match_index, item in enumerate(match_payload.get("matches", []), start=1)
    }
    for index, box in enumerate(match_payload.get("reference_blocks", [])):
        color = (0, 180, 0) if index in matched_reference_indices else (0, 0, 255)
        match_label = reference_match_labels.get(index, "")
        _draw_labeled_box(reference_overlay, box, f"R{index + 1} {match_label}:{box.get('name', '')}", color)
    for index, box in enumerate(predicted_display_blocks):
        color = (0, 180, 0) if index in matched_predicted_indices else (255, 120, 0)
        match_label = predicted_match_labels.get(index, "")
        _draw_labeled_box(predicted_overlay, box, f"P{index + 1} {match_label}:{box.get('name', '')}", color)

    reference_output_path = get_layout_overlay_artifact_path(sample_id=artifacts.sample_id, kind="reference_overlay")
    predicted_output_path = get_layout_overlay_artifact_path(sample_id=artifacts.sample_id, kind="predicted_overlay")
    cv2.imwrite(str(reference_output_path), reference_overlay)
    cv2.imwrite(str(predicted_output_path), predicted_overlay)

    overview_height = max(reference_overlay.shape[0], predicted_overlay.shape[0])
    if reference_overlay.shape[0] != overview_height:
        scale = overview_height / reference_overlay.shape[0]
        reference_overlay = cv2.resize(reference_overlay, (int(reference_overlay.shape[1] * scale), overview_height))
    if predicted_overlay.shape[0] != overview_height:
        scale = overview_height / predicted_overlay.shape[0]
        predicted_overlay = cv2.resize(predicted_overlay, (int(predicted_overlay.shape[1] * scale), overview_height))
    overview = np.hstack([reference_overlay, predicted_overlay])
    left_width = reference_overlay.shape[1]
    for match_index, item in enumerate(match_payload.get("matches", []), start=1):
        left_x, left_y, left_w, left_h = _box_to_pixel_rect(item["reference_box"], reference_overlay.shape[1], reference_overlay.shape[0])
        predicted_box = _overlay_display_box(item["predicted_box"], predicted_ocr_blocks)
        right_x, right_y, right_w, right_h = _box_to_pixel_rect(predicted_box, predicted_overlay.shape[1], predicted_overlay.shape[0])
        start_point = (left_x + left_w // 2, left_y + left_h // 2)
        end_point = (left_width + right_x + right_w // 2, right_y + right_h // 2)
        cv2.line(overview, start_point, end_point, (90, 200, 255), 1, cv2.LINE_AA)
        mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
        score_label = f"M{match_index} s={float(item.get('score', 0.0)):.2f}"
        _draw_match_score(overview, mid_point, score_label, (40, 90, 200))

    overview_output_path = get_layout_overlay_artifact_path(sample_id=artifacts.sample_id, kind="match_overview")
    cv2.imwrite(str(overview_output_path), overview)
    return {
        "reference_overlay_path": str(reference_output_path),
        "predicted_overlay_path": str(predicted_output_path),
        "match_overview_path": str(overview_output_path),
    }


def _html_text_blocks_for_completion(artifacts: ArtifactBundle) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    html_blocks, html_debug = _extract_predicted_text_blocks(artifacts)
    return html_blocks, html_debug




def _merge_strategy_a(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strategy A paragraph merge — combine OCR fragments that share x-range
    and have a tight vertical gap, into one block per paragraph.

    Conditions for merging two blocks i, j (both in normalized coords):
        - x-range IoU > 0.7   (left/right independent elements have IoU≈0,
                               so they are never merged — Phase 2 verified
                               0 false-merges on 30 infographic samples)
        - y_gap  < 0.5 × min(h_i, h_j)   (intra-paragraph line spacing is
                                          typically much smaller than half
                                          a line height)

    Iterated to fixed point: a single greedy pass can miss merges where
    extending an absorbed bbox newly satisfies the y_gap condition with a
    later block, so we repeat until no more pairs qualify.

    Phase 8 spike validated this merging step lifts layout discriminator
    score (A-C) from 0.568 → 0.666 with std(B) 0.155 → 0.172, by fixing
    the failure mode where ref-side OCR splits a paragraph into 3 lines
    while pred-side splits into 2, leaving per-line text_sim < 0.3 and
    every block scoring 0.
    """
    if not blocks:
        return blocks

    def _to_internal(b):
        return {
            "x": float(b.get("x", 0.0)),
            "y": float(b.get("y", 0.0)),
            "w": float(b.get("width", 0.0)),
            "h": float(b.get("height", 0.0)),
            "text": (b.get("text") or ""),
            "orig": b,
        }

    items = [_to_internal(b) for b in blocks]
    changed = True
    while changed:
        changed = False
        items.sort(key=lambda b: (b["y"], b["x"]))
        n = len(items)
        used = [False] * n
        new_items: List[Dict[str, Any]] = []
        for i in range(n):
            if used[i]:
                continue
            cur = items[i]
            used[i] = True
            for j in range(i + 1, n):
                if used[j]:
                    continue
                other = items[j]
                # x-range IoU
                x_lo = max(cur["x"], other["x"])
                x_hi = min(cur["x"] + cur["w"], other["x"] + other["w"])
                inter = max(0.0, x_hi - x_lo)
                union = max(cur["x"] + cur["w"], other["x"] + other["w"]) - min(cur["x"], other["x"])
                iou_x = inter / union if union > 0 else 0.0
                if iou_x <= 0.7:
                    continue
                # y gap (overlap counts as 0 gap)
                cy1 = cur["y"] + cur["h"]
                oy1 = other["y"] + other["h"]
                y_gap = max(0.0, max(cur["y"], other["y"]) - min(cy1, oy1))
                min_h = min(cur["h"], other["h"])
                if min_h <= 0 or y_gap < 0.5 * min_h:
                    # absorb other into cur (extend bbox, concat text)
                    nx = min(cur["x"], other["x"])
                    ny = min(cur["y"], other["y"])
                    nx2 = max(cur["x"] + cur["w"], other["x"] + other["w"])
                    ny2 = max(cur["y"] + cur["h"], other["y"] + other["h"])
                    cur = {
                        "x": nx,
                        "y": ny,
                        "w": nx2 - nx,
                        "h": ny2 - ny,
                        "text": (cur["text"].strip() + " " + other["text"].strip()).strip(),
                        "orig": cur["orig"],
                    }
                    used[j] = True
                    changed = True
            new_items.append(cur)
        items = new_items

    out: List[Dict[str, Any]] = []
    for it in items:
        rec = dict(it["orig"])
        rec.update({"x": it["x"], "y": it["y"], "width": it["w"], "height": it["h"], "text": it["text"]})
        out.append(rec)
    return out


def _coverage_based_score(
    reference_blocks: List[Dict[str, Any]],
    predicted_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Semantic matching + proximity-based layout scoring.

    Pre-step (Phase 8 / M5): merge OCR fragments into paragraph-level blocks
    via strategy A.  Without this, identical content split into different
    numbers of OCR lines on ref vs pred (common with varying viewport widths)
    gets text_sim < 0.3 per line and is scored 0.

    Then for each (paragraph) reference block:
      1. Find all pred blocks with text_sim >= 0.3 (SequenceMatcher)
      2. Among matched, pick the one with highest proximity score
         (Gaussian decay on center-point distance, sigma=0.15)
      3. Score = proximity (how close the matched block is spatially)

    Final score (revised 2026-06-17 from 5-case _inspect/ study):

        score = matched_mean * (0.9 + 0.1 * match_rate)

    where:
        matched_mean = mean proximity among ref blocks that found a match
        match_rate   = n_matched / n_ref

    Rationale: previous formula was mean over ALL ref blocks (unmatched=0).
    That double-penalised content gaps that TextValidator already covers,
    and made layout score collapse on samples where some ref blocks
    legitimately have no pred counterpart (OCR noise, image-text not in DOM,
    etc).  The new formula treats matched_mean as the dominant signal
    (×0.9-1.0 modulation only).  Validated on 5 _inspect samples:
    +5.36% mean, no regressions, no over-penalisation.
    """
    import math
    from difflib import SequenceMatcher as _SM

    TEXT_SIM_THRESHOLD = 0.3
    SIGMA = 0.15
    # Match-rate floor for final score:
    #   score = matched_mean * (FLOOR + (1-FLOOR) * match_rate)
    # FLOOR=0.9 → matched_mean dominates, match_rate only modulates ±10%.
    MATCH_RATE_FLOOR = 0.9

    # Paragraph-level merge before scoring (M5)
    reference_blocks = _merge_strategy_a(reference_blocks)
    predicted_blocks = _merge_strategy_a(predicted_blocks)

    if not reference_blocks:
        return {"score": 0.0, "mean_coverage": 0.0, "matched_mean": 0.0,
                "match_rate": 0.0, "match_factor": 0.0,
                "per_block": [],
                "n_ref": 0, "n_pred": 0, "n_fully_covered": 0, "n_uncovered": 0,
                "n_matched": 0}

    per_block = []
    for rb in reference_blocks:
        ref_text = (rb.get("text") or "").strip().lower()
        # ref block center in normalised coords
        rcx = float(rb.get("x", 0)) + float(rb.get("width", 0)) / 2
        rcy = float(rb.get("y", 0)) + float(rb.get("height", 0)) / 2

        best_score = 0.0
        best_match_text = ""
        best_tsim = 0.0
        best_dist: float = 999.0

        for pb in predicted_blocks:
            pred_text = (pb.get("text") or "").strip().lower()
            if not ref_text or not pred_text:
                continue
            tsim = _SM(None, ref_text, pred_text).ratio()
            if tsim < TEXT_SIM_THRESHOLD:
                continue

            pcx = float(pb.get("x", 0)) + float(pb.get("width", 0)) / 2
            pcy = float(pb.get("y", 0)) + float(pb.get("height", 0)) / 2
            dist = ((rcx - pcx) ** 2 + (rcy - pcy) ** 2) ** 0.5
            prox = math.exp(-(dist ** 2) / (2 * SIGMA ** 2))

            if prox > best_score:
                best_score = prox
                best_match_text = (pb.get("text") or "")[:30]
                best_tsim = tsim
                best_dist = dist

        per_block.append({
            "text": (rb.get("text") or "")[:30],
            "coverage": round(best_score, 4),   # keep "coverage" key for backward compat
            "matched": 1 if best_score > 0 else 0,
            "text_sim": round(best_tsim, 4),
            "center_dist": round(best_dist, 4) if best_dist < 999 else None,
            "match": best_match_text,
        })

    n_ref = len(reference_blocks)
    n_matched = sum(b["matched"] for b in per_block)

    # mean over ALL ref (kept for diagnostic logging — old metric)
    mean_coverage = sum(b["coverage"] for b in per_block) / n_ref

    # New score: matched-only mean × match-rate modulation
    if n_matched > 0:
        matched_mean = sum(b["coverage"] for b in per_block if b["matched"]) / n_matched
    else:
        matched_mean = 0.0
    match_rate = n_matched / n_ref
    match_factor = MATCH_RATE_FLOOR + (1.0 - MATCH_RATE_FLOOR) * match_rate
    score = matched_mean * match_factor

    return {
        "score": round(score, 4),
        "mean_coverage": round(mean_coverage, 4),
        "matched_mean": round(matched_mean, 4),
        "match_rate": round(match_rate, 4),
        "match_factor": round(match_factor, 4),
        "n_ref": n_ref,
        "n_pred": len(predicted_blocks),
        "n_matched": n_matched,
        "n_fully_covered": sum(1 for b in per_block if b["coverage"] >= 0.5),
        "n_uncovered": sum(1 for b in per_block if b["coverage"] < 0.01),
        "per_block": per_block,
    }


def _status_from_textblock_score(score: float, reference_coverage: float, has_matches: bool) -> ValidationStatus:
    return ValidationStatus.PASS


class SemanticLayoutValidator(BaseValidator):
    """Semantic topology verification — OCR reference vs OCR predicted.

    Algorithm: run PaddleOCR on both the reference image and the rendered HTML
    screenshot to extract text BBoxes at the same granularity (ink-level lines).

    Matching: for each ref block, find pred blocks with text similarity >= 0.3
    (SequenceMatcher), then score by center-point proximity (Gaussian decay,
    sigma=0.15).  Each ref block takes the best-matching pred block's proximity
    score.  Final score = mean across all ref blocks.

    This approach avoids the granularity mismatch between DOM nodes (paragraph-
    level) and OCR (line-level), and naturally handles many-to-many matching.
    """

    name = "SemanticLayoutValidator"
    supported_categories = ("layout",)

    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        reference_image_path = artifacts.reference_image_path or artifacts.metadata.get("reference_image_path")
        if not reference_image_path:
            return self.build_result(
                requirement,
                status=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
                message="Reference image path is unavailable for text block validation.",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
                debug={"implemented": True, "reason": "missing_reference_image"},
            )

        # --- Reference side: OCR on the reference image ---
        reference_blocks, reference_debug = _extract_reference_text_blocks(str(reference_image_path))

        # --- Predicted side: OCR on the render image (same granularity as ref) ---
        render_image_path = artifacts.render_image_path
        predicted_blocks: List[Dict[str, Any]] = []
        predicted_debug: Dict[str, Any] = {}
        if render_image_path and Path(render_image_path).exists():
            predicted_blocks, predicted_debug = _extract_reference_text_blocks(
                str(render_image_path),
                source="predicted_render_ocr",
                name_prefix="pred_ocr_text",
            )
            predicted_debug["source"] = "render_image_ocr"
        else:
            # Fallback to DOM text blocks if render image unavailable
            predicted_blocks, predicted_debug = _html_text_blocks_for_completion(artifacts)
            predicted_debug["source"] = "dom_fallback"

        if not reference_blocks:
            return self.build_result(
                requirement,
                status=ValidationStatus.FAIL,
                score=0.0,
                message="Text block validation could not extract any OCR reference text blocks.",
                evidence={
                    "evidence_basis": "fallback",
                    "evidence_sources": ["reference_image_ocr"],
                    "reference_image_path": str(reference_image_path),
                },
                debug={
                    "implemented": True,
                    "reference_debug": reference_debug,
                    "predicted_debug": predicted_debug,
                },
            )

        if not predicted_blocks:
            return self.build_result(
                requirement,
                status=ValidationStatus.FAIL,
                score=0.0,
                message="No text blocks could be extracted from the predicted output.",
                evidence={
                    "evidence_basis": "fallback",
                    "evidence_sources": ["predicted_render_ocr"],
                    "reference_image_path": str(reference_image_path),
                },
                debug={
                    "implemented": True,
                    "reference_debug": reference_debug,
                    "predicted_debug": predicted_debug,
                },
            )

        # --- Semantic matching + proximity scoring ---
        coverage_result = _coverage_based_score(reference_blocks, predicted_blocks)
        score = coverage_result["score"]
        status = ValidationStatus.PASS

        artifacts.layout_matching = {
            "reference_blocks": reference_blocks,
            "predicted_blocks": predicted_blocks,
            "per_block": coverage_result["per_block"],
            "summary": {
                "reference_block_count": len(reference_blocks),
                "predicted_block_count": len(predicted_blocks),
                "score": score,
                "mean_coverage": coverage_result["mean_coverage"],
                "matched_mean": coverage_result["matched_mean"],
                "match_rate": coverage_result["match_rate"],
                "n_matched": coverage_result["n_matched"],
                "n_fully_covered": coverage_result["n_fully_covered"],
                "n_uncovered": coverage_result["n_uncovered"],
                "predicted_source": predicted_debug.get("source", "render_image_ocr"),
                "matching_strategy": "semantic_proximity",
            },
        }
        overlay_artifacts = _write_overlay_images(artifacts, artifacts.layout_matching)
        if overlay_artifacts:
            artifacts.layout_matching["artifacts"] = overlay_artifacts
        artifacts.metadata["layout_matching"] = artifacts.layout_matching

        message = (
            f"Layout semantic matching: {coverage_result['n_fully_covered']}/{len(reference_blocks)} ref blocks matched, "
            f"score={score:.3f}."
        )

        return self.build_result(
            requirement,
            status=status,
            score=score,
            measured=dict(artifacts.layout_matching["summary"]),
            message=message,
            evidence={
                "evidence_basis": "hybrid",
                "evidence_sources": ["reference_image_ocr", "predicted_render_ocr"],
                "reference_image_path": str(reference_image_path),
                "render_image_path": artifacts.render_image_path,
                "per_block_coverage_preview": coverage_result["per_block"][:12],
            },
            debug={
                "implemented": True,
                "reference_debug": reference_debug,
                "predicted_debug": predicted_debug,
                "matching_algorithm": "semantic_proximity",
                "n_uncovered": coverage_result["n_uncovered"],
            },
        )


# ---------------------------------------------------------------------------
# BoundaryValidator  (overflow + size — formerly split across gate layer)
# ---------------------------------------------------------------------------


def _require_render_image(artifacts: ArtifactBundle) -> "Path | None":
    """Return resolved render image Path, or None if missing/unreadable."""
    render_path = artifacts.render_image_path
    if not render_path:
        return None
    path = Path(render_path)
    return path if path.exists() else None


class BoundaryValidator(BaseValidator):
    """Verify that the rendered output stays within its intended boundaries.

    Checks two independent aspects and averages their scores:

    Overflow — area-based penalty::

        For each element, compute total_area and clipped_area (within viewport).
        S_overflow = sum(clipped_area) / sum(total_area)
        This measures the fraction of element area that stays inside bounds.

    Size (aspect-ratio) — exponential decay on relative error::

        S_size = exp(−λ × |r_pred − r_ref| / r_ref)

    Final score = (S_overflow + S_size) / 2   (both ∈ (0, 1]).

    Both sub-scores are reported independently in *measured* for
    interpretability.  If size metadata is unavailable the size factor
    is treated as 1.0 (overflow alone is still validated).
    """

    name = "BoundaryValidator"
    supported_categories = ("boundary", "size")

    # Size (aspect-ratio)
    SIZE_LAMBDA: float = 2.5



    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        import math

        render_path = _require_render_image(artifacts)
        if render_path is None:
            return self.build_result(
                requirement,
                status=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
                message="Boundary check skipped: render image unavailable.",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
            )

        # ── Overflow (area-based) ─────────────────────────────────────
        overflow_count = len(artifacts.overflowing_elements) if artifacts.overflowing_elements else 0

        # Get viewport/poster bounds for clipping
        poster_w = (artifacts.dom_snapshot or {}).get("poster_width", 0)
        poster_h = (artifacts.dom_snapshot or {}).get("poster_height", 0)
        if not poster_w or not poster_h:
            # Fallback: try render image size
            from PIL import Image as PILImage
            img = PILImage.open(render_path)
            poster_w, poster_h = img.size

        total_area = 0.0
        clipped_area = 0.0
        max_overflow_px = 0

        if overflow_count:
            for elem in artifacts.overflowing_elements:
                el_left = float(elem.get("left", 0))
                el_top = float(elem.get("top", 0))
                el_right = float(elem.get("right", 0))
                el_bottom = float(elem.get("bottom", 0))
                el_w = max(0, el_right - el_left)
                el_h = max(0, el_bottom - el_top)
                el_area = el_w * el_h
                if el_area <= 0:
                    continue

                # Clip element rect to poster bounds
                clip_left = max(el_left, 0)
                clip_top = max(el_top, 0)
                clip_right = min(el_right, poster_w)
                clip_bottom = min(el_bottom, poster_h)
                clip_w = max(0, clip_right - clip_left)
                clip_h = max(0, clip_bottom - clip_top)
                clip_area = clip_w * clip_h

                total_area += el_area
                clipped_area += clip_area

                elem_overflow = max(int(elem.get("overflow_x", 0)), int(elem.get("overflow_y", 0)))
                max_overflow_px = max(max_overflow_px, elem_overflow)

        if total_area > 0:
            overflow_score = clipped_area / total_area
        else:
            overflow_score = 1.0  # no overflowing elements = perfect

        # ── Size (aspect-ratio) ───────────────────────────────────────
        reference_wh = (
            (requirement.expected or {}).get("reference_image_wh")
            or artifacts.metadata.get("reference_image_wh")
            or ""
        )
        size_score = 1.0
        size_measured: Dict[str, Any] = {}
        if isinstance(reference_wh, str) and "x" in reference_wh:
            from PIL import Image as PILImage  # type: ignore[import-not-found]

            img = PILImage.open(render_path)
            actual_w, actual_h = img.size
            declared_w, declared_h = map(int, reference_wh.lower().split("x", 1))
            actual_ratio = float(actual_w) / float(max(actual_h, 1))
            declared_ratio = float(declared_w) / float(max(declared_h, 1))
            relative_delta = abs(actual_ratio - declared_ratio) / max(abs(declared_ratio), 1e-6)

            lam = float((requirement.tolerance or {}).get("size_lambda", self.SIZE_LAMBDA))
            size_score = math.exp(-lam * relative_delta)
            size_measured = {
                "actual": [actual_w, actual_h],
                "declared": [declared_w, declared_h],
                "actual_aspect_ratio": round(actual_ratio, 4),
                "declared_aspect_ratio": round(declared_ratio, 4),
                "relative_aspect_ratio_delta": round(relative_delta, 4),
                "size_lambda": lam,
            }

        # ── Combined score ────────────────────────────────────────────
        score = round((overflow_score + size_score) / 2.0, 4)

        message = f"Boundary check completed (score={score:.3f})."

        measured: Dict[str, Any] = {
            "overflow_score": round(overflow_score, 4),
            "size_score": round(size_score, 4),
            "overflowing_element_count": overflow_count,
            "max_overflow_px": max_overflow_px,
            "total_overflow_area": round(total_area, 1),
            "clipped_area": round(clipped_area, 1),
            "poster_width": poster_w,
            "poster_height": poster_h,
            **size_measured,
        }

        return self.build_result(
            requirement,
            status=ValidationStatus.PASS,
            score=score,
            measured=measured,
            message=message,
            evidence={
                "evidence_basis": "explicit",
                "evidence_sources": ["render_image", "reference_metadata"],
                "overflowing_elements_preview": (artifacts.overflowing_elements or [])[:5],
                "render_overflow_debug_path": artifacts.metadata.get("render_overflow_debug_path"),
            },
        )