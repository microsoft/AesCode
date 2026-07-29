"""HTML generation prompts and utilities for building the user text."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Refine-stage feature flag: when enabled, the refine prompt includes a second
# image (a low-resolution render of the previous HTML) in addition to the
# reference image. SFT/RL/Inference/Eval all read this same env var so the
# behavior is consistent across the stack.
#
# Default: off (backward-compatible: existing single-image checkpoints keep
# working). Set REFINE_USE_PREV_RENDER=1 to enable.
# ---------------------------------------------------------------------------
def refine_use_prev_render() -> bool:
    """Return True iff the refine stage should include a prev-render image."""
    val = os.environ.get("REFINE_USE_PREV_RENDER", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def no_image_mode() -> bool:
    """Return True iff RL/data-prep should build no-image prompts (VREDO_NO_IMAGE=1).

    Default off: image-based SFT/RL/eval keep the reference <image> token and
    feed images to the model. Set VREDO_NO_IMAGE=1 to strip image input while
    still using the reference image for reward computation.
    """
    val = os.environ.get("VREDO_NO_IMAGE", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


# Recommended pixel budget for the prev-render image (~252 vision tokens at
# 588x336). Both SFT (qwen_vl_utils min/max_pixels) and the inference
# providers (when resizing client-side) should clamp to this.
REFINE_PREV_RENDER_MAX_PIXELS = 512 * 28 * 28   # 401_408
REFINE_PREV_RENDER_MIN_PIXELS = 128 * 28 * 28   # 100_352
REFINE_PREV_RENDER_WH = (588, 336)              # exact resolution to render to


_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


DEFAULT_SYSTEM_PROMPT = (
    "You create polished standalone HTML/CSS infographics, slides, and editorial layouts. "
    "You receive a generation prompt and a reference image. "
    "The generation prompt is the primary contract: follow it exactly wherever it specifies required text, charts, tables, or data. "
    "The reference image is visual guidance only: reuse strong ideas in hierarchy, palette, density, composition, and typography, but do not treat it as a screenshot to copy. "
    "Its visible text, numbers, and labels may be incomplete, inaccurate, or stylistically decorative, so do not trust them over the prompt or inline data contracts. "
    "HTML output constraints: "
    "(1) Produce exactly one complete <!DOCTYPE html> document with all CSS in a single inline <style> block. "
    "(2) The root artboard element (e.g. .poster, .artboard, or body > div) MUST declare explicit width and height in pixels matching the canvas size specified in the prompt. Use a single top-level container with fixed px dimensions — never 100vh, 100%, or viewport-relative units for the root canvas. "
    "(3) External resources are limited to https://placehold.co for content images and https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js for charts. "
    "(4) ECharts series.data must be literal arrays, never computed at runtime. "
    "(5) Inline '(Data contract [...])' clauses contain exact required values and must be copied verbatim into ECharts series.data or visible HTML table cells. "
    "(6) Tables must use real <table>/<tr>/<td> elements with visible text. "
    "(7) The root artboard must use overflow:hidden, and all content must fit inside the declared canvas size. "
    "(8) Each chart, table, title, heading, and footer must remain a separate DOM element, not merged into a decorative background or shared node. "
    "When refining a previous implementation, you also receive a low-resolution screenshot of that implementation; compare it against the reference image to identify visual discrepancies and fix the HTML accordingly."
)

DEFAULT_USER_TEMPLATE = """Create one complete HTML infographic page that follows the generation prompt faithfully and uses the reference image only as visual design guidance.

[Reference Image]
{reference_note}

[Canvas]
{canvas_spec}

[Generation Prompt]
{generation_prompt}

[Output]
Briefly describe your design approach (1-2 sentences), then output the complete HTML in a ```html code block.
"""

# Marker used to locate the [Output] section in a rendered user template so that
# the refine pipeline can replace it with the refinement instruction.
OUTPUT_SECTION_MARKER = "\n[Output]"

# Suffix appended after the [Generation Prompt] section in refine-step prompts.
# The {prev_html} placeholder is filled at call time.
# Format matches SFT repair format: brief diagnosis + HTML fence.
#
# NOTE: Kept as a string constant for backward-compatible imports. Prefer
# build_refine_suffix() below, which handles the optional prev-render image
# notice via the REFINE_USE_PREV_RENDER feature flag.
REFINE_SUFFIX = (
    "\n\n[Previous Implementation]\n"
    "```html\n{prev_html}\n```\n\n"
    "Briefly diagnose the key problems (2-3 sentences: structural errors first, "
    "then visual/content issues), then output the complete corrected HTML in a "
    "```html code block."
)

# Prefix used when the refine prompt also carries a low-res screenshot of the
# previous implementation as a second image. The reference image is image #1,
# the previous render is image #2. We state this explicitly so the model
# doesn't confuse the two.
_REFINE_PREV_RENDER_NOTE = (
    "\n\n[Previous Implementation]\n"
    "Image 1 above is the target reference. Image 2 above is a low-resolution "
    "screenshot of your previous implementation — use it to spot visual gaps "
    "(layout, proportions, palette, missing elements) versus the reference.\n"
    "```html\n{prev_html}\n```\n\n"
    "Briefly diagnose the key problems (2-3 sentences: structural errors first, "
    "then visual/content issues), then output the complete corrected HTML in a "
    "```html code block."
)


def build_refine_suffix(prev_html: str, with_prev_render: bool | None = None) -> str:
    """Build the refine suffix appended after the base user text.

    Args:
        prev_html: the previous HTML implementation to inject between fences.
        with_prev_render: whether the message also carries a 2nd image (the
            prev-render screenshot). If None, falls back to the global
            REFINE_USE_PREV_RENDER env flag.

    Returns:
        The fully-formatted suffix string (already includes leading "\\n\\n").
    """
    if with_prev_render is None:
        with_prev_render = refine_use_prev_render()
    template = _REFINE_PREV_RENDER_NOTE if with_prev_render else REFINE_SUFFIX
    return template.format(prev_html=prev_html)


# ---------------------------------------------------------------------------
# Dedicated refine user template (with prev-render image)
# ---------------------------------------------------------------------------
REFINE_USER_TEMPLATE = """You are refining a previous HTML implementation. Compare it against the reference image and the generation prompt, then fix the code.

[Reference Image (Image 1)]
{reference_note}

[Canvas]
{canvas_spec}

[Generation Prompt]
{generation_prompt}

[Previous Render (Image 2)]
Image 2 (attached above) is a low-resolution screenshot of your previous implementation. Compare it against the reference image (Image 1) to spot visual discrepancies — layout, proportions, palette, spacing, and missing elements.

[Previous Code]
```html
{prev_html}
```

Diagnose the key problems (2-3 sentences: structural errors first, then visual/content issues), then output the complete corrected HTML in a ```html code block.
"""

# Variant without prev-render image (single-image fallback)
REFINE_USER_TEMPLATE_SINGLE_IMAGE = """You are refining a previous HTML implementation. Compare it against the reference image and the generation prompt, then fix the code.

[Reference Image]
{reference_note}

[Canvas]
{canvas_spec}

[Generation Prompt]
{generation_prompt}

[Previous Code]
```html
{prev_html}
```

Diagnose the key problems (2-3 sentences: structural errors first, then visual/content issues), then output the complete corrected HTML in a ```html code block.
"""


def build_refine_user_text(
    sample: Dict[str, Any],
    prev_html: str,
    *,
    has_image: bool = False,
    with_prev_render: bool | None = None,
) -> str:
    """Build the complete user text for a refine task.

    Uses REFINE_USER_TEMPLATE (dual-image) or REFINE_USER_TEMPLATE_SINGLE_IMAGE
    depending on whether a prev-render image is attached.
    """
    if with_prev_render is None:
        with_prev_render = refine_use_prev_render()

    template = REFINE_USER_TEMPLATE if with_prev_render else REFINE_USER_TEMPLATE_SINGLE_IMAGE

    reference_metadata = sample.get("reference_metadata") or {}
    generation_prompt = sample.get("generation_prompt", sample.get("refine_prompt", "")).strip()
    requirements = sample.get("verifiable_requirements")
    if isinstance(requirements, list):
        generation_prompt = _inject_data_contracts(generation_prompt, requirements)

    wh = reference_metadata.get("reference_image_wh", "")
    orientation = reference_metadata.get("reference_orientation", "")
    canvas_note = f" ({wh}, {orientation})" if wh else ""

    if with_prev_render and has_image:
        reference_note = (
            f"Image 1 (attached above){canvas_note} is the target reference. "
            "Use it for visual inspiration: color palette, layout grid, typographic hierarchy, section arrangement, and overall design quality. "
            "Its visible text, numbers, and labels may be inaccurate or incomplete — do not copy them literally unless the generation prompt or inline data contracts require them."
        )
    elif has_image:
        reference_note = (
            f"The reference image is attached above{canvas_note}. "
            "Use it for visual inspiration only: color palette, layout grid, typographic hierarchy, section arrangement, and overall design quality. "
            "Its visible text, numbers, and labels may be inaccurate or incomplete — do not copy them literally unless the generation prompt or inline data contracts require them."
        )
    else:
        reference_note = "No reference image is available. Follow the generation prompt's visual direction closely."

    if wh and "x" in wh:
        w_str, h_str = wh.split("x", 1)
        canvas_spec = (
            f"Output size: {w_str}×{h_str} px ({orientation}). "
            f"Your root container element MUST set width: {w_str}px; height: {h_str}px; overflow: hidden. "
            f"Do not use percentage, vh, vw, or other relative units for the root canvas dimensions."
        )
    else:
        canvas_spec = (
            "No specific canvas size given. Default to width: 1280px; height: 720px; overflow: hidden "
            "on the root container element."
        )

    return template.format(
        reference_note=reference_note,
        canvas_spec=canvas_spec,
        generation_prompt=generation_prompt,
        prev_html=prev_html,
    )


# Canonical [Output] instruction for generate tasks across SFT, RL, and inference.
# Matches the SFT generate format so train/inference distributions are identical.
# Inference `extract_html_document()` already handles ```html fences gracefully.
GENERATE_OUTPUT_INSTRUCTION = (
    "\n\n[Output]\n"
    "Briefly describe your design approach (1-2 sentences), "
    "then output the complete HTML in a ```html code block."
)

# COT prefix used for generate-task assistant outputs so that the format is
# consistent with repair tasks (diagnosis + HTML fence).
GENERATE_COT_STUB = "Here is the complete implementation following the design specifications."

# Legacy user text for dir-format datasets (no generation_prompt available).
# Shared across SFT and RL for consistency.
LEGACY_USER_GENERATE = (
    "Implement the design shown in the screenshot as a single, complete HTML file. "
    "Use a single <style> block in <head> for all CSS. "
    "Match the layout, typography, colours, and spacing as closely as possible.\n\n"
    "Briefly describe your approach (1-2 sentences), then output the complete HTML "
    "in a ```html code block."
)

LEGACY_USER_REPAIR = (
    "Below is a previous HTML implementation of the design shown in the screenshot. "
    "Compare it with the reference image and identify the main issues.\n\n"
    "Previous implementation:\n"
    "```html\n{buggy_html}\n```\n\n"
    "Briefly diagnose the key problems (2-3 sentences: structural errors first, "
    "then visual/content issues), then output the complete corrected HTML in a "
    "```html code block."
)


def _sentence_end_after(text: str, from_pos: int) -> int:
    """Return the index just after the sentence boundary nearest to from_pos.

    Skips:
    - Periods inside quoted titles like "4. Global…" (digit before period)
    - Periods followed by a lowercase letter (abbreviations, decimals)
    - Periods inside double-quotes that are followed by a space + uppercase
      when the surrounding context looks like a quoted title
    Accepts:
    - `<alpha>. <uppercase>` — normal sentence end
    - `<alpha>.\n` — sentence end at newline
    """
    # Match period preceded by a letter and followed by space+uppercase, or newline
    # Exclude period preceded by digit (list items: "1. ", "4. ") or inside quotes
    for m in re.finditer(r'(?<=[a-zA-Z])\. +(?=[A-Z])|(?<=[a-zA-Z])\.$|(?<=[a-zA-Z])\.\n', text[from_pos:]):
        end = from_pos + m.end()
        # Skip if the period is preceded or followed by a digit context (e.g. "4. Global")
        preceding = text[max(0, end - m.end() + from_pos - 5): end - m.end() + from_pos + 5]
        _ = preceding  # context captured but heuristic already in regex
        return end
    # Fallback: bare newline
    nl = text.find('\n', from_pos)
    return nl + 1 if nl != -1 else len(text)


def _find_insertion_point(text: str, req: Dict[str, Any]) -> int:
    """Return the character index after the sentence that best describes this requirement."""
    category = req.get("category", "")
    target = req.get("target") or {}
    expected = req.get("expected") or {}
    data = expected.get("data") or []
    tl = text.lower()

    def first_match(keywords: List[str]) -> int:
        for kw in keywords:
            pos = tl.find(kw.lower())
            if pos != -1:
                return pos
        return -1

    if category == "chart_data":
        subtype = str(target.get("chart_subtype") or "").lower()
        count = len(data)
        if subtype == "combo":
            pos = first_match(["combo chart", "combo"])
        elif subtype == "pie":
            # Distinguish multiple pie series by segment count word
            kws: List[str] = []
            for word, n in _COUNT_WORDS.items():
                if n == count:
                    kws += [f"{word} segment", f"{word} ring", f"{word}-segment"]
            kws += [f"{count} segment", f"{count} ring"]
            kws += ["inner ring", "smaller ring"] if count <= 4 else ["outer ring", "larger ring"]
            kws += ["donut", "pie chart", "ring"]
            pos = first_match(kws)
        elif subtype in ("bar", "bar_horizontal"):
            pos = first_match(["horizontal bar", "bar chart", "bar"])
        else:
            pos = first_match([subtype + " chart", subtype])

    elif category == "table_data":
        # Use a distinctive row-label (non-digit, >3 chars) to anchor the table mention
        kws = []
        for row in (data[1:] if isinstance(data, list) else []):
            if isinstance(row, list) and row:
                label = str(row[0]).strip()
                if len(label) > 3 and not label.isdigit():
                    kws.append(label)
        kws += ["visible table", "html table", "actual table", "table"]
        pos = first_match(kws)
    else:
        pos = -1

    if pos == -1:
        return len(text)  # fallback: end of prompt
    return _sentence_end_after(text, pos)


def _format_data_snippet(req: Dict[str, Any]) -> str:
    """Format a chart_data or table_data requirement as an inline parenthetical."""
    req_id = str(req.get("id") or "")
    category = req.get("category", "")
    expected = req.get("expected") or {}
    target = req.get("target") or {}
    data = expected.get("data") or []

    if category == "chart_data":
        subtype = str(target.get("chart_subtype") or "chart")
        match_mode = str(expected.get("match_mode") or "exact").lower()
        if match_mode == "count":
            return (
                f" (Data contract [{req_id}]: ECharts {subtype} must render "
                f"exactly {len(data)} segments.)"
            )
        if all(isinstance(item, dict) for item in data):
            rows_str = ", ".join(
                "{" + ", ".join(f"{k}: {v}" for k, v in item.items()) + "}"
                for item in data
            )
            return (
                f" (Data contract [{req_id}]: ECharts {subtype} series.data must be "
                f"[{rows_str}].)"
            )
        else:
            vals = ", ".join(str(v) for v in data)
            return (
                f" (Data contract [{req_id}]: ECharts {subtype} must have exactly "
                f"{len(data)} segments with literal values [{vals}].)"
            )

    elif category == "table_data":
        col_widths: List[int] = []
        for row in data:
            if not isinstance(row, list):
                continue
            for i, cell in enumerate(row):
                w = len(str(cell))
                if i >= len(col_widths):
                    col_widths.append(w)
                else:
                    col_widths[i] = max(col_widths[i], w)
        table_rows = []
        for row in data:
            if not isinstance(row, list):
                continue
            cells = [
                str(cell).ljust(col_widths[i]) if i < len(col_widths) else str(cell)
                for i, cell in enumerate(row)
            ]
            table_rows.append("  | " + " | ".join(cells) + " |")
        return (
            f" (Data contract [{req_id}]: render as HTML <table> with these exact rows:\n"
            + "\n".join(table_rows)
            + ")"
        )

    return ""


def _inject_data_contracts(generation_prompt: str, requirements: List[Dict[str, Any]]) -> str:
    """Insert each chart_data/table_data data contract inline after its sentence in the prompt."""
    active = [
        req for req in requirements
        if req.get("enabled", True)
        and req.get("category") in ("chart_data", "table_data")
        and isinstance((req.get("expected") or {}).get("data"), list)
        and (req.get("expected") or {}).get("data")
    ]
    if not active:
        return generation_prompt

    # Collect (insertion_point, snippet) pairs; inject in reverse order to preserve offsets
    insertions: List[tuple[int, str]] = []
    for req in active:
        pos = _find_insertion_point(generation_prompt, req)
        snippet = _format_data_snippet(req)
        if snippet:
            insertions.append((pos, snippet))

    insertions.sort(key=lambda x: x[0], reverse=True)
    result = generation_prompt
    for pos, snippet in insertions:
        result = result[:pos] + snippet + result[pos:]
    return result


def build_user_text(sample: Dict[str, Any], user_template: str, *, has_image: bool = False) -> str:
    reference_metadata = sample.get("reference_metadata") or {}
    generation_prompt = sample.get("generation_prompt", sample.get("refine_prompt", "")).strip()
    requirements = sample.get("verifiable_requirements")
    if isinstance(requirements, list):
        generation_prompt = _inject_data_contracts(generation_prompt, requirements)
    wh = reference_metadata.get("reference_image_wh", "")
    orientation = reference_metadata.get("reference_orientation", "")
    canvas_note = f" ({wh}, {orientation})" if wh else ""
    reference_note = (
        f"The reference image is attached above{canvas_note}. "
        "Use it for visual inspiration only: color palette, layout grid, typographic hierarchy, section arrangement, and overall design quality. "
        "Its visible text, numbers, and labels may be inaccurate or incomplete — do not copy them literally unless the generation prompt or inline data contracts require them."
        if has_image else
        "No reference image is available. Follow the generation prompt's visual direction closely."
    )

    # Build explicit canvas specification
    if wh and "x" in wh:
        w_str, h_str = wh.split("x", 1)
        canvas_spec = (
            f"Output size: {w_str}×{h_str} px ({orientation}). "
            f"Your root container element MUST set width: {w_str}px; height: {h_str}px; overflow: hidden. "
            f"Do not use percentage, vh, vw, or other relative units for the root canvas dimensions."
        )
    else:
        canvas_spec = (
            "No specific canvas size given. Default to width: 1280px; height: 720px; overflow: hidden "
            "on the root container element."
        )

    return user_template.format(
        id=sample.get("id", ""),
        generation_prompt=generation_prompt,
        reference_note=reference_note,
        canvas_spec=canvas_spec,
    )
