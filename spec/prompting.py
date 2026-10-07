"""System/user prompts, contract building, payload normalization, and response parsing."""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Dict, List

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

from spec.schema import DEFAULT_EVALUATION_POLICY, DEFAULT_SEMANTIC_ATTRIBUTE_CONTRACT


SPEC_SYSTEM_PROMPT = (
    "You write assessment-oriented image-generation prompts by analysing one real reference infographic. "
    "Given one reference image, return exactly one JSON object and nothing else. "
    "Your job is to construct a generation prompt that preserves the meaningful information load and evaluation value of the reference, not to demand a pixel-identical recreation. "
    "Use the reference image as the primary source of truth to infer the main layout, information density, reading flow, major visible text, charts, tables, and other important graphic structure. For visual style (palette, typography, iconography, decoration) capture ONLY a coarse category-level cue (e.g. cool/warm palette, serif/sans-serif, flat/illustrative) — do not describe specific colors, font families, or fine decorative detail; leave those for the downstream image generator to fill in. "
    "Focus on what is clear, salient, and worth evaluating. Omit illegible details, tiny decorative elements, and incidental clutter unless they are clearly central to the image's meaning. "
    "Do not guess corrupted text, partially legible OCR fragments, or decorative microcopy. If a string is uncertain, omit it rather than hallucinating it. "
    "Every text_exact requirement must be backed by the generation_prompt itself: if exact visible text matters, explicitly include that exact string in the generation_prompt before extracting it as a requirement. "
    "The generation_prompt should be one coherent instruction block that describes the intended infographic, including the important mix of layout, design, text, charts, and tables, while keeping the requested information volume reasonable and intentional. "
    "Only create high-confidence structured requirements, and only use text_exact, chart_data, and table_data. "
    "Use exact schema field names and enum values only. Do not output semantic_attribute_contract or evaluation_policy. "
    "IMPORTANT — Charts: When the reference image contains any chart (bar, line, pie, combo, etc.), the generation_prompt MUST explicitly instruct the downstream generator to render it as an ECharts chart with literal data arrays. For example: 'Render this as an ECharts bar chart with literal series.data values [...].' Never leave the chart implementation method ambiguous. "
    "IMPORTANT — Non-reproducible visuals: The downstream HTML generator can only produce HTML/CSS code and cannot render real photographs, complex illustrations, hand-drawn art, brand logos, or photographic imagery. "
    "When the reference image contains such elements (e.g. photos of people, office scenes, product shots, landscapes, brand logos, vector illustrations, hand-drawn artwork), the generation_prompt MUST instruct the generator to use a https://placehold.co placeholder image with appropriate dimensions and a brief label describing what the image represents, rather than attempting to reproduce the visual with code. "
    "For example: 'Place a 400×300 placeholder image (https://placehold.co/400x300?text=Team+Photo) where the team photograph appears.' "
    "Simple geometric icons, flat color shapes, and decorative CSS elements are fine to describe for code reproduction; only use placeholders for content that requires actual photography or complex artwork."
)

SPEC_USER_TEMPLATE = (
    "Please write an image-generation prompt for later evaluation based on the reference image. Return one JSON object that matches the required schema.\n\n"
    "[Reference Metadata]\n"
    "- id: {id}\n"
    "- reference image size: {reference_image_wh}\n"
    "- computed reference aspect ratio: {reference_aspect_ratio}\n"
    "- computed reference orientation: {reference_orientation}\n\n"
    "Requirements:\n"
    "1) generation_prompt should be one coherent image-generation prompt body, not a checklist dump. It should describe a plausible infographic with enough information content to support later assessment, without over-specifying every small visible detail. Do not exceed 3000 characters. If you find yourself listing every text string or structural detail verbatim, compress into descriptive paragraphs instead.\n"
    "2) Preserve the important information mix seen in the reference when it is clear: major layout zones, key headings, charts, tables, and quantitative blocks. For visual style, include only one short coarse cue (palette temperature, typography family class, decoration density) — do NOT describe specific colors, font names, icon styles, gradients, or textures. Do not force exact reproduction of ambiguous, illegible, or low-value details.\n"
    "3) Use text_exact for the most important clearly legible visible strings that are worth evaluating in the final HTML. Every text_exact string must appear explicitly and verbatim inside generation_prompt; if it is not written into generation_prompt, do not include it in verifiable_requirements. Prioritise by evaluation value: strings that are central to the infographic's meaning, structure, or data story. Cap at 12 text_exact requirements and calibrate to the image — information-dense infographics may use more, simple ones fewer. Use chart_data only for real charts whose literal values should remain in ECharts data. Use table_data for real visible tables or matrix blocks whose rows and cells should remain visible in HTML. Do not create structured requirements for layout, order, grouping, repeated structure, or count.\n"
    "4) Tell the downstream HTML generator to expose correctness through separate visible text nodes, natural DOM structure, literal ECharts data arrays for chart_data, and actual HTML tables for table_data. Do not make custom data-entity/data-role/data-scope naming a requirement.\n"
    "5) Charts MUST be ECharts: When the reference contains any chart, the generation_prompt must explicitly say to render it using ECharts with literal series.data arrays. Example phrasing: 'Render as an ECharts [type] chart with literal data values [...].' Do not describe charts generically — always specify ECharts.\n"
    "6) Non-reproducible visuals MUST use placeholders: The downstream generator produces HTML/CSS only — it cannot render photographs, complex illustrations, hand-drawn art, or brand logos. When such content appears in the reference, the generation_prompt must instruct the generator to use a https://placehold.co/WxH?text=Description placeholder image. Describe what the placeholder represents and give approximate dimensions. Simple geometric icons, flat CSS shapes, and decorative elements can still be described for code reproduction.\n"
    "7) Return valid JSON only.\n\n"
    "[Output Rules]\n"
    "- The top-level object must contain exactly these keys: id, reference_metadata, generation_prompt, verifiable_requirements.\n"
    "- Copy id exactly from the provided metadata.\n"
    "- reference_metadata must use the provided declared aspect ratio label, reference image size, computed reference aspect ratio, and computed reference orientation.\n"
    "- verifiable_requirements must use only these categories: text_exact, chart_data, table_data.\n"
    "- Use only schema-legal field names and enum values. Do not invent undeclared fields. For text_exact output only id, category, and expected.text — no target, no normalization. For chart_data include target with chart_type and chart_subtype. For table_data output only id, category, and expected.data — no target needed.\n"
    "- Be conservative and selective: prefer fewer high-confidence requirements over broader but noisy extraction.\n"
    "- generation_prompt must not exceed 3000 characters. If it is longer, trim by removing low-value detail and collapsing lists into descriptive phrases.\n"
    "- text_exact is a validation-point extraction from generation_prompt, not an independent source of truth. If an exact string is not present verbatim in generation_prompt, omit it from verifiable_requirements. Select the most important strings that are worth evaluating; cap at 12 and balance against the image's information density.\n"
    "- If text appears corrupted, guessed, partially legible, or linguistically incoherent, omit it rather than preserving a bad transcription.\n"
    "- For multi-line titles or clearly separated headers, preserve that visible granularity instead of merging everything into one requirement.\n"
    "- Use chart_data only when the numeric payload is reasonably clear, and use table_data only when a real visible table or matrix is clearly present. Pair numeric requirements with a small number of text_exact items when titles or headers are needed to disambiguate the data block.\n"
    "\n"
    "[Schema Contract]\n{schema_contract}"
)

# Backward-compatible aliases — prefer the SPEC_* names to avoid confusion
# with the identically-named symbols in inference.prompts.
DEFAULT_SYSTEM_PROMPT = SPEC_SYSTEM_PROMPT
DEFAULT_USER_TEMPLATE = SPEC_USER_TEMPLATE


# ── Detailed spec variant ─────────────────────────────────────────────────────
# Tightens the base contract so the model copies visible text verbatim and
# emits well-formed chart_data/table_data payloads. This is the variant used to
# build the released specs; use build_detailed_spec_prompts() to apply it.

SPEC_DETAIL_SUFFIX = (
    "\n\nCritical output requirements:\n"
    "- generation_prompt should be 1200-2000 characters. Focus on information density, not length.\n"
    "- Prioritize these in order: (1) all clearly legible text copied verbatim, "
    "(2) chart/table data values and labels, (3) overall layout structure and section arrangement, "
    "(4) color palette using specific color names (e.g. teal, navy blue, coral red).\n"
    "- Do NOT pad with generic descriptions of typography, spacing, readability, or visual hierarchy. "
    "Every sentence should convey concrete, image-specific information.\n"
    "- Copy each distinct heading, label, and body text block verbatim into generation_prompt. "
    "Do not summarize or paraphrase visible text.\n\n"
    "STRICT chart_data and table_data formatting rules:\n"
    "- chart_data requirements MUST include a \"target\" object with \"chart_type\": \"echarts\" "
    "and \"chart_subtype\" set to one of: bar, bar_horizontal, line, pie, combo, scatter, funnel, heatmap, radar. "
    "Requirements missing target or chart_subtype will be DISCARDED.\n"
    "- For donut/pie charts, use chart_subtype \"pie\". For area charts, use \"line\". "
    "For horizontal bars, use \"bar_horizontal\".\n"
    "- chart_data expected.data must be a flat array of numeric values, e.g. [85, 15] or [12, 18, 24]. "
    "Do NOT use {\"data\": [85]} alone — include ALL data points visible in the chart.\n"
    "- table_data expected.data must be a 2D array where the first row is headers, "
    "e.g. [[\"Region\",\"Q1\"],[\"North\",12]]. Do NOT put headers in a separate \"headers\" field.\n"
    "- Correct chart_data example: "
    "{\"id\":\"req_pie\",\"category\":\"chart_data\","
    "\"target\":{\"chart_type\":\"echarts\",\"chart_subtype\":\"pie\"},"
    "\"expected\":{\"data\":[85,15]}}\n"
    "- Correct table_data example: "
    "{\"id\":\"req_table\",\"category\":\"table_data\","
    "\"expected\":{\"data\":[[\"Category\",\"Value\"],[\"A\",100],[\"B\",200]]}}"
)

SPEC_USER_TEMPLATE_PATCHES = {
    "Do not exceed 3000 characters.":
        "Do not exceed 2000 characters.",
    "If you find yourself listing every text string or structural detail verbatim, "
    "compress into descriptive paragraphs instead.":
        "Include all visible text verbatim. Describe layout, color palette, and data concisely. "
        "Do not waste characters on generic visual descriptions.",
}

# Quality gates applied to a parsed spec before it is accepted.
MIN_PROMPT_LENGTH = 800
MIN_REQUIREMENTS = 3


def build_detailed_spec_prompts() -> tuple[str, str]:
    """Return the (system prompt, user template) pair for the detailed variant."""
    user_template = SPEC_USER_TEMPLATE
    for old, new in SPEC_USER_TEMPLATE_PATCHES.items():
        user_template = user_template.replace(old, new)
    return SPEC_SYSTEM_PROMPT + SPEC_DETAIL_SUFFIX, user_template


def text_exact_is_supported(text: str, generation_prompt: str) -> bool:
    """Whether a text_exact string appears verbatim in the generation prompt.

    A text_exact requirement is an extraction from the prompt, never an
    independent source of truth, so editing one means editing both.
    """
    return _text_exact_is_supported_by_prompt(text, generation_prompt)

HARD_REQUIREMENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bmust\b",
        r"\bexactly\b",
        r"\bfixed order\b",
        r"\btotal of\b",
        r"\bin that exact order\b",
        r"\bin this exact order\b",
        r"\bfrom left to right\b",
        r"\bfrom top to bottom\b",
        r"\bleft to right\b",
        r"\btop to bottom\b",
        r"\baligned\b",
        r"\bcentered\b",
        r"\bgrouped\b",
        r"\barranged\b",
        r"\bplaced\b",
        r"\bcontains?\b",
        r"\binclude exactly\b",
        r"\bthere (?:should|must) be\b",
        r"\bshould span\b",
        r"\boccup(?:y|ies)\b",
        r"\bpositioned\b",
    ]
]

SOFT_REQUIREMENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bmay\b",
        r"\bcan\b",
        r"\boptionally\b",
        r"\bsuggested\b",
        r"\bcould\b",
        r"\bmight\b",
        r"\bfeel\b",
        r"\batmosphere\b",
        r"\bmood\b",
        r"\btexture\b",
        r"\bdecorative\b",
        r"\bwhimsical\b",
        r"\bplayful\b",
    ]
]


# ── Text helpers ───────────────────────────────────────────────────────────────

def normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    normalized.append(text)
        return normalized
    return []


def split_prompt_sentences(text: str) -> List[str]:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", compact)
    return [part.strip() for part in parts if part.strip()]


def extract_verifiable_requirements(refined_prompt: str) -> List[str]:
    requirements = []
    for sentence in split_prompt_sentences(refined_prompt):
        if any(pattern.search(sentence) for pattern in SOFT_REQUIREMENT_PATTERNS):
            continue
        if any(pattern.search(sentence) for pattern in HARD_REQUIREMENT_PATTERNS):
            requirements.append(sentence)
            continue
        if re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", sentence, re.IGNORECASE) and re.search(
            r"\b(?:section|column|row|panel|card|bullet|heading|title|label|icon|circle|rectangle|band|badge|arrow|line|text|block|group|list|item|step|dot)\b",
            sentence,
            re.IGNORECASE,
        ):
            requirements.append(sentence)
    deduped: List[str] = []
    seen: set = set()
    for sentence in requirements:
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    return deduped


def extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start == -1:
        raise ValueError("Model output does not contain a JSON object.")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]
    raise ValueError("Failed to isolate a complete JSON object from model output.")


# ── Contract building ──────────────────────────────────────────────────────────

def build_semantic_attribute_contract(requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
    _ = requirements
    return json.loads(json.dumps(DEFAULT_SEMANTIC_ATTRIBUTE_CONTRACT))


def build_generator_execution_priorities(
    requirements: List[Dict[str, Any]], contract: Dict[str, Any]
) -> str:
    lines: List[str] = []
    enabled_requirements = [
        requirement
        for requirement in requirements
        if isinstance(requirement, dict) and requirement.get("enabled", True)
    ]

    _ = contract
    lines.append(
        "Use visible text and literal chart data as the primary validation surface. Do not rely on custom data-entity/data-role/data-scope names or renamed classes for correctness."
    )

    exact_text_targets: List[str] = []
    for requirement in enabled_requirements:
        if requirement.get("category") != "text_exact":
            continue
        expected = requirement.get("expected") if isinstance(requirement.get("expected"), dict) else {}
        text = expected.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if len(text.strip()) > 80:
            continue
        exact_text_targets.append(text.strip())
    if exact_text_targets:
        lines.append(
            "Keep exact text verbatim in separate visible nodes, especially for titles, headings, and source lines. Do not merge multiple required strings into one element."
        )

    data_sections: List[str] = []
    for requirement in enabled_requirements:
        category = requirement.get("category")
        expected = requirement.get("expected") if isinstance(requirement.get("expected"), dict) else {}
        req_id = str(requirement.get("id") or "").strip()
        data = expected.get("data")
        if category == "chart_data" and isinstance(data, list):
            data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            data_sections.append(
                f"  - [{req_id}] ECharts series.data must be exactly: {data_json}"
            )
        elif category == "table_data" and isinstance(data, list):
            rows_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            data_sections.append(
                f"  - [{req_id}] HTML <table> rows must match exactly: {rows_json}"
            )
    if data_sections:
        lines.append(
            "The following data payloads MUST appear verbatim as literal JSON in ECharts series.data "
            "or visible HTML table cells — do NOT alter, reorder, or recompute these values:\n"
            + "\n".join(data_sections)
        )

    if not lines:
        return ""

    return "\n\nGenerator execution priorities:\n" + "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, start=1)
    )


def ensure_semantic_contract_in_prompt(
    generation_prompt: str, contract: Dict[str, Any], requirements: List[Dict[str, Any]]
) -> str:
    _ = contract
    _ = requirements
    return str(generation_prompt or "").strip()


# ── Reference metadata ─────────────────────────────────────────────────────────

def build_reference_metadata(
    item: Dict[str, Any],
    reference_aspect_ratio: str,
    reference_orientation: str,
) -> Dict[str, Any]:
    return {
        "reference_image_wh": item.get("reference_image_wh", ""),
        "reference_aspect_ratio": float(reference_aspect_ratio) if reference_aspect_ratio else 0.0,
        "reference_orientation": reference_orientation,
        "reference_image_path": f"data/testset/{item.get('id', '')}.png",
    }


# ── Payload normalization + parsing ────────────────────────────────────────────

_VALID_CHART_SUBTYPES: frozenset = frozenset([
    "bar", "bar_horizontal", "line", "pie", "combo", "scatter", "funnel", "heatmap", "radar",
])


def _has_text_signal(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\u3400-\u9fff]", text or ""))


def _is_low_confidence_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if "\ufffd" in stripped:
        return True
    if not _has_text_signal(stripped):
        return True
    if re.fullmatch(r"[0-9IVXivx一二三四五六七八九十]+", stripped):
        return False
    if len(stripped) == 1 and not re.search(r"[\u3400-\u9fff]", stripped):
        return True
    return False


def _normalize_text_exact_requirement(req: Dict[str, Any]) -> Dict[str, Any] | None:
    """Validate text quality and strip target from text_exact requirements."""
    req = dict(req)
    expected = dict(req.get("expected") or {})
    text = str(expected.get("text") or "").strip()

    if _is_low_confidence_text(text):
        warnings.warn(
            f"Dropping requirement {req.get('id', '?')!r} (low-confidence or empty text).",
            stacklevel=4,
        )
        return None

    expected["text"] = text
    req["expected"] = expected
    req.pop("target", None)
    return req


def _normalize_prompt_text_for_match(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _text_exact_is_supported_by_prompt(text: str, generation_prompt: str) -> bool:
    normalized_text = _normalize_prompt_text_for_match(text)
    normalized_prompt = _normalize_prompt_text_for_match(generation_prompt)
    if not normalized_text or not normalized_prompt:
        return False
    return normalized_text in normalized_prompt


def _manual_filter_requirements(
    requirements: List[Dict[str, Any]], *, generation_prompt: str = ""
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for req in requirements:
        if not isinstance(req, dict):
            continue

        category = str(req.get("category") or "").strip()
        candidate: Dict[str, Any] | None = None

        if category == "text_exact":
            candidate = _normalize_text_exact_requirement(req)
            if candidate is not None:
                expected = candidate.get("expected") or {}
                text = str(expected.get("text") or "")
                if not _text_exact_is_supported_by_prompt(text, generation_prompt):
                    warnings.warn(
                        f"Dropping requirement {req.get('id', '?')!r} (text_exact not explicitly present in generation_prompt).",
                        stacklevel=4,
                    )
                    continue
                key = (category, str(expected.get("text") or "").casefold())
                if key in seen:
                    continue
                seen.add(key)
        elif category == "chart_data":
            target = req.get("target") if isinstance(req.get("target"), dict) else {}
            expected = req.get("expected") if isinstance(req.get("expected"), dict) else {}
            data = expected.get("data")
            if target.get("chart_type") != "echarts":
                continue
            if str(target.get("chart_subtype") or "").lower() not in _VALID_CHART_SUBTYPES:
                continue
            if not isinstance(data, list) or not data:
                continue
            candidate = req
        elif category == "table_data":
            expected = req.get("expected") if isinstance(req.get("expected"), dict) else {}
            data = expected.get("data")
            if not isinstance(data, list) or not data:
                continue
            candidate = {k: v for k, v in req.items() if k != "target"}

        if candidate is not None:
            filtered.append(candidate)

    # Strip metadata fields that the verifier handles via defaults.
    _strip = frozenset({"validation_mode", "enabled"})
    return [{k: v for k, v in r.items() if k not in _strip} for r in filtered]


def _expand_chart_requirements(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Post-process chart_data requirements to prevent two systematic failure modes at scale.

    Transform 1 — combo + object-array data:
      The spec LLM tends to emit ``expected.data = [{year, metric1, metric2}, ...]`` for
      combo charts.  ECharts uses *separate* series, not combined-row objects, so a single
      combined requirement can never match.  Detect this pattern and expand into one
      scalar-array requirement per numeric column so the validator can match each split
      series independently.

    Transform 2 — pie/donut + all-equal scalar placeholder:
      When the exact pie weights are unknown, the spec LLM emits ``[1, 1, 1]`` (all equal
      scalars).  Those placeholders carry no value information — only segment *count* is
      meaningful.  Auto-inject ``match_mode: "count"`` so the validator checks only that
      the right number of segments is rendered.
    """
    result: List[Dict[str, Any]] = []
    for req in requirements:
        if req.get("category") != "chart_data":
            result.append(req)
            continue

        target = req.get("target") or {}
        expected = req.get("expected") or {}
        subtype = str(target.get("chart_subtype") or "").lower()
        data = expected.get("data")

        # ── Transform 1: combo object-array → per-series scalar arrays ────────
        if (
            subtype == "combo"
            and isinstance(data, list)
            and data
            and all(isinstance(item, dict) for item in data)
        ):
            first_keys = list(data[0].keys())
            # Numeric columns are series; string/mixed columns are x-axis dimensions
            series_cols = [
                k for k in first_keys
                if all(isinstance(row.get(k), (int, float)) for row in data)
            ]
            if len(series_cols) >= 2:
                for i, col in enumerate(series_cols, start=1):
                    new_req = dict(req)
                    new_req["id"] = f"{req.get('id', 'req_combo')}_{i}"
                    new_req["target"] = dict(target)
                    new_req["expected"] = {
                        k: v for k, v in expected.items() if k != "data"
                    }
                    new_req["expected"]["data"] = [
                        row[col] for row in data if isinstance(row.get(col), (int, float))
                    ]
                    result.append(new_req)
                continue  # drop the original combined-object requirement

        # ── Transform 2: pie all-equal placeholder → count mode ───────────────
        if (
            subtype == "pie"
            and isinstance(data, list)
            and len(data) >= 2
            and all(isinstance(v, (int, float)) for v in data)
            and len(set(data)) == 1  # all values identical → placeholder
        ):
            req = dict(req)
            req["expected"] = dict(expected)
            req["expected"].setdefault("match_mode", "count")
            result.append(req)
            continue

        result.append(req)
    return result


def _fix_requirement_shape(req: Dict[str, Any]) -> Dict[str, Any]:
    """Move fields the model commonly misplaces into their correct locations."""
    req = dict(req)
    return req


def normalize_model_payload(
    payload: Dict[str, Any],
    item: Dict[str, Any],
    reference_aspect_ratio: str,
    reference_orientation: str,
) -> Dict[str, Any]:
    generation_prompt = str(payload.get("generation_prompt", "")).strip()
    raw_requirements = payload.get("verifiable_requirements", [])
    if not isinstance(raw_requirements, list):
        raise ValueError("verifiable_requirements must be a list.")
    fixed_requirements: List[Dict[str, Any]] = []
    for requirement in raw_requirements:
        if not isinstance(requirement, dict):
            raise ValueError("Each verifiable requirement must be an object.")
        fixed_requirements.append(_fix_requirement_shape(requirement))
    raw_requirements = _expand_chart_requirements(fixed_requirements)
    raw_requirements = _manual_filter_requirements(
        raw_requirements,
        generation_prompt=generation_prompt,
    )

    normalized: Dict[str, Any] = {
        "id": item.get("id", ""),
        "reference_metadata": build_reference_metadata(item, reference_aspect_ratio, reference_orientation),
        "generation_prompt": generation_prompt,
        "verifiable_requirements": raw_requirements,
    }

    return normalized


def _filter_invalid_requirements(
    normalized: Dict[str, Any], schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate each requirement individually; silently drop ones that fail.

    This prevents a single malformed requirement from invalidating an
    otherwise-correct sample.
    """
    if jsonschema is None:
        return normalized
    req_schema = {"$ref": "#/$defs/requirement", "$defs": schema["$defs"]}
    valid: List[Dict[str, Any]] = []
    for req in normalized.get("verifiable_requirements", []):
        try:
            jsonschema.validate(instance=req, schema=req_schema)
            valid.append(req)
        except jsonschema.ValidationError as exc:
            warnings.warn(
                f"Dropping requirement {req.get('id', '?')!r} "
                f"(schema violation): {exc.message}",
                stacklevel=4,
            )
    result = dict(normalized)
    result["verifiable_requirements"] = valid
    return result


def parse_spec_response(
    raw_text: str,
    item: Dict[str, Any],
    reference_aspect_ratio: str,
    reference_orientation: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    object_text = extract_json_object_text(raw_text)
    payload = json.loads(object_text)
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    normalized = normalize_model_payload(
        payload,
        item=item,
        reference_aspect_ratio=reference_aspect_ratio,
        reference_orientation=reference_orientation,
    )
    if jsonschema is not None:
        normalized = _filter_invalid_requirements(normalized, schema)
        jsonschema.validate(instance=normalized, schema=schema)
    return normalized
