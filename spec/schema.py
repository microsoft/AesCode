"""Static schema constants and prompt-contract helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List


DEFAULT_EVALUATION_POLICY = {
    "hard_fail_on_missing_core_text": True,
    "hard_fail_on_wrong_section_count": True,
    "hard_fail_on_wrong_order": True,
    "allow_soft_cue_absence": True,
    "render_similarity_weight": 0.2,
    "aesthetic_score_weight": 0.1,
}

DEFAULT_SEMANTIC_ATTRIBUTE_CONTRACT = {
    "version": "1.0",
    "attributes": {},
    "selector_policy": {
        "prefer_semantic_attributes": False,
        "prefer_standard_html": True,
        "allow_text_based_fallback": True,
    },
    "required_entities": [],
    "required_role_scopes": [],
}

CANONICAL_ENTITY_NAME_HINTS = ["title", "footer_band"]

CANONICAL_ENTITY_ROLES = [
    "title",
    "section_heading",
    "chart_title",
    "table_header",
    "timeline_header",
    "cta_label",
    "footer",
    "footer_band",
]

COUNT_ENTITY_TYPES: set[str] = set()

ORDER_ENTITY_TYPES: set[str] = set()

ENTITY_NAME_PATTERNS = [
    re.compile(r"^title$"),
    re.compile(r"^footer_band$"),
]

COUNT_ENTITY_ROLE_MAP: Dict[str, str] = {}


def build_output_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://dataset-pipeline.local/schemas/verifiable_requirements.schema.json",
        "title": "Refined Sample With Verifiable Requirements",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "reference_metadata",
            "generation_prompt",
            "verifiable_requirements",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "reference_metadata": {"$ref": "#/$defs/referenceMetadata"},
            "generation_prompt": {"type": "string", "minLength": 1},
            "verifiable_requirements": {
                "type": "array",
                "items": {"$ref": "#/$defs/requirement"},
            },
        },
        "$defs": {
            "orientation": {
                "type": "string",
                "enum": ["landscape", "portrait", "square"],
            },
            "severity": {
                "type": "string",
                "enum": ["blocker", "critical", "normal", "advisory"],
            },
            "validationMode": {
                "type": "string",
                "enum": ["dom", "render", "hybrid"],
            },
            "scopeId": {"type": "string", "minLength": 1},
            "stringArray": {
                "type": "array",
                "items": {"type": "string"},
            },
            "chartScalarValue": {
                "type": ["string", "number", "integer", "boolean", "null"]
            },
            "chartTableRow": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/chartScalarValue"},
            },
            "tableDataRow": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/chartScalarValue"},
            },
            "locator": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "selector": {"type": "string"},
                    "css_selector": {"type": "string"},
                    "text_contains": {"type": "string"},
                    "scope": {"$ref": "#/$defs/scopeId"},
                    "role": {"type": "string"},
                    "index": {"type": "integer", "minimum": 0},
                },
            },
            "requirementBase": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "category",
                    "expected",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "category": {
                        "type": "string",
                        "enum": ["text_exact", "chart_data", "table_data"],
                    },
                    "validation_mode": {"$ref": "#/$defs/validationMode"},
                    "target": {"type": "object"},
                    "locator": {
                        "oneOf": [
                            {"$ref": "#/$defs/locator"},
                            {"type": "null"},
                        ]
                    },
                    "expected": {"type": "object"},
                    "enabled": {"type": "boolean"},
                },
            },
            "textExactRequirement": {
                "allOf": [
                    {"$ref": "#/$defs/requirementBase"},
                    {
                        "properties": {
                            "category": {"const": "text_exact"},
                            "expected": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["text"],
                                "properties": {
                                    "text": {"type": "string", "minLength": 1},
                                },
                            },
                        }
                    },
                ]
            },
            "chartDataRequirement": {
                "allOf": [
                    {"$ref": "#/$defs/requirementBase"},
                    {
                        "required": ["target"],
                        "properties": {
                            "category": {"const": "chart_data"},
                            "target": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["chart_type", "chart_subtype"],
                                "properties": {
                                    "chart_type": {
                                        "type": "string",
                                        "enum": ["echarts"],
                                    },
                                    "chart_subtype": {
                                        "type": "string",
                                        "enum": ["bar", "bar_horizontal", "line", "pie", "combo", "scatter", "funnel", "heatmap", "radar"],
                                    },
                                    "scope": {"$ref": "#/$defs/scopeId"},
                                },
                            },
                            "expected": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["data"],
                                "properties": {
                                    "series_name": {"type": "string", "minLength": 1},
                                    "x_key": {"type": "string", "minLength": 1},
                                    "y_key": {"type": "string", "minLength": 1},
                                    "match_mode": {
                                        "type": "string",
                                        "enum": ["exact", "subset"],
                                    },
                                    "data": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {
                                            "anyOf": [
                                                {
                                                    "type": "object",
                                                    "additionalProperties": {
                                                        "anyOf": [
                                                            {"$ref": "#/$defs/chartScalarValue"},
                                                            {"$ref": "#/$defs/chartTableRow"},
                                                        ]
                                                    },
                                                },
                                                {
                                                    "$ref": "#/$defs/chartScalarValue"
                                                },
                                                {
                                                    "$ref": "#/$defs/chartTableRow"
                                                }
                                            ]
                                        },
                                    },
                                },
                            },
                        }
                    },
                ]
            },
            "tableDataRequirement": {
                "allOf": [
                    {"$ref": "#/$defs/requirementBase"},
                    {
                        "properties": {
                            "category": {"const": "table_data"},
                            "target": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "scope": {"$ref": "#/$defs/scopeId"},
                                },
                            },
                            "expected": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["data"],
                                "properties": {
                                    "match_mode": {
                                        "type": "string",
                                        "enum": ["exact", "subset"],
                                    },
                                    "data": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"$ref": "#/$defs/tableDataRow"},
                                    },
                                },
                            },
                        }
                    },
                ]
            },
            "requirement": {
                "oneOf": [
                    {"$ref": "#/$defs/textExactRequirement"},
                    {"$ref": "#/$defs/chartDataRequirement"},
                    {"$ref": "#/$defs/tableDataRequirement"},
                ]
            },
            "referenceMetadata": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "reference_image_wh",
                    "reference_aspect_ratio",
                    "reference_orientation",
                    "reference_image_path",
                ],
                "properties": {
                    "reference_image_wh": {"type": "string", "pattern": "^[0-9]+x[0-9]+$"},
                    "reference_aspect_ratio": {"type": "number", "exclusiveMinimum": 0},
                    "reference_orientation": {"$ref": "#/$defs/orientation"},
                    "reference_image_path": {"type": "string", "minLength": 1},
                },
            },
            "semanticAttributeContract": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "version",
                    "attributes",
                    "selector_policy",
                    "required_entities",
                    "required_role_scopes",
                ],
                "properties": {
                    "version": {"type": "string", "const": "1.0"},
                    "attributes": {
                        "type": "object",
                        "additionalProperties": False,
                        "maxProperties": 0,
                    },
                    "selector_policy": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "prefer_semantic_attributes",
                            "prefer_standard_html",
                            "allow_text_based_fallback",
                        ],
                        "properties": {
                            "prefer_semantic_attributes": {"const": False},
                            "prefer_standard_html": {"const": True},
                            "allow_text_based_fallback": {"const": True},
                        },
                    },
                    "required_entities": {
                        "type": "array",
                        "maxItems": 0,
                    },
                    "required_role_scopes": {
                        "type": "array",
                        "maxItems": 0,
                    },
                },
            },
            "evaluationPolicy": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "hard_fail_on_missing_core_text",
                    "hard_fail_on_wrong_section_count",
                    "hard_fail_on_wrong_order",
                    "allow_soft_cue_absence",
                    "render_similarity_weight",
                    "aesthetic_score_weight",
                ],
                "properties": {
                    "hard_fail_on_missing_core_text": {"type": "boolean"},
                    "hard_fail_on_wrong_section_count": {"type": "boolean"},
                    "hard_fail_on_wrong_order": {"type": "boolean"},
                    "allow_soft_cue_absence": {"type": "boolean"},
                    "render_similarity_weight": {"type": "number"},
                    "aesthetic_score_weight": {"type": "number"},
                },
            },
        },
    }


def build_schema_prompt_contract(schema: Dict[str, Any]) -> str:
    categories = schema.get("$defs", {}).get("requirementBase", {}).get("properties", {}).get("category", {}).get("enum", [])
    category_text = ", ".join(categories)
    lines = [
        "Return one JSON object with these top-level keys only: id, reference_metadata, generation_prompt, verifiable_requirements.",
        "reference_metadata must include reference_image_wh, reference_aspect_ratio, reference_orientation, reference_image_path.",
        f"Allowed requirement categories: {category_text}.",
        "Each verifiable requirement must include id, category, expected. chart_data also requires target.",
        "text_exact: use for clearly legible, high-value visible text such as title lines, section headings, chart titles, table headers, timeline headers, CTA labels, and footer/source lines. Avoid exhaustive body-copy extraction.",
        "chart_data: use only when a real chart should preserve literal values. target.chart_type must be echarts; target.chart_subtype is required — choose from bar, bar_horizontal, line, pie, combo, scatter, funnel, heatmap, radar based on what the reference shows. expected.data must stay a literal JSON array whose items may be objects or scalar values. For combo charts that show two separate numeric series (e.g. bar + line), emit one chart_data requirement per series, each with a flat scalar data array. Do not merge both series into one combined-row object array.",
        "table_data: use when a real visible table or matrix should be preserved. expected.data must be a row-array matrix that maps to visible HTML table rows and cells.",
        "Do not invent target.type, expected.texts, or any other undeclared fields. For text_exact output only expected.text (no target). For chart_data use target.chart_type plus expected.data. For table_data use expected.data only (no target).",
        "Example text_exact shape: {\"id\":\"req_title\",\"category\":\"text_exact\",\"expected\":{\"text\":\"Example Title\"}}.",
        "Example chart_data shape (bar): {\"id\":\"req_bar\",\"category\":\"chart_data\",\"target\":{\"chart_type\":\"echarts\",\"chart_subtype\":\"bar\"},\"expected\":{\"data\":[12,18,24]}}.",
        "Example chart_data shape (combo, two requirements): {\"id\":\"req_combo_1\",\"category\":\"chart_data\",\"target\":{\"chart_type\":\"echarts\",\"chart_subtype\":\"combo\"},\"expected\":{\"data\":[2,2.8,4.1]}} and {\"id\":\"req_combo_2\",...,\"expected\":{\"data\":[7,21,26]}}.",
        "Example table_data shape: {\"id\":\"req_table\",\"category\":\"table_data\",\"expected\":{\"data\":[[\"Region\",\"Q1\"],[\"North\",12]]}}.",
        "Do not output semantic_attribute_contract, evaluation_policy, questions, or free-form requirement strings. Output structured JSON only.",
    ]
    return "\n".join(lines)
