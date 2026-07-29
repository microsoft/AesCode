"""Sample and requirement parsing: JSONL loading and payload deserialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from verifier.types import (
    EvaluationPolicy,
    RefinedSample,
    ReferenceMetadata,
    Requirement,
    RequirementCategory,
    Severity,
    ValidationMode,
)


def _normalize_requirement_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(payload)
    target = dict(item.get("target", {}))
    expected = dict(item.get("expected", {}))
    category = item.get("category")
    if category == "text_exact":
        # Apply cjk_basic normalization for all text_exact requirements; role is no longer
        # required in the spec output so we cannot condition on it here.
        expected.setdefault("normalization", "cjk_basic")
    elif category == "order":
        entity_type = str(target.get("entity_type", ""))
        if entity_type == "section_heading":
            expected.setdefault("normalization", "cjk_basic")
            expected.setdefault("case_sensitive", True)
    item["target"] = target
    item["expected"] = expected
    return item


def _parse_requirement(payload: Dict[str, Any]) -> Requirement:
    payload = _normalize_requirement_payload(payload)
    raw_severity = payload.get("severity") or "normal"
    return Requirement(
        id=payload["id"],
        category=RequirementCategory(payload["category"]),
        severity=Severity(raw_severity),
        validation_mode=ValidationMode(payload.get("validation_mode") or "dom"),
        target=payload.get("target", {}),
        expected=payload.get("expected", {}),
        enabled=payload.get("enabled", True),
        scope=payload.get("scope"),
        locator=payload.get("locator"),
        tolerance=payload.get("tolerance"),
        rationale=payload.get("rationale"),
        source_span=payload.get("source_span"),
    )


def _parse_sample(payload: Dict[str, Any]) -> RefinedSample:
    if "reference_metadata" not in payload:
        if {"reports", "html_path", "sample_id"}.intersection(payload.keys()):
            raise ValueError(
                "Received a batch-output JSONL row (has 'reports'/'html_path' keys) instead of a "
                "sample-input row.  Pass the refined sample file "
                "(e.g. dataset/eval/refined_image_0001_0005.jsonl) instead of the batch results file "
                "(e.g. dataset/eval/validate_image_0001_0005.jsonl)."
            )
        raise KeyError(
            f"'reference_metadata' key missing from JSONL row. "
            f"Present keys: {sorted(payload.keys())}"
        )
    _known_meta_fields = {"reference_image_wh", "reference_aspect_ratio", "reference_orientation", "reference_image_path"}
    reference_metadata = ReferenceMetadata(**{k: v for k, v in payload["reference_metadata"].items() if k in _known_meta_fields})
    evaluation_policy = EvaluationPolicy(**payload.get("evaluation_policy", {}))
    requirements = [_parse_requirement(item) for item in payload.get("verifiable_requirements", [])]
    return RefinedSample(
        id=payload["id"],
        application=payload.get("application", ""),
        reference_metadata=reference_metadata,
        generation_prompt=payload.get("generation_prompt", payload.get("refine_prompt", "")),
        verifiable_requirements=requirements,
        semantic_attribute_contract=dict(payload.get("semantic_attribute_contract", {})),
        evaluation_policy=evaluation_policy,
    )


def load_sample_from_jsonl(jsonl_path: Path, sample_id: str) -> RefinedSample:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("id") == sample_id:
                return _parse_sample(payload)
    raise KeyError(f"Sample id not found in JSONL: {sample_id}")


def load_samples_from_jsonl(jsonl_path: Path) -> List[RefinedSample]:
    samples: List[RefinedSample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            samples.append(_parse_sample(payload))
    return samples
