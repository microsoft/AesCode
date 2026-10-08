"""Report serialization helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from verifier.types import ValidationReport, ValidationResult

from verifier.pipeline.scoring import _result_evidence_basis


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, str)):
        return len(value) == 0
    return False


def _compact_result_payload(result: ValidationResult) -> Dict[str, Any]:
    evidence = result.evidence if isinstance(result.evidence, dict) else {}
    evidence_sources = list(evidence.get("evidence_sources", []))
    extra_evidence = {
        key: value
        for key, value in evidence.items()
        if key not in {"evidence_basis", "evidence_sources"} and not _is_empty_value(value)
    }

    payload: Dict[str, Any] = {
        "requirement_id": result.requirement_id,
        "validator_name": result.validator_name,
        "status": result.status.value,
        "score": result.score,
        "evidence_basis": _result_evidence_basis(result),
    }
    if evidence_sources:
        payload["evidence_sources"] = evidence_sources
    if not _is_empty_value(result.expected):
        payload["expected"] = result.expected
    if not _is_empty_value(result.measured):
        payload["measured"] = result.measured
    if not _is_empty_value(result.tolerance):
        payload["tolerance"] = result.tolerance
    if result.message:
        payload["message"] = result.message
    if extra_evidence:
        payload["evidence"] = extra_evidence
    if isinstance(result.debug, dict) and result.debug:
        payload["debug"] = result.debug
    return payload


def serialize_validation_report(report: ValidationReport) -> Dict[str, Any]:
    return {
        "sample_id": report.sample_id,
        "execution_pass": report.execution_pass,
        "quality_pass": report.quality_pass,
        "overall_pass": report.overall_pass,
        "hard_failures": list(report.hard_failures),
        "summary": dict(report.summary),
        "results": [_compact_result_payload(result) for result in report.results],
    }


serialize_validation_report = serialize_validation_report
