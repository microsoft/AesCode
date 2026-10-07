"""Scoring computation for validation results.

Flat scoring:
    overall_score = simple average of ALL per-validator scores.
    Per-validator scores are reported individually (e.g. ExecutionValidator,
    TextValidator, DesignValidator ...) so callers can inspect any dimension.
    There are NO bucket groupings (no "content" or "visual" categories).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from verifier.types import (
    RequirementCategory,
    ValidationResult,
    ValidationStatus,
)


EVIDENCE_BASIS_VALUES = ("explicit", "fallback", "hybrid", "unknown")

ASSESSMENT_BANDS = (
    (0.9, "excellent"),
    (0.8, "strong"),
    (0.65, "fair"),
    (0.0, "weak"),
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _result_base_score(result: ValidationResult) -> Optional[float]:
    """Return the validator's raw continuous score, clamped to [0, 1].

    status is intentionally ignored — each validator is responsible for
    setting a meaningful ``result.score`` and we pass it through as-is
    so that continuous validators (Design / Whitespace / etc.)
    retain full discriminability.
    """
    if result.status == ValidationStatus.NOT_APPLICABLE:
        return None
    return _clamp(result.score)


def _assessment_label(score: float) -> str:
    for minimum, label in ASSESSMENT_BANDS:
        if score >= minimum:
            return label
    return "weak"


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _result_evidence_basis(result: ValidationResult) -> str:
    basis = result.evidence.get("evidence_basis") if isinstance(result.evidence, dict) else None
    if basis in EVIDENCE_BASIS_VALUES:
        return str(basis)
    return "unknown"


def _summarize_evidence_bases(results: List[ValidationResult]) -> Dict[str, float]:
    counts = {basis: 0.0 for basis in EVIDENCE_BASIS_VALUES}
    for result in results:
        counts[_result_evidence_basis(result)] += 1.0
    return {
        "explicit_evidence_result_count": counts["explicit"],
        "fallback_evidence_result_count": counts["fallback"],
        "hybrid_evidence_result_count": counts["hybrid"],
        "unknown_evidence_result_count": counts["unknown"],
    }


# ---------------------------------------------------------------------------
# Scored rows — flat list, no bucket concept
# ---------------------------------------------------------------------------

def _build_scored_rows(
    results: List[ValidationResult],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        base_score = _result_base_score(result)
        if base_score is None:
            continue
        rows.append(
            {
                "requirement_id": result.requirement_id,
                "validator_name": result.validator_name,
                "status": result.status,
                "base_score": base_score,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Top-level scoring summary
# ---------------------------------------------------------------------------

def _summarize_scores(
    sample: Any,  # RefinedSample
    results: List[ValidationResult],
    execution_pass: bool,
) -> Dict[str, Any]:
    scored_rows = _build_scored_rows(results)

    # ── Per-validator scores ──────────────────────────────────────────────
    # Group by validator_name, compute mean score per validator.
    validator_scores: Dict[str, List[float]] = {}
    for row in scored_rows:
        validator_scores.setdefault(row["validator_name"], []).append(row["base_score"])

    per_validator: Dict[str, float] = {}
    for vname, scores in sorted(validator_scores.items()):
        per_validator[vname] = round(_clamp(sum(scores) / len(scores)), 4)

    # ── Overall = simple average of ALL per-validator mean scores ─────────
    if per_validator:
        overall_score = _clamp(
            sum(per_validator.values()) / len(per_validator)
        )
    else:
        overall_score = 1.0

    # Quality gate
    quality_gate_threshold = 0.6
    quality_pass = overall_score >= quality_gate_threshold
    overall_pass = execution_pass and quality_pass

    # Degraded / failed requirements for diagnostics
    degraded = [r["requirement_id"] for r in scored_rows
                if r["status"] in {ValidationStatus.FAIL, ValidationStatus.PARTIAL, ValidationStatus.UNCERTAIN}]
    critical_fails = [r["requirement_id"] for r in scored_rows
                      if r["status"] == ValidationStatus.FAIL]

    return {
        "overall_score": round(overall_score, 4),
        "per_validator": per_validator,
        "validator_count": len(per_validator),
        "quality_gate_threshold": round(quality_gate_threshold, 4),
        "quality_pass": 1.0 if quality_pass else 0.0,
        "overall_pass": 1.0 if overall_pass else 0.0,
        "overall_assessment": _assessment_label(overall_score),
        "degraded_requirement_count": float(len(degraded)),
        "critical_failure_count": float(len(critical_fails)),
        "degraded_requirements": degraded,
        "critical_failures": critical_fails,
    }
