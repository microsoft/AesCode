from __future__ import annotations

from abc import ABC, abstractmethod

from verifier.types import ArtifactBundle, Requirement, ValidationResult, ValidationStatus


VALID_EVIDENCE_BASES = {"explicit", "fallback", "hybrid", "unknown"}


def _normalize_evidence_payload(evidence: dict | None) -> dict:
    payload = dict(evidence or {})
    sources = payload.get("evidence_sources")
    if isinstance(sources, str):
        payload["evidence_sources"] = [sources]
    elif isinstance(sources, list):
        payload["evidence_sources"] = [source for source in sources if isinstance(source, str) and source]
    else:
        discovery_source = payload.get("discovery_source")
        payload["evidence_sources"] = [discovery_source] if isinstance(discovery_source, str) and discovery_source else []

    evidence_basis = payload.get("evidence_basis")
    if evidence_basis not in VALID_EVIDENCE_BASES:
        payload["evidence_basis"] = "unknown"
    return payload


class BaseValidator(ABC):
    name: str = "BaseValidator"
    supported_categories: tuple[str, ...] = ()

    @abstractmethod
    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        raise NotImplementedError

    def not_implemented(self, requirement: Requirement, message: str) -> ValidationResult:
        return ValidationResult(
            requirement_id=requirement.id,
            validator_name=self.name,
            status=ValidationStatus.NOT_APPLICABLE,
            score=0.0,
            expected=requirement.expected,
            measured=None,
            tolerance=requirement.tolerance,
            message=message,
            evidence=_normalize_evidence_payload({"evidence_basis": "unknown"}),
            debug={"implemented": False},
        )

    def build_result(
        self,
        requirement: Requirement,
        *,
        status: ValidationStatus,
        score: float,
        measured: object = None,
        message: str = "",
        evidence: dict | None = None,
        debug: dict | None = None,
    ) -> ValidationResult:
        return ValidationResult(
            requirement_id=requirement.id,
            validator_name=self.name,
            status=status,
            score=score,
            expected=requirement.expected,
            measured=measured,
            tolerance=requirement.tolerance,
            message=message,
            evidence=_normalize_evidence_payload(evidence),
            debug=debug or {"implemented": True},
        )