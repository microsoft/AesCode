"""verifier -- canonical stage-3 package for validation, scoring, and reward.

Public surface:
  Types          : ArtifactBundle, RefinedSample, ValidationReport, ValidationResult, ValidationStatus, ...
  Validators     : ExecutionValidator, BoundaryValidator, WhitespaceValidator,
                   TableChartValidator, TextValidator, SemanticLayoutValidator
                   (CompileValidator, RenderValidator are backward-compat aliases for ExecutionValidator)
  Registry       : build_default_registry, get_validator, get_validator_for_category, ...
  Runner         : validate_sample, validate_batch
  Parsing        : load_sample_from_jsonl, load_samples_from_jsonl
  Artifacts      : build_artifact_bundle, apply_semantic_attribute_contract, write_enriched_html_artifact
"""
from verifier.api import (
    ArtifactBundle,
    BaseValidator,
    BoundaryValidator,
    CompileValidator,
    EvaluationPolicy,
    ExecutionValidator,
    RefinedSample,
    ReferenceMetadata,
    RenderValidator,
    Requirement,
    RequirementCategory,
    SemanticLayoutValidator,
    Severity,
    TableChartValidator,
    TextValidator,
    ValidationMode,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
    WhitespaceValidator,
    apply_semantic_attribute_contract,
    build_artifact_bundle,
    build_default_registry,
    get_validator,
    get_validator_for_category,
    get_validator_for_requirement,
    load_sample_from_jsonl,
    load_samples_from_jsonl,
    validate_batch,
    validate_sample,
    write_enriched_html_artifact,
)

__all__ = [
    # Types
    "ArtifactBundle",
    "EvaluationPolicy",
    "RefinedSample",
    "ReferenceMetadata",
    "Requirement",
    "RequirementCategory",
    "Severity",
    "ValidationMode",
    "ValidationReport",
    "ValidationResult",
    "ValidationStatus",
    # Base
    "BaseValidator",
    # Validators
    "BoundaryValidator",
    "CompileValidator",
    "ExecutionValidator",
    "RenderValidator",
    "SemanticLayoutValidator",
    "TableChartValidator",
    "TextValidator",
    "WhitespaceValidator",
    # Registry
    "build_default_registry",
    "get_validator",
    "get_validator_for_category",
    "get_validator_for_requirement",
    # Runner
    "validate_sample",
    "validate_batch",
    # Parsing
    "load_sample_from_jsonl",
    "load_samples_from_jsonl",
    # Artifacts
    "build_artifact_bundle",
    "apply_semantic_attribute_contract",
    "write_enriched_html_artifact",
]
