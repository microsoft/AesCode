from __future__ import annotations

import os
from typing import Dict, Iterable, Type

from verifier.base import BaseValidator
from verifier.validators import (
    BoundaryValidator,
    DesignValidator,
    ExecutionValidator,
    SemanticLayoutValidator,
    TableChartValidator,
    TextValidator,
    WhitespaceValidator,
)


DEFAULT_VALIDATOR_REGISTRY: Dict[str, Type[BaseValidator]] = {
    ExecutionValidator.name: ExecutionValidator,
    BoundaryValidator.name: BoundaryValidator,
    WhitespaceValidator.name: WhitespaceValidator,
    DesignValidator.name: DesignValidator,
    TableChartValidator.name: TableChartValidator,
    SemanticLayoutValidator.name: SemanticLayoutValidator,
    TextValidator.name: TextValidator,
}

CATEGORY_FALLBACK_VALIDATOR_NAMES: Dict[str, tuple[str, ...]] = {
    "layout": (SemanticLayoutValidator.name,),
    "text_exact": (TextValidator.name,),
    "chart_data": (TableChartValidator.name,),
    "table_data": (TableChartValidator.name,),
    "style_cue": (WhitespaceValidator.name,),
    "compile": (ExecutionValidator.name,),
    "render": (ExecutionValidator.name,),
}


def _get_disabled_validators() -> set[str]:
    """Parse DISABLED_VALIDATORS env var (comma-separated validator names)."""
    raw = os.environ.get("DISABLED_VALIDATORS", "")
    if not raw:
        return set()
    return {v.strip() for v in raw.split(",") if v.strip()}


def build_default_registry() -> Dict[str, BaseValidator]:
    disabled = _get_disabled_validators()
    return {
        validator_name: validator_class()
        for validator_name, validator_class in DEFAULT_VALIDATOR_REGISTRY.items()
        if validator_name not in disabled
    }


def _iter_validators(registry: Dict[str, BaseValidator]) -> Iterable[BaseValidator]:
    seen_ids = set()
    for validator in registry.values():
        marker = id(validator)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)
        yield validator


def get_validator(validator_name: str, registry: Dict[str, BaseValidator] | None = None) -> BaseValidator:
    active_registry = registry or build_default_registry()
    if validator_name in active_registry:
        return active_registry[validator_name]
    # Backward compatibility: map old names to ExecutionValidator
    if validator_name in ("CompileValidator", "RenderValidator"):
        ev_name = ExecutionValidator.name
        if ev_name in active_registry:
            return active_registry[ev_name]
    for validator in _iter_validators(active_registry):
        if validator.name == validator_name:
            return validator
    raise KeyError(f"No validator registered with name: {validator_name}")


def get_validator_for_category(category: str, registry: Dict[str, BaseValidator] | None = None) -> BaseValidator:
    active_registry = registry or build_default_registry()
    preferred_validator_names = CATEGORY_FALLBACK_VALIDATOR_NAMES.get(category, ())
    for validator_name in preferred_validator_names:
        if validator_name in active_registry:
            return active_registry[validator_name]
    for validator in _iter_validators(active_registry):
        if category in getattr(validator, "supported_categories", ()):  # pragma: no branch
            return validator
    raise KeyError(f"No validator registered for category: {category}")


def get_validator_for_requirement(requirement, registry: Dict[str, BaseValidator] | None = None) -> BaseValidator:
    target = getattr(requirement, "target", {}) or {}
    expected = getattr(requirement, "expected", {}) or {}
    explicit_validator_name = target.get("validator_name") if isinstance(target, dict) else None
    if isinstance(explicit_validator_name, str) and explicit_validator_name:
        return get_validator(explicit_validator_name, registry=registry)

    category = getattr(requirement, "category", None)
    category_value = getattr(category, "value", category)
    if category_value == "style_cue":
        cue_type = str(expected.get("cue_type") or "").lower()
        if "whitespace" in cue_type or "spacing" in cue_type or "margin" in cue_type or "padding" in cue_type:
            return get_validator(WhitespaceValidator.name, registry=registry)
        if "design" in cue_type or "visual" in cue_type:
            return get_validator(DesignValidator.name, registry=registry)
        # Default style_cue → WhitespaceValidator
        return get_validator(WhitespaceValidator.name, registry=registry)

    if not isinstance(category_value, str) or not category_value:
        raise KeyError("Requirement does not declare a resolver-friendly category or validator_name")
    return get_validator_for_category(category_value, registry=registry)
