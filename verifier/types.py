from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RequirementCategory(str, Enum):
    LAYOUT = "layout"
    CANVAS = "canvas"
    TEXT_EXACT = "text_exact"
    COUNT = "count"
    CHART_DATA = "chart_data"
    TABLE_DATA = "table_data"
    ORDER = "order"
    GROUPING = "grouping"
    LAYOUT_RELATION = "layout_relation"
    LAYOUT_DISTRIBUTION = "layout_distribution"
    REPEATED_STRUCTURE = "repeated_structure"
    STYLE_CUE = "style_cue"


class Severity(str, Enum):
    BLOCKER = "blocker"
    CRITICAL = "critical"
    NORMAL = "normal"
    ADVISORY = "advisory"


class ValidationMode(str, Enum):
    DOM = "dom"
    RENDER = "render"
    HYBRID = "hybrid"


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class ReferenceMetadata:
    reference_image_wh: str
    reference_aspect_ratio: float
    reference_orientation: str
    reference_image_path: Optional[str] = None


@dataclass(slots=True)
class Requirement:
    id: str
    category: RequirementCategory
    severity: Severity
    validation_mode: ValidationMode
    target: Dict[str, Any]
    expected: Dict[str, Any]
    enabled: bool = True
    scope: Optional[str] = None
    locator: Optional[Dict[str, Any]] = None
    tolerance: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None
    source_span: Optional[str | List[str]] = None


@dataclass(slots=True)
class EvaluationPolicy:
    hard_fail_on_missing_core_text: bool = True
    hard_fail_on_wrong_section_count: bool = True
    hard_fail_on_wrong_order: bool = True
    allow_soft_cue_absence: bool = True
    render_similarity_weight: float = 0.0
    aesthetic_score_weight: float = 0.0
    quality_gate_threshold: float = 0.6
    quality_score_weight: float = 0.55


@dataclass(slots=True)
class RefinedSample:
    id: str
    reference_metadata: ReferenceMetadata
    generation_prompt: str
    application: str = ""
    verifiable_requirements: List[Requirement] = field(default_factory=list)
    semantic_attribute_contract: Dict[str, Any] = field(default_factory=dict)
    evaluation_policy: EvaluationPolicy = field(default_factory=EvaluationPolicy)


@dataclass(slots=True)
class ArtifactBundle:
    sample_id: str
    html_source: Optional[str] = None
    dom_snapshot: Optional[Dict[str, Any]] = None
    css_snapshot: Optional[Dict[str, Any]] = None
    render_image_path: Optional[str] = None
    reference_image_path: Optional[str] = None
    ocr_result: Optional[Dict[str, Any]] = None
    layout_boxes: List[Dict[str, Any]] = field(default_factory=list)
    detected_elements: List[Dict[str, Any]] = field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    layout_matching: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    playwright_errors: List[Dict[str, Any]] = field(default_factory=list)
    playwright_console_errors: List[str] = field(default_factory=list)
    render_duration_ms: Optional[float] = None
    overflowing_elements: List[Dict[str, Any]] = field(default_factory=list)
    render_blank: bool = False


@dataclass(slots=True)
class ValidationResult:
    requirement_id: str
    validator_name: str
    status: ValidationStatus
    score: float
    expected: Any = None
    measured: Any = None
    tolerance: Any = None
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationReport:
    sample_id: str
    execution_pass: bool
    quality_pass: bool
    overall_pass: bool
    hard_failures: List[str] = field(default_factory=list)
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


BBox = Tuple[float, float, float, float]