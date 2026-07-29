"""Validator implementations -- Gate, Content/Layout, and Visual layers.

Layer 1 (Gate)          : ExecutionValidator (unified compile + render)
Layer 2 (Content/Layout): BoundaryValidator, TableChartValidator, TextValidator, SemanticLayoutValidator
Layer 3 (Visual)        : WhitespaceValidator, DesignValidator (VLM-based)
"""
from verifier.validators.contentlayout import (
    BoundaryValidator,
    SemanticLayoutValidator,
    TableChartValidator,
    TextValidator,
)
from verifier.validators.gates import CompileValidator, ExecutionValidator, RenderValidator
from verifier.validators.visual import DesignValidator, WhitespaceValidator

__all__ = [
    "BoundaryValidator",
    "CompileValidator",
    "DesignValidator",
    "ExecutionValidator",
    "RenderValidator",
    "SemanticLayoutValidator",
    "TableChartValidator",
    "TextValidator",
    "WhitespaceValidator",
]
