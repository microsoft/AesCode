"""Stage 3 sample validation runner."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from verifier.artifacts.paths import (
    get_sample_artifact_dir,
    resolve_reference_image_path,
    write_source_html_artifact,
    write_layout_matching_artifact,
    write_validation_report_artifact,
)
from verifier.artifacts.html import build_artifact_bundle
from verifier.artifacts.semantic import apply_semantic_attribute_contract, write_enriched_html_artifact
from verifier.pipeline.registry import build_default_registry, get_validator, get_validator_for_requirement
from verifier.types import (
    RefinedSample,
    Requirement,
    RequirementCategory,
    Severity,
    ValidationMode,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)

_logger = logging.getLogger(__name__)

# Module-level executor shared across async calls
_VALIDATOR_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="validator")

# ---------------------------------------------------------------------------
# Playwright render executor — MUST be a separate ProcessPool, not a thread.
# ---------------------------------------------------------------------------
# Why: Playwright's sync API spawns chromium via its own internal uvloop
# (see playwright/_impl/_transport.py). When the host process is itself an
# AgentLoopWorker running its own uvloop, two uvloops in the same process
# race when both try to spawn subprocesses concurrently, raising:
#     RuntimeError: Racing with another loop to spawn a process.
# That exception lives inside an unawaited Future ("Future exception was
# never retrieved"), silently hanging the rollout. ProcessPoolExecutor
# physically isolates each playwright invocation in its own process so
# there is exactly one uvloop per process — race is impossible.
#
# Tunables:
#   RENDER_PROCESS_WORKERS (defaults to PLAYWRIGHT_CONCURRENCY in reward_fn,
#       fallback 4): one chromium per worker, ~400MB RAM each. Keep aligned
#       with the asyncio Semaphore that gates concurrent renders so we don't
#       waste memory on idle workers.
#   RENDER_BUILD_TIMEOUT_S (default 120): per-call timeout. If exceeded we
#       assume the worker is wedged (chromium crash, OOM, network stall) and
#       rebuild the pool. Should be > p99 render latency × safety margin.
import os as _os_for_pool
import multiprocessing as _mp_for_pool
from concurrent.futures.process import BrokenProcessPool

_RENDER_WORKERS = int(_os_for_pool.environ.get(
    "RENDER_PROCESS_WORKERS",
    _os_for_pool.environ.get("PLAYWRIGHT_CONCURRENCY", "4"),
))
_RENDER_BUILD_TIMEOUT_S = float(_os_for_pool.environ.get("RENDER_BUILD_TIMEOUT_S", "120"))
# Use 'spawn' to ensure a clean process (no fork-inherited uvloop / CUDA state).
_RENDER_EXECUTOR: ProcessPoolExecutor | None = None
_RENDER_EXECUTOR_LOCK = None  # threading.Lock, lazy-init


def _build_render_executor() -> ProcessPoolExecutor:
    """Construct a fresh render ProcessPool. Caller holds the lock."""
    ctx = _mp_for_pool.get_context("spawn")
    ex = ProcessPoolExecutor(max_workers=_RENDER_WORKERS, mp_context=ctx)
    _logger.info(
        "[runner] Initialised render ProcessPool: workers=%d timeout=%.0fs (spawn)",
        _RENDER_WORKERS, _RENDER_BUILD_TIMEOUT_S,
    )
    return ex


def _get_render_executor() -> ProcessPoolExecutor:
    """Lazy-init render ProcessPool. Safe under concurrent first-call."""
    global _RENDER_EXECUTOR, _RENDER_EXECUTOR_LOCK
    if _RENDER_EXECUTOR is not None:
        return _RENDER_EXECUTOR
    import threading
    if _RENDER_EXECUTOR_LOCK is None:
        _RENDER_EXECUTOR_LOCK = threading.Lock()
    with _RENDER_EXECUTOR_LOCK:
        if _RENDER_EXECUTOR is None:
            _RENDER_EXECUTOR = _build_render_executor()
    return _RENDER_EXECUTOR


def _rebuild_render_executor() -> ProcessPoolExecutor:
    """Tear down a broken pool and construct a new one. Idempotent under the lock.
    Called when a worker crashes (BrokenProcessPool) or wedges (timeout)."""
    global _RENDER_EXECUTOR, _RENDER_EXECUTOR_LOCK
    import threading
    if _RENDER_EXECUTOR_LOCK is None:
        _RENDER_EXECUTOR_LOCK = threading.Lock()
    with _RENDER_EXECUTOR_LOCK:
        old = _RENDER_EXECUTOR
        if old is not None:
            try:
                # cancel_futures avoids waiting on already-wedged tasks.
                old.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                _logger.warning("[runner] old render pool shutdown failed: %s", e)
            # FD-LEAK FIX (2026-06-23): shutdown(wait=False) returns before worker
            # processes exit. A worker wedged on a chromium render never closes its
            # multiprocessing pipe fds, leaking ~socket fds per rebuild (919/1024 after
            # ~17h -> OSError: Too many open files -> reward poisoned to 0). Force the
            # old workers dead so the OS reclaims their pipe/socket fds immediately.
            try:
                _procs = list(getattr(old, "_processes", {}).values()) if old else []
                for _p in _procs:
                    try:
                        if _p.is_alive():
                            _p.terminate()
                    except Exception:
                        pass
                for _p in _procs:
                    try:
                        _p.join(timeout=2.0)
                        if _p.is_alive():
                            _p.kill()
                    except Exception:
                        pass
            except Exception as e:
                _logger.warning("[runner] old render pool worker reap failed: %s", e)
        _RENDER_EXECUTOR = _build_render_executor()
        _logger.warning("[runner] Render ProcessPool REBUILT after failure")
    return _RENDER_EXECUTOR


async def _run_render_with_recovery(
    sample_id: str,
    html_source: str,
    html_path_str: str | None,
    artifact_root_str: str | None = None,
):
    """Submit a render to the ProcessPool with timeout + BrokenProcessPool self-heal.
    On failure the pool is rebuilt and the exception is re-raised so the caller's
    asyncio.gather(return_exceptions=True) can convert it into a FAIL result.

    ``artifact_root_str`` is forwarded into the render worker because
    ``set_artifact_root`` writes to a thread-local that doesn't cross the
    process boundary — without this the worker falls back to the DesignCoder repo
    ``artifacts/samples/`` directory, which may be outside the intended output
    root (a cause of render_image_missing if that path is on a volume that gets
    cleaned up concurrently).
    """
    loop = asyncio.get_event_loop()
    ex = _get_render_executor()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                ex,
                _render_worker_build_bundle,
                sample_id, html_source, html_path_str, artifact_root_str,
            ),
            timeout=_RENDER_BUILD_TIMEOUT_S,
        )
    except (BrokenProcessPool, asyncio.TimeoutError) as e:
        _logger.warning(
            "[runner] Render failed for sample=%s err=%s — rebuilding pool",
            sample_id, type(e).__name__,
        )
        _rebuild_render_executor()
        raise


def _render_worker_build_bundle(
    sample_id: str,
    html_source: str,
    html_path_str: str | None,
    artifact_root_str: str | None = None,
):
    """Top-level (picklable) wrapper invoked inside render workers.
    Imports happen inside the worker so the parent process doesn't pay
    playwright's import cost unnecessarily.

    Re-applies ``set_artifact_root`` because the parent's thread-local does
    NOT propagate across ProcessPoolExecutor's fork/spawn boundary. Without
    this the worker writes to ``<repo>/verifier/artifacts/samples/``, which may
    be outside the intended output root and get cleaned up concurrently,
    causing render_image_missing for ~all samples.
    """
    from pathlib import Path as _Path
    from verifier.artifacts.html import build_artifact_bundle_fast
    from verifier.artifacts.paths import set_artifact_root
    if artifact_root_str:
        set_artifact_root(_Path(artifact_root_str))
    return build_artifact_bundle_fast(
        sample_id,
        html_source,
        html_path=_Path(html_path_str) if html_path_str else None,
    )

from verifier.pipeline.parser import load_sample_from_jsonl, load_samples_from_jsonl
from verifier.pipeline.scoring import _result_base_score, _summarize_evidence_bases, _summarize_scores
from verifier.pipeline.serialization import serialize_validation_report
from verifier.validators import (
    BoundaryValidator,
    DesignValidator,
    ExecutionValidator,
    SemanticLayoutValidator,
    TableChartValidator,
    TextValidator,
    WhitespaceValidator,
)


def _extract_required_display_texts(sample: RefinedSample) -> List[str]:
    required_texts: List[str] = []

    def remember(value: str) -> None:
        text = str(value or "").strip()
        if len(text) < 2:
            return
        if text not in required_texts:
            required_texts.append(text)

    for requirement in sample.verifiable_requirements:
        if not requirement.enabled:
            continue
        if requirement.category == RequirementCategory.TEXT_EXACT:
            remember(str(requirement.expected.get("text") or ""))

    return required_texts
def _make_runtime_requirement(
    *,
    requirement_id: str,
    validator_name: str,
    category: RequirementCategory,
    severity: Severity,
    validation_mode: ValidationMode,
    target: Dict[str, Any] | None = None,
    expected: Dict[str, Any] | None = None,
) -> Requirement:
    merged_target = dict(target or {})
    merged_target.setdefault("validator_name", validator_name)
    return Requirement(
        id=requirement_id,
        category=category,
        severity=severity,
        validation_mode=validation_mode,
        target=merged_target,
        expected=dict(expected or {}),
    )


def _gate_requirements(sample: RefinedSample) -> List[Requirement]:
    return [
        _make_runtime_requirement(
            requirement_id="execution",
            validator_name=ExecutionValidator.name,
            category=RequirementCategory.LAYOUT,
            severity=Severity.BLOCKER,
            validation_mode=ValidationMode.HYBRID,
        ),
    ]


def _runtime_requirements(
    sample: RefinedSample,
    artifacts: Any,
    registry: Dict[str, Any] | None = None,
    *,
    include_design_validator: bool = True,
) -> List[Requirement]:
    color_signal = next(
        (
            requirement
            for requirement in sample.verifiable_requirements
            if requirement.enabled
            and getattr(requirement.category, "value", requirement.category) == "style_cue"
            and "color" in str(requirement.expected.get("cue_type") or "").lower()
        ),
        None,
    )
    requirements = [
        _make_runtime_requirement(
            requirement_id="boundary",
            validator_name=BoundaryValidator.name,
            category=RequirementCategory.CANVAS,
            severity=Severity.NORMAL,
            validation_mode=ValidationMode.RENDER,
            expected={"reference_image_wh": sample.reference_metadata.reference_image_wh},
        ),
        _make_runtime_requirement(
            requirement_id="layout",
            validator_name=SemanticLayoutValidator.name,
            category=RequirementCategory.LAYOUT,
            severity=Severity.CRITICAL,
            validation_mode=ValidationMode.HYBRID,
            target={"strategy": "text_block_validation"},
            expected={
                "reference_image_path": str(artifacts.reference_image_path) if artifacts.reference_image_path else None,
                "predicted_source": (
                    "text_blocks"
                    if artifacts.text_blocks
                    else ("render_image" if artifacts.render_image_path else "artifact_boxes")
                ),
                "required_display_texts": _extract_required_display_texts(sample),
            },
        ),
        _make_runtime_requirement(
            requirement_id="whitespace",
            validator_name=WhitespaceValidator.name,
            category=RequirementCategory.LAYOUT_DISTRIBUTION,
            severity=Severity.NORMAL,
            validation_mode=ValidationMode.RENDER,
        ),
    ]

    # DesignValidator is only added when it's registered (not disabled via
    # DISABLED_VALIDATORS env var).
    _active = registry or {}
    if include_design_validator and DesignValidator.name in _active:
        requirements.append(
            _make_runtime_requirement(
                requirement_id="design",
                validator_name=DesignValidator.name,
                category=RequirementCategory.STYLE_CUE,
                severity=Severity.NORMAL,
                validation_mode=ValidationMode.RENDER,
                expected={"cue_type": "design"},
            ),
        )

    count_requirements = [
        requirement
        for requirement in sample.verifiable_requirements
        if requirement.enabled and requirement.category == RequirementCategory.COUNT
    ]
    requirements.extend(count_requirements)

    chart_requirements = [
        requirement
        for requirement in sample.verifiable_requirements
        if requirement.enabled and requirement.category == RequirementCategory.CHART_DATA
    ]
    for requirement in chart_requirements:
        if not requirement.target.get("validator_name"):
            requirement.target["validator_name"] = TableChartValidator.name
    requirements.extend(chart_requirements)

    table_requirements = [
        requirement
        for requirement in sample.verifiable_requirements
        if requirement.enabled and requirement.category == RequirementCategory.TABLE_DATA
    ]
    for requirement in table_requirements:
        if not requirement.target.get("validator_name"):
            requirement.target["validator_name"] = TableChartValidator.name
    requirements.extend(table_requirements)

    text_exact_requirements = [
        requirement
        for requirement in sample.verifiable_requirements
        if requirement.enabled and requirement.category == RequirementCategory.TEXT_EXACT
    ]
    for requirement in text_exact_requirements:
        if not requirement.target.get("validator_name"):
            requirement.target["validator_name"] = TextValidator.name
    requirements.extend(text_exact_requirements)
    return requirements


def _build_summary(
    sample: RefinedSample,
    results: List[ValidationResult],
    execution_pass: bool,
    enrichment_summary: Dict[str, Any],
    *,
    include_legacy_requirements: bool,
) -> Dict[str, Any]:
    def _status_counts(items: List[Any], attr: str) -> Dict[str, float]:
        return {
            ValidationStatus.PASS.value: float(
                sum(1 for item in items if getattr(item, attr) == ValidationStatus.PASS.value or getattr(item, attr) == ValidationStatus.PASS)
            ),
            ValidationStatus.FAIL.value: float(
                sum(1 for item in items if getattr(item, attr) == ValidationStatus.FAIL.value or getattr(item, attr) == ValidationStatus.FAIL)
            ),
            ValidationStatus.PARTIAL.value: float(
                sum(1 for item in items if getattr(item, attr) == ValidationStatus.PARTIAL.value or getattr(item, attr) == ValidationStatus.PARTIAL)
            ),
            ValidationStatus.UNCERTAIN.value: float(
                sum(1 for item in items if getattr(item, attr) == ValidationStatus.UNCERTAIN.value or getattr(item, attr) == ValidationStatus.UNCERTAIN)
            ),
            ValidationStatus.NOT_APPLICABLE.value: float(
                sum(
                    1
                    for item in items
                    if getattr(item, attr) == ValidationStatus.NOT_APPLICABLE.value
                    or getattr(item, attr) == ValidationStatus.NOT_APPLICABLE
                )
            ),
        }

    validator_groups: Dict[str, List[ValidationResult]] = {}
    for result in results:
        validator_groups.setdefault(result.validator_name, []).append(result)

    validator_summary: List[Dict[str, Any]] = []
    for validator_name, grouped_results in validator_groups.items():
        applicable_results = [
            result for result in grouped_results if result.status != ValidationStatus.NOT_APPLICABLE
        ]
        if not applicable_results:
            aggregated_status = ValidationStatus.NOT_APPLICABLE
            aggregated_score = 0.0
        else:
            base_scores = [
                base_score
                for result in applicable_results
                if (base_score := _result_base_score(result)) is not None
            ]
            aggregated_score = (sum(base_scores) / len(base_scores)) if base_scores else 0.0
            status_set = {result.status for result in applicable_results}
            if len(status_set) == 1:
                aggregated_status = applicable_results[0].status
            else:
                aggregated_status = ValidationStatus.PARTIAL

        validator_summary.append(
            {
                "validator_name": validator_name,
                "status": aggregated_status.value,
                "score": round(float(aggregated_score), 4),
                "requirement_count": float(len(grouped_results)),
                "applicable_requirement_count": float(len(applicable_results)),
            }
        )

    validator_summary.sort(key=lambda item: item["validator_name"])
    applicable_validator_summary = [
        item for item in validator_summary if item["applicable_requirement_count"] > 0.0
    ]
    score_summary = _summarize_scores(sample, results, execution_pass)
    evidence_counts = _summarize_evidence_bases(results)
    layout_result = next((result for result in results if result.requirement_id == "layout"), None)
    layout_measured = layout_result.measured if layout_result and isinstance(layout_result.measured, dict) else {}
    enabled_requirement_count = float(len([r for r in sample.verifiable_requirements if r.enabled]))
    summary: Dict[str, Any] = {
        "validator_count": float(len(validator_summary)),
        "applicable_validator_count": float(len(applicable_validator_summary)),
        "validator_status_counts": _status_counts(
            [type("ValidatorRow", (), item)() for item in validator_summary],
            "status",
        ),
        "result_count": float(len(results)),
        "result_status_counts": _status_counts(results, "status"),
        "validators": applicable_validator_summary,
        "scores": score_summary,
        "passes": {
            "quality_pass": bool(score_summary["quality_pass"]),
            "overall_pass": bool(score_summary["overall_pass"]),
        },
        "evidence_result_counts": evidence_counts,
        "semantic_enrichment": {
            "changed": bool(enrichment_summary.get("changed")),
            "touched_nodes": float(enrichment_summary.get("touched_nodes", 0)),
        },
        "requirement_counts": {
            "declared": float(len(sample.verifiable_requirements)),
            "enabled": enabled_requirement_count,
            "included_in_results": float(len(results)),
        },
    }
    if layout_result and layout_result.status != ValidationStatus.NOT_APPLICABLE:
        summary["layout"] = {
            "validator_name": layout_result.validator_name,
            "score": float(layout_result.score),
            "reference_block_count": float(layout_measured.get("reference_block_count", 0.0)),
            "predicted_block_count": float(layout_measured.get("predicted_block_count", 0.0)),
            "matched_block_count": float(layout_measured.get("matched_block_count", 0.0)),
        }
    if score_summary["degraded_requirement_count"] > 0.0 or score_summary["critical_failure_count"] > 0.0:
        summary["issues"] = {
            "degraded_requirement_count": score_summary["degraded_requirement_count"],
            "critical_failure_count": score_summary["critical_failure_count"],
            "degraded_requirements": score_summary["degraded_requirements"],
            "critical_failures": score_summary["critical_failures"],
        }
    return summary

def validate_sample(
    sample: RefinedSample,
    html_path: Path,
    *,
    artifact_backend: str = "auto",
    artifact_root: Path | None = None,
    include_legacy_requirements: bool = False,
) -> ValidationReport:
    from verifier.artifacts.paths import set_artifact_root
    set_artifact_root(artifact_root)

    html_source = html_path.read_text(encoding="utf-8")
    source_artifact_path = write_source_html_artifact(sample_id=sample.id, html_source=html_source)
    enriched_html_source, enrichment_summary = apply_semantic_attribute_contract(
        html_source,
        sample.semantic_attribute_contract,
    )
    enriched_html_path = write_enriched_html_artifact(
        original_html_path=html_path,
        sample_id=sample.id,
        enriched_html=enriched_html_source,
    )
    artifacts = build_artifact_bundle(
        sample.id,
        enriched_html_source,
        html_path=enriched_html_path,
        geometry_backend=artifact_backend,
    )
    reference_image_path = resolve_reference_image_path(
        sample.id,
        declared_path=sample.reference_metadata.reference_image_path,
    )
    if reference_image_path is not None:
        artifacts.reference_image_path = str(reference_image_path)
    print('[XIN_DEBUG_BLD] sid=' + str(sample.id) + ' declared=' + repr(sample.reference_metadata.reference_image_path) + ' resolved=' + repr(reference_image_path) + ' artifacts_ref=' + repr(artifacts.reference_image_path), flush=True)
    artifacts.metadata["semantic_enrichment"] = enrichment_summary
    artifacts.metadata["original_html_path"] = str(html_path)
    artifacts.metadata["source_artifact_path"] = str(source_artifact_path)
    artifacts.metadata["enriched_html_path"] = str(enriched_html_path)
    artifacts.metadata["sample_artifact_dir"] = str(get_sample_artifact_dir(sample.id))
    artifacts.metadata["reference_image_path"] = (
        str(reference_image_path) if reference_image_path is not None else None
    )
    artifacts.metadata["reference_image_wh"] = sample.reference_metadata.reference_image_wh
    artifacts.metadata["semantic_attribute_contract"] = sample.semantic_attribute_contract
    registry = build_default_registry()
    results: List[ValidationResult] = []
    hard_failures: List[str] = []
    render_failed = False

    for gate_requirement in _gate_requirements(sample):
        gate_validator = get_validator(gate_requirement.target["validator_name"], registry=registry)
        gate_result = gate_validator.validate(gate_requirement, artifacts)
        results.append(gate_result)
        if gate_result.status == ValidationStatus.FAIL:
            hard_failures.append(gate_requirement.id)
            # Check evidence to determine if render-specific blockers were hit
            # (render_image_missing, blank_output, render_timeout) or if
            # compile failed (which also means render can't succeed).
            evidence = gate_result.evidence if isinstance(gate_result.evidence, dict) else {}
            compile_failed = evidence.get("compile_failed", False)
            blocker_issues = evidence.get("blocker_issues", [])
            if compile_failed or blocker_issues:
                render_failed = True

    # Validators that require a rendered image -- skip (score 0) when render failed.
    _RENDER_DEPENDENT_IDS = {"boundary", "layout", "whitespace", "color", "design"}

    for runtime_requirement in _runtime_requirements(sample, artifacts, registry=registry):
        # Skip render-dependent validators when render failed, but still
        # record a FAIL result so they appear in the report with score 0.
        if render_failed and runtime_requirement.id in _RENDER_DEPENDENT_IDS:
            results.append(ValidationResult(
                requirement_id=runtime_requirement.id,
                validator_name=runtime_requirement.target.get("validator_name", "unknown"),
                status=ValidationStatus.FAIL,
                score=0.0,
                expected=runtime_requirement.expected,
                measured=None,
                tolerance=runtime_requirement.tolerance,
                message="Skipped: render failed.",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
                debug={"implemented": True, "render_failed": True},
            ))
            continue
        if runtime_requirement.id == "layout" and artifacts.layout_matching:
            artifacts.layout_matching = {}
        runtime_validator = get_validator_for_requirement(runtime_requirement, registry=registry)
        result = runtime_validator.validate(runtime_requirement, artifacts)
        results.append(result)
        if runtime_requirement.id == "layout" and artifacts.layout_matching:
            layout_matching_path = write_layout_matching_artifact(sample_id=sample.id, payload=artifacts.layout_matching)
            artifacts.metadata["layout_matching_path"] = str(layout_matching_path)
        if result.status == ValidationStatus.FAIL and runtime_requirement.severity.value in {"blocker", "critical"}:
            hard_failures.append(runtime_requirement.id)

    if include_legacy_requirements:
        for requirement in sample.verifiable_requirements:
            if not requirement.enabled:
                continue
            if requirement.category in {
                RequirementCategory.LAYOUT_RELATION,
                RequirementCategory.LAYOUT_DISTRIBUTION,
                RequirementCategory.ORDER,
                RequirementCategory.GROUPING,
                RequirementCategory.REPEATED_STRUCTURE,
            }:
                continue
            if requirement.category in {RequirementCategory.COUNT, RequirementCategory.CHART_DATA}:
                continue
            validator = get_validator_for_requirement(requirement, registry=registry)
            result = validator.validate(requirement, artifacts)
            results.append(result)
            if result.status == ValidationStatus.FAIL and requirement.severity.value in {"blocker", "critical"}:
                hard_failures.append(requirement.id)

    execution_pass = not hard_failures
    summary = _build_summary(
        sample,
        results,
        execution_pass,
        enrichment_summary,
        include_legacy_requirements=include_legacy_requirements,
    )
    report = ValidationReport(
        sample_id=sample.id,
        execution_pass=execution_pass,
        quality_pass=bool((summary.get("passes") or {}).get("quality_pass")),
        overall_pass=bool((summary.get("passes") or {}).get("overall_pass")),
        hard_failures=hard_failures,
        results=results,
        summary=summary,
    )
    write_validation_report_artifact(
        sample_id=sample.id,
        backend=artifact_backend,
        payload=serialize_validation_report(report),
    )
    return report


async def validate_sample_async(
    sample: RefinedSample,
    html_source: str,
    html_path: Path,
    *,
    artifact_backend: str = "auto",
    artifact_root: Path | None = None,
    include_design_validator: bool = True,
) -> Dict[str, Any]:
    """Async fast-path validation for RL reward computation.

    Differences from validate_sample():
    - Uses build_artifact_bundle_fast (no disk artifact writes)
    - Runs non-gate validators in parallel via asyncio
    - Returns scoring dict directly (no ValidationReport serialization)
    - Skips legacy requirements
    - Skips write_validation_report_artifact

    Returns:
        {"overall_score": float, "per_validator": {str: float},
         "result_feedback": [str], "render_image_path": str, ...}
    """
    artifacts = await build_artifacts_async(sample, html_source, html_path, artifact_root=artifact_root)
    return await score_validators_async(
        sample,
        artifacts,
        include_design_validator=include_design_validator,
    )


async def build_artifacts_async(
    sample: RefinedSample,
    html_source: str,
    html_path: Path,
    *,
    artifact_root: Path | None = None,
) -> Any:
    """Build artifact bundle (Playwright render + reference image) asynchronously.

    This is the expensive I/O step (~2-3s). Returns the ArtifactBundle with
    render_image_path populated, ready for both validators and VLM scoring.

    Split from validate_sample_async so callers can start VLM scoring
    immediately after artifacts are built, in parallel with validators.
    """
    loop = asyncio.get_running_loop()

    from verifier.artifacts.html import build_artifact_bundle_fast
    from verifier.artifacts.semantic import apply_semantic_attribute_contract
    from verifier.artifacts.paths import set_artifact_root, resolve_reference_image_path

    set_artifact_root(artifact_root)

    enriched_html_source, enrichment_summary = apply_semantic_attribute_contract(
        html_source,
        sample.semantic_attribute_contract,
    )

    # Write enriched HTML to temp path for Playwright to load
    import tempfile
    enriched_html_path = html_path
    if enriched_html_source != html_source:
        tmp_enriched = Path(artifact_root or tempfile.gettempdir()) / f"{sample.id}_enriched.html"
        tmp_enriched.parent.mkdir(parents=True, exist_ok=True)
        tmp_enriched.write_text(enriched_html_source, encoding="utf-8")
        enriched_html_path = tmp_enriched

    # Build artifacts in a SEPARATE PROCESS (Playwright must be isolated from
    # the host AgentLoopWorker uvloop — see _get_render_executor docstring).
    # Wrapped with timeout + BrokenProcessPool self-heal so a single chromium
    # crash/OOM doesn't poison the pool for the rest of training.
    # NOTE: artifact_root must be forwarded explicitly because thread-local
    # set_artifact_root() in this process does NOT propagate to the worker.
    artifacts = await _run_render_with_recovery(
        sample.id,
        enriched_html_source,
        str(enriched_html_path) if enriched_html_path else None,
        str(artifact_root) if artifact_root else None,
    )

    # Reference image
    reference_image_path = resolve_reference_image_path(
        sample.id,
        declared_path=sample.reference_metadata.reference_image_path,
    )
    if reference_image_path is not None:
        artifacts.reference_image_path = str(reference_image_path)
    print('[XIN_DEBUG_BLD] sid=' + str(sample.id) + ' declared=' + repr(sample.reference_metadata.reference_image_path) + ' resolved=' + repr(reference_image_path) + ' artifacts_ref=' + repr(artifacts.reference_image_path), flush=True)
    artifacts.metadata["reference_image_wh"] = sample.reference_metadata.reference_image_wh

    return artifacts


async def score_validators_async(
    sample: RefinedSample,
    artifacts: Any,
    *,
    include_design_validator: bool = True,
) -> Dict[str, Any]:
    """Run gate + runtime validators on pre-built artifacts asynchronously.

    Split from validate_sample_async so this can run in parallel with VLM scoring
    (both only need the artifact bundle, not each other's results).

    Returns:
        {"overall_score": float, "per_validator": {str: float},
         "result_feedback": [str], "render_image_path": str, ...}
    """
    loop = asyncio.get_running_loop()

    # --- Gate validators (serial) ---
    registry = build_default_registry()
    results: List[ValidationResult] = []
    render_failed = False

    for gate_req in _gate_requirements(sample):
        gate_validator = get_validator(gate_req.target["validator_name"], registry=registry)
        gate_result = gate_validator.validate(gate_req, artifacts)
        results.append(gate_result)
        if gate_result.status == ValidationStatus.FAIL:
            evidence = gate_result.evidence if isinstance(gate_result.evidence, dict) else {}
            if evidence.get("compile_failed") or evidence.get("blocker_issues"):
                render_failed = True

    # --- Runtime validators (parallel) ---
    _RENDER_DEPENDENT_IDS = {"boundary", "layout", "whitespace", "color", "design"}
    runtime_reqs = _runtime_requirements(
        sample,
        artifacts,
        registry=registry,
        include_design_validator=include_design_validator,
    )

    async def _run_one_validator(req: Requirement) -> ValidationResult:
        print('[XIN_DEBUG_RUN] req.id=' + str(req.id) + ' validator=' + str(req.target.get('validator_name')) + ' ref_path=' + repr(artifacts.reference_image_path), flush=True)
        if render_failed and req.id in _RENDER_DEPENDENT_IDS:
            return ValidationResult(
                requirement_id=req.id,
                validator_name=req.target.get("validator_name", "unknown"),
                status=ValidationStatus.FAIL,
                score=0.0,
                expected=req.expected,
                measured=None,
                tolerance=req.tolerance,
                message="Skipped: render failed.",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
                debug={"implemented": True, "render_failed": True},
            )
        validator = get_validator_for_requirement(req, registry=registry)
        return await loop.run_in_executor(
            _VALIDATOR_EXECUTOR,
            validator.validate, req, artifacts,
        )

    runtime_results_raw = await asyncio.gather(
        *[_run_one_validator(req) for req in runtime_reqs],
        return_exceptions=True,
    )

    # Convert exceptions to FAIL results
    runtime_results = []
    for i, r in enumerate(runtime_results_raw):
        if isinstance(r, Exception):
            req = runtime_reqs[i]
            _logger.warning("Validator %s failed with exception: %s", req.id, r)
            runtime_results.append(ValidationResult(
                requirement_id=req.id,
                validator_name=req.target.get("validator_name", "unknown"),
                status=ValidationStatus.FAIL,
                score=0.0,
                expected=req.expected,
                measured=None,
                tolerance=req.tolerance,
                message=f"Validator exception: {r}",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
                debug={"implemented": True, "exception": str(r)},
            ))
        else:
            runtime_results.append(r)
    results.extend(runtime_results)

    # --- Scoring ---
    execution_pass = not any(
        r.status == ValidationStatus.FAIL
        for r in results
        if r.requirement_id == "execution"
    )
    score_summary = _summarize_scores(sample, results, execution_pass)

    # Collect feedback for refine prompt
    result_feedback: List[str] = []
    for r in results:
        status_val = r.status.value if hasattr(r.status, "value") else str(r.status)
        if status_val in ("fail", "partial") and r.message and "Skipped" not in r.message:
            result_feedback.append(f"[{r.requirement_id}] {r.message}")

    return {
        "overall_score": float(score_summary["overall_score"]),
        "per_validator": {str(k): float(v) for k, v in score_summary["per_validator"].items()},
        "degraded_requirements": score_summary.get("degraded_requirements", []),
        "critical_failures": score_summary.get("critical_failures", []),
        "result_feedback": result_feedback,
        "render_image_path": artifacts.render_image_path,
    }
