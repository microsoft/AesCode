"""Gate validators -- ExecutionValidator (unified compile + render).

Layer 1 of stage-3 verifier. The execution gate runs first in strict sequence.
Only blocker failures short-circuit downstream validators.

Key design notes
----------------
* ExecutionValidator : HTML + CSS structural checks (compile phase);
                       render image presence + blank detection + timeout +
                       JS/console error health scoring (render phase).

Backward-compatible aliases ``CompileValidator`` and ``RenderValidator``
point to ``ExecutionValidator``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from verifier.base import BaseValidator
from verifier.types import ArtifactBundle, Requirement, ValidationResult, ValidationStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_render_image(artifacts: ArtifactBundle) -> Optional[Path]:
    render_path = artifacts.render_image_path
    if not render_path:
        return None
    path = Path(render_path)
    return path if path.exists() else None


def _render_blank_signals(render_path: Path) -> Dict[str, float | bool]:
    import numpy as np
    from PIL import Image, ImageFile

    try:
        _orig = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img = Image.open(str(render_path))
            img.load()
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = _orig
        image = np.asarray(img.convert("L"))
    except Exception:
        import cv2
        image = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return {
            "readable": False,
            "variance": 0.0,
            "dark_pixel_ratio": 0.0,
            "near_white_ratio": 1.0,
            "looks_blank": False,
        }

    variance = float(image.var())
    dark_pixel_ratio = float((image < 235).mean())
    near_white_ratio = float((image >= 245).mean())

    # Extremely conservative: only classify as blank when the page is both
    # close to monochrome and has almost no darker-than-background pixels.
    looks_blank = variance < 8.0 and dark_pixel_ratio < 0.005
    return {
        "readable": True,
        "variance": variance,
        "dark_pixel_ratio": dark_pixel_ratio,
        "near_white_ratio": near_white_ratio,
        "looks_blank": looks_blank,
    }


# ---------------------------------------------------------------------------
# External resource allowlist (per html_contracts.md spec)
# ---------------------------------------------------------------------------
# Physical enforcement happens in verifier/artifacts/html.py via Playwright
# network interception (route.abort).  The compile phase only performs a
# static audit for observability -- it records violations in measured/evidence
# but does NOT penalise the score.  The natural consequence of a blocked
# resource (broken layout, missing styles) is picked up by downstream
# validators, creating a clean RL reward signal without heuristic constants.

_ALLOWED_EXTERNAL_PREFIXES: tuple[str, ...] = (
    "https://placehold.co",
    "http://placehold.co",
)
_ALLOWED_EXTERNAL_EXACT: frozenset[str] = frozenset({
    "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js",
})
_EXTERNAL_RESOURCE_ATTRS: tuple[tuple[str, str], ...] = (
    ("script", "src"),
    ("link", "href"),
    ("img", "src"),
    ("source", "src"),
    ("iframe", "src"),
)


def _is_allowed_external_url(url: str) -> bool:
    if url in _ALLOWED_EXTERNAL_EXACT:
        return True
    return any(url.startswith(prefix) for prefix in _ALLOWED_EXTERNAL_PREFIXES)


def _check_external_resources(soup: Any) -> List[str]:
    """Return violation strings for any disallowed external resource URLs."""
    violations: List[str] = []
    seen: set[str] = set()
    for tag, attr in _EXTERNAL_RESOURCE_ATTRS:
        for element in soup.find_all(tag):
            url = str(element.get(attr) or "").strip()
            if not url or url.startswith("data:"):
                continue
            if not url.startswith(("http://", "https://")):
                continue  # relative paths are fine
            if url in seen:
                continue
            seen.add(url)
            if not _is_allowed_external_url(url):
                violations.append(f"disallowed_external_resource:{tag}:{url[:120]}")
    return violations


# ---------------------------------------------------------------------------
# JS/Console error filtering and classification
# ---------------------------------------------------------------------------

# Messages matching any of these patterns are silently dropped before scoring.
# * 404 / network failures protect semantic placeholder images (placehold.co)
#   that intentionally return 404 during offline/test runs.
# * Internal probe errors (overflow detection machinery) are not page errors.
_NETWORK_ERROR_FRAGMENTS: tuple[str, ...] = (
    "404",
    "net::ERR_",
    "Failed to load resource",
    "ERR_NAME_NOT_RESOLVED",
    "ERR_CONNECTION_REFUSED",
    "ERR_INTERNET_DISCONNECTED",
    "ERR_ABORTED",
)

# JS error types that indicate broken logic -> N_fatal
_FATAL_JS_FRAGMENTS: tuple[str, ...] = (
    "SyntaxError",
    "TypeError",
    "ReferenceError",
    "RangeError",
    "URIError",
    "EvalError",
)

# playwright_errors whose type is set internally by build_artifact_bundle;
# they reflect instrumentation failures, not page-level JS errors.
_INTERNAL_ERROR_TYPES: frozenset[str] = frozenset({
    "overflow_probe_failed",
    "overflow_debug_capture_failed",
    "playwright_runtime_failed",
})


def _is_network_error(message: str) -> bool:
    return any(fragment in message for fragment in _NETWORK_ERROR_FRAGMENTS)


def _classify_playwright_errors(
    js_errors: List[Dict[str, Any]],
    console_errors: List[str],
) -> Dict[str, Any]:
    """Filter and classify Playwright errors into N_fatal and N_warn.

    Filtering rules (applied in order):
    1. Internal probe errors (type in _INTERNAL_ERROR_TYPES) -> ignored.
    2. Any error whose message matches a network/404 pattern -> ignored.
    3. console.info strings never reach here (dropped at source in html.py).

    Classification:
    * js_runtime errors matching _FATAL_JS_FRAGMENTS -> fatal
    * all other surviving js_runtime errors          -> warn
    * surviving console.error strings                -> warn
    """
    fatal_messages: List[str] = []
    warn_messages: List[str] = []

    for error in js_errors:
        error_type = str(error.get("type") or "")
        if error_type in _INTERNAL_ERROR_TYPES:
            continue
        msg = str(error.get("message") or "")
        if _is_network_error(msg):
            continue
        if any(fragment in msg for fragment in _FATAL_JS_FRAGMENTS):
            fatal_messages.append(msg)
        else:
            warn_messages.append(msg)

    for text in console_errors:
        msg = str(text or "")
        if _is_network_error(msg):
            continue
        # console.info is already excluded at source; all remaining are console.error
        warn_messages.append(msg)

    return {
        "n_fatal": len(fatal_messages),
        "n_warn": len(warn_messages),
        "fatal_messages_preview": fatal_messages[:5],
        "warn_messages_preview": warn_messages[:5],
    }


# ---------------------------------------------------------------------------
# ExecutionValidator (merged compile + render)
# ---------------------------------------------------------------------------


class ExecutionValidator(BaseValidator):
    """Unified execution gate: compile checks then render checks.

    Compile phase
    -------------
    HTML parse      BLOCKER  BeautifulSoup -- empty source is a hard error;
                             missing html/body root is a warning only.
    CSS syntax      AUDIT    cssutils -- lint all <style> tags via
                             rule-inspection (ERROR nodes in parsed sheet).
                             Recorded as warning only -- actual CSS breakage
                             is caught downstream by render-based validators.
    External res    AUDIT    Static scan for disallowed external URLs.
                             Recorded in measured/evidence for observability
                             but carries **zero score penalty**.

    Render phase
    ------------
    Render image    BLOCKER  render image missing or unreadable.
    Blank output    BLOCKER  pixel variance < threshold.
    Timeout         BLOCKER  render_duration_ms > RENDER_TIMEOUT_MS.
    JS errors       NORMAL   fatal JS errors penalize via exp(-alpha * N_fatal).

    Scoring
    -------
    If compile fails (empty HTML, CSS errors) -> score 0.0, FAIL.
    If render has blocker issues (missing image, blank, timeout) -> score 0.0, FAIL.
    If JS fatal errors -> score = exp(-0.4 * n_fatal).
    Otherwise score = 1.0.
    """

    name = "ExecutionValidator"
    supported_categories = ("compile", "render")

    # Hard timeout: renders exceeding this are treated as episode termination.
    RENDER_TIMEOUT_MS: float = 15_000
    # JS fatal error health score -- exp(-alpha * N_fatal)
    ALPHA: float = 0.4   # fatal error decay rate (1 fatal -> x0.67)

    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        source = artifacts.html_source or ""
        compile_errors: List[str] = []
        compile_warnings: List[str] = []
        soup = BeautifulSoup(source, "html.parser")

        # ── Compile phase ────────────────────────────────────────────

        # HTML structure checks
        if not source.strip():
            compile_errors.append("html_source_empty")
        if not soup.find("html") and not soup.find("body"):
            compile_warnings.append("no_html_root")

        # CSS syntax -- rule-inspection via cssutils error nodes
        import cssutils  # type: ignore[import-not-found]

        cssutils.log.setLevel(logging.CRITICAL)
        for style_tag in soup.find_all("style"):
            css_text = style_tag.string or style_tag.get_text("\n", strip=False)
            try:
                sheet = cssutils.parseString(css_text)
                for rule in sheet:
                    if hasattr(rule, "cssText") and "ERROR" in str(rule.cssText):
                        compile_warnings.append(f"css_error:{str(rule.cssText)[:80]}")
            except Exception as exc:  # noqa: BLE001
                compile_warnings.append(f"css_error:{str(exc)[:160]}")

        # External resource audit (observability only -- zero score impact)
        resource_violations = _check_external_resources(soup)

        # If compile fails, short-circuit -- no render checks needed.
        if compile_errors:
            measured: Dict[str, Any] = {
                "compile_errors": compile_errors,
                "compile_warnings": compile_warnings,
                "resource_violation_count": len(resource_violations),
                # render fields (not executed)
                "render_image_path": None,
                "render_duration_ms": None,
                "render_blank": False,
                "render_variance": 0.0,
                "render_dark_pixel_ratio": 0.0,
                "render_near_white_ratio": 0.0,
                "n_fatal": 0,
                "n_warn": 0,
            }
            evidence: Dict[str, Any] = {
                "evidence_basis": "explicit",
                "evidence_sources": ["html_source"],
                "resource_violations": resource_violations,
                "compile_failed": True,
                "blocker_issues": [],
                "normal_issues": [],
                "fatal_messages_preview": [],
                "warn_messages_preview": [],
            }
            return self.build_result(
                requirement,
                status=ValidationStatus.FAIL,
                score=0.0,
                measured=measured,
                message="Execution gate failed: compile errors.",
                evidence=evidence,
            )

        # ── Render phase ─────────────────────────────────────────────

        blocker_issues: List[str] = []
        normal_issues: List[str] = []
        variance = 0.0
        dark_pixel_ratio = 0.0
        near_white_ratio = 1.0
        render_path = _require_render_image(artifacts)
        if render_path is None:
            blocker_issues.append("render_image_missing")
        else:
            blank_signals = _render_blank_signals(render_path)
            if not bool(blank_signals["readable"]):
                blocker_issues.append("render_image_unreadable")
            else:
                variance = float(blank_signals["variance"])
                dark_pixel_ratio = float(blank_signals["dark_pixel_ratio"])
                near_white_ratio = float(blank_signals["near_white_ratio"])
                if bool(blank_signals["looks_blank"]):
                    blocker_issues.append("blank_output")

        render_duration_ms = artifacts.render_duration_ms
        if render_duration_ms is not None and render_duration_ms > self.RENDER_TIMEOUT_MS:
            blocker_issues.append("render_timeout")

        error_classification = _classify_playwright_errors(
            list(artifacts.playwright_errors or []),
            list(artifacts.playwright_console_errors or []),
        )
        n_fatal = error_classification["n_fatal"]
        n_warn = error_classification["n_warn"]
        if n_fatal:
            normal_issues.append("js_fatal_errors")

        # ── Scoring ──────────────────────────────────────────────────

        if blocker_issues:
            status = ValidationStatus.FAIL
            score = 0.0
        elif n_fatal:
            score = round(math.exp(-self.ALPHA * n_fatal), 4)
            status = ValidationStatus.PASS if score >= 0.9 else ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.PASS
            score = 1.0

        measured = {
            "compile_errors": compile_errors,
            "compile_warnings": compile_warnings,
            "resource_violation_count": len(resource_violations),
            "render_image_path": str(render_path) if render_path else None,
            "render_duration_ms": render_duration_ms,
            "render_blank": bool(artifacts.render_blank),
            "render_variance": round(variance, 4),
            "render_dark_pixel_ratio": round(dark_pixel_ratio, 4),
            "render_near_white_ratio": round(near_white_ratio, 4),
            "n_fatal": n_fatal,
            "n_warn": n_warn,
        }
        evidence = {
            "evidence_basis": "explicit" if render_path else "unknown",
            "evidence_sources": ["html_source", "render_image", "playwright_runtime"],
            "resource_violations": resource_violations,
            "compile_failed": False,
            "blocker_issues": blocker_issues,
            "normal_issues": normal_issues,
            "fatal_messages_preview": error_classification["fatal_messages_preview"],
            "warn_messages_preview": error_classification["warn_messages_preview"],
        }

        if status == ValidationStatus.PASS:
            message = "Execution gate passed."
        else:
            message = "Execution gate reported issues."

        # DEBUG (2026-05-29): dump measured + evidence whenever score < 1.0
        if score < 1.0:
            try:
                import logging as _l
                _l.getLogger(__name__).warning(
                    "EXEC_DEBUG score=%s blockers=%s normals=%s render_path=%s render_dur_ms=%s "
                    "variance=%.4f dark_ratio=%.4f near_white=%.4f n_fatal=%s n_warn=%s "
                    "fatal_preview=%s warn_preview=%s",
                    score, blocker_issues, normal_issues,
                    str(render_path) if render_path else None,
                    render_duration_ms, variance, dark_pixel_ratio, near_white_ratio,
                    n_fatal, n_warn,
                    error_classification.get("fatal_messages_preview"),
                    error_classification.get("warn_messages_preview"),
                )
            except Exception:
                pass

        return self.build_result(
            requirement,
            status=status,
            score=score,
            measured=measured,
            message=message,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
CompileValidator = ExecutionValidator
RenderValidator = ExecutionValidator
