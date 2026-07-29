from __future__ import annotations

import logging
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def _response_text(completion) -> str:
    if isinstance(completion, list) and completion:
        item = completion[0]
        return item.get("content", "") if isinstance(item, dict) else str(item)
    return str(completion)


def _extract_html(text: str) -> str:
    match = re.search(r"```(?:html)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(<html[\s\S]*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _html_quality(html: str) -> float:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return 0.0

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return 0.0

    score = 0.0
    score += 0.10 * bool(soup.find("html"))
    score += 0.10 * bool(soup.find("head"))
    score += 0.10 * bool(soup.find("body"))

    body = soup.find("body")
    body_text = body.get_text(strip=True) if body else ""
    score += 0.20 * (len(body_text) > 30)

    all_css = " ".join(str(style.string or "") for style in soup.find_all("style"))
    css_props = {prop for prop in re.findall(r"([\w-]+)\s*:", all_css) if len(prop) > 2}
    score += 0.30 * min(1.0, len(css_props) / 20.0)

    max_same = 0
    for parent in soup.find_all(True):
        tag_counts: dict[str, int] = {}
        for child in parent.children:
            if hasattr(child, "name") and child.name:
                tag_counts[child.name] = tag_counts.get(child.name, 0) + 1
        if tag_counts:
            max_same = max(max_same, max(tag_counts.values()))
    no_halluc = 1.0 if max_same <= 50 else max(0.0, 1.0 - (max_same - 50) / 200.0)
    score += 0.20 * no_halluc

    return float(min(1.0, score))


# ---------------------------------------------------------------------------
# Verifier-based reward helpers
# ---------------------------------------------------------------------------

# Per-call cache: keyed by (sample_id, hash(html)) to avoid re-running
# the verifier pipeline on the same HTML within a single training step.
# Cleared at the start of each batch via _clear_verifier_cache().
_verifier_cache: Dict[tuple, Dict[str, Any]] = {}
_verifier_temp_dirs: List[str] = []


def _clear_verifier_cache() -> None:
    """Clear the per-batch verifier cache and clean up temp dirs."""
    import shutil
    for d in _verifier_temp_dirs:
        shutil.rmtree(d, ignore_errors=True)
    _verifier_temp_dirs.clear()
    _verifier_cache.clear()


def _cache_key(sample_id: str, html: str) -> tuple:
    return (sample_id, hash(html))


def _run_verifier_fast(
    html: str,
    sample_id: str,
    verifiable_requirements: List[Dict[str, Any]],
    reference_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the verifier pipeline on an HTML string using static (DOM-only) backend.

    Uses a per-batch cache so the same (sample_id, html) pair is only verified once,
    even when multiple reward functions call this for the same completion.

    Returns the scoring summary dict with overall_score, per_validator scores,
    degraded_requirements, critical_failures,
    and per-result feedback messages for refine prompt injection.
    Falls back to heuristic scoring on import/runtime errors.
    """
    key = _cache_key(sample_id, html)
    if key in _verifier_cache:
        return _verifier_cache[key]

    try:
        from verifier.pipeline.parser import _parse_sample
        from verifier.pipeline.runner import validate_sample
        from verifier.artifacts.paths import set_artifact_root
    except ImportError:
        logger.debug("verifier not importable; falling back to heuristic.")
        q = _html_quality(html)
        return {"overall_score": q, "per_validator": {}}

    try:
        payload = {
            "id": sample_id,
            "reference_metadata": reference_metadata,
            "generation_prompt": "",
            "verifiable_requirements": verifiable_requirements,
        }
        sample = _parse_sample(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            html_file = tmp_path / f"{sample_id}.html"
            html_file.write_text(html, encoding="utf-8")
            set_artifact_root(tmp_path / "artifacts")
            report = validate_sample(
                sample,
                html_file,
                artifact_backend="auto",
                artifact_root=tmp_path / "artifacts",
            )
        summary = report.summary or {}
        scores = summary.get("scores", summary)
        overall_score = float(scores.get("overall_score", 0.0))
        per_validator = {str(k): float(v) for k, v in scores.get("per_validator", {}).items()}

        # Collect per-result feedback messages for failed/partial items
        result_feedback: List[str] = []
        for r in report.results:
            status_val = r.status.value if hasattr(r.status, "value") else str(r.status)
            if status_val in ("fail", "partial") and r.message and "Skipped" not in r.message:
                result_feedback.append(f"[{r.requirement_id}] {r.message}")

        result = {
            "overall_score": overall_score,
            "per_validator": per_validator,
            "degraded_requirements": scores.get("degraded_requirements", []),
            "critical_failures": scores.get("critical_failures", []),
            "result_feedback": result_feedback,
        }
        _verifier_cache[key] = result
        return result
    except Exception as exc:
        logger.debug("Verifier failed for %s: %s. Falling back to heuristic.", sample_id, exc)
        q = _html_quality(html)
        result = {"overall_score": q, "per_validator": {}}
        _verifier_cache[key] = result
        return result


async def _run_verifier_fast_async(
    html: str,
    sample_id: str,
    verifiable_requirements: List[Dict[str, Any]],
    reference_metadata: Dict[str, Any],
    include_design_validator: bool = True,
) -> Dict[str, Any]:
    """Async version of _run_verifier_fast using validate_sample_async.

    Uses the async fast-path: parallel validators, no disk artifact writes.
    Results are cached per (sample_id, html_hash).
    """
    import asyncio

    key = _cache_key(sample_id, html)
    if key in _verifier_cache:
        return _verifier_cache[key]

    try:
        from verifier.pipeline.parser import _parse_sample
        from verifier.pipeline.runner import validate_sample_async
    except ImportError:
        logger.debug("verifier async not importable; falling back to sync.")
        return _run_verifier_fast(html, sample_id, verifiable_requirements, reference_metadata)

    try:
        import tempfile
        from pathlib import Path

        payload = {
            "id": sample_id,
            "reference_metadata": reference_metadata,
            "generation_prompt": "",
            "verifiable_requirements": verifiable_requirements,
        }
        sample = _parse_sample(payload)

        tmpdir = tempfile.mkdtemp(prefix=f"vredo2_reward_{sample_id}_")
        _verifier_temp_dirs.append(tmpdir)
        tmp_path = Path(tmpdir)
        html_file = tmp_path / f"{sample_id}.html"
        html_file.write_text(html, encoding="utf-8")

        result = await validate_sample_async(
            sample,
            html_source=html,
            html_path=html_file,
            artifact_backend="playwright",
            artifact_root=tmp_path / "artifacts",
            include_design_validator=include_design_validator,
        )

        _verifier_cache[key] = result
        return result
    except Exception as exc:
        # Clean up on failure
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        logger.debug("Async verifier failed for %s: %s. Falling back to sync.", sample_id, exc)
        return _run_verifier_fast(html, sample_id, verifiable_requirements, reference_metadata)


async def _build_artifacts_and_score_async(
    html: str,
    sample_id: str,
    verifiable_requirements: List[Dict[str, Any]],
    reference_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build artifacts + run validators, returning artifacts handle for VLM.

    Unlike _run_verifier_fast_async, this returns the artifacts alongside the
    scoring dict so callers can start VLM scoring using the render image in
    parallel with validators.

    Returns:
        {"overall_score": float, "per_validator": {...}, "render_image_path": str,
         "_artifacts": ArtifactBundle, "_sample": RefinedSample, ...}
    """
    key = _cache_key(sample_id, html)
    if key in _verifier_cache:
        return _verifier_cache[key]

    try:
        from verifier.pipeline.parser import _parse_sample
        from verifier.pipeline.runner import build_artifacts_async, score_validators_async
    except ImportError:
        logger.debug("verifier split-async not importable; falling back to unified.")
        return await _run_verifier_fast_async(html, sample_id, verifiable_requirements, reference_metadata)

    try:
        import tempfile
        from pathlib import Path

        payload = {
            "id": sample_id,
            "reference_metadata": reference_metadata,
            "generation_prompt": "",
            "verifiable_requirements": verifiable_requirements,
        }
        sample = _parse_sample(payload)

        tmpdir = tempfile.mkdtemp(prefix=f"vredo2_reward_{sample_id}_")
        _verifier_temp_dirs.append(tmpdir)
        tmp_path = Path(tmpdir)
        html_file = tmp_path / f"{sample_id}.html"
        html_file.write_text(html, encoding="utf-8")

        # Step 1: Build artifacts (Playwright render) — the slow I/O step
        artifacts = await build_artifacts_async(
            sample, html, html_file, artifact_root=tmp_path / "artifacts",
        )

        # Step 2: Run validators (can be gathered with VLM by caller)
        result = await score_validators_async(sample, artifacts)

        _verifier_cache[key] = result
        return result
    except Exception as exc:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        logger.debug("Split-async verifier failed for %s: %s. Falling back.", sample_id, exc)
        return await _run_verifier_fast_async(html, sample_id, verifiable_requirements, reference_metadata)


def _run_verifier_batch(
    htmls: List[str],
    sample_ids: List[str],
    verifiable_requirements_list: List[List[Dict[str, Any]]],
    reference_metadata_list: List[Dict[str, Any]],
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Run verifier on a batch of HTMLs in parallel using threads.

    Results are cached, so subsequent calls to _run_verifier_fast for the same
    HTML will be instant cache hits.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # First pass: identify which items need computation (not in cache)
    results: List[Optional[Dict[str, Any]]] = [None] * len(htmls)
    to_compute: List[int] = []
    for i, (html, sid) in enumerate(zip(htmls, sample_ids)):
        key = _cache_key(sid, html)
        if key in _verifier_cache:
            results[i] = _verifier_cache[key]
        else:
            to_compute.append(i)

    if not to_compute:
        return results  # type: ignore[return-value]

    # Run uncached items in parallel
    def _verify_one(idx: int) -> tuple:
        return idx, _run_verifier_fast(
            htmls[idx],
            sample_ids[idx],
            verifiable_requirements_list[idx],
            reference_metadata_list[idx],
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(to_compute))) as pool:
        futures = {pool.submit(_verify_one, i): i for i in to_compute}
        for future in as_completed(futures):
            idx, score = future.result()
            results[idx] = score

    return results  # type: ignore[return-value]


def physical_constraint_reward(prompts, completions, **kwargs) -> list[float]:
    del prompts, kwargs
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        warnings.warn("beautifulsoup4 missing; physical_constraint_reward returns 0.")
        return [0.0] * len(completions)

    rewards: list[float] = []
    for completion in completions:
        html = _extract_html(_response_text(completion))

        try:
            soup = BeautifulSoup(html, "html.parser")
            syntax = 1.0
        except Exception:
            rewards.append(0.0)
            continue

        structure = (
            float(bool(soup.find("html")))
            + float(bool(soup.find("head")))
            + float(bool(soup.find("body")))
        ) / 3.0

        body = soup.find("body")
        body_text = body.get_text(strip=True) if body else ""
        non_empty = 1.0 if len(body_text) > 20 else 0.0

        max_same = 0
        for parent in soup.find_all(True):
            tag_counts: dict[str, int] = {}
            for child in parent.children:
                if hasattr(child, "name") and child.name:
                    tag_counts[child.name] = tag_counts.get(child.name, 0) + 1
            if tag_counts:
                max_same = max(max_same, max(tag_counts.values()))
        no_halluc = 1.0 if max_same <= 50 else max(0.0, 1.0 - (max_same - 50) / 200.0)

        style_css = " ".join(str(style.string or "") for style in soup.find_all("style"))
        inline_css = " ".join(tag.get("style", "") for tag in soup.find_all(style=True))
        all_css = style_css + " " + inline_css
        overflow_penalty = 0
        if re.search(r"margin(?:-\w+)?:\s*-\d{3,}px", all_css):
            overflow_penalty += 1
        if re.search(r"(?:left|top):\s*-\d{3,}px", all_css):
            overflow_penalty += 1
        if re.search(r"overflow\s*:\s*visible", all_css) and re.search(r"position\s*:\s*fixed", all_css):
            overflow_penalty += 1
        no_overflow = max(0.0, 1.0 - overflow_penalty * 0.33)

        rewards.append(float((syntax + structure + non_empty + no_halluc + no_overflow) / 5.0))
    return rewards


_LAYOUT_PROPS = frozenset(
    {
        "display",
        "flex",
        "grid",
        "position",
        "float",
        "align-items",
        "justify-content",
        "align-self",
        "flex-direction",
        "grid-template",
        "grid-column",
        "grid-row",
        "gap",
        "column-gap",
        "row-gap",
    }
)
_TYPO_PROPS = frozenset(
    {
        "font-family",
        "font-size",
        "font-weight",
        "line-height",
        "letter-spacing",
        "text-align",
        "text-decoration",
        "white-space",
        "word-break",
        "font-style",
    }
)
_COLOR_PROPS = frozenset(
    {
        "color",
        "background-color",
        "background",
        "border-color",
        "border",
        "outline-color",
        "box-shadow",
        "text-shadow",
    }
)


def aesthetic_quality_reward(prompts, completions, **kwargs) -> list[float]:
    del prompts, kwargs
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        warnings.warn("beautifulsoup4 missing; aesthetic_quality_reward returns 0.")
        return [0.0] * len(completions)

    rewards: list[float] = []
    for completion in completions:
        html = _extract_html(_response_text(completion))
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            rewards.append(0.0)
            continue

        style_css = " ".join(str(style.string or "") for style in soup.find_all("style"))
        inline_css = " ".join(tag.get("style", "") for tag in soup.find_all(style=True))
        all_css = style_css + " " + inline_css

        css_props = {prop for prop in re.findall(r"([\w-]+)\s*:", all_css) if len(prop) > 2}
        css_richness = min(1.0, len(css_props) / 40.0)
        layout = min(1.0, len(css_props & _LAYOUT_PROPS) / 5.0)
        typography = min(1.0, len(css_props & _TYPO_PROPS) / 4.0)
        colour = min(1.0, len(css_props & _COLOR_PROPS) / 3.0)
        has_viewport = float(bool(soup.find("meta", attrs={"name": "viewport"})))
        has_rel_unit = float(bool(re.search(r"\d+(?:em|rem|%|vw|vh)", all_css)))
        responsive = (has_viewport + has_rel_unit) / 2.0
        rewards.append(float((css_richness + layout + typography + colour + responsive) / 5.0))
    return rewards


def _apply_progress_mode(delta: float, mode: str, neg_scale: float = 0.3) -> float:
    """Apply a reward shaping mode to a raw progress delta.

    Modes:
      clip       (default) max(0, delta) — reward gains only; regressions ignored.
                 Regression is captured by absolute content/visual rewards, so
                 double-penalising is avoided.
      asymmetric           pos delta rewarded fully; neg delta scaled by neg_scale.
                 Useful as an ablation when you want a soft regression signal.
      full                 raw delta, positive or negative.
                 High variance; generally not recommended.
    """
    if mode == "clip":
        return max(0.0, delta)
    if mode == "asymmetric":
        return delta if delta >= 0.0 else neg_scale * delta
    # mode == "full"
    return delta


def progress_reward(
    prompts,
    completions,
    prev_htmls: Optional[list[str]] = None,
    current_step: "int | list[int]" = 1,
    progress_mode: str = "clip",
    progress_neg_scale: float = 0.3,
    **kwargs,
) -> list[float]:
    """Heuristic progress reward (delta of _html_quality scores).

    Args:
        current_step: Rollout step index. Can be an int (same for all) or
            list[int] (per-sample, as provided by TRL 1.0's reward kwargs).
        progress_mode: One of 'clip' (default), 'asymmetric', 'full'.
        progress_neg_scale: Scale applied to negative delta in 'asymmetric' mode.
    """
    del prompts, kwargs
    # TRL 1.0 passes per-sample kwargs as lists; accept both scalar and list
    _step = current_step[0] if isinstance(current_step, list) else current_step
    if _step <= 1 or prev_htmls is None:
        return [0.0] * len(completions)

    rewards: list[float] = []
    for completion, prev_html in zip(completions, prev_htmls):
        current_html = _extract_html(_response_text(completion))
        delta = _html_quality(current_html) - _html_quality(prev_html)
        rewards.append(float(_apply_progress_mode(delta, progress_mode, progress_neg_scale)))
    return rewards


# ---------------------------------------------------------------------------
# Verifier-backed reward functions (TRL GDPO compatible signature)
# ---------------------------------------------------------------------------

_DEFAULT_REF_META = {"reference_image_wh": "1920x1080", "reference_aspect_ratio": 1.78,
                     "reference_orientation": "landscape"}


def _prepare_verifier_args(
    completions, verifiable_requirements, reference_metadata, sample_id,
):
    """Normalize optional kwargs and extract HTMLs. Shared by all verifier rewards."""
    n = len(completions)
    if verifiable_requirements is None:
        verifiable_requirements = [[] for _ in range(n)]
    if reference_metadata is None:
        reference_metadata = [_DEFAULT_REF_META for _ in range(n)]
    if sample_id is None:
        sample_id = [f"sample_{i}" for i in range(n)]
    htmls = [_extract_html(_response_text(c)) for c in completions]
    return n, htmls, verifiable_requirements, reference_metadata, sample_id


def _ensure_batch_verified(
    htmls, sample_ids, verifiable_requirements, reference_metadata,
):
    """Pre-compute verifier scores for the entire batch in parallel.

    Uses the cache so that subsequent reward function calls are instant lookups.
    """
    # Check if all items are already cached
    uncached = [i for i, (h, s) in enumerate(zip(htmls, sample_ids))
                if _cache_key(s, h) not in _verifier_cache]
    if not uncached:
        return

    _run_verifier_batch(
        [htmls[i] for i in uncached],
        [sample_ids[i] for i in uncached],
        [verifiable_requirements[i] for i in uncached],
        [reference_metadata[i] for i in uncached],
    )


def verifier_execution_reward(
    prompts,
    completions,
    *,
    verifiable_requirements: Optional[List[List[Dict[str, Any]]]] = None,
    reference_metadata: Optional[List[Dict[str, Any]]] = None,
    sample_id: Optional[List[str]] = None,
    **kwargs,
) -> list[float]:
    """Execution (gate) reward: unified compile + render score.

    This is typically the FIRST reward function in the list, so it triggers
    the batch-parallel verifier computation for all completions. Subsequent
    reward functions will hit the cache.

    Intended to be used with a small weight (e.g. 0.2) since most outputs
    pass the execution gate.  Reads the single ExecutionValidator score
    from per_validator.
    """
    del prompts
    _clear_verifier_cache()  # fresh batch -> clear stale cache
    n, htmls, vr, rm, sids = _prepare_verifier_args(
        completions, verifiable_requirements, reference_metadata, sample_id,
    )
    _ensure_batch_verified(htmls, sids, vr, rm)
    results = []
    for i in range(n):
        pv = _verifier_cache.get(_cache_key(sids[i], htmls[i]), {}).get("per_validator", {})
        execution_s = pv.get("ExecutionValidator", 1.0)
        results.append(float(execution_s))
    return results


def verifier_overall_reward(
    prompts,
    completions,
    *,
    verifiable_requirements: Optional[List[List[Dict[str, Any]]]] = None,
    reference_metadata: Optional[List[Dict[str, Any]]] = None,
    sample_id: Optional[List[str]] = None,
    **kwargs,
) -> list[float]:
    """Overall reward: flat average of all per-validator scores.

    If batch was already verified by execution_reward, this is a pure cache lookup.
    """
    del prompts
    n, htmls, vr, rm, sids = _prepare_verifier_args(
        completions, verifiable_requirements, reference_metadata, sample_id,
    )
    _ensure_batch_verified(htmls, sids, vr, rm)
    return [float(_verifier_cache.get(_cache_key(sids[i], htmls[i]), {}).get("overall_score", 0.0))
            for i in range(n)]


# Legacy aliases — kept so existing training configs don't break immediately.
verifier_content_reward = verifier_overall_reward
verifier_visual_reward = verifier_overall_reward


def verifier_progress_reward(
    prompts,
    completions,
    *,
    prev_htmls: Optional[list[str]] = None,
    prev_scores: Optional[list[Dict[str, Any]]] = None,
    current_step: "int | list[int]" = 1,
    verifiable_requirements: Optional[List[List[Dict[str, Any]]]] = None,
    reference_metadata: Optional[List[Dict[str, Any]]] = None,
    sample_id: Optional[List[str]] = None,
    progress_mode: str = "clip",
    progress_neg_scale: float = 0.3,
    **kwargs,
) -> list[float]:
    """Verifier-backed progress reward: delta of overall_score between steps.

    If batch was already verified by execution_reward, current scores are
    pure cache lookups.  See _apply_progress_mode() for mode options.

    Args:
        current_step: Rollout step index. Can be an int (same for all) or
            list[int] (per-sample, as provided by TRL 1.0's reward kwargs).
        progress_mode: One of 'clip' (default), 'asymmetric', 'full'.
        progress_neg_scale: Scale applied to negative delta in 'asymmetric' mode.
    """
    del prompts
    n = len(completions)
    # TRL 1.0 passes per-sample kwargs as lists; accept both scalar and list
    _step = current_step[0] if isinstance(current_step, list) else current_step
    if _step <= 1 or prev_htmls is None:
        return [0.0] * n

    n, htmls, vr, rm, sids = _prepare_verifier_args(
        completions, verifiable_requirements, reference_metadata, sample_id,
    )
    _ensure_batch_verified(htmls, sids, vr, rm)

    rewards: list[float] = []
    for i in range(n):
        current_scores = _verifier_cache.get(_cache_key(sids[i], htmls[i]), {})
        if prev_scores and i < len(prev_scores):
            prev_overall = prev_scores[i].get("overall_score", 0.0)
        else:
            prev_s = _run_verifier_fast(prev_htmls[i], sids[i], vr[i], rm[i])
            prev_overall = prev_s.get("overall_score", 0.0)
        delta = current_scores.get("overall_score", 0.0) - prev_overall
        rewards.append(float(_apply_progress_mode(delta, progress_mode, progress_neg_scale)))
    return rewards