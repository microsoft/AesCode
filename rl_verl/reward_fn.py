"""Custom reward function for verl's reward manager.

verl's NaiveRewardManager calls compute_score() for each generated response.
This module bridges our AesCode verifier pipeline into verl's reward system.

The function is registered via Hydra config:
    reward.custom_reward_function.path=rl_verl/reward_fn.py
    reward.custom_reward_function.name=compute_score

Function signature (required by verl):
    compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float | dict

When returning a dict, verl's reward manager extracts result["score"] as the
reward and logs all other keys as reward_extra_info (visible in wandb).

Reward strategy:
    GDPO (Group Distributional Policy Optimization) uses per-dimension advantages,
    so each validator's score is passed through independently — no weighted average
    is needed for training. The VALIDATOR_WEIGHTS below are all 1.0 and only used
    for the aggregated "score" field (monitoring/logging), not for GDPO advantage.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

# Limit concurrent Playwright renders to avoid truncated screenshots (libpng errors).
# Each AgentLoopWorker gets its own semaphore; 4 concurrent renders is safe.
_PLAYWRIGHT_CONCURRENCY = int(os.environ.get("PLAYWRIGHT_CONCURRENCY", "4"))
_playwright_sem: asyncio.Semaphore | None = None


def _get_playwright_sem() -> asyncio.Semaphore:
    """Lazy-init semaphore (must be created inside a running event loop)."""
    global _playwright_sem
    if _playwright_sem is None:
        _playwright_sem = asyncio.Semaphore(_PLAYWRIGHT_CONCURRENCY)
    return _playwright_sem

# Ensure repo root is on sys.path so we can import rl_verl.rewards and verifier
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Side-channel for render image paths (cannot go through VERL reward dict)
# ---------------------------------------------------------------------------
# Keyed by sample_id → latest render image path. Read by multi_step_agent.py
# to pass to TrainingMonitor. Cleared when step-scoped tempdirs are GC'd.
_last_render_paths: dict[str, str] = {}

# Render sample saving: batch counter for directory naming
_render_batch_counter: int = 0


def get_last_render_path(sample_id: str) -> str | None:
    """Get the last render image path for a sample (side-channel)."""
    return _last_render_paths.get(sample_id)


def _clear_render_paths():
    """Clear the render path cache."""
    _last_render_paths.clear()


# Age-based reward tempdir GC.
# Tracks (created_at_monotonic, tmpdir) per entry; on every reward call we
# sweep entries older than ``REWARD_TMPDIR_TTL`` (default 600s).  This is
# independent of any step / token plumbing — the caller never has to thread
# a ``global_step`` through, and the leak risk is naturally bounded by the
# TTL regardless of how the trainer is driving us.
_reward_tmpdirs: list[tuple[float, str]] = []
_reward_tmpdirs_lock = threading.Lock()


def _reward_tmpdir_ttl() -> float:
    try:
        return float(os.environ.get("REWARD_TMPDIR_TTL", "600"))
    except (TypeError, ValueError):
        return 600.0


def _register_reward_tmpdir(tmpdir: str) -> None:
    with _reward_tmpdirs_lock:
        _reward_tmpdirs.append((time.monotonic(), tmpdir))


def _gc_stale_reward_tmpdirs() -> None:
    """Delete tmpdirs registered more than TTL seconds ago.

    Cheap to call on every reward path entry — typical list length is
    bounded by (batch_size * TTL / step_time) and we sweep in O(n).
    """
    ttl = _reward_tmpdir_ttl()
    if ttl <= 0:
        return
    cutoff = time.monotonic() - ttl
    stale: list[str] = []
    with _reward_tmpdirs_lock:
        kept: list[tuple[float, str]] = []
        for ts, d in _reward_tmpdirs:
            if ts <= cutoff:
                stale.append(d)
            else:
                kept.append((ts, d))
        _reward_tmpdirs[:] = kept

    if stale:
        for d in stale:
            shutil.rmtree(d, ignore_errors=True)
        # Side-channel render paths in deleted tmpdirs are now dangling;
        # purging the cache is safe because TrainingMonitor only reads
        # the *current* sample's path right after reward computation
        # returns (well within TTL).
        _clear_render_paths()
        logger.debug("reward tmpdir gc: removed %d stale dirs (ttl=%.0fs)",
                     len(stale), ttl)


# ---------------------------------------------------------------------------
# Import reward helpers from the rl_verl reward backend
# ---------------------------------------------------------------------------

from rl_verl.rewards import (
    _extract_html,
    _html_quality,
    _run_verifier_fast,
)

# ---------------------------------------------------------------------------
# Validator weights and short names
# ---------------------------------------------------------------------------

VALIDATOR_WEIGHTS = {
    "ExecutionValidator": 1.0,
    "TextValidator": 1.0,
    "BoundaryValidator": 1.0,
    "SemanticLayoutValidator": 1.0,
    "TableChartValidator": 1.0,
    "WhitespaceValidator": 1.0,
    "DesignValidator": 1.0,           # VLM-based visual design alignment
}

# Which validators are active.  Defaults to ALL validators in VALIDATOR_WEIGHTS.
# Two env vars for control (DISABLED_VALIDATORS is the preferred approach):
#   ENABLED_VALIDATORS  — explicit whitelist (comma-separated full names).
#   DISABLED_VALIDATORS — blacklist subtracted from default full set.
# If both are set, ENABLED_VALIDATORS takes priority and DISABLED_VALIDATORS
# is applied on top of it.
# Example: DISABLED_VALIDATORS="DesignValidator"  (disable just one)
_all_validator_names = list(VALIDATOR_WEIGHTS.keys())
_env_enabled = os.environ.get("ENABLED_VALIDATORS", "")
_env_disabled = os.environ.get("DISABLED_VALIDATORS", "")
ENABLED_VALIDATORS: set[str] = (
    {v.strip() for v in _env_enabled.split(",") if v.strip()}
    if _env_enabled
    else set(_all_validator_names)
)
if _env_disabled:
    _disabled = {v.strip() for v in _env_disabled.split(",") if v.strip()}
    ENABLED_VALIDATORS -= _disabled


def _is_design_validator_enabled() -> bool:
    """Whether DesignValidator is enabled by the unified validator controls."""
    return "DesignValidator" in ENABLED_VALIDATORS


_design_route_logged = False


def _log_design_route_once() -> None:
    """Emit one startup log line clarifying RL design routing semantics."""
    global _design_route_logged
    if _design_route_logged:
        return
    _design_route_logged = True
    design_enabled = _is_design_validator_enabled()
    logger.info(
        "[DesignRoute] design_validator_enabled=%s routing=training_branch_only "
        "verifier_runtime_design=disabled_for_rl",
        "yes" if design_enabled else "no",
    )

# Validators/rewards that may be absent from per_validator dict when not
# evaluated (e.g. TableChartValidator absent for samples without tables,
# DesignValidator absent when VLM API call fails or no render image).
# We use dict presence (not score==0) to decide — a validator that ran but
# scored 0.0 DOES participate in the average.

DEFAULT_WEIGHT = 1.0

_SHORT_NAMES = {
    "ExecutionValidator": "execution",
    "TextValidator": "text",
    "BoundaryValidator": "boundary",
    "SemanticLayoutValidator": "layout",
    "TableChartValidator": "tablechart",
    "WhitespaceValidator": "whitespace",
    "DesignValidator": "design",
}


# ---------------------------------------------------------------------------
# DesignValidator coordinator wiring
# ---------------------------------------------------------------------------
#
# Generate (cross-rollout Swiss)         → rl_verl.design_coordinator.get_generate_coordinator()
# Eval     (single-sample absolute)      → verifier.validators.visual.DesignValidator.validate (unchanged)
#
# Three scoring strategies (switchable via DESIGN_SCORER env var):
#   swiss   — Swiss tournament (group barrier, VLM pairwise)
#   clip    — CLIP cosine similarity (per-rollout)
#   rubric  — VLM rubric Q&A accuracy (per-rollout)

from rl_verl.design_coordinator import (
    DesignResult,
    get_generate_coordinator,
)
from rl_verl.design_scorer import (
    BaseDesignScorer,
    DesignScoreResult,
    SwissDesignScorer,
    get_design_scorer,
)


def _design_result_to_extras(r, mode: str) -> dict:
    """Map a DesignResult into reward_extra_info-friendly fields.

    Returns ONLY numeric fields — verl's reward_manager floats every key
    in reward_extra_info (agent_loop.py:960), so string diagnostics would
    crash training. The string diagnostics (mode / reason_code / fire_reason)
    are emitted via logger.info instead; downstream observability should
    scrape the log line below.
    """
    logger.info(
        "design_diag mode=%s reason=%s fire=%s confidence=%.3f applicable=%d",
        mode, r.reason_code, r.fire_reason, r.confidence, int(r.applicable),
    )
    return {
        "design_confidence": float(r.confidence),
        "design_applicable": float(int(r.applicable)),
    }


# ---------------------------------------------------------------------------
# Blank placeholder render — used when a rollout fails so it can still
# participate in the swiss tournament (as the worst-case baseline) instead
# of being skipped (which would deadlock the N-rollout barrier).
# ---------------------------------------------------------------------------

_BLANK_RENDER_PATH: str | None = None


def _get_blank_render_path() -> str:
    """Return path to a 1024x1024 white PNG, generating it once per process."""
    global _BLANK_RENDER_PATH
    if _BLANK_RENDER_PATH is not None and Path(_BLANK_RENDER_PATH).exists():
        return _BLANK_RENDER_PATH
    import tempfile
    from PIL import Image
    tmp_dir = Path(tempfile.gettempdir()) / "aescode_blank_render"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / "blank_1024.png"
    if not p.exists():
        Image.new("RGB", (1024, 1024), (255, 255, 255)).save(p, "PNG")
    _BLANK_RENDER_PATH = str(p)
    return _BLANK_RENDER_PATH


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_extra_info(extra_info):
    if extra_info is None:
        extra_info = {}
    if isinstance(extra_info, str):
        import json as _json
        try:
            extra_info = _json.loads(extra_info)
        except (_json.JSONDecodeError, TypeError):
            extra_info = {}
    return extra_info


def _parse_json_field(value):
    if isinstance(value, str):
        import json
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [] if value.strip().startswith("[") else {}
    return value


def _maybe_save_html_sample(html: str, sample_id: str, score: float):
    """Optionally save HTML samples for debugging (high/low scoring)."""
    import os
    save_dir = os.environ.get("REWARD_HTML_SAVE_DIR")
    if not save_dir:
        return
    try:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        tag = "good" if score >= 0.7 else "bad" if score < 0.3 else "mid"
        out = save_path / f"{sample_id}_{tag}_{score:.3f}.html"
        out.write_text(html, encoding="utf-8")
    except Exception:
        pass




# ---------------------------------------------------------------------------
# Reward: weighted average of all validators + VLM
# ---------------------------------------------------------------------------

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    """Compute reward score (async).

    Must be async so that verl's NaiveRewardManager (which checks
    inspect.iscoroutinefunction) awaits it directly instead of dispatching
    to a ThreadPoolExecutor.  Running in the same event loop is critical:
    the GenerateSwissCoordinator uses asyncio.Event / asyncio.Lock to
    synchronise across rollouts — these primitives only work within a
    single event loop.  When each rollout ran in its own executor thread
    (with its own asyncio.run → separate loop), Event.set() in one loop
    could never wake Event.wait() in another, causing 100 % SCORE_TIMEOUT.
    """
    return await compute_score_async(data_source, solution_str, ground_truth, extra_info)


def _compute_score_sync(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    """Original synchronous compute_score implementation (fallback)."""
    extra_info = _parse_extra_info(extra_info)
    html = _extract_html(solution_str)

    zero_result = {
        "score": 0.0,
        "execution": 0.0, "text": 0.0, "boundary": 0.0,
        "layout": 0.0, "tablechart": 0.0, "whitespace": 0.0,
        "design": 0.0, "vlm": 0.0, "reward_unweighted": 0.0,
        "tablechart_applicable": 0.0,
    }

    if not html or len(html.strip()) < 10:
        return zero_result

    sample_id = extra_info.get("sample_id", "unknown")
    verifiable_requirements = _parse_json_field(extra_info.get("verifiable_requirements", []))
    reference_metadata = _parse_json_field(extra_info.get("reference_metadata", {}))

    try:
        verifier_result = _run_verifier_fast(
            html=html,
            sample_id=sample_id,
            verifiable_requirements=verifiable_requirements,
            reference_metadata=reference_metadata,
        )
    except Exception:
        logger.exception(
            "[_compute_score_sync] verifier failed for sample %s", sample_id
        )
        raise

    per_validator = dict(verifier_result.get("per_validator", {}))

    # Build result with short names
    result = {"score": 0.0}
    for full_name, short_name in _SHORT_NAMES.items():
        result[short_name] = float(per_validator.get(full_name, 0.0))
    result["tablechart_applicable"] = 1.0 if "TableChartValidator" in per_validator else 0.0

    if per_validator:
        weighted_sum = 0.0
        weight_sum = 0.0
        unweighted_sum = 0.0
        count = 0
        for vname, vscore in per_validator.items():
            w = VALIDATOR_WEIGHTS.get(vname, DEFAULT_WEIGHT)
            weighted_sum += w * vscore
            weight_sum += w
            unweighted_sum += vscore
            count += 1
        reward = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        result["score"] = float(max(0.0, min(1.0, reward)))
        result["reward_unweighted"] = float(unweighted_sum / count) if count > 0 else 0.0
    else:
        fallback = verifier_result.get("overall_score", 0.0)
        result["score"] = float(max(0.0, min(1.0, fallback)))
        result["reward_unweighted"] = result["score"]

    _maybe_save_html_sample(html, sample_id, result["score"])
    return result


async def compute_score_async(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    """Async version of compute_score with parallel verifier + VLM.

    Key optimization: after Playwright renders the HTML (build_artifacts),
    validators and VLM scoring run in parallel via asyncio.gather, since
    both only need the render image — neither depends on the other.

    Timeline:
        build_artifacts (Playwright ~2-3s)
            ├── score_validators (gate + runtime ~0.5-1.5s)  ← parallel
            └── vlm_score_batch  (GPT-5.4 API ~2-5s)         ← parallel
        merge → weighted average → final score
    """
    import asyncio

    extra_info = _parse_extra_info(extra_info)
    html = _extract_html(solution_str)

    zero_result = {
        "score": 0.0,
        "execution": 0.0, "text": 0.0, "boundary": 0.0,
        "layout": 0.0, "tablechart": 0.0, "whitespace": 0.0,
        "design": 0.0, "vlm": 0.0, "reward_unweighted": 0.0,
        "tablechart_applicable": 0.0,
    }

    sample_id = extra_info.get("sample_id", "unknown")
    reference_metadata = _parse_json_field(extra_info.get("reference_metadata", {}))
    ikw = extra_info.get("interaction_kwargs", {})
    if isinstance(ikw, str):
        import json as _j
        try:
            ikw = _j.loads(ikw)
        except Exception:
            ikw = {}

    _log_design_route_once()
    design_enabled = _is_design_validator_enabled()
    # Validation phase: run DesignValidator in eval mode (single-sample absolute
    # 0-5 fidelity → [0,1]) via the verifier pipeline instead of swiss tournament.
    # val uses rollout_n=1 so swiss can never assemble its 8-member group; eval
    # mode gives val reward a real design signal so the metric tracks training.
    _is_val = bool(isinstance(ikw, dict) and ikw.get("_is_val"))

    # GC stale tmpdirs from previous reward calls (age-based, see
    # _gc_stale_reward_tmpdirs).  Cheap; safe to call on every entry.
    _gc_stale_reward_tmpdirs()

    if not html or len(html.strip()) < 10:
        # MODEL FAILURE: empty / malformed output.
        # Generate-only: submit a model-failure marker via the configured scorer.
        logger.warning(
            "[%s] HTML too short / missing — submitting model-failure marker",
            sample_id,
        )
        ref_path = reference_metadata.get("reference_image_path")
        sub_step_idx = ikw.get("sub_step_idx", 0) if isinstance(ikw, dict) else 0
        _gs = ikw.get("global_step") if isinstance(ikw, dict) else None
        global_step = int(_gs) if _gs is not None else None

        if not design_enabled or _is_val:
            # val: model-failure → zero design; no swiss notification needed
            # (val rollouts don't participate in swiss coordinator groups).
            logger.info(
                "design_diag mode=%s reason=%s fire=skipped "
                "confidence=0.000 applicable=0",
                "eval" if _is_val else "generate",
                "val_model_failure" if _is_val else "validator_disabled",
            )
            result = dict(zero_result)
            result["design_confidence"] = 0.0
            result["design_applicable"] = 0.0
            return result

        scorer = get_design_scorer()
        if isinstance(scorer, SwissDesignScorer):
            # Swiss: submit blank-render marker to coordinator
            blank_path = _get_blank_render_path()
            try:
                d_res = await get_generate_coordinator().submit_and_wait(
                    sample_id=sample_id, sub_step_idx=sub_step_idx,
                    ref_path=ref_path, render_path=blank_path,
                    model_failed=True,
                    global_step=global_step,
                )
            except Exception:
                logger.exception("[%s] generate coordinator (model-fail) raised", sample_id)
                d_res = DesignScoreResult(
                    score=0.0, confidence=0.0, applicable=0,
                    reason_code="infra_vlm_error", fire_reason="unknown",
                )
        else:
            # Per-rollout scorer: just return model failure result
            d_res = scorer.model_failure_result()

        result = dict(zero_result)
        if hasattr(d_res, 'applicable') and d_res.applicable:
            result["design"] = float(d_res.score)
            w = VALIDATOR_WEIGHTS.get("DesignValidator", DEFAULT_WEIGHT)
            result["score"] = float(max(0.0, min(1.0, (w * d_res.score) / max(w, 1e-9))))
            result["reward_unweighted"] = float(d_res.score)
        result.update(_design_result_to_extras(d_res, "generate"))
        return result

    verifiable_requirements = _parse_json_field(extra_info.get("verifiable_requirements", []))
    # reference_metadata + ikw already parsed above (shared with the early-return branch)

    # ── Parse routing info ──
    sub_step_idx = ikw.get("sub_step_idx", 0) if isinstance(ikw, dict) else 0
    ref_path = reference_metadata.get("reference_image_path")
    # global_step for deterministic coordinator group key (fixes epoch-split bug)
    _gs = ikw.get("global_step") if isinstance(ikw, dict) else None
    global_step = int(_gs) if _gs is not None else None

    # Try the split path: build artifacts first, then validators + design in parallel
    design_result = None
    generate_submit_task: asyncio.Task | None = None
    generate_submit_started = False
    # Pre-bind so the except handler can reference them even if exception
    # fires before they're set in the try block.
    scorer: BaseDesignScorer | None = None
    scorer_is_swiss = False
    try:
        from verifier.pipeline.parser import _parse_sample
        from verifier.pipeline.runner import build_artifacts_async, score_validators_async
        import tempfile

        payload = {
            "id": sample_id,
            "reference_metadata": reference_metadata,
            "generation_prompt": "",
            "verifiable_requirements": verifiable_requirements,
        }
        sample = _parse_sample(payload)

        tmpdir = tempfile.mkdtemp(prefix=f"aescode_reward_{sample_id}_")
        _register_reward_tmpdir(tmpdir)
        tmp_path = Path(tmpdir)
        html_file = tmp_path / f"{sample_id}.html"
        html_file.write_text(html, encoding="utf-8")

        # Phase 1: Build artifacts (Playwright render) — slow I/O.
        artifacts = await build_artifacts_async(
            sample, html, html_file, artifact_root=tmp_path / "artifacts",
        )
        render_image_path = getattr(artifacts, "render_image_path", None)
        if render_image_path:
            _last_render_paths[sample_id] = str(render_image_path)
        render_str = str(render_image_path) if render_image_path else None

        # Phase 2: Non-VLM validators ‖ Design scoring.
        #
        # Routing:
        #   SwissDesignScorer:
        #     train → SwissCoordinator (cross-rollout group barrier)
        #     val   → score_validators_async include_design_validator=True
        #             (eval mode, single-sample 0-5 fidelity via visual.py)
        #             because swiss group barrier can't assemble during val.
        #   RubricDesignScorer / CLIPDesignScorer:
        #     train → per-rollout scorer
        #     val   → per-rollout scorer (SAME scorer, same scale).
        #             Critical: when DESIGN_SCORER=rubric, val MUST also call
        #             rubric, not the visual.py eval mode — otherwise train
        #             reports rubric accuracy in [0,1] while val reports
        #             absolute fidelity in [0,1] and they are not comparable.
        scorer = get_design_scorer() if design_enabled else None
        scorer_is_swiss = isinstance(scorer, SwissDesignScorer) if scorer else False
        # Only include design inside the verifier pipeline for the legacy
        # swiss+val path; per-rollout scorers handle val themselves below.
        include_design_in_pipeline = bool(
            _is_val and design_enabled and scorer_is_swiss
        )
        validator_task = score_validators_async(
            sample,
            artifacts,
            include_design_validator=include_design_in_pipeline,
        )

        # Run the per-rollout scorer for BOTH train and val (when scorer is
        # rubric/clip). Only skip when (a) design fully disabled, or
        # (b) swiss scorer in val (handled by verifier pipeline above).
        run_per_rollout_scorer = bool(
            design_enabled and scorer is not None and not scorer_is_swiss
        )

        if design_enabled and not _is_val and scorer_is_swiss:
            # Swiss tournament — uses group barrier via coordinator (train only)
            async def _submit_generate_member() -> DesignResult:
                nonlocal generate_submit_started
                generate_submit_started = True
                return await get_generate_coordinator().submit_and_wait(
                    sample_id=sample_id,
                    sub_step_idx=int(sub_step_idx),
                    ref_path=ref_path,
                    render_path=render_str,
                    model_failed=False,
                    global_step=global_step,
                )

            generate_submit_task = asyncio.create_task(_submit_generate_member())
            design_task = generate_submit_task

            verifier_result, design_result = await asyncio.gather(
                validator_task, design_task,
            )
        elif run_per_rollout_scorer:
            # Per-rollout scorer (CLIP / Rubric) — no group barrier.
            # Runs for both train AND val so design metric is on the same scale.
            design_extra = dict(extra_info) if extra_info else {}
            design_extra["interaction_kwargs"] = ikw

            async def _score_per_rollout() -> DesignScoreResult:
                return await scorer.score(
                    sample_id=sample_id,
                    render_path=render_str,
                    ref_path=ref_path,
                    extra_info=design_extra,
                )

            verifier_result, design_result = await asyncio.gather(
                validator_task, _score_per_rollout(),
            )
        else:
            # Design disabled, OR swiss-in-val (handled inside verifier pipeline)
            verifier_result = await validator_task
            design_result = None

    except Exception:
        logger.exception("Split-async reward failed for %s", sample_id)
        # Do NOT raise here: upstream may convert this rollout to reward=0
        # without notifying GenerateSwissCoordinator, which leaves peers
        # waiting for a missing member and causes timeout cascades.
        verifier_result = {"per_validator": {}, "overall_score": 0.0}

        if design_enabled and scorer_is_swiss and not _is_val:
            # Swiss train: must still notify coordinator even on failure
            if generate_submit_task is not None:
                try:
                    design_result = await generate_submit_task
                except Exception:
                    logger.exception("[%s] in-flight coordinator submit failed", sample_id)
                    design_result = None

            if design_result is None and not generate_submit_started:
                try:
                    design_result = await get_generate_coordinator().submit_and_wait(
                        sample_id=sample_id,
                        sub_step_idx=int(sub_step_idx),
                        ref_path=ref_path,
                        render_path=None,
                        model_failed=False,
                        global_step=global_step,
                    )
                except Exception:
                    logger.exception("[%s] coordinator submit on artifact_error failed", sample_id)
                    design_result = None

            if design_result is None:
                design_result = DesignScoreResult(
                    score=0.0, confidence=0.0, applicable=0,
                    reason_code="infra_artifact_error", fire_reason="artifact_error",
                )
        elif design_enabled and scorer is not None and not scorer_is_swiss:
            # Per-rollout scorer (CLIP/Rubric) — train or val: return infra failure
            design_result = scorer.infra_failure_result("infra_artifact_error")
        else:
            design_result = None

    per_validator = dict(verifier_result.get("per_validator", {}))

    # Inject design score from scorer.
    # design_result can be DesignResult (Swiss) or DesignScoreResult (CLIP/Rubric)
    if design_result and design_result.applicable:
        per_validator["DesignValidator"] = float(design_result.score)

    # DEBUG (2026-05-29): dump full per_validator + verifier_result keys so we
    # can see which validators actually returned which score for every sample.
    # Triggered for ALL samples — remove once root cause is identified.
    try:
        _vr_keys = sorted(verifier_result.keys()) if isinstance(verifier_result, dict) else []
        _overall = verifier_result.get("overall_score") if isinstance(verifier_result, dict) else None
        _html_len = len(html) if isinstance(html, str) else -1
        logger.warning(
            "REWARD_DEBUG sample=%s html_len=%d verifier_keys=%s overall=%s per_validator=%s design_app=%s design_score=%s",
            sample_id, _html_len, _vr_keys, _overall, per_validator,
            getattr(design_result, "applicable", None) if design_result else None,
            getattr(design_result, "score", None) if design_result else None,
        )
    except Exception:
        pass

    # Build result with short names
    result = {"score": 0.0}
    for full_name, short_name in _SHORT_NAMES.items():
        result[short_name] = float(per_validator.get(full_name, 0.0))
    result["tablechart_applicable"] = 1.0 if "TableChartValidator" in per_validator else 0.0

    if per_validator:
        weighted_sum = 0.0
        weight_sum = 0.0
        unweighted_sum = 0.0
        count = 0
        for vname, vscore in per_validator.items():
            w = VALIDATOR_WEIGHTS.get(vname, DEFAULT_WEIGHT)
            weighted_sum += w * vscore
            weight_sum += w
            unweighted_sum += vscore
            count += 1
        reward = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        result["score"] = float(max(0.0, min(1.0, reward)))
        result["reward_unweighted"] = float(unweighted_sum / count) if count > 0 else 0.0
    else:
        fallback = verifier_result.get("overall_score", 0.0)
        result["score"] = float(max(0.0, min(1.0, fallback)))
        result["reward_unweighted"] = result["score"]

    _maybe_save_html_sample(html, sample_id, result["score"])
    # Attach design diagnostic fields (mode / reason_code / fire_reason /
    # confidence / applicable) — these flow into reward_extra_info → wandb
    # and are the primary observability signal for the DesignValidator
    # pipeline.  Fall back to a neutral OK record if the design scorer
    # produced nothing (e.g. ENABLED_VALIDATORS excluded it).
    if design_result is not None:
        result.update(_design_result_to_extras(design_result, "generate"))
    elif _is_val and "DesignValidator" in per_validator:
        # val eval mode: design ran inside the pipeline (visual.py:_validate_eval).
        # Emit log diagnostic so wandb / watchdog see the eval signal; only
        # numeric fields go into result[] (verl floats reward_extra_info).
        _design_score = float(per_validator.get("DesignValidator", 0.0))
        logger.info(
            "design_diag mode=eval reason=val_pipeline fire=ok "
            "confidence=1.000 applicable=1 score=%.3f", _design_score,
        )
        result["design_confidence"] = 1.0
        result["design_applicable"] = 1.0
    else:
        logger.info(
            "design_diag mode=generate reason=validator_disabled fire=skipped "
            "confidence=0.000 applicable=0",
        )
        result["design_confidence"] = 0.0
        result["design_applicable"] = 0.0
    return result




def compute_score_detailed(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    """Extended reward function that returns per-dimension scores.

    This version returns a dict with detailed score breakdown, useful for
    GDPO-style multi-objective advantage computation if verl supports it.

    Returns:
        dict: {
            "score": float,  # Combined reward (same as compute_score)
            "extra_info": {
                "execution_score": float,
                "overall_score": float,
                "per_validator": dict[str, float],
                "html_quality": float,  # Heuristic quality for comparison
            }
        }
    """
    if extra_info is None:
        extra_info = {}

    # extra_info may be a JSON string (from parquet serialization)
    if isinstance(extra_info, str):
        import json as _json
        try:
            extra_info = _json.loads(extra_info)
        except (_json.JSONDecodeError, TypeError):
            extra_info = {}

    html = _extract_html(solution_str)

    if not html or len(html.strip()) < 10:
        return {
            "score": 0.0,
            "extra_info": {
                "execution_score": 0.0,
                "overall_score": 0.0,
                "per_validator": {},
                "html_quality": 0.0,
            },
        }

    sample_id = extra_info.get("sample_id", "unknown")
    verifiable_requirements = extra_info.get("verifiable_requirements", [])
    reference_metadata = extra_info.get("reference_metadata", {})

    if isinstance(verifiable_requirements, str):
        import json
        try:
            verifiable_requirements = json.loads(verifiable_requirements)
        except (json.JSONDecodeError, TypeError):
            verifiable_requirements = []

    if isinstance(reference_metadata, str):
        import json
        try:
            reference_metadata = json.loads(reference_metadata)
        except (json.JSONDecodeError, TypeError):
            reference_metadata = {}

    try:
        verifier_result = _run_verifier_fast(
            html=html,
            sample_id=sample_id,
            verifiable_requirements=verifiable_requirements,
            reference_metadata=reference_metadata,
        )
    except Exception as exc:
        logger.warning("Verifier failed for %s: %s", sample_id, exc)
        q = _html_quality(html)
        return {
            "score": q,
            "extra_info": {
                "execution_score": 0.0,
                "overall_score": q,
                "per_validator": {},
                "html_quality": q,
            },
        }

    per_validator = verifier_result.get("per_validator", {})
    html_q = _html_quality(html)

    if per_validator:
        weighted_sum = sum(VALIDATOR_WEIGHTS.get(k, DEFAULT_WEIGHT) * v for k, v in per_validator.items())
        weight_sum = sum(VALIDATOR_WEIGHTS.get(k, DEFAULT_WEIGHT) for k in per_validator)
        reward = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    else:
        reward = verifier_result.get("overall_score", 0.0)

    return {
        "score": float(max(0.0, min(1.0, reward))),
        "extra_info": {
            "execution_score": float(per_validator.get("ExecutionValidator", 0.0)),
            "overall_score": float(verifier_result.get("overall_score", 0.0)),
            "per_validator": {k: float(v) for k, v in per_validator.items()},
            "html_quality": float(html_q),
        },
    }
