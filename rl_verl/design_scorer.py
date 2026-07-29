"""Design scorer abstraction — Swiss / CLIP / Rubric.

Three design scoring strategies for RL training, switchable via
the DESIGN_SCORER environment variable (default: "swiss").

  swiss   — Cross-rollout Swiss tournament (group barrier, VLM pairwise)
  clip    — Per-rollout CLIP cosine similarity (render vs reference image)
  rubric  — Per-rollout VLM rubric Q&A (render image + rubric questions)

Swiss requires all rollouts in a group to be present before scoring.
CLIP and Rubric are per-rollout and score independently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass (unified across all scorers)
# ---------------------------------------------------------------------------
@dataclass
class DesignScoreResult:
    """Unified result from any design scorer."""
    score: float = 0.5            # [0, 1]
    confidence: float = 1.0       # 1.0 for per-rollout scorers
    applicable: int = 1           # 0 → mask in training (zero advantage)
    reason_code: str = "ok"
    fire_reason: str = "per_rollout"
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseDesignScorer(ABC):
    """Base class for design scorers."""

    @abstractmethod
    async def score(
        self,
        sample_id: str,
        render_path: str | None,
        ref_path: str | None,
        extra_info: dict | None = None,
    ) -> DesignScoreResult:
        """Score a single rollout's rendered output.

        Args:
            sample_id: Unique sample identifier.
            render_path: Path to rendered HTML screenshot (may be None on failure).
            ref_path: Path to reference image.
            extra_info: Additional info (interaction_kwargs, rubric_questions, etc.)

        Returns:
            DesignScoreResult with score in [0, 1].
        """
        ...

    def model_failure_result(self) -> DesignScoreResult:
        """Default result for model failures (empty HTML)."""
        return DesignScoreResult(
            score=0.0, confidence=1.0, applicable=1,
            reason_code="model_empty_html", fire_reason="model_failure",
        )

    def infra_failure_result(self, reason: str = "infra_error") -> DesignScoreResult:
        """Default result for infrastructure failures."""
        return DesignScoreResult(
            score=0.5, confidence=0.0, applicable=0,
            reason_code=reason, fire_reason="infra_failure",
        )


# ---------------------------------------------------------------------------
# 1. Swiss tournament scorer (wraps existing design_coordinator)
# ---------------------------------------------------------------------------
class SwissDesignScorer(BaseDesignScorer):
    """Cross-rollout Swiss tournament via GenerateSwissCoordinator.

    This is the only scorer that requires a group barrier — all rollouts
    in a group must be submitted before scores are returned.
    """

    async def score(
        self, sample_id: str, render_path: str | None,
        ref_path: str | None, extra_info: dict | None = None,
    ) -> DesignScoreResult:
        from rl_verl.design_coordinator import (
            DesignResult,
            DesignReason,
            get_generate_coordinator,
        )
        extra_info = extra_info or {}
        ikw = extra_info.get("interaction_kwargs", {})
        sub_step_idx = ikw.get("sub_step_idx", 0)
        _gs = ikw.get("global_step")
        global_step = int(_gs) if _gs is not None else None
        model_failed = render_path is None

        try:
            d_res: DesignResult = await get_generate_coordinator().submit_and_wait(
                sample_id=sample_id,
                sub_step_idx=int(sub_step_idx),
                ref_path=ref_path,
                render_path=render_path,
                model_failed=model_failed,
                global_step=global_step,
            )
        except Exception as exc:
            logger.exception("[%s] Swiss coordinator raised", sample_id)
            return self.infra_failure_result("infra_vlm_error")

        return DesignScoreResult(
            score=float(d_res.score),
            confidence=float(d_res.confidence),
            applicable=int(d_res.applicable),
            reason_code=d_res.reason_code,
            fire_reason=d_res.fire_reason,
        )

    @property
    def needs_group_barrier(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 2. CLIP scorer (per-rollout, no group barrier)
# ---------------------------------------------------------------------------

# Lazy-loaded CLIP model singleton
_clip_model = None
_clip_preprocess = None
_clip_device = None


def _load_clip_model(model_name: str = "ViT-L/14", device: str | None = None):
    """Load CLIP model (lazy singleton)."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_device

    import torch
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP scorer requires the `clip` package. "
            "Install with: pip install git+https://github.com/openai/CLIP.git"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading CLIP model %s on %s...", model_name, device)
    _clip_model, _clip_preprocess = clip.load(model_name, device=device)
    _clip_device = device
    logger.info("CLIP model loaded.")
    return _clip_model, _clip_preprocess, _clip_device


class CLIPDesignScorer(BaseDesignScorer):
    """Per-rollout CLIP similarity between render screenshot and reference image.

    Supports two backends:
      - local:     OpenAI CLIP ViT-L/14 (or SigLIP) running locally
      - azure_api: Azure OpenAI embedding endpoint (no local GPU needed)

    Config via env vars:
      CLIP_BACKEND:  local | azure_api (default: local)
      CLIP_MODEL:    model name for local backend (default: ViT-L/14)
      CLIP_DEVICE:   cuda | cpu (default: auto)
      CLIP_MIN_SIM:  minimum similarity for rescaling (default: 0.5)
      CLIP_MAX_SIM:  maximum similarity for rescaling (default: 1.0)
    """

    def __init__(self):
        self.backend = os.environ.get("CLIP_BACKEND", "local")
        self.model_name = os.environ.get("CLIP_MODEL", "ViT-L/14")
        self.device = os.environ.get("CLIP_DEVICE", None)
        self.min_sim = float(os.environ.get("CLIP_MIN_SIM", "0.5"))
        self.max_sim = float(os.environ.get("CLIP_MAX_SIM", "1.0"))

    async def score(
        self, sample_id: str, render_path: str | None,
        ref_path: str | None, extra_info: dict | None = None,
    ) -> DesignScoreResult:
        if not render_path or not Path(render_path).exists():
            return self.infra_failure_result("infra_render_missing")
        if not ref_path or not Path(ref_path).exists():
            return self.infra_failure_result("infra_ref_missing")

        try:
            if self.backend == "azure_api":
                similarity = await self._score_azure_api(render_path, ref_path)
            else:
                similarity = await self._score_local(render_path, ref_path)

            # Rescale from [min_sim, max_sim] → [0, 1]
            score = max(0.0, min(1.0,
                (similarity - self.min_sim) / (self.max_sim - self.min_sim)
            ))

            return DesignScoreResult(
                score=score, confidence=1.0, applicable=1,
                reason_code="ok", fire_reason="per_rollout",
                details={"raw_similarity": similarity, "backend": self.backend},
            )
        except Exception as exc:
            logger.warning("[%s] CLIP scoring failed: %s", sample_id, exc)
            return self.infra_failure_result("infra_clip_error")

    async def _score_local(self, render_path: str, ref_path: str) -> float:
        """Compute CLIP cosine similarity using local model."""
        import torch
        from PIL import Image as PILImage

        model, preprocess, device = _load_clip_model(self.model_name, self.device)

        loop = asyncio.get_running_loop()

        def _compute():
            render_img = preprocess(PILImage.open(render_path).convert("RGB")).unsqueeze(0).to(device)
            ref_img = preprocess(PILImage.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                render_feat = model.encode_image(render_img)
                ref_feat = model.encode_image(ref_img)
                render_feat = render_feat / render_feat.norm(dim=-1, keepdim=True)
                ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
                similarity = (render_feat @ ref_feat.T).item()
            return float(similarity)

        return await loop.run_in_executor(None, _compute)

    async def _score_azure_api(self, render_path: str, ref_path: str) -> float:
        """Compute CLIP similarity via Azure OpenAI embedding API."""
        import base64
        from PIL import Image as PILImage
        import io

        # Get Azure client
        from rl_verl.vlm_reward import _get_shared_client, _ensure_dotenv
        _ensure_dotenv()

        client = await _get_shared_client(timeout=30.0)

        async def _get_embedding(image_path: str) -> list[float]:
            img = PILImage.open(image_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = await client.embeddings.create(
                model=os.environ.get("CLIP_API_MODEL", "clip"),
                input=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
            )
            return response.data[0].embedding

        render_emb, ref_emb = await asyncio.gather(
            _get_embedding(render_path),
            _get_embedding(ref_path),
        )

        # Cosine similarity
        import math
        dot = sum(a * b for a, b in zip(render_emb, ref_emb))
        norm_r = math.sqrt(sum(a * a for a in render_emb))
        norm_f = math.sqrt(sum(a * a for a in ref_emb))
        similarity = dot / (norm_r * norm_f) if norm_r > 0 and norm_f > 0 else 0.0
        return float(similarity)

    @property
    def needs_group_barrier(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 3. Rubric scorer (per-rollout, VLM Q&A)
# ---------------------------------------------------------------------------

_RUBRIC_BATCH_PROMPT_HEADER = """You are a strict but fair visual judge.

You will see a SINGLE GENERATED IMAGE and a numbered list of yes/no questions about it.

Rules:
- Answer ONLY based on what is actually visible in the generated image.
- Judge each question independently.
- Gray placeholder boxes for image regions are expected; do not penalize missing photo content.
- Do not verify exact text wording; use text only as a location anchor.

Output STRICT JSON (no markdown fences):
{"judgments": [{"id": "q1", "answer": "yes"}, {"id": "q2", "answer": "no"}]}
"""


def _build_rubric_batch_prompt(questions: list[tuple[str, str]]) -> str:
    """Build one strict-JSON prompt for all rubric questions.

    Args:
        questions: list of (question_id, question_text)
    """
    q_block = "\n".join(f"  {qid}. {qtext}" for qid, qtext in questions)
    return (
        f"{_RUBRIC_BATCH_PROMPT_HEADER}\n\n"
        f"Judge the following generated image against {len(questions)} yes/no questions.\n\n"
        f"Questions:\n{q_block}\n\n"
        "For each question, answer yes / no / unclear based ONLY on what is visible in the image.\n"
        "Return strict JSON only."
    )


def _parse_loose_json(raw_text: str) -> dict:
    """Parse JSON output that may be wrapped in markdown fences."""
    s = (raw_text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        if s.startswith("json"):
            s = s.split("\n", 1)[1] if "\n" in s else s
    return json.loads(s)


def _normalize_ynu(value: str) -> str | None:
    v = str(value or "").strip().lower()
    if v in {"yes", "no", "unclear"}:
        return v
    return None


def _parse_batch_answers(raw_text: str, valid_ids: set[str]) -> dict[str, str]:
    """Parse strict JSON judgments into {qid: yes|no|unclear}."""
    out: dict[str, str] = {}
    if not raw_text:
        return out
    try:
        parsed = _parse_loose_json(raw_text)
    except Exception:
        return out
    judgments = parsed.get("judgments", []) if isinstance(parsed, dict) else []
    if not isinstance(judgments, list):
        return out
    for item in judgments:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip()
        ans = _normalize_ynu(item.get("answer", ""))
        if qid in valid_ids and ans is not None:
            out[qid] = ans
    return out


# ---------------------------------------------------------------------------
# Process-wide in-flight concurrency cap for rubric VLM calls.
#
# ROOT CAUSE (2026-06-16 training hang, GPU->0% from step ~37):
#   verl's AgentLoopWorker dispatches every rollout of a batch concurrently
#   via asyncio.gather, so an entire step's worth of RubricDesignScorer.score()
#   coroutines (G*N rollouts, often 100+) await the VLM AT ONCE on the SAME
#   event loop that also drives vLLM's AsyncMPClient zmq poll. The rubric path
#   -- unlike Swiss / refine / eval -- had NO concurrency Semaphore, only an
#   AsyncLimiter (a requests-per-MINUTE rate cap, which does NOT bound in-flight
#   concurrency). Under that unbounded fan-out the event loop saturates (sync
#   per-call image encode + the occasional blocking AzureCliCredential token
#   refresh that forks `az account get-access-token`), vLLM's zmq poll is
#   starved, GPU drops to 0%, the step hangs forever -- and the
#   asyncio.wait_for HARD-CAP, scheduled on that same frozen loop, never fires.
#
#   Swiss never hit this: its group barrier + Semaphore keep the number of
#   simultaneously-inflight VLM coroutines low. This Semaphore restores the
#   rubric path to that same, already-proven-safe concurrency regime. It is
#   shared process-wide (lazy singleton) so the cap is real -- a fresh sem per
#   call would leave concurrency effectively unbounded.
# ---------------------------------------------------------------------------
_rubric_inflight_sem: Optional[asyncio.Semaphore] = None


def _get_rubric_inflight_sem() -> asyncio.Semaphore:
    """Lazily build the shared rubric in-flight Semaphore.

    Cap = RUBRIC_VLM_MAX_INFLIGHT, defaulting to VLM_REWARD_CONCURRENCY (the
    same budget Swiss/eval/refine use), falling back to 32. Built lazily so it
    binds to the worker's running event loop.
    """
    global _rubric_inflight_sem
    if _rubric_inflight_sem is None:
        cap = int(os.environ.get(
            "RUBRIC_VLM_MAX_INFLIGHT",
            os.environ.get("VLM_REWARD_CONCURRENCY", "32"),
        ))
        _rubric_inflight_sem = asyncio.Semaphore(max(1, cap))
    return _rubric_inflight_sem


class RubricDesignScorer(BaseDesignScorer):
    """Per-rollout VLM rubric scorer.

    Each sample has a list of rubric questions. We send the rendered
    screenshot + ALL questions to the VLM in a single call (with short
    Observation+Answer CoT per question). Score = correct / valid.

        Rubric questions are passed via extra_info["rubric_questions"]:
            [{"id": "q1", "question": "...", "pass_answer": "yes"}, ...]

        Backward-compatible legacy format is still accepted:
            [{"question": "...", "expected": "yes"}, ...]

        Behaviour:
            - 1 batched call answers all N questions.
            - Per call: exponential backoff with up to RUBRIC_VLM_MAX_RETRIES
                attempts (default 3 = total 4 tries) on 429/500/timeout/conn.
            - Any questions whose answer can't be parsed after the first batched
                call get a single follow-up retry call (also with backoff).
            - Questions still missing are counted as incorrect (same denominator),
                matching v5.2 release semantics: reward = correct / total_questions.
            - **Hard outer cap** of RUBRIC_VLM_HARD_CAP seconds on score() —
                prevents a single sample from hanging the whole reward gather and
                therefore the entire training step. If exceeded, applicable=0 and
                the sample is silently dropped.

    Config via env vars:
      RUBRIC_VLM_TIMEOUT:        per-call timeout seconds (default: 90)
      RUBRIC_VLM_MAX_TOKENS:     max_completion_tokens per call (default: 250)
      RUBRIC_VLM_MAX_RETRIES:    additional retries on 429/500/timeout (default: 3)
      RUBRIC_VLM_BACKOFF_BASE:   initial backoff seconds (default: 1.0)
      RUBRIC_VLM_HARD_CAP:       absolute cap for whole score() call (default: 300)
      RUBRIC_VLM_CONCURRENCY:    legacy, unused (kept for backwards compat)
    """

    def __init__(self):
        # 90s gives headroom over the underlying client's own retries+timeouts.
        # max_tokens 250 = 5 Qs * (~10w obs + "Answer N: yes" + slack) ≈ 200,
        # +50 buffer. Way down from 800 — main cause of APITimeoutError under load.
        self.timeout = float(os.environ.get("RUBRIC_VLM_TIMEOUT", "90"))
        self.max_tokens = int(os.environ.get("RUBRIC_VLM_MAX_TOKENS", "250"))
        self.max_retries = int(os.environ.get("RUBRIC_VLM_MAX_RETRIES", "3"))
        self.backoff_base = float(os.environ.get("RUBRIC_VLM_BACKOFF_BASE", "1.0"))
        # Outer hard cap: even if every retry hits its own 90s timeout, we
        # bound the entire score() call at 300s so we can't hang the
        # asyncio.gather() in reward_fn that has NO outer timeout itself.
        self.hard_cap = float(os.environ.get("RUBRIC_VLM_HARD_CAP", "300"))
        # Reasoning-style models (gpt-5.x family deployed in Azure) only
        # accept the default temperature (1.0) and reject temperature=0.0
        # with HTTP 400 unsupported_value. Set to empty / "default" to skip
        # the temperature arg entirely. For older models set to "0.0" or
        # similar via env.
        _t = os.environ.get("RUBRIC_VLM_TEMPERATURE", "").strip()
        if _t == "" or _t.lower() == "default":
            self.temperature: float | None = None
        else:
            self.temperature = float(_t)
        # Legacy, kept for env-var compat with skill docs
        self.concurrency = int(os.environ.get("RUBRIC_VLM_CONCURRENCY", "1"))

    async def score(
        self, sample_id: str, render_path: str | None,
        ref_path: str | None, extra_info: dict | None = None,
    ) -> DesignScoreResult:
        # ── In-flight concurrency cap (ROOT-CAUSE fix for 2026-06-16 hang) ──
        # Bound how many rubric VLM coroutines are simultaneously awaiting on
        # this worker's event loop. Without this, an entire step's rollouts
        # (B*N, 60-128 typical) fan out at once via asyncio.gather and
        # saturate the loop — PIL+JPEG+base64 of 60-128 1024px screenshots
        # serialised on the same loop that drives vLLM's zmq poll easily
        # adds tens of seconds of pure loop block, starving the engine
        # control channel → GPU 0% → permanent hang. Swiss never hit this:
        # its barrier serialises samples and its inner Semaphore caps inflight
        # pairs to ~32, restoring the same regime here.
        #
        # Use ``async with sem`` (not manual acquire/release) so the permit
        # is never leaked on cancellation — important because the outer
        # REWARD_TIMEOUT (600s in run_gdpo.sh) cancels this coroutine if the
        # whole reward computation runs over budget; manual
        # ``await asyncio.wait_for(sem.acquire(), ...)`` patterns can leak
        # a permit when the timeout fires exactly as acquire completes
        # (bpo-42130), draining the cap to 0 over hours and silently
        # permanently hanging every subsequent rubric call.
        #
        # No timeout on the acquire itself: REWARD_TIMEOUT is the global cap
        # already; stacking another timer here is just more code on the
        # same potentially-busy loop, and ``async with`` on a Semaphore is
        # fully cancellation-safe in modern Python.
        sem = _get_rubric_inflight_sem()
        async with sem:
            return await self._score_guarded(
                sample_id, render_path, ref_path, extra_info,
            )

    async def _score_guarded(
        self, sample_id: str, render_path: str | None,
        ref_path: str | None, extra_info: dict | None = None,
    ) -> DesignScoreResult:
        # OUTER HARD CAP — never let one sample hang the whole step.
        # reward_fn.py does `asyncio.gather(validator_task, scorer.score(...))`
        # without an outer wait_for; if our per-call timeout chain ever fails
        # us (SDK internal retry loops, AsyncLimiter starvation, network
        # half-close that the SDK can't detect, ...), this cap is the last
        # line of defence. Beyond it: applicable=0, training continues.
        try:
            return await asyncio.wait_for(
                self._score_inner(sample_id, render_path, ref_path, extra_info),
                timeout=self.hard_cap,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] rubric scorer HARD-CAPPED at %.0fs -> applicable=0",
                sample_id, self.hard_cap,
            )
            return DesignScoreResult(
                score=0.0, confidence=0.0, applicable=0,
                reason_code="rubric_hard_cap_timeout", fire_reason="per_rollout",
            )
        except Exception as exc:
            # Last-resort catch-all: never propagate from a scorer; better to
            # be applicable=0 than to crash the entire training step.
            logger.warning(
                "[%s] rubric scorer unhandled %s -> applicable=0: %s",
                sample_id, type(exc).__name__, str(exc)[:200],
            )
            return DesignScoreResult(
                score=0.0, confidence=0.0, applicable=0,
                reason_code="rubric_unhandled_error", fire_reason="per_rollout",
            )

    async def _score_inner(
        self, sample_id: str, render_path: str | None,
        ref_path: str | None, extra_info: dict | None = None,
    ) -> DesignScoreResult:
        if not render_path or not Path(render_path).exists():
            return self.infra_failure_result("infra_render_missing")

        extra_info = extra_info or {}
        rubric_questions = extra_info.get("rubric_questions", [])
        if isinstance(rubric_questions, str):
            try:
                rubric_questions = json.loads(rubric_questions)
            except (json.JSONDecodeError, TypeError):
                rubric_questions = []

        # Fallback: some pipelines fold rubric_questions under interaction_kwargs
        # instead of top-level extra_info. Honor that path so we don't silently
        # treat the sample as unscoreable.
        if not rubric_questions:
            ikw = extra_info.get("interaction_kwargs") or {}
            if isinstance(ikw, str):
                try:
                    ikw = json.loads(ikw)
                except (json.JSONDecodeError, TypeError):
                    ikw = {}
            rubric_questions = ikw.get("rubric_questions") or [] if isinstance(ikw, dict) else []
            if isinstance(rubric_questions, str):
                try:
                    rubric_questions = json.loads(rubric_questions)
                except (json.JSONDecodeError, TypeError):
                    rubric_questions = []

        if not rubric_questions:
            # Log the keys we received so it's debuggable when this fires
            # in production. Keep at WARNING since each rollout hits this
            # once if data is missing — it's signal, not noise.
            try:
                _keys = sorted(extra_info.keys())[:25] if isinstance(extra_info, dict) else type(extra_info).__name__
            except Exception:
                _keys = "<unhashable>"
            logger.warning(
                "[%s] No rubric questions found -> applicable=0  extra_info_keys=%s",
                sample_id, _keys,
            )
            return DesignScoreResult(
                score=0.0, confidence=0.0, applicable=0,
                reason_code="no_rubric_questions", fire_reason="per_rollout",
            )

        try:
            correct, total, details = await self._evaluate_rubric_batched(
                render_path, rubric_questions, sample_id
            )
        except Exception as exc:
            logger.warning("[%s] Rubric scoring failed: %s", sample_id, exc)
            return self.infra_failure_result("infra_rubric_error")

        score = correct / total if total > 0 else 0.0  # normalized to [0, 1]
        return DesignScoreResult(
            score=score, confidence=1.0, applicable=1,
            reason_code="ok", fire_reason="per_rollout",
            details={
                "correct": correct,
                "total": total,
                "asked": len(rubric_questions),
                "per_question": details,
            },
        )

    async def _evaluate_rubric_batched(
        self,
        render_path: str,
        rubric_questions: list[dict],
        sample_id: str,
    ) -> tuple[int, int, list[dict]]:
        """Single batched VLM call (+1 retry for missing answers).

        Returns (correct_count, total_count, per_question_details).
            total_count always equals number of rubric questions. Missing/
            unparseable answers are counted as incorrect.
        """
        from rl_verl.vlm_reward import _get_shared_client, _ensure_dotenv, _img_part, _get_model
        _ensure_dotenv()

        # _get_shared_client returns (client, limiter): the limiter is an
        # AsyncLimiter scoped to one endpoint and MUST wrap every chat call
        # so multi-process rate limiting applies. Round-robin between
        # endpoints happens here on every call.
        client, limiter = await _get_shared_client(timeout=self.timeout)
        model = _get_model()

        normalized_questions: list[dict] = []
        for i, q in enumerate(rubric_questions, start=1):
            qid = str(q.get("id") or f"q{i}").strip()
            qtext = str(q.get("question") or "").strip()
            exp = str(q.get("pass_answer", q.get("expected", "yes"))).strip().lower()
            if exp not in {"yes", "no"}:
                exp = "yes"
            normalized_questions.append({"id": qid, "question": qtext, "expected": exp})

        n = len(normalized_questions)
        question_pairs = [(q["id"], q["question"]) for q in normalized_questions]

        # ── First pass: ask all N at once ──
        first_raw = await self._call_vlm_batch(
            client, limiter, model, render_path, question_pairs, sample_id, attempt=1,
        )
        valid_ids = {q["id"] for q in normalized_questions}
        parsed = _parse_batch_answers(first_raw, valid_ids)

        # ── Retry pass: only the unparsed indices ──
        # Re-acquire client to round-robin to (possibly) the other endpoint.
        retry_raw = ""
        missing = [q for q in normalized_questions if q["id"] not in parsed]
        if missing:
            retry_questions = [(q["id"], q["question"]) for q in missing]
            logger.debug(
                "[%s] %d/%d rubric Qs unparsed in first call, retrying", sample_id, len(missing), n
            )
            retry_client, retry_limiter = await _get_shared_client(timeout=self.timeout)
            retry_raw = await self._call_vlm_batch(
                retry_client, retry_limiter, model, render_path, retry_questions, sample_id, attempt=2,
            )
            retry_valid_ids = {q["id"] for q in missing}
            retry_parsed = _parse_batch_answers(retry_raw, retry_valid_ids)
            parsed.update(retry_parsed)

        # ── Tally ──
        details = []
        correct = 0
        total = n
        for q in normalized_questions:
            qid = q["id"]
            qtext = q["question"]
            exp = q["expected"]
            ans = parsed.get(qid)
            if ans is None:
                details.append({
                    "id": qid,
                    "question": qtext,
                    "expected": exp,
                    "answer": "invalid",
                    "correct": False,
                    "counted": True,
                })
                continue
            is_correct = (ans == exp)
            details.append({
                "id": qid,
                "question": qtext,
                "expected": exp,
                "answer": ans,
                "correct": is_correct,
                "counted": True,
            })
            if is_correct:
                correct += 1

        # Log raw VLM outputs at debug level for offline review
        logger.debug(
            "[%s] rubric raw first=%r retry=%r parsed=%s",
            sample_id, first_raw[:300], retry_raw[:300], parsed,
        )
        return correct, total, details

    async def _call_vlm_batch(
        self,
        client,
        limiter,
        model: str,
        render_path: str,
        questions: list[tuple[str, str]],
        sample_id: str,
        attempt: int,
    ) -> str:
        """One logical VLM call (with internal backoff retries) returning raw
        text. Empty string only if all retries exhausted.

        The limiter (AsyncLimiter or None) enforces per-endpoint RPM caps
        shared across all workers; we wrap each chat call with it.

        Retries on transient errors: 429 RateLimit, 500 InternalServerError,
        APITimeoutError, APIConnectionError, TimeoutError. Other exceptions
        (auth fail, bad request) abort immediately.

        `attempt` arg is for log labelling (1 = first batched call across
        all Qs, 2 = follow-up call for the questions whose answers were
        missing); internal retries don't bump it.
        """
        import random
        from rl_verl.vlm_reward import _img_part

        prompt = _build_rubric_batch_prompt(questions)
        # Encode the (high-detail, up-to-1024px) render off the event loop.
        # _img_part does PIL open + LANCZOS resize + JPEG + base64 — all
        # synchronous CPU work. Running it inline on the loop (as the original
        # code did) adds to the per-rollout loop-blocking that, fanned out
        # across an unbounded number of rollouts, starves vLLM's zmq poll.
        # CLIPDesignScorer already offloads its heavy compute the same way.
        loop = asyncio.get_running_loop()
        img_part = await loop.run_in_executor(None, _img_part, render_path, "rubric")
        content_parts = [
            img_part,
            {"type": "text", "text": prompt},
        ]

        # Errors worth retrying with backoff (Azure transient).
        # Imported lazily so module import doesn't fail if openai is absent.
        try:
            from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
            _TRANSIENT = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, asyncio.TimeoutError)
        except Exception:
            _TRANSIENT = (asyncio.TimeoutError,)

        last_exc: Exception | None = None
        for tries in range(self.max_retries + 1):  # 0..max_retries inclusive
            try:
                async def _do_call():
                    # Only pass temperature if explicitly configured (reasoning
                    # models reject any non-default value and even rejecting
                    # 0.0 — see Azure 400 unsupported_value for gpt-5.x).
                    kw = dict(
                        model=model,
                        messages=[{"role": "user", "content": content_parts}],
                        max_completion_tokens=self.max_tokens,
                    )
                    if self.temperature is not None:
                        kw["temperature"] = self.temperature
                    return await asyncio.wait_for(
                        client.chat.completions.create(**kw),
                        timeout=self.timeout,
                    )

                if limiter is not None:
                    async with limiter:
                        response = await _do_call()
                else:
                    response = await _do_call()
                text = (response.choices[0].message.content or "").strip()
                if tries > 0:
                    logger.info(
                        "[%s] rubric VLM recovered on retry %d/%d (attempt=%d)",
                        sample_id, tries, self.max_retries, attempt,
                    )
                return text
            except _TRANSIENT as exc:
                last_exc = exc
                if tries >= self.max_retries:
                    break
                # Exponential backoff with jitter: 1s, 2s, 4s, 8s ...
                wait = self.backoff_base * (2 ** tries) + random.uniform(0, 0.3)
                logger.debug(
                    "[%s] rubric VLM transient %s on attempt=%d try=%d, backing off %.1fs",
                    sample_id, type(exc).__name__, attempt, tries + 1, wait,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                # Non-transient (auth fail, malformed request, etc.) — abort.
                logger.warning(
                    "[%s] rubric VLM non-transient attempt=%d: %s: %s",
                    sample_id, attempt, type(exc).__name__, str(exc)[:200],
                )
                return ""

        # All retries exhausted
        logger.warning(
            "[%s] rubric VLM call attempt=%d failed after %d retries: %s: %s",
            sample_id, attempt, self.max_retries,
            type(last_exc).__name__ if last_exc else "Unknown",
            str(last_exc)[:200] if last_exc else "",
        )
        return ""

    @property
    def needs_group_barrier(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Factory — singleton per worker process
# ---------------------------------------------------------------------------
_scorer_instance: BaseDesignScorer | None = None


def get_design_scorer() -> BaseDesignScorer:
    """Get the configured design scorer (lazy singleton).

    Reads DESIGN_SCORER env var: swiss | clip | rubric (default: swiss).
    """
    global _scorer_instance
    if _scorer_instance is not None:
        return _scorer_instance

    scorer_type = os.environ.get("DESIGN_SCORER", "swiss").lower().strip()

    if scorer_type == "clip":
        _scorer_instance = CLIPDesignScorer()
        logger.info("Design scorer: CLIP (backend=%s)", _scorer_instance.backend)
    elif scorer_type == "rubric":
        _scorer_instance = RubricDesignScorer()
        logger.info("Design scorer: Rubric (concurrency=%d)", _scorer_instance.concurrency)
    elif scorer_type == "swiss":
        _scorer_instance = SwissDesignScorer()
        logger.info("Design scorer: Swiss tournament")
    else:
        logger.warning("Unknown DESIGN_SCORER=%r, falling back to Swiss", scorer_type)
        _scorer_instance = SwissDesignScorer()

    return _scorer_instance
