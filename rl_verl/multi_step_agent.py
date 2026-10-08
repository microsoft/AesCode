"""Single-step AgentLoop for AesCode GDPO training (generate-only).

Architecture — Single-step generate:
  [system + user(image + prompt)] → HTML

Registration:
  @register("single_step_generate") makes this available as:
    actor_rollout_ref.rollout.agent.default_agent_loop=single_step_generate
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from PIL import Image

# Ensure repo root on path for our imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from inference.prompts import DEFAULT_SYSTEM_PROMPT
from rl_verl.data_prep import IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS
from rl_verl.rewards import _extract_html
from rl_verl.training_monitor import get_monitor
from rl_verl.reward_fn import get_last_render_path

# Quality dimensions for GDPO
_QUALITY_DIMS = ["execution", "text", "boundary", "layout", "tablechart", "whitespace", "design"]


async def _compute_reward_async(
    html: str,
    sample_id: str,
    verifiable_requirements: list,
    reference_metadata: dict,
    interaction_kwargs: dict | None = None,
) -> tuple[float, dict]:
    """Async verifier + VLM reward with weighted scoring."""
    if not html or len(html.strip()) < 10:
        return 0.0, {
            "score": 0.0, "execution": 0.0, "text": 0.0, "boundary": 0.0,
            "layout": 0.0, "tablechart": 0.0, "whitespace": 0.0,
            "design": 0.0, "vlm": 0.0, "reward_unweighted": 0.0,
        }
    try:
        from rl_verl.reward_fn import compute_score_async

        _ikw = interaction_kwargs or {}
        extra_info = {
            "sample_id": sample_id,
            "verifiable_requirements": json.dumps(verifiable_requirements)
                if not isinstance(verifiable_requirements, str) else verifiable_requirements,
            "reference_metadata": json.dumps(reference_metadata)
                if not isinstance(reference_metadata, str) else reference_metadata,
            "interaction_kwargs": _ikw,
        }
        # Surface rubric_questions at extra_info TOP LEVEL too (RubricDesignScorer's
        # preferred read path; ikw is only its fallback). Belt-and-suspenders so the
        # design dim can never silently collapse to applicable=0 again.
        if isinstance(_ikw, dict) and _ikw.get("rubric_questions"):
            extra_info["rubric_questions"] = _ikw["rubric_questions"]
        result = await compute_score_async("train", f"```html\n{html}\n```", "", extra_info)
        return result["score"], result
    except Exception as exc:
        logger.warning("Async verifier failed for %s: %s", sample_id, exc)
        # Best-effort fallback: register to coordinator so peers don't wait
        try:
            ikw = interaction_kwargs if isinstance(interaction_kwargs, dict) else {}
            design_enabled = True
            try:
                from rl_verl.reward_fn import _is_design_validator_enabled as _design_on
                design_enabled = bool(_design_on())
            except Exception:
                pass

            if design_enabled:
                from rl_verl.design_coordinator import get_generate_coordinator
                ref_path = reference_metadata.get("reference_image_path") if isinstance(reference_metadata, dict) else None
                _gs = ikw.get("global_step") if isinstance(ikw, dict) else None
                global_step = int(_gs) if _gs is not None else None
                await get_generate_coordinator().submit_and_wait(
                    sample_id=sample_id,
                    sub_step_idx=0,
                    ref_path=ref_path,
                    render_path=None,
                    model_failed=False,
                    global_step=global_step,
                )
        except Exception as reg_exc:
            logger.warning("Coordinator fallback submit failed for %s: %s", sample_id, reg_exc)
        return 0.0, {
            "score": 0.0, "execution": 0.0, "text": 0.0, "boundary": 0.0,
            "layout": 0.0, "tablechart": 0.0, "whitespace": 0.0,
            "design": 0.0, "vlm": 0.0, "reward_unweighted": 0.0,
        }


def _safe_json(val):
    """Parse JSON string if needed, return dict/list or empty fallback."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {} if val.startswith("{") else []
    return val


@register("single_step_generate")
class SingleStepGenerateAgentLoop(AgentLoopBase):
    """Single-step generate agent loop for GDPO training.

    Each rollout: prompt + image → HTML → reward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def run(
        self, sampling_params: dict[str, Any], *, validate: bool = False, **kwargs
    ) -> AgentLoopOutput:
        """Single-step generate: [system + user(image + prompt)] → HTML"""
        messages = list(kwargs["raw_prompt"])
        extra_info = _safe_json(kwargs.get("extra_info", {}))
        ikw = extra_info.get("interaction_kwargs", {})
        if isinstance(ikw, dict) and "global_step" not in ikw:
            kw_step = kwargs.get("global_step", kwargs.get("global_steps"))
            if kw_step is not None:
                ikw["global_step"] = kw_step

        # ── Fold top-level rubric_questions into interaction_kwargs ──
        # data_prep writes rubric_questions at extra_info TOP LEVEL, but the
        # reward path only forwards interaction_kwargs (see _compute_reward_async
        # below, which builds a 4-key extra_info: sample_id / verifiable_requirements
        # / reference_metadata / interaction_kwargs). The RubricDesignScorer reads
        # extra_info["rubric_questions"] then falls back to
        # interaction_kwargs["rubric_questions"]; if neither is present it returns
        # applicable=0 and design silently collapses to 0.0 for the whole run.
        # Fold the top-level rubric into ikw (idempotent: don't clobber an existing
        # one) so the scorer's fallback path always sees it. Without this, the
        # no-image run's design dim reads 0.0 while the image baseline had ~0.82.
        if isinstance(ikw, dict) and "rubric_questions" not in ikw:
            _rq = extra_info.get("rubric_questions")
            if _rq:
                ikw["rubric_questions"] = _rq

        sample_id = ikw.get("sample_id", "unknown")
        v_reqs = _safe_json(ikw.get("verifiable_requirements", []))
        ref_meta = _safe_json(ikw.get("reference_metadata", {}))

        # ── Extract images from the dataset-prepared messages ──
        multi_modal_data = await self.process_vision_info(messages)
        multi_modal_data = dict(multi_modal_data)
        images = list(multi_modal_data.get("images") or [])
        videos = multi_modal_data.get("videos")
        # No-image mode: an EMPTY images list must not be propagated. verl's
        # apply_chat_template and _agent_loop_postprocess->_compute_multi_modal_inputs
        # both unconditionally feed multi_modal_data["images"] to the Qwen3-VL
        # processor, whose group_images_by_shape() does images[0].device and
        # raises IndexError on []. Drop the key so every downstream .get("images")
        # returns None and the text-only path is taken.
        if images:
            multi_modal_data["images"] = images
        else:
            multi_modal_data.pop("images", None)

        # ── Build prompt (use dataset-prepared messages directly) ──
        # No-image mode: images==[] must be passed as None, otherwise verl's
        # apply_chat_template still routes through the Qwen3-VL processor, whose
        # group_images_by_shape() does images[0].device on an empty list and
        # raises IndexError. Empty list -> None keeps the text-only path intact.
        _img_arg = images if images else None
        _vid_arg = videos if videos else None
        prompt_ids = await self.apply_chat_template(
            messages, images=_img_arg, videos=_vid_arg,
        )

        metrics: dict[str, Any] = {}

        # ── Generate ──
        sp = dict(sampling_params)
        sp["max_tokens"] = min(
            sp.get("max_tokens", self.response_length),
            self.response_length,
        )

        # Validation phase detection:
        # Primary: verl explicitly passes validate=True (forwarded from
        # batch.meta_info["validate"] via patched _run_agent_loop).
        # Fallback: temperature==0 was the legacy signal — kept so we don't
        # silently regress to swiss-deadlock if verl_custom.patch isn't
        # reapplied. When val_kwargs sets temperature>0 (new config), the
        # primary check kicks in instead.
        try:
            _temp_zero = float(sp.get("temperature", 1.0)) == 0.0
        except (TypeError, ValueError):
            _temp_zero = False
        _is_val = bool(validate) or _temp_zero
        if _is_val and isinstance(ikw, dict):
            ikw = dict(ikw)
            ikw["_is_val"] = True

        with simple_timer("generate_step", metrics):
            print(f"[Agent] generate() call: prompt_len={len(prompt_ids)}, max_tokens={sp.get('max_tokens')}, "
                  f"has_images={images is not None and len(images) > 0}", flush=True)
            import time as _atime
            _astart = _atime.time()
            try:
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sp,
                    image_data=images,
                    video_data=videos,
                )
            except Exception as _aerr:
                print(f"[Agent] generate() FAILED after {_atime.time() - _astart:.1f}s: {_aerr}", flush=True)
                import traceback; traceback.print_exc()
                raise
            print(f"[Agent] generate() done in {_atime.time() - _astart:.1f}s, output_len={len(output.token_ids)}", flush=True)

        gen_ids = output.token_ids
        routed_experts = output.routed_experts

        # Decode → extract HTML → score
        text = await self.loop.run_in_executor(
            None,
            lambda ids=gen_ids: self.tokenizer.decode(
                ids, skip_special_tokens=True
            ),
        )
        generated_html = _extract_html(text)

        # ── Compute reward ──
        _reward_timeout = float(os.environ.get("REWARD_TIMEOUT", "0"))
        if _reward_timeout > 0:
            try:
                reward, per_validator = await asyncio.wait_for(
                    _compute_reward_async(
                        generated_html, sample_id, v_reqs, ref_meta,
                        interaction_kwargs=ikw,
                    ),
                    timeout=_reward_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[%s] reward computation TIMEOUT after %.0fs — assigning reward=0",
                    sample_id, _reward_timeout,
                )
                reward = 0.0
                per_validator = {
                    "score": 0.0, "execution": 0.0, "text": 0.0, "boundary": 0.0,
                    "layout": 0.0, "tablechart": 0.0, "whitespace": 0.0,
                    "design": 0.0, "vlm": 0.0, "reward_unweighted": 0.0,
                    "_reward_timeout": True,
                }
        else:
            reward, per_validator = await _compute_reward_async(
                generated_html, sample_id, v_reqs, ref_meta,
                interaction_kwargs=ikw,
            )

        # Record to training monitor
        try:
            monitor = get_monitor()
            global_step = int(ikw.get("global_step", os.environ.get("VERL_GLOBAL_STEP", 0)))
            step_render_path = get_last_render_path(sample_id)
            step_ref_path = ref_meta.get("reference_image_path") if isinstance(ref_meta, dict) else None
            monitor.record(
                global_step, sample_id, per_validator,
                render_image_path=step_render_path,
                ref_image_path=step_ref_path,
            )
        except Exception as exc:
            logger.debug("Monitor record failed: %s", exc)

        # Design diagnostic fields
        design_applicable = int(per_validator.get("design_applicable", 1))
        design_confidence = float(per_validator.get("design_confidence", 1.0))

        # Log
        _score_keys = {"execution", "text", "boundary", "layout", "tablechart",
                       "whitespace", "design", "vlm"}
        pv_display = {k: f"{per_validator[k]:.2f}" for k in _score_keys
                      if k in per_validator and per_validator[k]} if per_validator else {}
        _log_msg = (
            f"[{sample_id}] GENERATE  "
            f"reward={reward:.3f}  "
            f"gen={len(gen_ids)}  prompt={len(prompt_ids)}  validators={pv_display or 'N/A'}  "
            f"design=app={design_applicable}/c={design_confidence:.2f}"
        )
        logger.info(_log_msg)
        print(_log_msg, flush=True)

        # ── Training monitor report ──
        try:
            monitor = get_monitor()
            global_step = int(ikw.get("global_step", os.environ.get("VERL_GLOBAL_STEP", 0)))
            monitor.maybe_generate_report(global_step)
        except Exception as exc:
            logger.debug("Monitor report failed: %s", exc)

        # Safety: ensure prompt_ids fits within prompt_length
        if len(prompt_ids) > self.prompt_length:
            logger.warning(
                "Prompt_ids (%d) exceeds prompt_length (%d), truncating",
                len(prompt_ids), self.prompt_length,
            )
            prompt_ids = prompt_ids[: self.prompt_length]

        resp_ids = gen_ids[: self.response_length]
        resp_mask = [1] * len(resp_ids)
        resp_lp = (output.log_probs or [])[: self.response_length] if output.log_probs else None

        out = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=resp_ids,
            response_mask=resp_mask,
            response_logprobs=resp_lp,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            reward_score=reward,
            num_turns=1,
            metrics=metrics,
            extra_fields={},
        )

        # reward_extra_info: GDPO reads these for per-dimension advantage
        reward_extra_info = {
            "score": reward,
            "reward_unweighted": per_validator.get("reward_unweighted", 0.0),
        }
        for dim in _QUALITY_DIMS:
            reward_extra_info[dim] = per_validator.get(dim, 0.0)
        # Zero design dim if not applicable (infra failure)
        if not design_applicable:
            reward_extra_info["design"] = 0.0

        reward_extra_info["tablechart_applicable"] = per_validator.get("tablechart_applicable", 0.0)
        reward_extra_info["design_confidence"] = float(design_confidence)
        reward_extra_info["design_applicable"] = float(design_applicable)

        out.extra_fields.update(
            {
                "reward_extra_info": reward_extra_info,
                "per_validator": per_validator,
                "quality_reward": reward,
                "generated_html": generated_html,
            }
        )

        return out
