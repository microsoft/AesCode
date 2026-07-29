"""Visual validators — WhitespaceValidator, DesignValidator.

Layer 3 of stage-3 verifier (Visual).

* WhitespaceValidator  : Canny edge + background-deviation content detection
                         on a 12×8 grid, cosine similarity between ref and pred.
* DesignValidator      : VLM-based visual design quality assessment via GPT-5.4.
                         Requires external API; returns NOT_APPLICABLE when API
                         is unavailable or unconfigured (e.g. single-sample eval).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from verifier.base import BaseValidator
from verifier.types import ArtifactBundle, Requirement, ValidationResult, ValidationStatus


# ===================================================================
# WhitespaceValidator  — ref vs pred spatial-distribution comparison
# ===================================================================

# Background-deviation threshold (uint8 intensity scale).
_WS_BG_DEVIATION: float = 25.0


def _safe_imread(path: str | Path, grayscale: bool = False) -> Any:
    """Read image via PIL (thread-safe) and return as numpy BGR array."""
    import numpy as np
    from PIL import Image, ImageFile

    try:
        _orig = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img = Image.open(str(path))
            img.load()
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = _orig
        if grayscale:
            img = img.convert("L")
            return np.asarray(img)
        else:
            img = img.convert("RGB")
            return np.asarray(img)[:, :, ::-1].copy()
    except Exception:
        import cv2
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        result = cv2.imread(str(path), flag)
        if result is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return result


def _ws_content_density_grid(image_path: str, rows: int, cols: int):
    """Load image → detect content → return R×C density grid.

    Each image is processed independently on its own pixel space, then
    divided into an R×C grid.  No shared canvas or padding needed —
    aspect ratio differences are irrelevant because both grids are the
    same (rows, cols) shape regardless of source image dimensions.

    Returns (rows, cols) float64 array where each cell is the fraction
    of content pixels in that cell.
    """
    import cv2
    import numpy as np

    image = _safe_imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Content detection (Canny + background deviation) ──────────
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_filled = cv2.dilate(edges, kernel, iterations=3)
    edge_filled = cv2.morphologyEx(edge_filled, cv2.MORPH_CLOSE, kernel, iterations=1)

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_value = float(np.median(border))
    deviation = np.abs(gray.astype(np.float64) - bg_value)
    bg_content = (deviation > _WS_BG_DEVIATION).astype(np.uint8) * 255

    content = cv2.bitwise_or(edge_filled, bg_content) > 0

    # ── Build density grid ────────────────────────────────────────
    h, w = gray.shape[:2]
    density = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            y0 = r * h // rows
            y1 = (r + 1) * h // rows
            x0 = c * w // cols
            x1 = (c + 1) * w // cols
            cell = content[y0:y1, x0:x1]
            density[r, c] = float(cell.mean())

    return density


class WhitespaceValidator(BaseValidator):
    """Compare spatial distribution of content between rendered and reference images.

    Algorithm (Phase 6 / W1 multi-scale cosine):
      1. Content detection  — Canny edges (dilated) + background-deviation.
      2. Each image independently → density grids at TWO scales:
            coarse 12×8 (preserves the original validator's translation tolerance)
            fine   24×16 (provides finer gradient for small layout shifts;
                          theoretical cosine ceiling jumps from ~0.85 to ~0.92)
      3. Flatten each grid to a vector and compute cosine similarity per scale.
      4. Final score = 0.5 × cos(coarse) + 0.5 × cos(fine).

    Why multi-scale (vs the previous single 12×8 grid): the single-grid
    cosine saturated around 0.78 during training with std collapsing to
    0.12, leaving GRPO no advantage signal.  The fine grid keeps producing
    distinguishable scores for sub-cell layout changes, restoring spread.
    Phase 6 spike: discr +12%, std +11%, robust unchanged.
    """

    name = "WhitespaceValidator"
    supported_categories = ("style_cue",)
    GRID_ROWS: int = 12
    GRID_COLS: int = 8
    FINE_GRID_ROWS: int = 24
    FINE_GRID_COLS: int = 16
    COARSE_WEIGHT: float = 0.5
    FINE_WEIGHT: float = 0.5

    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        render_path = artifacts.render_image_path
        reference_path = (
            Path(str(artifacts.reference_image_path)) if artifacts.reference_image_path else None
        )
        if not render_path or reference_path is None or not reference_path.exists():
            return self.build_result(
                requirement,
                status=ValidationStatus.NOT_APPLICABLE,
                score=0.0,
                message="Whitespace analysis skipped: render/reference image unavailable.",
                evidence={"evidence_basis": "unknown", "evidence_sources": []},
            )

        try:
            import numpy as np

            pred_coarse = _ws_content_density_grid(
                str(render_path), self.GRID_ROWS, self.GRID_COLS,
            )
            ref_coarse = _ws_content_density_grid(
                str(reference_path), self.GRID_ROWS, self.GRID_COLS,
            )
            pred_fine = _ws_content_density_grid(
                str(render_path), self.FINE_GRID_ROWS, self.FINE_GRID_COLS,
            )
            ref_fine = _ws_content_density_grid(
                str(reference_path), self.FINE_GRID_ROWS, self.FINE_GRID_COLS,
            )
        except Exception as exc:  # noqa: BLE001
            return self.build_result(
                requirement,
                status=ValidationStatus.UNCERTAIN,
                score=0.0,
                message=f"Whitespace analysis failed: {exc}",
                evidence={
                    "evidence_basis": "unknown",
                    "evidence_sources": ["render_image", "reference_image"],
                },
            )

        import numpy as np

        eps = 1e-8

        def _cos(pred, ref):
            p = pred.flatten()
            r = ref.flatten()
            return max(0.0, float(
                np.dot(p, r) / (np.linalg.norm(p) * np.linalg.norm(r) + eps)
            ))

        score_coarse = _cos(pred_coarse, ref_coarse)
        score_fine = _cos(pred_fine, ref_fine)
        score = self.COARSE_WEIGHT * score_coarse + self.FINE_WEIGHT * score_fine

        score = round(score, 4)
        score_coarse_r = round(score_coarse, 4)
        score_fine_r = round(score_fine, 4)

        return self.build_result(
            requirement,
            status=ValidationStatus.PASS,
            score=score,
            measured={
                "grid_size_coarse": [self.GRID_ROWS, self.GRID_COLS],
                "grid_size_fine": [self.FINE_GRID_ROWS, self.FINE_GRID_COLS],
                "score_coarse": score_coarse_r,
                "score_fine": score_fine_r,
                "pred_coverage_coarse": round(float(pred_coarse.mean()), 4),
                "ref_coverage_coarse": round(float(ref_coarse.mean()), 4),
            },
            message=f"Whitespace: score={score} (coarse={score_coarse_r}, fine={score_fine_r}).",
            evidence={
                "evidence_basis": "hybrid",
                "evidence_sources": ["render_image", "reference_image"],
            },
        )


# ===================================================================
# DesignValidator  — VLM-based visual design quality assessment
# ===================================================================


class DesignValidator(BaseValidator):
    """Visual design quality assessment via VLM — the unified entry point.

    Three modes:

      1. **eval** — Single-sample evaluation: ref + 1 render → 0-5 fidelity → [0,1].
         Called via ``validate()`` from the verifier pipeline.

      2. **generate** — RL generate sub-step: ref + N renders → Swiss tournament → [0,1].
         Coordinated by ``rl_verl.design_coordinator.GenerateSwissCoordinator``,
         which calls ``rl_verl.vlm_reward.swiss_tournament_judge`` directly.

      3. **refine** — RL refine sub-step: ref + prev + cur → pairwise delta → [-1,+1].
         Handled per-rollout by ``rl_verl.design_coordinator.RefineDesignScorer``,
         which calls ``rl_verl.vlm_reward.pairwise_one_with_meta`` directly.

    ``validate()`` only handles eval mode. Generate/refine bypass this class
    entirely and live in the rl_verl coordinator layer.

    To disable: ``DISABLED_VALIDATORS=DesignValidator``
    """

    name = "DesignValidator"
    supported_categories = ("style_cue",)

    # ------------------------------------------------------------------
    # Per-sample entry (eval mode, called by verifier pipeline)
    # ------------------------------------------------------------------

    def validate(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        render_path = artifacts.render_image_path
        if not render_path or not Path(render_path).exists():
            return self._not_applicable(requirement, "render image unavailable")

        return self._validate_eval(requirement, artifacts)

    def _not_applicable(self, requirement: Requirement, reason: str) -> ValidationResult:
        return self.build_result(
            requirement,
            status=ValidationStatus.NOT_APPLICABLE,
            score=0.0,
            message=f"Design analysis skipped: {reason}",
            evidence={"evidence_basis": "unknown", "evidence_sources": []},
        )

    def _validate_eval(self, requirement: Requirement, artifacts: ArtifactBundle) -> ValidationResult:
        """Eval mode: compare render against oracle reference image via VLM (0-5 scale)."""
        import os

        # Load .env if not already loaded (validator may run in worker thread
        # that hasn't seen vlm_reward's _ensure_dotenv() yet).
        try:
            from rl_verl.vlm_reward import _ensure_dotenv
            _ensure_dotenv()
        except Exception:
            pass

        render_path = artifacts.render_image_path
        reference_path = (
            Path(str(artifacts.reference_image_path)) if artifacts.reference_image_path else None
        )
        if reference_path is None or not reference_path.exists():
            return self._not_applicable(requirement, "reference image unavailable for eval")

        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "") or os.environ.get("API_KEY", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "") or os.environ.get("ENDPOINT", "")
        # Also check numbered endpoints (ENDPOINT1, ENDPOINT2, ...)
        if not (api_key and endpoint):
            for i in range(1, 10):
                api_key = os.environ.get(f"API_KEY{i}", "")
                endpoint = os.environ.get(f"ENDPOINT{i}", "")
                if api_key and endpoint:
                    break
        # DesignCoder: also accept VLM_REWARD_* env vars (used by RL training endpoint pool).
        # vlm_score_batch_sync below ultimately reads VLM_REWARD_BASE_URL/API_KEY via
        # _load_all_endpoints(); we only need this presence check to avoid a needless
        # _not_applicable early-return when only VLM_REWARD_* is configured.
        if not (api_key and endpoint):
            api_key = os.environ.get("VLM_REWARD_API_KEY", "")
            endpoint = os.environ.get("VLM_REWARD_BASE_URL", "")
        if not (api_key and endpoint):
            for i in range(1, 10):
                api_key = os.environ.get(f"VLM_REWARD_API_KEY{i}", "")
                endpoint = os.environ.get(f"VLM_REWARD_BASE_URL{i}", "")
                if api_key and endpoint:
                    break
        if not (api_key and endpoint):
            return self._not_applicable(requirement, "VLM API credentials not configured")

        try:
            from rl_verl.vlm_reward import vlm_score_batch_sync

            scores, rationales = vlm_score_batch_sync(
                [str(render_path)],
                [str(reference_path)],
                mode_override="eval",
            )
            score = scores[0] if scores else 0.0
            rationale = rationales[0] if rationales else ""
        except Exception as exc:  # noqa: BLE001
            return self.build_result(
                requirement,
                status=ValidationStatus.UNCERTAIN,
                score=0.0,
                message=f"Design eval failed: {exc}",
                evidence={
                    "evidence_basis": "unknown",
                    "evidence_sources": ["render_image", "reference_image"],
                },
            )

        score = round(max(0.0, min(1.0, score)), 4)
        return self.build_result(
            requirement,
            status=ValidationStatus.PASS,
            score=score,
            measured={
                "vlm_score": score,
                "vlm_rationale": rationale[:300],
                "vlm_mode": "eval",
            },
            message=f"Design (eval fidelity): score={score}.",
            evidence={
                "evidence_basis": "hybrid",
                "evidence_sources": ["render_image", "reference_image", "vlm_api"],
            },
        )


