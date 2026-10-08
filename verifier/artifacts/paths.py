from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict


def _unique_suffix() -> str:
    """Return a per-process-per-thread suffix to avoid file collisions."""
    return f"{os.getpid()}_{threading.get_ident()}"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Thread-local artifact root override. Set via set_artifact_root().
# Using thread-local storage so concurrent verifier calls don't clobber
# each other's paths.
_tls = threading.local()


def set_artifact_root(root: Path | None) -> None:
    """Override where per-sample artifact directories are created (thread-safe)."""
    _tls.artifact_root = root


def _get_artifact_root() -> Path | None:
    return getattr(_tls, "artifact_root", None)


def get_sample_artifact_dir(sample_id: str) -> Path:
    artifact_root = _get_artifact_root()
    if artifact_root is not None:
        output_dir = artifact_root / sample_id
    else:
        output_dir = get_repo_root() / "artifacts" / "samples" / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_source_html_artifact(*, sample_id: str, html_source: str) -> Path:
    output_path = get_sample_artifact_dir(sample_id) / "source.html"
    output_path.write_text(html_source, encoding="utf-8")
    return output_path


def write_semantic_html_artifact(*, sample_id: str, enriched_html: str) -> Path:
    output_path = get_sample_artifact_dir(sample_id) / "semantic.html"
    output_path.write_text(enriched_html, encoding="utf-8")
    return output_path


def get_render_artifact_path(*, sample_id: str, backend: str = "playwright") -> Path:
    return get_sample_artifact_dir(sample_id) / f"render_{backend}_{_unique_suffix()}.png"


def get_render_debug_artifact_path(*, sample_id: str, backend: str = "playwright", kind: str = "overflow") -> Path:
    return get_sample_artifact_dir(sample_id) / f"render_{backend}_{kind}_{_unique_suffix()}.png"


def get_layout_matching_artifact_path(*, sample_id: str) -> Path:
    return get_sample_artifact_dir(sample_id) / "layout_matching.json"


def get_layout_overlay_artifact_path(*, sample_id: str, kind: str) -> Path:
    return get_sample_artifact_dir(sample_id) / f"layout_{kind}.png"


def write_layout_matching_artifact(*, sample_id: str, payload: Dict[str, Any]) -> Path:
    output_path = get_layout_matching_artifact_path(sample_id=sample_id)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def resolve_reference_image_path(
    sample_id: str,
    *,
    declared_path: str | None = None,
) -> Path | None:
    """Resolve the reference image for *sample_id*.

    Resolution order:
      1. ``declared_path`` as-is (absolute or relative to repo root).
      2. If the absolute ``declared_path`` doesn't exist, try the **filename**
         part inside every directory listed in ``REFERENCE_IMAGE_DIRS`` (env var,
         colon-separated) or ``REFERENCE_IMAGE_DIR`` (single dir, legacy).
      3. ``{sample_id}.png`` / ``.jpg`` inside those same override directories.
      4. Hard-coded well-known directories under the repo root.

    This makes the pipeline portable across machines where the image folder
    may live under different mount-point prefixes (e.g. ``/mnt/data`` vs
    ``/srv/data``).  Set the env var once and every callsite benefits.
    """
    import os

    repo_root = Path(__file__).resolve().parents[2]

    # -- Collect override image directories from env vars -----------------
    _override_dirs: list[Path] = []
    for env_key in ("REFERENCE_IMAGE_DIRS", "REFERENCE_IMAGE_DIR"):
        raw = os.environ.get(env_key, "")
        if raw:
            for part in raw.split(":"):
                p = Path(part.strip())
                if p.is_dir():
                    _override_dirs.append(p)

    # -- 1. Declared path (exact match) -----------------------------------
    if declared_path:
        declared = Path(declared_path)
        if declared.is_absolute():
            if declared.exists():
                return declared
        else:
            resolved = repo_root / declared
            if resolved.exists():
                return resolved

        # -- 2. Filename from declared_path in override dirs ---------------
        filename = declared.name  # e.g. "benchmark_in_domain_original_178.png"
        for d in _override_dirs:
            candidate = d / filename
            if candidate.exists():
                return candidate

    # -- 3. sample_id-based lookup in override dirs -----------------------
    for d in _override_dirs:
        for ext in ("png", "jpg", "jpeg", "webp"):
            candidate = d / f"{sample_id}.{ext}"
            if candidate.exists():
                return candidate

    # -- 4. Fallback: well-known directories under repo root ---------------
    candidates = [
        repo_root / "data" / "testset" / f"{sample_id}.png",
        repo_root / "data" / "testset" / f"{sample_id}.jpg",
        repo_root / "dataset" / "filter_image" / f"{sample_id}.png",
        repo_root / "dataset" / "filter_image" / "new_image" / f"{sample_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_validation_report_artifact(*, sample_id: str, backend: str, payload: Dict[str, Any]) -> Path:
    output_path = get_sample_artifact_dir(sample_id) / f"report_{backend}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path