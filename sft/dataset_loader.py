from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from inference.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_TEMPLATE,
    GENERATE_COT_STUB,
    GENERATE_OUTPUT_INSTRUCTION,
    LEGACY_USER_GENERATE,
    LEGACY_USER_REPAIR,
    OUTPUT_SECTION_MARKER,
    REFINE_PREV_RENDER_MAX_PIXELS,
    REFINE_PREV_RENDER_MIN_PIXELS,
    build_refine_user_text,
    build_user_text,
    refine_use_prev_render,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy user-text templates are now imported from inference.prompts
# (LEGACY_USER_GENERATE, LEGACY_USER_REPAIR) so SFT/RL share a single source.
# Keep backward-compatible aliases for any external callers.
# ---------------------------------------------------------------------------
_LEGACY_USER_GENERATE = LEGACY_USER_GENERATE
_LEGACY_USER_REPAIR = LEGACY_USER_REPAIR

# Assistant output templates — unified COT + fence format for both tasks so the
# model learns one consistent output distribution.
ASSISTANT_COT_HTML = "{cot}\n\n```html\n{clean_html}\n```"


def _load_image(path_or_b64: str) -> Image.Image:
    path = Path(path_or_b64)
    if path.exists():
        return Image.open(path).convert("RGB")
    raw = base64.b64decode(path_or_b64)
    import io

    return Image.open(io.BytesIO(raw)).convert("RGB")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _strip_outer_fences(html: str) -> str:
    match = re.search(r"```(?:html)?\s*([\s\S]+?)```", html, re.IGNORECASE)
    return match.group(1).strip() if match else html.strip()


class SFTDataset(Dataset):
    def __init__(
        self,
        processor,
        dir_path: Optional[str] = None,
        jsonl_path: Optional[str] = None,
        # Pipeline-dirs mode: pass spec_jsonl + at least refine_dir or draft_dir
        spec_jsonl: Optional[str] = None,
        refine_dir: Optional[str] = None,
        draft_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        cot_dir: Optional[str] = None,
        cot_generate_dir: Optional[str] = None,
        cot_repair_dir: Optional[str] = None,
        max_length: int = 8192,
        image_min_pixels: int = 256 * 28 * 28,
        image_max_pixels: int = 1280 * 28 * 28,
        difficulty_levels: Optional[list] = None,
        generate_ratio: float = 0.0,
        repair_max_buggy_chars: int = 30000,
        duplicate_tasks: bool = False,
        # Dual-JSONL mode: separate spec files for generate and refine tasks
        spec_jsonl_generate: Optional[str] = None,
        spec_jsonl_refine: Optional[str] = None,
    ):
        dual_jsonl_mode = spec_jsonl_generate is not None or spec_jsonl_refine is not None
        pipeline_mode = spec_jsonl is not None
        legacy_mode = dir_path is not None or jsonl_path is not None

        if dual_jsonl_mode and (pipeline_mode or legacy_mode):
            raise ValueError(
                "Cannot mix dual-jsonl mode (spec_jsonl_generate/spec_jsonl_refine) "
                "with spec_jsonl or dir_path/jsonl_path."
            )
        if not dual_jsonl_mode and not pipeline_mode and not legacy_mode:
            raise ValueError(
                "Provide one of: (1) spec_jsonl + refine_dir/draft_dir, "
                "(2) spec_jsonl_generate/spec_jsonl_refine + refine_dir, "
                "(3) jsonl_path, or (4) dir_path."
            )
        if pipeline_mode and legacy_mode:
            raise ValueError("Cannot mix pipeline-dirs mode with dir_path/jsonl_path.")
        if legacy_mode and (dir_path is None) == (jsonl_path is None):
            raise ValueError("Provide exactly one of dir_path or jsonl_path.")

        self.processor = processor
        self.max_length = max_length
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self._allowed_levels = set(difficulty_levels) if difficulty_levels else None
        self.generate_ratio = generate_ratio
        self.repair_max_buggy_chars = repair_max_buggy_chars
        self.duplicate_tasks = duplicate_tasks
        # For lazy loading in pipeline mode
        self._lazy = pipeline_mode or dual_jsonl_mode
        self._draft_dir = Path(draft_dir) if draft_dir else None
        self._refine_dir = Path(refine_dir) if refine_dir else None
        self._image_dir = Path(image_dir) if image_dir else None
        # COT directories: new separate dirs take priority over legacy single dir
        self._cot_generate_dir = Path(cot_generate_dir) if cot_generate_dir else (Path(cot_dir) / "generate" if cot_dir else None)
        self._cot_repair_dir = Path(cot_repair_dir) if cot_repair_dir else (Path(cot_dir) / "repair" if cot_dir else None)

        self.samples: list[dict] = []
        if dual_jsonl_mode:
            self._load_from_dual_pipeline(
                spec_jsonl_generate, spec_jsonl_refine,
                refine_dir, draft_dir, image_dir,
            )
        elif pipeline_mode:
            self._load_from_pipeline(spec_jsonl, refine_dir, draft_dir, image_dir)
        elif dir_path is not None:
            self._load_from_dir(dir_path)
        else:
            self._load_from_jsonl(jsonl_path)

        if not dual_jsonl_mode:
            self._assign_task_types()

        if self.samples:
            from collections import Counter

            difficulty_dist = Counter(sample.get("difficulty", "?") for sample in self.samples)
            task_dist = Counter(sample["task"] for sample in self.samples)
            logger.info(
                "SFTDataset: %d samples. Task: %s Difficulty: %s",
                len(self.samples),
                dict(task_dist),
                dict(sorted(difficulty_dist.items())),
            )

    def _load_from_pipeline(
        self,
        spec_jsonl: str,
        refine_dir: Optional[str],
        draft_dir: Optional[str],
        image_dir: Optional[str],
    ) -> None:
        """Load from pipeline directories — fast, no packing step needed.

        Indexes by whichever HTML dir is the 'clean' source (refine_dir if given,
        else draft_dir).  Only samples with a matching clean HTML are kept.
        HTML content is loaded lazily in __getitem__.
        """
        clean_dir = Path(refine_dir) if refine_dir else Path(draft_dir) if draft_dir else None
        if clean_dir is None or not clean_dir.is_dir():
            raise ValueError("Pipeline mode requires at least one of refine_dir or draft_dir.")

        # 1) Index existing clean HTML files.
        #    Use a shared cache file so only one process pays the FUSE listdir cost.
        import hashlib
        _cache_key = hashlib.md5(str(clean_dir).encode()).hexdigest()[:12]
        _cache_path = Path(f"/tmp/_sft_listing_cache_{_cache_key}.json")
        available_ids: dict[str, Path] = {}

        if _cache_path.exists():
            try:
                _cached = json.loads(_cache_path.read_text())
                available_ids = {k: Path(v) for k, v in _cached.items()}
                logger.info("Pipeline mode: loaded %d cached HTML listings from %s", len(available_ids), _cache_path)
            except Exception:
                available_ids = {}

        if not available_ids:
            for fname_str in os.listdir(str(clean_dir)):
                if fname_str.endswith(".html"):
                    available_ids[fname_str[:-5]] = clean_dir / fname_str
            # Write cache for other ranks
            try:
                _cache_path.write_text(json.dumps({k: str(v) for k, v in available_ids.items()}))
            except Exception:
                pass
            logger.info("Pipeline mode: found %d clean HTML files in %s", len(available_ids), clean_dir)

        # 2) Build spec lookup — only load specs that have clean HTML
        #    First pass: read all spec ids without full parse to filter fast
        spec_lookup: dict[str, dict] = {}
        with open(spec_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Quick id extraction to avoid full JSON parse for 19K entries
                # when only ~268 are needed
                obj = json.loads(line)
                sid = str(obj.get("id", ""))
                if sid in available_ids:
                    spec_lookup[sid] = obj
        logger.info("Pipeline mode: matched %d specs", len(spec_lookup))

        # 3) Index COT files from separate generate/repair directories
        #    Each dir contains flat {id}.json files with {"id": ..., "cot": ...}
        #    Use pre-packed JSON cache if available to avoid 30K+ FUSE reads
        cot_lookup: dict[str, dict[str, str]] = {}  # sid -> {"generate": cot, "repair": cot}
        _cot_packed = Path("/tmp/_sft_cot_packed.json")
        if _cot_packed.exists():
            try:
                cot_lookup = json.loads(_cot_packed.read_text(encoding="utf-8"))
                logger.info("Pipeline mode: loaded %d COT entries from packed cache", len(cot_lookup))
            except Exception:
                cot_lookup = {}
        if not cot_lookup:
            cot_dirs_to_scan: list[tuple[str, Path | None]] = [
                ("generate", self._cot_generate_dir),
                ("repair", self._cot_repair_dir),
            ]
            for task_name, cot_path in cot_dirs_to_scan:
                if cot_path is None or not cot_path.is_dir():
                    continue
                count = 0
                for fname in cot_path.iterdir():
                    if fname.suffix == ".json":
                        try:
                            cot_data = json.loads(fname.read_text(encoding="utf-8"))
                            cot_text = cot_data.get("cot", "")
                            if cot_text:
                                sid_key = fname.stem
                                cot_lookup.setdefault(sid_key, {})[task_name] = cot_text
                                count += 1
                        except Exception:
                            pass
                logger.info("Pipeline mode: loaded %d COT-%s entries from %s", count, task_name, cot_path)
            # Write packed cache so other ranks (and restarts) skip the slow scan
            if cot_lookup:
                try:
                    _cot_packed.write_text(json.dumps(cot_lookup, ensure_ascii=False), encoding="utf-8")
                    logger.info("Pipeline mode: wrote COT packed cache (%d entries) to %s", len(cot_lookup), _cot_packed)
                except Exception:
                    pass
                logger.info("Pipeline mode: loaded %d COT-%s entries from %s", count, task_name, cot_path)

        # 4) Build samples (metadata only — HTML is loaded lazily)
        img_dir = Path(image_dir) if image_dir else None
        _draft_dir = Path(draft_dir) if draft_dir else None
        has_draft = _draft_dir is not None and _draft_dir.is_dir()

        # Pre-index image files using cache to avoid concurrent FUSE listdir
        def _cached_listdir(dir_path: str) -> set[str]:
            """os.listdir with /tmp JSON cache to survive multi-rank FUSE contention."""
            ck = hashlib.md5(dir_path.encode()).hexdigest()[:12]
            cp = Path(f"/tmp/_sft_listing_cache_{ck}.json")
            if cp.exists():
                try:
                    return set(json.loads(cp.read_text()).keys())
                except Exception:
                    pass
            names = os.listdir(dir_path)
            stems = {Path(n).stem: os.path.join(dir_path, n) for n in names}
            try:
                cp.write_text(json.dumps(stems))
            except Exception:
                pass
            return set(stems.keys())

        _img_stems: set[str] = set()
        if img_dir and img_dir.is_dir():
            _img_stems = _cached_listdir(str(img_dir))

        # Implicit prev-render discovery: when refine-with-prev-render is on,
        # look for a sibling "draft_render" directory (under final/).
        _prev_render_dir: Path | None = None
        _prev_render_stems: set[str] = set()
        if refine_use_prev_render() and _draft_dir is not None:
            cand = _draft_dir.parent / "draft_render"
            if cand.is_dir():
                _prev_render_dir = cand
                _prev_render_stems = _cached_listdir(str(cand))
                logger.info(
                    "Pipeline mode: prev_render dir resolved to %s (%d files)",
                    cand, len(_prev_render_stems),
                )
            else:
                logger.warning(
                    "REFINE_USE_PREV_RENDER is on but no prev_render dir found near %s "
                    "(tried draft_render). Repair samples "
                    "will fall back to single-image refine.",
                    _draft_dir,
                )

        # If COT directories were provided, only keep samples that have at least
        # a generate COT — the COT set defines the SFT training subset (e.g. 15K)
        # while refine_dir may contain extra samples used for RL/eval.
        _filter_by_cot = False  # disabled: train without COT

        for sid, clean_path in sorted(available_ids.items()):
            spec = spec_lookup.get(sid)
            if spec is None:
                continue

            difficulty = spec.get("difficulty", 2)
            if not self._keep(difficulty):
                continue

            # Resolve COT: prefer cot_dir lookup, fallback to spec field
            sample_cots = cot_lookup.get(sid, {})

            # When COT dirs are provided, skip samples without COT
            if _filter_by_cot and not sample_cots:
                continue

            # Resolve image path (using pre-indexed set, no per-sample stat)
            target_image = ""
            if img_dir and sid in _img_stems:
                target_image = str(img_dir / f"{sid}.png")
            if not target_image:
                ref_meta = spec.get("reference_metadata") or {}
                target_image = ref_meta.get("reference_image_path", "")

            # Resolve prev_render path (only meaningful when feature flag is on
            # AND a draft HTML exists for this sample; loaded lazily in __getitem__).
            _prev_render_path: str | None = None
            if (
                _prev_render_dir is not None
                and has_draft
                and sid in _prev_render_stems
            ):
                _prev_render_path = str(_prev_render_dir / f"{sid}.png")

            self.samples.append({
                "id": sid,
                "difficulty": difficulty,
                "target_image": target_image,
                # Lazy — store paths, load content in __getitem__
                "_clean_html_path": str(clean_path),
                "_buggy_html_path": str(_draft_dir / f"{sid}.html") if has_draft else None,
                "_prev_render_path": _prev_render_path,
                "buggy_html": None,  # placeholder, loaded lazily
                "clean_html": None,  # placeholder, loaded lazily
                "cot": spec.get("cot"),  # fallback from spec (usually None)
                "_cot_generate": sample_cots.get("generate"),
                "_cot_repair": sample_cots.get("repair"),
                "task": None,
                "generation_prompt": spec.get("generation_prompt", ""),
                "reference_metadata": spec.get("reference_metadata") or {},
                "verifiable_requirements": spec.get("verifiable_requirements") or [],
            })

        logger.info("Pipeline mode: %d samples ready", len(self.samples))

    def _load_from_dual_pipeline(
        self,
        spec_jsonl_generate: Optional[str],
        spec_jsonl_refine: Optional[str],
        refine_dir: Optional[str],
        draft_dir: Optional[str],
        image_dir: Optional[str],
    ) -> None:
        """Load from two separate spec JSONLs with pre-determined task types.

        Generate samples come from spec_jsonl_generate (task="generate"),
        refine samples come from spec_jsonl_refine (task="repair").
        Both use refine_dir for clean HTML; refine samples also use draft_dir
        for buggy HTML. No random task assignment — tasks are deterministic.
        """
        import hashlib

        clean_dir = Path(refine_dir) if refine_dir else None
        if clean_dir is None or not clean_dir.is_dir():
            raise ValueError("Dual-jsonl mode requires refine_dir.")

        _draft_dir = Path(draft_dir) if draft_dir else None
        img_dir = Path(image_dir) if image_dir else None

        # Reuse the same caching infrastructure as _load_from_pipeline
        def _cached_listdir_map(dir_path: str) -> dict[str, Path]:
            ck = hashlib.md5(dir_path.encode()).hexdigest()[:12]
            cp = Path(f"/tmp/_sft_listing_cache_{ck}.json")
            if cp.exists():
                try:
                    return {k: Path(v) for k, v in json.loads(cp.read_text()).items()}
                except Exception:
                    pass
            result = {}
            for fname in os.listdir(dir_path):
                if fname.endswith(".html"):
                    result[fname[:-5]] = Path(dir_path) / fname
            try:
                cp.write_text(json.dumps({k: str(v) for k, v in result.items()}))
            except Exception:
                pass
            return result

        def _cached_listdir_stems(dir_path: str) -> set[str]:
            ck = hashlib.md5(dir_path.encode()).hexdigest()[:12]
            cp = Path(f"/tmp/_sft_listing_cache_{ck}.json")
            if cp.exists():
                try:
                    return set(json.loads(cp.read_text()).keys())
                except Exception:
                    pass
            names = os.listdir(dir_path)
            stems = {Path(n).stem: os.path.join(dir_path, n) for n in names}
            try:
                cp.write_text(json.dumps(stems))
            except Exception:
                pass
            return set(stems.keys())

        # Index clean HTML files
        available_ids = _cached_listdir_map(str(clean_dir))
        logger.info("Dual-jsonl mode: found %d clean HTML files in %s", len(available_ids), clean_dir)

        # Index image files
        _img_stems: set[str] = set()
        if img_dir and img_dir.is_dir():
            _img_stems = _cached_listdir_stems(str(img_dir))

        # Index COT files (same logic as _load_from_pipeline)
        cot_lookup: dict[str, dict[str, str]] = {}
        _cot_packed = Path("/tmp/_sft_cot_packed.json")
        if _cot_packed.exists():
            try:
                cot_lookup = json.loads(_cot_packed.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not cot_lookup:
            for task_name, cot_path in [("generate", self._cot_generate_dir), ("repair", self._cot_repair_dir)]:
                if cot_path is None or not cot_path.is_dir():
                    continue
                count = 0
                for fname in cot_path.iterdir():
                    if fname.suffix == ".json":
                        try:
                            cot_data = json.loads(fname.read_text(encoding="utf-8"))
                            cot_text = cot_data.get("cot", "")
                            if cot_text:
                                cot_lookup.setdefault(fname.stem, {})[task_name] = cot_text
                                count += 1
                        except Exception:
                            pass
                logger.info("Dual-jsonl mode: loaded %d COT-%s entries from %s", count, task_name, cot_path)
            if cot_lookup:
                try:
                    _cot_packed.write_text(json.dumps(cot_lookup, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

        # Prev-render discovery (same as _load_from_pipeline)
        _prev_render_dir: Path | None = None
        _prev_render_stems: set[str] = set()
        if refine_use_prev_render() and _draft_dir is not None:
            cand = _draft_dir.parent / "draft_render"
            if cand.is_dir():
                _prev_render_dir = cand
                _prev_render_stems = _cached_listdir_stems(str(cand))
                logger.info("Dual-jsonl mode: prev_render dir %s (%d files)", cand, len(_prev_render_stems))

        has_draft = _draft_dir is not None and _draft_dir.is_dir()

        def _build_sample(sid: str, spec: dict, task: str) -> dict | None:
            clean_path = available_ids.get(sid)
            if clean_path is None:
                return None
            difficulty = spec.get("difficulty", 2)
            if not self._keep(difficulty):
                return None

            sample_cots = cot_lookup.get(sid, {})

            # Resolve image
            target_image = ""
            if img_dir and sid in _img_stems:
                target_image = str(img_dir / f"{sid}.png")
            if not target_image:
                ref_meta = spec.get("reference_metadata") or {}
                target_image = ref_meta.get("reference_image_path", "")

            # Prev render (only for repair)
            _prev_render_path: str | None = None
            if task == "repair" and _prev_render_dir is not None and has_draft and sid in _prev_render_stems:
                _prev_render_path = str(_prev_render_dir / f"{sid}.png")

            buggy_path = None
            if task == "repair" and has_draft:
                buggy_path = str(_draft_dir / f"{sid}.html")

            return {
                "id": sid,
                "difficulty": difficulty,
                "target_image": target_image,
                "_clean_html_path": str(clean_path),
                "_buggy_html_path": buggy_path,
                "_prev_render_path": _prev_render_path,
                "buggy_html": None,
                "clean_html": None,
                "cot": spec.get("cot"),
                "_cot_generate": sample_cots.get("generate"),
                "_cot_repair": sample_cots.get("repair"),
                "task": task,
                "generation_prompt": spec.get("generation_prompt", ""),
                "reference_metadata": spec.get("reference_metadata") or {},
                "verifiable_requirements": spec.get("verifiable_requirements") or [],
            }

        # Load generate samples
        if spec_jsonl_generate:
            with open(spec_jsonl_generate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    spec = json.loads(line)
                    sid = str(spec.get("id", ""))
                    sample = _build_sample(sid, spec, "generate")
                    if sample is not None:
                        self.samples.append(sample)
            logger.info("Dual-jsonl mode: %d generate samples from %s", len(self.samples), spec_jsonl_generate)

        # Load refine samples
        n_before = len(self.samples)
        if spec_jsonl_refine:
            if not has_draft:
                raise ValueError("Dual-jsonl mode with spec_jsonl_refine requires draft_dir.")
            with open(spec_jsonl_refine, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    spec = json.loads(line)
                    sid = str(spec.get("id", ""))
                    sample = _build_sample(sid, spec, "repair")
                    if sample is not None:
                        self.samples.append(sample)
            logger.info("Dual-jsonl mode: %d refine samples from %s", len(self.samples) - n_before, spec_jsonl_refine)

        # Shuffle all samples together with fixed seed
        import random
        random.Random(42).shuffle(self.samples)
        logger.info("Dual-jsonl mode: %d total samples ready", len(self.samples))

    def _load_on_demand(self, sample: dict) -> dict:
        """Load HTML content on first access (lazy loading for pipeline mode)."""
        if sample.get("clean_html") is None and sample.get("_clean_html_path"):
            path = Path(sample["_clean_html_path"])
            if path.exists():
                sample["clean_html"] = _strip_outer_fences(path.read_text(encoding="utf-8"))
            else:
                sample["clean_html"] = ""
        if sample.get("buggy_html") is None and sample.get("_buggy_html_path"):
            # Only load if task is repair
            if sample.get("task") == "repair":
                path = Path(sample["_buggy_html_path"])
                if path.exists():
                    sample["buggy_html"] = _strip_outer_fences(path.read_text(encoding="utf-8"))
        return sample

    def _keep(self, difficulty: int) -> bool:
        return self._allowed_levels is None or difficulty in self._allowed_levels

    def _resolve_buggy_existing(self) -> set[str] | None:
        """Pre-compute set of existing buggy HTML files to avoid per-sample
        Path.exists() calls which are extremely slow on FUSE / network mounts."""
        _buggy_dirs: set[str] = set()
        for sample in self.samples:
            p = sample.get("_buggy_html_path")
            if p:
                _buggy_dirs.add(str(Path(p).parent))
        if not _buggy_dirs:
            return None
        _buggy_existing: set[str] = set()
        for d in _buggy_dirs:
            try:
                import hashlib as _hl
                _ck = _hl.md5(d.encode()).hexdigest()[:12]
                _cp = Path(f"/tmp/_sft_listing_cache_{_ck}.json")
                if _cp.exists():
                    _cached = json.loads(_cp.read_text())
                    _buggy_existing.update(_cached.values())
                else:
                    _buggy_existing.update(
                        str(Path(d) / f) for f in os.listdir(d)
                    )
            except (OSError, Exception):
                pass
        return _buggy_existing

    def _has_buggy(self, sample: dict, _buggy_existing: set[str] | None) -> bool:
        if sample["buggy_html"] is not None:
            return True
        if sample.get("_buggy_html_path"):
            return _buggy_existing is not None and sample["_buggy_html_path"] in _buggy_existing
        return False

    def _assign_task_types(self) -> None:
        import random
        import copy

        rng = random.Random(42)
        _buggy_existing = self._resolve_buggy_existing()

        # When COT directories are provided, only assign a task if
        # the corresponding COT exists. Samples without any applicable
        # COT are dropped entirely — training with a stub placeholder
        # would introduce noise.
        _require_cot = False  # disabled: train without COT

        if self.duplicate_tasks:
            new_samples = []
            for sample in self.samples:
                has_buggy = self._has_buggy(sample, _buggy_existing)
                has_gen_cot = bool(sample.get("_cot_generate"))
                has_rep_cot = bool(sample.get("_cot_repair"))

                # Generate copy — only if generate COT exists (or COT not required)
                if has_gen_cot or not _require_cot:
                    gen_sample = copy.copy(sample)
                    gen_sample["task"] = "generate"
                    gen_sample["buggy_html"] = None
                    gen_sample["_buggy_html_path"] = None
                    new_samples.append(gen_sample)

                # Repair copy — only if buggy HTML + repair COT exist (or COT not required)
                if has_buggy and (has_rep_cot or not _require_cot):
                    rep_sample = copy.copy(sample)
                    rep_sample["task"] = "repair"
                    new_samples.append(rep_sample)

            rng.shuffle(new_samples)
            self.samples = new_samples
        else:
            kept = []
            # When generate_ratio >= 1.0 → generate-only mode (Phase 1 generate)
            # When generate_ratio <= 0.0 → repair-only mode (Phase 1 repair)
            repair_only = self.generate_ratio <= 0.0
            generate_only = self.generate_ratio >= 1.0

            for sample in self.samples:
                has_buggy = self._has_buggy(sample, _buggy_existing)
                has_gen_cot = bool(sample.get("_cot_generate"))
                has_rep_cot = bool(sample.get("_cot_repair"))

                if not has_buggy:
                    if repair_only:
                        # Repair-only mode but no buggy HTML → skip
                        continue
                    # Generate only — need generate COT
                    if _require_cot and not has_gen_cot:
                        continue
                    sample["task"] = "generate"
                elif generate_only:
                    # Generate-only mode — ignore buggy, use as generate
                    if _require_cot and not has_gen_cot:
                        continue
                    sample["task"] = "generate"
                    sample["buggy_html"] = None
                    sample["_buggy_html_path"] = None
                elif rng.random() < self.generate_ratio:
                    # Chose generate over repair — need generate COT
                    if _require_cot and not has_gen_cot:
                        # Fall back to repair if repair COT exists
                        if has_rep_cot:
                            sample["task"] = "repair"
                        else:
                            continue
                    else:
                        sample["task"] = "generate"
                        sample["buggy_html"] = None
                        sample["_buggy_html_path"] = None
                else:
                    # Repair — need repair COT
                    if _require_cot and not has_rep_cot:
                        # Fall back to generate if generate COT exists
                        if has_gen_cot and not repair_only:
                            sample["task"] = "generate"
                            sample["buggy_html"] = None
                            sample["_buggy_html_path"] = None
                        else:
                            continue
                    else:
                        sample["task"] = "repair"
                kept.append(sample)
            self.samples = kept

        if _require_cot:
            from collections import Counter
            task_dist = Counter(s["task"] for s in self.samples)
            logger.info(
                "COT-filtered task assignment: %d samples kept (generate=%d, repair=%d)",
                len(self.samples), task_dist.get("generate", 0), task_dist.get("repair", 0),
            )

    def _load_from_dir(self, dir_path: str) -> None:
        root = Path(dir_path)
        for sample_dir in sorted(root.iterdir()):
            if not sample_dir.is_dir():
                continue
            target_img = sample_dir / "target.png"
            buggy_html = sample_dir / "buggy.html"
            clean_html = sample_dir / "clean.html"
            cot_file = sample_dir / "cot.txt"
            diff_file = sample_dir / "difficulty.txt"
            if not (target_img.exists() and clean_html.exists()):
                continue
            difficulty = int(_read_text(diff_file)) if diff_file.exists() else 2
            if not self._keep(difficulty):
                continue
            self.samples.append(
                {
                    "id": sample_dir.name,
                    "difficulty": difficulty,
                    "target_image": str(target_img),
                    "buggy_html": _read_text(buggy_html) if buggy_html.exists() else None,
                    "clean_html": _strip_outer_fences(_read_text(clean_html)),
                    "cot": _read_text(cot_file) if cot_file.exists() else None,
                    "task": None,
                }
            )

    def _load_from_jsonl(self, jsonl_path: str) -> None:
        assert jsonl_path is not None
        with open(jsonl_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                difficulty = obj.get("difficulty", 2)
                if not self._keep(difficulty):
                    continue
                buggy = obj.get("buggy_html") or None
                self.samples.append(
                    {
                        "id": obj.get("id", ""),
                        "difficulty": difficulty,
                        "target_image": obj["target_image"],
                        "buggy_html": buggy,
                        "clean_html": _strip_outer_fences(obj["clean_html"]),
                        "cot": obj.get("cot"),
                        "task": None,
                        # Spec fields — used to build the user text via build_user_text()
                        # so SFT input format is identical to the inference pipeline.
                        "generation_prompt": obj.get("generation_prompt", ""),
                        "reference_metadata": obj.get("reference_metadata") or {},
                        "verifiable_requirements": obj.get("verifiable_requirements") or [],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if self._lazy:
            sample = self._load_on_demand(sample)
        try:
            image = _load_image(sample["target_image"])
            if sample["task"] == "repair":
                try:
                    return self._build_repair(sample, image)
                except ValueError as err:
                    if "No supervised tokens" not in str(err):
                        raise
                    logger.warning(
                        "Sample %s has no supervised tokens in REPAIR mode after truncation; falling back to GENERATE.",
                        sample.get("id", "<unknown>"),
                    )
                    return self._build_generate(sample, image)
            return self._build_generate(sample, image)
        except (ValueError, Exception) as err:
            if "No supervised tokens" in str(err):
                # Fall back to a nearby sample that fits within max_length
                logger.warning("Sample %s too long, falling back to next sample.", sample.get("id", "<unknown>"))
                return self.__getitem__((idx + 1) % len(self.samples))
            raise

    def _build_generate(self, sample: dict, image) -> dict:
        if sample.get("generation_prompt"):
            # Spec-format: user text matches inference pipeline exactly.
            user_text = build_user_text(sample, DEFAULT_USER_TEMPLATE, has_image=image is not None)
            # Replace [Output] with the canonical generate instruction (COT + fence).
            idx = user_text.find(OUTPUT_SECTION_MARKER)
            if idx >= 0:
                user_text = user_text[:idx]
            user_text += GENERATE_OUTPUT_INSTRUCTION
        else:
            # Legacy dir-format: no generation_prompt available.
            user_text = _LEGACY_USER_GENERATE
        assistant_text = "```html\n{clean_html}\n```".format(clean_html=sample["clean_html"])
        # Generate path: always single image (the target reference).
        return self._encode([image], user_text, assistant_text)

    def _build_repair(self, sample: dict, image) -> dict:
        buggy_html = sample["buggy_html"] or ""
        if self.repair_max_buggy_chars > 0 and len(buggy_html) > self.repair_max_buggy_chars:
            buggy_html = buggy_html[: self.repair_max_buggy_chars]

        # Decide whether this particular sample carries a prev-render image.
        # Only when (a) feature flag is on, AND (b) a prev_render PNG exists
        # for this sample. Per-sample fallback ensures missing renders don't
        # crash a whole batch — they just degrade to single-image refine.
        prev_render_img = None
        use_prev_render = refine_use_prev_render()
        prev_render_path = sample.get("_prev_render_path")
        if use_prev_render and prev_render_path:
            try:
                prev_render_img = _load_image(prev_render_path)
            except Exception as exc:
                logger.warning(
                    "Sample %s: prev_render load failed (%s); falling back to single-image refine.",
                    sample.get("id", "<unknown>"), exc,
                )
                prev_render_img = None

        with_prev_render_in_text = prev_render_img is not None

        if sample.get("generation_prompt"):
            user_text = build_refine_user_text(
                sample, buggy_html,
                has_image=image is not None,
                with_prev_render=with_prev_render_in_text,
            )
        else:
            # Legacy dir-format fallback.
            user_text = _LEGACY_USER_REPAIR.format(buggy_html=buggy_html)
        cot = sample.get("_cot_repair") or sample.get("cot")
        if cot:
            assistant_text = ASSISTANT_COT_HTML.format(cot=cot, clean_html=sample["clean_html"])
        else:
            raise ValueError(f"Sample {sample.get('id', '?')} has no repair COT — should have been filtered.")

        images: list = [image]
        if prev_render_img is not None:
            images.append(prev_render_img)
        return self._encode(images, user_text, assistant_text)

    def _encode(self, images: list, user_text: str, assistant_text: str) -> dict:
        """Encode messages with one or more images.

        The first image uses the dataset's default min/max pixel budget
        (typically the high-res target reference). Any additional images are
        treated as low-res prev-render images and capped at
        REFINE_PREV_RENDER_MAX_PIXELS (~252 vision tokens) to keep the
        prompt within budget.
        """
        from qwen_vl_utils import process_vision_info

        # Backward compat: accept a single PIL.Image as well.
        if not isinstance(images, list):
            images = [images]

        user_content: list[dict] = []
        for i, img in enumerate(images):
            if i == 0:
                user_content.append({
                    "type": "image",
                    "image": img,
                    "min_pixels": self.image_min_pixels,
                    "max_pixels": self.image_max_pixels,
                })
            else:
                # Auxiliary images (currently: prev_render). Keep them small.
                user_content.append({
                    "type": "image",
                    "image": img,
                    "min_pixels": REFINE_PREV_RENDER_MIN_PIXELS,
                    "max_pixels": REFINE_PREV_RENDER_MAX_PIXELS,
                })
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        labels = _build_labels(
            input_ids=input_ids,
            processor=self.processor,
            max_length=self.max_length,
        )

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
        if int((labels != -100).sum().item()) == 0:
            raise ValueError("No supervised tokens after truncation.")

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw
        return result


import numpy as np

# Cache for tokenized assistant header to avoid re-tokenizing every sample
_HEADER_IDS_CACHE: dict[int, np.ndarray] = {}


def _get_header_ids(processor) -> np.ndarray:
    tok_id = id(processor.tokenizer)
    if tok_id not in _HEADER_IDS_CACHE:
        assistant_header = "<|im_start|>assistant\n"
        ids = processor.tokenizer(
            assistant_header,
            add_special_tokens=False,
            return_tensors="np",
        )["input_ids"].squeeze(0)
        _HEADER_IDS_CACHE[tok_id] = ids
    return _HEADER_IDS_CACHE[tok_id]


def _build_labels(
    input_ids: torch.Tensor,
    processor,
    max_length: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    del max_length
    labels = input_ids.clone()
    header_np = _get_header_ids(processor)
    header_len = len(header_np)

    # Vectorized reverse search using numpy
    seq_np = input_ids.numpy()
    seq_len = len(seq_np)
    start_pos = -1

    if header_len <= seq_len:
        # Compare first element to narrow candidates, then verify full match
        first_val = header_np[0]
        candidates = np.where(seq_np == first_val)[0]
        # Reverse iterate candidates for last match
        for idx in candidates[::-1]:
            if idx + header_len <= seq_len and np.array_equal(
                seq_np[idx : idx + header_len], header_np
            ):
                start_pos = idx + header_len
                break

    if start_pos == -1:
        labels[:] = ignore_index
    else:
        labels[:start_pos] = ignore_index

    return labels


def _build_rope_index_single(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build 3D RoPE position_ids [3, seq_len] for a single (unbatched) sequence.

    This is a standalone extraction of Qwen3VLModel.get_rope_index logic
    so that PackingDataset can construct position_ids without holding a model ref.
    """
    seq_len = input_ids.shape[0]
    input_tokens = input_ids.tolist()

    if image_grid_thw is not None and image_grid_thw.numel() > 0:
        vision_start_indices = (input_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum().item())
    else:
        image_nums = 0

    if image_nums == 0:
        # Pure text: all 3 dims share the same sequential positions
        pos = torch.arange(seq_len, dtype=torch.long)
        return torch.stack([pos, pos, pos])  # [3, seq_len]

    llm_pos_ids_list: list[torch.Tensor] = []
    st = 0
    image_index = 0
    for _ in range(image_nums):
        ed = input_tokens.index(image_token_id, st)
        t, h, w = (
            image_grid_thw[image_index][0].item(),
            image_grid_thw[image_index][1].item(),
            image_grid_thw[image_index][2].item(),
        )
        image_index += 1
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        llm_grid_t = t

        text_len = ed - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(
            torch.stack([t_index, h_index, w_index]) + text_len + st_idx
        )
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < seq_len:
        st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
        text_len = seq_len - st
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    return torch.cat(llm_pos_ids_list, dim=1)  # [3, seq_len]


class PackingDataset(Dataset):
    """Wraps an SFTDataset to pack multiple samples into max_length sequences.

    Each __getitem__ returns a single packed sequence with:
      - input_ids [total_len]
      - labels [total_len]
      - position_ids [3, total_len] (each sub-seq starts from 0 → triggers FA2 varlen)
      - pixel_values, image_grid_thw (concatenated)
      - NO attention_mask (critical for FA2 varlen path)
    """

    def __init__(
        self,
        base_dataset: SFTDataset,
        max_length: int,
        image_token_id: int,
        vision_start_token_id: int,
        spatial_merge_size: int,
        pad_token_id: int,
    ):
        self.base = base_dataset
        self.max_length = max_length
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = spatial_merge_size
        self.pad_token_id = pad_token_id

        # Estimate token lengths without loading full samples.
        # Use a heuristic: text chars / 3.5 + vision tokens from image_grid_thw.
        # This avoids expensive image loading and tokenization for 38K samples.
        logger.info("PackingDataset: estimating sample lengths for bin packing...")
        self._sample_lengths: list[int] = []
        sms = spatial_merge_size
        for sample in self.base.samples:
            # Rough text length estimate from generation_prompt + clean_html path size
            text_chars = len(sample.get("generation_prompt", "")) + 8000  # avg HTML ~8K chars
            text_tokens = int(text_chars / 3.5)
            # Vision token count from image: assume standard image grid
            # Actual grid depends on image resolution, use conservative estimate
            img_tokens = int((base_dataset.image_max_pixels / (14 * 14)) / (sms * sms))
            est_len = min(text_tokens + img_tokens + 200, max_length)  # 200 for chat template overhead
            self._sample_lengths.append(est_len)

        # First-fit-decreasing bin packing
        self.bins: list[list[int]] = self._bin_pack()
        total_samples = sum(len(b) for b in self.bins)
        avg_per_bin = total_samples / len(self.bins) if self.bins else 0
        efficiency = sum(self._sample_lengths) / (len(self.bins) * self.max_length) if self.bins else 0
        logger.info(
            "PackingDataset: %d samples → %d packed sequences (avg %.1f samples/pack, %.1f%% est. efficiency)",
            total_samples, len(self.bins), avg_per_bin, efficiency * 100,
        )

    def _bin_pack(self) -> list[list[int]]:
        """First-fit-decreasing bin packing."""
        indexed = sorted(
            enumerate(self._sample_lengths), key=lambda x: x[1], reverse=True
        )
        bins: list[list[int]] = []
        bin_remaining: list[int] = []
        for idx, length in indexed:
            if length > self.max_length:
                # Sample too long even alone — keep it as a solo bin (will be truncated)
                bins.append([idx])
                bin_remaining.append(0)
                continue
            placed = False
            for b_i in range(len(bins)):
                if bin_remaining[b_i] >= length:
                    bins[b_i].append(idx)
                    bin_remaining[b_i] -= length
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_remaining.append(self.max_length - length)
        return bins

    def __len__(self) -> int:
        return len(self.bins)

    def __getitem__(self, idx: int) -> dict:
        sample_indices = self.bins[idx]

        all_input_ids = []
        all_labels = []
        all_position_ids = []
        all_pixel_values = []
        all_image_grid_thw = []
        total_len = 0

        for s_idx in sample_indices:
            try:
                sample = self.base[s_idx]
            except Exception:
                continue
            ids = sample["input_ids"]
            # Stop packing if adding this sample would exceed max_length
            if total_len + ids.shape[0] > self.max_length and total_len > 0:
                break
            labs = sample["labels"]
            pv = sample.get("pixel_values")
            igthw = sample.get("image_grid_thw")

            # Build per-subsequence position_ids [3, seq_len]
            pos_ids = _build_rope_index_single(
                ids, igthw,
                self.image_token_id,
                self.vision_start_token_id,
                self.spatial_merge_size,
            )
            all_input_ids.append(ids)
            all_labels.append(labs)
            all_position_ids.append(pos_ids)
            total_len += ids.shape[0]
            if pv is not None:
                all_pixel_values.append(pv)
            if igthw is not None:
                all_image_grid_thw.append(igthw)

        if not all_input_ids:
            # Fallback: return a single sample
            sample = self.base[sample_indices[0]]
            all_input_ids.append(sample["input_ids"])
            all_labels.append(sample["labels"])
            pos_ids = _build_rope_index_single(
                sample["input_ids"], sample.get("image_grid_thw"),
                self.image_token_id, self.vision_start_token_id, self.spatial_merge_size,
            )
            all_position_ids.append(pos_ids)
            total_len = sample["input_ids"].shape[0]
            if sample.get("pixel_values") is not None:
                all_pixel_values.append(sample["pixel_values"])
            if sample.get("image_grid_thw") is not None:
                all_image_grid_thw.append(sample["image_grid_thw"])

        # Concatenate
        input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.cat(all_labels, dim=0)
        position_ids = torch.cat(all_position_ids, dim=1)  # [3, total_len]

        # Truncate to max_length
        total_len = input_ids.shape[0]
        if total_len > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            position_ids = position_ids[:, : self.max_length]
            total_len = self.max_length

        # Pad to max_length for uniform batch stacking
        pad_len = self.max_length - total_len
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            # Pad position_ids: continue incrementing from last position (doesn't matter, will be masked)
            pad_pos = torch.zeros(3, pad_len, dtype=torch.long)
            position_ids = torch.cat([position_ids, pad_pos], dim=1)

        result: dict = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }
        if all_pixel_values:
            result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            result["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
        return result


class PackingCollator:
    """Simple collator for PackingDataset — just stacks pre-packed sequences."""

    def __call__(self, batch: list[dict]) -> dict:
        # position_ids per sample: [3, seq_len] → stack → [bs, 3, seq_len] → permute → [3, bs, seq_len]
        output = {
            "input_ids": torch.stack([s["input_ids"] for s in batch]),
            "labels": torch.stack([s["labels"] for s in batch]),
            "position_ids": torch.stack([s["position_ids"] for s in batch]).permute(1, 0, 2),
        }
        if "pixel_values" in batch[0]:
            output["pixel_values"] = torch.cat(
                [s["pixel_values"] for s in batch if "pixel_values" in s], dim=0,
            )
        if "image_grid_thw" in batch[0]:
            output["image_grid_thw"] = torch.cat(
                [s["image_grid_thw"] for s in batch if "image_grid_thw" in s], dim=0,
            )
        return output


class SFTDataCollator:
    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(sample["input_ids"].shape[0] for sample in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for sample in batch:
            seq_len = sample["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids_list.append(
                torch.cat(
                    [
                        sample["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_mask_list.append(
                torch.cat(
                    [sample["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
                )
            )
            labels_list.append(
                torch.cat(
                    [
                        sample["labels"],
                        torch.full((pad_len,), self.ignore_index, dtype=torch.long),
                    ]
                )
            )

        output = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
        if "pixel_values" in batch[0]:
            output["pixel_values"] = torch.cat(
                [sample["pixel_values"] for sample in batch if "pixel_values" in sample],
                dim=0,
            )
        if "image_grid_thw" in batch[0]:
            output["image_grid_thw"] = torch.cat(
                [sample["image_grid_thw"] for sample in batch if "image_grid_thw" in sample],
                dim=0,
            )
        return output