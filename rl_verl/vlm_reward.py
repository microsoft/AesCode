"""Unified VLM visual assessment for DesignValidator — eval, generate, refine.

Three modes corresponding to DesignValidator's three scenarios:

1. **Eval** (single sample): Reference + 1 render → fidelity score 0-5 → [0,1].
   Used by eval pipeline (eval_vllm_validate.py).

2. **Generate** (RL group): Reference + N renders → Swiss pairwise tournament
   → ranking scores [0,1].  Used by reward_fn.py generate sub-step.

3. **Refine** (RL pairwise): Reference + prev_render + cur_render → 3-level
   verdict {A_better, same, B_better} → delta in [-1, +1].
   Used by reward_fn.py refine sub-step.

Configuration via environment variables:
  VLM_REWARD_MODE:        render_only | render_ref | eval (default render_only)
    VLM_REWARD_MODEL:       model name (default gpt-5.4-mini)
  VLM_REWARD_CONCURRENCY: max parallel API calls (default 16)
  VLM_REWARD_TIMEOUT:     per-request timeout in seconds (default 60)
    VLM_REWARD_BASE_URL:    OpenAI-compatible API base URL (preferred)
    VLM_REWARD_API_KEY:     OpenAI-compatible API key (preferred)

Enable/disable is controlled by the unified ENABLED_VALIDATORS /
DISABLED_VALIDATORS mechanism in reward_fn.py — no separate toggle here.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading for Ray workers (they don't go through _bootstrap)
# ---------------------------------------------------------------------------

_dotenv_loaded = False

def _ensure_dotenv():
    """Load .env file if not already loaded. Safe to call multiple times."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _get_mode() -> str:
    return os.environ.get("VLM_REWARD_MODE", "render_only")

def _get_model() -> str:
    return os.environ.get("VLM_REWARD_MODEL", "gpt-5.4-mini")

def _get_concurrency() -> int:
    return int(os.environ.get("VLM_REWARD_CONCURRENCY", "16"))

def _get_timeout() -> float:
    return float(os.environ.get("VLM_REWARD_TIMEOUT", "60"))

def _get_rpm_per_worker() -> float:
    """Per-endpoint request-rate cap, measured in requests/minute, applied
    *inside this worker process*. Total cluster RPM hitting one endpoint
    is roughly: rpm_per_worker * num_workers / num_endpoints.

    Tune via env VLM_RPM_PER_WORKER. Default 20 ⇒ with 16 Ray workers and
    2 endpoints, each endpoint sees ~160 RPM cluster-wide (well under the
    Azure default 500-1000 RPM per gpt-4o deployment).
    """
    return float(os.environ.get("VLM_RPM_PER_WORKER", "20"))

# ---------------------------------------------------------------------------
# Image encoding (with resize + caching for tournament efficiency)
# ---------------------------------------------------------------------------

# Per-mode image quality settings.
# Eval uses higher resolution for accurate VLM judgements;
# generate/refine (RL) uses lower resolution for speed.
_IMG_SETTINGS = {
    "eval":        {"max_dim": 1024, "detail": "high"},
    "generate":    {"max_dim": 768,  "detail": "low"},
    "refine":      {"max_dim": 768,  "detail": "low"},
    "render_only": {"max_dim": 768,  "detail": "low"},
    "render_ref":  {"max_dim": 768,  "detail": "low"},
    # Rubric scoring needs higher fidelity to read chart types, alignment,
    # element counts, and small layout details from the render-only screenshot.
    # Matched to eval quality; can be overridden via VLM_IMG_MAX_DIM/VLM_DETAIL.
    "rubric":      {"max_dim": 1024, "detail": "high"},
}

# Legacy module-level defaults (used when mode is unknown)
IMG_MAX_DIM = 768
DETAIL = "low"


def _get_img_settings(mode: str | None = None) -> tuple[int, str]:
    """Return (max_dim, detail) for the given mode, with env var override."""
    if mode is None:
        mode = _get_mode()
    settings = _IMG_SETTINGS.get(mode, _IMG_SETTINGS["generate"])
    max_dim = int(os.environ.get("VLM_IMG_MAX_DIM", settings["max_dim"]))
    detail = os.environ.get("VLM_DETAIL", settings["detail"])
    return max_dim, detail


# Separate caches per (max_dim, detail) to avoid mixing resolutions
_img_cache: dict[tuple[int, str, str], dict] = {}


def _img_part(path: str | Path, mode: str | None = None) -> dict:
    """Encode image as base64 with mode-aware resize and detail level.

    Eval mode: resize to ≤2048, detail=high (precise VLM analysis).
    RL modes: resize to ≤768, detail=low (~85 tokens/image, fast).
    Override via VLM_IMG_MAX_DIM / VLM_DETAIL env vars.

    Results are cached by (path, max_dim, detail) to avoid redundant work.
    """
    max_dim, detail = _get_img_settings(mode)
    path_str = str(path)
    cache_key = (path_str, max_dim, detail)
    if cache_key in _img_cache:
        return _img_cache[cache_key]

    try:
        from PIL import Image
    except ImportError:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        ext = Path(path).suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
        result = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}
        _img_cache[cache_key] = result
        return result

    img = Image.open(path)
    ratio = min(max_dim / img.width, max_dim / img.height)
    if ratio < 1:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    # JPEG is ~10x smaller than PNG for infographics → less bandwidth, fewer 403s
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=60)
    b64 = base64.b64encode(buf.getvalue()).decode()
    result = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}}
    _img_cache[cache_key] = result
    return result


def clear_img_cache():
    """Clear the image encoding cache between batches to prevent memory growth."""
    _img_cache.clear()

# ---------------------------------------------------------------------------
# Prompts — three modes
# ---------------------------------------------------------------------------

# Mode: render_only (legacy, single render without reference)
_PROMPT_RENDER_ONLY = """\
You are a visual quality inspector. Do NOT read text content. Just glance at this infographic like a normal person would.

Ignore: Gray/colored placeholder blocks are intentional (pure code model cannot generate real images). Do NOT penalize placeholders.

Evaluate two things:
1. Element placement (50%): Are boxes, cards, title blocks overlapping, misaligned, or crammed together?
2. Design sense (50%): Does it look polished at a glance — clear hierarchy, balanced spacing, visually pleasing — or rough like a draft?

0-3: Multiple overlapping/occluded elements, clearly broken
4-5: Some misalignment/overlap, or overall very rough with no design sense
6-7: Mostly tidy, average design sense
8-9: Clean layout with good design sense, looks comfortable
10: Perfect

Format:
Issue: <one sentence>
Score: <integer 0-10>"""

# Mode: render_ref (RL, single render + reference)
_PROMPT_RENDER_REF = """\
First image is the reference design, second is the code-generated render. Do NOT read text content. Just glance and evaluate the render.

Ignore: Gray/colored placeholder blocks replacing real images are expected. Do NOT penalize them. Minor differences in fonts, colors, or exact wording are also fine.

Evaluate three things:
1. Element integrity (30%): Are any elements overlapping, occluded, cropped off-screen, or misaligned with each other?
2. Layout fidelity (40%): Does the overall spatial arrangement follow the reference — similar section grouping, relative positions of major blocks (title, charts, sidebars), and general proportions? It does NOT need to be pixel-perfect, just structurally similar.
3. Design sense (30%): At a glance, does it look polished — clear hierarchy, balanced whitespace, visually comfortable — or rough like a draft?

0-3: Broken — major overlaps, elements cut off, or layout completely different from reference
4-5: Noticeable issues — some overlap/misalignment, or spatial structure clearly deviates from reference
6-7: Decent — mostly clean, layout roughly follows reference with minor deviations
8-9: Good — clean layout, spatial structure matches reference well, polished appearance
10: Excellent — no visible issues, layout faithfully reflects reference structure

Format:
Issue: <one sentence>
Score: <integer 0-10>"""

# Mode: eval (evaluation, single render + oracle reference — fidelity assessment, 0-5 scale)
_PROMPT_EVAL_FIDELITY = """\
First image is the oracle reference design, second image is the code-generated HTML render.

Context: The render was produced by an LLM that sees the reference image as visual guidance only. Gray/colored placeholder blocks replacing real photographs are expected — do NOT penalize them. Minor differences in fonts, exact colors, or decorative text are acceptable.

Evaluate these dimensions holistically:
- Layout fidelity: Does the overall spatial structure match — section grouping, relative positions of major blocks (title, charts, sidebars, footer), and general proportions?
- Element completeness: Are all major visual elements from the reference present? Are any key sections missing, duplicated, or severely misplaced?
- Overlap & clipping: Are elements overlapping, occluded, or cropped off the canvas edge?
- Design quality: At a glance, does it look polished — clear hierarchy, balanced whitespace, consistent styling — or rough like a draft?

Score on a 0-5 integer scale:
0 — Completely broken: layout bears no resemblance to reference, most elements missing or entirely overlapping. Unusable.
1 — Major issues: layout structure is vaguely recognizable but heavily distorted. Multiple sections missing, severely misplaced, or overlapping. Looks like a failed attempt.
2 — Significant deviations: some structural similarity to the reference, but with notable layout errors — wrong proportions, misplaced sections, moderate overlaps, or several missing elements. Rough overall appearance.
3 — Decent reproduction: layout broadly follows the reference with correct major sections in approximately right positions. Minor issues remain — small overlaps, slight misalignment, one or two missing secondary elements. Acceptable but clearly imperfect.
4 — Good fidelity: layout matches the reference well. All major and most secondary elements present and correctly positioned. Clean appearance with only very minor deviations (e.g., slight spacing differences, minor styling inconsistencies). Looks polished.
5 — Excellent: faithful reproduction of the reference structure. All elements present, correctly positioned, no overlaps or clipping. Professional finish with consistent styling and balanced whitespace.

Format:
Issue: <one sentence>
Score: <integer 0-5>"""

# Mode: generate (RL Swiss tournament, pairwise comparison)
_PROMPT_SWISS_PAIRWISE = """\
You are judging two HTML renders against a reference design.

Image 1 = REFERENCE (high-quality design inspiration, NOT a pixel-perfect target).
Image 2 = render A.
Image 3 = render B.

A and B are HTML renders; gray rectangles in A/B are intentional image
placeholders — treat them as neutral, do NOT penalize.

Compare A and B by how well each captures the REFERENCE's overall design
(layout, visual hierarchy, structural balance, element completeness).
- Ignore text accuracy.
- Ignore the actual content inside placeholder rectangles.
- HEAVILY penalize: significant element overlap, broken/clipped components,
  missing major elements, or large unintended blank regions.
- You MUST pick a winner — even when close, choose whichever is marginally better.

Reply in EXACTLY this format (one line each):
Reason: <≤25 words: first name 2-3 key REFERENCE elements, then say which side matches them better>
Better: A or B, Margin: large or small"""

# Mode: refine (RL pairwise delta, prev vs current)
_PROMPT_REFINE_PAIRWISE = """\
You are judging two candidate HTML renders against a reference design.

Image 1 = REFERENCE (high-quality design inspiration, NOT a pixel-perfect target).
Image 2 = candidate A.
Image 3 = candidate B.

A and B are HTML renders; gray rectangles are intentional image placeholders —
treat them as neutral, do NOT penalize.

Compare A and B by how well each captures the REFERENCE's overall design
(layout, visual hierarchy, structural balance, element completeness).
- Ignore text accuracy.
- Ignore the actual content inside placeholder rectangles.
- HEAVILY penalize: significant element overlap, broken/clipped components,
  missing major elements, or large unintended blank regions.
- Use 'same' ONLY when truly indistinguishable; prefer A_better/B_better
  whenever you can detect any meaningful difference.

Reply in EXACTLY this format (one line each):
Reason: <≤25 words: first name 2-3 key REFERENCE elements, then say which side matches them better>
Verdict: A_better, same, or B_better"""

# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_score_response(text: str, max_score: int = 10) -> tuple[float, str]:
    """Extract score and rationale from VLM response → [0,1].

    Args:
        text: Raw VLM response text.
        max_score: Scale upper bound (5 for eval mode, 10 for render_only/render_ref).
    """
    rationale = ""
    rat_match = re.search(r"(?:问题|Issue)[：:]\s*(.+)", text)
    if rat_match:
        rationale = rat_match.group(1).strip()

    score_match = re.search(r"(?:分数|Score)[：:]\s*(\d{1,2})", text)
    if not score_match:
        score_match = re.search(r"\b(\d{1,2})\b", text.strip())

    if score_match:
        raw = int(score_match.group(1))
        return min(max(raw / float(max_score), 0.0), 1.0), rationale

    logger.warning("Could not parse VLM score from response: %r", text[:200])
    return 0.0, rationale

# ---------------------------------------------------------------------------
# Multi-endpoint round-robin support
# ---------------------------------------------------------------------------

def _is_azure_endpoint(base_url: str) -> bool:
    s = (base_url or "").lower()
    return (
        ".openai.azure.com" in s
        or "cognitiveservices.azure.com" in s
    )


def _load_all_endpoints() -> list:
    """Load OpenAI-compatible endpoint/key pairs from env vars.

    Preferred vars:
      - VLM_REWARD_BASE_URL / VLM_REWARD_API_KEY
      - VLM_REWARD_BASE_URL{i} / VLM_REWARD_API_KEY{i}

    Backward-compatible fallbacks:
      - AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY
      - ENDPOINT / API_KEY, ENDPOINT{i} / API_KEY{i}
    """
    _ensure_dotenv()
    pairs = []

    vlm_ep = os.environ.get("VLM_REWARD_BASE_URL", "")
    vlm_key = os.environ.get("VLM_REWARD_API_KEY", "")
    use_vlm_prefixed = bool(vlm_ep and vlm_key)

    if use_vlm_prefixed:
        # Strict mode for RL training: when VLM_REWARD_BASE_URL/API_KEY is set,
        # only use VLM_REWARD_BASE_URL{i}/API_KEY{i} peers. This avoids
        # accidentally mixing legacy ENDPOINT{i}/API_KEY{i} pools.
        pairs.append((vlm_ep, vlm_key))
        for i in range(1, 10):
            ep_i = os.environ.get(f"VLM_REWARD_BASE_URL{i}", "")
            key_i = os.environ.get(f"VLM_REWARD_API_KEY{i}", "")
            if ep_i and key_i:
                pairs.append((ep_i, key_i))
        return pairs

    # Backward-compatible legacy mode
    ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "") or os.environ.get("ENDPOINT", "")
    key = os.environ.get("AZURE_OPENAI_API_KEY", "") or os.environ.get("API_KEY", "")
    if ep and key:
        pairs.append((ep, key))
    for i in range(1, 10):
        ep_i = os.environ.get(f"ENDPOINT{i}", "")
        key_i = os.environ.get(f"API_KEY{i}", "")
        if ep_i and key_i:
            pairs.append((ep_i, key_i))
    return pairs


class _CleanEnvCliCredential:
    """AzureCliCredential replacement that forks the az CLI with a SANITIZED env.

    Why: the training worker leaks the verl (python3.10) site-packages via
    PYTHONPATH/PYTHONHOME into az. The az CLI ships its OWN python3.14; when it
    imports rpds it loads the 3.10-compiled rpds.cpython-310 .so into 3.14 and
    dies with "No module named rpds.rpds" (ABI mismatch) -> token fetch fails
    -> design VLM judge silently returns applicable=0 -> design collapses to 0.0.
    Stripping PYTHON*/u2c-PATH from the child env makes az use its bundled interp.
    """
    def get_token(self, *scopes, **kwargs):
        import subprocess, json
        from azure.core.credentials import AccessToken
        scope = scopes[0] if scopes else "https://cognitiveservices.azure.com/.default"
        resource = scope[:-len("/.default")] if scope.endswith("/.default") else scope
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
        env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        out = subprocess.run(
            ["/usr/bin/az", "account", "get-access-token", "--resource", resource,
             "--output", "json"],
            env=env, capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            raise RuntimeError("az get-access-token failed: " + out.stderr.strip()[:300])
        # az can print harmless noise (e.g. "No module named 'rpds.rpds'") to
        # STDOUT before the JSON body; slice from the first '{' so json.loads
        # does not choke on the prefix.
        raw = out.stdout
        brace = raw.find("{")
        if brace > 0:
            raw = raw[brace:]
        data = json.loads(raw)
        return AccessToken(data["accessToken"], int(data["expires_on"]))


def _get_azure_ad_token_provider():
    """Azure AD bearer token provider backed by _CleanEnvCliCredential."""
    from azure.identity import get_bearer_token_provider
    return get_bearer_token_provider(
        _CleanEnvCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

# Module-level singleton so all workers share one credential object.
_azure_ad_provider = None

def _ensure_azure_ad_provider():
    global _azure_ad_provider
    if _azure_ad_provider is None:
        _azure_ad_provider = _get_azure_ad_token_provider()
    return _azure_ad_provider


def _build_async_client(base_url: str, api_key: str, timeout: float):
    if _is_azure_endpoint(base_url):
        from openai import AsyncAzureOpenAI

        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        # Azure AD token via azure-identity — set api_key to "azure_ad" in .env.
        # Uses AzureCliCredential (reads ~/.azure/); the SDK caches and
        # auto-refreshes tokens internally — no manual refresh logic.
        if api_key == "azure_ad":
            return AsyncAzureOpenAI(
                azure_ad_token_provider=_ensure_azure_ad_provider(),
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
            )

        # Plain API key authentication.
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            timeout=timeout,
        )

    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

_endpoint_counter = 0
_endpoint_counter_lock = asyncio.Lock()

async def _next_endpoint_index(n: int) -> int:
    """Thread-safe round-robin counter."""
    global _endpoint_counter
    async with _endpoint_counter_lock:
        idx = _endpoint_counter % n
        _endpoint_counter += 1
        return idx

# ---------------------------------------------------------------------------
# Shared async client pool (one per endpoint, reused across groups)
# ---------------------------------------------------------------------------

_shared_clients: list = []
_shared_limiters: list = []  # one AsyncLimiter per endpoint, parallel to _shared_clients
_shared_client_lock = asyncio.Lock()

async def _get_shared_client(timeout: float):
    """Get or create a shared async OpenAI client (round-robin if multiple endpoints).

    Returns (client, limiter). The limiter is an aiolimiter.AsyncLimiter scoped
    to that one endpoint; callers should `async with limiter:` around the actual
    chat.completions.create. The limiter is shared across all calls within this
    worker process for that endpoint, which is what allows it to enforce a rate.
    """
    global _shared_clients, _shared_limiters
    async with _shared_client_lock:
        if not _shared_clients:
            try:
                from aiolimiter import AsyncLimiter
            except ImportError:
                AsyncLimiter = None  # graceful fallback: no rate limiting
                logger.warning("aiolimiter not installed; VLM rate-limit disabled.")
            pairs = _load_all_endpoints()
            if not pairs:
                logger.warning("VLM API credentials not set; VLM disabled.")
                return None, None
            rpm = _get_rpm_per_worker()
            for ep, key in pairs:
                try:
                    _shared_clients.append(_build_async_client(ep, key, timeout))
                except ImportError:
                    logger.warning("openai package not installed; VLM disabled.")
                    return None, None
                if AsyncLimiter is not None and rpm > 0:
                    _shared_limiters.append(AsyncLimiter(max_rate=rpm, time_period=60))
                else:
                    _shared_limiters.append(None)
            logger.info(
                "VLM: initialized %d endpoint client(s), rpm_per_worker=%.1f",
                len(_shared_clients), rpm,
            )
    if not _shared_clients:
        return None, None
    idx = await _next_endpoint_index(len(_shared_clients))
    return _shared_clients[idx], _shared_limiters[idx]

async def close_shared_client():
    """Close all shared clients (call at end of batch)."""
    global _shared_clients, _shared_limiters
    async with _shared_client_lock:
        for c in _shared_clients:
            try: await c.close()
            except Exception: pass
        _shared_clients = []
        _shared_limiters = []

# ---------------------------------------------------------------------------
# Eval mode: single-sample scoring (ref + render → 0-5 score → [0,1])
# ---------------------------------------------------------------------------

async def _score_one(
    client,
    semaphore: asyncio.Semaphore,
    render_image_path: str | Path,
    reference_image_path: str | Path | None,
    mode: str,
    model: str,
    max_retries: int = 2,
) -> tuple[float, str]:
    """Score a single rendered screenshot via the VLM API. Returns (score, rationale)."""
    async with semaphore:
        content_parts: list[dict] = []

        # In render_ref / eval mode, reference image comes first
        if mode in ("render_ref", "eval") and reference_image_path and Path(reference_image_path).exists():
            content_parts.append(_img_part(reference_image_path, mode=mode))

        content_parts.append(_img_part(render_image_path, mode=mode))

        if mode == "eval" and reference_image_path:
            prompt = _PROMPT_EVAL_FIDELITY
        elif mode == "render_ref" and reference_image_path:
            prompt = _PROMPT_RENDER_REF
        else:
            prompt = _PROMPT_RENDER_ONLY
        content_parts.append({"type": "text", "text": prompt})

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content_parts}],
                    max_completion_tokens=80,
                    temperature=0.0,
                )
                text = response.choices[0].message.content or ""
                return _parse_score_response(text, max_score=5 if mode == "eval" else 10)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                # Retry on rate-limit (429) or temporary block (403 Forbidden)
                if ("429" in exc_str or "403" in exc_str or "Forbidden" in exc_str) and attempt < max_retries:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.debug("VLM retry %d/%d for %s (%.1fs): %s",
                                 attempt + 1, max_retries, render_image_path, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                break
        logger.warning("VLM scoring failed for %s: %s", render_image_path, last_exc)
        return 0.0, f"error: {last_exc}"


async def vlm_score_batch(
    render_image_paths: list[str | Path],
    reference_image_paths: list[str | Path | None] | None = None,
    mode_override: str | None = None,
) -> tuple[list[float], list[str]]:
    """Score a batch of rendered screenshots via the VLM API.

    Returns (scores, rationales). Scores are float in [0, 1].
    """
    n = len(render_image_paths)
    _ensure_dotenv()

    if reference_image_paths is None:
        reference_image_paths = [None] * n

    mode = mode_override or _get_mode()
    model = _get_model()
    concurrency = _get_concurrency()
    timeout = _get_timeout()

    pairs = _load_all_endpoints()
    if not pairs:
        logger.warning("VLM API credentials not set; VLM reward disabled.")
        return [0.0] * n, [""] * n

    try:
        clients = [_build_async_client(ep, key, timeout) for ep, key in pairs]
    except ImportError:
        logger.warning("openai package not installed; VLM reward disabled.")
        return [0.0] * n, [""] * n
    semaphore = asyncio.Semaphore(concurrency)

    try:
        tasks = [
            _score_one(clients[i % len(clients)], semaphore, rp, ref, mode, model)
            for i, (rp, ref) in enumerate(zip(render_image_paths, reference_image_paths))
        ]
        results = await asyncio.gather(*tasks)
        scores = [r[0] for r in results]
        rationales = [r[1] for r in results]
        return scores, rationales
    finally:
        for c in clients:
            await c.close()


def vlm_score_batch_sync(
    render_image_paths: list[str | Path],
    reference_image_paths: list[str | Path | None] | None = None,
    mode_override: str | None = None,
) -> tuple[list[float], list[str]]:
    """Synchronous wrapper around :func:`vlm_score_batch`.

    Handles the case where an event loop is already running (e.g. inside
    multiprocessing workers) by creating a new loop in a dedicated thread.
    """
    coro = vlm_score_batch(render_image_paths, reference_image_paths, mode_override=mode_override)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Already inside an event loop — run in a new thread to avoid
        # "asyncio.run() cannot be called from a running event loop".
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)

# ---------------------------------------------------------------------------
# Generate mode: Swiss pairwise tournament (ref + N renders → ranking)
# ---------------------------------------------------------------------------

async def _swiss_compare_pair(
    client, sem: asyncio.Semaphore,
    ref_path: str, path_a: str, path_b: str,
    max_retries: int = 3,
    limiter=None,
) -> tuple[str | None, int]:
    """Single pairwise comparison. Returns (winner 'A'/'B'/None, margin 1 or 2)."""
    async with sem:
        # Order matters for Azure prompt caching: fixed prefix first.
        # [text(fixed) + ref(fixed per sample)] is cached across all
        # pairs in this sample's tournament; only render_A/B vary.
        content = [
            {"type": "text", "text": _PROMPT_SWISS_PAIRWISE},
            _img_part(ref_path, mode="generate"),
            _img_part(path_a, mode="generate"),
            _img_part(path_b, mode="generate"),
        ]
        text = ""
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                if limiter is not None:
                    async with limiter:
                        resp = await client.chat.completions.create(
                            model=_get_model(),
                            messages=[{"role": "user", "content": content}],
                            max_completion_tokens=200,
                            temperature=0.0,
                        )
                else:
                    resp = await client.chat.completions.create(
                        model=_get_model(),
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=200,
                        temperature=0.0,
                    )
                text = resp.choices[0].message.content or ""
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.debug("[DesignValidator] swiss pair retry %d/%d (%.1fs): %s",
                                 attempt + 1, max_retries, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                break
        if last_exc is not None:
            logger.warning("[DesignValidator] swiss pair API error after %d retries: %s",
                           max_retries, last_exc)
            return None, 0

    m = re.search(r"Better:\s*([AB])", text, re.IGNORECASE)
    winner = m.group(1).upper() if m else None
    mm = re.search(r"Margin:\s*(large|small)", text, re.IGNORECASE)
    margin = 2 if (mm and mm.group(1).lower() == "large") else 1
    return winner, margin


async def swiss_tournament_judge(
    ref_path: str,
    render_paths: list[str],
) -> list[Optional[float]]:
    """Score renders via Swiss pairwise tournament, return per-render [0,1] scores.

    Runs log2(n) rounds of Swiss pairwise comparisons. Each round pairs
    candidates with similar cumulative scores. Margin scoring (large=±2,
    small=±1) gives fine-grained ranking.

    None at index i = missing render → caller fills with group mean.
    """
    n = len(render_paths)
    fail = [None] * n

    _ensure_dotenv()
    client, limiter = await _get_shared_client(_get_timeout())
    if client is None:
        return fail

    if not ref_path or not Path(ref_path).exists():
        logger.warning("[DesignValidator] swiss: missing ref %s", ref_path)
        return fail

    valid_indices = [i for i in range(n) if render_paths[i] and Path(render_paths[i]).exists()]
    if len(valid_indices) < 2:
        logger.warning("[DesignValidator] swiss: <2 valid renders (%d/%d)", len(valid_indices), n)
        return fail

    valid_renders = [render_paths[i] for i in valid_indices]
    nv = len(valid_renders)
    num_rounds = max(1, math.ceil(math.log2(nv)))
    scores = [0] * nv
    sem = asyncio.Semaphore(_get_concurrency())

    try:
        for round_idx in range(num_rounds):
            ranked = sorted(range(nv), key=lambda i: scores[i], reverse=True)
            pairs = [(ranked[j], ranked[j + 1]) for j in range(0, nv - 1, 2)]

            tasks = [
                _swiss_compare_pair(client, sem, ref_path, valid_renders[ia], valid_renders[ib], limiter=limiter)
                for ia, ib in pairs
            ]
            results = await asyncio.gather(*tasks)

            for (ia, ib), (winner, margin) in zip(pairs, results):
                if winner == "A":
                    scores[ia] += margin
                    scores[ib] -= margin
                elif winner == "B":
                    scores[ib] += margin
                    scores[ia] -= margin
    except Exception:
        logger.exception("[DesignValidator] swiss tournament error")
        return fail

    # Normalize: score ∈ [-2*num_rounds, +2*num_rounds] → [0, 1]
    max_score = 2 * num_rounds
    normalized = [(s + max_score) / (2 * max_score) for s in scores]

    out: list[Optional[float]] = [None] * n
    for idx, norm in zip(valid_indices, normalized):
        out[idx] = norm

    logger.info("[DesignValidator] swiss tournament: %d renders, %d rounds, %d pairs total",
                nv, num_rounds, num_rounds * (nv // 2))
    return out

# ---------------------------------------------------------------------------
# Refine mode: pairwise delta (ref + prev + cur → verdict)
# ---------------------------------------------------------------------------

# Image 2 = A = prev, Image 3 = B = refined
# Positive = refined is better
_VERDICT_MAP = {
    "b_better":  +1.0,
    "same":       0.0,
    "a_better":  -1.0,
}


async def _pairwise_one(client, sem, ref_path, prev_path, refine_path, limiter=None) -> Optional[float]:
    if not refine_path or not Path(refine_path).exists():
        logger.warning("[_pairwise_one] cur_render missing: %s (exists=%s)", refine_path, Path(refine_path).exists() if refine_path else "N/A")
        return None
    if not prev_path or not Path(prev_path).exists():
        logger.warning("[_pairwise_one] prev_render missing: %s (exists=%s)", prev_path, Path(prev_path).exists() if prev_path else "N/A")
        return None
    if not ref_path or not Path(ref_path).exists():
        logger.warning("[_pairwise_one] ref_path missing: %s (exists=%s)", ref_path, Path(ref_path).exists() if ref_path else "N/A")
        return None

    # Random A/B swap to de-bias the VLM's positional prior. The prompt
    # itself does not reveal which input is prev / refine, but VLMs still
    # tend to lean toward "A_better". By randomizing which physical image
    # sits in slot A vs slot B and then inverting the verdict accordingly,
    # the returned value remains semantically: +1 = refine better than prev.
    swap = random.random() < 0.5
    if swap:
        img_a, img_b = refine_path, prev_path
    else:
        img_a, img_b = prev_path, refine_path

    max_retries = 3
    async with sem:
        # Order matters for Azure prompt caching: fixed prefix first.
        # [text(fixed) + ref(fixed per sample)] is cached across all
        # rollouts in this sample; prev/refine vary per rollout.
        content = [
            {"type": "text", "text": _PROMPT_REFINE_PAIRWISE},
            _img_part(ref_path, mode="refine"),
            _img_part(img_a, mode="refine"),
            _img_part(img_b, mode="refine"),
        ]
        text = ""
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                if limiter is not None:
                    async with limiter:
                        resp = await client.chat.completions.create(
                            model=_get_model(),
                            messages=[{"role": "user", "content": content}],
                            max_completion_tokens=150,
                            temperature=0.0,
                        )
                else:
                    resp = await client.chat.completions.create(
                        model=_get_model(),
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=150,
                        temperature=0.0,
                    )
                text = resp.choices[0].message.content or ""
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.debug("[DesignValidator] pairwise retry %d/%d (%.1fs): %s",
                                 attempt + 1, max_retries, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                break
        if last_exc is not None:
            logger.warning("[DesignValidator] pairwise API error after %d retries: %s",
                           max_retries, last_exc)
            return None

    m = re.search(r"Verdict:\s*(A_better|same|B_better)", text, re.IGNORECASE)
    verdict: Optional[float] = None
    if m:
        verdict = _VERDICT_MAP.get(m.group(1).lower(), 0.0)
    else:
        low = text.lower()
        if "b_better" in low:  verdict = +1.0
        elif "a_better" in low: verdict = -1.0
        elif "same" in low:    verdict = 0.0

    if verdict is None:
        return None
    # Verdict is in (img_a, img_b) frame. Re-map to (prev, refine) frame
    # so that +1 always means refine beat prev.
    return -verdict if swap else verdict


async def pairwise_one_with_meta(
    ref_path: str,
    prev_path: str,
    refine_path: str,
) -> Optional[float]:
    """Single per-rollout pairwise call — no group state, no batching.

    Used by ``rl_verl.design_coordinator.RefineDesignScorer``.  Returns the
    delta in [-1, +1] frame where +1 means ``refine_path`` beat ``prev_path``,
    or ``None`` on any infra failure (caller decides how to map to a score).

    Shares the same client pool / limiter as the Swiss tournament so we
    never blow past the per-endpoint RPM budget.  A **process-wide**
    in-flight semaphore (``DESIGN_REFINE_MAX_INFLIGHT``, default
    ``VLM_REWARD_CONCURRENCY``) bounds total concurrent pairwise calls
    so a burst of refine rollouts cannot starve the limiter queue.
    """
    _ensure_dotenv()
    client, limiter = await _get_shared_client(_get_timeout())
    if client is None:
        return None
    sem = _get_refine_inflight_sem()
    return await _pairwise_one(client, sem, ref_path, prev_path, refine_path, limiter=limiter)


# Process-wide inflight cap for per-rollout pairwise calls.  ``_pairwise_one``
# itself takes the semaphore inside ``async with``, so we must reuse the SAME
# semaphore across calls — otherwise each caller creates a fresh sem and the
# cap is effectively unbounded (this was the supplementary issue raised in
# the review).
_refine_inflight_sem: Optional[asyncio.Semaphore] = None


def _get_refine_inflight_sem() -> asyncio.Semaphore:
    global _refine_inflight_sem
    if _refine_inflight_sem is None:
        cap = int(os.environ.get("DESIGN_REFINE_MAX_INFLIGHT",
                                 str(_get_concurrency())))
        _refine_inflight_sem = asyncio.Semaphore(max(1, cap))
    return _refine_inflight_sem


async def pairwise_judge_better(
    ref_path: str,
    prev_path: str,
    refine_paths: list[str],
) -> list[Optional[float]]:
    """Ref-anchored pairwise delta verdicts. Returns [-1, 0, +1] or None on failure."""
    n = len(refine_paths)
    _ensure_dotenv()
    client, limiter = await _get_shared_client(_get_timeout())
    if client is None:
        logger.warning("[DesignValidator] missing creds; pairwise → all None")
        return [None] * n
    sem = asyncio.Semaphore(_get_concurrency())
    results = await asyncio.gather(*[
        _pairwise_one(client, sem, ref_path, prev_path, p, limiter=limiter)
        for p in refine_paths
    ])
    results = list(results)

    # Sanity: if every successful verdict in the group is identical, the VLM
    # is collapsing — likely a positional / prompt-prior issue worth surfacing.
    succ = [v for v in results if v is not None]
    if len(succ) >= 2 and len(set(succ)) == 1:
        logger.warning(
            "[DesignValidator] pairwise verdicts have zero variance "
            "(n=%d, all=%+.1f) — possible VLM bias/collapse",
            len(succ), succ[0],
        )

    return results

# ---------------------------------------------------------------------------
# Utility: fill None with group mean
# ---------------------------------------------------------------------------

def fill_none_with_group_mean(values: list[Optional[float]]) -> list[float]:
    """Soft-exclude failed rollouts: replace None with mean of successful ones."""
    succ = [v for v in values if v is not None]
    if not succ:
        return [0.0] * len(values)
    mean = sum(succ) / len(succ)
    return [float(v) if v is not None else float(mean) for v in values]
