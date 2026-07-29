from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup

from verifier.artifacts.paths import get_render_artifact_path, get_render_debug_artifact_path
from verifier.types import ArtifactBundle

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local Playwright browser pool — avoids launching a new Chromium
# process for every sample.  Each thread gets its own browser instance that
# is reused across validate_sample() calls within the same ThreadPoolExecutor.
# ---------------------------------------------------------------------------
_browser_local = threading.local()


def _get_thread_browser():
    """Return a reusable Playwright browser for the current thread.

    Creates a Playwright context + browser on first call.  Subsequent calls
    in the same thread return the existing browser.
    """
    browser = getattr(_browser_local, "browser", None)
    pw = getattr(_browser_local, "playwright", None)
    if browser is not None and pw is not None:
        try:
            # Quick liveness check — Playwright throws if the browser process died
            browser.contexts  # noqa: B018
            return pw, browser
        except Exception:
            # Browser died — clean up and recreate
            _close_thread_browser()

    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    _browser_local.playwright = pw
    _browser_local.browser = browser
    return pw, browser


def _close_thread_browser():
    """Close the thread-local browser and Playwright context (cleanup)."""
    browser = getattr(_browser_local, "browser", None)
    pw = getattr(_browser_local, "playwright", None)
    if browser is not None:
        try:
            browser.close()
        except Exception:
            pass
    if pw is not None:
        try:
            pw.stop()
        except Exception:
            pass
    _browser_local.browser = None
    _browser_local.playwright = None


def close_all_thread_browsers():
    """Public API to close the current thread's browser. Call after batch is done."""
    _close_thread_browser()


# ---------------------------------------------------------------------------
# Playwright network interception — external resource allowlist
# ---------------------------------------------------------------------------
# Only these external URLs are permitted during rendering.  Everything else
# is physically blocked via route.abort() so that the rendered page reflects
# exactly what the allowed resources provide.  This replaces any explicit
# scoring penalty: if a model hallucinates an external CDN, the resource
# simply never loads and downstream validators (layout, colour, design)
# naturally penalise the broken output.

_ALLOWED_EXTERNAL_PREFIXES: tuple[str, ...] = (
    # placehold.co is intercepted and synthesised locally (see
    # _synth_placehold_svg) so it works with or without network.  Listed here
    # too so callers searching for "what's allowed?" still find it.
    "https://placehold.co",
    "http://placehold.co",
)
# NOTE: echarts CDN URLs used to be on the exact-allow-list, but on no-network
# nodes they hang the renderer for ~120s.  They are now intercepted and served
# from a local bundled copy at verifier/artifacts/assets/echarts.min.js so
# HTML using ECharts renders correctly on both online and offline nodes.
# See _make_network_interceptor below.
_ALLOWED_EXTERNAL_EXACT: frozenset[str] = frozenset()

# Local ECharts bundle (served when HTML loads echarts from any CDN).
# All known CDN paths for echarts get intercepted and fulfilled from this file.
_ECHARTS_LOCAL_PATH = Path(__file__).resolve().parent / "assets" / "echarts.min.js"
_ECHARTS_BODY_CACHE: bytes | None = None
def _get_echarts_body() -> bytes | None:
    """Lazy-load the bundled echarts.min.js (cached after first read)."""
    global _ECHARTS_BODY_CACHE
    if _ECHARTS_BODY_CACHE is None:
        try:
            _ECHARTS_BODY_CACHE = _ECHARTS_LOCAL_PATH.read_bytes()
        except FileNotFoundError:
            return None
    return _ECHARTS_BODY_CACHE


def _is_echarts_cdn_url(url: str) -> bool:
    """Match any known CDN URL hosting echarts.* (any version)."""
    if "echarts" not in url.lower():
        return False
    return any(host in url for host in (
        "jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com",
        "npmmirror.com", "bootcdn.net", "staticfile.org",
    ))


def _is_allowed_url(url: str) -> bool:
    """Return True if *url* is a local resource or on the allowlist."""
    if not url.startswith(("http://", "https://")):
        return True  # file://, data:, blob:, about: — all local
    if url in _ALLOWED_EXTERNAL_EXACT:
        return True
    return any(url.startswith(prefix) for prefix in _ALLOWED_EXTERNAL_PREFIXES)


# ---------------------------------------------------------------------------
# placehold.co local SVG synthesiser
# ---------------------------------------------------------------------------
# placehold.co is a remote service that returns an SVG (grey rect + centred
# label) for URLs like:
#   https://placehold.co/640x480
#   https://placehold.co/640x480/cccccc/333333?text=Hello+World
#   https://placehold.co/640x480?text=Hi&font=arial
# On no-network nodes the request hangs until the outer timeout SIGTERMs the
# render worker (~120s) — even though the response is trivial to reconstruct
# locally.  We synthesise an equivalent SVG and fulfill the request directly,
# so the rendered page looks the same with or without network access.

import urllib.parse as _urlparse

_PLACEHOLD_HOSTS = ("placehold.co", "www.placehold.co")
_PLACEHOLD_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


def _normalise_colour(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if token.startswith("#"):
        return token
    # placehold.co accepts bare hex like 'cccccc' or named colours like 'red'
    if re.fullmatch(r"[0-9A-Fa-f]{3,8}", token):
        return "#" + token
    return token  # assume CSS-named colour ('red', 'navy', ...)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )


def _synth_placehold_svg(url: str) -> Optional[bytes]:
    """Build an SVG body matching what placehold.co would return for *url*.

    Returns None if the URL doesn't look like a placehold.co URL we can parse.
    """
    parsed = _urlparse.urlparse(url)
    host = parsed.netloc.lower()
    if host not in _PLACEHOLD_HOSTS:
        return None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if not parts:
        return None
    size_m = _PLACEHOLD_SIZE_RE.match(parts[0])
    if not size_m:
        return None
    w, h = int(size_m.group(1)), int(size_m.group(2))
    bg = "#cccccc"
    fg = "#333333"
    if len(parts) >= 2:
        bg = _normalise_colour(parts[1]) or bg
    if len(parts) >= 3:
        fg = _normalise_colour(parts[2]) or fg
    qs = _urlparse.parse_qs(parsed.query)
    raw_text = qs.get("text", [f"{w}x{h}"])[0]
    text = _xml_escape(raw_text.replace("+", " "))
    font_size = max(min(w, h) // 10, 12)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
        f'<rect width="100%" height="100%" fill="{bg}"/>'
        f'<text x="50%" y="50%" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="{font_size}" fill="{fg}" text-anchor="middle" '
        f'dominant-baseline="middle">{text}</text>'
        f'</svg>'
    )
    return svg.encode("utf-8")


def _make_network_interceptor(blocked_urls: list[str]):
    """Return a Playwright route handler that handles disallowed URLs.

    Behaviour:
      - local URLs (file://, data:, blob:, about:) → continue_
      - placehold.co URLs → fulfill with locally synthesised SVG (works
        identically with or without network access)
      - other allow-listed URLs → continue_ (will only succeed if network is
        actually reachable; on no-net nodes they will still be slow but at
        least the common placehold.co case never hangs)
      - everything else → abort("blockedbyclient")
    """
    def _handler(route):
        url = route.request.url
        # Fast path: ECharts CDN (any host, any version) — fulfill from local bundle
        if _is_echarts_cdn_url(url):
            body = _get_echarts_body()
            if body is not None:
                try:
                    route.fulfill(status=200, content_type="application/javascript", body=body)
                    return
                except Exception:
                    pass  # Fall through to abort if fulfill failed
        # Fast path: placehold.co — synthesise locally regardless of network
        if "placehold.co" in url:
            body = _synth_placehold_svg(url)
            if body is not None:
                try:
                    route.fulfill(status=200, content_type="image/svg+xml", body=body)
                    return
                except Exception:
                    # Fall through to abort if fulfill itself failed
                    pass
        if _is_allowed_url(url):
            route.continue_()
        else:
            blocked_urls.append(url)
            route.abort("blockedbyclient")
    return _handler


PAGE_ROOT_SELECTOR_CANDIDATES = (
    ".poster",
    ".artboard",
    '[data-entity="page"]',
    '[data-role="page"]',
)

PLAYWRIGHT_PAGE_ROOT_SELECTOR = ", ".join([*PAGE_ROOT_SELECTOR_CANDIDATES, "body"])


SIZE_DECLARATION_RE = re.compile(
    r"(?P<selector>\.[A-Za-z0-9_-]+)\s*\{(?P<body>.*?)\}",
    re.DOTALL,
)
CSS_VALUE_RE = re.compile(r"(?P<name>[A-Za-z-]+)\s*:\s*(?P<value>[^;]+);")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_style_text(soup: BeautifulSoup) -> str:
    return "\n".join(node.get_text("\n", strip=False) for node in soup.select("style"))


def _parse_selector_declarations(css_text: str, selector: str) -> Dict[str, str]:
    declarations: Dict[str, str] = {}
    for match in SIZE_DECLARATION_RE.finditer(css_text):
        if match.group("selector") != selector:
            continue
        body = match.group("body")
        declarations.update(
            {
                value_match.group("name").strip(): value_match.group("value").strip()
                for value_match in CSS_VALUE_RE.finditer(body)
            }
        )
    return declarations


def _parse_inline_style(style_value: Optional[str]) -> Dict[str, str]:
    if not style_value:
        return {}
    return {
        match.group("name").strip(): match.group("value").strip()
        for match in CSS_VALUE_RE.finditer(style_value)
    }


def _resolve_static_page_root_selector(soup: BeautifulSoup) -> str:
    for selector in PAGE_ROOT_SELECTOR_CANDIDATES:
        if soup.select_one(selector):
            return selector
    return ".poster"


def _resolve_static_page_content_selector(root_selector: str) -> Optional[str]:
    if root_selector == ".artboard":
        return ".artboard-inner"
    if root_selector == ".poster":
        return ".poster-inner"
    return None


def _extract_numeric_px(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)px", value)
    if not match:
        return None
    return float(match.group(1))


def _extract_numeric(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", value)
    if not match:
        return None
    return float(match.group(1))


def _parse_box_sides(value: Optional[str]) -> tuple[float, float, float, float]:
    if not value:
        return (0.0, 0.0, 0.0, 0.0)
    parts = [part for part in value.split() if part]
    numeric_parts = [_extract_numeric_px(part) or 0.0 for part in parts]
    if len(numeric_parts) == 1:
        top = right = bottom = left = numeric_parts[0]
    elif len(numeric_parts) == 2:
        top = bottom = numeric_parts[0]
        right = left = numeric_parts[1]
    elif len(numeric_parts) == 3:
        top = numeric_parts[0]
        right = left = numeric_parts[1]
        bottom = numeric_parts[2]
    else:
        top, right, bottom, left = (numeric_parts + [0.0, 0.0, 0.0, 0.0])[:4]
    return (top, right, bottom, left)


def _is_before_in_dom(left_node: Any, right_node: Any) -> bool:
    if left_node is None or right_node is None:
        return False
    current = left_node
    while current is not None:
        if current == right_node:
            return True
        current = current.find_next()
    return False


def _column_snapshot(column_node: Any) -> Dict[str, Any]:
    heading_nodes = column_node.select(".col-heading")
    bullet_items = column_node.select(":scope > .bullet-item")
    return {
        "heading_texts": [normalize_text(node.get_text(" ", strip=True)) for node in heading_nodes],
        "bullet_item_count": len(bullet_items),
        "bullet_marker_count": len(column_node.select(":scope .bullet-marker")),
        "storyboard_panel_count": len(column_node.select(":scope .story-panel")),
        "colored_circle_count": len(column_node.select(":scope .cc")),
        "has_blank_rect": column_node.select_one(":scope .blank-rect") is not None,
        "has_eye_wrap": column_node.select_one(":scope .eye-wrap") is not None,
        "raw_text": normalize_text(column_node.get_text(" ", strip=True)),
    }


def _estimate_text_width(text: str, font_size: float) -> float:
    width_units = 0.0
    for char in text:
        if char.isspace():
            width_units += 0.25
        elif ord(char) > 127:
            width_units += 1.0
        else:
            width_units += 0.6
    return max(font_size * width_units, font_size * 2.0)


def _make_box(name: str, x: float, y: float, width: float, height: float) -> Dict[str, Any]:
    return {
        "name": name,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "right": round(x + width, 2),
        "bottom": round(y + height, 2),
        "cx": round(x + width / 2.0, 2),
        "cy": round(y + height / 2.0, 2),
        "bbox": [round(x, 2), round(y, 2), round(width, 2), round(height, 2)],
    }


def _union_boxes(name: str, boxes: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    box_list = [box for box in boxes if box]
    if not box_list:
        return None
    min_x = min(box["x"] for box in box_list)
    min_y = min(box["y"] for box in box_list)
    max_right = max(box["right"] for box in box_list)
    max_bottom = max(box["bottom"] for box in box_list)
    return _make_box(name, min_x, min_y, max_right - min_x, max_bottom - min_y)


def _normalize_bbox(box: Dict[str, Any], page_width: float, page_height: float) -> List[float]:
    if not page_width or not page_height:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(box["x"] / page_width, 4),
        round(box["y"] / page_height, 4),
        round(box["width"] / page_width, 4),
        round(box["height"] / page_height, 4),
    ]


def _parse_alias_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in re.split(r"[|,]", value) if part.strip()]


def _merge_aliases(existing: Iterable[str], incoming: Iterable[str]) -> List[str]:
    merged: List[str] = []
    for alias in [*(existing or []), *(incoming or [])]:
        if alias and alias not in merged:
            merged.append(alias)
    return merged


def _collect_declared_semantic_spec(node: Any) -> Optional[Dict[str, Any]]:
    if not getattr(node, "attrs", None):
        return None
    name = normalize_text(node.get("data-entity") or "")
    role = normalize_text(node.get("data-role") or name)
    scope = normalize_text(node.get("data-scope") or "") or None
    aliases = _parse_alias_list(node.get("data-aliases"))
    selector = node.get("data-selector") or (f'[data-entity="{name}"]' if name else None)
    text = normalize_text(node.get_text(" ", strip=True)) if hasattr(node, "get_text") else ""
    if not any([name, role, scope, aliases, selector]):
        return None
    return {
        "name": name or None,
        "role": role or None,
        "scope": scope,
        "aliases": aliases,
        "selector": selector,
        "text": text,
    }


def _apply_declared_semantic_overrides(bundle: ArtifactBundle, soup: BeautifulSoup) -> None:
    declared: List[Dict[str, Any]] = []
    for node in soup.select("[data-entity], [data-role]"):
        spec = _collect_declared_semantic_spec(node)
        if not spec:
            continue
        declared.append(spec)
        name = spec.get("name")
        if not name:
            continue
        element = _find_named_box(bundle.detected_elements, name)
        if not element:
            continue
        if spec.get("role"):
            element["role"] = spec["role"]
        if spec.get("scope"):
            element["scope"] = spec["scope"]
        if spec.get("selector"):
            element["selector"] = spec["selector"]
        if spec.get("text"):
            element["text"] = spec["text"]
        element["aliases"] = _merge_aliases(element.get("aliases") or [], spec.get("aliases") or [])
        element.setdefault("metadata", {}).update(
            {
                "declared_semantic": True,
                "declared_role": spec.get("role"),
                "declared_scope": spec.get("scope"),
            }
        )
    if bundle.dom_snapshot is not None:
        bundle.dom_snapshot["declared_semantic_elements"] = declared


def _upsert_detected_element(
    bundle: ArtifactBundle,
    *,
    name: str,
    role: str,
    box: Dict[str, Any],
    selector: Optional[str] = None,
    scope: Optional[str] = None,
    text: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    page_width = float(bundle.dom_snapshot.get("poster_width") or box["right"] or 1.0)
    page_height = float(bundle.dom_snapshot.get("poster_height") or box["bottom"] or 1.0)
    existing_element = _find_named_box(bundle.detected_elements, name)
    if existing_element:
        _update_box_geometry(existing_element, box)
        existing_element["normalized_bbox"] = _normalize_bbox(box, page_width, page_height)
        existing_element["role"] = role or existing_element.get("role")
        existing_element["scope"] = scope or existing_element.get("scope")
        existing_element["selector"] = selector or existing_element.get("selector")
        if text:
            existing_element["text"] = text
        existing_element["aliases"] = _merge_aliases(existing_element.get("aliases") or [], aliases or [])
        existing_element["source"] = source
        if metadata:
            existing_element.setdefault("metadata", {}).update(metadata)
    else:
        _add_detected_element(
            bundle.detected_elements,
            page_width,
            page_height,
            name=name,
            role=role,
            box=box,
            selector=selector,
            scope=scope,
            text=text,
            aliases=aliases,
            source=source,
            metadata=metadata,
        )
    existing_layout = _find_named_box(bundle.layout_boxes, name)
    if existing_layout:
        _update_box_geometry(existing_layout, box)
    else:
        bundle.layout_boxes.append(box)


def _extract_playwright_semantic_elements(page: Any, root_selector: Optional[str]) -> List[Dict[str, Any]]:
    script = r"""
        () => {
            const rootSelector = __ROOT_SELECTOR__;
            const root = (rootSelector ? document.querySelector(rootSelector) : null)
                || document.body
                || document.documentElement;
            if (!root) {
                return [];
            }
            const rootRect = root.getBoundingClientRect();
            const counters = new Map();
            const normalizeText = value => String(value || '').replace(/\s+/g, ' ').trim();
            const nodes = Array.from(document.querySelectorAll('[data-entity], [data-role]'));
            const items = [];

            for (const node of nodes) {
                if (!root.contains(node)) {
                    continue;
                }
                const style = window.getComputedStyle(node);
                if (
                    style.display === 'none' ||
                    style.visibility === 'hidden' ||
                    Number.parseFloat(style.opacity || '1') === 0
                ) {
                    continue;
                }
                const rect = node.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) {
                    continue;
                }
                let name = normalizeText(node.getAttribute('data-entity'));
                const role = normalizeText(node.getAttribute('data-role')) || name;
                const scope = normalizeText(node.getAttribute('data-scope')) || null;
                const aliases = normalizeText(node.getAttribute('data-aliases'))
                    .split(/[|,]/)
                    .map(part => part.trim())
                    .filter(Boolean);
                const selector = node.getAttribute('data-selector') || (name ? `[data-entity="${name}"]` : null);
                const text = normalizeText(node.innerText || node.textContent || '');

                if (!name && role) {
                    const key = `${role}::${scope || 'page'}`;
                    const count = (counters.get(key) || 0) + 1;
                    counters.set(key, count);
                    name = `${role}_${count}`;
                }
                if (!name) {
                    continue;
                }

                items.push({
                    name,
                    role: role || name,
                    scope,
                    aliases,
                    selector,
                    text,
                    bbox: [
                        rect.left - rootRect.left,
                        rect.top - rootRect.top,
                        rect.width,
                        rect.height,
                    ],
                });
            }

            return items;
        }
    """
    script = script.replace("__ROOT_SELECTOR__", json.dumps(root_selector))
    return page.evaluate(script)


def _extract_playwright_text_blocks(page: Any, root_selector: Optional[str]) -> List[Dict[str, Any]]:
    script = r"""
        () => {
            const rootSelector = __ROOT_SELECTOR__;
            const root = (rootSelector ? document.querySelector(rootSelector) : null)
                || document.body
                || document.documentElement;
            if (!root) {
                return [];
            }
            const rootRect = root.getBoundingClientRect();
            const normalizeText = value => String(value || '').replace(/\s+/g, ' ').trim();
            const parseColor = value => {
                const match = String(value || '').match(/rgba?\(([^)]+)\)/i);
                if (!match) {
                    return [0, 0, 0];
                }
                return match[1]
                    .split(',')
                    .slice(0, 3)
                    .map(channel => Math.max(0, Math.min(255, Math.round(parseFloat(channel) || 0))));
            };
            const blocks = [];
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
                acceptNode(node) {
                    if (!node || !node.nodeValue || !node.nodeValue.trim()) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    const parent = node.parentElement;
                    if (!parent || parent.closest('script, style, noscript')) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    const style = window.getComputedStyle(parent);
                    if (
                        style.display === 'none' ||
                        style.visibility === 'hidden' ||
                        Number.parseFloat(style.opacity || '1') === 0
                    ) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return NodeFilter.FILTER_ACCEPT;
                },
            });

            while (walker.nextNode()) {
                const node = walker.currentNode;
                const parent = node.parentElement;
                const text = normalizeText(node.nodeValue);
                if (!text) {
                    continue;
                }

                const range = document.createRange();
                range.selectNodeContents(node);
                const rects = Array.from(range.getClientRects()).filter(rect => rect.width > 0 && rect.height > 0);
                if (!rects.length) {
                    continue;
                }

                const left = Math.min(...rects.map(rect => rect.left)) - rootRect.left;
                const top = Math.min(...rects.map(rect => rect.top)) - rootRect.top;
                const right = Math.max(...rects.map(rect => rect.right)) - rootRect.left;
                const bottom = Math.max(...rects.map(rect => rect.bottom)) - rootRect.top;
                const anchor = parent.closest('[data-entity], [data-role]');
                const style = window.getComputedStyle(anchor || parent);
                const parentClasses = Array.from(parent.classList || []).join('.');
                const parentSignature = [
                    String(parent.tagName || '').toLowerCase(),
                    parent.id || '',
                    parentClasses,
                    parent.getAttribute('data-entity') || '',
                    parent.getAttribute('data-role') || '',
                ].join('|');

                blocks.push({
                    text,
                    bbox: [left, top, right - left, bottom - top],
                    color: parseColor(style.color),
                    parent_signature: parentSignature,
                    semantic_name: anchor ? normalizeText(anchor.getAttribute('data-entity')) || null : null,
                    semantic_role: anchor ? normalizeText(anchor.getAttribute('data-role')) || null : null,
                    semantic_scope: anchor ? normalizeText(anchor.getAttribute('data-scope')) || null : null,
                });
            }
            return blocks;
        }
    """
    script = script.replace("__ROOT_SELECTOR__", json.dumps(root_selector))
    return page.evaluate(script)


def _hydrate_playwright_text_blocks(bundle: ArtifactBundle, raw_blocks: List[Dict[str, Any]]) -> None:
    page_width = float(bundle.dom_snapshot.get("poster_width") or 1.0)
    page_height = float(bundle.dom_snapshot.get("poster_height") or 1.0)
    text_blocks: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_blocks, start=1):
        bbox = raw.get("bbox") or []
        if len(bbox) != 4:
            continue
        text = normalize_text(str(raw.get("text") or ""))
        if not text:
            continue
        box = _make_box(f"text_block_{index}", float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        text_blocks.append(
            {
                **box,
                "name": f"text_block_{index}",
                "role": "text_block",
                "text": text,
                "color": [int(value) for value in (raw.get("color") or [0, 0, 0])[:3]],
                "normalized_bbox": _normalize_bbox(box, page_width, page_height),
                "parent_signature": raw.get("parent_signature"),
                "semantic_name": raw.get("semantic_name"),
                "semantic_role": raw.get("semantic_role"),
                "semantic_scope": raw.get("semantic_scope"),
                "source": "playwright_text_rects",
            }
        )
    bundle.text_blocks = text_blocks


def _add_detected_element(
    detected_elements: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    *,
    name: str,
    role: str,
    box: Dict[str, Any],
    selector: Optional[str] = None,
    scope: Optional[str] = None,
    text: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    source: str = "inferred",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    detected_elements.append(
        {
            **box,
            "name": name,
            "role": role,
            "scope": scope,
            "selector": selector,
            "text": text,
            "aliases": aliases or [],
            "normalized_bbox": _normalize_bbox(box, page_width, page_height),
            "source": source,
            "metadata": metadata or {},
        }
    )


def _update_box_geometry(target: Dict[str, Any], box: Dict[str, Any]) -> None:
    for key in ("x", "y", "width", "height", "right", "bottom", "cx", "cy", "bbox"):
        target[key] = box[key]


def _replace_named_geometry(
    bundle: ArtifactBundle,
    *,
    name: str,
    box: Dict[str, Any],
    source: str,
    selector: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    for layout_box in bundle.layout_boxes:
        if layout_box.get("name") == name:
            _update_box_geometry(layout_box, box)
    for element in bundle.detected_elements:
        if element.get("name") == name:
            _update_box_geometry(element, box)
            element["normalized_bbox"] = _normalize_bbox(
                box,
                float(bundle.dom_snapshot.get("poster_width") or 0.0),
                float(bundle.dom_snapshot.get("poster_height") or 0.0),
            )
            element["source"] = source
            if selector is not None:
                element["selector"] = selector
            if metadata:
                element.setdefault("metadata", {}).update(metadata)


def _derived_box_from_parent(
    parent_box: Dict[str, Any],
    *,
    name: str,
    x_ratio: float,
    y_ratio: float,
    width_ratio: float,
    height_ratio: float,
) -> Dict[str, Any]:
    return _make_box(
        name,
        parent_box["x"] + parent_box["width"] * x_ratio,
        parent_box["y"] + parent_box["height"] * y_ratio,
        parent_box["width"] * width_ratio,
        parent_box["height"] * height_ratio,
    )


def _measure_playwright_box(page: Any, selector: str, *, index: int = 0) -> Optional[Dict[str, float]]:
    locator = page.locator(selector)
    if locator.count() <= index:
        return None
    box = locator.nth(index).bounding_box()
    if not box:
        return None
    return {
        "x": float(box["x"]),
        "y": float(box["y"]),
        "width": float(box["width"]),
        "height": float(box["height"]),
    }


def _playwright_locator_tag_name(page: Any, selector: str, *, index: int = 0) -> Optional[str]:
    locator = page.locator(selector)
    if locator.count() <= index:
        return None
    try:
        return str(locator.nth(index).evaluate("node => (node.tagName || '').toLowerCase()") or "")
    except Exception:
        return None


def _measure_first_playwright_box(page: Any, selectors: Iterable[str]) -> tuple[Optional[str], Optional[Dict[str, float]]]:
    fallback: tuple[Optional[str], Optional[Dict[str, float]]] = (None, None)
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        for index in range(count):
            measured = _measure_playwright_box(page, selector, index=index)
            if not measured:
                continue
            tag_name = _playwright_locator_tag_name(page, selector, index=index)
            if tag_name not in {"body", "html"}:
                return selector, measured
            if fallback == (None, None):
                fallback = (selector, measured)
    return fallback


def _measure_playwright_boxes(page: Any, selector: str) -> List[Dict[str, float]]:
    locator = page.locator(selector)
    boxes: List[Dict[str, float]] = []
    for index in range(locator.count()):
        box = locator.nth(index).bounding_box()
        if not box:
            continue
        boxes.append(
            {
                "x": float(box["x"]),
                "y": float(box["y"]),
                "width": float(box["width"]),
                "height": float(box["height"]),
            }
        )
    return boxes


def _relative_playwright_box(measured: Dict[str, float], poster_origin: Dict[str, float], name: str) -> Dict[str, Any]:
    return _make_box(
        name,
        measured["x"] - poster_origin["x"],
        measured["y"] - poster_origin["y"],
        measured["width"],
        measured["height"],
    )


def _capture_playwright_render(bundle: ArtifactBundle, page: Any, poster_measured: Dict[str, float], html_path: Path) -> str:
    _ = html_path
    render_path = get_render_artifact_path(sample_id=bundle.sample_id, backend="playwright")
    page.screenshot(
        path=str(render_path),
        clip={
            "x": float(poster_measured["x"]),
            "y": float(poster_measured["y"]),
            "width": float(poster_measured["width"]),
            "height": float(poster_measured["height"]),
        },
    )
    return str(render_path)


def _capture_playwright_overflow_debug_render(
    bundle: ArtifactBundle,
    page: Any,
    root_selector: Optional[str],
    poster_measured: Dict[str, float],
    viewport_width: int,
    viewport_height: int,
) -> Optional[str]:
    overflow_items = bundle.overflowing_elements or []
    if not overflow_items:
        return None

    debug_bounds = page.evaluate(
        r"""
        (rootSelector) => {
            const root = (rootSelector ? document.querySelector(rootSelector) : null)
                || document.body
                || document.documentElement;
            if (!root) {
                return null;
            }
            const originalRootOverflow = root.style.overflow || '';
            const originalBodyOverflow = document.body ? (document.body.style.overflow || '') : '';
            root.style.overflow = 'visible';
            if (document.body) {
                document.body.style.overflow = 'visible';
            }

            const rootRect = root.getBoundingClientRect();
            const candidates = [rootRect];
            for (const el of root.querySelectorAll('*')) {
                const style = getComputedStyle(el);
                const text = (el.innerText || '').replace(/\s+/g, ' ').trim();
                const hasSemantic = el.hasAttribute('data-role') || el.hasAttribute('data-entity') || el.hasAttribute('data-scope');
                if (style.display === 'none' || style.visibility === 'hidden' || Number.parseFloat(style.opacity || '1') === 0) {
                    continue;
                }
                if (!hasSemantic && text.length < 2) {
                    continue;
                }
                const rect = el.getBoundingClientRect();
                if (rect.width < 1 || rect.height < 1) {
                    continue;
                }
                if (
                    rect.left < rootRect.left - 1 ||
                    rect.right > rootRect.right + 1 ||
                    rect.top < rootRect.top - 1 ||
                    rect.bottom > rootRect.bottom + 1
                ) {
                    candidates.push(rect);
                }
            }

            const minLeft = Math.min(...candidates.map((rect) => rect.left));
            const minTop = Math.min(...candidates.map((rect) => rect.top));
            const maxRight = Math.max(...candidates.map((rect) => rect.right));
            const maxBottom = Math.max(...candidates.map((rect) => rect.bottom));
            return {
                minLeft,
                minTop,
                maxRight,
                maxBottom,
                rootLeft: rootRect.left,
                rootTop: rootRect.top,
                rootWidth: rootRect.width,
                rootHeight: rootRect.height,
                originalRootOverflow,
                originalBodyOverflow,
            };
        }
        """,
        root_selector,
    )
    if not debug_bounds:
        return None

    required_width = int(max(viewport_width, debug_bounds["maxRight"] + 40))
    required_height = int(max(viewport_height, debug_bounds["maxBottom"] + 40))
    if required_width > viewport_width or required_height > viewport_height:
        page.set_viewport_size({"width": required_width, "height": required_height})

    margin = 8.0
    clip_left = max(0.0, min(float(debug_bounds["minLeft"]), float(poster_measured["x"])) - margin)
    clip_top = max(0.0, min(float(debug_bounds["minTop"]), float(poster_measured["y"])) - margin)
    clip_right = max(float(debug_bounds["maxRight"]), float(poster_measured["x"] + poster_measured["width"])) + margin
    clip_bottom = max(float(debug_bounds["maxBottom"]), float(poster_measured["y"] + poster_measured["height"])) + margin

    debug_path = get_render_debug_artifact_path(sample_id=bundle.sample_id, backend="playwright", kind="overflow")
    page.screenshot(
        path=str(debug_path),
        clip={
            "x": clip_left,
            "y": clip_top,
            "width": max(1.0, clip_right - clip_left),
            "height": max(1.0, clip_bottom - clip_top),
        },
    )

    page.evaluate(
        r"""
        (payload) => {
            const rootSelector = payload?.rootSelector || null;
            const originalRootOverflow = payload?.originalRootOverflow || '';
            const originalBodyOverflow = payload?.originalBodyOverflow || '';
            const root = (rootSelector ? document.querySelector(rootSelector) : null)
                || document.body
                || document.documentElement;
            if (root) {
                root.style.overflow = originalRootOverflow;
            }
            if (document.body) {
                document.body.style.overflow = originalBodyOverflow;
            }
        }
        """,
        {
            "rootSelector": root_selector,
            "originalRootOverflow": debug_bounds["originalRootOverflow"],
            "originalBodyOverflow": debug_bounds["originalBodyOverflow"],
        },
    )

    # Restore viewport to the original size so that subsequent text-block
    # extraction (and any other measurements) uses the same coordinate
    # system as the poster screenshot that was captured earlier.
    if required_width > viewport_width or required_height > viewport_height:
        page.set_viewport_size({"width": viewport_width, "height": viewport_height})

    return str(debug_path)


def _render_image_looks_blank(image_path: str) -> bool:
    """Conservative blank-screen detector.

    We intentionally bias toward false negatives rather than false positives:
    a page is considered blank only when it is both near-monochrome and has
    almost no darker-than-background pixels. This avoids killing low-contrast
    but visibly rendered pages.
    """
    try:
        import cv2
    except ImportError:
        return False
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    variance = float(image.var())
    dark_pixel_ratio = float((image < 235).mean())
    return variance < 8.0 and dark_pixel_ratio < 0.005


def _measure_locator_descendant_union_box(locator: Any) -> Optional[Dict[str, float]]:
    descendant_locator = locator.locator("*")
    boxes: List[Dict[str, float]] = []
    for index in range(descendant_locator.count()):
        box = descendant_locator.nth(index).bounding_box()
        if not box:
            continue
        boxes.append(
            {
                "x": float(box["x"]),
                "y": float(box["y"]),
                "width": float(box["width"]),
                "height": float(box["height"]),
            }
        )
    if not boxes:
        return None
    min_x = min(box["x"] for box in boxes)
    min_y = min(box["y"] for box in boxes)
    max_right = max(box["x"] + box["width"] for box in boxes)
    max_bottom = max(box["y"] + box["height"] for box in boxes)
    return {
        "x": min_x,
        "y": min_y,
        "width": max_right - min_x,
        "height": max_bottom - min_y,
    }


def _find_named_box(items: Iterable[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if item.get("name") == name:
            return item
    return None


def _estimated_column_content_box(
    column_box: Dict[str, Any],
    heading_box: Dict[str, Any],
    column_snapshot: Dict[str, Any],
    *,
    name: str,
    extra_boxes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    bullet_count = int(column_snapshot.get("bullet_item_count", 0))
    storyboard_panel_count = int(column_snapshot.get("storyboard_panel_count", 0))
    colored_circle_count = int(column_snapshot.get("colored_circle_count", 0))
    estimated_height = 92.0 + bullet_count * 56.0
    estimated_height += storyboard_panel_count * 12.0
    estimated_height += colored_circle_count * 10.0
    if column_snapshot.get("has_blank_rect"):
        estimated_height += 90.0
    if column_snapshot.get("has_eye_wrap"):
        estimated_height += 110.0
    base_box = _make_box(
        name,
        column_box["x"] + 10.0,
        heading_box["y"],
        max(column_box["width"] - 20.0, 0.0),
        max(min(estimated_height, column_box["height"] - 14.0), 32.0),
    )
    union_box = _union_boxes(name, [base_box, heading_box, *(extra_boxes or [])])
    if not union_box:
        return base_box
    max_bottom = min(union_box["bottom"], column_box["bottom"] - 8.0)
    return _make_box(
        name,
        union_box["x"],
        union_box["y"],
        union_box["width"],
        max(max_bottom - union_box["y"], 24.0),
    )


_OVERFLOW_PROBE_JS = r"""
(posterRect) => {
    const toleranceX = 8;
    const toleranceY = 24;
    const decorativeClassPattern = /(^|\s)(doodle|brain|sprout|notepad|palette|crystal|child-bubble|child-figure|main-arrow|tiny-person|col3cluster|col4cluster|gap-entity|heading-back)(\s|$)/i;
    const svgTags = new Set(['SVG', 'PATH', 'G', 'LINE', 'POLYLINE', 'POLYGON', 'RECT', 'CIRCLE', 'ELLIPSE']);

    const pLeft = posterRect.x;
    const pTop = posterRect.y;
    const pRight = posterRect.x + posterRect.width;
    const pBottom = posterRect.y + posterRect.height;

    const normalizeClassName = (value) => {
        if (typeof value === 'string') return value;
        if (value && typeof value.baseVal === 'string') return value.baseVal;
        return '';
    };

    const hasMeaningfulText = (el) => ((el.innerText || '').replace(/\s+/g, '').length >= 2);
    const hasSemanticAttrs = (el) => (
        el.hasAttribute('data-role') || el.hasAttribute('data-entity') || el.hasAttribute('data-scope')
    );
    const isHidden = (style) => (
        style.display === 'none' ||
        style.visibility === 'hidden' ||
        Number.parseFloat(style.opacity || '1') === 0
    );
    const isDecorative = (el, style) => {
        const className = normalizeClassName(el.className);
        if (svgTags.has(el.tagName)) return true;
        if (decorativeClassPattern.test(className)) return true;
        if (style.pointerEvents === 'none' && !hasMeaningfulText(el) && !hasSemanticAttrs(el)) return true;
        return false;
    };

    return [...document.querySelectorAll('*')]
        .map((el) => {
            const style = getComputedStyle(el);
            if (isHidden(style) || isDecorative(el, style)) return null;
            const rect = el.getBoundingClientRect();
            if (rect.width < 1 || rect.height < 1) return null;

            // Check if element's bounding rect exceeds poster bounds
            const exceedsLeft = Math.max(0, pLeft - rect.left);
            const exceedsTop = Math.max(0, pTop - rect.top);
            const exceedsRight = Math.max(0, rect.right - pRight);
            const exceedsBottom = Math.max(0, rect.bottom - pBottom);

            if (exceedsLeft <= toleranceX && exceedsRight <= toleranceX &&
                exceedsTop <= toleranceY && exceedsBottom <= toleranceY) return null;

            // Must be meaningful content
            if (!hasMeaningfulText(el) && !hasSemanticAttrs(el)) return null;

            const text = (el.innerText || '').replace(/\s+/g, ' ').trim();
            return {
                tag: el.tagName,
                cls: normalizeClassName(el.className),
                id: el.id || '',
                overflow_x: Math.max(exceedsLeft, exceedsRight),
                overflow_y: Math.max(exceedsTop, exceedsBottom),
                left: rect.left,
                top: rect.top,
                right: rect.right,
                bottom: rect.bottom,
                width: rect.width,
                height: rect.height,
                text_preview: text.slice(0, 80),
            };
        })
        .filter(Boolean)
        .filter((item, index, list) => {
            const key = `${item.tag}|${item.id}|${item.cls}`;
            return list.findIndex((candidate) => `${candidate.tag}|${candidate.id}|${candidate.cls}` === key) === index;
        });
}
"""


def _apply_playwright_geometry(bundle: ArtifactBundle, html_path: Path, backend: str) -> ArtifactBundle:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - optional dependency path
        bundle.metadata.update(
            {
                "geometry_mode": "static_semantic_inference",
                "geometry_backend_requested": backend,
                "geometry_backend_used": "static",
                "geometry_backend_fallback_reason": f"playwright_import_failed: {exc}",
            }
        )
        return bundle

    try:
        started_at = time.perf_counter()

        # Try to reuse a thread-local browser; fall back to one-shot if needed
        reused_browser = False
        try:
            _, browser = _get_thread_browser()
            reused_browser = True
        except Exception:
            # Fallback: create a new browser for this call
            _pw_ctx = sync_playwright().start()
            browser = _pw_ctx.chromium.launch(headless=True)

        viewport_width = max(int(bundle.dom_snapshot.get("poster_width") or 1440) + 200, 1600)
        viewport_height = max(int(bundle.dom_snapshot.get("poster_height") or 810) + 200, 1200)
        page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})

        # Network interception: physically block any external URL not on
        # the allowlist.  This makes broken-resource consequences flow
        # through downstream validators (layout, colour, design) instead
        # of requiring explicit penalty arithmetic.
        blocked_urls: list[str] = []
        page.route("**/*", _make_network_interceptor(blocked_urls))

        page.on(
            "pageerror",
            lambda exc: bundle.playwright_errors.append({"type": "js_runtime", "message": str(exc)}),
        )
        page.on(
            "console",
            lambda message: bundle.playwright_console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(html_path.resolve().as_uri(), wait_until="load")

        root_selector, poster_measured = _measure_first_playwright_box(page, PAGE_ROOT_SELECTOR_CANDIDATES)
        if not poster_measured:
            raise RuntimeError("Playwright could not measure a page root selector")

        # If the poster is taller than the initial viewport, expand the viewport so
        # that the full poster is rendered and capturable.  Without this the bottom
        # portion of the poster is outside the rendered area and the screenshot will
        # be truncated, causing coordinate normalisation mismatches downstream.
        required_h = int(poster_measured["height"] + poster_measured.get("y", 0.0)) + 60
        if required_h > viewport_height:
            viewport_height = required_h
            page.set_viewport_size({"width": viewport_width, "height": viewport_height})
            # Re-measure after resize in case layout reflowed.
            _, remeasured = _measure_first_playwright_box(page, PAGE_ROOT_SELECTOR_CANDIDATES)
            if remeasured:
                poster_measured = remeasured

        # Override dom_snapshot dimensions with the actual rendered poster size so that
        # _hydrate_playwright_text_blocks normalises text-block coordinates correctly.
        # CSS-extracted values may be missing (e.g. no explicit height) or use
        # modern functions (min(), clamp()) that the static parser cannot evaluate.
        if bundle.dom_snapshot is None:
            bundle.dom_snapshot = {}
        bundle.dom_snapshot["poster_width"] = poster_measured["width"]
        bundle.dom_snapshot["poster_height"] = poster_measured["height"]

        bundle.render_image_path = _capture_playwright_render(bundle, page, poster_measured, html_path)
        bundle.render_blank = _render_image_looks_blank(bundle.render_image_path)
        try:
            bundle.overflowing_elements = page.evaluate(_OVERFLOW_PROBE_JS, poster_measured)
        except Exception as exc:  # pragma: no cover - optional runtime path
            bundle.playwright_errors.append({"type": "overflow_probe_failed", "message": str(exc)})

        overflow_debug_path: Optional[str] = None
        if bundle.overflowing_elements:
            try:
                overflow_debug_path = _capture_playwright_overflow_debug_render(
                    bundle,
                    page,
                    root_selector,
                    poster_measured,
                    viewport_width,
                    viewport_height,
                )
            except Exception as exc:  # pragma: no cover - optional runtime path
                bundle.playwright_errors.append({"type": "overflow_debug_capture_failed", "message": str(exc)})
        if overflow_debug_path:
            bundle.metadata["render_overflow_debug_path"] = overflow_debug_path

        # Playwright's clip is bounded by the actual viewport height.  If the poster
        # is taller than the viewport, the screenshot will be shorter than
        # poster_measured["height"].  Correct poster_height now so that
        # _hydrate_playwright_text_blocks (called below) normalises text-block
        # coordinates against the real render dimensions – otherwise overlay boxes
        # drawn on the render image will be shifted vertically.
        _effective_render_h = viewport_height - poster_measured.get("y", 0.0)
        if _effective_render_h < poster_measured["height"] - 0.5:
            bundle.dom_snapshot["poster_height"] = round(_effective_render_h, 2)

        poster_origin = {"x": poster_measured["x"], "y": poster_measured["y"]}
        page_box = _make_box("page", 0.0, 0.0, poster_measured["width"], poster_measured["height"])
        _replace_named_geometry(bundle, name="page", box=page_box, source="playwright_dom", selector=root_selector)

        single_selectors = {
            "page_content": ".poster-inner, .artboard-inner, [data-entity='page_content'], [data-role='page_content']",
            "title_area": ".title-area",
            "title": ".main-title",
            "main_columns": ".columns-area",
            "instructional_columns": ".columns-area",
            "blank_rectangular_frame": ".blank-rect",
            "eye_illustration": ".eye-wrap",
        }
        for name, selector in single_selectors.items():
            measured = _measure_playwright_box(page, selector)
            if not measured:
                continue
            _replace_named_geometry(
                bundle,
                name=name,
                box=_relative_playwright_box(measured, poster_origin, name),
                source="playwright_dom",
                selector=selector,
            )

        for index, measured in enumerate(_measure_playwright_boxes(page, ".col"), start=1):
            column_name = f"column_{index}"
            _replace_named_geometry(
                bundle,
                name=column_name,
                box=_relative_playwright_box(measured, poster_origin, column_name),
                source="playwright_dom",
                selector=".col",
                metadata={"playwright_index": index - 1},
            )

        for index, measured in enumerate(_measure_playwright_boxes(page, ".heading-row"), start=1):
            heading_area_name = f"column_{index}_heading_area"
            _replace_named_geometry(
                bundle,
                name=heading_area_name,
                box=_relative_playwright_box(measured, poster_origin, heading_area_name),
                source="playwright_dom",
                selector=".heading-row",
                metadata={"playwright_index": index - 1},
            )

        for index, measured in enumerate(_measure_playwright_boxes(page, ".col-heading"), start=1):
            heading_name = f"column_{index}_section_heading"
            _replace_named_geometry(
                bundle,
                name=heading_name,
                box=_relative_playwright_box(measured, poster_origin, heading_name),
                source="playwright_dom",
                selector=".col-heading",
                metadata={"playwright_index": index - 1},
            )

        column_locators = page.locator(".columns-area > .col")
        for index in range(column_locators.count()):
            cluster_measured = _measure_locator_descendant_union_box(column_locators.nth(index))
            if not cluster_measured:
                continue
            cluster_name = f"column_{index + 1}_content_cluster"
            _replace_named_geometry(
                bundle,
                name=cluster_name,
                box=_relative_playwright_box(cluster_measured, poster_origin, cluster_name),
                source="playwright_dom_descendants",
                selector=".columns-area > .col *",
                metadata={"playwright_index": index},
            )

        divider_boxes: List[Dict[str, Any]] = []
        for index, measured in enumerate(_measure_playwright_boxes(page, ".divider"), start=1):
            divider_name = f"divider_{index}"
            divider_box = _relative_playwright_box(measured, poster_origin, divider_name)
            divider_boxes.append(divider_box)
            _replace_named_geometry(
                bundle,
                name=divider_name,
                box=divider_box,
                source="playwright_dom",
                selector=".divider",
                metadata={"playwright_index": index - 1},
            )
            gap_name = f"column_{index}_and_column_{index + 1}_gap"
            _replace_named_geometry(
                bundle,
                name=gap_name,
                box=divider_box,
                source="playwright_dom",
                selector=".divider",
                metadata={"playwright_index": index - 1},
            )

        chibi_measured = _measure_playwright_box(page, ".chibi-wrap")
        if chibi_measured:
            chibi_wrap_box = _relative_playwright_box(chibi_measured, poster_origin, "central_child_wrap")
            for layout_box in bundle.layout_boxes:
                if layout_box.get("name") == "central_child_wrap":
                    _update_box_geometry(layout_box, chibi_wrap_box)
            child_box = _derived_box_from_parent(
                chibi_wrap_box,
                name="central_child_character",
                x_ratio=18.0 / 90.0,
                y_ratio=50.0 / 160.0,
                width_ratio=54.0 / 90.0,
                height_ratio=108.0 / 160.0,
            )
            thought_box = _derived_box_from_parent(
                chibi_wrap_box,
                name="child_thought_bubble",
                x_ratio=44.5 / 90.0,
                y_ratio=2.0 / 160.0,
                width_ratio=41.5 / 90.0,
                height_ratio=32.0 / 160.0,
            )
            _replace_named_geometry(
                bundle,
                name="central_child_character",
                box=child_box,
                source="playwright_derived",
                selector=".chibi-wrap",
            )
            _replace_named_geometry(
                bundle,
                name="child_thought_bubble",
                box=thought_box,
                source="playwright_derived",
                selector=".chibi-wrap",
            )

        blank_rect_element = next((item for item in bundle.detected_elements if item.get("name") == "blank_rectangular_frame"), None)
        if blank_rect_element:
            tiny_person_box = _derived_box_from_parent(
                blank_rect_element,
                name="tiny_person",
                x_ratio=14.0 / 38.0,
                y_ratio=41.0 / 52.0,
                width_ratio=10.0 / 38.0,
                height_ratio=11.0 / 52.0,
            )
            _replace_named_geometry(
                bundle,
                name="tiny_person",
                box=tiny_person_box,
                source="playwright_derived",
                selector=".blank-rect",
            )

        column_content_boxes = [
            _find_named_box(bundle.detected_elements, f"column_{index}_content_cluster")
            for index in range(1, 5)
        ]
        columns_content_box = _union_boxes(
            "columns_content_cluster",
            [box for box in column_content_boxes if box],
        )
        if columns_content_box:
            _replace_named_geometry(
                bundle,
                name="columns_content_cluster",
                box=columns_content_box,
                source="playwright_dom_descendants",
                selector=".columns-area > .col *",
            )

        main_content_boxes = list(column_content_boxes)
        main_content_boxes.extend(
            [
                _find_named_box(bundle.detected_elements, "central_child_character"),
                _find_named_box(bundle.detected_elements, "child_thought_bubble"),
            ]
        )
        main_content_box = _union_boxes("main_content_cluster", [box for box in main_content_boxes if box])
        if main_content_box:
            _replace_named_geometry(
                bundle,
                name="main_content_cluster",
                box=main_content_box,
                source="playwright_dom_descendants",
                selector=".columns-area",
            )

        semantic_elements = _extract_playwright_semantic_elements(page, root_selector)
        for semantic_element in semantic_elements:
            bbox = semantic_element.get("bbox") or []
            if len(bbox) != 4:
                continue
            name = str(semantic_element.get("name") or "").strip()
            if not name:
                continue
            box = _make_box(name, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            _upsert_detected_element(
                bundle,
                name=name,
                role=str(semantic_element.get("role") or name),
                box=box,
                selector=semantic_element.get("selector"),
                scope=semantic_element.get("scope"),
                text=normalize_text(str(semantic_element.get("text") or "")) or None,
                aliases=[str(alias) for alias in (semantic_element.get("aliases") or []) if str(alias).strip()],
                source="playwright_semantic_attr",
                metadata={"declared_semantic": True},
            )

        _hydrate_playwright_text_blocks(bundle, _extract_playwright_text_blocks(page, root_selector))

        # Collect bounding boxes for chart containers and tables for visibility checks
        try:
            element_bboxes = page.evaluate("""() => {
                const results = {};
                // ECharts containers: echarts.init sets _echarts_instance_ on the DOM element
                document.querySelectorAll('[_echarts_instance_]').forEach(el => {
                    const r = el.getBoundingClientRect();
                    const id = el.id || el.className || 'echart';
                    results['chart:' + id] = {x: r.x, y: r.y, width: r.width, height: r.height};
                });
                // Fallback: canvas parents (for charts that don't set _echarts_instance_)
                document.querySelectorAll('canvas').forEach(el => {
                    const parent = el.parentElement;
                    if (parent) {
                        const key = 'chart_canvas:' + (parent.id || parent.className || 'canvas');
                        if (!Object.keys(results).some(k => k === 'chart:' + parent.id)) {
                            const r = parent.getBoundingClientRect();
                            results[key] = {x: r.x, y: r.y, width: r.width, height: r.height};
                        }
                    }
                });
                // HTML tables
                document.querySelectorAll('table').forEach((el, i) => {
                    const r = el.getBoundingClientRect();
                    results['table:' + i] = {x: r.x, y: r.y, width: r.width, height: r.height};
                });
                return results;
            }""")
            bundle.dom_snapshot["element_bboxes"] = element_bboxes
        except Exception:
            pass  # non-critical; don't block validation

        # Extract ECharts runtime data (series type + data) from live instances
        try:
            echarts_runtime = page.evaluate("""() => {
                const results = [];
                document.querySelectorAll('[_echarts_instance_]').forEach(el => {
                    try {
                        const inst = echarts.getInstanceByDom(el);
                        if (!inst) return;
                        const opt = inst.getOption();
                        if (!opt || !opt.series) return;
                        opt.series.forEach((s, si) => {
                            results.push({
                                chart_index: results.length,
                                series_index: si,
                                type: s.type || null,
                                name: s.name || null,
                                data: s.data || null,
                            });
                        });
                    } catch(e) {}
                });
                return results;
            }""")
            bundle.dom_snapshot["echarts_runtime_data"] = echarts_runtime
        except Exception:
            pass

        # Collect bounding boxes for text leaf nodes (for overlap detection)
        try:
            text_bboxes = page.evaluate("""() => {
                const boxes = [];
                const walker = document.createTreeWalker(
                    document.body, NodeFilter.SHOW_TEXT, null);
                let node;
                while (node = walker.nextNode()) {
                    const text = node.textContent.trim();
                    if (!text) continue;
                    const parent = node.parentElement;
                    if (!parent) continue;
                    const tag = parent.tagName.toLowerCase();
                    if (tag === 'script' || tag === 'style') continue;
                    const range = document.createRange();
                    range.selectNodeContents(node);
                    const rects = range.getClientRects();
                    for (let i = 0; i < rects.length; i++) {
                        const r = rects[i];
                        if (r.width > 0 && r.height > 0) {
                            boxes.push({x: r.x, y: r.y, width: r.width, height: r.height});
                        }
                    }
                }
                return boxes;
            }""")
            bundle.dom_snapshot["text_bboxes"] = text_bboxes
        except Exception:
            pass

        page.close()  # Close the page but keep the browser for reuse
        if not reused_browser:
            browser.close()
            _pw_ctx.stop()
        bundle.render_duration_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        # Record which external URLs were physically blocked by the network
        # interceptor.  This is purely for observability / debugging — the
        # score impact comes naturally from broken layout downstream.
        if blocked_urls:
            bundle.metadata["blocked_external_urls"] = blocked_urls[:50]
            bundle.metadata["blocked_external_url_count"] = len(blocked_urls)
    except Exception as exc:  # pragma: no cover - optional runtime path
        # Clean up the page on error
        try:
            page.close()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        if not reused_browser:  # type: ignore[possibly-undefined]
            try:
                browser.close()  # type: ignore[possibly-undefined]
                _pw_ctx.stop()  # type: ignore[possibly-undefined]
            except Exception:
                pass
        bundle.playwright_errors.append({"type": "playwright_runtime_failed", "message": str(exc)})
        bundle.metadata.update(
            {
                "geometry_mode": "static_semantic_inference",
                "geometry_backend_requested": backend,
                "geometry_backend_used": "static",
                "geometry_backend_fallback_reason": f"playwright_runtime_failed: {exc}",
            }
        )
        return bundle

    bundle.metadata.update(
        {
            "geometry_mode": "playwright_dom_boxes",
            "geometry_backend_requested": backend,
            "geometry_backend_used": "playwright",
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "page_root_selector": root_selector,
            "render_backend_used": "playwright_screenshot",
            "render_image_path": bundle.render_image_path,
            "render_overflow_debug_path": bundle.metadata.get("render_overflow_debug_path"),
        }
    )
    return bundle


def _build_static_artifact_bundle(sample_id: str, html_source: str) -> ArtifactBundle:
    soup = BeautifulSoup(html_source, "lxml")
    css_text = _extract_style_text(soup)
    page_root_selector = _resolve_static_page_root_selector(soup)
    page_content_selector = _resolve_static_page_content_selector(page_root_selector)
    poster_declarations = _parse_selector_declarations(css_text, page_root_selector)
    poster_inner_declarations = _parse_selector_declarations(css_text, page_content_selector) if page_content_selector else {}
    title_area_declarations = _parse_selector_declarations(css_text, ".title-area")
    columns_area_declarations = _parse_selector_declarations(css_text, ".columns-area")
    col_declarations = _parse_selector_declarations(css_text, ".col")
    divider_declarations = _parse_selector_declarations(css_text, ".divider")
    chibi_declarations = _parse_selector_declarations(css_text, ".chibi-wrap")
    main_title_declarations = _parse_selector_declarations(css_text, ".main-title")
    heading_declarations = _parse_selector_declarations(css_text, ".col-heading")
    blank_rect_declarations = _parse_selector_declarations(css_text, ".blank-rect")

    main_title = soup.select_one(".main-title")
    title_area = soup.select_one(".title-area")
    columns_area = soup.select_one(".columns-area")
    column_nodes = columns_area.select(":scope > .col") if columns_area else []
    columns = [_column_snapshot(node) for node in column_nodes]

    svg_text = "\n".join(node.decode() for node in soup.select("svg"))
    poster_width = _extract_numeric_px(poster_declarations.get("width"))
    poster_height = _extract_numeric_px(poster_declarations.get("height"))

    dom_snapshot = {
        "document_title": normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else "",
        "main_title": normalize_text(main_title.get_text(" ", strip=True)) if main_title else "",
        "title_text_align": title_area_declarations.get("text-align", ""),
        "page_root_selector": page_root_selector,
        "poster_width": poster_width,
        "poster_height": poster_height,
        "column_count": len(columns),
        "columns": columns,
        "column_headings": [snapshot["heading_texts"][0] for snapshot in columns if snapshot["heading_texts"]],
        "total_bullet_marker_count": len(soup.select(".bullet-marker")),
        "divider_count": len(soup.select(".divider")),
        "storyboard_panel_count": len(soup.select(".story-panel")),
        "colored_circle_count": len(soup.select(".cc")),
        "title_before_columns": _is_before_in_dom(title_area, columns_area),
        "title_centered": title_area_declarations.get("text-align") == "center",
        "columns_evenly_spaced": columns_area_declarations.get("display") == "flex"
        and columns_area_declarations.get("flex-direction") == "row"
        and col_declarations.get("flex") == "1",
        "chibi_centered": chibi_declarations.get("left") == "50%",
        "chibi_absolute": chibi_declarations.get("position") == "absolute",
        "css_text": css_text,
        "svg_text": svg_text,
    }

    layout_boxes: List[Dict[str, Any]] = []
    detected_elements: List[Dict[str, Any]] = []
    if poster_width and poster_height:
        page_box = _make_box("page", 0.0, 0.0, poster_width, poster_height)
        layout_boxes.append(page_box)
        _add_detected_element(
            detected_elements,
            poster_width,
            poster_height,
            name="page",
            role="page",
            box=page_box,
            selector=page_root_selector,
            source="css_absolute",
        )

        inner_top, inner_right, inner_bottom, inner_left = _parse_box_sides(poster_inner_declarations.get("padding"))
        inner_box = _make_box(
            "page_content",
            inner_left,
            inner_top,
            max(poster_width - inner_left - inner_right, 0.0),
            max(poster_height - inner_top - inner_bottom, 0.0),
        )
        layout_boxes.append(inner_box)

        title_font_size = _extract_numeric_px(main_title_declarations.get("font-size")) or 42.0
        title_line_height = _extract_numeric(main_title_declarations.get("line-height")) or 1.1
        title_height = title_font_size * title_line_height
        title_margin_bottom = _extract_numeric_px(title_area_declarations.get("margin-bottom")) or 10.0
        title_text = dom_snapshot["main_title"] or dom_snapshot["document_title"]
        title_width = min(_estimate_text_width(title_text, title_font_size), inner_box["width"] * 0.9)
        title_x = inner_box["x"] + (inner_box["width"] - title_width) / 2.0
        title_y = inner_box["y"]
        title_box = _make_box("title", title_x, title_y, title_width, title_height)
        title_area_box = _make_box(
            "title_area",
            inner_box["x"],
            inner_box["y"],
            inner_box["width"],
            title_height + 14.0,
        )
        layout_boxes.extend([title_area_box, title_box])
        _add_detected_element(
            detected_elements,
            poster_width,
            poster_height,
            name="title",
            role="title",
            box=title_box,
            selector=".main-title",
            scope="page",
            text=title_text,
            aliases=["main_title"],
            source="text_estimate",
        )
        _add_detected_element(
            detected_elements,
            poster_width,
            poster_height,
            name="title_area",
            role="title_area",
            box=title_area_box,
            selector=".title-area",
            scope="page",
            source="layout_inferred",
        )

        columns_y = title_area_box["bottom"] + title_margin_bottom
        columns_height = max(inner_box["bottom"] - columns_y, 0.0)
        columns_box = _make_box("main_columns", inner_box["x"], columns_y, inner_box["width"], columns_height)
        layout_boxes.append(columns_box)
        _add_detected_element(
            detected_elements,
            poster_width,
            poster_height,
            name="main_columns",
            role="main_columns",
            box=columns_box,
            selector=".columns-area",
            scope="page",
            aliases=["instructional_columns_region"],
            source="flex_inferred",
        )

        divider_width = _extract_numeric_px(divider_declarations.get("width")) or 22.0
        direct_children = [node for node in columns_area.find_all(recursive=False) if getattr(node, "name", None)] if columns_area else []
        divider_count = sum(1 for node in direct_children if "divider" in (node.get("class") or []))
        column_count = sum(1 for node in direct_children if "col" in (node.get("class") or [])) or len(columns)
        remaining_width = max(columns_box["width"] - divider_count * divider_width, 0.0)
        column_width = remaining_width / column_count if column_count else 0.0

        cursor_x = columns_box["x"]
        column_boxes: List[Dict[str, Any]] = []
        divider_boxes: List[Dict[str, Any]] = []
        column_index = 0
        divider_index = 0
        heading_height = max((_extract_numeric_px(heading_declarations.get("font-size")) or 17.5) * 1.35 + 8.0, 42.0)
        for child in direct_children:
            classes = child.get("class") or []
            if "col" in classes:
                column_index += 1
                column_name = f"column_{column_index}"
                column_box = _make_box(column_name, cursor_x, columns_box["y"], column_width, columns_box["height"])
                column_boxes.append(column_box)
                layout_boxes.append(column_box)

                inline_style = _parse_inline_style(child.get("style"))
                pad_left = _extract_numeric_px(inline_style.get("padding-left"))
                if pad_left is None:
                    pad_left = _extract_numeric_px(col_declarations.get("padding-left")) or 10.0
                pad_right = _extract_numeric_px(inline_style.get("padding-right"))
                if pad_right is None:
                    pad_right = _extract_numeric_px(col_declarations.get("padding-right")) or 10.0

                heading_box = _make_box(
                    f"{column_name}_heading_area",
                    column_box["x"] + pad_left,
                    column_box["y"],
                    max(column_box["width"] - pad_left - pad_right, 0.0),
                    heading_height,
                )
                layout_boxes.append(heading_box)
                heading_texts = columns[column_index - 1].get("heading_texts", []) if column_index - 1 < len(columns) else []
                heading_text = heading_texts[0] if heading_texts else ""

                _add_detected_element(
                    detected_elements,
                    poster_width,
                    poster_height,
                    name=column_name,
                    role="instructional_column",
                    box=column_box,
                    selector=".col",
                    scope=column_name,
                    aliases=[f"instructional_{column_name}"],
                    source="flex_inferred",
                )
                _add_detected_element(
                    detected_elements,
                    poster_width,
                    poster_height,
                    name=f"{column_name}_heading_area",
                    role="column_heading_area",
                    box=heading_box,
                    selector=".heading-row",
                    scope=column_name,
                    text=heading_text,
                    aliases=[f"{column_name}_heading"],
                    source="flow_inferred",
                )
                if heading_text:
                    heading_text_width = min(_estimate_text_width(heading_text, 17.5), heading_box["width"])
                    heading_text_box = _make_box(
                        f"{column_name}_section_heading",
                        heading_box["x"] + max(heading_box["width"] - heading_text_width, 0.0) / 2.0,
                        heading_box["y"] + 6.0,
                        heading_text_width,
                        max(17.5 * 1.3, 24.0),
                    )
                    layout_boxes.append(heading_text_box)
                    _add_detected_element(
                        detected_elements,
                        poster_width,
                        poster_height,
                        name=f"{column_name}_section_heading",
                        role="section_heading",
                        box=heading_text_box,
                        selector=".col-heading",
                        scope=column_name,
                        text=heading_text,
                        aliases=[f"heading_{column_index}"],
                        source="text_estimate",
                    )
                cursor_x = column_box["right"]
            elif "divider" in classes:
                divider_index += 1
                divider_box = _make_box(
                    f"divider_{divider_index}",
                    cursor_x,
                    columns_box["y"],
                    divider_width,
                    columns_box["height"],
                )
                divider_boxes.append(divider_box)
                layout_boxes.append(divider_box)
                _add_detected_element(
                    detected_elements,
                    poster_width,
                    poster_height,
                    name=f"divider_{divider_index}",
                    role="vertical_separator",
                    box=divider_box,
                    selector=".divider",
                    scope=f"between_column_{divider_index}_and_column_{divider_index + 1}",
                    source="flex_inferred",
                )
                cursor_x = divider_box["right"]

        for gap_index, divider_box in enumerate(divider_boxes, start=1):
            gap_name = f"column_{gap_index}_and_column_{gap_index + 1}_gap"
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name=gap_name,
                role="column_gap",
                box=divider_box,
                selector=".divider",
                scope="page",
                aliases=[f"gap_{gap_index}"],
                source="flex_inferred",
            )

        column_union = _union_boxes("instructional_columns", column_boxes)
        if column_union:
            layout_boxes.append(column_union)
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name="instructional_columns",
                role="instructional_columns",
                box=column_union,
                selector=".columns-area",
                scope="page",
                source="flex_inferred",
                metadata={"child_count": len(column_boxes)},
            )

        if len(column_boxes) >= 3:
            column_3 = column_boxes[2]
            blank_rect_width = _extract_numeric_px(blank_rect_declarations.get("width")) or 38.0
            blank_rect_height = _extract_numeric_px(blank_rect_declarations.get("height")) or 52.0
            blank_rect_box = _make_box(
                "blank_rectangular_frame",
                column_3["x"] + 30.0,
                column_3["y"] + column_3["height"] * 0.62,
                blank_rect_width,
                blank_rect_height,
            )
            layout_boxes.append(blank_rect_box)
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name="blank_rectangular_frame",
                role="blank_rectangular_frame",
                box=blank_rect_box,
                selector=".blank-rect",
                scope="column_3",
                source="flow_inferred",
            )
            tiny_person_box = _make_box(
                "tiny_person",
                blank_rect_box["x"] + 14.0,
                blank_rect_box["y"] + 41.0,
                10.0,
                11.0,
            )
            layout_boxes.append(tiny_person_box)
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name="tiny_person",
                role="tiny_person",
                box=tiny_person_box,
                selector=".blank-rect svg",
                scope="column_3",
                source="svg_local",
            )

        if len(column_boxes) >= 4:
            column_4 = column_boxes[3]
            eye_box = _make_box(
                "eye_illustration",
                column_4["x"] + 20.0,
                column_4["y"] + column_4["height"] * 0.56,
                150.0,
                72.0,
            )
            layout_boxes.append(eye_box)
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name="eye_illustration",
                role="eye_illustration",
                box=eye_box,
                selector=".eye-wrap",
                scope="column_4",
                source="flow_inferred",
            )

        chibi_svg = soup.select_one(".chibi-wrap svg")
        chibi_width = _extract_numeric(chibi_svg.get("width")) if chibi_svg else 90.0
        chibi_height = _extract_numeric(chibi_svg.get("height")) if chibi_svg else 160.0
        chibi_left = chibi_declarations.get("left", "50%")
        left_percent_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", chibi_left.strip()) if chibi_left else None
        left_anchor = poster_width * float(left_percent_match.group(1)) / 100.0 if left_percent_match else poster_width / 2.0
        chibi_bottom = _extract_numeric_px(chibi_declarations.get("bottom")) or 18.0
        chibi_wrap_x = left_anchor - chibi_width / 2.0
        chibi_wrap_y = poster_height - chibi_bottom - chibi_height
        chibi_wrap_box = _make_box("central_child_wrap", chibi_wrap_x, chibi_wrap_y, chibi_width, chibi_height)
        layout_boxes.append(chibi_wrap_box)

        thought_bubble_box = _make_box(
            "child_thought_bubble",
            chibi_wrap_box["x"] + 44.5,
            chibi_wrap_box["y"] + 2.0,
            41.5,
            32.0,
        )
        child_body_box = _make_box(
            "central_child_character",
            chibi_wrap_box["x"] + 18.0,
            chibi_wrap_box["y"] + 50.0,
            54.0,
            108.0,
        )
        layout_boxes.extend([thought_bubble_box, child_body_box])
        _add_detected_element(
            detected_elements,
            poster_width,
            poster_height,
            name="central_child_character",
            role="central_child_character",
            box=child_body_box,
            selector=".chibi-wrap",
            scope="page",
            aliases=["child_character", "chibi"],
            source="css_absolute+svg_local",
        )
        _add_detected_element(
            detected_elements,
            poster_width,
            poster_height,
            name="child_thought_bubble",
            role="child_thought_bubble",
            box=thought_bubble_box,
            selector=".chibi-wrap svg",
            scope="page",
            source="svg_local",
        )

        content_cluster_boxes: List[Dict[str, Any]] = []
        for column_index, column_box in enumerate(column_boxes, start=1):
            heading_box = _find_named_box(layout_boxes, f"column_{column_index}_heading_area")
            if not heading_box:
                continue
            extra_boxes = [
                box
                for box_name in (
                    "blank_rectangular_frame" if column_index == 3 else None,
                    "tiny_person" if column_index == 3 else None,
                    "eye_illustration" if column_index == 4 else None,
                )
                if box_name
                for box in [_find_named_box(layout_boxes, box_name)]
                if box
            ]
            cluster_box = _estimated_column_content_box(
                column_box,
                heading_box,
                columns[column_index - 1] if column_index - 1 < len(columns) else {},
                name=f"column_{column_index}_content_cluster",
                extra_boxes=extra_boxes,
            )
            layout_boxes.append(cluster_box)
            content_cluster_boxes.append(cluster_box)
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name=f"column_{column_index}_content_cluster",
                role="content_cluster",
                box=cluster_box,
                selector=".col",
                scope=f"column_{column_index}",
                aliases=[f"column_{column_index}_content"],
                source="content_estimate",
            )

        columns_content_box = _union_boxes(
            "columns_content_cluster",
            content_cluster_boxes,
        )
        if columns_content_box:
            layout_boxes.append(columns_content_box)
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name="columns_content_cluster",
                role="content_cluster",
                box=columns_content_box,
                selector=".columns-area",
                scope="page",
                aliases=["instructional_content_cluster"],
                source="content_estimate",
            )

        main_content_box = _union_boxes(
            "main_content_cluster",
            [*content_cluster_boxes, child_body_box, thought_bubble_box],
        )
        if main_content_box:
            layout_boxes.append(main_content_box)
            _add_detected_element(
                detected_elements,
                poster_width,
                poster_height,
                name="main_content_cluster",
                role="content_cluster",
                box=main_content_box,
                selector=".columns-area",
                scope="page",
                aliases=["content_cluster"],
                source="content_estimate",
            )

    dom_snapshot["detected_element_names"] = [element["name"] for element in detected_elements]

    bundle = ArtifactBundle(
        sample_id=sample_id,
        html_source=html_source,
        dom_snapshot=dom_snapshot,
        css_snapshot={
            "poster": poster_declarations,
            "poster_inner": poster_inner_declarations,
            "page_root_selector": page_root_selector,
            "title_area": title_area_declarations,
            "columns_area": columns_area_declarations,
            "col": col_declarations,
            "divider": divider_declarations,
            "main_title": main_title_declarations,
            "col_heading": heading_declarations,
            "blank_rect": blank_rect_declarations,
            "chibi_wrap": chibi_declarations,
        },
        layout_boxes=layout_boxes,
        detected_elements=detected_elements,
        metadata={
            "soup_parser": "lxml",
            "geometry_mode": "static_semantic_inference",
        },
    )
    _apply_declared_semantic_overrides(bundle, soup)
    if bundle.dom_snapshot is not None:
        bundle.dom_snapshot["detected_element_names"] = [element["name"] for element in bundle.detected_elements]
    return bundle


def build_artifact_bundle(
    sample_id: str,
    html_source: str,
    *,
    html_path: Optional[Path] = None,
    geometry_backend: str = "auto",
) -> ArtifactBundle:
    bundle = _build_static_artifact_bundle(sample_id, html_source)
    if geometry_backend == "static":
        bundle.metadata.update(
            {
                "geometry_backend_requested": geometry_backend,
                "geometry_backend_used": "static",
            }
        )
        return bundle
    if geometry_backend not in {"auto", "playwright"}:
        raise ValueError(f"Unsupported geometry backend: {geometry_backend}")
    if html_path is None:
        bundle.metadata.update(
            {
                "geometry_backend_requested": geometry_backend,
                "geometry_backend_used": "static",
                "geometry_backend_fallback_reason": "html_path_missing",
            }
        )
        return bundle
    return _apply_playwright_geometry(bundle, html_path, geometry_backend)


def _apply_playwright_geometry_fast(bundle: ArtifactBundle, html_path: Path, backend: str) -> ArtifactBundle:
    """Like _apply_playwright_geometry but skips debug screenshots and disk writes."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover
        bundle.metadata.update(
            {
                "geometry_mode": "static_semantic_inference",
                "geometry_backend_used": "static",
            }
        )
        return bundle

    try:
        started_at = time.perf_counter()

        reused_browser = False
        try:
            _, browser = _get_thread_browser()
            reused_browser = True
        except Exception:
            _pw_ctx = sync_playwright().start()
            browser = _pw_ctx.chromium.launch(headless=True)

        viewport_width = max(int(bundle.dom_snapshot.get("poster_width") or 1440) + 200, 1600)
        viewport_height = max(int(bundle.dom_snapshot.get("poster_height") or 810) + 200, 1200)
        page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})

        blocked_urls: list[str] = []
        page.route("**/*", _make_network_interceptor(blocked_urls))

        page.on(
            "pageerror",
            lambda exc: bundle.playwright_errors.append({"type": "js_runtime", "message": str(exc)}),
        )
        page.on(
            "console",
            lambda message: bundle.playwright_console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(html_path.resolve().as_uri(), wait_until="load")

        root_selector, poster_measured = _measure_first_playwright_box(page, PAGE_ROOT_SELECTOR_CANDIDATES)
        if not poster_measured:
            raise RuntimeError("Playwright could not measure a page root selector")

        required_h = int(poster_measured["height"] + poster_measured.get("y", 0.0)) + 60
        if required_h > viewport_height:
            viewport_height = required_h
            page.set_viewport_size({"width": viewport_width, "height": viewport_height})
            _, remeasured = _measure_first_playwright_box(page, PAGE_ROOT_SELECTOR_CANDIDATES)
            if remeasured:
                poster_measured = remeasured

        if bundle.dom_snapshot is None:
            bundle.dom_snapshot = {}
        bundle.dom_snapshot["poster_width"] = poster_measured["width"]
        bundle.dom_snapshot["poster_height"] = poster_measured["height"]

        bundle.render_image_path = _capture_playwright_render(bundle, page, poster_measured, html_path)
        bundle.render_blank = _render_image_looks_blank(bundle.render_image_path)

        try:
            bundle.overflowing_elements = page.evaluate(_OVERFLOW_PROBE_JS, poster_measured)
        except Exception as exc:  # pragma: no cover
            bundle.playwright_errors.append({"type": "overflow_probe_failed", "message": str(exc)})

        # Skip _capture_playwright_overflow_debug_render entirely

        _effective_render_h = viewport_height - poster_measured.get("y", 0.0)
        if _effective_render_h < poster_measured["height"] - 0.5:
            bundle.dom_snapshot["poster_height"] = round(_effective_render_h, 2)

        poster_origin = {"x": poster_measured["x"], "y": poster_measured["y"]}
        page_box = _make_box("page", 0.0, 0.0, poster_measured["width"], poster_measured["height"])
        _replace_named_geometry(bundle, name="page", box=page_box, source="playwright_dom", selector=root_selector)

        single_selectors = {
            "page_content": ".poster-inner, .artboard-inner, [data-entity='page_content'], [data-role='page_content']",
            "title_area": ".title-area",
            "title": ".main-title",
            "main_columns": ".columns-area",
            "instructional_columns": ".columns-area",
            "blank_rectangular_frame": ".blank-rect",
            "eye_illustration": ".eye-wrap",
        }
        for name, selector in single_selectors.items():
            measured = _measure_playwright_box(page, selector)
            if not measured:
                continue
            _replace_named_geometry(
                bundle,
                name=name,
                box=_relative_playwright_box(measured, poster_origin, name),
                source="playwright_dom",
                selector=selector,
            )

        for index, measured in enumerate(_measure_playwright_boxes(page, ".col"), start=1):
            column_name = f"column_{index}"
            _replace_named_geometry(
                bundle,
                name=column_name,
                box=_relative_playwright_box(measured, poster_origin, column_name),
                source="playwright_dom",
                selector=".col",
                metadata={"playwright_index": index - 1},
            )

        for index, measured in enumerate(_measure_playwright_boxes(page, ".heading-row"), start=1):
            heading_area_name = f"column_{index}_heading_area"
            _replace_named_geometry(
                bundle,
                name=heading_area_name,
                box=_relative_playwright_box(measured, poster_origin, heading_area_name),
                source="playwright_dom",
                selector=".heading-row",
                metadata={"playwright_index": index - 1},
            )

        for index, measured in enumerate(_measure_playwright_boxes(page, ".col-heading"), start=1):
            heading_name = f"column_{index}_section_heading"
            _replace_named_geometry(
                bundle,
                name=heading_name,
                box=_relative_playwright_box(measured, poster_origin, heading_name),
                source="playwright_dom",
                selector=".col-heading",
                metadata={"playwright_index": index - 1},
            )

        column_locators = page.locator(".columns-area > .col")
        for index in range(column_locators.count()):
            cluster_measured = _measure_locator_descendant_union_box(column_locators.nth(index))
            if not cluster_measured:
                continue
            cluster_name = f"column_{index + 1}_content_cluster"
            _replace_named_geometry(
                bundle,
                name=cluster_name,
                box=_relative_playwright_box(cluster_measured, poster_origin, cluster_name),
                source="playwright_dom_descendants",
                selector=".columns-area > .col *",
                metadata={"playwright_index": index},
            )

        for index, measured in enumerate(_measure_playwright_boxes(page, ".divider"), start=1):
            divider_name = f"divider_{index}"
            divider_box = _relative_playwright_box(measured, poster_origin, divider_name)
            _replace_named_geometry(
                bundle,
                name=divider_name,
                box=divider_box,
                source="playwright_dom",
                selector=".divider",
                metadata={"playwright_index": index - 1},
            )
            gap_name = f"column_{index}_and_column_{index + 1}_gap"
            _replace_named_geometry(
                bundle,
                name=gap_name,
                box=divider_box,
                source="playwright_dom",
                selector=".divider",
                metadata={"playwright_index": index - 1},
            )

        chibi_measured = _measure_playwright_box(page, ".chibi-wrap")
        if chibi_measured:
            chibi_wrap_box = _relative_playwright_box(chibi_measured, poster_origin, "central_child_wrap")
            for layout_box in bundle.layout_boxes:
                if layout_box.get("name") == "central_child_wrap":
                    _update_box_geometry(layout_box, chibi_wrap_box)
            child_box = _derived_box_from_parent(
                chibi_wrap_box,
                name="central_child_character",
                x_ratio=18.0 / 90.0,
                y_ratio=50.0 / 160.0,
                width_ratio=54.0 / 90.0,
                height_ratio=108.0 / 160.0,
            )
            thought_box = _derived_box_from_parent(
                chibi_wrap_box,
                name="child_thought_bubble",
                x_ratio=44.5 / 90.0,
                y_ratio=2.0 / 160.0,
                width_ratio=41.5 / 90.0,
                height_ratio=32.0 / 160.0,
            )
            _replace_named_geometry(
                bundle,
                name="central_child_character",
                box=child_box,
                source="playwright_derived",
                selector=".chibi-wrap",
            )
            _replace_named_geometry(
                bundle,
                name="child_thought_bubble",
                box=thought_box,
                source="playwright_derived",
                selector=".chibi-wrap",
            )

        blank_rect_element = next((item for item in bundle.detected_elements if item.get("name") == "blank_rectangular_frame"), None)
        if blank_rect_element:
            tiny_person_box = _derived_box_from_parent(
                blank_rect_element,
                name="tiny_person",
                x_ratio=14.0 / 38.0,
                y_ratio=41.0 / 52.0,
                width_ratio=10.0 / 38.0,
                height_ratio=11.0 / 52.0,
            )
            _replace_named_geometry(
                bundle,
                name="tiny_person",
                box=tiny_person_box,
                source="playwright_derived",
                selector=".blank-rect",
            )

        column_content_boxes = [
            _find_named_box(bundle.detected_elements, f"column_{index}_content_cluster")
            for index in range(1, 5)
        ]
        columns_content_box = _union_boxes(
            "columns_content_cluster",
            [box for box in column_content_boxes if box],
        )
        if columns_content_box:
            _replace_named_geometry(
                bundle,
                name="columns_content_cluster",
                box=columns_content_box,
                source="playwright_dom_descendants",
                selector=".columns-area > .col *",
            )

        main_content_boxes = list(column_content_boxes)
        main_content_boxes.extend(
            [
                _find_named_box(bundle.detected_elements, "central_child_character"),
                _find_named_box(bundle.detected_elements, "child_thought_bubble"),
            ]
        )
        main_content_box = _union_boxes("main_content_cluster", [box for box in main_content_boxes if box])
        if main_content_box:
            _replace_named_geometry(
                bundle,
                name="main_content_cluster",
                box=main_content_box,
                source="playwright_dom_descendants",
                selector=".columns-area",
            )

        semantic_elements = _extract_playwright_semantic_elements(page, root_selector)
        for semantic_element in semantic_elements:
            bbox = semantic_element.get("bbox") or []
            if len(bbox) != 4:
                continue
            name = str(semantic_element.get("name") or "").strip()
            if not name:
                continue
            box = _make_box(name, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            _upsert_detected_element(
                bundle,
                name=name,
                role=str(semantic_element.get("role") or name),
                box=box,
                selector=semantic_element.get("selector"),
                scope=semantic_element.get("scope"),
                text=normalize_text(str(semantic_element.get("text") or "")) or None,
                aliases=[str(alias) for alias in (semantic_element.get("aliases") or []) if str(alias).strip()],
                source="playwright_semantic_attr",
                metadata={"declared_semantic": True},
            )

        _hydrate_playwright_text_blocks(bundle, _extract_playwright_text_blocks(page, root_selector))

        # Collect bounding boxes for chart containers and tables for visibility checks
        try:
            element_bboxes = page.evaluate("""() => {
                const results = {};
                document.querySelectorAll('[_echarts_instance_]').forEach(el => {
                    const r = el.getBoundingClientRect();
                    const id = el.id || el.className || 'echart';
                    results['chart:' + id] = {x: r.x, y: r.y, width: r.width, height: r.height};
                });
                document.querySelectorAll('canvas').forEach(el => {
                    const parent = el.parentElement;
                    if (parent) {
                        const key = 'chart_canvas:' + (parent.id || parent.className || 'canvas');
                        if (!Object.keys(results).some(k => k === 'chart:' + parent.id)) {
                            const r = parent.getBoundingClientRect();
                            results[key] = {x: r.x, y: r.y, width: r.width, height: r.height};
                        }
                    }
                });
                document.querySelectorAll('table').forEach((el, i) => {
                    const r = el.getBoundingClientRect();
                    results['table:' + i] = {x: r.x, y: r.y, width: r.width, height: r.height};
                });
                return results;
            }""")
            bundle.dom_snapshot["element_bboxes"] = element_bboxes
        except Exception:
            pass

        # Extract ECharts runtime data (series type + data) from live instances
        try:
            echarts_runtime = page.evaluate("""() => {
                const results = [];
                document.querySelectorAll('[_echarts_instance_]').forEach(el => {
                    try {
                        const inst = echarts.getInstanceByDom(el);
                        if (!inst) return;
                        const opt = inst.getOption();
                        if (!opt || !opt.series) return;
                        opt.series.forEach((s, si) => {
                            results.push({
                                chart_index: results.length,
                                series_index: si,
                                type: s.type || null,
                                name: s.name || null,
                                data: s.data || null,
                            });
                        });
                    } catch(e) {}
                });
                return results;
            }""")
            bundle.dom_snapshot["echarts_runtime_data"] = echarts_runtime
        except Exception:
            pass

        # Collect bounding boxes for text leaf nodes (for overlap detection)
        try:
            text_bboxes = page.evaluate("""() => {
                const boxes = [];
                const walker = document.createTreeWalker(
                    document.body, NodeFilter.SHOW_TEXT, null);
                let node;
                while (node = walker.nextNode()) {
                    const text = node.textContent.trim();
                    if (!text) continue;
                    const parent = node.parentElement;
                    if (!parent) continue;
                    const tag = parent.tagName.toLowerCase();
                    if (tag === 'script' || tag === 'style') continue;
                    const range = document.createRange();
                    range.selectNodeContents(node);
                    const rects = range.getClientRects();
                    for (let i = 0; i < rects.length; i++) {
                        const r = rects[i];
                        if (r.width > 0 && r.height > 0) {
                            boxes.push({x: r.x, y: r.y, width: r.width, height: r.height});
                        }
                    }
                }
                return boxes;
            }""")
            bundle.dom_snapshot["text_bboxes"] = text_bboxes
        except Exception:
            pass

        page.close()
        if not reused_browser:
            browser.close()
            _pw_ctx.stop()
        bundle.render_duration_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    except Exception as exc:  # pragma: no cover
        try:
            page.close()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        if not reused_browser:  # type: ignore[possibly-undefined]
            try:
                browser.close()  # type: ignore[possibly-undefined]
                _pw_ctx.stop()  # type: ignore[possibly-undefined]
            except Exception:
                pass
        bundle.playwright_errors.append({"type": "playwright_runtime_failed", "message": str(exc)})
        bundle.metadata.update(
            {
                "geometry_mode": "static_semantic_inference",
                "geometry_backend_used": "static",
            }
        )
        return bundle

    bundle.metadata.update(
        {
            "geometry_mode": "playwright_dom_boxes",
            "geometry_backend_used": "playwright",
            "render_image_path": bundle.render_image_path,
        }
    )
    return bundle


def build_artifact_bundle_fast(
    sample_id: str,
    html_source: str,
    *,
    html_path: Optional[Path] = None,
) -> ArtifactBundle:
    """Fast artifact bundle for RL reward computation.
    Skips disk artifact writes. Only produces in-memory data for scoring."""
    bundle = _build_static_artifact_bundle(sample_id, html_source)
    if html_path is None:
        return bundle
    return _apply_playwright_geometry_fast(bundle, html_path, "playwright")