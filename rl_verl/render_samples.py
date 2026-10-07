#!/usr/bin/env python3
"""Render saved HTML samples to PNG images for visual inspection.

During GDPO training, reward_fn.py periodically saves generated HTML files
to RENDER_SAMPLE_DIR. This script renders them to PNG using Playwright.

Usage:
    python rl_verl/render_samples.py /path/to/render_samples
    python rl_verl/render_samples.py /path/to/render_samples --limit 10
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def render_html_to_png(html_path: Path, png_path: Path, width: int = 1280, height: int = 960):
    """Render a single HTML file to PNG using Playwright."""
    from playwright.sync_api import sync_playwright

    html_content = html_path.read_text(encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html_content, wait_until="networkidle")
        page.screenshot(path=str(png_path), full_page=False)
        browser.close()


def render_all(render_dir: str, limit: int = 0):
    """Render all unrendered HTML files in the directory."""
    rdir = Path(render_dir)
    if not rdir.exists():
        print(f"Render directory does not exist: {render_dir}")
        return

    html_files = sorted(rdir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not html_files:
        print("No HTML files found.")
        return

    if limit > 0:
        html_files = html_files[:limit]

    png_dir = rdir / "png"
    png_dir.mkdir(exist_ok=True)

    rendered = 0
    for html_path in html_files:
        png_path = png_dir / html_path.with_suffix(".png").name
        if png_path.exists():
            continue
        try:
            print(f"  Rendering {html_path.name} ...", end=" ")
            render_html_to_png(html_path, png_path)
            print("OK")
            rendered += 1
        except Exception as e:
            print(f"FAILED: {e}")

    total_png = len(list(png_dir.glob("*.png")))
    print(f"\nRendered {rendered} new images. Total PNGs: {total_png}")
    print(f"View at: {png_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render HTML samples to PNG")
    parser.add_argument("render_dir", help="Directory containing saved HTML files")
    parser.add_argument("--limit", type=int, default=10, help="Max files to render (0=all)")
    args = parser.parse_args()
    render_all(args.render_dir, args.limit)
