"""Training monitor — generates live HTML reports during GDPO training.

Records per-sample, per-step reward breakdowns and generates a continuously
updated HTML report with:
  - Reference image + rendered images for EACH refine step (embedded as base64)
  - Per-validator score bars
  - VLM rationale text
  - Score trend across training steps (inline SVG line charts)
  - Summary statistics

Report files are named report_step_{N}.html so different training steps
can be compared side by side.

Design: Each Ray worker writes per-sample JSON files to a shared directory.
The report generator reads all JSON files to build the aggregated view.
This avoids the problem of each worker having its own in-memory singleton.

Configuration via environment variables:
  REWARD_MONITOR_ENABLED   (0/1, default 1)
  REWARD_MONITOR_INTERVAL  (report every N batch steps, default 10)
  REWARD_MONITOR_OUTPUT    (output directory, default {OUTPUT_DIR}/monitor)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Score dimension keys (short names from reward_fn.py)
SCORE_KEYS = [
    "execution", "text", "boundary", "layout",
    "tablechart", "whitespace", "design", "vlm",
]

# Colors for each dimension in charts/bars
SCORE_COLORS = {
    "execution": "#e74c3c",
    "text": "#3498db",
    "boundary": "#2ecc71",
    "layout": "#9b59b6",
    "tablechart": "#f39c12",
    "whitespace": "#1abc9c",
    "design": "#e67e22",
    "vlm": "#8e44ad",
}


def _is_enabled() -> bool:
    return os.environ.get("REWARD_MONITOR_ENABLED", "1") == "1"


def _get_interval() -> int:
    return int(os.environ.get("REWARD_MONITOR_INTERVAL", "10"))


def _get_output_dir() -> Path:
    default = os.path.join(
        os.environ.get("OUTPUT_DIR", "training/checkpoints/verl_gdpo"),
        "monitor",
    )
    return Path(os.environ.get("REWARD_MONITOR_OUTPUT", default))


def _img_to_base64(path: str | Path) -> str | None:
    """Read an image file and return a data URI string, or None if missing."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        data = p.read_bytes()
        ext = p.suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/png")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


class TrainingMonitor:
    """Collects per-sample reward data during training and generates HTML reports.

    Multi-worker safe: each record() call writes a JSON file to disk.
    generate_report() reads all files to build the aggregated report.
    """

    def __init__(self):
        self.enabled = _is_enabled()
        self.interval = _get_interval()
        self.output_dir = _get_output_dir()
        self._last_report_step = -1

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "images").mkdir(exist_ok=True)
            (self.output_dir / "records").mkdir(exist_ok=True)
            print(
                f"[TrainingMonitor] enabled: interval={self.interval}, output={self.output_dir}",
                flush=True,
            )

    def record(
        self,
        global_step: int,
        sample_id: str,
        result: dict,
        render_image_path: str | None = None,
        ref_image_path: str | None = None,
        refine_step: int = 0,
    ):
        """Record a reward result for a sample at a given training step.

        Args:
            refine_step: which refine step (0=generate, 1=refine1, etc.)
        Writes a JSON file per (sample_id, step, refine_step) so multiple workers
        can contribute.
        """
        if not self.enabled:
            return

        # Write per-sample record as JSON
        record_dir = self.output_dir / "records" / sample_id
        record_dir.mkdir(parents=True, exist_ok=True)
        record_file = record_dir / f"step_{global_step}_r{refine_step}_{os.getpid()}.json"
        try:
            # Filter out non-serializable fields
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    clean_result[k] = v
            record_file.write_text(
                json.dumps({
                    "step": global_step,
                    "refine_step": refine_step,
                    "sample_id": sample_id,
                    **clean_result,
                }),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[TrainingMonitor] record write failed: {exc}", flush=True)

        # Copy images to output dir for the report
        img_dir = self.output_dir / "images" / sample_id
        img_dir.mkdir(parents=True, exist_ok=True)

        # Save render image per (step, refine_step)
        if render_image_path and os.path.isfile(render_image_path):
            dest = img_dir / f"render_step{global_step}_r{refine_step}.png"
            try:
                shutil.copy2(render_image_path, dest)
            except Exception:
                pass

        if ref_image_path and os.path.isfile(ref_image_path):
            dest = img_dir / "reference.png"
            if not dest.exists():
                try:
                    shutil.copy2(ref_image_path, dest)
                except Exception:
                    pass

    def should_report(self, global_step: int) -> bool:
        """Check if we should generate a report at this step."""
        if not self.enabled:
            return False
        if global_step <= self._last_report_step:
            return False
        return global_step % self.interval == 0 or global_step == 0

    def maybe_generate_report(self, global_step: int):
        """Generate report if interval reached."""
        if self.should_report(global_step):
            self.generate_report(global_step)
            self._last_report_step = global_step

    def _load_history_from_disk(self) -> dict[str, list[tuple[int, int, dict]]]:
        """Load all per-sample records from disk.

        Returns: {sample_id: [(global_step, refine_step, result_dict), ...]}
        """
        history: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)
        records_dir = self.output_dir / "records"
        if not records_dir.exists():
            return history

        seen = set()  # deduplicate (sample_id, step, refine_step)
        for sample_dir in sorted(records_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_id = sample_dir.name
            for record_file in sorted(sample_dir.glob("step_*_r*.json")):
                try:
                    data = json.loads(record_file.read_text(encoding="utf-8"))
                    step = data.pop("step", 0)
                    refine_step = data.pop("refine_step", 0)
                    sid = data.pop("sample_id", sample_id)
                    key = (sid, step, refine_step)
                    if key not in seen:
                        seen.add(key)
                        history[sid].append((step, refine_step, data))
                except Exception:
                    continue

        # Sort entries by (step, refine_step)
        for sid in history:
            history[sid].sort(key=lambda x: (x[0], x[1]))

        return history

    def generate_report(self, global_step: int = 0):
        """Generate the HTML monitoring report with step suffix."""
        history = self._load_history_from_disk()
        if not history:
            return

        report_path = self.output_dir / f"report_step_{global_step}.html"
        # Also write a latest symlink/copy
        latest_path = self.output_dir / "report_latest.html"
        try:
            html = self._build_report_html(history, global_step)
            report_path.write_text(html, encoding="utf-8")
            # Copy as latest
            latest_path.write_text(html, encoding="utf-8")
            print(
                f"[TrainingMonitor] report: {report_path} ({len(history)} samples)",
                flush=True,
            )
        except Exception as exc:
            print(f"[TrainingMonitor] Failed to generate report: {exc}", flush=True)
            import traceback
            traceback.print_exc()

        # Also save aggregated history JSON
        try:
            json_path = self.output_dir / "history.json"
            data = {sid: [(s, r, d) for s, r, d in entries] for sid, entries in history.items()}
            json_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _build_report_html(self, history: dict[str, list[tuple[int, int, dict]]], current_step: int) -> str:
        """Build the full HTML report string with embedded base64 images."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        all_steps = set()
        for entries in history.values():
            for step, _, _ in entries:
                all_steps.add(step)
        max_step = max(all_steps) if all_steps else 0

        # Per-sample sections (cap to avoid huge reports)
        MAX_REPORT_SAMPLES = 50
        sample_ids = sorted(history.keys())
        if len(sample_ids) > MAX_REPORT_SAMPLES:
            # Pick evenly spaced samples to cover the full range
            step = len(sample_ids) / MAX_REPORT_SAMPLES
            sample_ids = [sample_ids[int(i * step)] for i in range(MAX_REPORT_SAMPLES)]
        sample_sections = []
        for sample_id in sample_ids:
            entries = history[sample_id]
            sample_sections.append(self._build_sample_section(sample_id, entries))

        # Summary section
        summary_html = self._build_summary_section(history)

        return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>GDPO Training Monitor — Step {current_step}</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1800px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #2c3e50; }}
.sample {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.sample h3 {{ margin: 0 0 12px; color: #34495e; }}
.images {{ display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
.img-box {{ text-align: center; }}
.img-box img {{ max-width: 250px; max-height: 200px; border-radius: 4px; border: 1px solid #ddd; object-fit: contain; }}
.img-box p {{ font-size: 11px; color: #666; margin: 0 0 4px; }}
.scores {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
.score-bar {{ display: flex; align-items: center; gap: 4px; font-size: 13px; }}
.score-bar .bar {{ height: 16px; border-radius: 3px; min-width: 2px; }}
.score-bar .label {{ min-width: 80px; text-align: right; color: #666; }}
.score-bar .val {{ min-width: 40px; font-weight: 600; }}
.rationale {{ font-size: 12px; color: #555; background: #f0f0f0; padding: 8px; border-radius: 4px; margin-top: 8px; white-space: pre-wrap; max-height: 120px; overflow-y: auto; }}
.summary {{ background: #2c3e50; color: white; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.summary table {{ width: 100%; color: white; border-collapse: collapse; }}
.summary td, .summary th {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
.step-compare {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; }}
.step-card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; background: #fafafa; min-width: 200px; }}
.step-card h4 {{ margin: 0 0 6px; font-size: 13px; color: #555; }}
.delta {{ font-size: 12px; font-weight: bold; }}
.delta.pos {{ color: #27ae60; }}
.delta.neg {{ color: #e74c3c; }}
</style></head>
<body>
<h1>GDPO Training Monitor</h1>
<p style="color:#666">Report for step {current_step} | Generated: {timestamp} | Samples: {len(history)} | Max step seen: {max_step}</p>
<hr>
{summary_html}
{"".join(sample_sections)}
</body></html>'''

    def _build_sample_section(self, sample_id: str, entries: list[tuple[int, int, dict]]) -> str:
        """Build HTML for one sample's monitoring card with per-step images."""
        # Group entries by global_step
        by_global_step: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for gstep, rstep, result in entries:
            by_global_step[gstep].append((rstep, result))

        latest_gstep = max(by_global_step.keys())
        latest_refine_entries = sorted(by_global_step[latest_gstep], key=lambda x: x[0])
        # Use the last refine step's result for the score bars
        _, latest_result = latest_refine_entries[-1]

        # Reference image (base64 embedded)
        img_dir = self.output_dir / "images" / sample_id
        ref_path = img_dir / "reference.png"
        ref_b64 = _img_to_base64(ref_path)

        # Build image gallery: ref + all refine step renders at latest global step
        # Always try to show all images from disk, even if records are incomplete
        images_html = ""
        if ref_b64:
            images_html += f'''<div class="img-box">
                <p>Reference</p>
                <img src="{ref_b64}">
            </div>\n'''

        # Collect scores from records for labels
        rstep_scores = {rstep: r.get("score", 0.0) for rstep, r in latest_refine_entries}

        # Show images for all refine steps found on disk (r0, r1, r2, ...)
        for rstep_idx in range(max(len(latest_refine_entries), 3)):
            render_path = img_dir / f"render_step{latest_gstep}_r{rstep_idx}.png"
            render_b64 = _img_to_base64(render_path)
            if not render_b64:
                continue
            step_label = "Generate" if rstep_idx == 0 else f"Refine {rstep_idx}"
            step_score = rstep_scores.get(rstep_idx, 0.0)
            images_html += f'''<div class="img-box">
                <p>{step_label} (step {latest_gstep}, score={float(step_score):.3f})</p>
                <img src="{render_b64}">
            </div>\n'''

        # Score bars for latest result (final refine step)
        bars_html = ""
        for key in SCORE_KEYS:
            val = latest_result.get(key, 0.0)
            if val is None:
                val = 0.0
            color = SCORE_COLORS.get(key, "#95a5a6")
            width = max(2, int(float(val) * 200))
            bars_html += f'''<div class="score-bar">
                <span class="label">{key}</span>
                <div class="bar" style="width:{width}px;background:{color}"></div>
                <span class="val">{float(val):.3f}</span>
            </div>\n'''

        weighted = latest_result.get("score", 0.0)

        # Per-step comparison cards (generate vs refine at latest step)
        step_compare_html = ""
        if len(latest_refine_entries) >= 2:
            step_compare_html = '<div class="step-compare">'
            prev_score = None
            for rstep, r in latest_refine_entries:
                step_label = "Generate" if rstep == 0 else f"Refine {rstep}"
                s = float(r.get("score", 0.0))
                delta_html = ""
                if prev_score is not None:
                    delta = s - prev_score
                    cls = "pos" if delta >= 0 else "neg"
                    sign = "+" if delta >= 0 else ""
                    delta_html = f' <span class="delta {cls}">{sign}{delta:.3f}</span>'
                prev_score = s
                # Mini score list
                mini_scores = " | ".join(
                    f"{k}: {float(r.get(k, 0.0)):.2f}" for k in SCORE_KEYS
                    if r.get(k)
                )
                step_compare_html += f'''<div class="step-card">
                    <h4>{step_label}: {s:.3f}{delta_html}</h4>
                    <div style="font-size:11px;color:#777">{mini_scores}</div>
                </div>'''
            step_compare_html += '</div>'

        # VLM rationale
        rationale = latest_result.get("vlm_rationale", "")
        rationale_html = f'<div class="rationale">{_escape(str(rationale))}</div>' if rationale else ""

        # Trend SVG (score over global steps, using final refine step)
        trend_entries = []
        for gstep in sorted(by_global_step.keys()):
            refine_entries = sorted(by_global_step[gstep], key=lambda x: x[0])
            _, r = refine_entries[-1]  # last refine step
            trend_entries.append((gstep, r))
        trend_svg = self._build_trend_svg(trend_entries)

        return f'''<div class="sample">
    <h3>{sample_id} <span style="font-weight:normal;color:#888">(step {latest_gstep}, weighted={float(weighted):.3f})</span></h3>
    <div class="images">{images_html}</div>
    {step_compare_html}
    <div class="scores">{bars_html}</div>
    {rationale_html}
    <div class="trend">{trend_svg}</div>
</div>'''

    def _build_trend_svg(self, entries: list[tuple[int, dict]]) -> str:
        """Build an inline SVG line chart showing score trends over steps."""
        if len(entries) < 2:
            return ""

        W, H = 600, 150
        PAD_L, PAD_R, PAD_T, PAD_B = 50, 20, 10, 25
        plot_w = W - PAD_L - PAD_R
        plot_h = H - PAD_T - PAD_B

        steps = [e[0] for e in entries]
        min_step, max_step = min(steps), max(steps)
        step_range = max(max_step - min_step, 1)

        def x(step):
            return PAD_L + (step - min_step) / step_range * plot_w

        def y(val):
            return PAD_T + (1.0 - float(val)) * plot_h

        # Grid lines
        grid = ""
        for v in [0, 0.25, 0.5, 0.75, 1.0]:
            yv = y(v)
            grid += f'<line x1="{PAD_L}" y1="{yv}" x2="{W-PAD_R}" y2="{yv}" stroke="#eee" stroke-width="1"/>'
            grid += f'<text x="{PAD_L-5}" y="{yv+4}" text-anchor="end" fill="#999" font-size="10">{v:.1f}</text>'

        # Lines for "score" (weighted) and each dimension
        lines = ""
        # Weighted score line (thick, black)
        points = " ".join(f"{x(s)},{y(r.get('score', 0))}" for s, r in entries)
        lines += f'<polyline points="{points}" fill="none" stroke="#2c3e50" stroke-width="2"/>'

        # Per-dimension lines (thin, colored)
        for key in SCORE_KEYS:
            color = SCORE_COLORS.get(key, "#999")
            points = " ".join(f"{x(s)},{y(r.get(key, 0))}" for s, r in entries)
            lines += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1" opacity="0.6"/>'

        # Step labels
        step_labels = ""
        for s in [min_step, max_step]:
            step_labels += f'<text x="{x(s)}" y="{H-2}" text-anchor="middle" fill="#999" font-size="10">step {s}</text>'

        return f'''<svg width="{W}" height="{H}" style="background:white;border:1px solid #eee;border-radius:4px">
{grid}{lines}{step_labels}
<text x="{W-PAD_R}" y="{PAD_T+10}" text-anchor="end" fill="#2c3e50" font-size="10" font-weight="bold">weighted</text>
</svg>'''

    def _build_summary_section(self, history: dict[str, list[tuple[int, int, dict]]]) -> str:
        """Build summary table averaging latest scores across all samples."""
        if not history:
            return ""

        # Collect latest result per sample (last refine step of last global step)
        all_latest = []
        for entries in history.values():
            # entries sorted by (gstep, rstep), last entry is latest
            _, _, result = entries[-1]
            all_latest.append(result)

        n = len(all_latest)
        rows = ""
        for key in ["score", "reward_unweighted"] + SCORE_KEYS:
            vals = [float(r.get(key, 0.0) or 0.0) for r in all_latest]
            avg = sum(vals) / len(vals) if vals else 0.0
            mn = min(vals) if vals else 0.0
            mx = max(vals) if vals else 0.0
            rows += f"<tr><td>{key}</td><td>{avg:.3f}</td><td>{mn:.3f}</td><td>{mx:.3f}</td></tr>\n"

        return f'''<div class="summary">
<h3 style="margin:0 0 8px;color:white">Summary ({n} samples)</h3>
<table><tr><th>Dimension</th><th>Mean</th><th>Min</th><th>Max</th></tr>
{rows}</table></div>'''


def _escape(text: str) -> str:
    """Basic HTML escape."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Module-level singleton
_monitor: TrainingMonitor | None = None


def get_monitor() -> TrainingMonitor:
    """Get or create the singleton TrainingMonitor."""
    global _monitor
    if _monitor is None:
        _monitor = TrainingMonitor()
    return _monitor
