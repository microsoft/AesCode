from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from verifier.artifacts.paths import get_sample_artifact_dir
from verifier.types import RefinedSample, ValidationReport
from verifier.pipeline.runner import load_samples_from_jsonl, validate_sample


def _load_html_manifest(manifest_path: Optional[Path]) -> Dict[str, str]:
    if manifest_path is None:
        return {}
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if manifest_path.suffix.lower() == ".jsonl":
        mapping: Dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("id") or payload.get("sample_id") or "")
            html_path = str(payload.get("html") or payload.get("html_path") or "")
            if sample_id and html_path:
                mapping[sample_id] = html_path
        return mapping
    payload = json.loads(text)
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    raise ValueError("HTML manifest must be a JSON object or JSONL mapping sample ids to html paths.")


def _resolve_html_path(
    sample: RefinedSample,
    *,
    html_dir: Optional[Path],
    html_pattern: str,
    html_manifest: Dict[str, str],
    manifest_base_dir: Optional[Path],
) -> Path:
    manifest_match = html_manifest.get(sample.id)
    if manifest_match:
        manifest_path = Path(manifest_match)
        if manifest_path.is_absolute():
            return manifest_path
        base_dir = manifest_base_dir or Path.cwd()
        return (base_dir / manifest_path).resolve()
    candidate_text = html_pattern.format(sample_id=sample.id, application=sample.application)
    candidate_path = Path(candidate_text)
    if candidate_path.is_absolute():
        return candidate_path
    if html_dir is not None:
        return (html_dir / candidate_path).resolve()
    return candidate_path.resolve()


def _serialize_report(report: ValidationReport) -> Dict[str, Any]:
    return {
        "sample_id": report.sample_id,
        "execution_pass": report.execution_pass,
        "quality_pass": report.quality_pass,
        "overall_pass": report.overall_pass,
        "hard_failures": list(report.hard_failures),
        "summary": dict(report.summary),
        "results": [
            {
                "requirement_id": result.requirement_id,
                "status": result.status.value,
                "score": result.score,
                "message": result.message,
                "measured": result.measured,
                "expected": result.expected,
                "evidence": result.evidence,
            }
            for result in report.results
        ],
    }


def _select_samples(
    samples: Sequence[RefinedSample],
    *,
    sample_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> List[RefinedSample]:
    selected = list(samples)

    if sample_ids:
        requested_ids = [sample_id for sample_id in sample_ids if sample_id]
        sample_index = {sample.id: sample for sample in selected}
        missing_ids = [sample_id for sample_id in requested_ids if sample_id not in sample_index]
        if missing_ids:
            raise ValueError(f"Sample ids not found in JSONL: {', '.join(missing_ids)}")
        selected = [sample_index[sample_id] for sample_id in requested_ids]

    if shuffle:
        import random as _random
        _random.shuffle(selected)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")
        selected = selected[:limit]

    return selected


def _compare_reports(baseline_backend: str, baseline: ValidationReport, candidate_backend: str, candidate: ValidationReport) -> Dict[str, Any]:
    baseline_index = {result.requirement_id: result for result in baseline.results}
    candidate_index = {result.requirement_id: result for result in candidate.results}
    requirement_ids = sorted(set(baseline_index) | set(candidate_index))
    differences: List[Dict[str, Any]] = []
    for requirement_id in requirement_ids:
        left = baseline_index.get(requirement_id)
        right = candidate_index.get(requirement_id)
        left_status = left.status.value if left else None
        right_status = right.status.value if right else None
        left_score = left.score if left else None
        right_score = right.score if right else None
        if left_status == right_status and left_score == right_score:
            continue
        differences.append(
            {
                "requirement_id": requirement_id,
                "baseline_status": left_status,
                "candidate_status": right_status,
                "baseline_score": left_score,
                "candidate_score": right_score,
            }
        )
    return {
        "baseline_backend": baseline_backend,
        "candidate_backend": candidate_backend,
        "baseline_execution_pass": baseline.execution_pass,
        "candidate_execution_pass": candidate.execution_pass,
        "baseline_quality_pass": baseline.quality_pass,
        "candidate_quality_pass": candidate.quality_pass,
        "baseline_overall_pass": baseline.overall_pass,
        "candidate_overall_pass": candidate.overall_pass,
        "baseline_hard_failures": list(baseline.hard_failures),
        "candidate_hard_failures": list(candidate.hard_failures),
        "difference_count": len(differences),
        "differences": differences,
    }


def validate_batch(
    jsonl_path: Path,
    *,
    html_dir: Optional[Path] = None,
    html_pattern: str = "{sample_id}.html",
    html_manifest_path: Optional[Path] = None,
    artifact_backends: Optional[List[str]] = None,
    artifact_root: Optional[Path] = None,
    sample_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> Dict[str, Any]:
    active_backends = artifact_backends or ["static", "playwright"]
    samples = _select_samples(
        load_samples_from_jsonl(jsonl_path),
        sample_ids=sample_ids,
        limit=limit,
        shuffle=shuffle,
    )
    html_manifest = _load_html_manifest(html_manifest_path)
    manifest_base_dir = html_manifest_path.parent if html_manifest_path else None

    rows: List[Dict[str, Any]] = []
    processed_count = 0
    missing_html: List[str] = []
    report_count_by_backend = {backend: 0 for backend in active_backends}
    inconsistent_pairs = 0

    for sample in samples:
        html_path = _resolve_html_path(
            sample,
            html_dir=html_dir,
            html_pattern=html_pattern,
            html_manifest=html_manifest,
            manifest_base_dir=manifest_base_dir,
        )
        if not html_path.exists():
            missing_html.append(sample.id)
            rows.append(
                {
                    "sample_id": sample.id,
                    "html_path": str(html_path),
                    "error": "html_not_found",
                }
            )
            continue

        processed_count += 1
        reports: Dict[str, ValidationReport] = {}
        serialized_reports: Dict[str, Any] = {}
        for backend in active_backends:
            report = validate_sample(sample, html_path, artifact_backend=backend, artifact_root=artifact_root)
            reports[backend] = report
            serialized_reports[backend] = _serialize_report(report)
            report_count_by_backend[backend] += 1

        comparisons: List[Dict[str, Any]] = []
        if len(active_backends) >= 2:
            baseline_backend = active_backends[0]
            baseline_report = reports[baseline_backend]
            for candidate_backend in active_backends[1:]:
                comparison = _compare_reports(
                    baseline_backend,
                    baseline_report,
                    candidate_backend,
                    reports[candidate_backend],
                )
                if comparison["difference_count"] > 0:
                    inconsistent_pairs += 1
                comparisons.append(comparison)

        rows.append(
            {
                "sample_id": sample.id,
                "application": sample.application,
                "html_path": str(html_path),
                "artifact_dir": str(get_sample_artifact_dir(sample.id)),
                "reports": serialized_reports,
                "comparisons": comparisons,
            }
        )

    return {
        "jsonl_path": str(jsonl_path),
        "selected_sample_ids": [sample.id for sample in samples],
        "processed_samples": processed_count,
        "missing_html_samples": missing_html,
        "artifact_backends": active_backends,
        "report_count_by_backend": report_count_by_backend,
        "inconsistent_backend_pairs": inconsistent_pairs,
        "rows": rows,
    }