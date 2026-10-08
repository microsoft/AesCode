#!/usr/bin/env python3
"""Stage 4: aggregate one or more finished runs into the benchmark's main table.

Nine columns, from two sources that are never mixed:

    Text  Bound  Chart   rule validators, read off results/validation_report.jsonl
    Content Layout Style  checklist judge, read off <block>/judgments.jsonl
    Rule    = mean(Text, Bound, Chart)
    Visual  = mean(Content, Layout, Style)
    Overall = mean(Rule, Visual)

Execution is deliberately not a column. It is a gate, not a dimension: a page that
does not render scores zero on everything downstream, so reporting it alongside the
others would count the same failure twice.

AGGREGATION, in the order it matters:

* Fixed denominator. Every sample in the spec with a checklist counts, for all N
  rollouts, whether or not that rollout produced a judgment. A rollout that failed
  to render, or whose judgment call failed, scores 0 and is NOT dropped. Dropping
  it would reward a model whose worst pages are the ones that crash the renderer.

* Per dimension: mean over samples of (mean over that sample's N rollouts). Samples
  are weighted equally regardless of how many questions their checklist holds.

* Applicability, not zero-fill, for the rule dimensions. A rule dimension counts
  for a sample only if the validator reported on that sample at all. The chart
  validator covers only the pages whose spec asks for a chart or a table; the pages
  it has nothing to say about stay out of the Chart denominator rather than scoring
  0 there. Within a sample that IS in the denominator, a rollout the validator did
  not reach still scores 0.

* Applicability for the visual buckets too: a bucket counts for a sample only if
  that sample's checklist has questions in it.

* A question is passed iff the judge's answer equals the checklist's pass_answer.
  `unclear` is therefore a failure, which is the intended reading: a property the
  judge could not verify from the image has not been demonstrated.

    python3 eval/build_table.py --runs-dir runs \\
            --rubric-dir data/checklists \\
            --spec data/eval_spec.jsonl --n 3
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rubric_io  # noqa: E402
from rubric_io import load_rubrics  # noqa: E402

# Column name -> validator name in validation_report.jsonl `per_validator`.
RULE_DIMS = {
    "Text": "TextValidator",
    "Bound": "BoundaryValidator",
    "Chart": "TableChartValidator",
}
# The visual columns are the checklist buckets, taken from the one place the
# taxonomy lives rather than restated here.
VISUAL_DIMS = rubric_io.BUCKETS
COLUMNS = ([*RULE_DIMS, "Rule", *VISUAL_DIMS, "Visual", "Overall"])

_K_IN_HTML_PATH = re.compile(r"_k(\d+)\.html")


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_spec_sids(spec: Path):
    return [json.loads(line)["id"] for line in spec.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_rule_scores(run: Path, n: int):
    """{(sid, k): {dim: score}} -- only dimensions the validator actually reported."""
    out = {}
    for row in read_jsonl(run / "results" / "validation_report.jsonl"):
        sid = row.get("sample_id")
        match = _K_IN_HTML_PATH.search(row.get("html_path") or "")
        if not sid or not match:
            continue
        k = int(match.group(1))
        if k not in range(n):
            continue
        per_validator = row.get("per_validator") or {}
        out[(sid, k)] = {dim: per_validator[name]
                         for dim, name in RULE_DIMS.items()
                         if per_validator.get(name) is not None}
    return out


def load_visual_scores(judgments: Path, rubrics, n: int):
    """{(sid, k): {bucket: pass-rate}} plus the count of usable judgments."""
    out, usable = {}, 0
    for row in read_jsonl(judgments):
        sid, k = row.get("sid"), row.get("k")
        if k not in range(n) or sid not in rubrics or not row.get("ok"):
            continue
        usable += 1
        answers = row.get("answers") or {}
        scores = {}
        for bucket in VISUAL_DIMS:
            questions = [q for q in rubrics[sid] if q.get("bucket") == bucket]
            if questions:
                passed = sum(answers.get(q["id"]) == q["pass_answer"]
                             for q in questions)
                scores[bucket] = passed / len(questions)
        out[(sid, k)] = scores
    return out, usable


def score_run(run: Path, rubrics, n: int, block: str):
    rule = load_rule_scores(run, n)
    visual, usable = load_visual_scores(run / block / "judgments.jsonl", rubrics, n)

    # A rule dimension applies to a sample if the validator reported on it for
    # any rollout of that sample.
    applies = {dim: set() for dim in RULE_DIMS}
    for (sid, _k), dims in rule.items():
        for dim in dims:
            applies[dim].add(sid)

    per_dim_per_sid = defaultdict(lambda: defaultdict(list))
    for sid in rubrics:
        buckets_here = {q["bucket"] for q in rubrics[sid]}
        for k in range(n):
            reported = rule.get((sid, k)) or {}
            for dim in RULE_DIMS:
                if sid in applies[dim]:
                    per_dim_per_sid[dim][sid].append(reported.get(dim, 0.0))
            judged = visual.get((sid, k)) or {}
            for bucket in VISUAL_DIMS:
                if bucket in buckets_here:
                    per_dim_per_sid[bucket][sid].append(judged.get(bucket, 0.0))

    dims = {}
    for dim, per_sid in per_dim_per_sid.items():
        if per_sid:
            dims[dim] = statistics.mean(statistics.mean(v) for v in per_sid.values())

    def group(names):
        present = [dims[name] for name in names if name in dims]
        return statistics.mean(present) if present else None

    dims["Rule"] = group(RULE_DIMS)
    dims["Visual"] = group(VISUAL_DIMS)
    both = [v for v in (dims["Rule"], dims["Visual"]) if v is not None]
    dims["Overall"] = statistics.mean(both) if both else None

    coverage = {
        "rollouts_expected": len(rubrics) * n,
        "rollouts_judged": usable,
        "rollouts_validated": len(rule),
        "samples_per_rule_dim": {dim: len(applies[dim]) for dim in RULE_DIMS},
    }
    return dims, coverage


def discover_runs(runs_dir: Path, suffix: str):
    if not runs_dir.is_dir():
        raise SystemExit(f"no runs directory at {runs_dir}; run stages 1-3 first")
    return sorted(p for p in runs_dir.iterdir()
                  if p.is_dir() and p.name.endswith(suffix))


def label_of(run: Path, suffix: str):
    name = run.name
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


def render_markdown(rows):
    header = "| Model | " + " | ".join(COLUMNS) + " |"
    rule = "|:--|" + "--:|" * len(COLUMNS)
    lines = [header, rule]
    for label, dims in rows:
        cells = []
        for column in COLUMNS:
            value = dims.get(column)
            cells.append("--" if value is None else f"{100 * value:.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate finished runs into the main benchmark table.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="directory holding the run directories")
    parser.add_argument("--runs", default="",
                        help="comma-separated run directory names or paths. "
                             "Default: every directory under --runs-dir whose "
                             "name ends in --suffix.")
    parser.add_argument("--rubric-dir", type=Path, required=True,
                        help="directory of per-sample checklist JSON files")
    parser.add_argument("--spec", type=Path, required=True,
                        help="eval spec jsonl; defines the sample universe")
    parser.add_argument("--n", type=int, default=3,
                        help="rollouts per sample (the table's denominator)")
    parser.add_argument("--suffix", default="_final300",
                        help="run directory suffix, stripped to form the row label")
    parser.add_argument("--block", default="block_final300",
                        help="subdirectory of the run holding judgments.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the markdown table here as well as to stdout")
    args = parser.parse_args()

    spec_sids = load_spec_sids(args.spec)
    rubrics = load_rubrics(
        args.rubric_dir, spec_sids,
        warn=lambda m: print(f"warn: {m}", file=sys.stderr))
    if not rubrics:
        raise SystemExit(f"no checklist in {args.rubric_dir} matches any sample "
                         f"in {args.spec}; there is nothing to score.")
    missing = len(spec_sids) - len(rubrics)
    if missing:
        # Silently shrinking the denominator is how two tables stop being
        # comparable, so say it out loud.
        print(f"WARNING: {missing} of {len(spec_sids)} spec samples have no "
              f"checklist in {args.rubric_dir} and are excluded from every row.",
              file=sys.stderr)

    if args.runs:
        runs = [Path(r) if "/" in r else args.runs_dir / r
                for r in args.runs.split(",") if r.strip()]
    else:
        runs = discover_runs(args.runs_dir, args.suffix)
    if not runs:
        raise SystemExit(f"no run directories found under {args.runs_dir}")

    print(f"samples={len(rubrics)}  n={args.n}  "
          f"rollouts/row={len(rubrics) * args.n}", file=sys.stderr)

    rows = []
    for run in runs:
        if not run.is_dir():
            print(f"skip {run}: not a directory", file=sys.stderr)
            continue
        dims, coverage = score_run(run, rubrics, args.n, args.block)
        label = label_of(run, args.suffix)
        rows.append((label, dims))
        print(f"  {label:<28} judged={coverage['rollouts_judged']}/"
              f"{coverage['rollouts_expected']} "
              f"validated={coverage['rollouts_validated']} "
              f"rule_dim_samples={coverage['samples_per_rule_dim']}",
              file=sys.stderr)

    table = render_markdown(rows)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
