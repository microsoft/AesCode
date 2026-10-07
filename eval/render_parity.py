#!/usr/bin/env python3
"""render_parity.py — catch renders that this harness lost but the original run had.

The archives ship the original `validation_report.jsonl`, so every rollout has a
known-good `execution_pass` from the run that produced the paper's numbers. If a
re-render here marks a rollout as failed while the original marked it passed,
that is a defect in THIS harness, not a property of the model, and letting it
through would silently subtract points (missing render -> no PNG -> no judgment
-> scored 0 under the locked denominator).

Observed cause: rendering at --workers 48 loses ~6% of rollouts on strong models;
the identical HTML renders 50/50 at --workers 8.

    check        report the mismatch count per run (exit 1 if any)
    repair       drop the mismatched rows from the report and delete their artifact
                 dirs, so a subsequent `eval_vllm_validate.py --resume` at a lower
                 concurrency re-renders exactly those rollouts
    retry-failed for runs generated here, which have no archive to compare
                 against: drop every failed row so the next --resume re-renders
                 it. Exits 3 when there is nothing to retry.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def load(path: Path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            m = re.search(r"_k(\d+)\.html", r.get("html_path", ""))
            if m:
                out[(r["sample_id"], int(m.group(1)))] = r
    return out


def mismatches(run: Path):
    mine = load(run / "results/validation_report.jsonl")
    orig = load(run / "results_orig/validation_report.jsonl")
    bad = []
    for sk, r in mine.items():
        if not r.get("execution_pass") and (orig.get(sk) or {}).get("execution_pass"):
            bad.append(sk)
    return sorted(bad), len(mine), len(orig)


def failed(run: Path):
    mine = load(run / "results/validation_report.jsonl")
    return sorted(sk for sk, r in mine.items() if not r.get("execution_pass")), len(mine)


def drop(run: Path, bad):
    """Remove these rollouts' report rows and artifacts so --resume redoes them."""
    badset = set(bad)
    rep = run / "results/validation_report.jsonl"
    kept = []
    with open(rep) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            m = re.search(r"_k(\d+)\.html", r.get("html_path", ""))
            if m and (r["sample_id"], int(m.group(1))) in badset:
                continue
            kept.append(ln.rstrip("\n"))
    rep.write_text("\n".join(kept) + ("\n" if kept else ""))
    for sid, k in bad:
        shutil.rmtree(run / f"results/artifacts/{sid}_k{k}", ignore_errors=True)
    return len(kept)


def retry_failed(a):
    dropped = 0
    for rd in a.runs:
        run = Path(rd)
        if (run / "results_orig/validation_report.jsonl").exists():
            print(f"{run.name}: has an archived report; use `repair` instead", file=sys.stderr)
            sys.exit(2)
        bad, nmine = failed(run)
        print(f"{run.name}: rows={nmine} failed={len(bad)}")
        if bad:
            kept = drop(run, bad)
            dropped += len(bad)
            print(f"  dropped {len(bad)} failed rows, kept {kept}")
    if not dropped:
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "repair", "retry-failed"])
    ap.add_argument("runs", nargs="+", help="runs/<LABEL>_final300 dirs")
    a = ap.parse_args()

    if a.mode == "retry-failed":
        retry_failed(a)
        return

    total = 0
    for rd in a.runs:
        run = Path(rd)
        bad, nmine, norig = mismatches(run)
        total += len(bad)
        print(f"{run.name}: rows={nmine} orig={norig} harness_lost={len(bad)}")
        if a.mode == "repair" and bad:
            kept = drop(run, bad)
            print(f"  repaired: dropped {len(bad)} rows, kept {kept}")

    if a.mode == "check" and total:
        print(f"\nFAIL: {total} rollouts rendered here but not in the original run.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
