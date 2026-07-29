#!/usr/bin/env python3
"""DesignCoder — evaluation VALIDATION stage (release).

Takes the HTML produced by run_inference.py and computes ALL evaluation
dimensions, then prints a clean pa@1 / pa@k score table. No separate
"build table" step — the table is printed at the end of this script.

Dimensions
  Deterministic (6, from the repo's validation engine, no LLM):
    Execution, Text, Boundary, TableChart, SemanticLayout, Whitespace
  Design (1, rubric judge):
    render each rollout → a vision LLM answers the sid's yes/no rubric.
    Uses a standard OpenAI-compatible endpoint (--judge-base-url /
    --judge-api-key / --judge-model). If no judge creds are given, Design
    is skipped and only the 6 deterministic dims are reported.

Aggregation protocol (identical to the paper table)
  Denominator LOCKED to (accountable sids) x N. A rollout the model never
  produced, or that failed to render, scores 0 — it is NOT dropped.
  pa@1 = mean over N rollouts of the per-rollout score, then mean over sids.
  pa@k = max over the N rollouts, then mean over sids.   (k == N == --n)
  TableChart stays applicable-only (denominator = sids whose GT has a table).
  Rule    = per-sample mean(Text, Boundary [, TableChart])
  Visual  = per-sample mean(SemanticLayout, Whitespace [, Design])
  Overall = per-sample mean(Text, Boundary, SemLayout, Whitespace
                            [, TableChart][, Design])   -- Execution excluded

Usage
  # 6 deterministic dims only:
  python run_validation.py --run ./runs/my_model --n 8 \
      --spec <data_root>/eval.jsonl --image-dir <data_root>/images

  # all dims incl. Design (rubric judge on your own OpenAI-compatible endpoint):
  python run_validation.py --run ./runs/my_model --n 8 \
      --spec <data_root>/eval.jsonl --image-dir <data_root>/images \
      --rubric-dir <data_root>/rubrics \
      --judge-base-url https://api.openai.com/v1 \
      --judge-api-key  $OPENAI_API_KEY \
      --judge-model    gpt-4o

Resumable: reruns skip already-validated rollouts and already-judged renders.
"""
import argparse
import base64
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent          # <REPO>/eval
REPO = EVAL_DIR.parent                               # <REPO>
VALIDATE_ENGINE = EVAL_DIR / "eval_vllm_validate.py"

# 6 deterministic validators as they appear in the validation report.
DET_VALIDATORS = [
    "ExecutionValidator", "BoundaryValidator", "TextValidator",
    "TableChartValidator", "SemanticLayoutValidator", "WhitespaceValidator",
]
FULL_COVERAGE_FRACTION = 0.967  # >= this fraction of sids => scored over the full set

# Design rubric judge system prompt (inlined; matches paper judge).
JUDGE_SYSTEM = """You are a strict but fair visual judge.

You will see a SINGLE GENERATED IMAGE and a numbered list of yes/no questions about it.
For EACH question, output a JSON line {"id": "...", "answer": "yes"|"no"|"unclear"}.

Rules:
- Answer ONLY based on what is actually visible in the generated image.
- "unclear" only if the relevant element is genuinely ambiguous or not visible at all (use sparingly).
- Do not assume. If the question asks about a "blue-to-pink gradient" and you see a flat blue, answer "no".
- Do not be lenient. The point is to find layout differences from the reference.

Output STRICT JSON (no fences):
{"judgments": [{"id":"q1","answer":"yes"}, {"id":"q2","answer":"no"}, ...]}
"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [validate] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1: deterministic 6-dim validation (repo engine, DesignValidator off)
# ---------------------------------------------------------------------------
def run_deterministic(run_dir, spec, image_dir, n, workers, retry):
    html_dir = run_dir / "html"
    res_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    res_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    report = res_dir / "validation_report.jsonl"

    n_html = len(list(html_dir.glob("*.html")))
    env = dict(os.environ, DISABLED_VALIDATORS="DesignValidator",
               TOKENIZERS_PARALLELISM="false")
    for attempt in range(1, retry + 1):
        log(f"deterministic validate attempt {attempt}/{retry} "
            f"(6 dims, DesignValidator disabled)")
        subprocess.run(
            [sys.executable, str(VALIDATE_ENGINE),
             "--spec-jsonl", str(spec), "--html-dir", str(html_dir),
             "--output-dir", str(res_dir), "--ref-image-dir", str(image_dir),
             "--pattern", "sample", "--n", str(n),
             "--workers", str(workers), "--resume"],
            cwd=str(REPO), env=env, check=False)
        n_val = sum(1 for _ in open(report)) if report.exists() else 0
        log(f"attempt {attempt}: rows={n_val}/html={n_html}")
        if n_val >= n_html:
            break
        time.sleep(5)
    return report


# ---------------------------------------------------------------------------
# Stage 2: Design rubric judge (OpenAI-compatible). Optional.
# ---------------------------------------------------------------------------
def parse_loose(t):
    s = t.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        if s.startswith("json"):
            s = s.split("\n", 1)[1] if "\n" in s else s
    return json.loads(s)


def find_png(art_dir, sid, k):
    import glob
    paths = glob.glob(str(art_dir / f"{sid}_k{k}/{sid}/render_playwright_*.png"))
    paths = [p for p in paths if "overflow" not in Path(p).name]
    return Path(paths[0]) if paths else None


def judge_one(client, model, image_path, rubric, timeout=180):
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    qs = "\n".join(f"  {q['id']}. {q['question']}" for q in rubric)
    user_text = (
        f"Judge the following GENERATED image against {len(rubric)} yes/no questions.\n\n"
        f"Questions:\n{qs}\n\n"
        "For each question, answer yes / no / unclear based ONLY on what is visible "
        'in the image.\nOutput strict JSON: {"judgments": [{"id":"q1","answer":"yes"}, ...]}')
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
        max_completion_tokens=2000,
        timeout=timeout,
    )
    return r.choices[0].message.content or ""


def judge_with_retry(client, model, art_dir, sid, k, rubric):
    img = find_png(art_dir, sid, k)
    if not img:
        return {"sid": sid, "k": k, "ok": False, "err": "no_png"}
    delay = 8
    last = ""
    for attempt in range(5):
        try:
            txt = judge_one(client, model, img, rubric)
            try:
                d = parse_loose(txt)
                return {"sid": sid, "k": k, "ok": True,
                        "answers": {it["id"]: it["answer"] for it in d.get("judgments", [])}}
            except Exception as e:
                return {"sid": sid, "k": k, "ok": False, "err": f"parse: {e}"}
        except Exception as e:
            msg = str(e)
            last = f"{type(e).__name__}: {msg[:160]}"
            if any(c in msg for c in ("400 ", "401 ", "403 ", "404 ", "BadRequest")):
                return {"sid": sid, "k": k, "ok": False, "err": last}
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return {"sid": sid, "k": k, "ok": False, "err": f"retries: {last}"}


def run_design_judge(run_dir, spec, rubric_dir, n, base_url, api_key, model, conc):
    """Judge every rendered rollout against its rubric. Returns judgments path
    (always the on-disk path, so aggregation reads it even if judging is skipped)."""
    out_jsonl = run_dir / "results" / "design_judgments.jsonl"
    try:
        from openai import OpenAI
    except ImportError:
        log("openai package not installed -> skip NEW judging (existing judgments still scored)")
        return out_jsonl if out_jsonl.exists() else None
    art_dir = run_dir / "results" / "artifacts"
    if not art_dir.exists():
        log(f"no render artifacts at {art_dir} -> skip NEW judging")
        return out_jsonl if out_jsonl.exists() else None

    sids = sorted(set(json.loads(l)["id"] for l in open(spec)))
    rubrics = {sid: json.loads((rubric_dir / f"{sid}.json").read_text())["rubric"]
               for sid in sids if (rubric_dir / f"{sid}.json").exists()}
    log(f"Design: {len(rubrics)}/{len(sids)} sids have a rubric")

    done = set()
    if out_jsonl.exists():
        for ln in out_jsonl.read_text().splitlines():
            if not ln:
                continue
            try:
                r = json.loads(ln)
                if r.get("ok"):
                    done.add((r["sid"], r["k"]))
            except Exception:
                pass

    tasks = [(sid, k) for sid in sids if sid in rubrics for k in range(n)
             if (sid, k) not in done and find_png(art_dir, sid, k)]
    log(f"Design: {len(done)} judged, {len(tasks)} pending")
    if tasks:
        client = OpenAI(base_url=base_url, api_key=api_key)
        t0, ok, fail = time.time(), len(done), 0
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(judge_with_retry, client, model, art_dir, sid, k, rubrics[sid]):
                    (sid, k) for sid, k in tasks}
            with open(out_jsonl, "a") as f:
                for fut in as_completed(futs):
                    r = fut.result()
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    if r.get("ok"):
                        ok += 1
                    else:
                        fail += 1
                    d = ok + fail - len(done)
                    if d % 100 == 0:
                        el = time.time() - t0
                        rate = d / el * 60 if el > 0 else 0
                        log(f"  Design {d}/{len(tasks)} ok={ok} fail={fail} {rate:.0f}/min")
        log(f"Design judge done: ok={ok} fail={fail} wall={time.time()-t0:.0f}s")
    return out_jsonl


# ---------------------------------------------------------------------------
# Stage 3: aggregation + table
# ---------------------------------------------------------------------------
def load_rows(report, final_sids):
    by_sk = {}
    if not report.exists():
        return []
    for line in open(report):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = r.get("sample_id")
        if sid not in final_sids:
            continue
        m = re.search(r"_k(\d+)\.html", r.get("html_path", ""))
        if not m:
            continue
        by_sk[(sid, int(m.group(1)))] = r
    return list(by_sk.values())


def accountable_sids(rows, final_sids):
    cov = set(r["sample_id"] for r in rows
              if re.search(r"_k\d+\.html", r.get("html_path", ""))) & final_sids
    threshold = FULL_COVERAGE_FRACTION * len(final_sids)
    return set(final_sids) if len(cov) >= threshold else cov


def index_rows(rows):
    by_sid = defaultdict(dict)
    for r in rows:
        m = re.search(r"_k(\d+)\.html", r["html_path"])
        if m:
            by_sid[r["sample_id"]][int(m.group(1))] = r
    return by_sid


def load_design_scores(judgments_path, rubric_dir, final_sids):
    """Return {(sid,k): design_score in [0,1]} and set of sids that have a rubric."""
    if not judgments_path or not judgments_path.exists():
        return {}, set()
    j = {}
    for ln in judgments_path.read_text().splitlines():
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            j[(r["sid"], r["k"])] = r["answers"]
    rub_cache = {}
    def rub(sid):
        if sid not in rub_cache:
            p = rubric_dir / f"{sid}.json"
            rub_cache[sid] = json.loads(p.read_text())["rubric"] if p.exists() else None
        return rub_cache[sid]
    scores = {}
    for (sid, k), ans in j.items():
        rb = rub(sid)
        if not rb:
            continue
        pa_map = {q["id"]: q.get("pass_answer", "yes") for q in rb}
        hit = sum(1 for q in rb if ans.get(q["id"]) == pa_map[q["id"]])
        scores[(sid, k)] = hit / len(rb)
    rubric_sids = {s for s in final_sids if (rubric_dir / f"{s}.json").exists()}
    return scores, rubric_sids


def main():
    ap = argparse.ArgumentParser(description="DesignCoder eval validation + scoring")
    ap.add_argument("--run", required=True, help="run dir produced by run_inference.py")
    ap.add_argument("--n", type=int, default=8, help="rollouts per spec (k for pa@k)")
    ap.add_argument("--spec", required=True, help="eval spec JSONL")
    ap.add_argument("--image-dir", required=True, help="reference image dir")
    ap.add_argument("--rubric-dir", default="", help="rubric dir (needed for Design)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--det-retry", type=int, default=3)
    ap.add_argument("--name", default="", help="model name shown in the table")
    # Design judge (OpenAI-compatible). Omit to skip Design.
    ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", ""))
    ap.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    ap.add_argument("--judge-conc", type=int, default=16)
    ap.add_argument("--skip-deterministic", action="store_true",
                    help="reuse existing validation_report.jsonl, don't re-run engine")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    spec = Path(args.spec)
    rubric_dir = Path(args.rubric_dir) if args.rubric_dir else None
    name = args.name or run_dir.name

    if not VALIDATE_ENGINE.exists():
        sys.exit(f"ERROR: validation engine not found: {VALIDATE_ENGINE}")
    if not (run_dir / "html").exists():
        sys.exit(f"ERROR: {run_dir}/html not found — run run_inference.py first")

    # Stage 1
    report = run_dir / "results" / "validation_report.jsonl"
    if not args.skip_deterministic:
        report = run_deterministic(run_dir, spec, args.image_dir,
                                   args.n, args.workers, args.det_retry)

    # Stage 2 (optional Design)
    judgments = None
    do_design = bool(args.judge_base_url and args.judge_api_key
                     and args.judge_model and rubric_dir)
    if do_design:
        judgments = run_design_judge(
            run_dir, spec, rubric_dir, args.n,
            args.judge_base_url, args.judge_api_key, args.judge_model, args.judge_conc)
    else:
        log("no judge creds / rubric-dir -> Design dimension skipped (6 deterministic dims only)")

    # Stage 3
    final_sids = set(json.loads(l)["id"] for l in open(spec))
    rows = load_rows(report, final_sids)
    if not rows:
        sys.exit(f"ERROR: no validation rows in {report}")
    expected = accountable_sids(rows, final_sids)
    by_sid = index_rows(rows)
    design_scores, rubric_sids = load_design_scores(judgments, rubric_dir, final_sids) \
        if rubric_dir else ({}, set())

    table = compute_table(by_sid, expected, args.n, design_scores,
                          rubric_sids, has_design=do_design)
    print_table(name, table, args.n, len(expected), rows, do_design)


# --- scoring primitives (locked denominator = expected x n) ---------------
def _det_score(r, v):
    if r is None:
        return 0.0
    s = (r.get("per_validator") or {}).get(v)
    return s if s is not None else 0.0


def _col_pa(by_sid, expected, n, v):
    """Single deterministic validator: (pa1, pak). TableChart applicable-only."""
    pa1s, paks = [], []
    for sid in expected:
        kd = by_sid.get(sid, {})
        if v == "TableChartValidator":
            applicable = any((r.get("per_validator") or {}).get("TableChartValidator") is not None
                             for r in kd.values())
            if not applicable:
                continue
        scores = [_det_score(kd.get(k), v) for k in range(n)]
        pa1s.append(statistics.mean(scores))
        paks.append(max(scores))
    if not pa1s:
        return None, None
    return statistics.mean(pa1s), statistics.mean(paks)


def _group_pa(by_sid, expected, n, base, design_scores, rubric_sids,
              include_tc=False, include_design=False):
    """Per-sample mean over applicable validators. (pa1, pak)."""
    pa1s, paks = [], []
    for sid in expected:
        kd = by_sid.get(sid, {})
        should_tc = include_tc and any(
            (r.get("per_validator") or {}).get("TableChartValidator") is not None
            for r in kd.values())
        has_design = include_design and (sid in rubric_sids)
        per_rollout = []
        for k in range(n):
            r = kd.get(k)
            pv = (r.get("per_validator") or {}) if r is not None else {}
            vals = [pv.get(v) if pv.get(v) is not None else 0.0 for v in base]
            if should_tc:
                tc = pv.get("TableChartValidator")
                vals.append(tc if tc is not None else 0.0)
            if has_design:
                vals.append(design_scores.get((sid, k), 0.0))
            if vals:
                per_rollout.append(statistics.mean(vals))
        if per_rollout:
            pa1s.append(statistics.mean(per_rollout))
            paks.append(max(per_rollout))
    if not pa1s:
        return None, None
    return statistics.mean(pa1s), statistics.mean(paks)


def _design_pa(expected, n, design_scores, rubric_sids):
    sids = [s for s in expected if s in rubric_sids]
    if not sids:
        return None, None
    pa1s, paks = [], []
    for sid in sids:
        scores = [design_scores.get((sid, k), 0.0) for k in range(n)]
        pa1s.append(statistics.mean(scores))
        paks.append(max(scores))
    return statistics.mean(pa1s), statistics.mean(paks)


def compute_table(by_sid, expected, n, design_scores, rubric_sids, has_design):
    t = {}
    for v in DET_VALIDATORS:
        t[v] = _col_pa(by_sid, expected, n, v)
    t["Design"] = _design_pa(expected, n, design_scores, rubric_sids) if has_design else (None, None)
    t["Rule"] = _group_pa(by_sid, expected, n, ["TextValidator", "BoundaryValidator"],
                          design_scores, rubric_sids, include_tc=True, include_design=False)
    t["Visual"] = _group_pa(by_sid, expected, n,
                            ["SemanticLayoutValidator", "WhitespaceValidator"],
                            design_scores, rubric_sids, include_tc=False,
                            include_design=has_design)
    t["Overall"] = _group_pa(by_sid, expected, n,
                             ["TextValidator", "BoundaryValidator",
                              "SemanticLayoutValidator", "WhitespaceValidator"],
                             design_scores, rubric_sids, include_tc=True,
                             include_design=has_design)
    return t


def _f(x):
    return f"{x:.4f}" if x is not None else "  --  "


def print_table(name, t, n, n_expected, rows, has_design):
    n_rows = len(rows)
    target = n_expected * n
    print("\n" + "=" * 78)
    print(f"DesignCoder eval — {name}")
    print(f"scored sids={n_expected}  rollouts n={n}  "
          f"validated rows={n_rows}/{target}"
          + ("" if has_design else "  (Design skipped: no judge)"))
    print("=" * 78)
    cols = ["Exec*", "Text", "Bound", "TblCh", "Rule",
            "SemL", "WSpc", "Design", "Visual", "Overall"]
    keys = ["ExecutionValidator", "TextValidator", "BoundaryValidator",
            "TableChartValidator", "Rule", "SemanticLayoutValidator",
            "WhitespaceValidator", "Design", "Visual", "Overall"]
    print("Exec* is reported but EXCLUDED from Rule/Visual/Overall (saturated).\n")
    for metric, idx in (("pa@1 (mean-of-N)", 0), (f"pa@{n} (max-of-N)", 1)):
        print(f"--- {metric} ---")
        print("| " + " | ".join(f"{c:>7}" for c in cols) + " |")
        print("|" + "|".join("-" * 9 for _ in cols) + "|")
        print("| " + " | ".join(f"{_f(t[k][idx]):>7}" for k in keys) + " |")
        print()


if __name__ == "__main__":
    main()
