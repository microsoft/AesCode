#!/usr/bin/env bash
# env.sh — source this before running any stage.
#
#   source eval/run/env.sh
#
# Everything below is either a path into the repo or a knob the four stage
# scripts read. Nothing here reaches outside BENCH_ROOT, which is the repo root.
set -u

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_ROOT="$(cd "$_here/../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$BENCH_ROOT:${PYTHONPATH:-}"

# ---- data ----
# No evaluation data ships with the repository; these are the paths the stages
# look in by default. Point them wherever your spec and checklists live. The
# formats are specified in eval/README.md.
export SPEC="${SPEC:-$BENCH_ROOT/data/eval_spec.jsonl}"
export RUBRIC_DIR="${RUBRIC_DIR:-$BENCH_ROOT/data/checklists}"
export IMAGE_DIR="${IMAGE_DIR:-$BENCH_ROOT/data/images}"
export SIDS="${SIDS:-$BENCH_ROOT/runs/_all.txt}"

# ---- run layout ----
export RUN_SUFFIX="${RUN_SUFFIX:-_final300}"
export BLOCK="${BLOCK:-block_final300}"
export N="${N:-3}"

# ---- judge (OpenAI-compatible; OPENAI_BASE_URL may be a comma-separated list) ----
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"

# ---- rendering ----
# The design validator is reference-dependent and contributes no column to the
# reference-free table. Leaving it on costs a VLM call per rollout and changes
# nothing, so it is off for every run.
export DISABLED_VALIDATORS="${DISABLED_VALIDATORS:-DesignValidator}"
export RUNTIME="${RUNTIME:-$BENCH_ROOT/.runtime}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$RUNTIME/ms-playwright}"
export TOKENIZERS_PARALLELISM=false

# The sample-id list is the shared denominator for every stage, so it is derived
# once from the spec rather than maintained by hand: a stage working off a
# different list reports a coverage number that means something different.
if [ ! -f "$SIDS" ] && [ -f "$SPEC" ]; then
  mkdir -p "$(dirname "$SIDS")"
  "$PYTHON_BIN" -c 'import json,sys
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if line:
        print(json.loads(line)["id"])' "$SPEC" > "$SIDS"
  echo "[env] wrote $SIDS from $SPEC"
fi

echo "[env] BENCH_ROOT=$BENCH_ROOT"
echo "[env] PYTHON_BIN=$PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
echo "[env] SPEC=$SPEC ($([ -f "$SPEC" ] && wc -l < "$SPEC" || echo MISSING) samples)"
echo "[env] RUBRIC_DIR=$RUBRIC_DIR ($(ls "$RUBRIC_DIR" 2>/dev/null | wc -l) checklists)"
echo "[env] N=$N  judge=$JUDGE_MODEL"
