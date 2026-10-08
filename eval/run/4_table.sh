#!/usr/bin/env bash
# 4_table.sh [RUN_LABEL,RUN_LABEL,...]
# ---------------------------------------------------------------------------
# Stage 4: aggregate finished runs into the markdown main table.
#
#   source eval/run/env.sh
#   bash eval/run/4_table.sh                  # every run under runs/
#   bash eval/run/4_table.sh labelA,labelB    # just these
#
# Runs that have not finished stage 3 are included and will score low rather
# than being skipped, which is the same convention the table itself uses for an
# unjudged rollout. Check the per-row coverage line printed on stderr before
# believing a row: it reports judged / expected rollouts for that arm.
set -uo pipefail
: "${BENCH_ROOT:?source eval/run/env.sh first}"
: "${PYTHON_BIN:?}" ; : "${SPEC:?}" ; : "${RUBRIC_DIR:?}"
cd "$BENCH_ROOT"

N="${N:-3}"
BLOCK="${BLOCK:-block_final300}"
SUFFIX="${RUN_SUFFIX:-_final300}"
OUT_MD="${OUT_MD:-$BENCH_ROOT/runs/main_table.md}"
RUNS_ARG=""
[ "${1:-}" != "" ] && RUNS_ARG="--runs $1"

"$PYTHON_BIN" eval/build_table.py \
    --runs-dir "$BENCH_ROOT/runs" $RUNS_ARG \
    --rubric-dir "$RUBRIC_DIR" --spec "$SPEC" \
    --n "$N" --suffix "$SUFFIX" --block "$BLOCK" --out "$OUT_MD"
