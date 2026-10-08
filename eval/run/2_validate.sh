#!/usr/bin/env bash
# 2_validate.sh <RUN_LABEL>
# ---------------------------------------------------------------------------
# Stage 2: render every rollout and run the rule validators over it.
#
#   source eval/run/env.sh
#   bash eval/run/2_validate.sh my_label
#
# The screenshot produced here is the only thing stage 3 ever sees, so this
# script's job is to get a screenshot for every rollout that can produce one,
# and to say precisely how many could not.
#
# THE GATE COUNTS ROLLOUT DIRECTORIES, NOT PNG FILES. One rollout can emit more
# than one PNG, so counting files under results/artifacts/ overcounts and a run
# can clear a target it has not actually met. What is counted here is: for each
# sample id and each k < N, does results/artifacts/<sid>_k<k>/ hold at least one
# non-overflow screenshot.
#
# RETRY UNTIL STALLED, NOT A FIXED NUMBER OF TIMES. Rendering loses a few
# percent of rollouts at high concurrency that render fine at low concurrency, so
# a failed render is retried; but a rollout whose HTML is genuinely truncated
# will never render, and retrying it forever is a hang. The loop therefore stops
# as soon as the screenshot count stops rising.
#
# A rollout that still has no screenshot is judged as all-wrong, which is correct
# for a model that emitted broken HTML and wrong if the harness dropped it. The
# final count is written to <block>/.shot_count_at_judge so the two can be told
# apart afterwards.
set -uo pipefail
: "${BENCH_ROOT:?source eval/run/env.sh first}"
: "${PYTHON_BIN:?}" ; : "${SPEC:?}" ; : "${IMAGE_DIR:?}"
cd "$BENCH_ROOT"

label="${1:?usage: bash eval/run/2_validate.sh <RUN_LABEL>}"
SIDS="${SIDS:-$BENCH_ROOT/runs/_all.txt}"
N="${N:-3}"
BLOCK="${BLOCK:-block_final300}"
VAL_WORKERS="${VAL_WORKERS:-8}"
export DISABLED_VALIDATORS="${DISABLED_VALIDATORS:-DesignValidator}"

OUT="$BENCH_ROOT/runs/${label}${RUN_SUFFIX:-_final300}"
[ -d "$OUT" ] || { echo "no such run: $OUT"; exit 2; }
[ -f "$SIDS" ] || { echo "no sample id list: $SIDS"; exit 2; }

n_sids=$(grep -c . "$SIDS")
want=$((n_sids * N))
ulimit -n 65535 2>/dev/null || true

count_shots () {
  "$PYTHON_BIN" - "$OUT" "$SIDS" "$N" <<'PY'
import sys
from pathlib import Path
out, sidsf, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
art = out / "results/artifacts"
hit = 0
for s in (x.strip() for x in sidsf.read_text().split()):
    if not s:
        continue
    for k in range(n):
        d = art / f"{s}_k{k}"
        if d.is_dir() and any(
            "overflow" not in p.name for p in d.rglob("render_playwright_*.png")
        ):
            hit += 1
print(hit)
PY
}

ns=$(count_shots)
if [ "$ns" -ge "$want" ]; then
  echo "[render] $label already complete ($ns / $want)"
else
  echo "=== [$(date '+%H:%M:%S')] render $label ($ns / $want) ==="
  prev=-1
  for attempt in 1 2 3; do
    "$PYTHON_BIN" eval/eval_vllm_validate.py \
        --spec-jsonl "$SPEC" --html-dir "$OUT/html" --output-dir "$OUT/results" \
        --ref-image-dir "$IMAGE_DIR" --pattern sample --n "$N" \
        --workers "$VAL_WORKERS" --resume 2>&1 | tail -2
    ns=$(count_shots)
    echo "[render] $label $ns / $want"
    [ "$ns" -ge "$want" ] && break
    [ "$ns" -le "$prev" ] && {
      echo "[render] $label stalled at $ns; $((want - ns)) rollouts never render"
      break
    }
    prev=$ns
    # Drop the failed rows so the next --resume re-renders exactly those.
    "$PYTHON_BIN" eval/render_parity.py retry-failed "$OUT" || true
  done
fi

mkdir -p "$OUT/$BLOCK"
echo "$ns / $want" > "$OUT/$BLOCK/.shot_count_at_judge"
echo "DONE validate $label ($ns / $want) -> next: bash eval/run/3_judge.sh $label"
