#!/usr/bin/env bash
# 3_judge.sh <RUN_LABEL>
# ---------------------------------------------------------------------------
# Stage 3: run the checklist judge over the screenshots from stage 2.
#
#   export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.openai.com/v1
#   source eval/run/env.sh
#   bash eval/run/3_judge.sh my_label
#
# One VLM call per category per rollout, over the rendered PNG and nothing else.
#
# ONE ARM AT A TIME, AND NO TABLE. Tabulating is a separate step on purpose, so a
# half-finished batch cannot quietly produce a table.
#
# --n is passed explicitly. Every stage must use the same N; a judge running at a
# different N from the generation silently produces a table nobody can reproduce.
#
# The .done marker is written only after the judge exits cleanly, so a rerun
# resumes rather than re-judging, and a crashed run is never mistaken for a
# finished one. The judge itself also resumes from judgments.jsonl.
set -uo pipefail
: "${BENCH_ROOT:?source eval/run/env.sh first}"
: "${PYTHON_BIN:?}" ; : "${RUBRIC_DIR:?}"
cd "$BENCH_ROOT"

label="${1:?usage: bash eval/run/3_judge.sh <RUN_LABEL>}"
SIDS="${SIDS:-$BENCH_ROOT/runs/_all.txt}"
N="${N:-3}"
BLOCK="${BLOCK:-block_final300}"
JUDGE_CONC="${JUDGE_CONC:-64}"

OUT="$BENCH_ROOT/runs/${label}${RUN_SUFFIX:-_final300}"
[ -d "$OUT" ] || { echo "no such run: $OUT"; exit 2; }

if [ -f "$OUT/$BLOCK/.done" ]; then
  echo "[judge] $label already done"
  exit 0
fi
mkdir -p "$OUT/$BLOCK"

shots="$(cat "$OUT/$BLOCK/.shot_count_at_judge" 2>/dev/null || echo 'unknown')"
echo "=== [$(date '+%H:%M:%S')] judge $label (screenshots $shots) ==="
"$PYTHON_BIN" eval/judge_checklist.py \
    --label "$label" --rubric-dir "$RUBRIC_DIR" --sids-file "$SIDS" \
    --art-dir "$OUT/results/artifacts" --out "$OUT/$BLOCK" \
    --conc "$JUDGE_CONC" --n "$N" \
    --judge-model "${JUDGE_MODEL:-}" || exit 1

touch "$OUT/$BLOCK/.done"
echo "DONE judge $label -> next: bash eval/run/4_table.sh"
