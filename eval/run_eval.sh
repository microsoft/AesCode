#!/usr/bin/env bash
# DesignCoder — one-shot INFERENCE + VALIDATION driver (release).
#
# Runs the two stages back-to-back on one node:
#   1. run_inference.py   ckpt -> html (n rollouts x specs)
#   2. run_validation.py  html -> 6 deterministic dims [+ Design] -> score table
#
# Both stages resume, so re-running after an interruption is safe.
#
# Usage:
#   bash run_eval.sh <CKPT> <OUT_DIR> <SPEC> <IMAGE_DIR> [N] [TP] [NGPU]
#
#   CKPT       HF safetensors dir (config.json + model-*.safetensors)
#   OUT_DIR    where html/ + results/ land
#   SPEC       eval spec JSONL
#   IMAGE_DIR  reference image dir
#   N          rollouts per spec (pa@N). default 8
#   TP         tensor-parallel size per vLLM endpoint. default 8
#   NGPU       total GPUs. default = TP
#
# Design dimension (optional): export judge creds for an OpenAI-compatible
# vision endpoint AND set RUBRIC_DIR before calling; otherwise only the 6
# deterministic dims run.
#   export JUDGE_BASE_URL=https://api.openai.com/v1
#   export JUDGE_API_KEY=sk-...
#   export JUDGE_MODEL=gpt-4o
#   export RUBRIC_DIR=/path/to/rubrics
#
# Example (8xGPU, n=8, with Design):
#   export JUDGE_BASE_URL=... JUDGE_API_KEY=... JUDGE_MODEL=gpt-4o RUBRIC_DIR=...
#   bash run_eval.sh /models/my-ckpt ./runs/my_model specs.jsonl images/ 8 8 8

set -euo pipefail

CKPT="${1:?usage: run_eval.sh <CKPT> <OUT_DIR> <SPEC> <IMAGE_DIR> [N] [TP] [NGPU]}"
OUT="${2:?need OUT_DIR}"
SPEC="${3:?need SPEC jsonl}"
IMAGE_DIR="${4:?need IMAGE_DIR}"
N="${5:-8}"
TP="${6:-8}"
NGPU="${7:-$TP}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"

echo "=== [1/2] inference: ckpt=$CKPT out=$OUT n=$N tp=$TP ngpu=$NGPU ==="
"$PY" "$HERE/run_inference.py" \
    --ckpt "$CKPT" --out "$OUT" --spec "$SPEC" --image-dir "$IMAGE_DIR" \
    --n "$N" --tp "$TP" --ngpu "$NGPU"

echo "=== [2/2] validation + scoring ==="
VAL_ARGS=(--run "$OUT" --n "$N" --spec "$SPEC" --image-dir "$IMAGE_DIR")
if [ -n "${JUDGE_BASE_URL:-}" ] && [ -n "${JUDGE_API_KEY:-}" ] && [ -n "${JUDGE_MODEL:-}" ] && [ -n "${RUBRIC_DIR:-}" ]; then
  echo "    Design judge enabled (model=$JUDGE_MODEL)"
  VAL_ARGS+=(--rubric-dir "$RUBRIC_DIR"
             --judge-base-url "$JUDGE_BASE_URL"
             --judge-api-key  "$JUDGE_API_KEY"
             --judge-model    "$JUDGE_MODEL")
else
  echo "    Design judge disabled (set JUDGE_BASE_URL/JUDGE_API_KEY/JUDGE_MODEL/RUBRIC_DIR to enable)"
fi
"$PY" "$HERE/run_validation.py" "${VAL_ARGS[@]}"

echo "=== done. score table printed above; artifacts under $OUT ==="
