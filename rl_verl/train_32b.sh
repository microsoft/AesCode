#!/usr/bin/env bash
#
# AesCode-32B — GDPO reinforcement learning stage.
#
# Identical recipe to train_8b.sh; the two scales differ only in capacity and
# in the memory settings needed to keep vLLM resident alongside the actor.
# Every hyper-parameter not set here is inherited from train_gdpo.sh.
#
# Required environment (see .env.example):
#   AESCODE_ROOT          writable root for checkpoints, caches and logs
#   AESCODE_HOME          this repository
#   OPENAI_BASE_URL       OpenAI-compatible endpoint backing the rubric judge
#   OPENAI_API_KEY        credential for that endpoint
#   VLM_MODEL             judge model name
#
# Usage:
#   bash rl_verl/train_32b.sh
#
set -euo pipefail

: "${AESCODE_ROOT:?set AESCODE_ROOT to a writable working root}"
: "${AESCODE_HOME:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AESCODE_HOME

cd "$AESCODE_HOME"

# ── Model & data ──
export MODEL="${MODEL:-${AESCODE_ROOT}/checkpoints/aescode_32b_sft/phase1-generate/final}"
export MODEL_DIR="$MODEL"
export TRAIN_DATA="${TRAIN_DATA:-${AESCODE_ROOT}/data/rl/train.parquet}"
export VAL_DATA="${VAL_DATA:-${AESCODE_ROOT}/data/rl/val.parquet}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${AESCODE_ROOT}/checkpoints}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aescode_32b_gdpo_$(date +%Y%m%d_%H%M%S)}"

# ── Reward configuration ──
export DESIGN_SCORER=rubric
export DISABLED_VALIDATORS=""

# ── Memory ──
# At 32B the defaults (0.85 / micro-batch 8) make the vLLM wake-up step OOM.
# These are the values the released run used.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
export ROLLOUT_LOG_PROB_MICRO_BATCH="${ROLLOUT_LOG_PROB_MICRO_BATCH:-2}"
export REF_LOG_PROB_MICRO_BATCH="${REF_LOG_PROB_MICRO_BATCH:-2}"

echo "[aescode-32b] MODEL=$MODEL"
echo "[aescode-32b] TRAIN=$TRAIN_DATA"
echo "[aescode-32b] VAL=$VAL_DATA"
echo "[aescode-32b] DESIGN_SCORER=$DESIGN_SCORER (all reward channels enabled)"
echo "[aescode-32b] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[aescode-32b] EXPERIMENT=$EXPERIMENT_NAME"

bash "${AESCODE_HOME}/rl_verl/train_gdpo.sh"
