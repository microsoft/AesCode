#!/usr/bin/env bash
#
# AesCode-8B — GDPO reinforcement learning stage.
#
# Starts from the cold-start SFT checkpoint and optimises the decoupled
# cross-modal rewards with the Visual Graph Rubric judge enabled. Every
# hyper-parameter not set here is inherited from train_gdpo.sh, which already
# carries the released defaults.
#
# Required environment (see .env.example):
#   AESCODE_ROOT          writable root for checkpoints, caches and logs
#   AESCODE_HOME          this repository
#   OPENAI_BASE_URL       OpenAI-compatible endpoint backing the rubric judge
#   OPENAI_API_KEY        credential for that endpoint
#   VLM_MODEL             judge model name
#
# Usage:
#   bash rl_verl/train_8b.sh
#
set -euo pipefail

: "${AESCODE_ROOT:?set AESCODE_ROOT to a writable working root}"
: "${AESCODE_HOME:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AESCODE_HOME

cd "$AESCODE_HOME"

# ── Model & data ──
# MODEL must point at the cold-start SFT checkpoint produced by sft/train_sft.sh.
export MODEL="${MODEL:-${AESCODE_ROOT}/checkpoints/aescode_8b_sft/phase1-generate/final}"
export MODEL_DIR="$MODEL"
export TRAIN_DATA="${TRAIN_DATA:-${AESCODE_ROOT}/data/rl/train.parquet}"
export VAL_DATA="${VAL_DATA:-${AESCODE_ROOT}/data/rl/val.parquet}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${AESCODE_ROOT}/checkpoints}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-aescode_8b_gdpo_$(date +%Y%m%d_%H%M%S)}"

# ── Reward configuration ──
# The design channel carries the Visual Graph Rubric judge; it stays on.
export DESIGN_SCORER=rubric
export DISABLED_VALIDATORS=""

echo "[aescode-8b] MODEL=$MODEL"
echo "[aescode-8b] TRAIN=$TRAIN_DATA"
echo "[aescode-8b] VAL=$VAL_DATA"
echo "[aescode-8b] DESIGN_SCORER=$DESIGN_SCORER (all reward channels enabled)"
echo "[aescode-8b] EXPERIMENT=$EXPERIMENT_NAME"

bash "${AESCODE_HOME}/rl_verl/train_gdpo.sh"
