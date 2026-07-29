#!/usr/bin/env bash
#================================================================
# DesignCoder 32B — GDPO RL, multi node
#
# 32B across NNODES>1. Requires a running multi-node Ray cluster with identical
# code + env on every node (see below). Keeps the 32B memory guards from the
# single-node launcher.
#
# Multi-node setup:
#   Head node:    ray start --head --num-gpus=8 --num-cpus=48 --disable-usage-stats
#   Worker nodes: ray start --address=<HEAD_IP>:6379 --num-gpus=8 --num-cpus=48
#   Then launch this script ON THE HEAD NODE only.
#
# Keep TRAIN_BATCH_SIZE a multiple of (NNODES × N_GPUS_PER_NODE).
#
# Usage:
#   NNODES=2 MODEL=/path/to/sft_32b bash rl_verl/launch_32b_multi_node.sh
#================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Topology: multi node ──
export NNODES="${NNODES:-2}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ── 32B model ──
export MODEL="${MODEL:?set MODEL to your 32B SFT checkpoint dir}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"

# ── 32B memory guards ──
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
export ROLLOUT_LOG_PROB_MICRO_BATCH="${ROLLOUT_LOG_PROB_MICRO_BATCH:-2}"
export REF_LOG_PROB_MICRO_BATCH="${REF_LOG_PROB_MICRO_BATCH:-2}"
export PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"

export DESIGN_SCORER="${DESIGN_SCORER:-rubric}"
export DISABLED_VALIDATORS="${DISABLED_VALIDATORS:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gdpo_32b_mn_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_gdpo.sh" "$@"
