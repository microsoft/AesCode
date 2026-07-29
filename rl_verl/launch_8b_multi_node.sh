#!/usr/bin/env bash
#================================================================
# DesignCoder 8B — GDPO RL, multi node
#
# Same as launch_8b_single_node.sh but for NNODES>1. verl/Ray handle the
# cross-node actor/rollout placement; you only need a running multi-node Ray
# cluster and identical code + env on every node.
#
# Multi-node setup:
#   Head node:    ray start --head --num-gpus=8 --num-cpus=48 --disable-usage-stats
#   Worker nodes: ray start --address=<HEAD_IP>:6379 --num-gpus=8 --num-cpus=48
#   Then launch this script ON THE HEAD NODE only.
#
# The global train batch scales with total GPUs; keep TRAIN_BATCH_SIZE a
# multiple of (NNODES × N_GPUS_PER_NODE). Reference: 2 nodes × 8 GPUs, batch 128.
#
# Usage:
#   NNODES=2 MODEL=/path/to/sft_8b bash rl_verl/launch_8b_multi_node.sh
#================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Topology: multi node (set NNODES to your node count) ──
export NNODES="${NNODES:-2}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ── 8B model + rollout sizing ──
export MODEL="${MODEL:?set MODEL to your 8B SFT checkpoint dir}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Global batch across all GPUs (must divide by NNODES × N_GPUS_PER_NODE).
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"

export DESIGN_SCORER="${DESIGN_SCORER:-rubric}"
export DISABLED_VALIDATORS="${DISABLED_VALIDATORS:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gdpo_8b_mn_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_gdpo.sh" "$@"
