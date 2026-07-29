#!/usr/bin/env bash
#================================================================
# DesignCoder 8B — GDPO RL, single node (8 GPUs)
#
# Thin launcher: sets 8B-specific paths/overrides, then execs the engine
# (run_gdpo.sh). All hyperparameters live in the engine; override any of them
# by exporting the corresponding env var before calling this script.
#
# Reference hardware: 1 × 8×B200 (183 GB). On smaller GPUs (e.g. A100-80GB),
# enable offload and lower GPU_MEM_UTIL / PPO_MICRO_BATCH — see engine header.
#
# Before running:
#   1. ray start --head --num-gpus=8 --num-cpus=48 --disable-usage-stats
#   2. prepare dataset/verl_{train,val}.parquet  (python -m rl_verl.data_prep --help)
#   3. set MODEL to your 8B SFT checkpoint (or DATA_ROOT so the default resolves)
#
# Usage:
#   MODEL=/path/to/sft_8b bash rl_verl/launch_8b_single_node.sh
#   MODEL=/path/to/sft_8b bash rl_verl/launch_8b_single_node.sh --debug
#================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Topology: single node, 8 GPUs ──
export NNODES="${NNODES:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ── 8B model + rollout sizing (fits TP=4 comfortably on B200) ──
export MODEL="${MODEL:?set MODEL to your 8B SFT checkpoint dir}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# ── Design reward on (rubric judge). Requires VLM_REWARD_* in .env. ──
export DESIGN_SCORER="${DESIGN_SCORER:-rubric}"
export DISABLED_VALIDATORS="${DISABLED_VALIDATORS:-}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gdpo_8b_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_gdpo.sh" "$@"
