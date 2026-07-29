#!/usr/bin/env bash
#================================================================
# DesignCoder 32B — GDPO RL, single node (8 GPUs)
#
# Thin launcher for the 32B policy. The 32B model is large enough that vLLM
# wake-up + FSDP can OOM at the engine's 8B defaults, so this launcher lowers
# GPU_MEM_UTIL and the log-prob micro-batches. Tune further for your hardware.
#
# Reference hardware: 1 × 8×B200 (183 GB). On smaller GPUs you will likely
# need offload (PARAM_OFFLOAD=True OPTIMIZER_OFFLOAD=True) and a lower TP/batch.
#
# Before running:
#   1. ray start --head --num-gpus=8 --num-cpus=48 --disable-usage-stats
#   2. prepare dataset/verl_{train,val}.parquet
#   3. set MODEL to your 32B SFT checkpoint
#
# Usage:
#   MODEL=/path/to/sft_32b bash rl_verl/launch_32b_single_node.sh
#================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Topology: single node, 8 GPUs ──
export NNODES="${NNODES:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ── 32B model ──
export MODEL="${MODEL:?set MODEL to your 32B SFT checkpoint dir}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"

# ── 32B memory guard: engine's 8B defaults (0.85 / mb=8) OOM on vLLM wake-up ──
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
export ROLLOUT_LOG_PROB_MICRO_BATCH="${ROLLOUT_LOG_PROB_MICRO_BATCH:-2}"
export REF_LOG_PROB_MICRO_BATCH="${REF_LOG_PROB_MICRO_BATCH:-2}"
export PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"

# ── Design reward on (rubric judge) ──
export DESIGN_SCORER="${DESIGN_SCORER:-rubric}"
export DISABLED_VALIDATORS="${DISABLED_VALIDATORS:-}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gdpo_32b_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_gdpo.sh" "$@"
