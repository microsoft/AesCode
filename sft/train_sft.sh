#!/usr/bin/env bash

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export DS_BUILD_OPS=0
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Data source: pipeline-dirs mode (preferred) or legacy JSONL mode ───────
# Pipeline-dirs mode: set SPEC_JSONL + CLEAN_HTML_DIR (+ optional IMAGE_DIR)
# Legacy mode:        set DATA_PATH to a .jsonl file or directory
_SFT_DATA_ROOT="${SFT_DATA_ROOT:-${AESCODE_ROOT}/spec_train15k/final}"
# Only fall back to the packaged spec file when the caller has not asked
# for legacy single-jsonl mode; otherwise the elif DATA_PATH arm below is
# unreachable, since ":-" also substitutes on an empty string.
if [[ -n "${DATA_PATH:-}" ]]; then
  SPEC_JSONL="${SPEC_JSONL:-}"
else
  SPEC_JSONL="${SPEC_JSONL:-${_SFT_DATA_ROOT}/split_sft.jsonl}"
fi
CLEAN_HTML_DIR="${CLEAN_HTML_DIR:-${_SFT_DATA_ROOT}/clean_html}"
IMAGE_DIR="${IMAGE_DIR:-${_SFT_DATA_ROOT}/images}"
DATA_PATH="${DATA_PATH:-}"

# ── Model & Checkpoint output ─────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-${AESCODE_ROOT}/models/Qwen3-VL-4B-Instruct}"
LR="${LR:-1e-5}"
DS_CONFIG="${DS_CONFIG:-${PROJECT_ROOT}/sft/configs/ds_zero2.json}"
# Auto-construct output dir on blob (shared storage, both nodes can see).
_CKPT_ROOT="${CKPT_ROOT:-${AESCODE_ROOT}/checkpoints}"
_MODEL_SHORT="$(basename "${MODEL_NAME}" | sed 's/-Instruct//')"
OUTPUT_DIR="${OUTPUT_DIR:-${_CKPT_ROOT}/${_MODEL_SHORT}_sft_lr${LR}}"

DEBUG_MODE=0
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_MODE=1
fi

NUM_GPUS="${NUM_GPUS:-8}"

# ── Multi-node configuration ────────────────────────────────────────────────
# Default: 2 nodes × 8 GPUs = 16 GPUs (clusters A/B).
# Override NNODES=1 for single-node training.
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"
NODE_RANK="${NODE_RANK:-0}"

PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-$((256 * 28 * 28))}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-$((1280 * 28 * 28))}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
PACKING="${PACKING:-1}"

# ── Model-size-aware defaults ──────────────────────────────────────────
# Auto-tune BS/GRAD_ACCUM based on model size.
# IMAGE_MAX_PIXELS=1280*28*28 across all sizes.
# Verified on 2×8 A100-80G, ZeRO-2 + optimizer offload=CPU + packing.
#   4B: BS=1, GRAD_ACCUM=4 (safe with 1280*28*28 pixels)
#   8B: BS=1, GRAD_ACCUM=4
# These apply ONLY if user hasn't explicitly overridden PER_DEVICE_BS.
if [[ "${PER_DEVICE_BS}" == "2" && "${GRAD_ACCUM}" == "2" ]]; then
  # Defaults not overridden — apply model-aware tuning
  _model_lower="$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${_model_lower}" == *"8b"* || "${_model_lower}" == *"4b"* ]]; then
    PER_DEVICE_BS=1
    GRAD_ACCUM=4
    echo "[auto-tune] ${_model_lower} detected → BS=1, GRAD_ACCUM=4, IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS}"
  fi
fi

if [[ "${DEBUG_MODE}" -eq 1 ]]; then
  NUM_GPUS=1
  NPROC_PER_NODE=1
  NNODES=1
  GRAD_ACCUM=1
  OUTPUT_DIR="${OUTPUT_DIR}_debug"
  MAX_LENGTH=2048
  IMAGE_MAX_PIXELS=$((448 * 28 * 28))
fi

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "DeepSpeed config not found: ${DS_CONFIG}"
  exit 1
fi

# ── Build data args ──────────────────────────────────────────────────────────
DATA_ARGS=""
if [[ -n "${SPEC_JSONL}" ]]; then
  # Pipeline-dirs mode
  DATA_ARGS="--spec_jsonl ${SPEC_JSONL}"
  [[ -n "${CLEAN_HTML_DIR}" ]] && DATA_ARGS="${DATA_ARGS} --clean_html_dir ${CLEAN_HTML_DIR}"
  [[ -n "${IMAGE_DIR}" ]]      && DATA_ARGS="${DATA_ARGS} --image_dir ${IMAGE_DIR}"
  DATA_SOURCE="${SPEC_JSONL} (+dirs)"
elif [[ -n "${DATA_PATH}" ]]; then
  # Legacy mode
  if [[ ! -e "${DATA_PATH}" ]]; then
    echo "Data not found: ${DATA_PATH}"
    exit 1
  fi
  DATA_ARGS="--data_path ${DATA_PATH}"
  DATA_SOURCE="${DATA_PATH}"
else
  echo "ERROR: set SPEC_JSONL (pipeline mode) or DATA_PATH (legacy mode)"
  exit 1
fi

echo "========================================================"
echo "  SFT Training"
echo "  Model  : ${MODEL_NAME}"
echo "  Data   : ${DATA_SOURCE}"
echo "  Output : ${OUTPUT_DIR}"
echo "  Mode   : $([[ "${DEBUG_MODE}" -eq 1 ]] && echo "debug" || echo "train")"
echo "  Length : ${MAX_LENGTH}"
echo "  Packing: $([[ "${PACKING}" -eq 1 ]] && echo "ON" || echo "OFF")"
echo "  GPUs   : ${NPROC_PER_NODE} × ${NNODES} node(s) | global_bs = $((PER_DEVICE_BS * NPROC_PER_NODE * NNODES * GRAD_ACCUM))"
echo "========================================================"

COMMON_ARGS=(
  --model_name_or_path "${MODEL_NAME}"
  ${DATA_ARGS}
  --output_dir "${OUTPUT_DIR}"
  --deepspeed "${DS_CONFIG}"
  --max_length "${MAX_LENGTH}"
  --image_min_pixels "${IMAGE_MIN_PIXELS}"
  --image_max_pixels "${IMAGE_MAX_PIXELS}"
  --learning_rate "${LR}"
  --per_device_train_batch_size "${PER_DEVICE_BS}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type cosine
  --weight_decay 0.0
  --logging_steps 1
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --dataloader_num_workers 8
  --report_to wandb
  --seed 42
)
if [[ "${PACKING}" -eq 1 ]]; then
  COMMON_ARGS+=(--packing)
fi

LAUNCH_CMD=("${PY:-python3.10}" -m torch.distributed.run
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  "${PROJECT_ROOT}/sft/train_sft.py"
)

"${LAUNCH_CMD[@]}" "${COMMON_ARGS[@]}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
