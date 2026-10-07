#!/usr/bin/env bash
#================================================================
# GDPO reinforcement learning — shared launcher.
#
# Carries the hyper-parameters both released models were trained with.
# Do not run this directly; use rl_verl/train_8b.sh or rl_verl/train_32b.sh,
# which set the model, the data and the per-scale memory settings and then
# source these defaults. Every value below can be overridden from the
# environment, so ablations need no edits here.
#
# The defaults assume one node with 8 GPUs of at least 180GB (the released
# runs used 8×B200-183GB). On smaller cards, lower PPO_MICRO_BATCH,
# ROLLOUT_LOG_PROB_MICRO_BATCH and GPU_MEM_UTIL, and turn the offload flags
# back on.
#
# Prerequisites:
#   1. The verl environment active, patched via verl_patches/apply_verl_patch.sh
#   2. A single-node Ray cluster running: ray start --head --num-gpus=8
#   3. RL parquet shards prepared (see rl_verl/data_prep.py)
#   4. A cold-start SFT checkpoint to initialise from
#   5. .env present with OPENAI_BASE_URL / OPENAI_API_KEY / VLM_MODEL
#      for the rubric judge
#
# Usage:
#   bash rl_verl/train_gdpo.sh --debug            # tiny batch, smoke test
#================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Load .env early so defaults below can inherit from it ──
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    _SCRIPT_PROJECT_ROOT="$PROJECT_ROOT"
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
    PROJECT_ROOT="$_SCRIPT_PROJECT_ROOT"
    unset _SCRIPT_PROJECT_ROOT
fi

# Force verl environment on PATH (the training cluster doesn't have conda activate)
export PATH="${AESCODE_PY_ENV}/bin:$PATH"
PYTHON="${PYTHON:-${AESCODE_PY_ENV}/bin/python}"

# -------------------------------------------------------------------------
# Chromium system-lib dependencies — vendor inside env (added 2026-06-21)
# Why:
#   - playwright chromium dynamically loads libpango / libpangocairo /
#     libpangoft2 / libfribidi / libthai / libdatrie. The training worker
#     base image is missing them (it ships nss/cairo/gbm/drm/xkbcommon
#     but not the pango stack).
#   - server (the build host, ubuntu 22.04 + apt) has them under
#     /usr/lib/x86_64-linux-gnu/; we cp them into "$ENV_ROOT/lib_extra/"
#     so the env rsync (every 10s server -> worker) carries them along.
#   - LD_LIBRARY_PATH below makes the in-env chrome find them at runtime,
#     and we re-export it into ray runtime_env so AgentLoopWorker children
#     inherit it (otherwise verl forks lose it).
#
# Re-vendor recipe (server, after worker reboot or fresh env build):
#   ENV=${AESCODE_PY_ENV}
#   mkdir -p $ENV/lib_extra
#   cp /usr/lib/x86_64-linux-gnu/libpango-1.0.so.0*       $ENV/lib_extra/
#   cp /usr/lib/x86_64-linux-gnu/libpangocairo-1.0.so.0*  $ENV/lib_extra/
#   cp /usr/lib/x86_64-linux-gnu/libpangoft2-1.0.so.0*    $ENV/lib_extra/
#   cp /usr/lib/x86_64-linux-gnu/libfribidi.so.0*         $ENV/lib_extra/
#   cp /usr/lib/x86_64-linux-gnu/libthai.so.0*            $ENV/lib_extra/
#   cp /usr/lib/x86_64-linux-gnu/libdatrie.so.1*          $ENV/lib_extra/
# -------------------------------------------------------------------------
ENV_ROOT="${AESCODE_PY_ENV}"
LIB_EXTRA="$ENV_ROOT/lib_extra"
export LD_LIBRARY_PATH="$LIB_EXTRA:${LD_LIBRARY_PATH:-}"

# Pre-launch self-check: chromium must resolve all libs OR we hard-fail fast
# (rather than dying mid-step-1 with render_image_missing for every rollout)
CHROME="${AESCODE_ROOT}/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome"
if [[ -x "$CHROME" ]]; then
    _MISSING=$(ldd "$CHROME" 2>&1 | grep "not found" || true)
    if [[ -n "$_MISSING" ]]; then
        echo "[train_gdpo.sh] FATAL: chromium has unresolved libraries:" >&2
        echo "$_MISSING" >&2
        echo "[train_gdpo.sh] Likely cause: worker reboot lost system libs." >&2
        echo "[train_gdpo.sh] Fix: cp missing libs from server's" >&2
        echo "[train_gdpo.sh] /usr/lib/x86_64-linux-gnu/ into $LIB_EXTRA/" >&2
        echo "[train_gdpo.sh] See the header comment of this script for the cp recipe." >&2
        exit 90
    fi
else
    echo "[train_gdpo.sh] FATAL: chromium binary missing at" >&2
    echo "[train_gdpo.sh]   $CHROME" >&2
    echo "[train_gdpo.sh] Likely cause: worker reboot lost ~/.cache (not rsync'd)." >&2
    echo "[train_gdpo.sh] Fix on worker: playwright install chromium" >&2
    exit 91
fi

# ── Cleanup trap: kill orphan GPU processes on exit (single node) ──
_cleanup_gpu_procs() {
    echo "[train_gdpo.sh] Cleaning up GPU processes..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | sort -u | xargs -r kill -9 2>/dev/null || true
    echo "[train_gdpo.sh] GPU cleanup done."
}
trap _cleanup_gpu_procs EXIT

# ── Model & Data ──
DATA_ROOT="${DATA_ROOT:-${AESCODE_ROOT}}"
SPEC_DIR="${SPEC_DIR:-${DATA_ROOT}/spec_train15k/final}"
# Point this at the cold-start SFT checkpoint produced by sft/train_sft.sh.
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/ckpt/sft_phase1_generate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/training/checkpoints}"
MODEL="${MODEL:-${MODEL_DIR}}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/dataset/verl_train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/dataset/verl_val.parquet}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/verl_gdpo_${RUN_TIMESTAMP}}"
export OUTPUT_DIR

# ── Cluster (single node) ──
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ── Training hyperparameters (tuned for 8×B200 instead of 16×A100) ──
# Keep global batch == 128 to match A100 16-gpu original training dynamics.
# B200 2.3× memory + 8 cards (vs 16 A100) → bigger micro batches to keep
# the same global batch with half the GPUs.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"  # 071231 sweet-spot: token-mean loss + larger LR -> ~5x faster val gain
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
ROLLOUT_N="${ROLLOUT_N:-8}"
KL_COEF="${KL_COEF:-0.001}"  # 071231 verified: 17h no-collapse run, val 0.804->0.895
TEMPERATURE="${TEMPERATURE:-1.0}"

# ── 071231-baked: training dynamics ──
# Below 4 vars were override flags on the prior run; now baked as defaults.
# Override at call-site if you really know what you're doing.
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"          # 071231: 0.001 prevents premature entropy collapse on long-CoT
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"      # 071231: token-mean > seq-mean-token-mean (no length dilution)
VAL_KWARGS_N="${VAL_KWARGS_N:-8}"                # 071231: 8 rollouts/val (was 16, halve to save 50% val time)
PLAYWRIGHT_CONCURRENCY="${PLAYWRIGHT_CONCURRENCY:-8}"  # 071231: 8 fixed render_fail 4.1% -> 0.04%
export ENTROPY_COEFF LOSS_AGG_MODE VAL_KWARGS_N PLAYWRIGHT_CONCURRENCY
TOP_P="${TOP_P:-1.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-4}"
# Whether to use gradient checkpointing in actor backward.
# 2026-06-09: tried False on B200, OOM in actor backward at 177-182 GB on GPU 0.
# vllm wake_up doesn't fully release KV cache, GPU 0 stays hot. With batch=128
# (A100 dynamics anchor) we MUST keep grad_ckpt=True. Verified-stable peak ~163 GB.
ENABLE_GRADIENT_CKPT="${ENABLE_GRADIENT_CKPT:-True}"

AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-${PROJECT_ROOT}/rl_verl/agent_loop_config.yaml}"

# ── vLLM rollout (B200-tuned) ──
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"   # Sanity-verified value; grad_ckpt=True keeps actor footprint ~12 GB
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"

# ── B200 advantages: turn OFF offload and enforce_eager ──
# (defaults differ from A100 baseline; override via env if needed)
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
MULTI_STAGE_WAKE_UP="${MULTI_STAGE_WAKE_UP:-True}"
ROLLOUT_LOG_PROB_MICRO_BATCH="${ROLLOUT_LOG_PROB_MICRO_BATCH:-8}"
REF_LOG_PROB_MICRO_BATCH="${REF_LOG_PROB_MICRO_BATCH:-8}"

# ── VLM Reward (DesignValidator) ──
# The design channel is part of the method: it carries the Visual Graph Rubric
# judge. Both released runs (8B and 32B) were trained with it enabled, so no
# validator is disabled by default. Set DISABLED_VALIDATORS to drop channels
# when running ablations.
DISABLED_VALIDATORS="${DISABLED_VALIDATORS-}"
VLM_REWARD_TIMEOUT="${VLM_REWARD_TIMEOUT:-60}"
DESIGN_GEN_ASSEMBLE_TIMEOUT="${DESIGN_GEN_ASSEMBLE_TIMEOUT:-3600}"
DESIGN_GEN_TOTAL_TIMEOUT="${DESIGN_GEN_TOTAL_TIMEOUT:-4200}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-600}"
# `rubric` is the Visual Graph Rubric judge used for both released models.
# `swiss` (pairwise tournament) and `clip` exist for ablations only.
DESIGN_SCORER="${DESIGN_SCORER:-rubric}"
CLIP_DEVICE="${CLIP_DEVICE:-cpu}"   # only the clip scorer loads CLIP; cpu = no-op otherwise

# ── RubricDesignScorer (DESIGN_SCORER=rubric) ──
# 2026-06-16 hang root cause was unbounded fan-out of B*N rollouts each doing
# PIL+JPEG+base64 of a 1024px screenshot serialised on the same event loop
# that drives vLLM AsyncMPClient zmq poll. Bound the in-flight rubric VLM calls
# at MAX_INFLIGHT (default = VLM_REWARD_CONCURRENCY) and rely on per-call
# TIMEOUT plus an absolute HARD_CAP (fast-fail inside REWARD_TIMEOUT) to
# release permits under any failure mode.
RUBRIC_VLM_MAX_INFLIGHT="${RUBRIC_VLM_MAX_INFLIGHT:-${VLM_REWARD_CONCURRENCY:-32}}"
RUBRIC_VLM_TIMEOUT="${RUBRIC_VLM_TIMEOUT:-90}"
RUBRIC_VLM_HARD_CAP="${RUBRIC_VLM_HARD_CAP:-300}"
RUBRIC_VLM_MAX_TOKENS="${RUBRIC_VLM_MAX_TOKENS:-1500}"
RUBRIC_VLM_MAX_RETRIES="${RUBRIC_VLM_MAX_RETRIES:-3}"
RUBRIC_VLM_BACKOFF_BASE="${RUBRIC_VLM_BACKOFF_BASE:-1.0}"
RUBRIC_VLM_TEMPERATURE="${RUBRIC_VLM_TEMPERATURE:-default}"
export RUBRIC_VLM_MAX_INFLIGHT RUBRIC_VLM_TIMEOUT RUBRIC_VLM_HARD_CAP
export RUBRIC_VLM_MAX_TOKENS RUBRIC_VLM_MAX_RETRIES RUBRIC_VLM_BACKOFF_BASE RUBRIC_VLM_TEMPERATURE

# The judge runs on the shared hosted endpoint unless it is given its own.
VLM_REWARD_BASE_URL="${VLM_REWARD_BASE_URL:-${OPENAI_BASE_URL:-}}"
VLM_REWARD_API_KEY="${VLM_REWARD_API_KEY:-${OPENAI_API_KEY:-}}"
VLM_REWARD_MODEL="${VLM_REWARD_MODEL:-${VLM_MODEL:-}}"

export VLM_REWARD_TIMEOUT DISABLED_VALIDATORS DESIGN_GEN_ASSEMBLE_TIMEOUT DESIGN_GEN_TOTAL_TIMEOUT REWARD_TIMEOUT DESIGN_SCORER CLIP_DEVICE
export VLM_REWARD_BASE_URL VLM_REWARD_API_KEY VLM_REWARD_MODEL VLM_REWARD_CONCURRENCY VLM_GLOBAL_RPM_LIMIT VLM_RPM_HEADROOM_PCT VLM_ACTIVE_ENDPOINTS

# ── OCR disk cache (TextValidator) ──
VERIFIER_OCR_CACHE_DIR="${VERIFIER_OCR_CACHE_DIR:-${DATA_ROOT}/ocr_cache_split_rl}"
export VERIFIER_OCR_CACHE_DIR

# ── GDPO reward dimensions (7 quality dims) ──
GDPO_REWARD_KEYS="[execution,text,boundary,layout,tablechart,whitespace,design]"
GDPO_REWARD_WEIGHTS="[1.0,1.0,1.0,1.0,3.0,1.0,1.0]"
# Per-validator metric mode: skip-zero means metrics aggregate over the
# subset of samples that returned a real (non-zero) score. Only tablechart
# is truly sparse (fires only on samples with tables); the rare true 0 from
# a fully-failed table parse is acceptable noise. design with rubric scorer
# always runs and a 0 is a real failure signal — do NOT skip it.
GDPO_SKIP_ZERO_KEYS="[tablechart]"

# ── Training Monitor ──
REWARD_MONITOR_ENABLED="${REWARD_MONITOR_ENABLED:-1}"
REWARD_MONITOR_INTERVAL="${REWARD_MONITOR_INTERVAL:-10}"
REWARD_MONITOR_OUTPUT="${REWARD_MONITOR_OUTPUT:-${OUTPUT_DIR}/monitor}"
export REWARD_MONITOR_ENABLED REWARD_MONITOR_INTERVAL REWARD_MONITOR_OUTPUT

# ── Logging & Checkpointing ──
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-aescode-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl_gdpo_${RUN_TIMESTAMP}}"
LOGGER="${LOGGER:-[\"console\"]}"
unset WANDB_RUN_ID 2>/dev/null || true
export WANDB_RUN_ID="${EXPERIMENT_NAME}"
export WANDB_RESUME="${WANDB_RESUME:-never}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
CKPT_KEEP_LAST="${CKPT_KEEP_LAST:-5}"
CKPT_KEEP_EVERY="${CKPT_KEEP_EVERY:-5}"
CKPT_MILESTONE_EVERY="${CKPT_MILESTONE_EVERY:-50}"
RENDER_SAMPLE_DIR="${RENDER_SAMPLE_DIR:-${OUTPUT_DIR}/render_samples}"
RENDER_SAMPLE_INTERVAL="${RENDER_SAMPLE_INTERVAL:-5}"
RENDER_SAMPLE_N="${RENDER_SAMPLE_N:-10}"

# ── Debug mode (smaller batch for quick verification) ──
DEBUG=false
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        DEBUG=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

if [[ "$DEBUG" == "true" ]]; then
    NNODES=1
    N_GPUS_PER_NODE=8
    TRAIN_BATCH_SIZE=8
    MAX_PROMPT_LEN=6144
    MAX_RESPONSE_LEN=4096
    ROLLOUT_N=2
    PPO_MINI_BATCH=8
    PPO_MICRO_BATCH=1
    TOTAL_EPOCHS=2
    SAVE_FREQ=5
    TEST_FREQ=-1
    GPU_MEM_UTIL=0.50
    VLLM_MAX_MODEL_LEN=10240
    EXPERIMENT_NAME="debug_gdpo_${RUN_TIMESTAMP}"
    REWARD_MONITOR_INTERVAL=2
    echo "[train_gdpo.sh] DEBUG mode: ${NNODES} node × ${N_GPUS_PER_NODE} GPUs, batch=8, generate-only"
fi

# ── Check prerequisites ──
if ! $PYTHON -c "import verl" &>/dev/null; then
    echo "ERROR: verl not importable. Check ${AESCODE_PY_ENV}"
    exit 1
fi

if ! $PYTHON -c "import ray" &>/dev/null; then
    echo "ERROR: ray not importable."
    exit 1
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: Train data not found: $TRAIN_DATA"
    echo "Run data prep first:"
    echo "  python -m rl_verl.data_prep \\"
    echo "    --input ${SPEC_DIR}/split_rl.jsonl \\"
    echo "    --val-input ${SPEC_DIR}/split_val.jsonl \\"
    echo "    --rubric-input ${SPEC_DIR}/rubrics.jsonl \\"
    echo "    --image-dir ${SPEC_DIR}/images \\"
    echo "    --output-dir dataset/"
    exit 1
fi

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: Model checkpoint not found: $MODEL"
    exit 1
fi

# ── Verify Ray is running (single-node, started via 'ray start --head') ──
if ! ray status &>/dev/null 2>&1; then
    echo "ERROR: Ray not running. Start it first:"
    echo "  ray start --head --num-gpus=${N_GPUS_PER_NODE} --num-cpus=48 --disable-usage-stats"
    exit 1
fi

# ── Create output directory ──
mkdir -p "$OUTPUT_DIR"

# ── Persist log to checkpoint dir ──
LOG_FILE="${OUTPUT_DIR}/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[train_gdpo.sh] Logging to $LOG_FILE"

# ── Print config ──
echo "================================================================"
echo "  AesCode — GDPO reinforcement learning"
echo "================================================================"
echo "  Model              : $MODEL"
echo "  Train data         : $TRAIN_DATA"
echo "  Val data           : $VAL_DATA"
echo "  Output             : $OUTPUT_DIR"
echo "  Nodes x GPUs       : ${NNODES} × ${N_GPUS_PER_NODE} = $((NNODES * N_GPUS_PER_NODE)) GPUs"
echo "  Rollout TP size    : $ROLLOUT_TP_SIZE"
echo "  Rollout N          : $ROLLOUT_N"
echo "  Batch size         : $TRAIN_BATCH_SIZE  (mini=$PPO_MINI_BATCH micro=$PPO_MICRO_BATCH)"
echo "  Max prompt / resp  : $MAX_PROMPT_LEN / $MAX_RESPONSE_LEN"
echo "  vLLM max model len : $VLLM_MAX_MODEL_LEN"
echo "  GPU mem util       : $GPU_MEM_UTIL"
echo "  Param offload      : $PARAM_OFFLOAD  (was True for A100)"
echo "  Optim offload      : $OPTIMIZER_OFFLOAD  (was True for A100)"
echo "  Ref offload        : $REF_PARAM_OFFLOAD"
echo "  Grad checkpoint    : $ENABLE_GRADIENT_CKPT  (False = faster, B200 has memory)"
echo "  Enforce eager      : $ENFORCE_EAGER  (was True for A100)"
echo "  LR / KL coef       : $LEARNING_RATE / $KL_COEF"
echo "  Epochs             : $TOTAL_EPOCHS"
echo "  Resume mode        : $RESUME_MODE"
echo "  Save / Test freq   : $SAVE_FREQ / $TEST_FREQ"
echo "  Debug              : $DEBUG"
echo "  Design scorer      : $DESIGN_SCORER  (CLIP on $CLIP_DEVICE)"
if [[ "$DESIGN_SCORER" == "rubric" ]]; then
    echo "  Rubric MAX_INFLIGHT: $RUBRIC_VLM_MAX_INFLIGHT (per-worker fan-out cap)"
    echo "  Rubric TIMEOUT     : ${RUBRIC_VLM_TIMEOUT}s per call, HARD_CAP ${RUBRIC_VLM_HARD_CAP}s outer"
    echo "  Rubric retries     : $RUBRIC_VLM_MAX_RETRIES (backoff base ${RUBRIC_VLM_BACKOFF_BASE}s)"
fi
echo "================================================================"

# ── Ensure PYTHONPATH includes project root ──
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export NCCL_ASYNC_ERROR_HANDLING=0
export RAY_DEDUP_LOGS=0
# Reduce memory fragmentation in actor backward (B200 OOM seen at 178 GB limit).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# HTML sample rendering env vars
export RENDER_SAMPLE_DIR="$RENDER_SAMPLE_DIR"
export RENDER_SAMPLE_INTERVAL="$RENDER_SAMPLE_INTERVAL"
export RENDER_SAMPLE_N="$RENDER_SAMPLE_N"

# ── Launch training ──
$PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gdpo \
    algorithm.use_kl_in_reward=False \
    "+algorithm.gdpo_reward_keys=$GDPO_REWARD_KEYS" \
    "+algorithm.gdpo_reward_weights=$GDPO_REWARD_WEIGHTS" \
    "+algorithm.skip_zero_reward_keys=$GDPO_SKIP_ZERO_KEYS" \
    "+algorithm.filter_groups.enable=True" \
    "+algorithm.filter_groups.metric=seq_reward" \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.val_max_samples=50 \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CKPT" \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
    actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
    actor_rollout_ref.actor.freeze_vision_tower=False \
    actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_KWARGS_N" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_stage_wake_up="$MULTI_STAGE_WAKE_UP" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.rollout.max_model_len="$VLLM_MAX_MODEL_LEN" \
    \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_step_generate \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH" \
    \
    reward.custom_reward_function.path="${PROJECT_ROOT}/rl_verl/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    "trainer.logger=$LOGGER" \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    "trainer.default_local_dir=$OUTPUT_DIR" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    trainer.max_actor_ckpt_to_keep=999 \
    trainer.max_critic_ckpt_to_keep=4 \
    \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PROJECT_ROOT" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1='1'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RENDER_SAMPLE_DIR=$RENDER_SAMPLE_DIR" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RENDER_SAMPLE_INTERVAL='$RENDER_SAMPLE_INTERVAL'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RENDER_SAMPLE_N='$RENDER_SAMPLE_N'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_REWARD_TIMEOUT='$VLM_REWARD_TIMEOUT'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DESIGN_GEN_ASSEMBLE_TIMEOUT='$DESIGN_GEN_ASSEMBLE_TIMEOUT'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DESIGN_GEN_TOTAL_TIMEOUT='$DESIGN_GEN_TOTAL_TIMEOUT'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_TIMEOUT='$REWARD_TIMEOUT'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DISABLED_VALIDATORS='$DISABLED_VALIDATORS'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DESIGN_SCORER='$DESIGN_SCORER'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CLIP_DEVICE='$CLIP_DEVICE'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PLAYWRIGHT_CONCURRENCY='$PLAYWRIGHT_CONCURRENCY'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.ROLLOUT_N='$ROLLOUT_N'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RUN_ID=${WANDB_RUN_ID}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RESUME=${WANDB_RESUME}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_INIT_TIMEOUT='180'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=${WANDB_API_KEY:-}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_REWARD_BASE_URL=${VLM_REWARD_BASE_URL:-}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_REWARD_API_KEY=${VLM_REWARD_API_KEY:-}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_REWARD_MODEL=${VLM_REWARD_MODEL:-}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_REWARD_CONCURRENCY='${VLM_REWARD_CONCURRENCY:-16}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_GLOBAL_RPM_LIMIT='${VLM_GLOBAL_RPM_LIMIT:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_RPM_HEADROOM_PCT='${VLM_RPM_HEADROOM_PCT:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLM_ACTIVE_ENDPOINTS='${VLM_ACTIVE_ENDPOINTS:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RUBRIC_VLM_MAX_INFLIGHT='${RUBRIC_VLM_MAX_INFLIGHT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RUBRIC_VLM_TIMEOUT='${RUBRIC_VLM_TIMEOUT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RUBRIC_VLM_HARD_CAP='${RUBRIC_VLM_HARD_CAP}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RUBRIC_VLM_MAX_TOKENS='${RUBRIC_VLM_MAX_TOKENS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RUBRIC_VLM_MAX_RETRIES='${RUBRIC_VLM_MAX_RETRIES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RUBRIC_VLM_BACKOFF_BASE='${RUBRIC_VLM_BACKOFF_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RUBRIC_VLM_TEMPERATURE='${RUBRIC_VLM_TEMPERATURE}'" \
    \
    "${EXTRA_ARGS[@]}"

# ── Checkpoint cleanup ──
echo ""
echo "[train_gdpo.sh] Running checkpoint cleanup..."
$PYTHON "${PROJECT_ROOT}/rl_verl/cleanup_checkpoints.py" \
    "$OUTPUT_DIR" \
    --keep-full-last "$CKPT_KEEP_LAST" \
    --keep-every "$CKPT_KEEP_EVERY" \
    --milestone-every "$CKPT_MILESTONE_EVERY"

echo ""
echo "================================================================"
echo "  Training complete!"
echo "  Checkpoints: $OUTPUT_DIR"
echo "================================================================"
