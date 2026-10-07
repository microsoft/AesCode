#!/usr/bin/env bash
# 1_infer.sh <CKPT_PATH> <RUN_LABEL> [TP] [NGPU]
# ---------------------------------------------------------------------------
# Stage 1: serve a checkpoint with vLLM and generate N rollouts per sample.
#
#   source eval/run/env.sh
#   bash eval/run/1_infer.sh /abs/path/to/ckpt my_label 4 4
#
# To evaluate a hosted API instead of local weights, skip this script and call
# eval/eval_vllm_infer.py directly with --endpoints and --model; the
# sampling knobs below are the ones to reproduce.
#
# NO REFERENCE IMAGE: set CLEAN_NOIMAGE=1 to drop the reference image and the
# [Reference Image] block from the prompt. That is the setting for the
# reference-free arms of the table. It is not the same as --no-image-eval,
# which adds a training-parity note that only a model trained with it has seen.
#
# Each run owns its port range and kills only the processes it started, so two
# checkpoints can share one node on disjoint GPUs: set GPU_OFFSET and PORT_BASE
# per run. There is deliberately no pattern-kill over the whole user's
# processes; that silently takes down a model a parallel run is serving.
set -uo pipefail
: "${BENCH_ROOT:?source eval/run/env.sh first}"
: "${PYTHON_BIN:?}" ; : "${SPEC:?}" ; : "${IMAGE_DIR:?}"

CKPT="${1:?usage: bash eval/run/1_infer.sh <CKPT_PATH> <RUN_LABEL> [TP] [NGPU]}"
LABEL="${2:?need RUN_LABEL}"
TP="${3:-4}"
NGPU="${4:-4}"
GPU_OFFSET="${GPU_OFFSET:-0}"
PORT_BASE="${PORT_BASE:-8000}"
case "$CKPT" in /*) ;; *) CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")";; esac

# ---- sampling knobs: these define the benchmark setting, do not vary them ----
N="${N:-3}"; TEMP=0.8; TOP_P=0.95; MAX_TOKENS=12000; MAX_MODEL_LEN=24576
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
READY_TICKS="${READY_TICKS:-360}"          # x5s; a large MoE takes well over 20min
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
INFER_EXTRA=""
[ "${CLEAN_NOIMAGE:-0}" = "1" ] && INFER_EXTRA="--clean-noimage"
N_ENDPOINT=$(( NGPU / TP )); INFER_WORKERS=$(( N_ENDPOINT * 48 ))

OUT="$BENCH_ROOT/runs/${LABEL}${RUN_SUFFIX:-_final300}"
HTML_DIR="$OUT/html"; LOGS="$OUT/logs"
mkdir -p "$HTML_DIR" "$LOGS"
TOTAL=$(wc -l < "$SPEC"); TARGET=$((TOTAL * N))
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

[ -f "$CKPT/config.json" ] || { log "ERROR: $CKPT has no config.json"; exit 2; }
[ $(( NGPU % TP )) -eq 0 ] || { log "ERROR: NGPU%TP != 0"; exit 2; }

# The prompt mode cannot be recovered from the HTML afterwards, so it is written
# next to the output before a single token is generated.
cat > "$OUT/gen_meta.json" <<EOF
{"label": "$LABEL", "ckpt": "$CKPT", "n": $N, "temperature": $TEMP,
 "top_p": $TOP_P, "max_tokens": $MAX_TOKENS,
 "with_ref_image": $([ "${CLEAN_NOIMAGE:-0}" = "1" ] && echo false || echo true),
 "prompt_mode": "$([ "${CLEAN_NOIMAGE:-0}" = "1" ] && echo clean_noimage || echo with_image)",
 "generated_at": "$(date -Iseconds)"}
EOF

log "infer $LABEL  ckpt=$CKPT  TP=$TP NGPU=$NGPU (offset $GPU_OFFSET) -> $N_ENDPOINT endpoint(s)"
log "  n=$N target=$TARGET html  mode=$([ -n "$INFER_EXTRA" ] && echo clean_noimage || echo with_image)"

ENDPOINTS=""; VLLM_PIDS=()
for e in $(seq 0 $((N_ENDPOINT-1))); do
  port=$((PORT_BASE+e)); gs=$(( GPU_OFFSET + e*TP )); ge=$(( gs+TP-1 )); devs=$(seq -s, $gs $ge)
  log "  endpoint $e GPUs [$devs] port $port"
  # Servers sharing one Triton cache race each other; the loser dies on a
  # half-written entry and only surfaces later as "n/m endpoints ready".
  CUDA_VISIBLE_DEVICES=$devs TRITON_CACHE_DIR="$LOGS/.triton_ep${e}" \
  nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$CKPT" --served-model-name model $EXTRA_VLLM_ARGS \
      --tensor-parallel-size $TP --port "$port" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --trust-remote-code --max-model-len "$MAX_MODEL_LEN" --host 0.0.0.0 \
      --enable-prefix-caching --max-num-seqs 128 \
      < /dev/null > "$LOGS/vllm_ep${e}.log" 2>&1 &
  VLLM_PIDS+=($!)
  [ -n "$ENDPOINTS" ] && ENDPOINTS="${ENDPOINTS},"
  ENDPOINTS="${ENDPOINTS}http://localhost:${port}/v1"
done

kill_vllm(){
  for p in "${VLLM_PIDS[@]}"; do
    pkill -9 -P "$p" 2>/dev/null || true
    kill -9 "$p" 2>/dev/null || true
  done
}
trap kill_vllm EXIT

log "waiting for vLLM..."
rc=0
for tick in $(seq 1 "$READY_TICKS"); do
  rc=0
  for e in $(seq 0 $((N_ENDPOINT-1))); do
    curl -s --connect-timeout 1 "http://localhost:$((PORT_BASE+e))/v1/models" 2>/dev/null \
      | grep -q '"object"' && rc=$((rc+1))
  done
  [ $rc -eq $N_ENDPOINT ] && { log "all ready (${tick}x5s)"; break; }
  sleep 5
done
[ $rc -eq $N_ENDPOINT ] || { log "ERROR: only $rc/$N_ENDPOINT endpoints ready"; exit 3; }

for try in $(seq 1 5); do
  log "infer attempt $try/5 ($TOTAL x n=$N, workers=$INFER_WORKERS)"
  "$PYTHON_BIN" "$BENCH_ROOT/eval/eval_vllm_infer.py" \
      --spec-jsonl "$SPEC" --image-dir "$IMAGE_DIR" --endpoints "$ENDPOINTS" \
      --mode sample --n "$N" --temperature "$TEMP" --top-p "$TOP_P" \
      --output-dir "$HTML_DIR" --max-tokens "$MAX_TOKENS" \
      --workers "$INFER_WORKERS" --resume $INFER_EXTRA \
      2>&1 | tee -a "$LOGS/infer.log"
  nh=$(find "$HTML_DIR" -maxdepth 1 -name '*.html' | wc -l)
  log "  html=$nh / $TARGET"
  [ "$nh" -ge "$TARGET" ] && { log "infer complete"; break; }
  sleep 5
done
kill_vllm; trap - EXIT
log "DONE infer $LABEL -> next: bash eval/run/2_validate.sh $LABEL"
