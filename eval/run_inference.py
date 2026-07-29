#!/usr/bin/env python3
"""DesignCoder — evaluation INFERENCE stage (release).

Serve an HF checkpoint (or a local safetensors dir) with vLLM and roll out
n generations for every eval spec. This stage ONLY produces HTML; scoring lives
entirely in run_validation.py.

  spec  ─vLLM(n)→  html/{sid}_k{0..n-1}.html

Fixed decoding protocol (do NOT change between checkpoints of the same family —
the whole point of a benchmark is that every model sees identical sampling):
  n=8  temperature=0.8  top_p=0.95  max_tokens=12000  max_model_len=24576

Usage (single 8xGPU node, tensor-parallel 8, one endpoint):
  python run_inference.py \
      --ckpt /path/to/checkpoint \
      --out  ./runs/my_model \
      --spec <data_root>/eval.jsonl \
      --image-dir <data_root>/images \
      --tp 8 --ngpu 8

Usage (4xGPU):
  python run_inference.py --ckpt ... --out ./runs/my_model --tp 4 --ngpu 4

Resumable: rerun the exact same command; already-produced *.html are skipped.

This is a thin wrapper around the repo's validated inference engine
(eval/eval_vllm_infer.py).
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout discovery. These release scripts live in <REPO>/eval/ alongside
# the inference/validation engines. Spec / images / rubrics are NOT bundled —
# pass them via --spec / --image-dir (see examples in README).
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent          # <REPO>/eval
REPO = EVAL_DIR.parent                               # <REPO>
INFER_ENGINE = EVAL_DIR / "eval_vllm_infer.py"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [infer] {msg}", flush=True)


def kill_stale_vllm(ports):
    for p in ports:
        subprocess.run(["pkill", "-9", "-f",
                        f"vllm.entrypoints.openai.api_server.*--port {p}"],
                       check=False)
        subprocess.run(["fuser", "-k", "-9", f"{p}/tcp"],
                       check=False, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-if", "EngineCore"], check=False)
    time.sleep(3)


def wait_gpu_free(ngpu, per_gpu_total_mb=143360, need_free_mb=110000, max_ticks=60):
    """Block until every GPU has >= need_free_mb free (previous job released)."""
    for tick in range(1, max_ticks + 1):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], text=True)
        except Exception:
            return  # no nvidia-smi (CPU box / container) → don't block
        used = [int(x) for x in out.split() if x.strip().isdigit()]
        busy = sum(1 for u in used if (per_gpu_total_mb - u) < need_free_mb)
        if busy == 0:
            log(f"GPU memory free after {tick} checks")
            return
        if tick % 6 == 0:
            log(f"waiting GPU free ({busy} still busy) tick {tick}/{max_ticks}")
        time.sleep(5)


def launch_vllm(ckpt, tp, ngpu, logs_dir, max_model_len, gpu_mem_util):
    """Launch ngpu/tp vLLM endpoints (tp GPUs each). Return (procs, endpoints, ports)."""
    n_endpoint = ngpu // tp
    ports = [8000 + e for e in range(n_endpoint)]
    procs, endpoints = [], []
    for e in range(n_endpoint):
        port = ports[e]
        devs = ",".join(str(g) for g in range(e * tp, e * tp + tp))
        log(f"endpoint {e}: GPUs [{devs}] port {port}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devs)
        logf = open(logs_dir / f"vllm_ep{e}.log", "w")
        p = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
             "--model", str(ckpt), "--served-model-name", "model",
             "--tensor-parallel-size", str(tp), "--port", str(port),
             "--gpu-memory-utilization", str(gpu_mem_util),
             "--trust-remote-code", "--max-model-len", str(max_model_len),
             "--host", "0.0.0.0", "--enable-prefix-caching", "--max-num-seqs", "128"],
            stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append(p)
        endpoints.append(f"http://localhost:{port}/v1")
    return procs, endpoints, ports


def wait_ready(ports, max_ticks=240):
    """Poll /v1/models until all endpoints answer. 32B TP load can take ~10min."""
    import urllib.request
    for tick in range(1, max_ticks + 1):
        ready = 0
        for port in ports:
            try:
                with urllib.request.urlopen(
                        f"http://localhost:{port}/v1/models", timeout=2) as r:
                    if b'"object"' in r.read():
                        ready += 1
            except Exception:
                pass
        if ready == len(ports):
            log(f"all {len(ports)} vLLM ready after {tick} checks")
            return True
        if tick % 12 == 0:
            log(f"vLLM ready {ready}/{len(ports)} (check {tick})")
        time.sleep(5)
    return False


def kill_procs(procs, ports):
    for p in procs:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    time.sleep(3)
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
    kill_stale_vllm(ports)


def count_html(html_dir):
    return len(list(Path(html_dir).glob("*.html")))


def main():
    ap = argparse.ArgumentParser(description="DesignCoder eval inference")
    ap.add_argument("--ckpt", required=True,
                    help="HF safetensors dir (config.json + model-*.safetensors)")
    ap.add_argument("--out", required=True,
                    help="output run dir; produces <out>/html/ and <out>/logs/")
    ap.add_argument("--spec", required=True, help="eval spec JSONL")
    ap.add_argument("--image-dir", required=True, help="reference image dir")
    ap.add_argument("--tp", type=int, default=8, help="tensor-parallel size per endpoint")
    ap.add_argument("--ngpu", type=int, default=8, help="total GPUs to use")
    # locked protocol knobs — exposed but defaulted to the benchmark values
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--infer-retry", type=int, default=5)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    out = Path(args.out).resolve()
    spec = Path(args.spec)

    # sanity
    if not INFER_ENGINE.exists():
        sys.exit(f"ERROR: inference engine not found: {INFER_ENGINE}")
    if not (ckpt / "config.json").exists():
        sys.exit(f"ERROR: {ckpt} has no config.json (need HF safetensors dir)")
    if not spec.exists():
        sys.exit(f"ERROR: spec not found: {spec}")
    if args.ngpu % args.tp != 0:
        sys.exit(f"ERROR: ngpu={args.ngpu} not divisible by tp={args.tp}")

    html_dir = out / "html"
    logs_dir = out / "logs"
    html_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    total = sum(1 for _ in open(spec))
    target = total * args.n
    n_endpoint = args.ngpu // args.tp
    infer_workers = n_endpoint * 48

    log("=" * 60)
    log(f"CKPT: {ckpt}")
    log(f"TP={args.tp} NGPU={args.ngpu} -> {n_endpoint} endpoint(s), workers={infer_workers}")
    log(f"SPEC: {spec} ({total} specs x n={args.n} = {target} rollouts)")
    log(f"OUT:  {out}")
    log("=" * 60)

    already = count_html(html_dir)
    if already >= target:
        log(f"already complete: {already}/{target} html — nothing to do")
        return

    ports = [8000 + e for e in range(n_endpoint)]
    log(f"killing leftover vLLM on ports {ports}")
    kill_stale_vllm(ports)
    wait_gpu_free(args.ngpu)

    procs, endpoints, ports = launch_vllm(
        ckpt, args.tp, args.ngpu, logs_dir, args.max_model_len, args.gpu_mem_util)
    log(f"endpoints: {endpoints}")

    try:
        if not wait_ready(ports):
            for e in range(n_endpoint):
                log(f"--- vllm_ep{e}.log tail ---")
                print((logs_dir / f"vllm_ep{e}.log").read_text()[-2000:])
            sys.exit("ERROR: vLLM did not become ready within timeout")

        for attempt in range(1, args.infer_retry + 1):
            log(f"inference attempt {attempt}/{args.infer_retry}")
            subprocess.run(
                [sys.executable, str(INFER_ENGINE),
                 "--spec-jsonl", str(spec), "--image-dir", str(args.image_dir),
                 "--endpoints", ",".join(endpoints),
                 "--mode", "sample", "--n", str(args.n),
                 "--temperature", str(args.temperature), "--top-p", str(args.top_p),
                 "--output-dir", str(html_dir), "--max-tokens", str(args.max_tokens),
                 "--workers", str(infer_workers), "--resume"],
                cwd=str(REPO), check=False)
            n = count_html(html_dir)
            log(f"attempt {attempt}: html={n}/{target}")
            if n >= target:
                break
            time.sleep(5)
    finally:
        log("killing vLLM (releasing GPUs)")
        kill_procs(procs, ports)

    n = count_html(html_dir)
    log("=" * 60)
    log(f"INFERENCE DONE: {n}/{target} html at {html_dir}")
    log("next: run_validation.py --run " + str(out))
    log("=" * 60)
    if n < target:
        sys.exit(1)


if __name__ == "__main__":
    main()
