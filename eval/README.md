# Evaluation

Standard, self-serve evaluation of an image→HTML checkpoint. Two stages, cleanly
separated:

```
run_inference.py    ckpt ──vLLM──► html/{sid}_k{0..N-1}.html      (N·|specs| rollouts)
run_validation.py   html ──────►  6 deterministic dims [+ Design] ──► pa@1 / pa@N table
run_eval.sh         both, back-to-back on one node
```

The score table is **printed at the end of `run_validation.py`** — there is no
separate "build table" step.

## Dimensions

Deterministic (6, no LLM, from the repo's validation engine):
`Execution`, `Text`, `Boundary`, `TableChart`, `SemanticLayout`, `Whitespace`.

Design (1, optional, rubric judge): each rollout is rendered to PNG and a vision
LLM answers the spec's yes/no rubric. Enabled only when you supply an
OpenAI-compatible endpoint and a rubric dir; otherwise the 6 deterministic dims
are reported alone.

Reported columns:
```
Exec* | Text | Bound | TblCh | Rule | SemL | WSpc | Design | Visual | Overall
```
- `Exec*` is shown but **excluded** from Rule/Visual/Overall (it saturates ~1.0).
- `Rule`    = per-sample mean(Text, Boundary [, TableChart])
- `Visual`  = per-sample mean(SemanticLayout, Whitespace [, Design])
- `Overall` = per-sample mean(Text, Boundary, SemLayout, Whitespace [, TableChart][, Design])

## Scoring protocol (locked)

- Denominator locked to **(accountable sids) × N**. A rollout the model never
  produced, or one that failed to render, scores **0** — never dropped. This
  stops sampling/render gaps from inflating averages.
- `pa@1` = mean over N rollouts of the per-rollout score, then mean over sids.
- `pa@N` = max over N rollouts, then mean over sids. (N is `--n`.)
- TableChart stays applicable-only (denominator = sids whose GT has a table).
- A model covering ≥96.7% of sids is scored over the full set; fewer → scored
  over its own covered set (a genuine partial eval).

## Quick start

One node, 8 GPUs, N=8, deterministic dims only:
```bash
bash run_eval.sh /path/to/checkpoint ./runs/my_model \
    <data_root>/eval.jsonl <data_root>/images 8 8 8
```

With Design (any OpenAI-compatible vision endpoint + a rubric dir):
```bash
export JUDGE_BASE_URL=https://api.openai.com/v1
export JUDGE_API_KEY=sk-...
export JUDGE_MODEL=gpt-4o
export RUBRIC_DIR=<data_root>/rubrics
bash run_eval.sh /path/to/checkpoint ./runs/my_model \
    <data_root>/eval.jsonl <data_root>/images 8 8 8
```

4 GPUs, N=16 rollouts:
```bash
bash run_eval.sh /path/to/checkpoint ./runs/my_model \
    <data_root>/eval.jsonl <data_root>/images 16 4 4
```

## Stages individually

Inference only:
```bash
python run_inference.py --ckpt /path/to/checkpoint --out ./runs/my_model \
    --spec <data_root>/eval.jsonl --image-dir <data_root>/images \
    --n 8 --tp 8 --ngpu 8
```

Validation + scoring only (after inference, or to re-score existing html):
```bash
python run_validation.py --run ./runs/my_model --n 8 \
    --spec <data_root>/eval.jsonl --image-dir <data_root>/images \
    [--rubric-dir <data_root>/rubrics --judge-base-url ... --judge-api-key ... --judge-model gpt-4o] \
    [--skip-deterministic]   # reuse existing validation_report.jsonl
```

## Decoding protocol (defaults)

`n=8  temperature=0.8  top_p=0.95  max_tokens=12000  max_model_len=24576`

These are the benchmark defaults. Keep them **identical** across checkpoints of
the same family — a benchmark is only meaningful when every model sees the same
sampling. All are overridable via flags on `run_inference.py` if you are
deliberately running a different protocol.

## Requirements

- `vllm` (serves the checkpoint; HF safetensors dir with `config.json`)
- `openai` python package (only for the Design judge)
- Playwright render stack + the repo's validation engine
  (`eval/eval_vllm_validate.py`) for the deterministic dims
- Your eval data (not shipped — see the [main README](../README.md#data--format)
  for the JSONL schema):
  - specs: an eval JSONL (`id` + `generation_prompt` + `verifiable_requirements`)
  - reference images: `<image_dir>/<id>.png`
  - Design rubrics (optional): `<rubric_dir>/<id>.json` with a `rubric` list

## Outputs

```
runs/my_model/
├── html/                              generated HTML (N × |specs|)
├── logs/                              vLLM + infer logs
└── results/
    ├── validation_report.jsonl        per-rollout deterministic scores
    ├── artifacts/                     rendered PNGs (for Design judge)
    └── design_judgments.jsonl         rubric judge answers (if Design enabled)
```

Both stages are resumable — rerun the same command to continue after an
interruption.
