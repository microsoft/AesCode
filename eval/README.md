# Reference-free evaluation benchmark

A generated page is scored without ever comparing it to a reference image. Two
independent sources of evidence are combined:

* **Rule dimensions** — `Text`, `Bound`, `Chart`. Deterministic validators run
  over the rendered DOM and screenshot: is every required string present, does
  anything overflow the canvas, do the charts and tables carry the stated data.
* **Visual dimensions** — `Content`, `Layout`, `Style`. A per-sample checklist of
  yes/no questions, answered by a VLM judge that sees the rendered page image and
  nothing else: no reference image, and not the generation prompt.

```
Rule    = mean(Text, Bound, Chart)
Visual  = mean(Content, Layout, Style)
Overall = mean(Rule, Visual)
```

Execution is not a column. It is a gate: a page that does not render scores zero
on everything downstream, so reporting it separately would count one failure
twice.

No evaluation data ships with this repository. The input formats are specified
below; point `SPEC` and `RUBRIC_DIR` at your own.

## Pipeline

Four stages, one script each. Source the environment first.

```bash
source eval/run/env.sh

bash eval/run/1_infer.sh /abs/path/to/ckpt my_label 4 4   # N rollouts per sample
bash eval/run/2_validate.sh my_label                      # render + rule validators
bash eval/run/3_judge.sh my_label                         # checklist judge
bash eval/run/4_table.sh                                  # markdown table
```

`N = 3` rollouts per sample at `temperature 0.8`, `top_p 0.95`, `max_tokens
12000`. Every stage must use the same `N`; the stage scripts pass it explicitly
for that reason.

To evaluate a hosted API rather than local weights, skip stage 1 and call
`eval/eval_vllm_infer.py` directly with `--endpoints` and `--model`, keeping
those sampling knobs. To evaluate without a reference image — the setting for the
reference-free arms — set `CLEAN_NOIMAGE=1` for stage 1.

### Model access

Everything that calls a large model speaks the OpenAI chat-completions schema.

```bash
export OPENAI_API_KEY=...                         # local servers ignore this
export OPENAI_BASE_URL=https://api.openai.com/v1  # comma-separated list allowed
export JUDGE_MODEL=gpt-5.5                        # or pass --judge-model
```

`OPENAI_BASE_URL` may list several base URLs serving the *same* model; calls are
spread round-robin over them. Pin the judge model for a whole table: a provider
alias repointed mid-run changes the judge between rows, and the rows then are not
comparable.

## Input formats

### Spec — `$SPEC`, one JSON object per line

```json
{
  "id": "<sid>",
  "generation_prompt": "Create a wide landscape infographic in HTML/CSS titled \"<TITLE>\" ... render the share breakdown as an ECharts pie chart with the literal values ... include a real HTML table of ...",
  "verifiable_requirements": [
    {"id": "<req_id>", "category": "text_exact",
     "expected": {"text": "<TITLE>"}},
    {"id": "<req_id>", "category": "chart_data",
     "target": {"chart_type": "echarts", "chart_subtype": "pie"},
     "expected": {"data": [<number>, <number>, "..."]}},
    {"id": "<req_id>", "category": "table_data",
     "expected": {"data": [["<header>", "..."], ["<cell>", "..."]]}}
  ],
  "reference_metadata": {
    "reference_image_wh": "<w>x<h>",
    "reference_aspect_ratio": 1.7778,
    "reference_orientation": "landscape"
  }
}
```

* `id` is the sample id (`sid`) used by every later stage and by every filename.
* `generation_prompt` is the brief given to the model. The checklist is written
  against this same text; a question about anything the brief does not state is
  asking a page to guess.
* `verifiable_requirements[].category` selects the deterministic validator:
  `text_exact` (with `expected.text`) feeds `Text`; `chart_data` and `table_data`
  (with `expected.data`, and `target` naming the chart library and subtype) feed
  `Chart`. `target` is optional. A sample with no chart or table requirement is
  absent from the `Chart` denominator rather than scoring 0 there.
* A requirement is only verifiable if the brief asked for it. Every `text_exact`
  string must appear verbatim inside `generation_prompt`, and a brief that
  carries a `chart_data` requirement must itself instruct the model to render
  that chart as an ECharts chart with literal data arrays — otherwise the
  validator is testing something the model was never told to produce.
* `reference_metadata.reference_image_wh` is a `"<w>x<h>"` string and fixes the
  render canvas. The page is a fixed-size canvas, like a poster: anything laid
  out past the edge is lost, and the judge is told to treat it as lost.

### Checklist — `$RUBRIC_DIR/<sid>.json`, one file per sample

```json
{
  "sid": "<sid>",
  "questions": [
    {"id": "q1",
     "kind": "located_presence",
     "bucket": "Content",
     "category": "completeness",
     "binds_to": "node:<node_id>",
     "question": "<Is the element the brief names present, in the region named here?>",
     "pass_answer": "yes"},

    {"id": "q2",
     "kind": "collision",
     "bucket": "Layout",
     "category": "relation",
     "binds_to": "node:<node_id>",
     "question": "<Does this element overlap or occlude a neighbour?>",
     "pass_answer": "no"},

    {"id": "q3",
     "kind": "page_density",
     "bucket": "Style",
     "category": "style_strict",
     "question": "<Does the page clear the stated density threshold, e.g. no band of more than one quarter of the canvas height left empty?>",
     "pass_answer": "yes"}
  ]
}
```

Required on every question: `id` (unique within the file; the key used in
`judgments.jsonl`), `kind`, `bucket`, `category`, `question`, `pass_answer`.
`binds_to` is optional — a reference of the form `node:<id>` or `edge:<id>` into
the scene graph the checklist was authored from. A question missing a required
key is dropped with a warning rather than guessed at: a question with no
`pass_answer` has no defined score.

`kind` is the primitive. It names the one way of looking at the page that the
question applies, and it decides the other two labels: `bucket` is a pure
function of `kind`, and `category` is a pure function of `bucket`.

| bucket    | category       | kinds |
| --------- | -------------- | ----- |
| `Content` | `completeness` | `located_presence`, `element_count`, `element_integrity` |
| `Layout`  | `relation`     | `collision`, `containment`, `structure`, `page_balance`, `connective`, `relation`, `subgraph` |
| `Style`   | `style_strict` | `concrete_attribute`, `element_quality`, `icon_quality`, `page_density`, `delivery_gate`, `design_labelled` |

That table lives in exactly one place in the code, `eval/rubric_io.py`, and both
the judge and the table read it from there. `bucket` and `category` are *derived
from `kind` on load*; a stored value that disagrees is overwritten and warned
about, so a checklist cannot be judged under one label and scored under another.
Keys outside the schema are dropped on load, so no stage can come to depend on
one. `category` decides which judge call a question is asked in; `bucket` decides
which table column it scores.

`page_balance`, `page_density` and `delivery_gate` are whole-page kinds: they are
about the canvas rather than one element, so they carry no `binds_to` and are
asked in their own judge call.

`pass_answer` is `yes` or `no`; a question is passed iff the judge's answer equals
it. `unclear` is therefore a failure, which is intended — a property the judge
could not verify from the image has not been demonstrated. Both polarities occur,
which is why the field is explicit: some questions state a property that should
hold, others ask whether a defect is present.

Checklists run 26 to 50 questions. A checklist directory may also hold
bookkeeping files such as `_manifest.json`; anything whose `questions` is not a
list is not a checklist and is skipped.

`data_prep/build_graph_rubric.py` builds checklists of this shape: it extracts a scene
graph from the brief, authors questions against that graph one call per bucket
under per-kind quotas, then lints the result against the question contract.

## Authoring a checklist

`data_prep/build_graph_rubric.py` turns a spec into a checklist in three stages.

```bash
python3 data_prep/build_graph_rubric.py \
  --input data/eval_spec.jsonl \
  --output my_rubrics \
  --save-graphs my_rubrics/_graphs
```

1. **Scene graph.** One call over the brief and `verifiable_requirements`, with
   no image: nodes are the elements the page must contain, edges the spatial and
   structural relations between them.
2. **Authoring.** One call per bucket, against that graph. Every question carries
   a `kind` and, unless the kind is a whole-page one, a `binds_to` into the graph.
   Binding to the graph rather than to the prose is what keeps a question about
   something the page was actually asked for.
3. **Linter.** Every question is checked against the contract below, and the
   checklist against its quotas. Flags are written to `<out>/_lint.json`; nothing
   is silently repaired.

Question contract — each question must be answerable yes/no from a single
rendering with no reference image and no access to the markup, name the element
it is about, state any threshold it depends on, carry an explicit `pass_answer`,
and declare its `kind`. A question is rejected if it asks about the content of a
placeholder image slot, prescribes an implementation rather than a visible
outcome, restates a browser default, or bundles several claims that could not
fail independently. `delivery_gate` is exempt from the last of those: it is a
disqualifier list by design.

Quotas keep a page from being scored almost entirely on whether its strings are
present. `page_balance` is exactly one question per page and `delivery_gate`
exactly five — colour, typography, hierarchy, finish, content treatment. The rest
scale with the size of the graph, targeting 26–50 questions split about
18/43/39 across Content/Layout/Style. A kind the graph has no material for —
`subgraph` on a page with no nested structure, `icon_quality` on a page with no
icons — is dropped from the plan rather than filled with invented questions.

### Run directory layout

```
runs/
  _all.txt                               # one sample id per line; the denominator
  <label>_final300/
    gen_meta.json                        # prompt mode and sampling knobs
    html/<sid>_k<k>.html                 # stage 1, k in 0..N-1
    logs/
    results/
      validation_report.jsonl            # stage 2, one row per rollout
      artifacts/<sid>_k<k>/<sid>/render_playwright_*.png
    block_final300/
      judgments.jsonl                    # stage 3, one row per rollout
      rubrics_used.json                  # snapshot of the checklists as judged
      .shot_count_at_judge               # "<rendered> / <expected>"
      .done                              # written only on a clean judge exit
  main_table.md                          # stage 4
```

`_all.txt` is derived from the spec by `eval/run/env.sh` if it does not exist, so
every stage shares one denominator.

`validation_report.jsonl` rows carry at least `sample_id`, `html_path` (whose
`_k<k>.html` suffix gives the rollout index) and `per_validator`, a map from
validator name to score. `TextValidator`, `BoundaryValidator` and
`TableChartValidator` become `Text`, `Bound` and `Chart`; a validator that has
nothing to say about a sample reports nothing for it rather than reporting zero.

`judgments.jsonl` rows are:

```json
{"sid": "<sid>", "k": 0, "ok": true,
 "answers": {"q1": "yes", "q2": "no", "q3": "unclear"},
 "where": {"q1": "left third of the page"}}
```

`where` records the evidence the judge claims for each answer and is advisory.
A rollout with no screenshot, no judgment, or `ok: false` scores 0 and is never
dropped from the denominator. Dropping it would reward a model whose worst pages
are the ones that crash the renderer.

## Layout

```
eval/
  build_table.py          stage 4: the aggregation, and the only place it lives
  judge_checklist.py      stage 3: one VLM call per category over the PNG
  rubric_io.py            the kind taxonomy and the checklist loader, shared by
                          the builder, the judge and the table
  llm_client.py           OpenAI-compatible client pool
  render_parity.py        find and retry rollouts this harness lost
  eval_vllm_infer.py      stage 1
  eval_vllm_validate.py   stage 2
  run/                    env.sh + the four stage drivers
data_prep/
  build_graph_rubric.py   checklist builder: scene graph -> questions -> linter
  build_vgr.py            its training-side sibling, same shape
tests/                    the aggregation and the checklist contract, pinned
```

The stages also use two packages that live at the repo root and are shared with
training: `verifier/` (the rule validators and the Playwright render harness) and
`inference/prompts.py` (the brief, so a page is authored and checklisted against
the same text). `eval/run/env.sh` puts the repo root on `PYTHONPATH`.

`python3 tests/test_aggregation.py` and `python3 tests/test_checklist_builder.py`
build their own fixtures, so they need neither model access nor evaluation data.

## Requirements

Python 3.10+ and the packages in the top-level `requirements.txt`. Two pieces are
installed separately and are covered in the Installation section of the main
README: Chromium with its system libraries (`playwright install --with-deps
chromium`) and `paddleocr`. Stage 1 additionally needs `vllm` when serving local
weights.

`DISABLED_VALIDATORS` in `run/env.sh` turns off the design validator, which is
reference-dependent and contributes no column here. The remaining validators run
on the rendering alone, so this stage does not need a reference image and does
not call OCR against one.

ECharts is vendored at `verifier/artifacts/assets/echarts.min.js` and served
locally, so pages that chart render the same way on a node with no network. Every
other external request is either served from the CDN cache or blocked.
