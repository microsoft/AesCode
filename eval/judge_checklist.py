#!/usr/bin/env python3
"""judge_checklist.py — judge a render against its checklist, one call per category.

ONE CALL PER CATEGORY, not one call per page. A checklist runs to between
twenty-six and fifty questions, and asking all of them in a single call invites
skimming: the model has to hold every question in working memory while looking at
one image, and the answers degrade towards the back of the list. Each category is
therefore a separate call over the same image, so each call carries a dozen or so
questions and the judge is doing one kind of reasoning at a time.

The categories are read off the rubric rather than hardcoded here: each question
declares its `kind`, rubric_io maps the kind to a bucket and the bucket to a
category, and this file groups by whatever categories come back. A checklist that
uses a different set is judged correctly without touching anything here. Each
category supplies its own brief from CATEGORY_BRIEF; one with no brief falls back
to the generic one.

The whole-page kinds -- page_balance, page_density, delivery_gate -- are asked in
their own call rather than appended to the Style call. Buried at position 25 of 34
the judge answers them badly; asked on their own against the same renders it gets
them right and its answers line up with the DOM overflow check. Eleven questions
in a call is a different task from thirty-four.

The judge sees the generated page image and nothing else: no reference image, and
not the generation prompt, so it cannot be talked into an answer by the
instruction text it is grading against.

Per rollout and per category:
    score = #questions answered as pass_answer / #questions in that category
Missing, unrendered and unjudged rollouts score 0 and are never dropped.

    python3 eval/judge_checklist.py --label L \\
            --rubric-dir data/checklists \\
            --sids-file runs/_all.txt --conc 24 --n 3
"""
import argparse
import base64
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm_client
import rubric_io
from rubric_io import load_rubrics

ROOT = Path(os.environ.get("BENCH_ROOT", Path(__file__).resolve().parents[1]))
SPEC = Path(os.environ.get("SPEC", ROOT / "data/eval_spec.jsonl"))
# Order to ask the categories in, derived from the one place the taxonomy lives.
# A category a checklist carries that is not in this list is asked last; the list
# is a preference, never a filter.
CATEGORY_ORDER = [rubric_io.category_of(b) for b in rubric_io.BUCKETS]
PAGE_CALL = "page"

GENERIC_BRIEF = (
    "Each question asks whether one stated property of the page holds. Judge only "
    "what the question asks, against the image in front of you. Partly true is not "
    "true. Where you cannot find the named elements or cannot resolve the property "
    "at this resolution, answer 'unclear' rather than assuming the page is fine."
)

CATEGORY_BRIEF = {
    "completeness": "Each question asks whether a required element is PRESENT and "
                    "visible. Judge presence only; do not deduct for it looking "
                    "different from what you might expect. Presence means you "
                    "located the element and can say where it is, not that the page "
                    "has a region where it would belong. Where a question fixes a "
                    "count or a region, both must hold. An element that is present "
                    "but cut through by the canvas edge is content that was lost.",
    "relation": "Each question asks whether a stated spatial or structural "
                "relation between elements holds: side by side, stacked, in a "
                "row or grid, contained, ordered, clear of one another. Answer "
                "these to a HIGHER standard of proof. A pass requires that you "
                "locate BOTH elements named in the question and can see the "
                "relation between them; a page that merely looks orderly is not "
                "evidence. If a question asks whether one thing stays clear of "
                "another, check the actual gap: if their marks touch, cross or "
                "print over one another anywhere, that is a failure even when the "
                "rest of the page is fine. If a question asks whether something is "
                "contained, the whole element must be inside, not most of it. "
                "Partly true is not true. Where you cannot find one of the named "
                "elements, or cannot resolve the relation at this resolution, "
                "answer 'unclear' rather than assuming the page is fine.",
    PAGE_CALL: "Each question is about the page AS A WHOLE: whether anything "
                   "is cut off or collides, whether the content fills the canvas, "
                   "and whether the margins and gaps are balanced. Take each one "
                   "separately and look at the whole image before answering. These "
                   "questions name a threshold, such as a fraction of the page or a "
                   "ratio between margins; check that threshold against the whole "
                   "canvas rather than against the neighbouring blocks, and do not "
                   "pass a page because it looks tidy. Where you cannot establish "
                   "the measurement the question asks for, answer 'unclear'.",
    "style_strict": "Each question asks whether ONE stated visual property holds, "
                     "or whether a complex construct such as a timeline, chart, "
                     "flow or icon is properly realised rather than crudely faked. "
                     "Answer these to a HIGHER standard of proof than the other "
                     "categories. A pass requires that you can point at the specific "
                     "pixels that establish the property, and that you would give the "
                     "same answer looking again. Partly true is not true. A construct "
                     "that merely gestures at the form, such as boxes in a row "
                     "standing in for a timeline with no connecting spine, or an icon "
                     "that is an unrecognisable blob, does not pass. Where you cannot "
                     "find that specific evidence, answer 'unclear' rather than "
                     "guessing; 'unclear' is scored as a failure, which is the correct "
                     "outcome for a property you cannot actually verify. Do not credit "
                     "a page for effort, for looking busy, or for browser defaults "
                     "such as a heading already being larger than body text.",}

JUDGE_SYSTEM = """You are a strict but fair judge of a generated web page.

You see ONE GENERATED PAGE IMAGE and a numbered list of yes/no questions about it.
There is no reference or target image; you are never asked whether the page
resembles anything.

{brief}

Rules:
- Answer ONLY from what is visible in the image.
- The image is a FIXED-SIZE CANVAS, like a slide or a poster. It is not a scrolling
  page and it was not extended to fit the content: anything the page laid out beyond
  this canvas is gone and you are seeing all there is. So an element that runs into
  the canvas edge and stops mid-way -- a severed word, a card with no bottom, a row
  of tiles whose last tile is sliced -- is content that was genuinely lost, and it is
  a real defect. Judge the composition as a design that had to fit this canvas.
- Full-bleed design is legitimate, not a defect: background bands, headers, footers
  and images may deliberately extend to the page edge. The test is whether something
  is CUT THROUGH, not whether it touches the edge.
- Read each question's polarity carefully. Some ask whether a DEFECT is present or
  whether a requirement was broken; for those, "yes" means the problem IS there.
- Grey placeholder boxes, often labelled with their pixel size, are REQUIRED by the
  brief: the models are told to stand images in with https://placehold.co rather
  than draw them. Treat a placeholder as the image being present and correctly
  placed. Never call it missing, unfinished, low-contrast or a design defect.
- Questions quote exact text. Match it by its distinctive words; differences in
  case or punctuation do not matter, but missing or different content does.
- A page carrying more content is not thereby worse, and a page carrying less is
  not thereby better. Do not report a defect unless you can point to the specific
  elements involved; adjacent or tightly spaced is not the same as overlapping.
- Do not be lenient and do not be charitable. If a property is only partly true,
  answer what the visible evidence shows.

How to decide each answer. This is a procedure, not advice, and it is the same
for every question and every category:

1. Before you answer, find the evidence in the image and put it in `where`: the
   region you looked at and what you saw there. `where` is REQUIRED on EVERY
   answer. "yes" and "no" are equally claims about the image and both must be
   shown. Neither one is the safe default.
2. If you cannot write that `where` from the image, you do not have the evidence,
   and "unclear" is your only remaining answer. You may not fall back on "yes",
   and you may not fall back on "no" either.
3. Answer "unclear" whenever the evidence is missing, ambiguous, too small to
   resolve, or you would answer differently on a second look. Say in `where` what
   you looked for and could not resolve.

Both answers need evidence because the questions do not share a polarity. Some
state a property the page should have, where "yes" is the good outcome. Others
ask whether a defect is present, where "no" is the good outcome. You are not told
which kind you are looking at, and you must not try to infer it: decide only what
the image shows.

What does NOT count as evidence, in either direction:
- Not seeing something. Failing to find a defect does not establish that a stated
  property holds, and failing to find an element does not establish that it is
  absent. A region you did not resolve is "unclear", not "no".
- The page looking clean, orderly, professional or complete overall.
- What a region probably contains, judged from the rest of the page.
- The brief, the question's own wording, or what a page like this usually does.
- A guess you would not repeat if you looked again.

"unclear" is scored as a FAILURE. That is intended. A question you cannot resolve
should not earn a point, whichever answer would have earned it. Do not avoid
"unclear" to seem decisive, and do not treat it as a last resort: on a question
whose evidence you cannot locate, it is the correct answer, not a failure of nerve.

Output STRICT JSON, no code fences. `where` must be filled on every judgment:
{{"judgments": [{{"id":"C_title","answer":"yes","where":"title bar, top-left, reads 'Quarterly Review'"}},
               {{"id":"L_overlap","answer":"no","where":"checked the gap between the chart and the legend at mid-right; roughly 20px of clear space, no marks crossing"}},
               {{"id":"L_grid","answer":"unclear","where":"looked for the fourth card in the lower row; that area is blank and I cannot tell if it was dropped or intended"}}]}}
"""


def parse_loose(t):
    s = t.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        if s.startswith("json"):
            s = s.split("\n", 1)[1] if "\n" in s else s
    return json.loads(s)




def find_png(art_dir: Path, sid: str, k: int):
    paths = [p for p in glob.glob(str(art_dir / f"{sid}_k{k}/{sid}/render_playwright_*.png"))
             if "overflow" not in Path(p).name]
    return Path(paths[0]) if paths else None


def judge_category(pool, model, img_b64, category, questions, timeout=240,
                   detail="auto"):
    qs = "\n".join(f"  {q['id']}. {q['question']}" for q in questions)
    user_text = (
        f"Judge this GENERATED page against {len(questions)} yes/no questions "
        f"about {category}.\n\n{qs}\n\n"
        "Answer yes / no / unclear for each, based ONLY on what is visible.\n"
        "Fill `where` on every judgment, whatever the answer: name the region you "
        "looked at and what you saw there. An answer you cannot locate in the "
        "image is 'unclear'.\n"
        'Output strict JSON: {"judgments": [{"id":"...","answer":"yes",'
        '"where":"region you looked at and what you saw"}, ...]}'
    )
    messages = [
        {"role": "system",
         "content": JUDGE_SYSTEM.format(
             brief=CATEGORY_BRIEF.get(category, GENERIC_BRIEF))},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}",
                           "detail": detail}},
            {"type": "text", "text": user_text}]},
    ]
    return llm_client.chat(
        pool, model, messages, max_tokens=8000, timeout=timeout, tries=1,
        reasoning_effort="low" if llm_client.wants_reasoning_effort(model) else None)


def categories_of(questions):
    """The categories this rubric actually uses, in CATEGORY_ORDER where known."""
    present = []
    for q in questions:
        c = q.get("category")
        if c and c not in present:
            present.append(c)
    known = [c for c in CATEGORY_ORDER if c in present]
    return known + [c for c in present if c not in known]


def is_page_level(q):
    return q.get("kind") in rubric_io.PAGE_LEVEL_KINDS


def call_groups(questions, split_page_level=True):
    """Which questions go in which call: one per category, page-level last."""
    split = bool(split_page_level)
    held = [q for q in questions if split and is_page_level(q)]
    rest = [q for q in questions if not (split and is_page_level(q))]
    out = []
    for cat in categories_of(rest):
        qs = [q for q in rest if q.get("category") == cat]
        if qs:
            out.append((cat, qs))
    if held:
        out.append((PAGE_CALL, held))
    return out


def judge_rollout(pool, model, sid, k, questions, art_dir, detail="auto",
                  split_page_level=True):
    img = find_png(art_dir, sid, k)
    if not img:
        return {"sid": sid, "k": k, "ok": False, "err": "no_png"}
    img_b64 = base64.b64encode(img.read_bytes()).decode()
    answers, where, per_cat = {}, {}, {}
    for cat, qs in call_groups(questions, split_page_level):
        got = None
        delay = 10
        last = ""
        for attempt in range(4):
            try:
                txt = judge_category(pool, model, img_b64, cat, qs, detail=detail)
                got = parse_loose(txt).get("judgments", [])
                break
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:160]}"
                if any(c in str(e) for c in ("400 ", "401 ", "403 ", "404 ", "BadRequest")):
                    break
                time.sleep(delay)
                delay = min(delay * 2, 120)
        if got is None:
            return {"sid": sid, "k": k, "ok": False, "err": f"{cat}: {last}"}
        for it in got:
            if not it.get("id"):
                continue
            answers[it["id"]] = it.get("answer")
            if it.get("where"):
                where[it["id"]] = it["where"]
        per_cat[cat] = {"asked": len(qs),
                        "answered": sum(1 for q in qs if q["id"] in answers)}
    missing = [q["id"] for q in questions if q["id"] not in answers]
    return {"sid": sid, "k": k, "ok": True, "answers": answers, "where": where,
            "per_category": per_cat, "missing": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--rubric-dir", default=str(
        os.environ.get("RUBRIC_DIR") or ROOT / "data/checklists"))
    ap.add_argument("--art-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--sids", default="")
    ap.add_argument("--sids-file", default="")
    ap.add_argument("--conc", type=int, default=24)
    ap.add_argument("--n", type=int, default=3, help="rollouts per sample")
    ap.add_argument("--judge-model", default="",
                    help="judge model name (default: $JUDGE_MODEL). Pin this for "
                         "a whole table: a provider alias repointed mid-run "
                         "changes the judge between rows.")
    ap.add_argument("--detail", default="auto", choices=["auto", "low", "high"],
                    help="vision detail for the page image. A downscaled screenshot "
                         "loses more from a dense page than from a sparse one, which "
                         "biases every question that needs small text.")
    ap.add_argument("--no-split-page-level", action="store_true",
                    help="append the whole-page questions to their category call "
                         "instead of asking them in their own call.")
    a = ap.parse_args()

    model = llm_client.judge_model(a.judge_model)
    pool = llm_client.ClientPool()
    print(f"[judge] model={model} endpoints={len(pool)}", flush=True)

    art_dir = Path(a.art_dir) if a.art_dir else \
        ROOT / f"runs/{a.label}_final300/results/artifacts"
    out_dir = Path(a.out) if a.out else ROOT / f"runs/{a.label}_final300/checklist"
    rd = Path(a.rubric_dir)

    if a.sids_file:
        sids = [s for s in Path(a.sids_file).read_text().split() if s]
    elif a.sids:
        sids = a.sids.split(",")
    else:
        sids = [json.loads(l)["id"] for l in open(SPEC)]

    rubrics = load_rubrics(rd, sids, warn=lambda m: print(f"warn: {m}", flush=True))
    nq = [len(v) for v in rubrics.values()]
    if not rubrics:
        raise SystemExit(
            f"none of the {len(sids)} sids has a rubric in {rd}. Judging nothing "
            f"and reporting success is how a run ends up silently scoring zero.")
    cat_counts = {}
    for qs in rubrics.values():
        for q in qs:
            cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    cats = ", ".join(f"{c}={cat_counts[c]}"
                     for c in categories_of(q for qs in rubrics.values() for q in qs))
    print(f"sids={len(sids)} with_rubric={len(rubrics)} "
          f"questions/sid mean={sum(nq)/max(1,len(nq)):.1f} "
          f"range={min(nq)}-{max(nq)} art_dir={art_dir}", flush=True)
    print(f"categories: {cats}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / "rubrics_used.json"
    if not snap.exists():
        snap.write_text(json.dumps(rubrics, indent=2, ensure_ascii=False))

    out_jsonl = out_dir / "judgments.jsonl"
    done = set()
    if out_jsonl.exists():
        for ln in out_jsonl.read_text().strip().split("\n"):
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                done.add((r["sid"], r["k"]))

    tasks = [(s, k) for s in sids if s in rubrics for k in range(a.n)
             if (s, k) not in done and find_png(art_dir, s, k)]
    split_page = not a.no_split_page_level
    ncalls = len(call_groups(next(iter(rubrics.values())), split_page))
    print(f"[{a.label}] done={len(done)} pending={len(tasks)} "
          f"(x{ncalls} calls each)", flush=True)
    if tasks:
        t0 = time.time()
        ok, fail = len(done), 0
        with ThreadPoolExecutor(max_workers=a.conc) as ex:
            futs = {ex.submit(judge_rollout, pool, model, s, k, rubrics[s],
                              art_dir, a.detail, split_page): (s, k)
                    for s, k in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                with open(out_jsonl, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                ok, fail = (ok + 1, fail) if r.get("ok") else (ok, fail + 1)
                dc = ok + fail - len(done)
                if dc % 25 == 0 or dc == len(tasks):
                    el = time.time() - t0
                    rate = dc / el * 60 if el else 0
                    print(f"  [{a.label} {dc}/{len(tasks)}] ok={ok} fail={fail} "
                          f"rate={rate:.0f}/min ETA={(len(tasks)-dc)/rate if rate else 0:.1f}min",
                          flush=True)
    print(f"DONE checklist judge {a.label} -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
