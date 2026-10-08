#!/usr/bin/env python3
"""Build the evaluation checklist for each sample.

Implements the two-call construction described in the paper (Appendix E.1), the
reference-free counterpart to the training-side rubric builder:

    Call 1  scene graph extraction
            Reads the generation prompt and its verifiable requirements and
            returns a graph of nodes and edges: the elements the brief asks for,
            their attributes, and the relations the brief asserts between them.

    Call 2  question authoring from the graph, ONE CALL PER BUCKET
            Receives that graph and writes yes/no questions for one dimension at
            a time, each question carrying the kind it belongs to and a binding
            into the graph. Content, Layout and Style ask for different kinds of
            looking, and a single call that has to produce all three spends its
            budget on whichever is easiest to write.

No reference image is read at any point. The graph is what the BRIEF asks for,
not what some particular rendering happens to contain, so a question can only
ever be about something the brief committed to.

Every question declares a `kind`; the kind determines the bucket and the bucket
determines the category (see rubric_io, which owns that mapping). Per-kind quotas
keep a page from being scored almost entirely on whether its strings are present:
without them an author writes located_presence questions all day, because they
are the easiest to write and the easiest to pass.

A programmatic linter then checks every question against the paper's checklist
question contract: answerable from one rendering, names its element, states its
threshold, declares its polarity and kind, and makes exactly one claim.

Usage:
    python3 data_prep/build_graph_rubric.py \\
        --input data/eval_spec.jsonl \\
        --output data/checklists \\
        --concurrency 8

Output is one file per sample in the released checklist schema:

    {"sid": ..., "questions": [{"id", "bucket", "category", "kind",
                                "binds_to", "question", "pass_answer"}, ...]}

plus, with --save-graphs, the stage-1 graph it was authored from.

The endpoint is any OpenAI-compatible chat completions API:

    export OPENAI_BASE_URL=https://api.openai.com/v1
    export OPENAI_API_KEY=...
    export VLM_MODEL=gpt-5.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# The kind -> bucket -> category taxonomy has one home, next to the stages that
# consume it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import rubric_io  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Quotas
# --------------------------------------------------------------------------
# Checklist length tracks graph size: a page with more elements has more that can
# go wrong, and a fixed length would either starve a dense page or pad a sparse
# one with questions it cannot fail. The constant is set so a typical graph of
# around twenty elements lands near the corpus mean of 35.8 questions.
QUESTIONS_PER_ELEMENT = 1.6
MIN_QUESTIONS, MAX_QUESTIONS = 26, 50

# Corpus bucket mix: 1,925 Content / 4,565 Layout / 4,237 Style.
BUCKET_MIX = {"Content": 0.18, "Layout": 0.43, "Style": 0.39}

# Kinds whose count per page is fixed rather than scaled.
#   page_balance  one per page by construction: there is one canvas to balance.
#   delivery_gate five per page, one each for colour, typography, hierarchy,
#                 finish and content treatment. These are the one sanctioned
#                 non-atomic form -- each is a disqualifier list bound to the
#                 canvas -- so they are exempted from the atomicity rule and
#                 their count is pinned rather than left to the author.
FIXED_QUOTAS = {"page_balance": 1, "delivery_gate": 5}

DELIVERY_GATE_ASPECTS = ("colour", "typography", "hierarchy", "finish",
                         "content treatment")

# Relative weight of each kind within its bucket, before graph support is applied.
KIND_WEIGHT = {
    "located_presence": 3, "element_count": 2, "element_integrity": 5,
    "collision": 4, "containment": 2, "structure": 5, "connective": 2,
    "relation": 3, "subgraph": 1,
    "concrete_attribute": 4, "element_quality": 3, "icon_quality": 2,
    "page_density": 2,
}

# `design_labelled` exists in the taxonomy because released checklists carry it,
# but it is a provenance label rather than a way of looking at a page, so nothing
# is authored under it. See the note at the end of this docstring block in the
# README: the kinds authored here are the four Style kinds the paper names, plus
# the sanctioned delivery gate.
NOT_AUTHORED = ("design_labelled",)

NODE_TYPES = ("region", "card", "panel", "list", "group", "text", "chart",
              "table", "icon", "image", "button", "input", "divider",
              "connector", "badge", "timeline", "banner")
EDGE_TYPES = ("contains", "left_of", "right_of", "above", "below",
              "aligned_with", "evenly_spaced", "connects", "timeline_step",
              "parent_of")
CONNECTIVE_EDGES = ("connects", "timeline_step", "parent_of")
SPATIAL_EDGES = ("left_of", "right_of", "above", "below", "aligned_with",
                 "evenly_spaced")
CONSTRUCT_NODES = ("chart", "table", "timeline", "connector", "icon", "image")


def _edge_types(graph: dict) -> list[str]:
    return [str(e.get("type", "")).split("{")[0].strip()
            for e in graph.get("edges", [])]


def _node_types(graph: dict) -> list[str]:
    return [str(n.get("type", "")).split("{")[0].strip()
            for n in graph.get("nodes", [])]


def supported_kinds(graph: dict) -> set[str]:
    """Kinds this graph can actually support a question about.

    Authoring a kind the graph has no material for produces a question invented
    to fill a quota, which is worse than a shorter checklist: it is unanswerable
    or trivially true, and either way it is noise in the column it lands in.
    """
    ntypes, etypes = _node_types(graph), _edge_types(graph)
    nodes, edges = graph.get("nodes", []), graph.get("edges", [])
    has_group = any(
        t in ("group", "list") or int((n.get("attrs") or {}).get("count", 0) or 0) >= 2
        for n, t in zip(nodes, ntypes)
    )
    ok = {"located_presence", "element_integrity", "collision", "structure",
          "concrete_attribute", "page_density", "page_balance", "delivery_gate"}
    if has_group:
        ok.add("element_count")
    if "contains" in etypes:
        ok.add("containment")
    if any(t in CONNECTIVE_EDGES for t in etypes):
        ok.add("connective")
    if any(t in SPATIAL_EDGES for t in etypes):
        ok.add("relation")
    # Do not force a subgraph question on a flat page; a fabricated one is worse
    # than none, which is the same call the training-side builder makes.
    if len(nodes) >= 6 and len(edges) >= 3:
        ok.add("subgraph")
    if "icon" in ntypes:
        ok.add("icon_quality")
    if any(t in CONSTRUCT_NODES for t in ntypes):
        ok.add("element_quality")
    return ok - set(NOT_AUTHORED)


def _apportion(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment, so the parts sum to exactly `total`."""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    pool = sum(weights.values())
    exact = {k: total * w / pool for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    short = total - sum(out.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[:short]:
        out[k] += 1
    return out


def plan_quotas(graph: dict, target: int | None = None) -> dict[str, dict[str, int]]:
    """{bucket: {kind: n}} for this graph."""
    size = len(graph.get("nodes", [])) + len(graph.get("edges", []))
    total = target or round(QUESTIONS_PER_ELEMENT * size)
    total = max(MIN_QUESTIONS, min(MAX_QUESTIONS, int(total)))
    ok = supported_kinds(graph)

    per_bucket = _apportion(total, {b: int(round(1000 * s))
                                    for b, s in BUCKET_MIX.items()})
    plan: dict[str, dict[str, int]] = {}
    for bucket, n in per_bucket.items():
        kinds = [k for k in rubric_io.KINDS_BY_BUCKET[bucket] if k in ok]
        fixed = {k: FIXED_QUOTAS[k] for k in kinds if k in FIXED_QUOTAS}
        scaled_kinds = [k for k in kinds if k not in FIXED_QUOTAS]
        remaining = max(0, n - sum(fixed.values()))
        scaled = _apportion(remaining, {k: KIND_WEIGHT[k] for k in scaled_kinds})
        plan[bucket] = {k: v for k, v in {**fixed, **scaled}.items() if v > 0}
    return plan


# --------------------------------------------------------------------------
# Call 1: scene graph extraction
# --------------------------------------------------------------------------

_GRAPH_SYSTEM = """You are a scene-graph extractor for page designs. Read the brief a page is to be generated from, together with the data and strings the page is required to carry, and describe the page THE BRIEF ASKS FOR as a graph. Do not write questions; a later step does that.

R1 BRIEF ONLY: you are not shown a rendering, and you must not invent one. Record only elements and relations the brief states or unambiguously implies. If the brief does not say where something sits, do not place it. Everything the brief leaves open is the designer's freedom and must not reach the graph.
R2 BOUNDARIES: a node is a region a reader could separate from its surroundings by an edge, a fill, or an unambiguous gap. Do not create nodes for internal padding, and do not merge elements the brief treats as separate.
R3 NO TEXT CONTENT: do not transcribe strings into the graph. A text node records its role, its place in the page and its emphasis. Required strings are supplied separately and are checked by a different part of the benchmark.
R4 NO DEPICTED SUBJECTS: do not record what a photo, illustration, icon or logo shows. Use image_kind / icon_kind and the slot geometry instead.
R5 ATTRIBUTES: record only properties that survive a redrawing, i.e. position, extent, aspect ratio, alignment, count, fill and border colour, emphasis, and chart or icon kind. Record an attribute ONLY where the brief states it.
R6 IDS: every node and every edge carries a stable id. Node ids look like n_<name>, edge ids like e_<name>.
R7 GROUP FOLDING: 3 or more elements the brief treats alike become ONE group node with attrs.count = N, not N separate nodes.

NODE_TYPES
  containers: region, card, panel, list, group
  atoms:      text, chart{chart_type}, table, icon{icon_kind}, image{image_kind},
              button, input, divider, connector, badge, timeline, banner
EDGE_TYPES
  spatial:    contains, left_of, right_of, above, below, aligned_with{axis},
              evenly_spaced{members}
  semantic:   connects{direction}, timeline_step{order}, parent_of

Emit a semantic edge only where the brief actually asserts the link, e.g. a stated flow, a numbered sequence, or a stated grouping under a heading.

OUTPUT (strict JSON, no fences):
{"layout_summary": "one sentence", "graph": {"nodes": [{"id": ..., "type": ..., "attrs": {...}}], "edges": [{"id": ..., "type": ..., "from": ..., "to": ..., "attrs": {...}}]}}"""

_GRAPH_USER = """Brief the page will be generated from:

{brief}

Data and strings the page is required to carry ({n_req}):
{requirements}

Emit the graph of the page this brief asks for, as strict JSON."""


# --------------------------------------------------------------------------
# Call 2: question authoring, one call per bucket
# --------------------------------------------------------------------------

_AUTHOR_SYSTEM = """You write yes/no questions that a judge will answer by looking at ONE rendered page image, with no reference image, no access to the markup, and without being shown the brief. You are given the brief and the scene graph already extracted from it. Write questions for the {bucket} dimension only.

{bucket_brief}

KINDS AND QUOTAS for this page. Write exactly this many of each kind, and put the kind on every question:
{quota_block}

THE QUESTION CONTRACT. Every question must:
C1 be answerable yes or no by a careful viewer from a single rendering, with no reference image and no access to the markup. Nothing finer than about 20 pixels is decidable.
C2 name the element it is about, the way a viewer would recognise it ("the dark panel down the left side"), never by node id. Whole-page questions are the exception and may address the canvas.
C3 state any threshold it depends on. "Is the heading large enough?" is not answerable; "Is the heading at least twice the height of the body text?" is.
C4 carry an explicit pass_answer, "yes" or "no". Some questions state a property that should hold; others ask whether a defect is present. The judge is not told which, so the polarity must be recorded.
C5 declare its kind, from the list above.

And every question must NOT:
C6 ask about the content of a placeholder image slot. The page is expected to draw a placeholder where a photo goes; what it depicts is not the page's choice. Ask about the slot's presence, position, and proportions instead.
C7 prescribe an implementation rather than a visible outcome. No CSS properties, no element names, no "is it a flex row". Ask what a viewer sees.
C8 restate a browser default. A heading already renders larger than body text and a block already starts on a new line; a question that cannot fail teaches the table nothing.
C9 bundle several claims that could not fail independently. One question, one claim, never two joined by "and".

BIND EVERY QUESTION to the graph element it tests: "node:<id>" or "edge:<id>", using an id that already exists in the given graph. Whole-page kinds ({page_level}) bind to nothing and omit the field.

ASK ABOUT WHOLE SETS where the graph gives you one. A question about a single element of a repeated set is passed by nearly every attempt, so a checklist built from them saturates and its ordering is decided by noise.
    weak   "Is the first metric card showing a value?"
    strong "Do all four metric cards show both their label and their value?"

ONLY WHAT THE BRIEF STATES. The graph was read off the brief, but if you find yourself asking about a colour, a size or a position the brief never fixed, drop the question and spend the budget on something the brief did commit to.

Output STRICT JSON, no fences:
{{"questions": [{{"id": "...", "kind": "...", "binds_to": "node:n_x", "question": "...", "pass_answer": "yes"}}]}}"""

_BUCKET_BRIEF = {
    "Content": """CONTENT asks whether a required element is PRESENT and INTACT. Nothing else.
  located_presence   is the required element there, in the region the brief puts it in
  element_count      are there exactly as many as the brief states, no fewer and no extras
  element_integrity  is each member whole: not truncated, not clipped, not collapsed to an empty box

Do not ask a bare defect question. "Is anything cut off?" is passed for free by a page too empty to go wrong. Conjoin the defect with the presence of what could cause it and set pass_answer "yes": "Are all six cards present, each showing its full label?".""",

    "Layout": """LAYOUT asks how elements sit relative to one another or to the canvas.
  structure     the stated arrangement: a row, a column, a grid of a stated shape, a stated order
  containment   members actually inside the container the brief puts them in
  collision     elements that must not overlap do not overlap, and nothing is clipped by the canvas
  relation      a stated binary spatial predicate: beside, above, aligned, evenly spaced
  connective    a stated link is actually drawn: an arrow, a numbered step, a connector between the two things it joins
  subgraph      a stated multi-element structure holds as a whole, not element by element
  page_balance  exactly one per page, about the whole canvas: margins, gutters and the distribution of weight

Bind to an edge wherever the relation you are testing is an edge in the graph.""",

    "Style": """STYLE asks about the quality of the rendering, and is judged to a HIGHER standard of proof than the other dimensions: a pass requires that a viewer could point at the pixels that establish it.
  concrete_attribute  one stated visual property: a stated colour, a stated weight, a stated relative size
  element_quality     a complex construct is properly realised rather than crudely faked. Boxes in a row standing in for a timeline with no connecting spine do not pass; a chart with no axes or no legend does not pass
  icon_quality        an icon reads as the thing it denotes rather than as a blob or a lettered square
  page_density        the page is neither starved nor crammed: content fills the canvas without colliding or running off it
  delivery_gate       exactly five per page, bound to the canvas, one each for {gate_aspects}. These are the one place a question may be a LIST: each names several specific, independently visible failures of its aspect and fails if ANY of them is present. Name concrete failures, not "looks unprofessional\"""",
}

_AUTHOR_USER = """Brief the page will be generated from:

{brief}

Scene graph extracted from that brief:

{graph_json}

Write the {bucket} questions for this page, as strict JSON."""


def _parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if "```" in text:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start:end])
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


async def _chat(client, model: str, system: str, user: str, *,
                timeout: float, max_tokens: int) -> str:
    kwargs: dict[str, Any] = {}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["reasoning_effort"] = "low"
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_completion_tokens=max_tokens,
            **kwargs,
        ),
        timeout=timeout,
    )
    return (response.choices[0].message.content or "").strip()


def format_requirements(sample: dict) -> str:
    """The data contract, as the brief's author would state it."""
    lines = []
    for req in sample.get("verifiable_requirements") or []:
        category = str(req.get("category", "")).strip()
        expected = req.get("expected") or {}
        if category == "text_exact":
            text = str(expected.get("text", "")).strip()
            if text:
                lines.append(f'  - must display the exact string "{text}"')
        elif category in ("chart_data", "table_data"):
            target = req.get("target") or {}
            what = target.get("chart_subtype") or target.get("chart_type") or category
            data = expected.get("data")
            n = len(data) if isinstance(data, list) else "?"
            lines.append(f"  - must carry a {what} with {n} data entries")
        elif category:
            lines.append(f"  - {category}: {json.dumps(expected, ensure_ascii=False)[:160]}")
    return "\n".join(lines) or "  (none)"


async def extract_scene_graph(client, model: str, sample: dict, *,
                              timeout: float = 180.0,
                              max_tokens: int = 6000) -> dict[str, Any]:
    """Call 1 — the graph of the page the brief asks for."""
    brief = str(sample.get("generation_prompt", "")).strip()
    requirements = format_requirements(sample)
    raw = await _chat(
        client, model, _GRAPH_SYSTEM,
        _GRAPH_USER.format(
            brief=brief[:9000],
            n_req=len(sample.get("verifiable_requirements") or []),
            requirements=requirements[:4000]),
        timeout=timeout, max_tokens=max_tokens)
    obj = _parse_json_object(raw)

    graph = obj.get("graph") or {}
    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("id")]
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict) and e.get("id")]
    if not nodes:
        raise ValueError("scene graph has no nodes")
    return {"layout_summary": str(obj.get("layout_summary", "")),
            "graph": {"nodes": nodes, "edges": edges}}


async def author_bucket(client, model: str, sample: dict, design_graph: dict,
                        bucket: str, quota: dict[str, int], *,
                        timeout: float = 180.0,
                        max_tokens: int = 8000) -> list[dict[str, Any]]:
    """Call 2 — the questions for one dimension."""
    if not quota:
        return []
    quota_block = "\n".join(f"  {k:<18} {n}" for k, n in quota.items())
    bucket_brief = _BUCKET_BRIEF[bucket].format(
        gate_aspects=", ".join(DELIVERY_GATE_ASPECTS))
    system = _AUTHOR_SYSTEM.format(
        bucket=bucket, bucket_brief=bucket_brief, quota_block=quota_block,
        page_level=", ".join(rubric_io.PAGE_LEVEL_KINDS))
    raw = await _chat(
        client, model, system,
        _AUTHOR_USER.format(
            brief=str(sample.get("generation_prompt", "")).strip()[:9000],
            graph_json=json.dumps(design_graph["graph"], ensure_ascii=False,
                                  indent=2)[:12000],
            bucket=bucket),
        timeout=timeout, max_tokens=max_tokens)
    obj = _parse_json_object(raw)

    allowed = set(quota)
    out = []
    for i, q in enumerate(obj.get("questions", []), start=1):
        if not isinstance(q, dict) or not str(q.get("question", "")).strip():
            continue
        kind = str(q.get("kind", "")).strip()
        if kind not in allowed:
            # The linter reports it; keeping it would put a question in a bucket
            # this call was not asked to write.
            kind = next(iter(quota))
            q = {**q, "kind": kind, "_kind_coerced": True}
        pass_answer = str(q.get("pass_answer", "yes")).strip().lower()
        out.append({
            "id": str(q.get("id") or f"{bucket[0]}{i:02d}").strip(),
            "kind": kind,
            "binds_to": str(q.get("binds_to", "") or "").strip(),
            "question": str(q["question"]).strip(),
            "pass_answer": pass_answer if pass_answer in ("yes", "no") else "yes",
            "_kind_coerced": bool(q.get("_kind_coerced")),
        })
    return out


# --------------------------------------------------------------------------
# Stage 3: the linter
# --------------------------------------------------------------------------
# One regex per clause of the question contract. These are cheap screens, not
# proofs: they catch the failure modes an author actually produces, and a flagged
# checklist is reported rather than silently rewritten.

# C6 — the depicted subject of a placeholder slot.
_PLACEHOLDER_RE = re.compile(
    r"\b(?:depict|depicts|depicting|show|shows|showing|portray|portrays|"
    r"illustrat\w+|represent|represents)\b[^?]*?"
    r"\b(?:photo|photograph|image|picture|illustration|logo|artwork|placeholder)\b|"
    r"\b(?:photo|photograph|image|picture|illustration|logo)\b[^?]*?"
    r"\b(?:depict|depicts|show|shows|portray|portrays)\b",
    re.IGNORECASE)

# C7 — implementation rather than visible outcome.
_IMPLEMENTATION_RE = re.compile(
    r"\b(?:css|html|<[a-z]+>|div|span|flex(?:box)?|grid-template|z-index|"
    r"border-radius|font-family|margin-(?:top|left|right|bottom)|padding|"
    r"stylesheet|class(?:es)?\s+named|inline\s+style|viewport\s+unit|"
    r"media\s+quer\w+|svg\s+element|dom\b)\b",
    re.IGNORECASE)

# C8 — a browser default, true before the page is written.
_BROWSER_DEFAULT_RE = re.compile(
    r"\b(?:heading|title|h1|h2)\b[^?]*\b(?:larger|bigger)\b[^?]*\bbody\b|"
    r"\bdoes\s+the\s+page\s+have\s+a\s+background\b|"
    r"\bis\s+the\s+(?:text|body\s+text)\s+readable\s*\?|"
    r"\bis\s+there\s+any\s+(?:text|content)\s+on\s+the\s+page\b",
    re.IGNORECASE)

# C9 — two claims that could fail independently.
_CONJUNCTION_RE = re.compile(
    r"\s+and\s+(?:is|are|does|do|has|have|was|were|the\s+\w+\s+(?:is|are))\b",
    re.IGNORECASE)

# C3 — a comparative with no stated threshold.
_VAGUE_RE = re.compile(
    r"\b(?:large|small|big|wide|narrow|tall|short|close|near|far)\s+enough\b|"
    r"\b(?:appropriately|properly|correctly|reasonably|sufficiently|adequately|"
    r"nicely|well)\s+\w+|"
    r"\b(?:good|bad|nice|clean|professional|polished|pleasing|attractive|"
    r"balanced|harmonious)\b",
    re.IGNORECASE)
_THRESHOLD_RE = re.compile(
    r"\d|\b(?:half|third|quarter|twice|double|triple|same|equal|at\s+least|"
    r"at\s+most|no\s+more\s+than|no\s+fewer\s+than|each\s+other)\b",
    re.IGNORECASE)

BINDING_PREFIXES = ("node:", "edge:")
_BINDING_RE = re.compile(r"^(node|edge):(.+)$")


def lint_rubric(questions: list[dict[str, Any]], design_graph: dict,
                quotas: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Check a checklist against the question contract and the quotas.

    Returns the flags raised and the share of questions bound to an element of
    their own graph. Flag codes are the contract clause they violate, so a flag
    points at the rule rather than at a symptom.
    """
    graph = design_graph.get("graph", {}) or {}
    known = {str(n.get("id")) for n in graph.get("nodes", []) if n.get("id")}
    known |= {str(e.get("id")) for e in graph.get("edges", []) if e.get("id")}

    flags: list[dict[str, str]] = []
    bound_ok = 0
    seen: dict[str, str] = {}

    def flag(qid: str, code: str, detail: str = "") -> None:
        flags.append({"id": qid, "code": code, "detail": detail})

    for q in questions:
        qid = str(q.get("id", "?"))
        text = str(q.get("question", ""))
        kind = str(q.get("kind", ""))
        page_level = kind in rubric_io.PAGE_LEVEL_KINDS

        # C5 — declares a kind that exists in the taxonomy.
        if not kind:
            flag(qid, "C5_missing_kind")
        elif rubric_io.bucket_of(kind) is None:
            flag(qid, "C5_unknown_kind", kind)
        if q.get("_kind_coerced"):
            flag(qid, "C5_kind_out_of_bucket", kind)

        # C4 — explicit polarity.
        if q.get("pass_answer") not in ("yes", "no"):
            flag(qid, "C4_bad_polarity", str(q.get("pass_answer")))

        # C2 — binds to its own graph, and names something.
        binds = str(q.get("binds_to", "") or "").strip()
        if page_level:
            if binds:
                flag(qid, "C2_page_kind_bound", binds)
            else:
                bound_ok += 1
        elif not binds:
            flag(qid, "C2_missing_binding")
        else:
            m = _BINDING_RE.match(binds)
            if not m:
                flag(qid, "C2_bad_binding_kind", binds)
            elif m.group(2) not in known:
                flag(qid, "C2_unknown_binding", binds)
            else:
                bound_ok += 1

        # C1 — decidable from one rendering. A question too short to name
        # anything is not decidable, whatever else it does.
        if len(text.split()) < 5:
            flag(qid, "C1_not_decidable", text[:60])

        # C3 — a comparative needs its threshold.
        if _VAGUE_RE.search(text) and not _THRESHOLD_RE.search(text):
            flag(qid, "C3_unstated_threshold", text[:80])

        if _PLACEHOLDER_RE.search(text):
            flag(qid, "C6_image_content_question", text[:80])
        if _IMPLEMENTATION_RE.search(text):
            flag(qid, "C7_prescribes_implementation", text[:80])
        if _BROWSER_DEFAULT_RE.search(text):
            flag(qid, "C8_browser_default", text[:80])
        # C9 exempts the delivery gate, which is a disqualifier list by design.
        if kind != "delivery_gate" and _CONJUNCTION_RE.search(text):
            flag(qid, "C9_not_atomic", text[:80])

        key = text.lower().strip(" ?.")
        if key in seen:
            flag(qid, "C9_duplicate", f"same as {seen[key]}")
        else:
            seen[key] = qid

    # Quotas.
    counts: dict[str, int] = {}
    for q in questions:
        counts[str(q.get("kind"))] = counts.get(str(q.get("kind")), 0) + 1
    for kind, want in FIXED_QUOTAS.items():
        got = counts.get(kind, 0)
        if got != want and any(kind in plan for plan in quotas.values()):
            flag("-", f"Q1_{kind}_quota", f"{got} != {want}")

    n = len(questions)
    if not MIN_QUESTIONS <= n <= MAX_QUESTIONS:
        flag("-", "Q2_question_count_out_of_range",
             f"{n} outside {MIN_QUESTIONS}-{MAX_QUESTIONS}")

    by_bucket: dict[str, int] = {}
    for q in questions:
        bucket = rubric_io.bucket_of(str(q.get("kind")))
        if bucket:
            by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
    for bucket in rubric_io.BUCKETS:
        if not by_bucket.get(bucket):
            flag("-", "Q3_empty_bucket", bucket)

    return {"flags": flags,
            "bound_to_own_graph": round(bound_ok / n, 4) if n else 0.0,
            "n_questions": n,
            "by_bucket": by_bucket,
            "by_kind": counts}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _build_client(timeout: float):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("Requires: pip install openai")
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "").split(",")[0].strip()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY to an OpenAI-compatible key.")
        sys.exit(1)
    return AsyncOpenAI(base_url=base_url or None, api_key=api_key, timeout=timeout)


def released(sid: str, questions: list[dict], warn=None) -> dict:
    """The checklist in released schema, with bucket and category derived."""
    out = []
    for i, q in enumerate(questions, start=1):
        norm = rubric_io.normalise_question(
            {**{k: v for k, v in q.items() if not k.startswith("_")},
             "id": q.get("id") or f"q{i}"},
            warn=warn, where=f"{sid}: ")
        if norm:
            out.append(norm)
    return {"sid": sid, "questions": out}


async def build_one(client, model: str, sample: dict, *, timeout: float,
                    max_tokens: int, target: int | None) -> dict[str, Any]:
    sid = str(sample["id"])
    design_graph = await extract_scene_graph(
        client, model, sample, timeout=timeout, max_tokens=max_tokens)
    quotas = plan_quotas(design_graph["graph"], target)

    authored = await asyncio.gather(*(
        author_bucket(client, model, sample, design_graph, bucket, quotas[bucket],
                      timeout=timeout, max_tokens=max_tokens)
        for bucket in rubric_io.BUCKETS))

    questions: list[dict] = []
    for bucket, got in zip(rubric_io.BUCKETS, authored):
        for i, q in enumerate(got, start=1):
            q["id"] = f"{bucket[0]}{i:02d}"
            questions.append(q)

    report = lint_rubric(questions, design_graph, quotas)
    return {"sid": sid, "design_graph": design_graph, "quotas": quotas,
            "questions": questions, "lint": report}


async def main(args: argparse.Namespace) -> None:
    client = _build_client(args.timeout)
    model = args.model or os.environ.get("VLM_MODEL", "gpt-5.5")

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.sids_file:
        want = {s for s in Path(args.sids_file).read_text().split() if s}
        samples = [s for s in samples if str(s.get("id")) in want]
    logger.info("Loaded %d samples from %s", len(samples), args.input)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = Path(args.save_graphs) if args.save_graphs else None
    if graph_dir:
        graph_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict | None] = [None] * len(samples)

    async def _process_one(idx: int, sample: dict) -> None:
        sid = str(sample.get("id", f"sample_{idx}"))
        if not str(sample.get("generation_prompt", "")).strip():
            logger.warning("[%s] no generation prompt, skipping", sid)
            return
        async with sem:
            try:
                built = await build_one(
                    client, model, sample, timeout=args.timeout,
                    max_tokens=args.max_tokens, target=args.target_questions)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] checklist construction failed: %s", sid, exc)
                return

        payload = released(sid, built["questions"],
                           warn=lambda m: logger.warning("%s", m))
        (out_dir / f"{sid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if graph_dir:
            (graph_dir / f"{sid}.json").write_text(
                json.dumps(built["design_graph"], ensure_ascii=False, indent=2),
                encoding="utf-8")
        results[idx] = built

        report = built["lint"]
        logger.info("[%s] %d nodes, %d edges -> %d questions %s, %d lint flag(s)",
                    sid, len(built["design_graph"]["graph"]["nodes"]),
                    len(built["design_graph"]["graph"]["edges"]),
                    len(payload["questions"]), report["by_bucket"],
                    len(report["flags"]))
        for f in report["flags"][:5]:
            logger.info("    [%s] %s %s", f["id"], f["code"], f["detail"][:60])

    await asyncio.gather(*(_process_one(i, s) for i, s in enumerate(samples)))

    done = [r for r in results if r]
    if not done:
        logger.error("No checklist was built.")
        return
    (out_dir / "_lint.json").write_text(json.dumps(
        {r["sid"]: r["lint"] for r in done}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    n_q = sum(r["lint"]["n_questions"] for r in done)
    flagged = [r for r in done if r["lint"]["flags"]]
    bound = sum(r["lint"]["bound_to_own_graph"] * r["lint"]["n_questions"] for r in done)
    mix: dict[str, int] = {}
    for r in done:
        for bucket, c in r["lint"]["by_bucket"].items():
            mix[bucket] = mix.get(bucket, 0) + c
    logger.info(
        "Done. %d/%d samples, %.1f questions on average, mix %s, "
        "%d flagged (%.1f%%), %.1f%% bound to their own graph. Output: %s",
        len(done), len(samples), n_q / len(done),
        {b: f"{100.0 * c / n_q:.0f}%" for b, c in mix.items()},
        len(flagged), 100.0 * len(flagged) / len(done),
        100.0 * bound / n_q if n_q else 0.0, out_dir)
    await client.close()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build evaluation checklists: scene graph extraction, then "
                    "one authoring call per dimension, then the contract linter.")
    parser.add_argument("--input", required=True, help="Spec JSONL file")
    parser.add_argument("--output", required=True, help="Directory for one <sid>.json checklist per sample")
    parser.add_argument("--sids-file", help="Restrict to the sample ids listed in this file")
    parser.add_argument("--save-graphs", help="Also write one scene graph JSON per sample into this directory")
    parser.add_argument("--model", help="Authoring model (default: $VLM_MODEL, else gpt-5.5)")
    parser.add_argument("--target-questions", type=int, help="Override the graph-size-derived checklist length")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent samples")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-call timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Max completion tokens per call")
    asyncio.run(main(parser.parse_args()))


if __name__ == "__main__":
    cli()
