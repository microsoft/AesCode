#!/usr/bin/env python3
"""Build the Visual Graph Rubric (VGR) for each sample.

Implements the two-call construction described in the paper (Appendix D.4):

    Call 1  scene graph extraction
            Reads the generation prompt and the reference rendering, commits to a
            layout class, and returns a skeleton plus a graph of nodes and edges.

    Call 2  rubric authoring from the graph
            Receives that graph with the same prompt and image and writes yes/no
            questions, each bound to a node, an edge, a set of nodes, or the whole
            canvas, and typed as structure, completeness, visual_attr, relation,
            forbidden, global or subgraph.

Fixing the graph before the questions are written keeps the structure a question
refers to from drifting, and lets every question cite the element it tests. A
programmatic linter then checks every rubric for missing bindings, inconsistent
polarity, and questions about text or image content.

Usage:
    python -m data_prep.build_vgr \
        --input dataset/split_rl.jsonl \
        --output dataset/rubrics.jsonl \
        --image-dir /path/to/spec_images \
        --concurrency 16

The rubric is a standalone artifact, not a second copy of the split. Each output
row holds the `id` it was built from and nothing else from the input:

    design_graph      {"skeleton": [...], "layout_class": ..., "graph": {...}}
    rubric_questions  [{"id", "binds_to", "qtype", "subtype", "question",
                        "pass_answer", "rationale"}, ...]
    vgr_lint          {"flags": [...], "bound_to_own_graph": 0.0-1.0}

`rl_verl.data_prep --rubric-input` joins this file back onto the split by `id`
when it builds the training parquet. `rubric_questions` is the field
RubricDesignScorer reads during RL training: the judge answers the whole list in
one call from the candidate rendering alone, and the reward is the pass rate.

The endpoint is any OpenAI-compatible chat completions API:

    export OPENAI_BASE_URL=https://api.openai.com/v1
    export OPENAI_API_KEY=...
    export VLM_MODEL=gpt-5.5

"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


LAYOUT_CLASSES = (
    "symmetric_grid",
    "hierarchical",
    "timeline_sequence",
    "hub_spoke",
    "free_form",
)

QTYPES = (
    "structure",
    "completeness",
    "visual_attr",
    "relation",
    "forbidden",
    "global",
    "subgraph",
)

RELATION_SUBTYPES = (
    "spatial",
    "semantic_connect",
    "semantic_sequence",
    "semantic_hierarchy",
)

# Semantic edges the rubric is required to exercise (R7), mapped to the relation
# subtype that discharges them and the number of edges that triggers the rule.
_SEMANTIC_EDGE_COVERAGE = {
    "connects": ("semantic_connect", 1),
    "timeline_step": ("semantic_sequence", 2),
    "parent_of": ("semantic_hierarchy", 2),
}


# --------------------------------------------------------------------------
# Call 1: scene graph extraction
# --------------------------------------------------------------------------

_GRAPH_SYSTEM = """You are a scene-graph extractor for page designs. Read the brief the design was produced from and the rendered reference, commit to a layout class before anything else, then describe the page as a graph. Describe only what is structurally present. Do not write questions; a later step does that.

R1 BOUNDARIES: a node is a region the eye can separate from its surroundings by an edge, a background fill, or an unambiguous gap. Do not invent nodes for internal padding, and do not merge two elements that a reader would treat as separate cards.
R2 NO DEPICTED SUBJECTS: do not record what a photo, illustration, icon or logo shows. Use image_kind / icon_kind and the slot geometry instead.
R3 NO TEXT CONTENT: do not transcribe strings. A text node records its role, position, extent and emphasis, not its wording.
R4 ATTRIBUTES: record only properties that are stable under redrawing, i.e. position, extent, aspect ratio, alignment, count, fill and border colour, shadow, and chart or icon kind.
R5 IDS: every node and every edge carries a stable id. Node ids look like n_<name>, edge ids like e_<name>.
R6 GROUP FOLDING: 3+ identical elements become ONE group node with attrs.count=N, not N separate nodes.

STEP 1  SKELETON + LAYOUT CLASS.
Choose one of symmetric_grid | hierarchical | timeline_sequence | hub_spoke | free_form.
List 5-7 essential elements; essential means removing it breaks the story. Each entry carries name, essential_because and the node_id it will take in the graph.

STEP 2  GRAPH.
NODE_TYPES
  containers: region, card, panel, list, group
  atoms:      text, chart{chart_type}, table, icon{icon_kind},
              image{image_kind}, button, input, divider, connector,
              badge, pyramid, funnel, timeline, banner
EDGE_TYPES
  spatial:    contains, left_of, right_of, above, below,
              aligned_with{axis}, evenly_spaced{members}
  semantic:   connects{direction}, timeline_step{order}, parent_of

Every node carries a stable id, a type, and an attrs object. Emit a semantic edge only where the design actually asserts the link, e.g. a drawn arrow, a numbered step, or containment under a parent heading.

OUTPUT (strict JSON, no fences): {"skeleton": [{"name": ..., "essential_because": ..., "node_id": ...}], "layout_class": ..., "layout_class_reason": ..., "graph": {"nodes": [{"id": ..., "type": ..., "attrs": {...}}], "edges": [{"id": ..., "type": ..., "from": ..., "to": ..., "attrs": {...}}]}}"""

_GRAPH_USER = """Brief the design was produced from:

{generation_prompt}

Read the attached reference rendering, commit to a layout class, then emit the skeleton and the graph as strict JSON."""


# --------------------------------------------------------------------------
# Call 2: rubric authoring from the graph
# --------------------------------------------------------------------------

_RUBRIC_SYSTEM = """You are a scene-graph-structured rubric author. You are given the brief, the rendered reference, and the scene graph already extracted from it. Write yes/no questions against that graph. Do not add, rename, or re-type graph elements. Adapt the subgraph quota to the given layout class; forced fake subgraph questions on flat symmetric layouts are worse than no subgraph at all.

R2 OPAQUE SLOTS: never make the depicted subject of a photo, illustration, icon, or logo the subject of a question; the reference generator may legitimately draw something else in the same slot.
  BAD:  "Does the portrait depict a smiling person?"
  BAD:  "Is the airplane fully visible?"
  GOOD: "Is there a photo-shaped placeholder slot in the upper-left, above the light-gray information panel?"
  GOOD: "Is the upper-left photo region rectangular and approximately 1:1 to 4:3 in aspect ratio?"
  A depicted-subject noun may appear only as an anchor naming which slot.
R3 POLARITY: every question carries pass_answer: "yes" | "no".
R5 BINDING: every question binds to node:<id> | edge:<id> | subgraph:<id,...> | global:page. The id must already exist in the given graph.
R7 SEMANTIC EDGE COVERAGE (mandatory): >=1 connects edge requires a semantic_connect question; >=2 timeline_step edges require a semantic_sequence question; >=2 parent_of edges require a semantic_hierarchy question.
R8 NO TEXT OR IMAGE CONTENT: do not ask whether a string is present or what it says, and do not ask what an image shows; both are scored elsewhere.

RUBRIC, 10-20 questions, quotas by layout class:
  structure      3-4  arrangement / split / alignment / row / column
  completeness   2-3  intact / no overlap / fully visible
  visual_attr    1-2  color / shadow / border
  relation       1-3  binary edge predicates; subtype required, one of
                      spatial | semantic_connect | semantic_sequence | semantic_hierarchy
  forbidden      1-2  at least one, e.g. "Are there extra X beyond N?" pass=no
  global         2    exactly two, last, pass=yes: no unintended overlap or
                      occlusion; all content fully contained and unclipped
  subgraph            symmetric_grid / free_form: 0-1, do not force;
                      hierarchical, timeline_sequence, hub_spoke: 1-2.
                      A subgraph question must require graph structure, e.g.
                      "Do the 4 timeline nodes follow strict order 2018->2020->2022->2024 via the connector arrows?"
                      and not "Are the 6 cards in a 3x2 grid?"

QUANTIFY OVER A WHOLE SET WHEREVER THE GRAPH GIVES YOU ONE. A question about one element is almost always passed by every competent attempt, so a rubric made of them saturates and its ordering is decided by noise. Ask about every member of a set, so that an attempt which did the work on some members and not the others fails. Ask about a single element only when the design really has just one of that thing.

ATOMICITY: one question = one binary check, no AND/OR.

OUTPUT (strict JSON, no fences): {"rubric": [{"id": ..., "binds_to": ..., "qtype": ..., "subtype": ..., "question": ..., "pass_answer": ..., "rationale": ...}], "qtype_count": {...}}"""

_RUBRIC_USER = """Brief the design was produced from:

{generation_prompt}

Layout class: {layout_class}

Scene graph already extracted from the attached reference:

{graph_json}

Write the rubric as strict JSON."""


def _encode_image(path: str | Path) -> str:
    from PIL import Image as PILImage

    img = PILImage.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating code fences."""
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


async def _chat(
    client,
    model: str,
    system: str,
    user: str,
    image_b64: str,
    *,
    timeout: float,
    max_tokens: int,
) -> str:
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": user},
                    ],
                },
            ],
            max_completion_tokens=max_tokens,
            temperature=0.3,
        ),
        timeout=timeout,
    )
    return (response.choices[0].message.content or "").strip()


async def extract_scene_graph(
    client,
    model: str,
    generation_prompt: str,
    image_b64: str,
    *,
    timeout: float = 120.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Call 1 — commit to a layout class and describe the page as a graph."""
    raw = await _chat(
        client,
        model,
        _GRAPH_SYSTEM,
        _GRAPH_USER.format(generation_prompt=generation_prompt),
        image_b64,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    obj = _parse_json_object(raw)

    graph = obj.get("graph") or {}
    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("id")]
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict) and e.get("id")]
    if not nodes:
        raise ValueError("scene graph has no nodes")

    layout_class = str(obj.get("layout_class", "")).strip()
    if layout_class not in LAYOUT_CLASSES:
        layout_class = "free_form"

    return {
        "skeleton": obj.get("skeleton", []),
        "layout_class": layout_class,
        "layout_class_reason": obj.get("layout_class_reason", ""),
        "graph": {"nodes": nodes, "edges": edges},
    }


async def author_rubric(
    client,
    model: str,
    generation_prompt: str,
    image_b64: str,
    design_graph: dict[str, Any],
    *,
    timeout: float = 120.0,
    max_tokens: int = 4096,
) -> list[dict[str, Any]]:
    """Call 2 — write yes/no questions against the already-extracted graph."""
    raw = await _chat(
        client,
        model,
        _RUBRIC_SYSTEM,
        _RUBRIC_USER.format(
            generation_prompt=generation_prompt,
            layout_class=design_graph.get("layout_class", "free_form"),
            graph_json=json.dumps(design_graph.get("graph", {}), ensure_ascii=False, indent=2),
        ),
        image_b64,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    obj = _parse_json_object(raw)

    questions: list[dict[str, Any]] = []
    for i, q in enumerate(obj.get("rubric", []), start=1):
        if not isinstance(q, dict) or not str(q.get("question", "")).strip():
            continue
        pass_answer = str(q.get("pass_answer", "yes")).strip().lower()
        if pass_answer not in {"yes", "no"}:
            pass_answer = "yes"
        qtype = str(q.get("qtype", "")).strip()
        questions.append(
            {
                "id": str(q.get("id") or f"q{i}").strip(),
                "binds_to": str(q.get("binds_to", "")).strip(),
                "qtype": qtype if qtype in QTYPES else "structure",
                "subtype": str(q.get("subtype", "") or "").strip(),
                "question": str(q["question"]).strip(),
                "pass_answer": pass_answer,
                "rationale": str(q.get("rationale", "") or "").strip(),
            }
        )
    return questions


# --------------------------------------------------------------------------
# Linter
# --------------------------------------------------------------------------

_TEXT_CONTENT_RE = re.compile(
    r"\b(say|says|read|reads|spell|spelled|wording|verbatim|text\s+content|"
    r"label\s+text|exact\s+(?:string|text|wording))\b|"
    r"(?:text|string|label|heading|title|caption)\s+(?:is|be|reads?|says?)\s+[\"']",
    re.IGNORECASE,
)

_IMAGE_CONTENT_RE = re.compile(
    r"\b(?:depict|depicts|depicting|show|shows|showing|portray|portrays)\b"
    r"[^?]*?\b(?:photo|photograph|image|picture|illustration|icon|logo|artwork)\b|"
    r"\b(?:photo|photograph|image|picture|illustration|icon|logo)\b[^?]*?"
    r"\b(?:depict|depicts|show|shows|portray|portrays)\b",
    re.IGNORECASE,
)

_CONJUNCTION_RE = re.compile(r"\s+(?:and|or)\s+(?:is|are|does|do|has|have)\b", re.IGNORECASE)


def _binding_ids(binds_to: str) -> list[str]:
    binds_to = (binds_to or "").strip()
    if not binds_to or ":" not in binds_to:
        return []
    kind, _, rest = binds_to.partition(":")
    if kind.strip() == "global":
        return []
    return [part.strip() for part in rest.split(",") if part.strip()]


# R5: the only binding kinds a question may use.
BINDING_PREFIXES = ("node:", "edge:", "subgraph:", "global:")


def lint_rubric(questions: list[dict[str, Any]], design_graph: dict[str, Any]) -> dict[str, Any]:
    """Check a rubric for missing bindings, bad polarity, and content questions.

    Returns the flags raised and the share of questions that bind to an element
    of their own graph. The paper reports 0.3% of training rubrics flagged and
    99.6% of questions bound to their own graph.
    """
    graph = design_graph.get("graph", {}) or {}
    known_ids = {str(n.get("id")) for n in graph.get("nodes", []) if n.get("id")}
    known_ids |= {str(e.get("id")) for e in graph.get("edges", []) if e.get("id")}

    flags: list[dict[str, str]] = []
    bound_ok = 0

    def flag(qid: str, code: str, detail: str = "") -> None:
        flags.append({"id": qid, "code": code, "detail": detail})

    for q in questions:
        qid = q.get("id", "?")
        binds_to = q.get("binds_to", "")
        text = q.get("question", "")

        ids = _binding_ids(binds_to)
        if not binds_to:
            flag(qid, "missing_binding")
        elif not any(binds_to.startswith(p) for p in BINDING_PREFIXES):
            # R5 restricts bindings to node:/edge:/subgraph:/global:page.
            flag(qid, "bad_binding_kind", binds_to)
        elif binds_to.startswith("global:"):
            if binds_to != "global:page":
                flag(qid, "bad_binding_kind", binds_to)
            else:
                bound_ok += 1
        elif not ids:
            flag(qid, "missing_binding", binds_to)
        else:
            unknown = [i for i in ids if i not in known_ids]
            if unknown:
                flag(qid, "unknown_binding", ",".join(unknown))
            else:
                bound_ok += 1

        if q.get("pass_answer") not in {"yes", "no"}:
            flag(qid, "bad_polarity", str(q.get("pass_answer")))
        if q.get("qtype") == "forbidden" and q.get("pass_answer") != "no":
            flag(qid, "forbidden_polarity")
        if q.get("qtype") == "relation" and q.get("subtype") not in RELATION_SUBTYPES:
            flag(qid, "missing_relation_subtype", str(q.get("subtype")))

        if _TEXT_CONTENT_RE.search(text):
            flag(qid, "text_content_question")
        if _IMAGE_CONTENT_RE.search(text):
            flag(qid, "image_content_question")
        if _CONJUNCTION_RE.search(text):
            flag(qid, "not_atomic")

    # R7: every semantic edge kind present in the graph must be exercised.
    edge_type_counts: dict[str, int] = {}
    for e in graph.get("edges", []):
        etype = str(e.get("type", "")).split("{")[0].strip()
        edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1
    present_subtypes = {q.get("subtype") for q in questions if q.get("qtype") == "relation"}
    for etype, (subtype, threshold) in _SEMANTIC_EDGE_COVERAGE.items():
        if edge_type_counts.get(etype, 0) >= threshold and subtype not in present_subtypes:
            flag("-", "semantic_edge_uncovered", f"{etype}->{subtype}")

    n = len(questions)
    if not 10 <= n <= 20:
        flag("-", "question_count_out_of_range", str(n))
    if sum(1 for q in questions if q.get("qtype") == "global") != 2:
        flag("-", "global_quota")
    if not any(q.get("qtype") == "forbidden" for q in questions):
        flag("-", "no_forbidden_question")

    return {
        "flags": flags,
        "bound_to_own_graph": round(bound_ok / n, 4) if n else 0.0,
        "n_questions": n,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _resolve_reference_image(sample: dict, image_dir: Path | None) -> str:
    ref_meta = sample.get("reference_metadata", {})
    if isinstance(ref_meta, str):
        try:
            ref_meta = json.loads(ref_meta)
        except Exception:
            ref_meta = {}
    ref_image = ref_meta.get("reference_image_path", "") if isinstance(ref_meta, dict) else ""
    if image_dir and ref_image:
        ref_image = str(image_dir / Path(ref_image).name)
    if ref_image and Path(ref_image).exists():
        return ref_image
    if image_dir:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = image_dir / f"{sample.get('id', '')}{ext}"
            if cand.exists():
                return str(cand)
    return ""


def _build_client(timeout: float):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("Requires: pip install openai")
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY to an OpenAI-compatible key.")
        sys.exit(1)
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


async def main(args: argparse.Namespace) -> None:
    client = _build_client(args.timeout)
    model = args.model or os.environ.get("VLM_MODEL", "gpt-5.5")

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("Loaded %d samples from %s", len(samples), args.input)

    image_dir = Path(args.image_dir) if args.image_dir else None
    graph_dir = Path(args.save_graphs) if args.save_graphs else None
    if graph_dir:
        graph_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict | None] = [None] * len(samples)

    async def _process_one(idx: int, sample: dict) -> None:
        sample_id = str(sample.get("id", f"sample_{idx}"))

        ref_image = _resolve_reference_image(sample, image_dir)
        if not ref_image:
            logger.warning("[%s] no reference image, skipping", sample_id)
            return

        generation_prompt = str(sample.get("generation_prompt", "")).strip()
        if not generation_prompt:
            logger.warning("[%s] no generation prompt, skipping", sample_id)
            return

        async with sem:
            try:
                image_b64 = _encode_image(ref_image)
                design_graph = await extract_scene_graph(
                    client, model, generation_prompt, image_b64,
                    timeout=args.timeout, max_tokens=args.max_tokens,
                )
                questions = await author_rubric(
                    client, model, generation_prompt, image_b64, design_graph,
                    timeout=args.timeout, max_tokens=args.max_tokens,
                )
            except Exception as exc:
                logger.warning("[%s] VGR construction failed: %s", sample_id, exc)
                return

        report = lint_rubric(questions, design_graph)
        results[idx] = {
            "id": sample_id,
            "design_graph": design_graph,
            "rubric_questions": questions,
            "vgr_lint": report,
        }

        if graph_dir:
            (graph_dir / f"{sample_id}.json").write_text(
                json.dumps(design_graph, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        logger.info(
            "[%s] %s, %d nodes, %d edges, %d questions, %d lint flag(s)",
            sample_id,
            design_graph["layout_class"],
            len(design_graph["graph"]["nodes"]),
            len(design_graph["graph"]["edges"]),
            len(questions),
            len(report["flags"]),
        )

    await asyncio.gather(*(_process_one(i, s) for i, s in enumerate(samples)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    done = [r for r in results if r and r.get("rubric_questions")]
    flagged = [r for r in done if r.get("vgr_lint", {}).get("flags")]
    n_q = sum(len(r["rubric_questions"]) for r in done)
    bound = sum(
        r["vgr_lint"]["bound_to_own_graph"] * r["vgr_lint"]["n_questions"] for r in done
    )
    logger.info(
        "Done. %d/%d samples have a rubric, %.1f questions on average, "
        "%d flagged (%.1f%%), %.1f%% of questions bind to their own graph. Output: %s",
        len(done), len(samples),
        n_q / len(done) if done else 0.0,
        len(flagged), 100.0 * len(flagged) / len(done) if done else 0.0,
        100.0 * bound / n_q if n_q else 0.0,
        output_path,
    )
    await client.close()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Visual Graph Rubric: scene graph extraction, then rubric authoring."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output rubric JSONL, keyed by sample id")
    parser.add_argument("--image-dir", help="Directory holding the reference images")
    parser.add_argument("--save-graphs", help="Also write one scene graph JSON per sample into this directory")
    parser.add_argument("--model", help="Judge model (default: $VLM_MODEL, else gpt-5.5)")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent samples")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-call timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max completion tokens per call")
    asyncio.run(main(parser.parse_args()))


if __name__ == "__main__":
    cli()
