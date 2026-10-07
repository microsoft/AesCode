#!/usr/bin/env python3
"""The checklist schema, and the one place the kind taxonomy is written down.

A checklist question declares what KIND of thing it asks. The kind determines the
dimension the question scores, and nothing else does:

    kind  ->  bucket  ->  category

`bucket` is the table column. `category` is the judge call the question is asked
in. Both are pure functions of `kind`, so both are derived here on load and the
stored values are only ever checked against the derivation. Two stages reading
the same file and disagreeing about which column a question belongs to is a
failure that produces a plausible table rather than an error, so the derivation
lives in one module and both stages import it.

Released question schema:

    {"id": "q1", "bucket": "Layout", "category": "relation", "kind": "structure",
     "binds_to": "node:n_left_region", "question": "...", "pass_answer": "yes"}

Required: id, bucket, category, kind, question, pass_answer. Optional: binds_to.
Any other key is a remnant of an authoring pass and is dropped on load, so a
downstream stage cannot come to depend on one.

A checklist file is {"sid": str, "questions": [...]}. A checklist directory may
also hold bookkeeping such as `_manifest.json`; anything whose `questions` is not
a list is not a checklist and is skipped.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# The taxonomy (paper Appendix E.1)
# ---------------------------------------------------------------------------
# Content collects the kinds that ask whether a required element is present and
# intact. Layout collects the kinds about how elements sit relative to one
# another or to the canvas. Style collects the kinds about the quality of the
# rendering.
KINDS_BY_BUCKET = {
    "Content": ("located_presence", "element_count", "element_integrity"),
    "Layout": ("collision", "containment", "structure", "page_balance",
               "connective", "relation", "subgraph"),
    "Style": ("concrete_attribute", "element_quality", "icon_quality",
              "page_density", "delivery_gate", "design_labelled"),
}

CATEGORY_BY_BUCKET = {
    "Content": "completeness",
    "Layout": "relation",
    "Style": "style_strict",
}

BUCKETS = tuple(KINDS_BY_BUCKET)
KIND_BUCKET = {k: b for b, ks in KINDS_BY_BUCKET.items() for k in ks}
KINDS = tuple(KIND_BUCKET)

# Kinds that are about the whole canvas rather than one element. They bind to no
# graph element, and the linter exempts them from the "name the element" rule.
PAGE_LEVEL_KINDS = ("page_balance", "page_density", "delivery_gate")

REQUIRED_KEYS = ("id", "bucket", "category", "kind", "question", "pass_answer")
OPTIONAL_KEYS = ("binds_to",)
RELEASED_KEYS = REQUIRED_KEYS + OPTIONAL_KEYS


def bucket_of(kind):
    """The dimension a kind scores, or None if the kind is not in the taxonomy."""
    return KIND_BUCKET.get(kind)


def category_of(bucket):
    """The judge call a bucket is asked in."""
    return CATEGORY_BY_BUCKET.get(bucket)


def normalise_question(q, warn=None, where=""):
    """One question in released form, or None if it cannot be placed.

    `bucket` and `category` are recomputed from `kind`. A stored value that
    disagrees is overwritten and reported: the file is wrong, not the taxonomy.
    """
    if not isinstance(q, dict):
        return None
    qid = str(q.get("id") or "").strip()
    kind = str(q.get("kind") or "").strip()
    text = str(q.get("question") or "").strip()
    pass_answer = str(q.get("pass_answer") or "").strip().lower()

    def say(msg):
        if warn:
            warn(f"{where}{qid or '?'}: {msg}")

    if not qid or not text:
        say("missing id or question text")
        return None
    if pass_answer not in ("yes", "no"):
        say(f"pass_answer is {q.get('pass_answer')!r}, expected yes or no")
        return None
    bucket = bucket_of(kind)
    if bucket is None:
        say(f"kind {kind!r} is not in the taxonomy")
        return None
    category = category_of(bucket)

    if q.get("bucket") and q["bucket"] != bucket:
        say(f"stored bucket {q['bucket']!r} disagrees with kind {kind!r} "
            f"({bucket!r}); using {bucket!r}")
    if q.get("category") and q["category"] != category:
        say(f"stored category {q['category']!r} disagrees with bucket {bucket!r} "
            f"({category!r}); using {category!r}")

    out = {"id": qid, "bucket": bucket, "category": category, "kind": kind,
           "question": text, "pass_answer": pass_answer}
    binds_to = str(q.get("binds_to") or "").strip()
    if binds_to:
        out["binds_to"] = binds_to
    return out


def load_rubric(path, warn=None):
    """The usable questions in one checklist file, or [] if it is not one."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if warn:
            warn(f"{path.name}: unreadable ({exc})")
        return []
    if not isinstance(payload, dict):
        return []
    questions = payload.get("questions")
    if not isinstance(questions, list):
        return []

    out, dropped, seen = [], 0, set()
    for q in questions:
        norm = normalise_question(q, warn=warn, where=f"{path.name}: ")
        if norm is None:
            dropped += 1
            continue
        if norm["id"] in seen:
            if warn:
                warn(f"{path.name}: duplicate question id {norm['id']!r}")
            dropped += 1
            continue
        seen.add(norm["id"])
        out.append(norm)
    if dropped and warn:
        warn(f"{path.name}: dropped {dropped} of {len(questions)} question(s)")
    return out


def load_rubrics(rubric_dir, sids, warn=None):
    """{sid: [question, ...]} for the sids that actually have a checklist."""
    rubric_dir = Path(rubric_dir)
    out = {}
    for sid in sids:
        path = rubric_dir / f"{sid}.json"
        if not path.exists():
            continue
        questions = load_rubric(path, warn=warn)
        if questions:
            out[sid] = questions
    return out
