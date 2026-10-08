#!/usr/bin/env python3
"""Tests for the aggregation rules the main table depends on.

Run with pytest, or directly:   python3 tests/test_aggregation.py

These pin the rules that are load-bearing for the published numbers and that
would otherwise fail silently. An aggregation bug does not raise; it produces a
plausible table, so the only way to notice is to check the arithmetic against
cases whose answer is known by hand.
"""
import json
import pathlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "data_prep"))

import build_table  # noqa: E402
import rubric_io  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def question(qid, kind, pass_answer="yes"):
    bucket = rubric_io.bucket_of(kind)
    return {"id": qid, "kind": kind, "bucket": bucket,
            "category": rubric_io.category_of(bucket),
            "question": f"{qid}?", "pass_answer": pass_answer}


def write_run(root, rule_rows, judgment_rows, block="block_final300"):
    """A run directory with the two files the table reads."""
    run = root / "demo_final300"
    (run / "results").mkdir(parents=True, exist_ok=True)
    (run / block).mkdir(parents=True, exist_ok=True)
    with (run / "results/validation_report.jsonl").open("w") as f:
        for sid, k, per_validator in rule_rows:
            f.write(json.dumps({"sample_id": sid,
                                "html_path": f"html/{sid}_k{k}.html",
                                "per_validator": per_validator}) + "\n")
    with (run / block / "judgments.jsonl").open("w") as f:
        for row in judgment_rows:
            f.write(json.dumps(row) + "\n")
    return run


def scored(rubrics, rule_rows, judgment_rows, n):
    root = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
    try:
        run = write_run(root, rule_rows, judgment_rows)
        return build_table.score_run(run, rubrics, n=n, block="block_final300")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def approx(a, b, tol=1e-9):
    return a is not None and abs(a - b) < tol


def answers_all_correct(qs):
    return {q["id"]: q["pass_answer"] for q in qs}


# ---------------------------------------------------------------------------
# Paper Appendix E.1: "the two prompts with no Content question have that
# dimension excluded rather than scored zero."
# ---------------------------------------------------------------------------

def test_bucket_with_no_questions_is_excluded_not_zeroed():
    """A sample whose checklist has no Content question leaves Content alone.

    Zero-filling would be the easy implementation and it is wrong: a sample that
    was never asked a Content question has not failed Content, and averaging a 0
    into the column drags every model down by the same amount while changing the
    gaps between them.
    """
    rubrics = {
        "s_full": [question("c1", "located_presence"),
                   question("l1", "structure"),
                   question("y1", "concrete_attribute")],
        "s_nocontent": [question("l1", "structure"),
                        question("y1", "concrete_attribute")],
    }
    judgments = [{"sid": sid, "k": k, "ok": True,
                  "answers": answers_all_correct(qs)}
                 for sid, qs in rubrics.items() for k in range(2)]
    dims, _ = scored(rubrics, [], judgments, n=2)

    assert approx(dims["Content"], 1.0), dims["Content"]
    assert approx(dims["Layout"], 1.0), dims["Layout"]
    assert approx(dims["Style"], 1.0), dims["Style"]
    # Had s_nocontent been zero-filled, Content would be 0.5 and Visual 0.833.
    assert approx(dims["Visual"], 1.0), dims["Visual"]


def test_zero_fill_would_have_changed_the_answer():
    """Guards the test above: the two conventions really do differ here."""
    rubrics = {
        "s_full": [question("c1", "located_presence"), question("l1", "structure")],
        "s_nocontent": [question("l1", "structure")],
    }
    judgments = [{"sid": sid, "k": 0, "ok": True,
                  "answers": answers_all_correct(qs)}
                 for sid, qs in rubrics.items()]
    dims, _ = scored(rubrics, [], judgments, n=1)
    assert approx(dims["Content"], 1.0)
    assert not approx(dims["Content"], 0.5)


def test_rule_dimension_applicability():
    """A validator that never reports on a sample does not score it zero.

    Same rule as the visual buckets, on the other half of the table: a page with
    no chart cannot fail the chart check.
    """
    rubrics = {"a": [question("c1", "located_presence")],
               "b": [question("c1", "located_presence")]}
    rule_rows = [
        ("a", 0, {"TextValidator": 1.0, "BoundaryValidator": 1.0,
                  "TableChartValidator": 0.0}),
        ("b", 0, {"TextValidator": 1.0, "BoundaryValidator": 1.0}),
    ]
    judgments = [{"sid": s, "k": 0, "ok": True, "answers": {"c1": "yes"}}
                 for s in rubrics]
    dims, coverage = scored(rubrics, rule_rows, judgments, n=1)
    assert coverage["samples_per_rule_dim"]["Chart"] == 1
    assert approx(dims["Chart"], 0.0), dims["Chart"]
    assert approx(dims["Text"], 1.0), dims["Text"]


def test_missing_rollout_scores_zero_without_leaving_the_denominator():
    """An unjudged rollout of a judged sample is a 0, not an absence.

    Dropping it would reward a model whose worst pages are the ones that crash
    the renderer.
    """
    rubrics = {"a": [question("c1", "located_presence")]}
    judgments = [{"sid": "a", "k": 0, "ok": True, "answers": {"c1": "yes"}},
                 {"sid": "a", "k": 1, "ok": False, "err": "no_png"}]
    dims, coverage = scored(rubrics, [], judgments, n=2)
    assert coverage["rollouts_judged"] == 1
    assert coverage["rollouts_expected"] == 2
    assert approx(dims["Content"], 0.5), dims["Content"]


def test_unclear_counts_as_failure():
    rubrics = {"a": [question("c1", "located_presence"),
                     question("c2", "located_presence")]}
    judgments = [{"sid": "a", "k": 0, "ok": True,
                  "answers": {"c1": "yes", "c2": "unclear"}}]
    dims, _ = scored(rubrics, [], judgments, n=1)
    assert approx(dims["Content"], 0.5), dims["Content"]


def test_pass_answer_no_is_scored_against_no():
    """Half the point of storing polarity: a defect probe passes by answering no."""
    rubrics = {"a": [question("c1", "located_presence", pass_answer="no")]}
    judgments = [{"sid": "a", "k": 0, "ok": True, "answers": {"c1": "no"}}]
    dims, _ = scored(rubrics, [], judgments, n=1)
    assert approx(dims["Content"], 1.0), dims["Content"]


def test_rollouts_beyond_n_are_ignored():
    """N is a knob, not a description of what happens to be on disk."""
    rubrics = {"a": [question("c1", "located_presence")]}
    judgments = [{"sid": "a", "k": 0, "ok": True, "answers": {"c1": "yes"}},
                 {"sid": "a", "k": 1, "ok": True, "answers": {"c1": "no"}}]
    dims, coverage = scored(rubrics, [], judgments, n=1)
    assert coverage["rollouts_judged"] == 1
    assert approx(dims["Content"], 1.0), dims["Content"]


def test_overall_is_the_mean_of_rule_and_visual():
    """Not the mean of six columns: Rule and Visual carry equal weight."""
    rubrics = {"a": [question("c1", "located_presence"),
                     question("l1", "structure"),
                     question("y1", "concrete_attribute")]}
    rule_rows = [("a", 0, {"TextValidator": 0.0, "BoundaryValidator": 0.0,
                           "TableChartValidator": 0.0})]
    judgments = [{"sid": "a", "k": 0, "ok": True,
                  "answers": answers_all_correct(rubrics["a"])}]
    dims, _ = scored(rubrics, rule_rows, judgments, n=1)
    assert approx(dims["Rule"], 0.0)
    assert approx(dims["Visual"], 1.0)
    assert approx(dims["Overall"], 0.5), dims["Overall"]


# ---------------------------------------------------------------------------
# The taxonomy
# ---------------------------------------------------------------------------

def test_bucket_and_category_are_derived_from_kind():
    for bucket, kinds in rubric_io.KINDS_BY_BUCKET.items():
        for kind in kinds:
            assert rubric_io.bucket_of(kind) == bucket
        assert rubric_io.category_of(bucket) in (
            "completeness", "relation", "style_strict")
    assert len(rubric_io.KINDS) == len(set(rubric_io.KINDS)), "a kind in two buckets"


def test_table_columns_follow_the_taxonomy():
    assert build_table.VISUAL_DIMS == rubric_io.BUCKETS
    assert build_table.COLUMNS == ["Text", "Bound", "Chart", "Rule",
                                   "Content", "Layout", "Style",
                                   "Visual", "Overall"]


def test_stored_bucket_is_overridden_by_kind():
    warnings = []
    got = rubric_io.normalise_question(
        {"id": "q", "kind": "structure", "bucket": "Style",
         "category": "completeness", "question": "is it?", "pass_answer": "yes"},
        warn=warnings.append)
    assert got["bucket"] == "Layout" and got["category"] == "relation"
    assert len(warnings) == 2, warnings


def test_vestigial_keys_are_dropped():
    got = rubric_io.normalise_question(
        {"id": "q", "kind": "structure", "question": "is it?",
         "pass_answer": "yes", "qtype": "x", "dim": "y", "aspect": "z",
         "source": "generic", "hardened": True})
    assert set(got) == {"id", "bucket", "category", "kind", "question",
                        "pass_answer"}, got


def test_unknown_kind_is_dropped_not_guessed():
    assert rubric_io.normalise_question(
        {"id": "q", "kind": "whatever", "bucket": "Style",
         "question": "is it?", "pass_answer": "yes"}) is None


def test_well_formed_checklist_loads_clean():
    """A checklist meeting the documented contract survives the loader untouched."""
    questions = []
    plan = ([("located_presence", 4), ("element_count", 2), ("element_integrity", 4)]
            + [("structure", 6), ("collision", 4), ("containment", 2), ("page_balance", 1)]
            + [("concrete_attribute", 5), ("element_quality", 2), ("delivery_gate", 5)])
    i = 0
    for kind, count in plan:
        for _ in range(count):
            i += 1
            questions.append({
                "id": f"q{i}", "kind": kind,
                "bucket": rubric_io.bucket_of(kind),
                "category": rubric_io.category_of(rubric_io.bucket_of(kind)),
                "question": f"question {i}?", "pass_answer": "yes",
            })

    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "s1.json").write_text(json.dumps({"sid": "s1", "questions": questions}))
        # a bookkeeping file in the same directory must be skipped, not parsed
        (d / "_manifest.json").write_text(json.dumps({"questions": 35}))

        warnings = []
        rubrics = rubric_io.load_rubrics(d, ["s1", "_manifest"], warn=warnings.append)

    assert list(rubrics) == ["s1"], list(rubrics)
    assert not warnings, warnings
    loaded = rubrics["s1"]
    assert 26 <= len(loaded) <= 50, len(loaded)
    kinds = [q["kind"] for q in loaded]
    assert kinds.count("page_balance") == 1
    assert kinds.count("delivery_gate") == 5
    assert {q["bucket"] for q in loaded} == set(rubric_io.BUCKETS)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
