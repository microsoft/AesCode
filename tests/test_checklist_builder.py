#!/usr/bin/env python3
"""Tests for the checklist builder's quotas and contract linter.

Run with pytest, or directly:   python3 tests/test_checklist_builder.py

The quota planner and the linter are the two parts of the builder that run
without a model, and both fail quietly: a drifted quota produces a checklist that
looks fine but scores a different mix of dimensions, and a linter that stops
matching produces a clean report on a bad checklist.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "data_prep"))

import build_graph_rubric as builder  # noqa: E402
import rubric_io  # noqa: E402


def graph(node_types, edge_types, counts=None):
    counts = counts or {}
    return {
        "nodes": [{"id": f"n{i}", "type": t,
                   "attrs": {"count": counts[i]} if i in counts else {}}
                  for i, t in enumerate(node_types)],
        "edges": [{"id": f"e{i}", "type": t} for i, t in enumerate(edge_types)],
    }


RICH = graph(
    ["region", "banner", "text", "group", "card", "card", "chart", "table",
     "icon", "icon", "panel", "text", "divider", "image"],
    ["contains", "contains", "left_of", "above", "evenly_spaced",
     "aligned_with", "connects", "timeline_step", "parent_of"],
    counts={3: 4})

FLAT = graph(["text", "text", "text"], [])


# ---------------------------------------------------------------------------
# quotas
# ---------------------------------------------------------------------------

def test_total_stays_in_the_corpus_range():
    for g in (FLAT, RICH,
              graph(["panel"] * 30 + ["chart"] * 10, ["contains"] * 25)):
        plan = builder.plan_quotas(g)
        total = sum(sum(v.values()) for v in plan.values())
        assert builder.MIN_QUESTIONS <= total <= builder.MAX_QUESTIONS, total


def test_fixed_quotas_are_exact():
    """page_balance once, delivery_gate five times, whatever the graph."""
    for g in (FLAT, RICH):
        plan = builder.plan_quotas(g)
        assert plan["Layout"]["page_balance"] == 1
        assert plan["Style"]["delivery_gate"] == 5


def test_bucket_mix_tracks_the_corpus():
    """18 / 43 / 39 Content / Layout / Style, within two points."""
    plan = builder.plan_quotas(RICH)
    total = sum(sum(v.values()) for v in plan.values())
    for bucket, want in builder.BUCKET_MIX.items():
        got = sum(plan[bucket].values()) / total
        assert abs(got - want) < 0.02, (bucket, got, want)


def test_mean_length_is_near_the_corpus_mean():
    """A typical graph should land near 35.8, not at either clamp."""
    total = sum(sum(v.values()) for v in builder.plan_quotas(RICH).values())
    assert 32 <= total <= 40, total


def test_every_bucket_is_populated_even_on_a_flat_page():
    plan = builder.plan_quotas(FLAT)
    for bucket in rubric_io.BUCKETS:
        assert sum(plan[bucket].values()) > 0, bucket


def test_unsupported_kinds_are_not_authored():
    """A flat page with no edges gets no relation, connective or subgraph work."""
    plan = builder.plan_quotas(FLAT)
    assert "connective" not in plan["Layout"]
    assert "subgraph" not in plan["Layout"]
    assert "containment" not in plan["Layout"]
    assert "relation" not in plan["Layout"]
    assert "icon_quality" not in plan["Style"]
    assert "element_count" not in plan["Content"]


def test_supported_kinds_turn_on_with_the_material():
    ok = builder.supported_kinds(RICH)
    for kind in ("connective", "containment", "relation", "subgraph",
                 "icon_quality", "element_quality", "element_count"):
        assert kind in ok, kind


def test_design_labelled_is_never_authored():
    """It is a provenance label, not a way of looking at a page."""
    for g in (FLAT, RICH):
        assert "design_labelled" not in builder.plan_quotas(g)["Style"]
    assert "design_labelled" not in builder.supported_kinds(RICH)
    # but it still loads, because released checklists carry it
    assert rubric_io.bucket_of("design_labelled") == "Style"


def test_quota_kinds_are_all_in_the_taxonomy():
    for bucket, plan in builder.plan_quotas(RICH).items():
        for kind in plan:
            assert rubric_io.bucket_of(kind) == bucket, (kind, bucket)


def test_apportionment_sums_exactly():
    for total in range(0, 60):
        parts = builder._apportion(total, {"a": 3, "b": 5, "c": 1})
        assert sum(parts.values()) == total, (total, parts)


# ---------------------------------------------------------------------------
# the contract linter
# ---------------------------------------------------------------------------

LINT_GRAPH = {"graph": {"nodes": [{"id": "n_hero", "type": "banner"},
                                  {"id": "n_cards", "type": "group"}],
                        "edges": [{"id": "e_above", "type": "above"}]}}


def codes(question):
    """Per-question flag codes only; whole-checklist quota flags use id '-'."""
    report = builder.lint_rubric([question], LINT_GRAPH, {})
    return {f["code"] for f in report["flags"] if f["id"] != "-"}


def q(**kw):
    base = {"id": "x", "kind": "structure", "binds_to": "node:n_hero",
            "question": "Are the four cards arranged in a single row?",
            "pass_answer": "yes"}
    base.update(kw)
    return base


def test_clean_question_raises_nothing():
    assert codes(q()) == set()


def test_c1_undecidable():
    assert "C1_not_decidable" in codes(q(question="Overlap?"))


def test_c2_binding():
    assert "C2_missing_binding" in codes(q(binds_to=""))
    assert "C2_bad_binding_kind" in codes(q(binds_to="n_hero"))
    assert "C2_unknown_binding" in codes(q(binds_to="node:n_nope"))
    assert "C2_page_kind_bound" in codes(
        q(kind="page_balance", binds_to="node:n_hero"))


def test_c3_unstated_threshold():
    assert "C3_unstated_threshold" in codes(
        q(question="Is the hero banner appropriately styled?"))
    # the same question with a threshold is fine
    assert "C3_unstated_threshold" not in codes(
        q(question="Is the hero banner at least twice the height of a card?"))


def test_c4_polarity():
    assert "C4_bad_polarity" in codes(q(pass_answer="maybe"))


def test_c5_kind():
    assert "C5_unknown_kind" in codes(q(kind="nonsense"))
    assert "C5_missing_kind" in codes(q(kind=""))


def test_c6_placeholder_content():
    assert "C6_image_content_question" in codes(
        q(question="Does the photo depict a smiling presenter?"))


def test_c7_implementation():
    assert "C7_prescribes_implementation" in codes(
        q(question="Is the banner built as a flex row with a z-index above the body?"))


def test_c8_browser_default():
    assert "C8_browser_default" in codes(
        q(question="Is the heading text larger than the body text?"))


def test_c9_atomicity_and_the_delivery_gate_exemption():
    bundled = "Is the hero above the cards and is the footer below them?"
    assert "C9_not_atomic" in codes(q(kind="collision", question=bundled))
    # the delivery gate is a disqualifier list by design and is exempt
    assert "C9_not_atomic" not in codes(
        q(kind="delivery_gate", binds_to="", question=bundled))


def test_duplicate_questions_are_caught():
    report = builder.lint_rubric([q(id="a"), q(id="b")], LINT_GRAPH, {})
    assert any(f["code"] == "C9_duplicate" for f in report["flags"])


def test_quota_violations_are_reported():
    plan = builder.plan_quotas(RICH)
    report = builder.lint_rubric([q()], LINT_GRAPH, plan)
    found = {f["code"] for f in report["flags"]}
    assert "Q1_page_balance_quota" in found
    assert "Q1_delivery_gate_quota" in found
    assert "Q2_question_count_out_of_range" in found
    assert "Q3_empty_bucket" in found


def test_bound_to_own_graph_is_reported():
    report = builder.lint_rubric(
        [q(binds_to="node:n_hero"), q(id="y", binds_to="node:n_nope")],
        LINT_GRAPH, {})
    assert report["bound_to_own_graph"] == 0.5


def test_released_output_is_in_schema():
    payload = builder.released("sid1", [
        q(kind="structure"), q(id="y", kind="delivery_gate", binds_to=""),
        q(id="z", kind="bogus")])
    assert payload["sid"] == "sid1"
    assert len(payload["questions"]) == 2, "the bogus kind should be dropped"
    for question in payload["questions"]:
        assert set(question) <= set(rubric_io.RELEASED_KEYS)
        assert question["bucket"] == rubric_io.bucket_of(question["kind"])
        assert question["category"] == rubric_io.category_of(question["bucket"])


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
