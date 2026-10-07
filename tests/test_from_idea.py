#!/usr/bin/env python3
"""Tests for the one-line idea pipeline.

Run with pytest, or directly:   python3 tests/test_from_idea.py

The pipeline reconstructs the two inputs AesCode was trained on from a single
sentence. Every assertion here pins a property that would otherwise fail
quietly: a spec that validates but carries the wrong canvas, a prompt that
reaches the model without the training-time template, or a retry loop that
spins instead of giving up. All network calls are stubbed.
"""
import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from inference.prompts import DEFAULT_USER_TEMPLATE, build_user_text  # noqa: E402

from_idea = __import__("inference.from_idea", fromlist=["x"])


def _png(path: Path, size):
    from PIL import Image

    Image.new("RGB", size, (250, 250, 250)).save(path)
    return path


TITLE = "European EV Adoption 2026"


def _spec(prompt_chars=1200, requirements=None):
    # text_exact only survives if the string is verbatim in generation_prompt.
    body = f'Landscape dashboard headlined "{TITLE}" with a quarterly bar chart. '
    return {
        "id": "idea_0001",
        "reference_metadata": {},
        "generation_prompt": body + "Three KPI cards sit in the left rail. " * (
            prompt_chars // 38 + 1
        ),
        "verifiable_requirements": requirements if requirements is not None else [
            {"id": "req_title_1", "category": "text_exact",
             "expected": {"text": TITLE}},
            {"id": "req_chart_1", "category": "chart_data",
             "target": {"chart_type": "echarts", "chart_subtype": "bar"},
             "expected": {"data": [120, 185, 240, 310]}},
            {"id": "req_table_1", "category": "table_data",
             "expected": {"data": [["Region", "Units"], ["North", 1200]]}},
        ],
    }


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(from_idea.time, "sleep", lambda _s: None)


def test_spec_canvas_follows_the_actual_image(tmp_path, monkeypatch, no_sleep):
    """Canvas metadata must come from the image on disk, not the requested size."""
    image = _png(tmp_path / "ref.png", (1024, 1536))
    monkeypatch.setattr(from_idea, "call_spec_model", lambda **_k: json.dumps(_spec()))

    spec = from_idea.build_spec(image, provider="openai", model="m")
    meta = spec["reference_metadata"]

    assert meta["reference_image_wh"] == "1024x1536"
    assert meta["reference_orientation"] == "portrait"
    assert meta["reference_image_path"] == str(image)
    # 1024/1536 rounded by spec.io
    assert meta["reference_aspect_ratio"] == pytest.approx(0.6667, abs=1e-3)


def test_requirements_survive_schema_validation(tmp_path, monkeypatch, no_sleep):
    """text_exact, chart_data and table_data must all pass the released filter."""
    image = _png(tmp_path / "ref.png", (1536, 1024))
    monkeypatch.setattr(from_idea, "call_spec_model", lambda **_k: json.dumps(_spec()))

    spec = from_idea.build_spec(image, provider="openai", model="m")
    kinds = {r["category"] for r in spec["verifiable_requirements"]}

    assert kinds == {"text_exact", "chart_data", "table_data"}


def test_fenced_json_is_accepted(tmp_path, monkeypatch, no_sleep):
    """Models commonly wrap JSON in a code fence; that must not fail the run."""
    image = _png(tmp_path / "ref.png", (1536, 1024))
    fenced = "```json\n" + json.dumps(_spec()) + "\n```"
    monkeypatch.setattr(from_idea, "call_spec_model", lambda **_k: fenced)

    assert from_idea.build_spec(image, provider="openai", model="m")["generation_prompt"]


def test_thin_spec_is_retried_exactly_once(tmp_path, monkeypatch, no_sleep):
    """A thin first answer costs one extra call, never an unbounded loop."""
    image = _png(tmp_path / "ref.png", (1536, 1024))
    thin = dict(_spec(), generation_prompt="Too short.", verifiable_requirements=[])
    replies = [json.dumps(thin), json.dumps(thin)]
    calls = []

    def fake(**_kwargs):
        calls.append(1)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(from_idea, "call_spec_model", fake)
    spec = from_idea.build_spec(image, provider="openai", model="m")

    assert len(calls) == 2
    assert spec["generation_prompt"] == "Too short."


def test_unparseable_output_raises_after_retries(tmp_path, monkeypatch, no_sleep):
    image = _png(tmp_path / "ref.png", (1536, 1024))
    calls = []

    def fake(**_kwargs):
        calls.append(1)
        return "I cannot help with that."

    monkeypatch.setattr(from_idea, "call_spec_model", fake)
    with pytest.raises(RuntimeError, match="Could not build a spec"):
        from_idea.build_spec(image, provider="openai", model="m", retries=2)

    assert len(calls) == 3  # initial attempt plus two retries


def test_run_writes_all_artifacts_and_uses_the_training_template(tmp_path, monkeypatch, no_sleep):
    """The model must receive the template SFT, RL and eval all build."""
    stub = types.ModuleType("inference.providers.image_api")
    stub.build_image_prompt = lambda idea, orientation: f"{orientation}::{idea}"
    stub.load_client = lambda **_k: None
    stub.generate_reference_image = (
        lambda _c, _m, _p, out_path, **_k: _png(out_path, (1536, 1024))
    )
    monkeypatch.setitem(sys.modules, "inference.providers.image_api", stub)
    monkeypatch.setattr(from_idea, "call_spec_model", lambda **_k: json.dumps(_spec()))

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return "```html\n<!DOCTYPE html>\n<html><body>ok</body></html>\n```"

    monkeypatch.setattr(from_idea, "generate", fake_generate)

    outdir = tmp_path / "out"
    from_idea.run(argparse.Namespace(
        idea="an EV dashboard", outdir=outdir, orientation="landscape",
        reference_image=None, spec_file=None, image_model="gpt-image-2",
        image_base_url=None, spec_provider="openai", spec_model="m", spec_base_url=None,
        spec_max_tokens=4000, retries=3, provider="openai", model="microsoft/AesCode-32B",
        base_url=None, max_tokens=12000, temperature=0.8, top_p=0.95,
        skip_generate=False,
    ))

    assert {p.name for p in outdir.iterdir()} == {
        "reference.png", "spec.json", "preview.txt", "page.html"
    }
    assert outdir.joinpath("page.html").read_text().startswith("<!DOCTYPE html>")

    user_text = captured["user_text"]
    assert "[Canvas]" in user_text and "width: 1536px" in user_text
    assert "120" in user_text  # chart data contract inlined
    assert captured["image"].name == "reference.png"

    saved = json.loads(outdir.joinpath("spec.json").read_text())
    assert user_text == build_user_text(saved, DEFAULT_USER_TEMPLATE, has_image=True)


def test_skip_generate_stops_before_the_model(tmp_path, monkeypatch, no_sleep):
    image = _png(tmp_path / "ref.png", (1536, 1024))
    monkeypatch.setattr(from_idea, "call_spec_model", lambda **_k: json.dumps(_spec()))

    def explode(**_kwargs):
        raise AssertionError("generate() must not run under --skip-generate")

    monkeypatch.setattr(from_idea, "generate", explode)

    outdir = tmp_path / "out"
    from_idea.run(argparse.Namespace(
        idea=None, outdir=outdir, orientation="landscape", reference_image=image,
        spec_file=None, image_model="gpt-image-2", image_base_url=None,
        spec_provider="openai", spec_model="m", spec_base_url=None,
        spec_max_tokens=4000, retries=3, provider="openai", model="m", base_url=None,
        max_tokens=12000, temperature=0.8, top_p=0.95, skip_generate=True,
    ))

    assert {p.name for p in outdir.iterdir()} == {"spec.json", "preview.txt"}
    assert outdir.joinpath("preview.txt").read_text().startswith("Create one complete HTML")


def test_edited_facts_reach_the_model(tmp_path, monkeypatch, no_sleep):
    """The whole point of writing spec.json out: edits must take effect."""
    image = _png(tmp_path / "reference.png", (1536, 1024))
    spec = _spec()
    spec["reference_metadata"] = {
        "reference_image_wh": "1536x1024", "reference_image_path": str(image)
    }
    spec_path = tmp_path / "spec.json"

    # The user renames the headline in both coupled places and rewrites the data.
    spec["generation_prompt"] = spec["generation_prompt"].replace(TITLE, "EU EV Adoption 2027")
    spec["verifiable_requirements"][0]["expected"]["text"] = "EU EV Adoption 2027"
    spec["verifiable_requirements"][1]["expected"]["data"] = [999, 888, 777]
    spec_path.write_text(json.dumps(spec, indent=2))

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return "<!DOCTYPE html><html></html>"

    monkeypatch.setattr(from_idea, "generate", fake_generate)

    def explode(**_kwargs):
        raise AssertionError("--spec-file must not call the spec model")

    monkeypatch.setattr(from_idea, "call_spec_model", explode)

    from_idea.run(argparse.Namespace(
        idea=None, outdir=tmp_path / "out", orientation="landscape",
        reference_image=None, spec_file=spec_path, image_model="gpt-image-2",
        image_base_url=None, spec_provider="openai", spec_model="m", spec_base_url=None,
        spec_max_tokens=4000, retries=3, provider="openai", model="m", base_url=None,
        max_tokens=12000, temperature=0.8, top_p=0.95, skip_generate=False,
    ))

    user_text = captured["user_text"]
    assert "EU EV Adoption 2027" in user_text
    assert TITLE not in user_text
    assert "[999, 888, 777]" in user_text
    assert "310" not in user_text          # the drafted numbers are gone
    assert captured["image"] == image      # reference recovered from the spec


def test_editing_one_half_of_a_coupled_pair_is_reported(tmp_path, capsys, no_sleep):
    """Renaming the prompt but not text_exact must not pass silently."""
    image = _png(tmp_path / "reference.png", (1536, 1024))
    spec = _spec()
    spec["reference_metadata"] = {
        "reference_image_wh": "1536x1024", "reference_image_path": str(image)
    }
    spec["verifiable_requirements"][0]["expected"]["text"] = "A Headline Never Written Down"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    result = from_idea.load_edited_spec(spec_path)

    assert "req_title_1" not in {r["id"] for r in result["verifiable_requirements"]}
    assert "not in generation_prompt verbatim" in capsys.readouterr().err


def test_malformed_spec_fails_with_a_readable_message(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text('{"id": "x", oops}')

    with pytest.raises(SystemExit, match="not valid JSON"):
        from_idea.load_edited_spec(spec_path)


def test_detailed_spec_prompts_keep_the_schema_placeholder():
    """build_spec() formats the template; a dropped placeholder breaks it."""
    from spec.prompting import build_detailed_spec_prompts

    system, template = build_detailed_spec_prompts()

    assert "{schema_contract}" in template
    assert "ECharts" in system
    for field in ("{id}", "{reference_image_wh}", "{reference_aspect_ratio}",
                  "{reference_orientation}"):
        assert field in template


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
