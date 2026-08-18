"""`create_task`'s routing: template-first, LLM fallback only when the template does not
cover the spec or fails verification (Phase 13 §7.2).

A harness failure or a review rejection on the template path is not an error — it is the
signal that a spec is outside the template's declared space — so it falls through to the
LLM path exactly once, carrying the harness report/findings as opening feedback, rather
than retrying a deterministic renderer that would fail identically every time.
"""

import uuid

import pytest

from adaptrna_agentic.agents.verifier import Review
from adaptrna_agentic.codegen import pipeline
from fixtures import broken_task_sources as sources


class ScriptedStructuredModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


class ExplodingModel:
    """Fails loudly if the template path ever calls a model — that is the whole point of
    'no model call' on the declared case."""

    def with_structured_output(self, schema):
        raise AssertionError("a model was invoked on the template path")

    def invoke(self, messages):
        raise AssertionError("a model was invoked on the template path")


def generated(files: dict):
    from adaptrna_agentic.agents.toolsmith import GeneratedFile, GeneratedTask

    return GeneratedTask(
        files=[GeneratedFile(filename=k, content=v) for k, v in files.items()],
        notes="",
    )


def unique_name():
    return f"gen_{uuid.uuid4().hex[:6]}"


@pytest.fixture
def dataset(tmp_path):
    """A directory holding both shapes the two paths' code actually reads: a single flat
    `data.csv` for the approved spec (and the template's own datamodule), and a
    `train.csv`/`val.csv` split for `sources.GOOD_DATAMODULE` — the scripted fallback
    tests' stand-in for "some working generated code", which follows the older
    directory-based convention broken_task_sources.py was written against."""
    root = tmp_path / "data"
    root.mkdir()
    rows = ["sequence,label"] + [f"{'ACGU' * 8},{i % 2}" for i in range(40)]
    (root / "data.csv").write_text("\n".join(rows) + "\n")
    (root / "train.csv").write_text("\n".join(rows) + "\n")
    (root / "val.csv").write_text("\n".join(rows[:8]) + "\n")
    return root / "data.csv"


@pytest.fixture
def covered_spec(dataset):
    """A spec `covers()` accepts: the template renders and verifies it with no model
    call at all."""

    def make(task_name):
        return {
            "spec_version": 1,
            "source": "confirm_data_profile",
            "path": str(dataset),
            "format": {"separator": ",", "compression": None},
            "sequence_column": "sequence",
            "label_column": "label",
            "target_type": "binary",
            "classes": ["0", "1"],
            "positive_class": "1",
            "head": {"primary_metric": "test/f1_score"},
            "split": {
                "mode": "random", "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 42, "stratify": True,
            },
            "task_name": task_name,
            "tool_description": "a spec the template covers",
        }

    return make


@pytest.fixture
def uncovered_spec(dataset):
    """No positive_class: `covers()` rejects a binary spec without one (D5/Fix 5)."""

    def make(task_name):
        return {
            "spec_version": 1,
            "source": "confirm_data_profile",
            "path": str(dataset),
            "format": {"separator": ",", "compression": None},
            "sequence_column": "sequence",
            "label_column": "label",
            "target_type": "binary",
            "classes": ["0", "1"],
            "head": {"primary_metric": "test/f1_score"},
            "split": {
                "mode": "random", "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 42, "stratify": True,
            },
            "task_name": task_name,
            "tool_description": "a spec the template does not cover",
        }

    return make


def files_for(task_name, dataset, source_fn=sources.good_task):
    return {
        "task.py": source_fn(task_name),
        "datamodule.py": sources.GOOD_DATAMODULE,
        "config.yaml": sources.CONFIG_TEMPLATE.format(task_name=task_name, data_root=dataset.parent),
    }


# ---------------------------------------------------------------- the template path

def test_the_template_path_renders_and_passes_with_no_generation_model_call(tmp_path, covered_spec):
    """D13: rendering itself needs no model. `toolsmith_model` would raise if the
    template path ever asked it to write code; skip_review keeps this test focused on
    that one claim (the review step's own model call is proven separately below, per
    §7.4 — "the identical harness, then the identical review" on both paths)."""
    name = unique_name()

    result = pipeline.create_task(
        covered_spec(name), toolsmith_model=ExplodingModel(), skip_review=True,
        data_dir=tmp_path / "hub",
    )

    assert result.ok, result.to_dict()
    assert result.path == "template"
    assert result.fell_back_from_template is False
    assert len(result.attempts) == 1
    assert result.attempts[0].index == 0


def test_the_template_path_still_runs_the_independent_review(tmp_path, covered_spec):
    """§7.4: the review is not a shortcut skipped because "the template is known good" —
    it runs on both paths, just with the narrower rendered=True framing."""
    name = unique_name()
    critic = ScriptedStructuredModel([Review(approved=True, findings=[])])

    result = pipeline.create_task(
        covered_spec(name), toolsmith_model=ExplodingModel(), verifier_model=critic,
        data_dir=tmp_path / "hub",
    )

    assert result.ok, result.to_dict()
    assert critic.calls                                   # the reviewer really ran
    assert "rendered deterministically" in critic.calls[0][-1]["content"]


def test_landed_spec_json_records_the_template_version(tmp_path, covered_spec):
    """`toolhub doctor` needs this to tell a stale template render apart from a task the
    LLM path produced, which carries no template version at all (plan §7.3)."""
    import json

    from adaptrna_agentic.codegen.templates.render import TEMPLATE_VERSION

    name = unique_name()

    result = pipeline.create_task(
        covered_spec(name), toolsmith_model=ExplodingModel(), skip_review=True,
        data_dir=tmp_path / "hub",
    )

    assert result.ok
    landed = json.loads(result.stage.files[f"adaptrna_custom/tasks/{name}/spec.json"])
    assert landed["template_version"] == TEMPLATE_VERSION


def test_the_staged_result_names_which_path_produced_it(tmp_path, covered_spec):
    """A user must never be unsure whether a human wrote the logic they are approving."""
    name = unique_name()

    result = pipeline.create_task(
        covered_spec(name), toolsmith_model=ExplodingModel(), skip_review=True,
        data_dir=tmp_path / "hub",
    )

    payload = result.to_dict()
    assert payload["path"] == "template"
    assert "fell_back_from_template" not in payload


# ---------------------------------------------------------------- uncovered -> straight to LLM

def test_a_spec_covers_rejects_goes_straight_to_the_llm_path(tmp_path, dataset, uncovered_spec):
    name = unique_name()
    smith = ScriptedStructuredModel([generated(files_for(name, dataset))])

    result = pipeline.create_task(
        uncovered_spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    assert result.ok
    assert result.path == "generated"
    # Never attempted the template at all -- this is not a fallback, it never tried.
    assert result.fell_back_from_template is False
    assert smith.calls
    assert all(attempt.index >= 1 for attempt in result.attempts)


def test_generated_spec_json_carries_no_template_version(tmp_path, dataset, uncovered_spec):
    """Nothing here was rendered, so nothing here claims a template version."""
    import json

    name = unique_name()
    smith = ScriptedStructuredModel([generated(files_for(name, dataset))])

    result = pipeline.create_task(
        uncovered_spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    assert result.ok
    landed = json.loads(result.stage.files[f"adaptrna_custom/tasks/{name}/spec.json"])
    assert "template_version" not in landed


# ---------------------------------------------------------------- fallback on failure

def test_a_harness_failure_on_rendered_code_falls_through_not_retries(
    tmp_path, dataset, covered_spec, monkeypatch
):
    import adaptrna_agentic.codegen.templates.render as render_module

    name = unique_name()
    monkeypatch.setattr(render_module, "covers", lambda spec: True)
    monkeypatch.setattr(render_module, "render", lambda spec: {
        "task.py": "this is not valid python(",
        "datamodule.py": sources.GOOD_DATAMODULE,
        "config.yaml": sources.CONFIG_TEMPLATE.format(task_name=name, data_root=dataset.parent),
    })

    smith = ScriptedStructuredModel([generated(files_for(name, dataset))])

    result = pipeline.create_task(
        covered_spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    assert result.ok                              # the fallback recovered
    assert result.path == "generated"
    assert result.fell_back_from_template is True
    assert "harness" in result.fallback_reason.lower()
    assert smith.calls                             # the model really was invoked

    template_attempts = [a for a in result.attempts if a.index == 0]
    assert len(template_attempts) == 1              # exactly one attempt -- never retried
    assert template_attempts[0].ok is False


def test_a_review_rejection_on_the_template_path_falls_through(
    tmp_path, dataset, covered_spec, monkeypatch
):
    """The template's own output can pass the harness and still be rejected by review —
    e.g. the reviewer decides the recipe does not actually fit this data."""
    import adaptrna_agentic.codegen.templates.render as render_module

    name = unique_name()
    monkeypatch.setattr(render_module, "covers", lambda spec: True)

    critic = ScriptedStructuredModel([
        Review(approved=False, findings=["this data does not fit the recipe"]),
    ])
    smith = ScriptedStructuredModel([generated(files_for(name, dataset))])

    result = pipeline.create_task(
        covered_spec(name), toolsmith_model=smith, verifier_model=critic,
        data_dir=tmp_path / "hub",
    )

    assert result.fell_back_from_template is True
    assert "review" in result.fallback_reason.lower()
    template_attempts = [a for a in result.attempts if a.index == 0]
    assert len(template_attempts) == 1
    assert template_attempts[0].ok is False
    assert template_attempts[0].review["findings"] == ["this data does not fit the recipe"]


def test_the_verifier_is_told_the_code_was_rendered_on_the_template_path(covered_spec):
    from adaptrna_agentic.codegen import prompts

    spec = covered_spec("t")
    rendered_prompt = prompts.verifier_user_prompt("d", spec, {"task.py": "x"}, "PASS", rendered=True)
    generated_prompt = prompts.verifier_user_prompt("d", spec, {"task.py": "x"}, "PASS", rendered=False)

    assert "rendered deterministically" in rendered_prompt
    assert "rendered deterministically" not in generated_prompt
