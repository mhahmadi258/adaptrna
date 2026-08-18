"""The bounded ToolSmith ⇄ Verifier loop, driven by scripted models.

No API key, no network: a fake structured-output model replays whatever files the test
wants ToolSmith to have "written", and the real harness verifies them for real.

Phase 13: `create_task` takes an approved DatasetSpec, not a name/description/profile
triple, and tries the deterministic template first (§7.2) — the routing itself is
`test_codegen_paths.py`'s job. Every spec here deliberately omits `positive_class`, which
`covers()` requires for a binary target (D5/Fix 5), so `covers(spec)` is False and these
tests always land on the ToolSmith/Verifier fallback loop this file is about.
"""

import uuid

import pytest

from adaptrna_agentic.agents.verifier import Review
from adaptrna_agentic.codegen import pipeline, staging
from adaptrna_agentic.codegen.discovery import CUSTOM_ROOT
from adaptrna_agentic.toolhub.errors import ToolHubError
from fixtures import broken_task_sources as sources


class ScriptedStructuredModel:
    """Replays structured results in order; records how many times it was asked."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


def generated(files: dict):
    from adaptrna_agentic.agents.toolsmith import GeneratedFile, GeneratedTask

    return GeneratedTask(
        files=[GeneratedFile(filename=k, content=v) for k, v in files.items()],
        notes="",
    )


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rows = ["sequence,label"] + [f"{'ACGU' * 8},{i % 2}" for i in range(8)]
    (root / "train.csv").write_text("\n".join(rows) + "\n")
    (root / "val.csv").write_text("\n".join(rows[:5]) + "\n")
    return root


@pytest.fixture
def spec(dataset):
    """A spec factory: every spec it builds is deliberately NOT covered by the template
    (no positive_class), so create_task() always falls straight to the fallback loop
    these tests exercise, without a model-free template attempt in the way."""

    def make(task_name, description="classify these sequences"):
        return {
            "spec_version": 1,
            "source": "confirm_data_profile",
            "path": str(dataset / "train.csv"),
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
            "tool_description": description,
        }

    return make


def files_for(task_name, dataset, source_fn=sources.good_task, datamodule=None):
    return {
        "task.py": source_fn(task_name),
        "datamodule.py": datamodule or sources.GOOD_DATAMODULE,
        "config.yaml": sources.CONFIG_TEMPLATE.format(task_name=task_name, data_root=dataset),
    }


def unique_name():
    return f"gen_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------- happy path

def test_first_attempt_succeeds_and_stages(tmp_path, dataset, spec):
    name = unique_name()
    smith = ScriptedStructuredModel([generated(files_for(name, dataset))])

    result = pipeline.create_task(
        spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    assert result.ok
    assert result.path == "generated"
    assert len(result.attempts) == 1
    assert result.stage is not None
    # Staged, NOT landed.
    assert not (CUSTOM_ROOT / "tasks" / name).exists()
    assert result.stage.package_dir.exists()

    payload = result.to_dict()
    assert payload["ok"] is True
    assert {f["path"] for f in payload["files"]} == {
        f"adaptrna_custom/tasks/{name}/{f}"
        for f in ("task.py", "datamodule.py", "config.yaml", "spec.json")
    }


def test_reviewer_can_reject_code_the_harness_passed(tmp_path, dataset, spec):
    name = unique_name()
    smith = ScriptedStructuredModel([generated(files_for(name, dataset))])
    critic = ScriptedStructuredModel([
        Review(approved=False, findings=["the loss does not match the target type"]),
        Review(approved=False, findings=["still wrong"]),
        Review(approved=False, findings=["still wrong"]),
    ])

    result = pipeline.create_task(
        spec(name), toolsmith_model=smith, verifier_model=critic, data_dir=tmp_path / "hub",
    )

    assert not result.ok
    assert len(result.attempts) == 3          # the harness passed; the reviewer did not
    assert result.stage is None
    assert "does not match" in result.to_dict()["history"][0]


# ---------------------------------------------------------------- convergence

def test_loop_converges_after_a_failed_attempt(tmp_path, dataset, spec):
    name = unique_name()
    smith = ScriptedStructuredModel([
        generated(files_for(name, dataset, sources.task_with_unsaved_state)),  # caught
        generated(files_for(name, dataset, sources.task_with_saved_state)),    # fixed
    ])

    result = pipeline.create_task(
        spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    assert result.ok
    assert len(result.attempts) == 2
    assert result.attempts[0].ok is False
    assert "adapter_roundtrip" in result.attempts[0].harness_report["failed"]

    # The second call carried the failure forward as feedback.
    second_prompt = smith.calls[1][-1]["content"]
    assert "previous attempt failed verification" in second_prompt
    assert "ADAPTER_EXTRA_PREFIXES" in second_prompt


def test_gives_up_after_three_attempts_and_writes_nothing(tmp_path, dataset, spec):
    name = unique_name()
    smith = ScriptedStructuredModel([
        generated(files_for(name, dataset, sources.task_with_unsaved_state))
    ])

    result = pipeline.create_task(
        spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    assert not result.ok
    assert len(result.attempts) == 3
    assert result.stage is None
    assert not (CUSTOM_ROOT / "tasks" / name).exists()
    assert "Gave up after 3" in result.to_dict()["conclusion"]


def test_incomplete_file_set_is_rejected_before_verification(tmp_path, dataset, spec):
    name = unique_name()
    smith = ScriptedStructuredModel([generated({"task.py": "x = 1"})])

    result = pipeline.create_task(
        spec(name), toolsmith_model=smith, skip_review=True,
        max_iterations=1, data_dir=tmp_path / "hub",
    )

    assert not result.ok
    assert "missing" in result.attempts[0].harness_report["checks"][0]["detail"]


def test_stray_paths_from_the_model_are_normalised(tmp_path, dataset, spec):
    name = unique_name()
    smith = ScriptedStructuredModel([generated({
        f"adaptrna_custom/tasks/{name}/task.py": sources.good_task(name),
        "datamodule.py": sources.GOOD_DATAMODULE,
        "./config.yaml": sources.CONFIG_TEMPLATE.format(task_name=name, data_root=dataset),
    })])

    result = pipeline.create_task(
        spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    assert result.ok, result.to_dict()


# ---------------------------------------------------------------- landing

def test_landing_writes_the_files_and_makes_the_task_importable(tmp_path, dataset, spec):
    from adaptrna_agentic.codegen.discovery import custom_task_names, load_all

    name = unique_name()
    smith = ScriptedStructuredModel([generated(files_for(name, dataset))])
    result = pipeline.create_task(
        spec(name), toolsmith_model=smith, skip_review=True, data_dir=tmp_path / "hub",
    )

    written = staging.land(result.stage)
    try:
        assert f"adaptrna_custom/tasks/{name}/task.py" in written
        assert f"adaptrna_custom/tasks/{name}/spec.json" in written
        assert name in custom_task_names()

        failures = load_all(only=[name])
        assert failures == []

        import rinalmo_hub.tasks  # noqa: F401
        from rinalmo_hub.registry import available_tasks

        assert name in available_tasks()
    finally:
        import shutil

        shutil.rmtree(CUSTOM_ROOT / "tasks" / name, ignore_errors=True)


def test_existing_task_name_is_never_silently_overwritten(tmp_path, dataset, spec, monkeypatch):
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.custom_task_names", lambda: ["already_here"]
    )

    with pytest.raises(ToolHubError, match="already exists"):
        pipeline.create_task(spec("already_here"), toolsmith_model=ScriptedStructuredModel([]))
