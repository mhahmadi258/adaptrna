"""The verification harness — controls and catches.

Controls: the shipped tasks must PASS. A harness that fails a known-good task is broken.
Catches: deliberately defective generated tasks must FAIL, one per failure mode. A harness
that passes everything is worse than no harness (plan §10).
"""

import uuid

import pytest

from adaptrna_agentic.codegen.harness import summarize, verify_task
from adaptrna_agentic.codegen.staging import stage_task
from fixtures import broken_task_sources as sources
from fixtures import target_type_tasks as target_types

HARNESS_TIMEOUT = 600

SEQS = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]


@pytest.fixture
def dataset(tmp_path):
    """A tiny sequence,label CSV dataset — the shape a generated datamodule reads."""
    root = tmp_path / "data"
    root.mkdir()
    rows = ["sequence,label"] + [
        f"{'ACGU' * 8},{i % 2}" for i in range(8)
    ]
    (root / "train.csv").write_text("\n".join(rows) + "\n")
    (root / "val.csv").write_text("\n".join(rows[:5]) + "\n")
    return root


def stage(tmp_path, dataset, task_source_fn, datamodule=None, name=None):
    """Stage a generated task exactly as the pipeline would, and verify from there."""
    task_name = name or f"gen_{uuid.uuid4().hex[:6]}"
    files = {
        "task.py": task_source_fn(task_name),
        "datamodule.py": datamodule or sources.GOOD_DATAMODULE,
        "config.yaml": sources.CONFIG_TEMPLATE.format(
            task_name=task_name, data_root=dataset
        ),
    }
    return task_name, stage_task(task_name, files, data_dir=tmp_path / "toolhub_data")


def stage_fixture_task(tmp_path, target_type: str):
    """Stage one of the hand-written, known-good fixture tasks (§target_type_tasks)."""
    info = target_types.TARGET_TYPES[target_type]

    data_root = tmp_path / "data"
    data_root.mkdir()
    train_text, val_text = target_types.split_train_val(info["csv"])
    (data_root / "train.csv").write_text(train_text)
    (data_root / "val.csv").write_text(val_text)

    task_name = f"fixture_{target_type}"
    files = {
        "task.py": info["task"](task_name),
        "datamodule.py": info["datamodule"],
        "config.yaml": info["config"](task_name, data_root),
    }
    return task_name, stage_task(task_name, files, data_dir=tmp_path / "toolhub_data")


def run(staged, task_name, **kwargs):
    return verify_task(
        task_name,
        task_module=staged.module_path,
        config_path=str(staged.config_path),
        sys_path=[str(staged.root)],
        sequences=SEQS,
        timeout=HARNESS_TIMEOUT,
        **kwargs,
    )


def failed(report) -> set:
    return {c["name"] for c in report["checks"] if c["status"] == "fail"}


def detail_of(report, check_name) -> str:
    return next(c["detail"] for c in report["checks"] if c["name"] == check_name)


# ---------------------------------------------------------------- controls

@pytest.mark.parametrize("target_type", ("binary", "multiclass", "regression"))
def test_fixture_tasks_pass_every_check(tmp_path, target_type):
    """The known-good control, one per supported target type (Phase 13, D1).

    Unlike the shipped-task control it replaces, this runs the *full* harness — not just
    the structural checks — because the fixture ships its own real data.
    """
    task_name, staged = stage_fixture_task(tmp_path, target_type)

    report = run(staged, task_name)

    assert report["ok"], summarize(report)
    assert failed(report) == set()


def test_a_correct_generated_task_passes_every_check(tmp_path, dataset):
    task_name, staged = stage(tmp_path, dataset, sources.good_task)

    report = run(staged, task_name)

    assert report["ok"], summarize(report)
    # The data-dependent checks really ran (they are the ones that catch datamodules).
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["datamodule"] == "pass"
    assert statuses["forward_backward"] == "pass"
    assert statuses["metrics"] == "pass"


def test_declared_extra_state_round_trips(tmp_path, dataset):
    """The positive control for the round-trip check: declared state survives."""
    task_name, staged = stage(tmp_path, dataset, sources.task_with_saved_state)

    report = run(staged, task_name)

    assert report["ok"], summarize(report)


# ---------------------------------------------------------------- catches

def test_unsaved_task_state_is_caught_by_the_round_trip(tmp_path, dataset):
    """THE check: prediction-affecting state missing from the adapter file."""
    task_name, staged = stage(tmp_path, dataset, sources.task_with_unsaved_state)

    report = run(staged, task_name)

    assert not report["ok"]
    assert "adapter_roundtrip" in failed(report)
    detail = detail_of(report, "adapter_roundtrip")
    assert "predictions changed" in detail
    assert "ADAPTER_EXTRA_PREFIXES" in detail        # names the fix


def test_head_ignoring_its_config_is_caught(tmp_path, dataset):
    task_name, staged = stage(tmp_path, dataset, sources.task_with_head_ignoring_config)

    report = run(staged, task_name)

    assert not report["ok"]
    assert "config_head" in failed(report)
    assert "hidden_dim" in detail_of(report, "config_head")


def test_wrong_extract_features_is_caught(tmp_path, dataset):
    task_name, staged = stage(tmp_path, dataset, sources.task_with_bad_extract_features)

    report = run(staged, task_name)

    assert not report["ok"]
    assert failed(report) & {"forward_backward", "adapter_roundtrip", "serving"}


def test_datamodule_that_does_not_match_the_data_is_caught(tmp_path, dataset):
    task_name, staged = stage(tmp_path, dataset, sources.good_task,
                              datamodule=sources.BAD_DATAMODULE)

    report = run(staged, task_name)

    assert not report["ok"]
    assert "datamodule" in failed(report)
    assert "seq" in detail_of(report, "datamodule")   # the column it looked for


def test_unregistered_task_is_caught(tmp_path, dataset):
    task_name, staged = stage(tmp_path, dataset, sources.good_task)
    # Ask for a name the module does not register.
    report = run(staged, "some_other_name")

    assert not report["ok"]
    assert "import" in failed(report)
    assert "not registered" in detail_of(report, "import")


def test_syntactically_broken_code_is_reported_not_raised(tmp_path, dataset):
    task_name, staged = stage(tmp_path, dataset, lambda name: "this is not python(")

    report = run(staged, task_name)

    assert not report["ok"]
    assert "import" in failed(report)


def test_skipped_data_checks_are_not_treated_as_verified(tmp_path, dataset):
    """A skipped datamodule check means the code never met the real data. Approving on
    that is the 'harness that passes everything' failure mode."""
    from adaptrna_agentic.codegen.harness import unmet_requirements

    task_name, staged = stage(tmp_path, dataset, sources.good_task)
    report = run(staged, task_name, only=["import", "config_head"])

    assert report["ok"]                                   # nothing FAILED...
    assert unmet_requirements(report) == [
        "datamodule", "forward_backward", "metrics",
    ]                                                     # ...but nothing was proven


def test_summarize_is_readable(tmp_path, dataset):
    task_name, staged = stage(tmp_path, dataset, sources.task_with_unsaved_state)

    text = summarize(run(staged, task_name))

    assert "FAIL" in text
    assert "adapter_roundtrip" in text
