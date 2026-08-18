"""`prune` is the one destructive command in the project.

The rules it must never break: a dry run by default, and nothing the manifest references
is ever deleted — those artifacts are hours of GPU time and reviewed code."""

import pytest

from adaptrna_agentic.toolhub.errors import ToolHubError
from adaptrna_agentic.toolhub.prune import prune


def _labels(report, key):
    return {item["label"] for item in report[key]}


def _kept_reasons(report):
    return {item["label"]: item["reason"] for item in report["kept"]}


@pytest.fixture
def hub(nano_registry, nano_splice_adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    nano_registry.register(nano_splice_adapter)
    return nano_registry


def _stage(hub, name="leftover"):
    from adaptrna_agentic.codegen import staging

    return staging.stage_task(
        name,
        {"task.py": "x = 1\n", "datamodule.py": "y = 2\n", "config.yaml": f"task: {name}\n"},
        data_dir=hub.data_dir,
    )


# ---------------------------------------------------------------- dry run

def test_dry_run_is_the_default(hub):
    stage = _stage(hub)

    report = prune("staging", data_dir=hub.data_dir)

    assert report["applied"] is False
    assert _labels(report, "would_remove") == {stage.id}
    assert stage.root.exists()               # nothing was touched


def test_apply_actually_removes(hub):
    stage = _stage(hub)

    report = prune("staging", apply=True, data_dir=hub.data_dir)

    assert report["applied"] is True
    assert not stage.root.exists()
    assert report["reclaimed_bytes"] > 0


# ---------------------------------------------------------------- protection

def test_referenced_artifacts_are_never_removed(hub):
    (hub.data_dir / "adapters" / "orphan.pt").write_bytes(b"junk")

    report = prune("artifacts", apply=True, data_dir=hub.data_dir)

    assert _labels(report, "removed") == {"orphan.pt"}
    assert "demo_binary.pt" in _kept_reasons(report)
    assert "referenced by tool 'demo_binary'" in _kept_reasons(report)["demo_binary.pt"]
    assert hub.get("demo_binary").artifact_path().exists()


def test_runs_that_produced_a_registered_tool_are_kept(hub, tmp_path, monkeypatch):
    """The output dir behind a live tool is protected even when it is old."""
    from adaptrna_agentic.settings import REPO_ROOT

    outputs = tmp_path / "outputs"
    (outputs / "kept_run").mkdir(parents=True)
    (outputs / "old_run").mkdir(parents=True)
    (outputs / "old_run" / "train.log").write_text("x" * 1000)

    entry = hub.get("demo_binary")
    entry.provenance["source"] = str(outputs / "kept_run" / "demo_binary_adapter.pt")
    hub.manifest.save()

    monkeypatch.setattr("adaptrna_agentic.toolhub.prune.REPO_ROOT", tmp_path)

    report = prune("runs", older_than=0, apply=True, data_dir=hub.data_dir)

    assert "kept_run" in _kept_reasons(report)
    assert "registered tool" in _kept_reasons(report)["kept_run"]
    assert (outputs / "kept_run").exists()
    assert not (outputs / "old_run").exists()


def test_running_jobs_are_never_pruned(hub, tmp_path):
    from adaptrna_agentic.jobs.store import JobRecord, JobStore

    store = JobStore(tmp_path / "jobs_data")
    store.add(JobRecord(id="live", task="t", arm="lora", command=["x"],
                        output_dir=str(tmp_path / "outputs" / "live"), state="running"))

    report = prune("jobs", older_than=0, data_dir=hub.data_dir,
                   jobs_dir=tmp_path / "jobs_data")

    assert _kept_reasons(report)["live"] == "still running"


# ---------------------------------------------------------------- guards

def test_runs_requires_an_explicit_age(hub):
    with pytest.raises(ToolHubError, match="Give an explicit age"):
        prune("runs", data_dir=hub.data_dir)


def test_unknown_target_is_rejected(hub):
    with pytest.raises(ToolHubError, match="Unknown prune target"):
        prune("everything", data_dir=hub.data_dir)


def test_age_filter_keeps_young_items(hub):
    stage = _stage(hub)

    report = prune("staging", older_than=30, data_dir=hub.data_dir)

    assert _labels(report, "would_remove") == set()
    assert "younger than 30" in _kept_reasons(report)[stage.id]


def test_report_names_the_dry_run(hub):
    from adaptrna_agentic.toolhub.prune import format_report

    _stage(hub)
    text = format_report(prune("staging", data_dir=hub.data_dir))

    assert "would remove" in text
    assert "--yes" in text
