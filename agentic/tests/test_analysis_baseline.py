"""A task with no knowledge-base band is compared against this project's own earlier runs.

This closes the gap Phase 6 exposed: the generated task's verdict was "nothing to
compare". A baseline is explicitly *not* a validated reference — the wording matters,
because a first run should establish a number, not be judged against someone else's."""

import pytest

from adaptrna_agentic.jobs.analysis import analyze_run
from adaptrna_agentic.jobs.store import JobRecord, JobStore

METRICS_HEADER = "epoch,step,train/loss,test/f1_score\n"


def _run_dir(tmp_path, name, f1, loss="0.10"):
    directory = tmp_path / "outputs" / name
    metrics = directory / "metrics" / "version_0"
    metrics.mkdir(parents=True)
    (metrics / "metrics.csv").write_text(
        METRICS_HEADER + f"0,50,{loss},\n1,100,,{f1}\n"
    )
    return directory


def _record(store, job_id, directory, task="generated_task", arm="lora", plan=None):
    store.add(JobRecord(
        id=job_id, task=task, arm=arm, command=["x"], output_dir=str(directory),
        state="succeeded", started_at="2026-01-01T00:00:00+00:00", plan=plan or {},
    ))


def _mock_band(monkeypatch, band, tolerance=1.0):
    """A reference band no real DatasetSpec-driven run can produce any more (Phase 13),
    kept reachable in tests so the comparison logic itself stays proven correct."""
    import adaptrna_agentic.jobs.analysis as module

    fake = {"reference": {"band": band, "tolerance": tolerance, "sources": ["test"]},
            "caveats": []}
    monkeypatch.setattr(module, "generic_knowledge", lambda: fake)


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    return JobStore(tmp_path / "jobs_data")


def test_first_run_of_an_unknown_task_is_the_baseline(jobs, tmp_path):
    directory = _run_dir(tmp_path, "first", "0.90")

    report = analyze_run(directory, task="generated_task", arm="lora",
                        plan={"primary_metric": "test/f1_score"})

    assert report["verdict"] == "ok"
    assert any("this run is the baseline" in c for c in report["checks"])
    assert "baseline" not in report


def test_later_run_is_compared_with_the_earlier_one(jobs, tmp_path):
    _record(jobs, "older", _run_dir(tmp_path, "older", "0.90"))
    directory = _run_dir(tmp_path, "newer", "0.95")

    report = analyze_run(directory, task="generated_task", arm="lora",
                        plan={"primary_metric": "test/f1_score"})

    assert report["baseline"]["job_id"] == "older"
    assert report["verdict"] == "ok"
    assert any("improves on the baseline" in c for c in report["checks"])


def test_a_drop_below_the_baseline_is_flagged_but_named_as_a_baseline(jobs, tmp_path):
    _record(jobs, "older", _run_dir(tmp_path, "older", "0.95"))
    directory = _run_dir(tmp_path, "newer", "0.60")

    report = analyze_run(directory, task="generated_task", arm="lora",
                        plan={"primary_metric": "test/f1_score"})

    assert report["verdict"] == "suspicious"
    text = " ".join(report["checks"])
    assert "below the baseline" in text
    assert "not a validated reference" in text


def test_a_difference_inside_tolerance_is_not_a_regression(jobs, tmp_path, monkeypatch):
    """Run-to-run non-determinism must not read as a regression (MASTER_PLAN §7).

    No real DatasetSpec-driven task carries a reference band any more (Phase 13) -- the
    band is mocked here so the comparison logic itself stays proven correct."""
    _mock_band(monkeypatch, [95.8, 97.5], tolerance=1.0)
    _record(jobs, "older", _run_dir(tmp_path, "older", "96.5"), task="demo_binary")
    directory = _run_dir(tmp_path, "newer", "95.8")

    report = analyze_run(directory, task="demo_binary", arm="lora",
                        plan={"primary_metric": "test/f1_score"})

    assert report["verdict"] == "ok"      # the knowledge band covers both anyway


def test_truncated_runs_are_not_used_as_baselines(jobs, tmp_path):
    _record(jobs, "smoke", _run_dir(tmp_path, "smoke", "0.55"),
            plan={"quick_run": True, "overrides": {"trainer.max_steps": 200}})
    directory = _run_dir(tmp_path, "real", "0.90")

    report = analyze_run(directory, task="generated_task", arm="lora",
                        plan={"primary_metric": "test/f1_score"})

    assert "baseline" not in report
    assert any("this run is the baseline" in c for c in report["checks"])


def test_a_knowledge_base_band_still_wins_over_history(jobs, tmp_path, monkeypatch):
    _mock_band(monkeypatch, [95.8, 97.5], tolerance=1.0)
    _record(jobs, "older", _run_dir(tmp_path, "older", "99.0"), task="demo_binary")
    directory = _run_dir(tmp_path, "newer", "96.5")

    report = analyze_run(directory, task="demo_binary", arm="lora",
                        plan={"primary_metric": "test/f1_score"})

    assert "baseline" not in report
    assert any("reference band" in c for c in report["checks"])


def test_baselines_only_come_from_the_same_arm(jobs, tmp_path):
    _record(jobs, "full_ft_run", _run_dir(tmp_path, "full_ft_run", "0.99"), arm="full_ft")
    directory = _run_dir(tmp_path, "lora_run", "0.90")

    report = analyze_run(directory, task="generated_task", arm="lora",
                        plan={"primary_metric": "test/f1_score"})

    assert "baseline" not in report
