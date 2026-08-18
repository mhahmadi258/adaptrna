"""RunAnalyzer against synthetic metrics.csv fixtures.

The two rules under test: a truncated run is never compared to a reference, and a
within-tolerance difference is never called a regression.

Phase 13: there are no known tasks any more, so the knowledge base carries no per-task
primary metric or reference band — `analyze_run` reads the primary metric from the plan
(set by `recommend()` from the approved spec's `head.primary_metric`) and always takes
the generic (`band: null`) path in real use. The reference-band comparison branch in
`analyze_run` is unreachable through the real knowledge base now, but it is still live
code (deliberately: jobs/** is untouched by this phase, per the plan), so the tests that
exercise it mock a band explicitly rather than dropping the coverage.
"""

import pytest

from adaptrna_agentic.jobs.analysis import (
    VERDICT_FAILED,
    VERDICT_OK,
    VERDICT_SUSPICIOUS,
    analyze_run,
)


def write_metrics(tmp_path, rows, header="epoch,step,train/loss,test/f1_score"):
    directory = tmp_path / "metrics" / "version_0"
    directory.mkdir(parents=True)
    (directory / "metrics.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    return tmp_path


def splice_run(tmp_path, f1, losses=("0.42", "0.11")):
    rows = [f"0,{50 * (i + 1)},{loss}," for i, loss in enumerate(losses)]
    rows.append(f"1,{50 * (len(losses) + 1)},,{f1}")
    return write_metrics(tmp_path, rows)


def _plan(task="generated_task", arm="lora", primary_metric="test/f1_score", **overrides):
    plan = {"task": task, "arm": arm, "primary_metric": primary_metric, "overrides": {}}
    plan.update(overrides)
    return plan


def _mock_band(monkeypatch, band, tolerance=1.0):
    """A reference band no real DatasetSpec-driven run can produce any more (Phase 13),
    kept reachable in tests so the comparison logic itself stays proven correct."""
    import adaptrna_agentic.jobs.analysis as module

    fake = {"reference": {"band": band, "tolerance": tolerance, "sources": ["test"]},
            "caveats": []}
    monkeypatch.setattr(module, "generic_knowledge", lambda: fake)


# ---------------------------------------------------------------- healthy runs, in-band

def test_healthy_splice_run_is_ok(tmp_path, monkeypatch):
    _mock_band(monkeypatch, [95.8, 97.5], tolerance=1.0)

    report = analyze_run(splice_run(tmp_path, 96.5), plan=_plan())

    assert report["verdict"] == VERDICT_OK
    assert report["primary_value"] == pytest.approx(96.5)
    assert any("within the reference band" in check for check in report["checks"])


def test_one_point_below_the_band_is_not_a_regression(tmp_path, monkeypatch):
    """FlashAttention non-determinism: 95.21 vs 95.82 from the same command and seed."""
    _mock_band(monkeypatch, [95.8, 97.5], tolerance=1.0)

    report = analyze_run(splice_run(tmp_path, 95.21), plan=_plan())

    assert report["verdict"] == VERDICT_OK
    assert not any("below the reference band" in check for check in report["checks"])


def test_clearly_below_the_band_is_flagged(tmp_path, monkeypatch):
    _mock_band(monkeypatch, [95.8, 97.5], tolerance=1.0)

    report = analyze_run(splice_run(tmp_path, 60.0), plan=_plan())

    assert report["verdict"] == VERDICT_SUSPICIOUS
    assert any("below the reference band" in check for check in report["checks"])
    assert any("second seed" in s for s in report["suggestions"])


def test_above_the_band_is_noted_not_failed(tmp_path, monkeypatch):
    _mock_band(monkeypatch, [95.8, 97.5], tolerance=1.0)

    report = analyze_run(splice_run(tmp_path, 99.9), plan=_plan())

    assert report["verdict"] == VERDICT_OK
    assert any("leakage" in check for check in report["checks"])


# ---------------------------------------------------------------- failures

def test_degenerate_metric_fails_with_the_arm_remedy(tmp_path):
    report = analyze_run(splice_run(tmp_path, 0.0), plan=_plan(arm="lora"))

    assert report["verdict"] == VERDICT_FAILED
    assert any("learned nothing" in check for check in report["checks"])
    assert any("3e-4" in s for s in report["suggestions"])


def test_destroyed_backbone_on_full_ft_points_at_1e_5(tmp_path):
    directory = write_metrics(
        tmp_path,
        ["0,50,0.9,", "1,100,0.9,", "1,101,,0.0"],
        header="epoch,step,train/loss,test/r2",
    )

    report = analyze_run(directory, plan=_plan(arm="full_ft", primary_metric="test/r2"))

    assert report["verdict"] == VERDICT_FAILED
    assert any("1e-5" in s for s in report["suggestions"])


def test_nan_loss_fails(tmp_path):
    directory = splice_run(tmp_path, 96.0, losses=("0.42", "nan"))

    report = analyze_run(directory, plan=_plan())

    assert report["verdict"] == VERDICT_FAILED
    assert any("non-finite loss" in check for check in report["checks"])
    assert any("gradient_clip_val" in s for s in report["suggestions"])


def test_collapsed_lora_run_is_suspicious_with_the_collapse_remedy(tmp_path):
    directory = splice_run(tmp_path, 50.0, losses=("0.42", "0.126", "0.126", "0.126", "0.126"))

    report = analyze_run(directory, plan=_plan(arm="lora"))

    assert report["verdict"] in (VERDICT_SUSPICIOUS, VERDICT_FAILED)
    assert any("flatlined" in check for check in report["checks"])
    assert any("constant-output" in s for s in report["suggestions"])


def test_missing_metrics_file_fails_with_a_log_hint(tmp_path):
    report = analyze_run(tmp_path, task="splice_site")

    assert report["verdict"] == VERDICT_FAILED
    assert any("no metrics.csv" in check for check in report["checks"])
    assert any("log" in s for s in report["suggestions"])


def test_no_primary_metric_on_the_plan_is_suspicious_not_a_crash(tmp_path):
    """The knowledge base carries no per-task metric name any more (Phase 13); without
    one on the plan, there is nothing to judge the run's metrics against."""
    report = analyze_run(splice_run(tmp_path, 96.0), task="generated_task")

    assert report["verdict"] == VERDICT_SUSPICIOUS
    assert any("no primary metric" in check for check in report["checks"])
    assert "primary_value" not in report


# ---------------------------------------------------------------- truncated runs

def test_truncated_run_is_not_compared_to_the_reference(tmp_path, monkeypatch):
    _mock_band(monkeypatch, [95.8, 97.5], tolerance=1.0)
    plan = _plan(quick_run=True, overrides={"trainer.max_steps": 200})

    report = analyze_run(splice_run(tmp_path, 60.0), plan=plan)

    assert report["truncated"] is True
    assert report["verdict"] == VERDICT_OK               # low F1 is expected, not a fault
    assert any("NOT comparable" in check for check in report["checks"])
    assert not any("below the reference band" in check for check in report["checks"])


def test_truncation_detected_from_overrides_without_quick_flag(tmp_path):
    plan = _plan(overrides={"trainer.max_steps": 50})

    assert analyze_run(splice_run(tmp_path, 70.0), plan=plan)["truncated"] is True


# ---------------------------------------------------------------- misc

def test_task_without_a_reference_band_says_so(tmp_path):
    """This is the ordinary path now (Phase 13): every task's band is null."""
    directory = write_metrics(
        tmp_path, ["0,50,0.4,", "1,100,,0.62"],
        header="epoch,step,train/loss,test/f1",
    )

    report = analyze_run(directory, plan=_plan(primary_metric="test/f1"))

    assert report["verdict"] == VERDICT_OK
    # Phase 7 replaced "nothing to compare" with a baseline statement: a first run of an
    # unknown task establishes the number future runs are measured against.
    assert any("baseline" in check for check in report["checks"])


def test_cost_columns_are_excluded_from_reported_metrics(tmp_path):
    """`cost/*` (rinalmo_hub/cost.py) is training instrumentation, not a task metric."""
    directory = write_metrics(
        tmp_path,
        ["0,50,0.42,,120.5", "1,100,0.11,,110.2", "1,101,,96.5,"],
        header="epoch,step,train/loss,test/f1_score,cost/train/iter_time_ms",
    )

    report = analyze_run(directory, plan=_plan())

    assert "cost/train/iter_time_ms" not in report["metrics"]
    assert report["primary_value"] == pytest.approx(96.5)


def test_plan_supplies_task_and_arm(tmp_path):
    plan = {"task": "splice_site", "arm": "lora", "overrides": {}}

    report = analyze_run(splice_run(tmp_path, 96.0), plan=plan)

    assert report["task"] == "splice_site"
    assert report["arm"] == "lora"
