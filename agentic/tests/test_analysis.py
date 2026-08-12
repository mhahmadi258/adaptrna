"""RunAnalyzer against synthetic metrics.csv fixtures.

The two rules under test: a truncated run is never compared to a reference, and a
within-tolerance difference is never called a regression.
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


# ---------------------------------------------------------------- healthy runs

def test_healthy_splice_run_is_ok(tmp_path):
    report = analyze_run(splice_run(tmp_path, 96.5), task="splice_site")

    assert report["verdict"] == VERDICT_OK
    assert report["primary_value"] == pytest.approx(96.5)
    assert any("within the reference band" in check for check in report["checks"])


def test_one_point_below_the_band_is_not_a_regression(tmp_path):
    """FlashAttention non-determinism: 95.21 vs 95.82 from the same command and seed."""
    report = analyze_run(splice_run(tmp_path, 95.21), task="splice_site")

    assert report["verdict"] == VERDICT_OK
    assert not any("below the reference band" in check for check in report["checks"])


def test_clearly_below_the_band_is_flagged(tmp_path):
    report = analyze_run(splice_run(tmp_path, 60.0), task="splice_site")

    assert report["verdict"] == VERDICT_SUSPICIOUS
    assert any("below the reference band" in check for check in report["checks"])
    assert any("second seed" in s for s in report["suggestions"])


def test_above_the_band_is_noted_not_failed(tmp_path):
    report = analyze_run(splice_run(tmp_path, 99.9), task="splice_site")

    assert report["verdict"] == VERDICT_OK
    assert any("leakage" in check for check in report["checks"])


# ---------------------------------------------------------------- failures

def test_degenerate_metric_fails_with_the_arm_remedy(tmp_path):
    report = analyze_run(splice_run(tmp_path, 0.0), task="splice_site", arm="lora")

    assert report["verdict"] == VERDICT_FAILED
    assert any("learned nothing" in check for check in report["checks"])
    assert any("3e-4" in s for s in report["suggestions"])


def test_destroyed_backbone_on_full_ft_points_at_1e_5(tmp_path):
    directory = write_metrics(
        tmp_path,
        ["0,50,0.9,", "1,100,0.9,", "1,101,,0.0"],
        header="epoch,step,train/loss,test/r2",
    )

    report = analyze_run(directory, task="mrl", arm="full_ft")

    assert report["verdict"] == VERDICT_FAILED
    assert any("1e-5" in s for s in report["suggestions"])


def test_nan_loss_fails(tmp_path):
    directory = splice_run(tmp_path, 96.0, losses=("0.42", "nan"))

    report = analyze_run(directory, task="splice_site")

    assert report["verdict"] == VERDICT_FAILED
    assert any("non-finite loss" in check for check in report["checks"])
    assert any("gradient_clip_val" in s for s in report["suggestions"])


def test_collapsed_lora_run_is_suspicious_with_the_collapse_remedy(tmp_path):
    directory = splice_run(tmp_path, 50.0, losses=("0.42", "0.126", "0.126", "0.126", "0.126"))

    report = analyze_run(directory, task="splice_site", arm="lora")

    assert report["verdict"] in (VERDICT_SUSPICIOUS, VERDICT_FAILED)
    assert any("flatlined" in check for check in report["checks"])
    assert any("constant-output" in s for s in report["suggestions"])


def test_missing_metrics_file_fails_with_a_log_hint(tmp_path):
    report = analyze_run(tmp_path, task="splice_site")

    assert report["verdict"] == VERDICT_FAILED
    assert any("no metrics.csv" in check for check in report["checks"])
    assert any("log" in s for s in report["suggestions"])


# ---------------------------------------------------------------- truncated runs

def test_truncated_run_is_not_compared_to_the_reference(tmp_path):
    plan = {"task": "splice_site", "arm": "lora", "quick_run": True,
            "overrides": {"trainer.max_steps": 200}}

    report = analyze_run(splice_run(tmp_path, 60.0), plan=plan)

    assert report["truncated"] is True
    assert report["verdict"] == VERDICT_OK               # low F1 is expected, not a fault
    assert any("NOT comparable" in check for check in report["checks"])
    assert not any("below the reference band" in check for check in report["checks"])


def test_truncation_detected_from_overrides_without_quick_flag(tmp_path):
    plan = {"task": "splice_site", "arm": "lora", "overrides": {"trainer.max_steps": 50}}

    assert analyze_run(splice_run(tmp_path, 70.0), plan=plan)["truncated"] is True


# ---------------------------------------------------------------- misc

def test_task_without_a_reference_band_says_so(tmp_path):
    directory = write_metrics(
        tmp_path, ["0,50,0.4,", "1,100,,0.62"],
        header="epoch,step,train/loss,test/f1",
    )

    report = analyze_run(directory, task="sec_struct")

    assert report["verdict"] == VERDICT_OK
    # Phase 7 replaced "nothing to compare" with a baseline statement: a first run of an
    # unknown task establishes the number future runs are measured against.
    assert any("baseline" in check for check in report["checks"])


def test_plan_supplies_task_and_arm(tmp_path):
    plan = {"task": "splice_site", "arm": "lora", "overrides": {}}

    report = analyze_run(splice_run(tmp_path, 96.0), plan=plan)

    assert report["task"] == "splice_site"
    assert report["arm"] == "lora"
