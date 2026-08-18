"""JobRunner against fake commands — no GPU, no engine training, seconds.

The fake command stands in for the engine CLI: it writes a metrics.csv the way the engine
does and exits with a chosen code, which is exactly the interface the runner consumes.
"""

import sys
import textwrap
import time

import pytest

from adaptrna_agentic.jobs.runner import JobRunner, read_progress
from adaptrna_agentic.jobs.store import JobStore
from adaptrna_agentic.toolhub.errors import ToolHubError

FAKE_TRAIN = textwrap.dedent("""
    import pathlib, sys, time
    out = pathlib.Path(sys.argv[1]); code = int(sys.argv[2]); delay = float(sys.argv[3])
    metrics = out / "metrics" / "version_0"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "metrics.csv").write_text(
        "epoch,step,train/loss,test/f1_score\\n"
        "0,50,0.42,\\n"
        "1,100,0.11,\\n"
        "1,101,,96.5\\n"
    )
    time.sleep(delay)
    if code == 0:
        (out / "demo_binary_adapter.pt").write_bytes(b"fake-adapter")
    (out / "exit_code").write_text(str(code))
    sys.exit(code)
""")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)
    return JobRunner(store=JobStore(tmp_path / "jobs_data"))


def plan(tmp_path, name="run_a", code=0, delay=0.0):
    script = tmp_path / "fake_train.py"
    script.write_text(FAKE_TRAIN)
    output_dir = tmp_path / "outputs" / name

    return {
        "task": "demo_binary", "arm": "lora", "output_dir": str(output_dir),
        "command": [sys.executable, str(script), str(output_dir), str(code), str(delay)],
    }


def wait_for(runner, job_id, state, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = runner.status(job_id)
        if status["state"] == state:
            return status
        time.sleep(0.1)
    pytest.fail(f"job '{job_id}' never reached '{state}' (last: {runner.status(job_id)})")


def test_successful_job_lifecycle(runner, tmp_path):
    record = runner.start(plan(tmp_path))

    assert record.state == "running"
    assert record.pid

    status = wait_for(runner, record.id, "succeeded")

    assert status["exit_code"] == 0
    assert status["ended_at"]
    assert status["adapter_path"].endswith("demo_binary_adapter.pt")


def test_failed_job_is_recorded(runner, tmp_path):
    record = runner.start(plan(tmp_path, name="run_bad", code=3))

    status = wait_for(runner, record.id, "failed")

    assert status["exit_code"] == 3
    assert status["adapter_path"] is None


def test_progress_parsed_from_metrics_csv(runner, tmp_path):
    record = runner.start(plan(tmp_path))
    wait_for(runner, record.id, "succeeded")

    progress = runner.status(record.id)["progress"]

    assert progress["epoch"] == 1
    assert progress["step"] == 101
    assert progress["latest_metrics"]["train/loss"] == pytest.approx(0.11)
    assert progress["latest_metrics"]["test/f1_score"] == pytest.approx(96.5)


def test_logs_are_captured(runner, tmp_path):
    script = tmp_path / "noisy.py"
    script.write_text("import pathlib,sys\n"
                      "print('hello from the run')\n"
                      "out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
                      "(out/'exit_code').write_text('0')\n")
    output_dir = tmp_path / "outputs" / "noisy"
    record = runner.start({
        "task": "demo_binary", "arm": "lora", "output_dir": str(output_dir),
        "command": [sys.executable, str(script), str(output_dir)],
    })

    wait_for(runner, record.id, "succeeded")

    assert "hello from the run" in runner.logs(record.id)


def test_concurrency_is_refused_naming_the_running_job(runner, tmp_path):
    first = runner.start(plan(tmp_path, name="long_run", delay=5.0))

    with pytest.raises(ToolHubError, match=f"Job '{first.id}'.*still running"):
        runner.start(plan(tmp_path, name="second_run"))

    runner.cancel(first.id)


def test_allow_concurrent_overrides_the_refusal(runner, tmp_path):
    first = runner.start(plan(tmp_path, name="long_run", delay=5.0))

    second = runner.start(plan(tmp_path, name="parallel"), allow_concurrent=True)
    assert second.state == "running"

    runner.cancel(first.id)
    runner.cancel(second.id)


def test_cancel_marks_the_job(runner, tmp_path):
    record = runner.start(plan(tmp_path, name="cancel_me", delay=10.0))

    result = runner.cancel(record.id)

    assert result["state"] == "cancelled"
    assert runner.status(record.id)["state"] == "cancelled"

    with pytest.raises(ToolHubError, match="not running"):
        runner.cancel(record.id)


def test_state_survives_a_fresh_store(runner, tmp_path):
    record = runner.start(plan(tmp_path))
    wait_for(runner, record.id, "succeeded")

    reopened = JobRunner(store=JobStore(runner.store.jobs_dir))

    assert reopened.status(record.id)["state"] == "succeeded"
    assert [j["id"] for j in reopened.list()] == [record.id]


def test_dead_process_without_exit_code_is_failed(runner, tmp_path):
    """A hard crash (OOM, kill -9) leaves no exit_code file — the PID check catches it."""
    script = tmp_path / "hard_exit.py"
    script.write_text("import os\nos._exit(9)\n")
    output_dir = tmp_path / "outputs" / "crashed"
    record = runner.start({
        "task": "demo_binary", "arm": "lora", "output_dir": str(output_dir),
        "command": [sys.executable, str(script)],
    })

    assert wait_for(runner, record.id, "failed")["exit_code"] == 9


def test_crashed_job_detected_without_a_process_handle(runner, tmp_path):
    """The cross-process path: a fresh runner has no Popen handle, so a zombie/dead PID
    must still be recognised as failed (this is what a restarted chat sees)."""
    script = tmp_path / "hard_exit2.py"
    script.write_text("import os\nos._exit(9)\n")
    output_dir = tmp_path / "outputs" / "crashed2"
    record = runner.start({
        "task": "demo_binary", "arm": "lora", "output_dir": str(output_dir),
        "command": [sys.executable, str(script)],
    })

    time.sleep(1.0)
    runner._processes.clear()            # simulate a different process observing the job
    reopened = JobRunner(store=JobStore(runner.store.jobs_dir))

    assert wait_for(reopened, record.id, "failed")


def test_read_progress_without_metrics_is_none(tmp_path):
    assert read_progress(tmp_path)["progress"] is None


def test_read_progress_surfaces_cost_columns_unfiltered(tmp_path):
    """Unlike `analysis._final_metrics`, the live job panel shows `cost/*` as-is."""
    metrics = tmp_path / "metrics" / "version_0"
    metrics.mkdir(parents=True)
    (metrics / "metrics.csv").write_text(
        "epoch,step,train/loss,cost/train/iter_time_ms\n"
        "0,10,0.5,412.3\n"
    )

    progress = read_progress(tmp_path)["progress"]
    assert progress["latest_metrics"]["cost/train/iter_time_ms"] == pytest.approx(412.3)


def test_unknown_job_lists_known_ids(runner, tmp_path):
    runner.start(plan(tmp_path))

    with pytest.raises(KeyError, match="Known jobs"):
        runner.status("nope")
