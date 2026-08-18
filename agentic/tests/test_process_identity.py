"""A PID is not an identity.

The kernel recycles PIDs, so a dead job whose PID has been reused would look alive
forever — and `cancel` would `killpg` whatever now owns it. Every check here exists
because that is a real hazard on the live install: all three job records written during
Phases 5–6 hold PIDs that are already free.
"""

import os
import subprocess
import sys
import time

import pytest

from adaptrna_agentic.jobs.runner import (
    JobRunner,
    _is_our_process,
    process_starttime,
)
from adaptrna_agentic.jobs.store import JobRecord, JobStore
from adaptrna_agentic.toolhub.errors import ToolHubError


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)
    return JobRunner(JobStore(tmp_path / "jobs_data"))


def _plan(tmp_path, name, body):
    script = tmp_path / f"{name}.py"
    script.write_text(body)
    return {
        "task": "demo_binary", "arm": "lora",
        "output_dir": str(tmp_path / "outputs" / name),
        "command": [sys.executable, str(script)],
        "overrides": {},
    }


# ---------------------------------------------------------------- identity primitives

def test_starttime_is_readable_and_stable():
    first = process_starttime(os.getpid())
    time.sleep(0.01)

    assert first is not None
    assert process_starttime(os.getpid()) == first     # does not drift


def test_our_own_process_matches():
    assert _is_our_process(os.getpid(), process_starttime(os.getpid())) is True


def test_wrong_starttime_is_not_ours():
    """The recycled-PID case: the PID exists, but it is somebody else's process now."""
    assert _is_our_process(os.getpid(), "1") is False


def test_missing_starttime_is_treated_as_gone():
    """Records written before start times were captured have no identity, so they must
    be assumed dead rather than signalled."""
    assert _is_our_process(os.getpid(), None) is False


def test_dead_pid_is_not_ours():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    starttime = process_starttime(child.pid)
    child.wait()

    assert _is_our_process(child.pid, starttime) is False   # reaped -> gone


# ---------------------------------------------------------------- runner behaviour

def test_launch_records_the_start_time(runner, tmp_path):
    record = runner.start(_plan(tmp_path, "quick", "pass\n"))

    assert record.pid_starttime is not None
    assert record.pid_starttime == process_starttime(record.pid) or record.state != "running"


def test_cancel_refuses_a_record_it_cannot_prove_is_ours(runner, tmp_path):
    """The dangerous case: without this guard, killpg would signal a stranger."""
    record = runner.start(_plan(tmp_path, "slow", "import time; time.sleep(30)\n"))
    real_pid, real_start = record.pid, record.pid_starttime

    # Simulate PID reuse: same PID, different process.
    record.pid_starttime = "1"
    runner.store.save()

    with pytest.raises(ToolHubError, match="may since have been reused"):
        runner.cancel(record.id)

    # The record is closed out, and the real process was left completely alone.
    assert runner.store.get(record.id).state == "failed"
    assert _is_our_process(real_pid, real_start) is True

    os.killpg(os.getpgid(real_pid), 15)     # clean up the process we did not kill


def test_legacy_record_without_starttime_is_not_signalled(runner, tmp_path):
    record = runner.start(_plan(tmp_path, "legacy", "import time; time.sleep(30)\n"))
    real_pid, real_start = record.pid, record.pid_starttime
    record.pid_starttime = None             # as written by a pre-Phase-7 build
    runner.store.save()

    with pytest.raises(ToolHubError):
        runner.cancel(record.id)

    assert _is_our_process(real_pid, real_start) is True
    os.killpg(os.getpgid(real_pid), 15)


def test_cancel_still_works_for_a_genuine_job(runner, tmp_path):
    record = runner.start(_plan(tmp_path, "real", "import time; time.sleep(30)\n"))

    assert runner.cancel(record.id)["state"] == "cancelled"

    for _ in range(50):
        if not _is_our_process(record.pid, record.pid_starttime):
            break
        time.sleep(0.1)
    assert not _is_our_process(record.pid, record.pid_starttime)


def test_stale_running_record_is_reconciled_as_failed(runner, tmp_path):
    """A job whose process vanished without an exit code (SIGKILL, OOM, a reboot)."""
    record = runner.start(_plan(tmp_path, "vanish", "pass\n"))
    for _ in range(50):
        if runner.status(record.id)["state"] != "running":
            break
        time.sleep(0.1)

    # Re-open the store as a fresh process would, with no Popen handle to poll.
    reopened = JobRunner(JobStore(runner.store.jobs_dir))
    stale = reopened.store.get(record.id)
    stale.state = "running"
    stale.exit_code = None
    (stale.output_path / "exit_code").unlink(missing_ok=True)
    stale.pid_starttime = "1"               # the PID now belongs to someone else
    reopened.store.save()

    assert reopened.status(record.id)["state"] == "failed"
