"""JobRunner: launch the engine's training CLI on the local GPU and track it.

Jobs are **detached**: a real run takes minutes to hours, the chat must stay responsive,
and the run must survive the chat exiting. State is therefore derived from disk — the
`exit_code` file the entrypoint writes, the engine's own `metrics.csv`, and PID liveness —
never from a live process handle.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import signal
import subprocess

from adaptrna_agentic.jobs.store import JobRecord, JobStore
from adaptrna_agentic.settings import REPO_ROOT
from adaptrna_agentic.toolhub.errors import ToolHubError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def process_starttime(pid: int) -> Optional[str]:
    """Process start time in clock ticks — `/proc/<pid>/stat` field 22.

    `(pid, starttime)` is a stable identity for a process on Linux: the kernel recycles
    PIDs, but a recycled PID always has a later start time.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # The comm field is parenthesised and may contain spaces, so split after it:
        # field 22 overall is index 19 of the remainder.
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _process_state(pid: int) -> Optional[str]:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def _is_our_process(pid: Optional[int], starttime: Optional[str]) -> bool:
    """Is this PID still *our* live process?

    Three ways the answer is no, and all three matter:
      * the PID is gone;
      * the PID exists but is a zombie — it has exited and merely awaits reaping, and a
        zombie still answers signal 0;
      * the PID exists but belongs to somebody else now, because it was recycled. Records
        written before start times were captured fall here too: without an identity we
        must assume the worst rather than signal a stranger.
    """
    if not pid:
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process — definitively not ours.
        return False

    if _process_state(pid) == "Z":
        return False

    if starttime is None:
        return False

    return process_starttime(pid) == starttime


class JobRunner:
    def __init__(self, store: Optional[JobStore] = None, jobs_dir=None):
        self.store = store or JobStore(jobs_dir)
        # Popen handles for jobs this process started: polling them reaps the child and
        # yields a real exit code even when the run died before writing `exit_code`.
        self._processes: Dict[str, subprocess.Popen] = {}

    # ------------------------------------------------------------------ launching

    def start(self, plan: Dict[str, Any], allow_concurrent: bool = False) -> JobRecord:
        """Launch a training plan as a detached background job."""
        self._refresh_all()

        if not allow_concurrent:
            running = self.store.running()
            if running:
                other = running[0]
                raise ToolHubError(
                    f"Job '{other.id}' ({other.task}) is still running — two giga runs on "
                    f"one GPU is how you get an out-of-memory failure mid-run. Wait for it "
                    f"to finish, cancel it, or pass allow_concurrent."
                )

        output_dir = Path(plan["output_dir"])
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        job_id = output_dir.name
        if job_id in self.store.jobs:
            raise ToolHubError(f"Job '{job_id}' already exists.")

        log_path = output_dir / "train.log"
        with open(log_path, "w") as log:
            process = subprocess.Popen(
                plan["command"],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,      # detached: survives the chat process
            )

        record = JobRecord(
            id=job_id,
            task=plan["task"],
            arm=plan["arm"],
            command=list(plan["command"]),
            output_dir=str(output_dir),
            state="running",
            pid=process.pid,
            pid_starttime=process_starttime(process.pid),
            started_at=_now(),
            plan=plan,
        )

        self._processes[job_id] = process

        return self.store.add(record)

    # ------------------------------------------------------------------ status

    def status(self, job_id: str) -> Dict[str, Any]:
        record = self._refresh(self.store.get(job_id))
        progress = read_progress(record.output_path)

        return {
            "id": record.id,
            "task": record.task,
            "arm": record.arm,
            "state": record.state,
            "exit_code": record.exit_code,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
            "output_dir": record.output_dir,
            "adapter_path": record.adapter_path,
            **progress,
        }

    def list(self) -> List[Dict[str, Any]]:
        self._refresh_all()
        return [
            {"id": r.id, "task": r.task, "arm": r.arm, "state": r.state,
             "started_at": r.started_at, "output_dir": r.output_dir}
            for r in self.store.list()
        ]

    def logs(self, job_id: str, tail: int = 40) -> str:
        record = self.store.get(job_id)
        if not record.log_path.exists():
            return ""

        lines = record.log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    def cancel(self, job_id: str) -> Dict[str, Any]:
        record = self.store.get(job_id)

        # Identity is checked *before* the state check so the specific reason wins: never
        # signal a PID we cannot prove is still ours, because killpg on a recycled PID
        # would take out an unrelated process group.
        if record.state == "running" and not _is_our_process(record.pid, record.pid_starttime):
            record.state = "failed"
            record.ended_at = record.ended_at or _now()
            self.store.save()
            raise ToolHubError(
                f"Job '{job_id}' is no longer running (its process is gone, and PID "
                f"{record.pid} may since have been reused — refusing to signal it). "
                f"The record has been marked failed."
            )

        record = self._refresh(record)
        if record.state != "running":
            raise ToolHubError(f"Job '{job_id}' is not running (state: {record.state}).")

        try:
            os.killpg(os.getpgid(record.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as exc:
            raise ToolHubError(f"Could not signal job '{job_id}': {exc}") from exc

        record.state = "cancelled"
        record.ended_at = _now()
        self.store.save()

        return {"id": job_id, "state": record.state}

    # ------------------------------------------------------------------ refresh

    def _refresh_all(self) -> None:
        for record in list(self.store.jobs.values()):
            self._refresh(record, save=False)
        self.store.save()

    def _refresh(self, record: JobRecord, save: bool = True) -> JobRecord:
        """Reconcile a record with what is on disk. Terminal states are never revisited."""
        if record.state != "running":
            return record

        # Reap our own child if we started it, so a finished process stops looking alive.
        process = self._processes.get(record.id)
        returncode = process.poll() if process is not None else None

        exit_file = record.output_path / "exit_code"
        if exit_file.exists():
            try:
                code = int(exit_file.read_text().strip())
            except ValueError:
                code = 1
            record.exit_code = code
            record.state = "succeeded" if code == 0 else "failed"
            record.ended_at = record.ended_at or _now()
            record.adapter_path = record.adapter_path or _find_adapter(record)
        elif returncode is not None or not _is_our_process(record.pid, record.pid_starttime):
            # Gone without writing an exit code: SIGKILL, OOM, or a hard crash.
            record.state = "failed"
            record.exit_code = returncode
            record.ended_at = record.ended_at or _now()

        if save:
            self.store.save()

        return record


def _find_adapter(record: JobRecord) -> Optional[str]:
    candidates = sorted(record.output_path.glob("*_adapter.pt"))
    return str(candidates[0]) if candidates else None


# ---------------------------------------------------------------------- progress

def read_progress(output_dir: Union[str, Path]) -> Dict[str, Any]:
    """Latest epoch/step and metrics from the engine's `metrics.csv`.

    The engine writes it as the run goes (that is the documented interface), and its rows
    are sparse — each row carries only the metrics logged at that moment.
    """
    metrics_file = latest_metrics_file(output_dir)
    if metrics_file is None:
        return {"progress": None}

    import pandas as pd

    try:
        frame = pd.read_csv(metrics_file)
    except (pd.errors.EmptyDataError, OSError):
        return {"progress": None}

    if frame.empty:
        return {"progress": None}

    latest: Dict[str, Any] = {}
    for column in frame.columns:
        if column in ("epoch", "step"):
            continue
        series = frame[column].dropna()
        if not series.empty:
            latest[column] = round(float(series.iloc[-1]), 6)

    return {
        "progress": {
            "epoch": int(frame["epoch"].dropna().iloc[-1]) if "epoch" in frame else None,
            "step": int(frame["step"].dropna().iloc[-1]) if "step" in frame else None,
            "latest_metrics": latest,
        }
    }


def latest_metrics_file(output_dir: Union[str, Path]) -> Optional[Path]:
    """`<output_dir>/metrics/version_N/metrics.csv` for the highest N."""
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root

    versions = sorted(
        (root / "metrics").glob("version_*"),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else -1,
    )
    for version in reversed(versions):
        candidate = version / "metrics.csv"
        if candidate.exists():
            return candidate

    return None
