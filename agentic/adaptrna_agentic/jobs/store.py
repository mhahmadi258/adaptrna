"""Job records on disk (`jobs_data/jobs.json`).

Same atomic-write discipline as the ToolHub manifest. Job state must be derivable from
disk alone: training runs are detached and outlive the chat process that started them.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import contextlib
import json
import os
import tempfile

from adaptrna_agentic.settings import REPO_ROOT

FORMAT_VERSION = 1

DATA_DIR_VAR = "ADAPTRNA_JOBS_DIR"

RUNNING_STATES = ("running",)
TERMINAL_STATES = ("succeeded", "failed", "cancelled")


def resolve_jobs_dir(jobs_dir: Optional[Union[str, Path]] = None) -> Path:
    if jobs_dir is not None:
        return Path(jobs_dir).expanduser()

    env = os.environ.get(DATA_DIR_VAR)
    if env:
        return Path(env).expanduser()

    return REPO_ROOT / "jobs_data"


@dataclass
class JobRecord:
    id: str
    task: str
    arm: str
    command: List[str]
    output_dir: str
    state: str = "running"          # running | succeeded | failed | cancelled
    pid: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    adapter_path: Optional[str] = None
    plan: Dict[str, Any] = field(default_factory=dict)

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def log_path(self) -> Path:
        return self.output_path / "train.log"


class JobStore:
    def __init__(self, jobs_dir: Optional[Union[str, Path]] = None):
        self.jobs_dir = resolve_jobs_dir(jobs_dir)
        self.jobs: Dict[str, JobRecord] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self.jobs_dir / "jobs.json"

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Job store '{self.path}' is not valid JSON: {exc}") from exc

        version = payload.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Job store '{self.path}' has format version {version}; "
                f"this build reads {FORMAT_VERSION}"
            )

        self.jobs = {
            job_id: JobRecord(id=job_id, **record)
            for job_id, record in payload.get("jobs", {}).items()
        }

    def save(self) -> Path:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": FORMAT_VERSION,
            "jobs": {
                job_id: {k: v for k, v in asdict(record).items() if k != "id"}
                for job_id, record in sorted(self.jobs.items())
            },
        }

        fd, tmp_name = tempfile.mkstemp(dir=self.jobs_dir, prefix=".jobs.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise

        return self.path

    # ------------------------------------------------------------------ queries

    def add(self, record: JobRecord) -> JobRecord:
        self.jobs[record.id] = record
        self.save()
        return record

    def get(self, job_id: str) -> JobRecord:
        if job_id not in self.jobs:
            raise KeyError(f"No job '{job_id}'. Known jobs: {sorted(self.jobs)}")
        return self.jobs[job_id]

    def list(self) -> List[JobRecord]:
        return sorted(self.jobs.values(), key=lambda r: r.started_at or "", reverse=True)

    def running(self) -> List[JobRecord]:
        return [record for record in self.jobs.values() if record.state in RUNNING_STATES]
