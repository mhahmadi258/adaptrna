"""`prune` — the one destructive command, and the only one.

Three rules, in this order (plans/PHASE_7_HARDENING.md §5):

1. **Never** delete anything the manifest references — a registered tool's artifact, or
   the output directory of the job that produced one. Referenced items are listed as
   skipped, with the reason.
2. Dry run by default. `--yes` performs it.
3. `runs` deletes GPU-hours artifacts, so it always requires an explicit age filter.

Deliberately not exposed as an agent tool: deletion stays a human action at the CLI.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from adaptrna_agentic.settings import REPO_ROOT
from adaptrna_agentic.toolhub.errors import ToolHubError

KINDS = ("staging", "sessions", "jobs", "runs", "artifacts")


@dataclass
class Candidate:
    label: str
    path: Optional[Path] = None
    bytes: int = 0
    age_days: Optional[float] = None
    skip_reason: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def keep(self) -> bool:
        return bool(self.skip_reason)


def prune(
    kind: str,
    *,
    older_than: Optional[float] = None,
    apply: bool = False,
    data_dir=None,
    jobs_dir=None,
) -> Dict[str, Any]:
    """Plan (and optionally perform) a prune. Returns what was, or would be, removed."""
    if kind not in KINDS:
        raise ToolHubError(f"Unknown prune target '{kind}'. Known: {list(KINDS)}")

    from adaptrna_agentic.toolhub.registry import Registry

    registry = Registry(data_dir)

    if kind == "runs" and older_than is None:
        raise ToolHubError(
            "`prune runs` deletes training output directories — hours of GPU time. "
            "Give an explicit age, e.g. `--older-than 7`."
        )

    candidates = {
        "staging": lambda: _staging_candidates(registry, older_than),
        "artifacts": lambda: _artifact_candidates(registry),
        "sessions": lambda: _session_candidates(older_than),
        "jobs": lambda: _job_candidates(registry, jobs_dir, older_than),
        "runs": lambda: _run_candidates(registry, jobs_dir, older_than),
    }[kind]()

    removable = [c for c in candidates if not c.keep]
    kept = [c for c in candidates if c.keep]

    removed = []
    if apply:
        for candidate in removable:
            _remove(candidate)
            removed.append(candidate.label)

    return {
        "kind": kind,
        "applied": apply,
        "would_remove" if not apply else "removed": [
            {"label": c.label, "path": str(c.path) if c.path else None,
             "bytes": c.bytes, "age_days": c.age_days}
            for c in removable
        ],
        "kept": [{"label": c.label, "reason": c.skip_reason} for c in kept],
        "reclaimed_bytes": sum(c.bytes for c in removable),
    }


# ---------------------------------------------------------------------- candidates

def _age_days(path: Path) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 86400


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _older_enough(age: float, older_than: Optional[float]) -> bool:
    return older_than is None or age >= older_than


def _staging_candidates(registry, older_than) -> List[Candidate]:
    from adaptrna_agentic.codegen import staging

    out = []
    for stage in staging.list_stages(registry.data_dir):
        age = _age_days(stage.root)
        candidate = Candidate(label=stage.id, path=stage.root,
                              bytes=_dir_size(stage.root), age_days=round(age, 2))
        if not _older_enough(age, older_than):
            candidate.skip_reason = f"younger than {older_than} day(s)"
        out.append(candidate)

    return out


def _artifact_candidates(registry) -> List[Candidate]:
    """Only ever the orphans: anything a tool references is skipped by construction."""
    out = []
    for orphan in registry.verify()["orphan_artifacts"]:
        path = Path(orphan["artifact"])
        out.append(Candidate(label=path.name, path=path, bytes=orphan["size_bytes"],
                             age_days=round(_age_days(path), 2)))

    for entry in registry.list():
        path = entry.artifact_path()
        if path is not None and path.exists():
            out.append(Candidate(label=path.name, path=path, bytes=path.stat().st_size,
                                 skip_reason=f"referenced by tool '{entry.name}'"))

    return out


def _session_candidates(older_than) -> List[Candidate]:
    import sqlite3

    from adaptrna_agentic.cli.chat import chat_db_path

    db = chat_db_path()
    if not db.exists():
        return []

    connection = sqlite3.connect(db)
    try:
        rows = connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        connection.close()

    age = _age_days(db)
    return [
        Candidate(
            label=thread_id, path=None, bytes=0, age_days=round(age, 2),
            data={"thread_id": thread_id},
            skip_reason="" if _older_enough(age, older_than)
            else f"store younger than {older_than} day(s)",
        )
        for (thread_id,) in rows
    ]


def _protected_run_dirs(registry) -> Dict[str, str]:
    """Output directories that produced a registered tool — never prunable."""
    protected = {}
    for entry in registry.list():
        source = (entry.provenance or {}).get("source")
        if source:
            protected[str(Path(source).parent)] = entry.name

    return protected


def _job_candidates(registry, jobs_dir, older_than) -> List[Candidate]:
    from adaptrna_agentic.jobs.store import JobStore

    store = JobStore(jobs_dir)
    protected = _protected_run_dirs(registry)
    out = []

    for record in store.list():
        candidate = Candidate(label=record.id, path=None, bytes=0,
                              data={"job_id": record.id, "state": record.state})
        if record.state == "running":
            candidate.skip_reason = "still running"
        elif str(record.output_path) in protected:
            candidate.skip_reason = f"produced the registered tool '{protected[str(record.output_path)]}'"
        elif record.started_at:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(record.started_at)).total_seconds() / 86400
            candidate.age_days = round(age, 2)
            if not _older_enough(age, older_than):
                candidate.skip_reason = f"younger than {older_than} day(s)"
        out.append(candidate)

    return out


def _run_candidates(registry, jobs_dir, older_than) -> List[Candidate]:
    from adaptrna_agentic.jobs.store import JobStore

    store = JobStore(jobs_dir)
    protected = _protected_run_dirs(registry)
    running = {str(r.output_path) for r in store.list() if r.state == "running"}

    outputs = REPO_ROOT / "outputs"
    if not outputs.is_dir():
        return []

    out = []
    for directory in sorted(p for p in outputs.iterdir() if p.is_dir()):
        age = _age_days(directory)
        candidate = Candidate(label=directory.name, path=directory,
                              bytes=_dir_size(directory), age_days=round(age, 2))
        if str(directory) in protected:
            candidate.skip_reason = f"produced the registered tool '{protected[str(directory)]}'"
        elif str(directory) in running:
            candidate.skip_reason = "a job is still writing to it"
        elif not _older_enough(age, older_than):
            candidate.skip_reason = f"younger than {older_than} day(s)"
        out.append(candidate)

    return out


# ---------------------------------------------------------------------- removal

def _remove(candidate: Candidate) -> None:
    if candidate.data.get("thread_id"):
        _delete_session(candidate.data["thread_id"])
        return
    if candidate.data.get("job_id"):
        _delete_job(candidate.data["job_id"])
        return

    path = candidate.path
    if path is None or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _delete_session(thread_id: str) -> None:
    import sqlite3

    from adaptrna_agentic.cli.chat import chat_db_path

    connection = sqlite3.connect(chat_db_path())
    try:
        for table in ("checkpoints", "writes", "checkpoint_blobs"):
            try:
                connection.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                continue
        connection.commit()
    finally:
        connection.close()


def _delete_job(job_id: str, jobs_dir=None) -> None:
    from adaptrna_agentic.jobs.store import JobStore

    store = JobStore(jobs_dir)
    store.jobs.pop(job_id, None)
    store.save()


def format_report(report: Dict[str, Any]) -> str:
    verb = "removed" if report["applied"] else "would remove"
    items = report.get("removed" if report["applied"] else "would_remove", [])

    lines = [f"prune {report['kind']}: {verb} {len(items)} item(s), "
             f"{report['reclaimed_bytes'] / 1e6:.1f} MB"]
    for item in items:
        age = f", {item['age_days']}d old" if item.get("age_days") is not None else ""
        lines.append(f"  - {item['label']} ({item['bytes'] / 1e6:.1f} MB{age})")
    for kept in report["kept"]:
        lines.append(f"  · kept {kept['label']} — {kept['reason']}")
    if not report["applied"] and items:
        lines.append("\nThis was a dry run. Add --yes to perform it.")

    return "\n".join(lines)
