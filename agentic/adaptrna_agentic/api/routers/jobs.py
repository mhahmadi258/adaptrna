"""Training jobs: read-only, plus cancel.

Starting a job is deliberately not here — it happens through the chat, behind the
approval gate, so the human sees the exact command before GPU time is spent.
"""

from fastapi import APIRouter, Depends, Query

from adaptrna_agentic.api.app import get_services
from adaptrna_agentic.api.deps import Services

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _runner():
    from adaptrna_agentic.jobs.runner import JobRunner

    return JobRunner()


@router.get("")
def list_jobs(_services: Services = Depends(get_services)) -> list:
    return _runner().list()


@router.get("/{job_id}")
def job_status(job_id: str, _services: Services = Depends(get_services)) -> dict:
    return _runner().status(job_id)


@router.get("/{job_id}/logs")
def job_logs(
    job_id: str,
    tail: int = Query(default=40, ge=1, le=2000),
    _services: Services = Depends(get_services),
) -> dict:
    return {"job_id": job_id, "tail": tail, "log": _runner().logs(job_id, tail=tail)}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, _services: Services = Depends(get_services)) -> dict:
    """Refuses — as a 409 — when the job is not running, or when its PID can no longer
    be proved to be ours (Phase 7's recycled-PID guard)."""
    return _runner().cancel(job_id)


@router.get("/{job_id}/analysis")
def job_analysis(job_id: str, _services: Services = Depends(get_services)) -> dict:
    from adaptrna_agentic.jobs.analysis import analyze_run

    record = _runner().store.get(job_id)

    return analyze_run(record.output_path, plan=record.plan)
