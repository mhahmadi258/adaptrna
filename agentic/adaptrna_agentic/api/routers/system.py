"""Liveness and install health."""

from fastapi import APIRouter, Depends

from adaptrna_agentic.api.app import get_services
from adaptrna_agentic.api.deps import Services

router = APIRouter(tags=["system"])


@router.get("/health")
def health(services: Services = Depends(get_services)) -> dict:
    """Cheap liveness: is the service up, and is the install broken?

    Deliberately status-only and unauthenticated — a probe should not need a token, and
    it should not leak paths.
    """
    from adaptrna_agentic.toolhub import doctor

    report = doctor.run_checks(services.registry.data_dir)

    return {
        "status": "ok",
        "install": report["status"],
        "failed_checks": report["failed"],
        "backbone_loaded": services.runtime.loaded,
    }


@router.get("/api/doctor")
def doctor_report(services: Services = Depends(get_services)) -> dict:
    """The full Phase 7 report: every check, with the command that fixes each failure."""
    from adaptrna_agentic.toolhub import doctor

    return doctor.run_checks(services.registry.data_dir)
