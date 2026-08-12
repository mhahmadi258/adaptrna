"""`doctor` — one command that says what is wrong with an install, and never changes it.

Every check reports `ok | warn | fail` with a concrete remedy. The rule this module lives
by: a health check that reports green on a broken install is worse than no health check,
so each check is tested against a purpose-built broken install (the same discipline as the
Phase 6 verification harness controls).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from adaptrna_agentic.settings import REPO_ROOT

OK, WARN, FAIL = "ok", "warn", "fail"

#: Warn when a directory grows past this; GPU-hours artifacts are big and easy to forget.
LARGE_DIR_BYTES = 5 * 1024**3


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


def run_checks(data_dir=None, jobs_dir=None) -> Dict[str, Any]:
    """Every check, as plain data. Read-only."""
    checks: List[Check] = []

    from adaptrna_agentic.toolhub.registry import Registry

    registry = Registry(data_dir)

    checks.append(_engine_check())
    checks.append(_backbone_check(registry))
    checks.extend(_artifact_checks(registry))
    checks.append(_external_tools_check(registry))
    checks.append(_custom_tasks_check())
    checks.extend(_job_checks(jobs_dir))
    checks.append(_staging_check(registry))
    checks.extend(_disk_checks(registry))

    worst = FAIL if any(c.status == FAIL for c in checks) else (
        WARN if any(c.status == WARN for c in checks) else OK
    )

    return {
        "status": worst,
        "checks": [vars(c) for c in checks],
        "failed": [c.name for c in checks if c.status == FAIL],
        "warned": [c.name for c in checks if c.status == WARN],
    }


# ---------------------------------------------------------------------- checks

def _engine_check() -> Check:
    try:
        import rinalmo_hub
        from rinalmo_hub.adapter import git_sha
    except ImportError as exc:
        return Check("engine", FAIL, f"the engine package cannot be imported: {exc}",
                     "pip install -e ./engine")

    return Check("engine", OK, f"importable from {Path(rinalmo_hub.__file__).parent}",
                 data={"git_sha": git_sha()})


def _backbone_check(registry) -> Check:
    backbone = registry.manifest.backbone
    if not backbone.weights:
        return Check("backbone", WARN,
                     "no checkpoint configured — training would start from random weights",
                     "toolhub config --weights /path/to/giga-v1.pt")

    from adaptrna_agentic.toolhub.manifest import resolve_path

    path = resolve_path(backbone.weights)
    if not path.exists():
        return Check("backbone", FAIL, f"checkpoint '{path}' does not exist",
                     "toolhub config --weights /path/to/giga-v1.pt")

    size = path.stat().st_size
    return Check("backbone", OK, f"{backbone.lm_config} checkpoint at {path} "
                                 f"({size / 1e9:.2f} GB)")


def _artifact_checks(registry) -> List[Check]:
    report = registry.verify()
    checks = []

    if report["missing_artifacts"]:
        for missing in report["missing_artifacts"]:
            checks.append(Check(
                f"artifact:{missing['tool']}", FAIL,
                f"'{missing['tool']}' points at '{missing['artifact']}', which is gone",
                f"restore the file, or `toolhub remove {missing['tool']}`",
            ))
    if report["orphan_artifacts"]:
        total = sum(o["size_bytes"] for o in report["orphan_artifacts"])
        checks.append(Check(
            "orphan_artifacts", WARN,
            f"{len(report['orphan_artifacts'])} adapter file(s) no tool references "
            f"({total / 1e6:.1f} MB)",
            "toolhub prune artifacts",
            data={"artifacts": report["orphan_artifacts"]},
        ))
    if not checks:
        checks.append(Check("artifacts", OK,
                            f"{len(registry.list())} tool(s), all artifacts present"))

    return checks


def _external_tools_check(registry) -> Check:
    from adaptrna_agentic.toolhub.external import contract

    broken = []
    externals = [e for e in registry.list() if e.type == "external"]
    for entry in externals:
        package = (entry.external or {}).get("package") or {}
        import_name = package.get("import_name")
        if not import_name:
            continue
        import importlib.util

        if importlib.util.find_spec(import_name) is None:
            broken.append({"tool": entry.name, "package": package.get("pip")})

    if broken:
        names = ", ".join(f"{b['tool']} (needs {b['package']})" for b in broken)
        return Check("external_tools", FAIL, f"package missing for: {names}",
                     "reinstall the package, or remove the tool", data={"broken": broken})

    return Check("external_tools", OK, f"{len(externals)} external tool(s) importable")


def _custom_tasks_check() -> Check:
    from adaptrna_agentic.codegen.discovery import custom_task_names, describe_failures, load_all

    names = custom_task_names()
    failures = load_all()
    if failures:
        return Check("custom_tasks", FAIL,
                     f"generated task(s) fail to import: {describe_failures(failures)}",
                     "fix the code in adaptrna_custom/tasks/, or delete the package")

    return Check("custom_tasks", OK, f"{len(names)} generated task(s) import cleanly",
                 data={"tasks": names})


def _job_checks(jobs_dir=None) -> List[Check]:
    from adaptrna_agentic.jobs.runner import _is_our_process
    from adaptrna_agentic.jobs.store import JobStore

    store = JobStore(jobs_dir)
    checks, stale, missing_outputs = [], [], []

    for record in store.list():
        if record.state == "running" and not _is_our_process(record.pid, record.pid_starttime):
            stale.append(record.id)
        if record.state == "succeeded" and not record.output_path.exists():
            missing_outputs.append(record.id)

    if stale:
        checks.append(Check(
            "stale_jobs", FAIL,
            f"{len(stale)} job(s) recorded as running whose process is gone: {stale}",
            "toolhub doctor reconciles nothing; run `toolhub job-status <id>` to close them out",
            data={"jobs": stale},
        ))
    if missing_outputs:
        checks.append(Check(
            "job_outputs", WARN,
            f"{len(missing_outputs)} succeeded job(s) whose output directory is gone: "
            f"{missing_outputs}",
            "harmless if the run was pruned deliberately",
            data={"jobs": missing_outputs},
        ))
    if not checks:
        checks.append(Check("jobs", OK, f"{len(store.jobs)} job record(s), all consistent"))

    return checks


def _staging_check(registry) -> Check:
    from adaptrna_agentic.codegen import staging

    stages = staging.list_stages(registry.data_dir)
    if not stages:
        return Check("staging", OK, "no generated code awaiting approval")

    total = sum(
        sum(f.stat().st_size for f in stage.root.rglob("*") if f.is_file())
        for stage in stages
    )
    return Check(
        "staging", WARN,
        f"{len(stages)} staged artifact(s) never landed or cleaned "
        f"({total / 1e3:.0f} KB): {[s.id for s in stages]}",
        "review with `list_staged_code`, then land it or `toolhub prune staging`",
        data={"stages": [s.id for s in stages]},
    )


def _disk_checks(registry) -> List[Check]:
    checks = []
    for label, path in (
        ("outputs", REPO_ROOT / "outputs"),
        ("toolhub_data", registry.data_dir),
        ("chat_data", REPO_ROOT / "chat_data"),
    ):
        if not path.exists():
            continue
        size = _dir_size(path)
        status = WARN if size > LARGE_DIR_BYTES else OK
        remedy = f"toolhub prune runs --older-than 7" if label == "outputs" and status == WARN else ""
        checks.append(Check(f"disk:{label}", status, f"{path} is {size / 1e9:.2f} GB",
                            remedy, data={"bytes": size}))

    return checks


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"install status: {report['status'].upper()}"]
    for check in report["checks"]:
        marker = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}[check["status"]]
        lines.append(f"  [{marker}] {check['name']}: {check['detail']}")
        if check["remedy"] and check["status"] != OK:
            lines.append(f"           -> {check['remedy']}")

    return "\n".join(lines)
