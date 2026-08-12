"""Run the verification harness for a task, in the sandbox, and summarise the report."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import tempfile

from adaptrna_agentic.codegen.sandbox import DEFAULT_TIMEOUT, run_python
from adaptrna_agentic.settings import REPO_ROOT

RUNNER_MODULE = "adaptrna_agentic.codegen._harness_runner"

#: Checks that need no dataset — used as the fast control over the shipped tasks.
STRUCTURAL_CHECKS = ("import", "config_head", "adapter_roundtrip", "serving")

#: Checks that must actually RUN before generated code may be approved. A skipped
#: datamodule check means the code was never exercised against real data — treating that
#: as success is precisely the "harness that passes everything" failure this design is
#: supposed to avoid.
REQUIRED_FOR_GENERATED = ("datamodule", "forward_backward", "metrics")


def verify_task(
    task_name: str,
    *,
    task_module: Optional[str] = None,
    config_path: Optional[str] = None,
    sequences: Optional[List[str]] = None,
    only: Optional[List[str]] = None,
    sys_path: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Verify a task (generated or shipped). Returns a report:

        {ok, task, checks: [{name, status, detail}], failed: [...], sandbox: {...}}

    Never raises for the task's misbehaviour — a crash, a hang and a failed check are all
    results the verification report needs to describe.
    """
    spec = {
        "task_name": task_name,
        "task_module": task_module,
        "config_path": config_path,
        "sequences": sequences,
        "only": only,
        "sys_path": [str(p) for p in (sys_path or [])],
    }

    with tempfile.TemporaryDirectory(prefix="adaptrna-harness-") as scratch:
        spec_path = Path(scratch) / "spec.json"
        spec_path.write_text(json.dumps(spec))

        # Run from the repo root, exactly as the JobRunner runs real training. Verifying
        # under a different working directory than the code will actually run under makes
        # repo-relative data paths unresolvable, which silently *skips* the checks that
        # matter most (the datamodule against real data) instead of failing.
        result = run_python(
            ["-m", RUNNER_MODULE, str(spec_path)], timeout=timeout, cwd=REPO_ROOT
        )

    if result.payload is None:
        return {
            "task": task_name,
            "ok": False,
            "checks": [{
                "name": "harness",
                "status": "fail",
                "detail": result.failure_summary,
                "stderr_tail": (result.stderr or "").strip().splitlines()[-15:],
            }],
            "failed": ["harness"],
            "sandbox": {"timed_out": result.timed_out, "returncode": result.returncode,
                        "limits": result.limits},
        }

    report = dict(result.payload)
    report["sandbox"] = {"timed_out": result.timed_out, "returncode": result.returncode,
                         "limits": result.limits}

    return report


def unmet_requirements(report: Dict[str, Any], required=REQUIRED_FOR_GENERATED) -> List[str]:
    """Required checks that did not actually pass (failed *or* skipped)."""
    statuses = {c["name"]: c["status"] for c in report.get("checks", [])}

    return [name for name in required if statuses.get(name) != "pass"]


def summarize(report: Dict[str, Any]) -> str:
    """One-line-per-check summary, for the verifier prompt and the CLI."""
    lines = [f"task '{report.get('task')}': {'PASS' if report.get('ok') else 'FAIL'}"]
    for check in report.get("checks", []):
        marker = {"pass": "ok  ", "skip": "skip", "fail": "FAIL"}.get(check["status"], "?")
        lines.append(f"  [{marker}] {check['name']}: {check.get('detail', '')}")

    return "\n".join(lines)
