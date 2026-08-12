"""The bounded ToolSmith ⇄ Verifier loop.

Plain Python: generate → stage → verify (deterministic harness) → review (independent
agent) → repeat with the failures as feedback, at most `max_iterations` times. A fourth
attempt at the same failing idea is rarely the fix, so the loop reports honestly and
stops (MASTER_PLAN §5).

Nothing here writes into the repository. A successful run leaves a *staged* artifact and
returns its diff; landing it is a separate, human-approved step.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adaptrna_agentic.agents import toolsmith, verifier
from adaptrna_agentic.codegen import harness, staging
from adaptrna_agentic.toolhub.errors import ToolHubError

MAX_ITERATIONS = 3


@dataclass
class Attempt:
    index: int
    files: Dict[str, str]
    harness_report: Dict[str, Any]
    review: Optional[Dict[str, Any]] = None
    ok: bool = False

    def summary(self) -> str:
        failed = self.harness_report.get("failed") or []
        findings = (self.review or {}).get("findings") or []
        if self.ok:
            return f"attempt {self.index}: passed"
        parts = []
        if failed:
            parts.append(f"harness failed: {failed}")
        if findings:
            parts.append(f"reviewer: {'; '.join(findings[:3])}")
        return f"attempt {self.index}: " + (" | ".join(parts) or "rejected")


@dataclass
class CodegenResult:
    ok: bool
    kind: str
    name: str
    attempts: List[Attempt] = field(default_factory=list)
    stage: Optional[staging.Stage] = None

    def to_dict(self) -> Dict[str, Any]:
        last = self.attempts[-1] if self.attempts else None
        payload: Dict[str, Any] = {
            "ok": self.ok,
            "kind": self.kind,
            "name": self.name,
            "iterations": len(self.attempts),
            "history": [a.summary() for a in self.attempts],
        }
        if last is not None:
            payload["harness"] = harness.summarize(last.harness_report)
            payload["review"] = last.review
        if self.stage is not None:
            payload["stage_id"] = self.stage.id
            payload["staging_path"] = str(self.stage.package_dir)
            payload["files"] = self.stage.summary()
        if not self.ok:
            payload["conclusion"] = (
                f"Gave up after {len(self.attempts)} attempt(s); nothing was written. "
                f"The last failure is above."
            )
        return payload


def create_task(
    task_name: str,
    description: str,
    profile: Dict[str, Any],
    *,
    sequences: Optional[List[str]] = None,
    max_iterations: int = MAX_ITERATIONS,
    toolsmith_model=None,
    verifier_model=None,
    data_dir=None,
    skip_review: bool = False,
) -> CodegenResult:
    """Generate, verify and review a new task. Leaves it staged; never lands it."""
    _reject_existing(task_name)

    result = CodegenResult(ok=False, kind="task", name=task_name)
    feedback: Optional[str] = None

    for index in range(1, max_iterations + 1):
        files = toolsmith.generate_task(
            task_name, description, profile, feedback=feedback, model=toolsmith_model
        )

        try:
            stage = staging.stage_task(task_name, files, data_dir=data_dir)
        except ToolHubError as exc:
            attempt = Attempt(index=index, files=files, harness_report={
                "ok": False, "task": task_name,
                "checks": [{"name": "files", "status": "fail", "detail": str(exc)}],
                "failed": ["files"],
            })
            result.attempts.append(attempt)
            feedback = f"Automated verification:\n{exc}"
            continue

        report = harness.verify_task(
            task_name,
            task_module=stage.module_path,
            config_path=str(stage.config_path) if stage.config_path else None,
            sys_path=[str(stage.root)],
            sequences=sequences,
        )
        summary = harness.summarize(report)

        # A generated task that was never exercised against the real data is not
        # verified, however green the structural checks look.
        unmet = harness.unmet_requirements(report)
        if unmet:
            report = dict(report)
            report["ok"] = False
            report["failed"] = list(report.get("failed") or []) + unmet
            summary += (
                f"\n  [FAIL] required checks did not run: {unmet}. The datamodule must "
                f"read the user's data from the paths in config.yaml (paths are resolved "
                f"from the repository root)."
            )

        review = None
        if report.get("ok") and not skip_review:
            review = verifier.review_task(
                description, profile, files, summary, model=verifier_model
            )

        approved = report.get("ok") and (skip_review or (review is not None and review.approved))
        attempt = Attempt(
            index=index, files=files, harness_report=report,
            review=review.model_dump() if review is not None else None,
            ok=bool(approved),
        )
        result.attempts.append(attempt)

        if approved:
            result.ok = True
            result.stage = stage
            return result

        feedback = verifier.format_feedback(summary, review)
        staging.discard(stage)

    return result


def create_external_tool(
    name: str,
    package: str,
    description: str,
    *,
    max_iterations: int = MAX_ITERATIONS,
    toolsmith_model=None,
    data_dir=None,
) -> CodegenResult:
    """Generate and verify an external wrapper against the Phase 3 contract."""
    from adaptrna_agentic.toolhub.external import contract

    result = CodegenResult(ok=False, kind="tool", name=name)
    feedback: Optional[str] = None

    for index in range(1, max_iterations + 1):
        content = toolsmith.generate_external_tool(
            package, description, feedback=feedback, model=toolsmith_model
        )
        stage = staging.stage_tool(name, content, data_dir=data_dir)

        checks, failures = _verify_wrapper(stage, contract)
        report = {"ok": not failures, "task": name, "checks": checks, "failed": failures}
        attempt = Attempt(index=index, files=stage.files, harness_report=report,
                          ok=not failures)
        result.attempts.append(attempt)

        if not failures:
            result.ok = True
            result.stage = stage
            return result

        feedback = harness.summarize(report)
        staging.discard(stage)

    return result


def _verify_wrapper(stage: staging.Stage, contract) -> tuple:
    """Contract validation + the wrapper's own golden cases, in-process.

    Wrapper modules are small and import a package the user already approved, so the
    sandbox's process isolation buys little here; the contract loader is the gate.
    """
    import sys

    checks, failures = [], []
    root = str(stage.root)
    sys.path.insert(0, root)
    try:
        try:
            spec, module = contract.load_spec(stage.module_path)
            checks.append({"name": "contract", "status": "pass",
                           "detail": f"SPEC valid; functions {[f.name for f in spec.functions]}"})
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": "contract", "status": "fail", "detail": str(exc)})
            return checks, ["contract"]

        if not contract.is_available(spec.package):
            checks.append({"name": "golden", "status": "skip",
                           "detail": f"package '{spec.package.pip}' is not installed"})
            return checks, failures

        for function in spec.functions:
            entry = _fake_entry(stage, function, contract)
            report = contract.run_golden(entry)
            status = "pass" if report["ok"] else "fail"
            checks.append({"name": f"golden:{function.name}", "status": status,
                           "detail": "; ".join(report["checks"])})
            if not report["ok"]:
                failures.append(f"golden:{function.name}")
    finally:
        if root in sys.path:
            sys.path.remove(root)

    return checks, failures


def _fake_entry(stage: staging.Stage, function, contract):
    from adaptrna_agentic.toolhub.manifest import ToolEntry

    return ToolEntry(
        name=f"{stage.name}_{function.name}", type="external", state="active",
        description=function.description,
        test={"golden": contract.golden_as_dicts(function)},
        external={"module": stage.module_path, "function": function.name, "package": {}},
    )


def _reject_existing(task_name: str) -> None:
    """Never silently overwrite a landed task."""
    from adaptrna_agentic.codegen.discovery import custom_task_names

    if task_name in custom_task_names():
        raise ToolHubError(
            f"A task named '{task_name}' already exists in adaptrna_custom/tasks/. "
            f"Landed code is yours: edit it directly, or choose another name."
        )
