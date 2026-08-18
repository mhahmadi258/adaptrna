"""The external-tool wrapper contract — and the template Phase 6's ToolSmith imitates.

A wrapper module must:

1. define `SPEC: ExternalToolSpec` at module level — the declarative description of the
   tool family: its package, its functions, and their golden test cases;
2. define one module-level callable per `FunctionSpec.name`, taking JSON-scalar keyword
   arguments and returning a JSON-serializable dict (agent-tool-ready for Phase 4);
3. validate inputs BEFORE importing the wrapped package, so a missing package fails at
   the call boundary (with the install hint) and validation tests run without it.

Golden `expect` values: plain values compare exactly; `{"approx": x, "tol": t}` compares
within tolerance. Goldens are captured against the installed package version (recorded in
the manifest provenance at registration) — never guessed.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import importlib
import importlib.util
import subprocess
import sys

from adaptrna_agentic.toolhub.errors import ToolHubError


@dataclass(frozen=True)
class PackageSpec:
    pip: str            # distribution name, e.g. "ViennaRNA"
    import_name: str    # top-level module it provides, e.g. "RNA"


@dataclass(frozen=True)
class GoldenCase:
    args: Dict[str, Any]
    expect: Dict[str, Any]


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    description: str
    golden: Tuple[GoldenCase, ...] = ()


@dataclass(frozen=True)
class ExternalToolSpec:
    name: str                              # family prefix: tools register as <name>_<fn>
    description: str
    package: PackageSpec
    functions: Tuple[FunctionSpec, ...]


# ---------------------------------------------------------------------- loading

def load_spec(module_path: str):
    """Import a wrapper module and validate the contract. Returns (spec, module)."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ToolHubError(f"Cannot import wrapper module '{module_path}': {exc}") from exc

    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, ExternalToolSpec):
        raise ToolHubError(
            f"'{module_path}' does not define `SPEC: ExternalToolSpec` — it is not an "
            f"external-tool wrapper module (see toolhub/external/contract.py)."
        )

    for function in spec.functions:
        target = getattr(module, function.name, None)
        if target is None:
            raise ToolHubError(
                f"SPEC of '{module_path}' declares function '{function.name}' but the "
                f"module does not define it."
            )
        if not callable(target):
            raise ToolHubError(
                f"'{module_path}.{function.name}' is declared in SPEC but is not callable."
            )

    return spec, module


# ---------------------------------------------------------------------- install flow

def is_available(package: PackageSpec) -> bool:
    return importlib.util.find_spec(package.import_name) is not None


def installed_version(package: PackageSpec) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package.pip)
    except PackageNotFoundError:
        return None


def install_command(package: PackageSpec) -> List[str]:
    """The exact command the gated install would run — constructed, never executed here."""
    return [sys.executable, "-m", "pip", "install", package.pip]


def install(package: PackageSpec) -> Optional[str]:
    """Run the install command (only ever the spec-named package, into this venv).
    Returns the installed version. Callers own the approval gate."""
    command = install_command(package)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ToolHubError(
            f"Install failed ({' '.join(command)}):\n{result.stderr.strip()[-2000:]}"
        )

    importlib.invalidate_caches()
    return installed_version(package)


# ---------------------------------------------------------------------- execution

def call_entry(entry, arguments: Dict[str, Any]):
    """Invoke an external tool entry's wrapped function with keyword arguments."""
    external = entry.external or {}
    module_path, function_name = external.get("module"), external.get("function")
    if not module_path or not function_name:
        raise ToolHubError(f"Tool '{entry.name}' has no external module/function recorded.")

    module = importlib.import_module(module_path)
    return getattr(module, function_name)(**arguments)


# ---------------------------------------------------------------------- golden tests

def run_golden(entry) -> Dict[str, Any]:
    """Golden-test report for an external tool entry (same shape as the adapter smoke
    test): {name, ok, checks, outputs}."""
    if not entry.active:
        return {"name": entry.name, "ok": False,
                "checks": [f"tool is disabled — `toolhub activate {entry.name}` first"]}

    golden = (entry.test or {}).get("golden") or []
    if not golden:
        return {"name": entry.name, "ok": False, "checks": ["no golden cases configured"]}

    checks: List[str] = []
    outputs: List[Any] = []
    ok = True

    for i, case in enumerate(golden):
        args, expect = case["args"], case["expect"]
        try:
            result = call_entry(entry, dict(args))
        except Exception as exc:  # noqa: BLE001 -- the report is the error channel
            checks.append(f"case {i} {args}: call failed: {exc}: FAIL")
            ok = False
            continue

        outputs.append(result)
        problems = _compare(result, expect)
        if problems:
            checks.extend(f"case {i} {args}: {p}: FAIL" for p in problems)
            ok = False
        else:
            checks.append(f"case {i} {args}: ok")

    return {"name": entry.name, "ok": ok, "checks": checks, "outputs": outputs}


def _compare(result: Dict[str, Any], expect: Dict[str, Any]) -> List[str]:
    problems = []
    for key, expected in expect.items():
        if not isinstance(result, dict) or key not in result:
            problems.append(f"missing key '{key}' in result")
            continue

        actual = result[key]
        if isinstance(expected, dict) and "approx" in expected:
            tolerance = float(expected.get("tol", 1e-6))
            try:
                delta = abs(float(actual) - float(expected["approx"]))
            except (TypeError, ValueError):
                problems.append(f"'{key}' is not numeric: {actual!r}")
                continue
            if delta > tolerance:
                problems.append(
                    f"'{key}' = {actual} differs from {expected['approx']} by {delta:.4g} "
                    f"(tol {tolerance})"
                )
        elif actual != expected:
            problems.append(f"'{key}' = {actual!r}, expected {expected!r}")

    return problems


def golden_as_dicts(function: FunctionSpec) -> List[Dict[str, Any]]:
    """Golden cases as plain dicts for the manifest (self-contained per entry)."""
    return [asdict(case) for case in function.golden]
