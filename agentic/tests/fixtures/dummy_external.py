"""A contract-compliant external wrapper for tests. Its "package" is the stdlib `json`
module, so every code path runs without touching PyPI."""

from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec,
    FunctionSpec,
    GoldenCase,
    PackageSpec,
)


def echo(value: str) -> dict:
    return {"value": value}


def add(a, b) -> dict:
    return {"total": float(a) + float(b)}


SPEC = ExternalToolSpec(
    name="dummy",
    description="Test-only external tool family.",
    package=PackageSpec(pip="dummy-package", import_name="json"),
    functions=(
        FunctionSpec(
            name="echo",
            description="Echo a value back.",
            golden=(GoldenCase(args={"value": "hi"}, expect={"value": "hi"}),),
        ),
        FunctionSpec(
            name="add",
            description="Add two numbers.",
            golden=(
                GoldenCase(args={"a": 2, "b": 3},
                           expect={"total": {"approx": 5.0, "tol": 0.001}}),
            ),
        ),
    ),
)
