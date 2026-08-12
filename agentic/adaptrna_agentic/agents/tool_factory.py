"""ToolHub → LangChain bridge: manifest entries and lifecycle operations as agent tools.

The toolhub package stays LangChain-free; this module is the only place the two meet.

Binding policy (plans/PHASE_4_ORCHESTRATOR.md §2): every registered tool is bound —
disabled entries carry a DISABLED note in their description — and *execution* enforces
state at call time through the shared Registry/Runtime. That is what lets the model
activate a tool and use it within the same turn, and the refusal message teaches it the
activate-first lifecycle.
"""

from dataclasses import asdict
from typing import Any, List
import functools
import importlib
import json

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from adaptrna_agentic.toolhub.errors import ToolHubError
from adaptrna_agentic.toolhub.registry import Registry
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

MANAGEMENT_TOOL_NAMES = (
    "list_tools",
    "tool_info",
    "activate_tool",
    "deactivate_tool",
    "test_tool",
)

#: Appended to adapter tool descriptions so the model knows the output type. A future
#: sec_struct adapter returns L×L matrices — its wrapper should cap and summarize rather
#: than dump matrices into context (see plan §7).
_TASK_OUTPUT_NOTES = {
    "splice_site": "Returns one probability per sequence that it contains a splice site.",
    "mrl": "Returns one predicted mean ribosome load per sequence (original scale).",
    "sec_struct": "Returns one base-pairing matrix per sequence (large output).",
}

_DISABLED_NOTE = " (currently DISABLED — call activate_tool('{name}') first.)"


def _surface_errors(func):
    """Convert ToolHub refusals/validation errors into ToolExceptions so
    `handle_tool_error=True` returns them as tool results the model can act on."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ToolHubError, KeyError, ValueError, FileNotFoundError) as exc:
            message = exc.args[0] if exc.args else str(exc)
            raise ToolException(str(message)) from exc

    return wrapper


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


# ---------------------------------------------------------------------- management tools

def _management_tools(registry: Registry, runtime: AdapterRuntime) -> List[BaseTool]:
    def list_tools() -> list:
        """List every registered tool: name, type, state, task and description."""
        return [
            {"name": e.name, "type": e.type, "state": e.state,
             "task": e.task, "description": e.description}
            for e in registry.list()
        ]

    def tool_info(name: str) -> dict:
        """Full manifest entry for one tool (serving policy, provenance, test spec)."""
        return asdict(registry.get(name))

    def activate_tool(name: str) -> str:
        """Activate a disabled tool so it can be called again."""
        entry = registry.activate(name)
        return f"'{entry.name}' is now {entry.state}"

    def deactivate_tool(name: str) -> str:
        """Disable a tool; it stays registered but refuses calls until re-activated."""
        entry = registry.deactivate(name)
        return f"'{entry.name}' is now {entry.state}"

    def test_tool(name: str) -> dict:
        """Run a tool's smoke/golden tests and return the report."""
        entry = registry.get(name)
        if entry.type == "external":
            from adaptrna_agentic.toolhub.external.contract import run_golden

            return run_golden(entry)
        return runtime.smoke_test(name)

    return [
        StructuredTool.from_function(func=_surface_errors(func), handle_tool_error=True)
        for func in (list_tools, tool_info, activate_tool, deactivate_tool, test_tool)
    ]


# ---------------------------------------------------------------------- capability tools

class _SequencesInput(BaseModel):
    sequences: List[str] = Field(
        description="RNA/DNA sequences as plain ACGU/T strings, one per prediction."
    )


def _check_active(registry: Registry, name: str) -> None:
    if not registry.get(name).active:
        raise ToolException(
            f"Tool '{name}' is disabled. Call activate_tool('{name}') first."
        )


def _adapter_tool(entry, registry: Registry, runtime: AdapterRuntime) -> BaseTool:
    name = entry.name

    @_surface_errors
    def run(sequences: List[str]) -> Any:
        _check_active(registry, name)
        return _jsonable(runtime.predict(name, sequences))

    description = entry.description
    note = _TASK_OUTPUT_NOTES.get(entry.task)
    if note and note not in description:
        description = f"{description} {note}"
    if not entry.active:
        description += _DISABLED_NOTE.format(name=name)

    return StructuredTool.from_function(
        func=run, name=name, description=description,
        args_schema=_SequencesInput, handle_tool_error=True,
    )


def _external_tool(entry, registry: Registry) -> BaseTool:
    name = entry.name
    module = importlib.import_module(entry.external["module"])
    target = getattr(module, entry.external["function"])

    # functools.wraps preserves the wrapper function's real signature, so
    # StructuredTool.from_function infers the args schema from it — the contract's
    # typed JSON-scalar kwargs paying off.
    @functools.wraps(target)
    def run(*args, **kwargs):
        _check_active(registry, name)
        try:
            return target(*args, **kwargs)
        except (ToolHubError, ValueError) as exc:
            raise ToolException(str(exc)) from exc

    description = entry.description
    if not entry.active:
        description += _DISABLED_NOTE.format(name=name)

    return StructuredTool.from_function(
        func=run, name=name, description=description, handle_tool_error=True,
    )


# ---------------------------------------------------------------------- entry point

def build_agent_tools(registry: Registry, runtime: AdapterRuntime) -> List[BaseTool]:
    """Every management operation plus every registered tool, as LangChain tools."""
    tools = _management_tools(registry, runtime)

    for entry in registry.list():
        if entry.name in MANAGEMENT_TOOL_NAMES:
            raise ToolHubError(
                f"Tool '{entry.name}' collides with a built-in management tool name; "
                f"re-register it under a different --name."
            )
        if entry.type == "adapter":
            tools.append(_adapter_tool(entry, registry, runtime))
        else:
            tools.append(_external_tool(entry, registry))

    return tools


def stringify_tool_output(output: Any) -> str:
    """ToolMessage content must be text; management/capability outputs may be data."""
    return output if isinstance(output, str) else json.dumps(output, default=str)
