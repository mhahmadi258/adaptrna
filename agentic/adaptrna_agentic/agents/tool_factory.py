"""ToolHub → LangChain bridge: manifest entries and lifecycle operations as agent tools.

The toolhub package stays LangChain-free; this module is the only place the two meet.

Binding policy (plans/PHASE_4_ORCHESTRATOR.md §2, amended by PHASE_10 §1): every registered
tool is bound — disabled entries carry a DISABLED note in their description — and
*execution* enforces state at call time through the shared Registry/Runtime.

Phase 4 bound everything so the model could activate a tool and use it within the same
turn. Phase 10 reversed that: a disabled tool is a statement of intent by the user, and an
agent that routes around it is not being helpful. Everything is still bound, but
activate_tool/deactivate_tool are in GATED_TOOLS, so the model can *ask* for a state change
and cannot *make* one.
"""

from dataclasses import asdict
from typing import Any, List, Optional
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
    "profile_dataset",
    "confirm_data_profile",
    "recommend_training_config",
    "start_training",
    "job_status",
    "list_jobs",
    "analyze_run",
    "register_trained_adapter",
    "create_task_tool",
    "create_external_tool",
    "list_staged_code",
    "land_generated_code",
)

#: Consequential operations: they burn GPU hours, write code into the repository, add a
#: servable tool, or change which tools the assistant may run at all — so the graph routes
#: them through the approval node before the tools node may execute them (MASTER_PLAN §3.2).
#:
#: The last of those is gated for a different reason than the others. Training and codegen
#: are gated for cost and blast radius; tool state is gated for *authority*. Enabling a tool
#: is cheap and trivially reversible, but the switch is how the user says which capabilities
#: they trust — so it is theirs, not the model's (PHASE_10 §1).
GATED_TOOLS = (
    "confirm_data_profile",
    "start_training",
    "register_trained_adapter",
    "land_generated_code",
    "activate_tool",
    "deactivate_tool",
)

#: Dotted paths a human may edit at the gate, per gated tool (Phase 13 §5). A path not on
#: a tool's list here is refused by `orchestrator._apply_edits` — an edit that cannot be
#: validated must fail loudly, because it is a training/codegen configuration.
#: `land_generated_code` is deliberately absent: editing generated code at the gate would
#: mean landing code the harness never actually verified.
EDITABLE_ARGS = {
    "confirm_data_profile": (
        "spec.sequence_column", "spec.label_column", "spec.target_type",
        "spec.task_name", "spec.tool_description", "spec.positive_class",
        "spec.on_invalid", "spec.split.*",
    ),
    "start_training": (
        "plan.overrides.*", "plan.seed", "plan.arm", "plan.quick_run",
    ),
}

#: Appended to adapter tool descriptions so the model knows the output type. A future
#: sec_struct adapter returns L×L matrices — its wrapper should cap and summarize rather
#: than dump matrices into context (see plan §7).
_TASK_OUTPUT_NOTES = {
    "splice_site": "Returns one probability per sequence that it contains a splice site.",
    "mrl": "Returns one predicted mean ribosome load per sequence (original scale).",
    "sec_struct": "Returns one base-pairing matrix per sequence (large output).",
}

_DISABLED_NOTE = (
    " (currently DISABLED — only the user can enable it. You may offer to ask them, which "
    "is what activate_tool('{name}') does; it does not switch the tool on by itself.)"
)


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
        """Ask the user to enable a disabled tool. Requires their approval.

        Tool state belongs to the user. This does not switch the tool on — it puts the
        request to them and only takes effect if they approve.
        """
        entry = registry.activate(name)
        return f"'{entry.name}' is now {entry.state}"

    def deactivate_tool(name: str) -> str:
        """Ask the user to disable a tool. Requires their approval.

        The tool stays registered and refuses calls while disabled.
        """
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
    ] + _pipeline_tools(registry, runtime)


def _pipeline_tools(registry: Registry, runtime: AdapterRuntime) -> List[BaseTool]:
    """Phase 5: profile data, recommend a config, train, watch, analyse, register."""
    from adaptrna_agentic.jobs.analysis import analyze_run as _analyze_run
    from adaptrna_agentic.jobs.runner import JobRunner
    from adaptrna_agentic.profiling.profiler import profile_dataset as _profile_dataset
    from adaptrna_agentic.profiling.recommender import recommend as _recommend

    job_runner = JobRunner()

    def profile_dataset(path: str) -> dict:
        """Profile one CSV/TSV table (optionally gzipped) and propose a dataset spec.

        Accepts exactly one delimited file with a sequence column and a label column;
        refuses a directory, a FASTA file, or a label column this build cannot train on
        (only binary, multiclass and regression targets are supported). Pass the result
        to confirm_data_profile for the user's approval — nothing is generated or
        trained from this call alone.
        """
        return _profile_dataset(path)

    def confirm_data_profile(spec: dict) -> dict:
        """Put the profiler's interpretation of a dataset to the user for approval.

        Pass the spec returned by profile_dataset unchanged. The user may correct the
        column choice, the target type, the split policy and the task name before
        approving. The approved spec is what create_task_tool consumes; nothing is
        generated without it. Requires user approval.
        """
        from adaptrna_agentic.profiling.profiler import confirm_profile as _confirm_profile

        return _confirm_profile(spec)

    def recommend_training_config(
        task: str,
        spec: Optional[dict] = None,
        arm: str = "lora",
        quick: bool = False,
        seed: int = 42,
    ) -> dict:
        """Propose a validated training plan for a landed task.

        Batch size and epoch count are derived from the spec the task landed with
        (adaptrna_custom/tasks/<task>/spec.json); pass spec explicitly only to point an
        existing task at a new file of the same shape (reuse — data.root changes, the
        code does not). Returns the task, arm, config overrides, the exact command, an
        ETA, the rationale behind every setting, and any caveats. All values come from
        the project's knowledge base of validated runs — never invent hyperparameters
        yourself.
        """
        return _recommend(
            task, spec=spec, arm=arm, quick=quick, seed=seed, registry=registry,
        )

    def start_training(plan: dict) -> dict:
        """Launch a training plan on the local GPU as a background job.

        Pass the plan object returned by recommend_training_config unchanged. Requires
        user approval.
        """
        from adaptrna_agentic.profiling.recommender import PLAN_SOURCE

        if plan.get("source") != PLAN_SOURCE:
            raise ToolHubError(
                "This plan did not come from recommend_training_config. Hyperparameters "
                "must come from the project's knowledge base of validated runs, not from "
                "you — call recommend_training_config and pass its result through "
                "unchanged. If it refused, report why rather than assembling a plan."
            )

        record = job_runner.start(plan)
        return {"job_id": record.id, "state": record.state,
                "output_dir": record.output_dir,
                "estimated_wall_clock": plan.get("estimated_wall_clock")}

    def job_status(job_id: str) -> dict:
        """Progress and state of a training job (epoch, step, latest metrics)."""
        return job_runner.status(job_id)

    def list_jobs() -> list:
        """Every training job this project has launched, newest first."""
        return job_runner.list()

    def analyze_run(job_id: str) -> dict:
        """Analyse a finished run: metrics, verdict, and remedies for anything wrong."""
        record = job_runner.store.get(job_id)
        return _analyze_run(record.output_path, plan=record.plan)

    def register_trained_adapter(
        job_id: str, name: Optional[str] = None, description: Optional[str] = None,
    ) -> dict:
        """Register a finished run's adapter as a servable tool. Requires user approval."""
        record = job_runner.store.get(job_id)
        status = job_runner.status(job_id)

        if status["state"] != "succeeded":
            raise ToolHubError(
                f"Job '{job_id}' is {status['state']}, not succeeded — nothing to register."
            )
        if not record.adapter_path:
            raise ToolHubError(
                f"Job '{job_id}' produced no adapter file. LoRA runs write one; full "
                f"fine-tuning runs do not (and cannot become served tools)."
            )

        entry = registry.register(record.adapter_path, name=name, description=description)
        entry.provenance["job_id"] = job_id
        entry.provenance["training_metrics"] = (status.get("progress") or {}).get(
            "latest_metrics", {}
        )
        registry.manifest.save()

        # A tool registered after the hub was built must become resident on next use.
        return asdict(entry)

    return [
        StructuredTool.from_function(func=_surface_errors(func), handle_tool_error=True)
        for func in (
            profile_dataset, confirm_data_profile, recommend_training_config, start_training,
            job_status, list_jobs, analyze_run, register_trained_adapter,
        )
    ] + _codegen_tools(registry, runtime)


#: Staged artifacts awaiting approval, keyed by stage id. Held in the chat process only:
#: a stage that is never landed leaves nothing behind but a temp directory.
_STAGES: dict = {}


def _codegen_tools(registry: Registry, runtime: AdapterRuntime) -> List[BaseTool]:
    """Phase 6: write, verify and land new tasks and external wrappers."""
    from adaptrna_agentic.codegen import pipeline, staging

    def create_task_tool(name: str, description: str, data_path: str) -> dict:
        """Build a NEW engine task for data no existing task can read.

        Writes the three files (task module, datamodule, config), verifies them against a
        real forward/backward pass and an adapter round trip, and has them reviewed. The
        code is only staged — call land_generated_code to write it into the project.
        """
        from adaptrna_agentic.profiling.profiler import profile_dataset as _profile

        profile = _profile(data_path)
        result = pipeline.create_task(name, description, profile)
        if result.stage is not None:
            _STAGES[result.stage.id] = result.stage

        return result.to_dict()

    def create_external_tool(name: str, package: str, description: str) -> dict:
        """Build a wrapper around a classical bioinformatics package.

        Verifies it against the wrapper contract and its own golden cases. Staged only —
        call land_generated_code to write it into the project.
        """
        result = pipeline.create_external_tool(name, package, description)
        if result.stage is not None:
            _STAGES[result.stage.id] = result.stage

        return result.to_dict()

    def list_staged_code() -> list:
        """Generated code waiting for approval, including from earlier sessions."""
        return [
            {"stage_id": s.id, "kind": s.kind, "name": s.name,
             "files": s.summary(), "staging_path": str(s.package_dir)}
            for s in staging.list_stages()
        ]

    def land_generated_code(stage_id: str) -> dict:
        """Write staged, verified code into the project. Requires user approval."""
        stage = _STAGES.get(stage_id) or staging.load_stage(stage_id)
        if stage is None:
            raise ToolHubError(
                f"No staged artifact '{stage_id}'. Use list_staged_code to see what is "
                f"awaiting approval."
            )

        written = staging.land(stage)
        _STAGES.pop(stage_id, None)

        result: dict = {
            "landed": written,
            "name": stage.name,
            "kind": stage.kind,
            "module": stage.module_path,
        }

        if stage.kind == "tool":
            from adaptrna_agentic.codegen import discovery
            import importlib
            import sys

            discovery.ensure_importable()

            # The verification pipeline imported the module from the staging
            # directory; evict those cached copies so the just-landed file is
            # picked up instead.
            for key in list(sys.modules):
                if key == stage.module_path or key in (
                    "adaptrna_custom", "adaptrna_custom.tools"
                ):
                    del sys.modules[key]
            importlib.invalidate_caches()

            entries = registry.register_external(stage.module_path)
            names = [e.name for e in entries]
            result["registered"] = names
            result["next"] = (
                f"Registered {len(names)} tool(s): {', '.join(names)} — "
                f"active and usable now."
            )
        else:
            result["next"] = (
                "The task is registered on next use — recommend_training_config can now "
                "target it."
            )

        return result

    return [
        StructuredTool.from_function(func=_surface_errors(func), handle_tool_error=True)
        for func in (create_task_tool, create_external_tool, list_staged_code,
                     land_generated_code)
    ]


def staged(stage_id: str):
    """The staged artifact behind a stage id (used by the approval gate to show a diff).

    Falls back to disk: staging directories outlive the session that created them, so a
    user can review the code in an editor and approve it later.
    """
    from adaptrna_agentic.codegen import staging

    return _STAGES.get(stage_id) or staging.load_stage(stage_id)


# ---------------------------------------------------------------------- capability tools

class _SequencesInput(BaseModel):
    sequences: List[str] = Field(
        description="RNA/DNA sequences as plain ACGU/T strings, one per prediction."
    )


def _check_active(registry: Registry, name: str) -> None:
    if not registry.get(name).active:
        raise ToolException(
            f"Tool '{name}' is disabled. Only the user can enable it — from the Tools "
            f"panel, or with 'toolhub activate {name}'. Tell them it is off and offer to "
            f"ask on their behalf; do not treat this as something you can resolve yourself."
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
    from adaptrna_agentic.codegen.discovery import ensure_importable

    ensure_importable()
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
