"""The AdaptRNA orchestrator — the one agent the user talks to (MASTER_PLAN §5).

Hand-built wiring (model ⇄ tools loop) with the ToolHub bound as agent tools, an optional
checkpointer for persistent sessions, and — from Phase 5 — a human approval gate in front
of consequential operations.

The gate is a **dedicated node**, not an `interrupt()` inside a tool. The tools node runs
every tool call of a turn in one pass, so interrupting from within it would re-execute the
calls that already completed when the turn resumes. A node containing only `interrupt()`
is idempotent: on resume it returns the decision instead of raising.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from adaptrna_agentic.agents.tool_factory import (
    EDITABLE_ARGS,
    GATED_TOOLS,
    build_agent_tools,
    stringify_tool_output,
)
from adaptrna_agentic.toolhub.registry import Registry
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

SYSTEM_PROMPT = """\
You are the AdaptRNA assistant: a conversational interface over one shared RNA
foundation-model backbone, plus whatever adapter and external tools have been built and
registered on it. A fresh install starts with **no tools at all** — your first job with a
new user is usually to turn one CSV of theirs into one.

Use the tools for any prediction or computation — never guess sequence properties
yourself. Report tool outputs faithfully: probabilities are probabilities, not
certainties. Use list_tools / tool_info / test_tool when the user asks about capabilities.

Which tools are enabled is the user's decision, not yours. A disabled tool is a deliberate
choice they made — never work around it, and never treat enabling it as a step you can take
to get on with a task. Say plainly that the tool is off, and stop there unless they want it
back on. activate_tool and deactivate_tool do not switch anything: they put the request to
the user and take effect only if they approve.

## Building a new tool from one file

The system accepts one CSV or TSV with a sequence column and a label column — binary,
multiclass, or regression. If what the user has is something else (multiple files, a
FASTA, per-position labels, more than a handful of free-text classes), say so plainly
rather than proposing a workaround.

Four steps, each a **separate request** — after a gate resolves, report what happened and
stop; do not begin the next step until the user asks for it:

1. **Profile.** `profile_dataset` reads the file and proposes an interpretation: which
   column is the sequence, which is the label, the target type, a split, and any data
   quality warnings. `confirm_data_profile` is the gate — present the interpretation and
   its warnings exactly as given, especially duplicate/leakage warnings; the user may edit
   any field at the gate, and their edits are decisions, not suggestions to argue with.
2. **Build.** `create_task_tool` turns the approved spec into runnable code (a template
   renders it deterministically when the spec allows; otherwise it's generated and
   independently reviewed) and stages it for review. `land_generated_code` is the gate.
3. **Train.** `recommend_training_config` proposes hyperparameters — never invent your
   own; they come from a knowledge base entry together with the reasoning behind each
   setting, which you present instead of your own reasoning. `start_training` is the gate;
   training then runs in the background, so tell the user how long it should take and let
   them keep working. `job_status` follows it; `analyze_run` judges the result and reports
   its verdict and reasons as given — never present a truncated smoke run as a real result.
4. **Serve.** `register_trained_adapter` turns a finished run into a servable tool; it is
   also a gate.

start_training, register_trained_adapter, land_generated_code, activate_tool and
deactivate_tool all pause for the user's approval; if the user declines, report that
plainly, with the reason if one was given, and do not retry the same action with different
arguments.

Sequences arrive as plain ACGU/T strings. Keep answers concise and grounded in the tool
results you actually received."""


class OrchestratorState(MessagesState):
    """Messages plus the human decisions recorded for this turn's gated tool calls."""

    approvals: Dict[str, Any]


def _tool_state(registry: Optional[Registry], name: str) -> str:
    """A gated toggle names the tool's current state; an unknown name must not break the
    gate — the tool call itself will fail with a proper error a moment later."""
    if registry is None:
        return "unknown"
    try:
        return registry.get(name).state
    except (KeyError, ValueError):
        return "unknown"


def _summarize(call: Dict[str, Any], registry: Optional[Registry] = None) -> str:
    """One line describing what approving this call would actually do."""
    name, args = call["name"], call.get("args", {})

    if name in ("activate_tool", "deactivate_tool"):
        target = args.get("name", "?")
        verb = "Enable" if name == "activate_tool" else "Disable"
        return f"{verb} the tool '{target}' (currently {_tool_state(registry, target)})"

    if name == "confirm_data_profile":
        return _summarize_profile(args.get("spec") or {})

    if name == "start_training":
        plan = args.get("plan") or {}
        return (
            f"Train {plan.get('task', '?')} ({plan.get('arm', '?')}) — "
            f"ETA {plan.get('estimated_wall_clock', 'unknown')}, "
            f"output {plan.get('output_dir', '?')}"
        )
    if name == "register_trained_adapter":
        return (
            f"Register job '{args.get('job_id', '?')}' as the servable tool "
            f"'{args.get('name') or 'its task name'}'"
        )
    if name == "land_generated_code":
        from adaptrna_agentic.agents.tool_factory import staged

        stage = staged(args.get("stage_id", ""))
        if stage is None:
            return f"Write staged code '{args.get('stage_id', '?')}' into the project"
        files = ", ".join(f["path"] for f in stage.summary())
        return f"Write generated {stage.kind} '{stage.name}' into the project: {files}"

    return f"{name}({args})"


def _summarize_profile(spec: Dict[str, Any]) -> str:
    row_counts = ((spec.get("split") or {}).get("row_counts")) or {}
    counts = "/".join(str(row_counts.get(k, "?")) for k in ("train", "val", "test"))
    rows = (spec.get("format") or {}).get("rows")
    rows_str = f"{rows:,}" if isinstance(rows, int) else "?"
    file_name = Path(spec.get("path") or "").name or "?"

    return (
        f"Use column '{spec.get('sequence_column', '?')}' as the sequence and "
        f"'{spec.get('label_column', '?')}' as a {spec.get('target_type', '?')} target "
        f"from {file_name} ({rows_str} rows → {counts} train/val/test), and build task "
        f"'{spec.get('task_name', '?')}'"
    )


def _details(call: Dict[str, Any], registry: Optional[Registry] = None) -> Dict[str, Any]:
    """Everything the human should see before approving — notably the exact command."""
    args = call.get("args", {})
    details: Dict[str, Any] = {}

    if call["name"] in ("activate_tool", "deactivate_tool"):
        target = args.get("name", "?")
        details["tool"] = target
        details["current_state"] = _tool_state(registry, target)
        details["after_approval"] = (
            "active" if call["name"] == "activate_tool" else "disabled"
        )

    if call["name"] == "confirm_data_profile":
        details["spec"] = args.get("spec") or {}
        details["warnings"] = (args.get("spec") or {}).get("warnings")

    if call["name"] == "start_training":
        plan = args.get("plan") or {}
        details["command"] = plan.get("command")
        details["output_dir"] = plan.get("output_dir")
        details["estimated_wall_clock"] = plan.get("estimated_wall_clock")
        details["warnings"] = plan.get("warnings")
        if plan.get("overrides", {}).get("data.prepare"):
            details["downloads"] = "Dataset download required (MRL is ~431 MB)."

    if call["name"] == "land_generated_code":
        from adaptrna_agentic.agents.tool_factory import staged

        stage = staged(args.get("stage_id", ""))
        if stage is not None:
            details["files"] = [
                {"path": d, "lines": len(c.splitlines()), "content": c}
                for d, c in sorted(stage.files.items())
            ]
            details["staging_path"] = str(stage.package_dir)

    return details


def _path_covered(path: str, whitelist) -> bool:
    for pattern in whitelist:
        if pattern == path:
            return True
        if pattern.endswith(".*") and path.startswith(pattern[:-1]):
            return True
    return False


def _get_path(obj: Any, path: str) -> Any:
    parts = path.split(".")
    node = obj
    for i, part in enumerate(parts):
        if not isinstance(node, dict):
            return None
        # "overrides" (a training plan's config overrides) is a FLAT dict keyed by
        # dotted strings ("optim.lr"), not a nested structure — everything after it is
        # one key, not further path segments.
        if part == "overrides" and i + 1 < len(parts):
            return (node.get("overrides") or {}).get(".".join(parts[i + 1:]))
        if part not in node:
            return None
        node = node[part]
    return node


def _set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    node = obj
    for i, part in enumerate(parts[:-1]):
        if part == "overrides" and i + 1 < len(parts):
            if not isinstance(node.get("overrides"), dict):
                node["overrides"] = {}
            node["overrides"][".".join(parts[i + 1:])] = value
            return
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _same_shape(old: Any, new: Any) -> bool:
    """Numbers may cross int/float; anything else must keep its type."""
    if old is None:
        return True
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        return isinstance(new, (int, float)) and not isinstance(new, bool)
    return type(old) is type(new)


def _apply_edits(args: Dict[str, Any], edits: Optional[Dict[str, Any]], tool_name: str) -> Dict[str, Any]:
    """Apply human `field=value` edits from the approval gate onto a tool's arguments.

    Whitelist-only (`EDITABLE_ARGS`) and type-checked — an edit naming a path the tool
    does not declare editable, or changing a value's type, is refused rather than
    silently ignored (plan §5). Every edit actually applied is recorded on the mutated
    top-level object itself (`spec["human_edits"]` / `plan["human_overrides"]`) so no
    later surface can present an edited value as if the system had recommended it.
    """
    if not edits:
        return args

    whitelist = EDITABLE_ARGS.get(tool_name)
    if not whitelist:
        raise ValueError(
            f"'{tool_name}' has no editable fields; {sorted(edits)} cannot be applied."
        )

    import copy

    args = copy.deepcopy(args)
    changes: Dict[str, Dict[str, Any]] = {}

    for path, new_value in edits.items():
        if not _path_covered(path, whitelist):
            raise ValueError(
                f"'{path}' is not editable for '{tool_name}'. Editable fields: "
                f"{', '.join(whitelist)}."
            )

        old_value = _get_path(args, path)
        # A CLI/UI edit arrives untyped: 'positive_class=1' means the string class label
        # '1', not the number 1, whenever the field it targets is itself a string (class
        # labels, task names, columns are never numbers here even when the data's own
        # labels look like integers). Coerce back to the field's own type rather than
        # rejecting a perfectly sensible edit typed the natural way.
        if isinstance(old_value, str) and isinstance(new_value, (int, float)) and not isinstance(new_value, bool):
            new_value = str(new_value)

        if not _same_shape(old_value, new_value):
            raise ValueError(
                f"'{path}' cannot change type from {type(old_value).__name__} to "
                f"{type(new_value).__name__}."
            )

        _set_path(args, path, new_value)
        changes[path] = {"recommended": old_value, "chosen": new_value}

    if changes:
        top_key = next(iter(whitelist)).split(".", 1)[0]
        record_key = "human_overrides" if top_key == "plan" else "human_edits"
        target = args.get(top_key)
        if isinstance(target, dict):
            target[record_key] = changes

            if top_key == "plan":
                # The gate shows plan["command"] verbatim (§4/§8): a training plan whose
                # overrides changed but whose command did not would mean the human
                # approved one command and a different one actually ran.
                from adaptrna_agentic.profiling.recommender import build_command

                target["command"] = build_command(target)

    return args


def build_orchestrator_graph(
    model: Optional[BaseChatModel] = None,
    registry: Optional[Registry] = None,
    runtime: Optional[AdapterRuntime] = None,
    checkpointer=None,
):
    """
    Compile the orchestrator graph.

    All arguments are injectable (the test seams). Defaults: the configured orchestrator
    model (resolved lazily, so compiling needs no API key), the default-data-dir
    Registry, and its AdapterRuntime. Pass a LangGraph checkpointer for persistent
    sessions and for the approval gate, which needs to suspend and resume.
    """
    registry = registry or Registry()
    runtime = runtime or AdapterRuntime(registry)
    injected_model = model

    def call_model(state: OrchestratorState):
        nonlocal injected_model
        if injected_model is None:
            from adaptrna_agentic.models import build_chat_model

            injected_model = build_chat_model("orchestrator")

        # Rebuilt every call: descriptions reflect current tool states, and tools
        # registered or toggled since the last turn are picked up.
        bound = injected_model.bind_tools(build_agent_tools(registry, runtime))

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        return {"messages": [bound.invoke(messages)]}

    def request_approval(state: OrchestratorState):
        """The gate. Contains ONLY the interrupt, so resuming re-runs nothing else."""
        approvals = dict(state.get("approvals") or {})
        pending = [
            call for call in state["messages"][-1].tool_calls
            if call["name"] in GATED_TOOLS and call["id"] not in approvals
        ]

        decisions = interrupt({
            "type": "approval_request",
            "requests": [
                {"id": call["id"], "tool": call["name"], "args": call.get("args", {}),
                 "summary": _summarize(call, registry),
                 "details": _details(call, registry)}
                for call in pending
            ],
        })

        # Accept either one decision for everything, or a per-call-id mapping.
        normalized = {}
        for call in pending:
            decision = decisions.get(call["id"], decisions) if isinstance(decisions, dict) else decisions
            if isinstance(decision, bool):
                decision = {"approved": decision}
            elif isinstance(decision, str):
                decision = {"approved": decision.strip().lower() in ("y", "yes", "approve", "true")}
            normalized[call["id"]] = decision

        return {"approvals": {**approvals, **normalized}}

    def run_tools(state: OrchestratorState):
        tools = {tool.name: tool for tool in build_agent_tools(registry, runtime)}
        approvals = state.get("approvals") or {}
        last = state["messages"][-1]

        results = []
        for call in last.tool_calls:
            decision = approvals.get(call["id"])

            if call["name"] in GATED_TOOLS and not (decision or {}).get("approved"):
                note = (decision or {}).get("note") or "no reason given"
                output = (
                    f"The user declined to run '{call['name']}' ({note}). "
                    f"Do not retry it; ask what they would like instead."
                )
            elif call["name"] not in tools:
                output = f"Unknown tool '{call['name']}'. Use list_tools to see what exists."
            else:
                try:
                    # Edits are applied to a *copy* of the model's original args before
                    # the tool ever sees them — the gated tool itself never has to know
                    # this mechanism exists. handle_tool_error=True turns ToolExceptions
                    # (refusals, validation) into result strings; anything else is caught
                    # here so one bad call never kills the turn.
                    args = _apply_edits(call["args"], (decision or {}).get("edits"), call["name"])
                    output = tools[call["name"]].invoke(args)
                except Exception as exc:  # noqa: BLE001
                    output = f"Tool '{call['name']}' failed: {exc}"

            results.append(ToolMessage(
                content=stringify_tool_output(output),
                tool_call_id=call["id"],
                name=call["name"],
            ))

        # Decisions are per turn: clear them so a later call of the same gated tool asks
        # again rather than inheriting an earlier "yes".
        return {"messages": results, "approvals": {}}

    def route_after_model(state: OrchestratorState):
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return END

        approvals = state.get("approvals") or {}
        if any(c["name"] in GATED_TOOLS and c["id"] not in approvals for c in calls):
            return "approval"

        return "tools"

    graph = StateGraph(OrchestratorState)
    graph.add_node("model", call_model)
    graph.add_node("approval", request_approval)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", route_after_model,
        {"tools": "tools", "approval": "approval", END: END},
    )
    graph.add_edge("approval", "tools")
    graph.add_edge("tools", "model")

    return graph.compile(checkpointer=checkpointer)
