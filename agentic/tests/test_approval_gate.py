"""The approval gate: a consequential tool must not run before the human says yes.

The load-bearing assertion is the *absence* of a side effect at the interrupt — if the
gate were implemented inside the tools node, the tool would already have executed.
"""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from adaptrna_agentic.agents.orchestrator import build_orchestrator_graph
from adaptrna_agentic.profiling.recommender import PLAN_SOURCE
from adaptrna_agentic.agents.tool_factory import GATED_TOOLS
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from scripted_model import scripted, tool_call

CONFIG = {"configurable": {"thread_id": "gate"}}


@pytest.fixture
def stack(nano_registry, tmp_path, monkeypatch):
    """A registry plus an isolated jobs dir, so `start_training` has somewhere to write."""
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)
    return nano_registry, AdapterRuntime(nano_registry)


@pytest.fixture
def saver(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "sessions.sqlite", check_same_thread=False))


def fake_plan(tmp_path, name="gated_run"):
    """A plan whose command is a trivial script — the gate is what is under test."""
    import sys
    script = tmp_path / "noop.py"
    script.write_text("import pathlib,sys\n"
                      "out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
                      "(out/'exit_code').write_text('0')\n")
    output_dir = tmp_path / "outputs" / name

    return {
        "source": PLAN_SOURCE, "task": "splice_site", "arm": "lora", "output_dir": str(output_dir),
        "estimated_wall_clock": "~7 min",
        "command": [sys.executable, str(script), str(output_dir)],
        "overrides": {}, "warnings": [],
    }


def _graph(stack, script, saver):
    registry, runtime = stack
    model = scripted(script)
    return model, build_orchestrator_graph(
        model=model, registry=registry, runtime=runtime, checkpointer=saver
    )


def _training_script(plan):
    return [
        AIMessage(content="", tool_calls=[tool_call("start_training", {"plan": plan})]),
        AIMessage(content="Started."),
    ]


def test_start_training_is_gated():
    assert "start_training" in GATED_TOOLS
    assert "register_trained_adapter" in GATED_TOOLS


def test_tool_state_changes_are_gated():
    assert "activate_tool" in GATED_TOOLS
    assert "deactivate_tool" in GATED_TOOLS


@pytest.fixture
def stack_with_tool(stack, nano_splice_adapter):
    """The toggle gate needs something to toggle — the training gate's stack is empty."""
    registry, runtime = stack
    registry.register(nano_splice_adapter)
    return registry, runtime


def _toggle_script(tool, name):
    return [
        AIMessage(content="", tool_calls=[tool_call(tool, {"name": name})]),
        AIMessage(content="Done."),
    ]


def test_activation_interrupts_before_the_registry_is_touched(stack_with_tool, saver):
    """The same load-bearing assertion as the training gate: no side effect at the
    interrupt. Here the side effect is the manifest, not a subprocess."""
    registry, _runtime = stack_with_tool
    registry.deactivate("splice_site")
    _model, graph = _graph(stack_with_tool, _toggle_script("activate_tool", "splice_site"), saver)

    result = graph.invoke({"messages": [HumanMessage(content="enable it")]}, CONFIG)

    assert "__interrupt__" in result
    request = result["__interrupt__"][0].value["requests"][0]
    assert request["tool"] == "activate_tool"
    assert request["summary"] == "Enable the tool 'splice_site' (currently disabled)"
    assert request["details"]["current_state"] == "disabled"
    assert request["details"]["after_approval"] == "active"

    assert registry.get("splice_site").state == "disabled"


def test_declining_activation_leaves_the_tool_disabled(stack_with_tool, saver):
    """The Phase 10 bug report, pinned: the assistant enabled a tool the user had
    deliberately switched off."""
    registry, _runtime = stack_with_tool
    registry.deactivate("splice_site")
    _model, graph = _graph(stack_with_tool, _toggle_script("activate_tool", "splice_site"), saver)

    graph.invoke({"messages": [HumanMessage(content="enable it")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": False, "note": "leave it off"}), CONFIG)

    assert registry.get("splice_site").state == "disabled"

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert "declined" in tool_messages[0].content
    assert "Do not retry" in tool_messages[0].content


def test_approving_activation_flips_the_tool(stack_with_tool, saver):
    registry, _runtime = stack_with_tool
    registry.deactivate("splice_site")
    _model, graph = _graph(stack_with_tool, _toggle_script("activate_tool", "splice_site"), saver)

    graph.invoke({"messages": [HumanMessage(content="enable it")]}, CONFIG)
    graph.invoke(Command(resume={"approved": True}), CONFIG)

    assert registry.get("splice_site").state == "active"


def test_deactivation_is_gated_too(stack_with_tool, saver):
    """Symmetry matters: an agent that can silently disable a tool can break a workflow
    just as effectively as one that can silently enable it."""
    registry, _runtime = stack_with_tool
    _model, graph = _graph(stack_with_tool, _toggle_script("deactivate_tool", "splice_site"), saver)

    result = graph.invoke({"messages": [HumanMessage(content="turn it off")]}, CONFIG)

    assert "__interrupt__" in result
    request = result["__interrupt__"][0].value["requests"][0]
    assert request["summary"] == "Disable the tool 'splice_site' (currently active)"
    assert registry.get("splice_site").state == "active"


def test_the_gate_survives_an_unknown_tool_name(stack_with_tool, saver):
    """`_summarize` reads the registry to name the current state; a hallucinated tool name
    must still produce a renderable request, not a crash inside the gate."""
    _model, graph = _graph(stack_with_tool, _toggle_script("activate_tool", "no_such_tool"), saver)

    result = graph.invoke({"messages": [HumanMessage(content="enable it")]}, CONFIG)

    request = result["__interrupt__"][0].value["requests"][0]
    assert request["details"]["current_state"] == "unknown"


def test_gate_interrupts_before_the_tool_runs(stack, saver, tmp_path):
    plan = fake_plan(tmp_path)
    _model, graph = _graph(stack, _training_script(plan), saver)

    result = graph.invoke({"messages": [HumanMessage(content="train it")]}, CONFIG)

    # The turn suspended at the gate...
    assert "__interrupt__" in result
    request = result["__interrupt__"][0].value
    assert request["type"] == "approval_request"
    assert request["requests"][0]["tool"] == "start_training"
    assert request["requests"][0]["details"]["command"] == plan["command"]
    assert "~7 min" in request["requests"][0]["summary"]

    # ...and — the point of the whole design — nothing was executed.
    from adaptrna_agentic.jobs.store import JobStore
    assert JobStore(tmp_path / "jobs_data").jobs == {}
    assert not (tmp_path / "outputs" / "gated_run").exists()


def test_resume_with_approval_executes_the_tool(stack, saver, tmp_path):
    plan = fake_plan(tmp_path)
    _model, graph = _graph(stack, _training_script(plan), saver)

    graph.invoke({"messages": [HumanMessage(content="train it")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": True}), CONFIG)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert "job_id" in tool_messages[0].content

    from adaptrna_agentic.jobs.store import JobStore
    assert list(JobStore(tmp_path / "jobs_data").jobs) == ["gated_run"]


def test_resume_with_denial_skips_the_tool(stack, saver, tmp_path):
    plan = fake_plan(tmp_path, name="denied_run")
    _model, graph = _graph(stack, _training_script(plan), saver)

    graph.invoke({"messages": [HumanMessage(content="train it")]}, CONFIG)
    final = graph.invoke(
        Command(resume={"approved": False, "note": "wrong dataset"}), CONFIG
    )

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert "declined" in tool_messages[0].content
    assert "wrong dataset" in tool_messages[0].content
    assert "Do not retry" in tool_messages[0].content

    from adaptrna_agentic.jobs.store import JobStore
    assert JobStore(tmp_path / "jobs_data").jobs == {}


def test_plain_string_resume_is_accepted(stack, saver, tmp_path):
    plan = fake_plan(tmp_path, name="string_yes")
    _model, graph = _graph(stack, _training_script(plan), saver)

    graph.invoke({"messages": [HumanMessage(content="train it")]}, CONFIG)
    final = graph.invoke(Command(resume="yes"), CONFIG)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert "job_id" in tool_messages[0].content


def test_ungated_tools_never_interrupt(stack, saver):
    _model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("list_tools", {})]),
        AIMessage(content="Here they are."),
    ], saver)

    result = graph.invoke({"messages": [HumanMessage(content="what tools?")]}, CONFIG)

    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "Here they are."


def test_approvals_do_not_leak_into_a_later_turn(stack, saver, tmp_path):
    """A second gated call must ask again rather than inherit the earlier 'yes'."""
    plan_one = fake_plan(tmp_path, name="first_run")
    plan_two = fake_plan(tmp_path, name="second_run")
    _model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("start_training", {"plan": plan_one})]),
        AIMessage(content="First started."),
        AIMessage(content="", tool_calls=[tool_call("start_training", {"plan": plan_two}, "call_2")]),
        AIMessage(content="Second started."),
    ], saver)

    graph.invoke({"messages": [HumanMessage(content="train one")]}, CONFIG)
    graph.invoke(Command(resume={"approved": True}), CONFIG)

    second = graph.invoke({"messages": [HumanMessage(content="train another")]}, CONFIG)

    assert "__interrupt__" in second       # asked again, as it must
