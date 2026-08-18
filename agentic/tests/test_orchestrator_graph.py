"""The orchestrator graph end to end with a scripted model: tool loops, tool-state changes
stopping at the approval gate (Phase 10's reversal of Phase 4), and turn survival."""

import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from adaptrna_agentic.agents.orchestrator import SYSTEM_PROMPT, build_orchestrator_graph
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from scripted_model import scripted, tool_call

DUMMY = "fixtures.dummy_external"
SEQS = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]

#: The gate suspends the graph, which needs somewhere to suspend *to*.
CONFIG = {"configurable": {"thread_id": "orchestrator"}}


@pytest.fixture
def stack(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external(DUMMY)
    return nano_registry, AdapterRuntime(nano_registry)


@pytest.fixture
def saver(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "sessions.sqlite", check_same_thread=False))


def _graph(stack, script, checkpointer=None):
    registry, runtime = stack
    model = scripted(script)
    graph = build_orchestrator_graph(
        model=model, registry=registry, runtime=runtime, checkpointer=checkpointer
    )
    return model, graph


def _tool_messages(state):
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_list_tools_turn(stack):
    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("list_tools", {})]),
        AIMessage(content="You have 3 tools."),
    ])

    state = graph.invoke({"messages": [HumanMessage(content="What tools are available?")]})

    tool_messages = _tool_messages(state)
    assert len(tool_messages) == 1
    assert "demo_binary" in tool_messages[0].content
    assert "dummy_add" in tool_messages[0].content

    assert state["messages"][-1].content == "You have 3 tools."
    assert len(model.calls) == 2


def test_fm_tool_turn_runs_a_real_nano_prediction(stack):
    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("demo_binary", {"sequences": SEQS})]),
        AIMessage(content="Here are the probabilities."),
    ])

    state = graph.invoke({"messages": [HumanMessage(content="Donor sites?")]})

    values = json.loads(_tool_messages(state)[0].content)
    assert len(values) == 2
    assert all(0.0 <= v <= 1.0 for v in values)


def test_activate_then_use_in_one_turn_now_stops_at_the_gate(stack, saver):
    """Phase 4 let the model activate a tool and use it in the same turn; Phase 10 does not.

    The script is the *same* one Phase 4 proved — activate, then use — so this test is a
    direct record of the reversal: the turn now suspends before either call runs.
    """
    registry, runtime = stack
    registry.deactivate("dummy_add")

    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("activate_tool", {"name": "dummy_add"})]),
        AIMessage(content="", tool_calls=[tool_call("dummy_add", {"a": 2, "b": 3}, "call_2")]),
        AIMessage(content="2 + 3 = 5."),
    ], checkpointer=saver)

    state = graph.invoke(
        {"messages": [HumanMessage(content="Enable dummy_add and add 2 and 3.")]}, CONFIG
    )

    assert "__interrupt__" in state
    assert not _tool_messages(state), "nothing may run before the human answers"
    assert not registry.get("dummy_add").active


def test_declining_activation_leaves_the_tool_disabled(stack, saver):
    """The reported bug, pinned: a decline must not flip the switch."""
    registry, runtime = stack
    registry.deactivate("dummy_add")

    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("activate_tool", {"name": "dummy_add"})]),
        AIMessage(content="Left it off."),
    ], checkpointer=saver)

    graph.invoke({"messages": [HumanMessage(content="Turn dummy_add on.")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": False, "note": "no"}), CONFIG)

    assert not registry.get("dummy_add").active
    assert "declined" in _tool_messages(final)[0].content


def test_approving_activation_flips_the_tool(stack, saver):
    registry, runtime = stack
    registry.deactivate("dummy_add")

    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("activate_tool", {"name": "dummy_add"})]),
        AIMessage(content="dummy_add is on."),
    ], checkpointer=saver)

    graph.invoke({"messages": [HumanMessage(content="Turn dummy_add on.")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": True}), CONFIG)

    assert registry.get("dummy_add").active
    assert "now active" in _tool_messages(final)[0].content


def test_the_gate_names_the_tool_and_both_states(stack, saver):
    registry, runtime = stack
    registry.deactivate("dummy_add")

    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("activate_tool", {"name": "dummy_add"})]),
        AIMessage(content="done"),
    ], checkpointer=saver)

    state = graph.invoke({"messages": [HumanMessage(content="enable it")]}, CONFIG)

    request = state["__interrupt__"][0].value["requests"][0]
    assert request["summary"] == "Enable the tool 'dummy_add' (currently disabled)"
    assert request["details"] == {
        "tool": "dummy_add", "current_state": "disabled", "after_approval": "active",
    }


def test_disabled_tool_refusal_reaches_the_model(stack):
    registry, runtime = stack
    registry.deactivate("dummy_echo")

    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("dummy_echo", {"value": "hi"})]),
        AIMessage(content="That tool is disabled; shall I ask the user to enable it?"),
    ])

    state = graph.invoke({"messages": [HumanMessage(content="Echo hi")]})

    refusal = _tool_messages(state)[0].content
    assert "Only the user can enable it" in refusal
    assert "disabled" in state["messages"][-1].content


def test_unknown_tool_call_is_survivable(stack):
    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("nonexistent", {})]),
        AIMessage(content="Sorry, no such tool."),
    ])

    state = graph.invoke({"messages": [HumanMessage(content="do the thing")]})

    assert "Unknown tool" in _tool_messages(state)[0].content
    assert state["messages"][-1].content == "Sorry, no such tool."


def test_system_prompt_injected_once(stack):
    model, graph = _graph(stack, [AIMessage(content="hi")])

    graph.invoke({"messages": [HumanMessage(content="hello")]})

    first_call = model.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert first_call[0].content == SYSTEM_PROMPT
    assert sum(isinstance(m, SystemMessage) for m in first_call) == 1


def test_the_prompt_does_not_tell_the_model_to_switch_tools_on():
    """The gate stops the action; the prompt has to stop the *intent*, or every disabled
    tool becomes an approval modal the user did not ask for."""
    assert "the user's decision, not yours" in SYSTEM_PROMPT
    assert "never work around it" in SYSTEM_PROMPT
    # It used to read "offer to activate it with activate_tool before using it".
    assert "before using it" not in SYSTEM_PROMPT
    for gated in ("activate_tool", "deactivate_tool"):
        assert gated in SYSTEM_PROMPT.split("pause for the user's approval")[0]
