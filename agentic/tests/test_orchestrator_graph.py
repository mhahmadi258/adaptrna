"""The orchestrator graph end to end with a scripted model: tool loops, the
activate-then-use-in-one-turn policy, and turn survival."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from adaptrna_agentic.agents.orchestrator import SYSTEM_PROMPT, build_orchestrator_graph
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from scripted_model import scripted, tool_call

DUMMY = "fixtures.dummy_external"
SEQS = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]


@pytest.fixture
def stack(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external(DUMMY)
    return nano_registry, AdapterRuntime(nano_registry)


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
    assert "splice_site" in tool_messages[0].content
    assert "dummy_add" in tool_messages[0].content

    assert state["messages"][-1].content == "You have 3 tools."
    assert len(model.calls) == 2


def test_fm_tool_turn_runs_a_real_nano_prediction(stack):
    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("splice_site", {"sequences": SEQS})]),
        AIMessage(content="Here are the probabilities."),
    ])

    state = graph.invoke({"messages": [HumanMessage(content="Donor sites?")]})

    values = json.loads(_tool_messages(state)[0].content)
    assert len(values) == 2
    assert all(0.0 <= v <= 1.0 for v in values)


def test_activate_then_use_in_one_turn(stack):
    registry, runtime = stack
    registry.deactivate("dummy_add")

    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("activate_tool", {"name": "dummy_add"})]),
        AIMessage(content="", tool_calls=[tool_call("dummy_add", {"a": 2, "b": 3}, "call_2")]),
        AIMessage(content="2 + 3 = 5."),
    ])

    state = graph.invoke(
        {"messages": [HumanMessage(content="Enable dummy_add and add 2 and 3.")]}
    )

    tool_messages = _tool_messages(state)
    assert "now active" in tool_messages[0].content
    assert json.loads(tool_messages[1].content)["total"] == 5.0
    assert registry.get("dummy_add").active
    assert state["messages"][-1].content == "2 + 3 = 5."


def test_disabled_tool_refusal_reaches_the_model(stack):
    registry, runtime = stack
    registry.deactivate("dummy_echo")

    model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("dummy_echo", {"value": "hi"})]),
        AIMessage(content="That tool is disabled; shall I activate it?"),
    ])

    state = graph.invoke({"messages": [HumanMessage(content="Echo hi")]})

    assert "activate_tool" in _tool_messages(state)[0].content
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
