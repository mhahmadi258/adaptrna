"""Gate 1: `confirm_data_profile` must not run before the human approves it.

Same load-bearing shape as the training gate (test_approval_gate.py): the interrupt fires
before anything is stamped, and a decline leaves nothing behind.
"""

import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from adaptrna_agentic.agents.orchestrator import build_orchestrator_graph
from adaptrna_agentic.agents.tool_factory import GATED_TOOLS
from adaptrna_agentic.profiling.profiler import PROFILE_SOURCE, SPEC_SOURCE, profile_dataset
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from scripted_model import scripted, tool_call

CONFIG = {"configurable": {"thread_id": "profile-gate"}}


@pytest.fixture
def stack(nano_registry):
    return nano_registry, AdapterRuntime(nano_registry)


@pytest.fixture
def saver(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "sessions.sqlite", check_same_thread=False))


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")
    return path


def _graph(stack, script, saver):
    registry, runtime = stack
    model = scripted(script)
    return model, build_orchestrator_graph(
        model=model, registry=registry, runtime=runtime, checkpointer=saver
    )


def _profile_script(spec):
    return [
        AIMessage(content="", tool_calls=[tool_call("confirm_data_profile", {"spec": spec})]),
        AIMessage(content="Approved."),
    ]


def test_confirm_data_profile_is_gated():
    assert "confirm_data_profile" in GATED_TOOLS


def test_confirm_data_profile_refuses_an_unstamped_spec(stack, saver, csv_path):
    forged = profile_dataset(csv_path)
    forged["source"] = None
    _model, graph = _graph(stack, _profile_script(forged), saver)

    graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": True}), CONFIG)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert "profile_dataset" in tool_messages[0].content


def test_gate_interrupts_before_anything_is_stamped(stack, saver, csv_path):
    spec = profile_dataset(csv_path)
    _model, graph = _graph(stack, _profile_script(spec), saver)

    result = graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)

    assert "__interrupt__" in result
    request = result["__interrupt__"][0].value["requests"][0]
    assert request["tool"] == "confirm_data_profile"
    # Not yet re-stamped: the request shows exactly what profile_dataset produced.
    assert request["args"]["spec"]["source"] == PROFILE_SOURCE
    assert request["details"]["spec"]["source"] == PROFILE_SOURCE
    assert "sequence" in request["summary"]


def test_declining_leaves_nothing_behind(stack, saver, csv_path):
    spec = profile_dataset(csv_path)
    _model, graph = _graph(stack, _profile_script(spec), saver)

    graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": False, "note": "wrong file"}), CONFIG)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert "declined" in tool_messages[0].content
    assert "wrong file" in tool_messages[0].content
    assert "Do not retry" in tool_messages[0].content


def test_approving_stamps_and_returns_the_approved_spec(stack, saver, csv_path):
    spec = profile_dataset(csv_path)
    _model, graph = _graph(stack, _profile_script(spec), saver)

    graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": True}), CONFIG)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    approved = json.loads(tool_messages[0].content)
    assert approved["source"] == SPEC_SOURCE


def test_a_later_profile_call_asks_again(stack, saver, csv_path):
    spec = profile_dataset(csv_path)
    _model, graph = _graph(stack, [
        AIMessage(content="", tool_calls=[tool_call("confirm_data_profile", {"spec": spec})]),
        AIMessage(content="First approved."),
        AIMessage(content="", tool_calls=[
            tool_call("confirm_data_profile", {"spec": spec}, "call_2")
        ]),
        AIMessage(content="Second approved."),
    ], saver)

    graph.invoke({"messages": [HumanMessage(content="profile it")]}, CONFIG)
    graph.invoke(Command(resume={"approved": True}), CONFIG)

    second = graph.invoke({"messages": [HumanMessage(content="profile again")]}, CONFIG)

    assert "__interrupt__" in second
