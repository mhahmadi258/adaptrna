"""Session persistence: the SQLite checkpointer owns history — same thread accumulates,
threads are isolated, and a fresh graph instance on the same DB file resumes (the
process-restart proof without a process restart)."""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from adaptrna_agentic.agents.orchestrator import build_orchestrator_graph
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from scripted_model import scripted


def _saver(path):
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture
def stack(nano_registry):
    return nano_registry, AdapterRuntime(nano_registry)


def _graph(stack, script, saver):
    registry, runtime = stack
    model = scripted(script)
    graph = build_orchestrator_graph(
        model=model, registry=registry, runtime=runtime, checkpointer=saver
    )
    return model, graph


def test_same_thread_accumulates_history(stack, tmp_path):
    model, graph = _graph(
        stack,
        [AIMessage(content="answer one"), AIMessage(content="answer two")],
        _saver(tmp_path / "sessions.sqlite"),
    )

    graph.invoke({"messages": [HumanMessage(content="first question")]}, _config("s1"))
    graph.invoke({"messages": [HumanMessage(content="second question")]}, _config("s1"))

    second_turn_input = [str(m.content) for m in model.calls[1]]
    assert any("first question" in text for text in second_turn_input)
    assert any("answer one" in text for text in second_turn_input)


def test_threads_are_isolated(stack, tmp_path):
    model, graph = _graph(
        stack,
        [AIMessage(content="a1"), AIMessage(content="a2")],
        _saver(tmp_path / "sessions.sqlite"),
    )

    graph.invoke({"messages": [HumanMessage(content="secret of thread one")]}, _config("s1"))
    graph.invoke({"messages": [HumanMessage(content="hello from thread two")]}, _config("s2"))

    other_turn_input = [str(m.content) for m in model.calls[1]]
    assert not any("secret of thread one" in text for text in other_turn_input)


def test_resume_with_a_new_graph_instance_on_the_same_db(stack, tmp_path):
    db = tmp_path / "sessions.sqlite"

    model_one, graph_one = _graph(stack, [AIMessage(content="noted: 42")], _saver(db))
    graph_one.invoke(
        {"messages": [HumanMessage(content="remember the number 42")]}, _config("dod")
    )

    # A brand-new graph + checkpointer on the same file — as after a process restart.
    model_two, graph_two = _graph(stack, [AIMessage(content="it was 42")], _saver(db))
    graph_two.invoke(
        {"messages": [HumanMessage(content="what number did I mention?")]}, _config("dod")
    )

    resumed_input = [str(m.content) for m in model_two.calls[0]]
    assert any("remember the number 42" in text for text in resumed_input)
    assert any("noted: 42" in text for text in resumed_input)


def test_list_sessions_cli_on_empty_store(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ADAPTRNA_CHAT_DIR", str(tmp_path))

    from adaptrna_agentic.cli.chat import main

    assert main(["--list-sessions"]) == 0
    assert "No sessions yet." in capsys.readouterr().out
