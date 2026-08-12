"""Streaming chat over HTTP, and the property the DoD turns on: a session written by the
terminal continues correctly through the API, against the same SQLite file."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from adaptrna_agentic.agents.orchestrator import build_orchestrator_graph
from adaptrna_agentic.api.deps import open_checkpointer
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from api_helpers import build_test_app, event_names, stream
from scripted_model import scripted, tool_call

SEQ = "GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "sessions.sqlite"


def _client(registry, db, script):
    app, services = build_test_app(registry, db, script=script)
    return TestClient(app), services


def test_a_plain_turn_streams_text_then_done(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="Hello there.")])

    events = stream(client, "/api/sessions/s1/messages", {"text": "hi"})

    assert "done" in event_names(events)
    assert events[-1]["data"]["answer"] == "Hello there."


def test_a_tool_turn_streams_call_then_result(nano_registry, nano_splice_adapter, db):
    nano_registry.register(nano_splice_adapter)
    client, _ = _client(nano_registry, db, [
        AIMessage(content="", tool_calls=[tool_call("splice_site", {"sequences": [SEQ]})]),
        AIMessage(content="The probability is above."),
    ])

    events = stream(client, "/api/sessions/s2/messages", {"text": "is this a site?"})
    names = event_names(events)

    assert names.index("tool_call") < names.index("tool_result") < names.index("done")
    call = next(e for e in events if e["event"] == "tool_call")
    assert call["data"]["name"] == "splice_site"

    result = next(e for e in events if e["event"] == "tool_result")
    values = json.loads(result["data"]["content"])
    assert 0.0 <= values[0] <= 1.0


def test_history_returns_the_turn(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="Noted.")])
    stream(client, "/api/sessions/s3/messages", {"text": "remember this"})

    body = client.get("/api/sessions/s3/history").json()
    roles = [m["role"] for m in body["messages"]]

    assert roles == ["user", "assistant"]
    assert body["messages"][0]["content"] == "remember this"
    assert body["pending_approval"] is None


def test_sessions_are_listed(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="ok")])
    stream(client, "/api/sessions/alpha/messages", {"text": "one"})
    stream(client, "/api/sessions/beta/messages", {"text": "two"})

    assert set(client.get("/api/sessions").json()) >= {"alpha", "beta"}


def test_threads_stay_isolated(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="ok"), AIMessage(content="ok")])
    stream(client, "/api/sessions/one/messages", {"text": "secret of thread one"})
    stream(client, "/api/sessions/two/messages", {"text": "hello"})

    other = client.get("/api/sessions/two/history").json()["messages"]
    assert not any("secret" in m.get("content", "") for m in other)


# ---------------------------------------------------------------- the DoD property

def test_a_terminal_session_continues_over_http(nano_registry, db):
    """Written by a graph built exactly as the terminal builds one, then continued by the
    API against the same SQLite file — the two front ends really do share sessions."""
    terminal_saver, _ = open_checkpointer(db)
    terminal_model = scripted([AIMessage(content="Noted: 42.")])
    terminal_graph = build_orchestrator_graph(
        model=terminal_model, registry=nano_registry,
        runtime=AdapterRuntime(nano_registry), checkpointer=terminal_saver,
    )
    config = {"configurable": {"thread_id": "shared"}}
    terminal_graph.invoke(
        {"messages": [HumanMessage(content="remember the number 42")]}, config
    )

    api_model = scripted([AIMessage(content="It was 42.")])
    app, _ = build_test_app(nano_registry, db, model=api_model)
    client = TestClient(app)

    events = stream(client, "/api/sessions/shared/messages",
                    {"text": "what number did I mention?"})

    assert events[-1]["data"]["answer"] == "It was 42."
    # The API's model saw the terminal's turn.
    seen = [str(m.content) for m in api_model.calls[0]]
    assert any("remember the number 42" in text for text in seen)
    assert any("Noted: 42." in text for text in seen)


def test_an_http_turn_is_visible_to_a_terminal_graph(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="Stored.")])
    stream(client, "/api/sessions/both/messages", {"text": "over http"})

    saver, _ = open_checkpointer(db)
    model = scripted([AIMessage(content="yes")])
    graph = build_orchestrator_graph(
        model=model, registry=nano_registry,
        runtime=AdapterRuntime(nano_registry), checkpointer=saver,
    )
    graph.invoke({"messages": [HumanMessage(content="and now from the terminal")]},
                 {"configurable": {"thread_id": "both"}})

    seen = [str(m.content) for m in model.calls[0]]
    assert any("over http" in text for text in seen)


def test_checkpointer_uses_wal(nano_registry, db):
    """Two processes share this file; rollback-journal mode would have them collide."""
    import sqlite3

    _client(nano_registry, db, [AIMessage(content="ok")])
    connection = sqlite3.connect(db)
    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    connection.close()

    assert mode.lower() == "wal"
