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
        AIMessage(content="", tool_calls=[tool_call("demo_binary", {"sequences": [SEQ]})]),
        AIMessage(content="The probability is above."),
    ])

    events = stream(client, "/api/sessions/s2/messages", {"text": "is this a site?"})
    names = event_names(events)

    assert names.index("tool_call") < names.index("tool_result") < names.index("done")
    call = next(e for e in events if e["event"] == "tool_call")
    assert call["data"]["name"] == "demo_binary"

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

    listed = {s["id"] for s in client.get("/api/sessions").json()}
    assert listed >= {"alpha", "beta"}


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


# ---------------------------------------------------------------- session management
#
# Phase 10. Phase 8 deliberately shipped no delete surface and Phase 9 agreed; §1 of
# plans/PHASE_10_SESSION_RAIL_AND_TOOL_GATE.md narrows that to "sessions only" rather than
# dropping it, so these tests are as much a record of the reversal as a check on the code.


def test_create_makes_a_session_that_survives_a_reload(nano_registry, db):
    """A thread only appears in the listing once it has a checkpoint. Without the seeded
    empty state, a session created in the rail would vanish on refresh."""
    client, _ = _client(nano_registry, db, [AIMessage(content="ok")])

    response = client.post("/api/sessions", json={"id": "fresh"})

    assert response.status_code == 201
    assert response.json()["id"] == "fresh"
    assert "fresh" in [s["id"] for s in client.get("/api/sessions").json()]


def test_create_refuses_a_duplicate(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="ok")])
    client.post("/api/sessions", json={"id": "twice"})

    response = client.post("/api/sessions", json={"id": "twice"})

    assert response.status_code == 409
    assert "already exists" in response.json()["error"]


def test_create_refuses_a_blank_name(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="ok")])

    assert client.post("/api/sessions", json={"id": "   "}).status_code == 400


def test_rename_carries_the_history(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="Noted.")])
    stream(client, "/api/sessions/before/messages", {"text": "remember this"})

    response = client.patch("/api/sessions/before", json={"id": "after"})

    assert response.status_code == 200
    assert response.json()["id"] == "after"

    listed = [s["id"] for s in client.get("/api/sessions").json()]
    assert "after" in listed and "before" not in listed

    messages = client.get("/api/sessions/after/history").json()["messages"]
    assert any("remember this" in str(m["content"]) for m in messages)


def test_rename_refuses_an_occupied_name(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="ok")])
    client.post("/api/sessions", json={"id": "one"})
    client.post("/api/sessions", json={"id": "two"})

    response = client.patch("/api/sessions/one", json={"id": "two"})

    assert response.status_code == 409
    assert "one" in [s["id"] for s in client.get("/api/sessions").json()]


def test_delete_removes_the_thread_and_its_writes(nano_registry, db):
    import sqlite3

    client, _ = _client(nano_registry, db, [AIMessage(content="Noted.")])
    stream(client, "/api/sessions/doomed/messages", {"text": "hello"})

    response = client.delete("/api/sessions/doomed")

    assert response.status_code == 200
    assert response.json() == {"deleted": "doomed"}
    assert "doomed" not in [s["id"] for s in client.get("/api/sessions").json()]

    connection = sqlite3.connect(db)
    try:
        for table in ("checkpoints", "writes"):
            left = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", ("doomed",)
            ).fetchone()[0]
            assert left == 0, f"{table} still holds rows for the deleted session"
    finally:
        connection.close()


@pytest.mark.parametrize("call", [
    lambda c: c.delete("/api/sessions/ghost"),
    lambda c: c.patch("/api/sessions/ghost", json={"id": "other"}),
])
def test_managing_an_unknown_session_is_a_404(nano_registry, db, call):
    client, _ = _client(nano_registry, db, [AIMessage(content="ok")])

    assert call(client).status_code == 404


def test_the_listing_is_newest_first_and_dated(nano_registry, db):
    client, _ = _client(nano_registry, db, [AIMessage(content="ok"), AIMessage(content="ok")])
    stream(client, "/api/sessions/older/messages", {"text": "first"})
    stream(client, "/api/sessions/newer/messages", {"text": "second"})

    sessions = client.get("/api/sessions").json()
    ids = [s["id"] for s in sessions]

    assert ids.index("newer") < ids.index("older")
    # `updated_at` is decoded from the checkpoint UUIDv6, not stored anywhere.
    assert all(s["updated_at"].endswith("Z") for s in sessions)


def test_rename_is_refused_while_an_approval_is_pending(nano_registry, nano_splice_adapter, db):
    """Renaming moves the rows a suspended turn is addressed by; the interrupt would be
    stranded with no way for either front end to answer it."""
    nano_registry.register(nano_splice_adapter)
    nano_registry.deactivate("demo_binary")
    client, _ = _client(nano_registry, db, [
        AIMessage(content="", tool_calls=[tool_call("activate_tool", {"name": "demo_binary"})]),
        AIMessage(content="done"),
    ])

    events = stream(client, "/api/sessions/held/messages", {"text": "enable it"})
    assert "approval_required" in event_names(events)

    response = client.patch("/api/sessions/held", json={"id": "moved"})

    assert response.status_code == 409
    assert "approval" in response.json()["error"]
