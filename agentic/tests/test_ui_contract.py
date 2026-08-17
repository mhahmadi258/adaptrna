"""The API surface the browser client reads, pinned.

`ui/*.js` is a separate codebase, in another language, with no compiler between it and
this one. These tests are that missing compiler: every assertion here names a field or an
event that the client reads *by name*, so a server-side rename fails in `pytest` — naming
the client file — instead of silently blanking a panel in someone's browser.

Where a test asserts on client source text it says so; those are pinning a specific
expression, and the browser suite (`-m ui`) checks the same properties against a real
rendered page.
"""

import inspect
import re
import sys
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from adaptrna_agentic.api.routers.ui import UI_DIR
from adaptrna_agentic.cli import chat as chat_cli
from adaptrna_agentic.jobs.runner import JobRunner
from adaptrna_agentic.profiling.recommender import PLAN_SOURCE
from api_helpers import build_test_app, stream
from scripted_model import tool_call

SESSION = "contract"

COMMAND = ["/bin/echo", "trained", "--task", "splice_site"]


def _plan(tmp_path):
    return {
        "source": PLAN_SOURCE, "task": "splice_site", "arm": "lora",
        "output_dir": str(tmp_path / "outputs" / "contract_run"),
        "command": COMMAND, "overrides": {},
        "estimated_wall_clock": "~7 min", "warnings": [],
    }


@pytest.fixture
def client(nano_registry, nano_splice_adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)
    nano_registry.register(nano_splice_adapter)

    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite", script=[
        AIMessage(content="Let me look.",
                  tool_calls=[tool_call("list_tools", {})]),
        AIMessage(content="Two tools are active."),
        AIMessage(content="", tool_calls=[tool_call("start_training", {"plan": _plan(tmp_path)})]),
        AIMessage(content="Started."),
    ])

    with TestClient(app) as test_client:
        yield test_client, tmp_path


def _events(test_client, text="what tools are there?"):
    return stream(test_client, f"/api/sessions/{SESSION}/messages", {"text": text})


def _by_name(events, name):
    return [e["data"] for e in events if e["event"] == name]


# ------------------------------------------------------------------ the stream

def test_every_streamed_event_is_one_the_client_handles():
    """The client's dispatch is a switch; anything the server emits must have a case.

    Read straight out of `app.js` so adding a server event without a client branch is a
    test failure rather than a frame the UI drops on the floor.
    """
    handled = set(re.findall(r'case "(\w+)":', (UI_DIR / "app.js").read_text()))

    assert {"text", "tool_call", "tool_result", "approval_required", "done", "error"} == handled


def test_a_turn_carries_the_fields_the_client_reads(client):
    test_client, _ = client

    events = _events(test_client)

    assert all("delta" in data for data in _by_name(events, "text"))

    call = _by_name(events, "tool_call")[0]
    assert call["name"] == "list_tools" and "args" in call

    result = _by_name(events, "tool_result")[0]
    assert "name" in result and "content" in result

    assert "answer" in _by_name(events, "done")[0]


def test_a_tool_result_is_not_also_streamed_as_text(client):
    """A ToolMessage must reach the client exactly once, as `tool_result`.

    Regression for a bug where the graph's `messages` stream mode re-emitted the same
    ToolMessage as a `text` delta, so the tool's raw output was rendered twice: once as a
    plain assistant bubble, once as the proper tool-result row. The duplicate only showed
    up live — a refresh (which rebuilds from the checkpointer) never had it — so this has
    to be caught here rather than in a `history()` test.
    """
    test_client, _ = client

    events = _events(test_client)

    results = _by_name(events, "tool_result")
    assert len(results) == 1

    streamed_text = "".join(data["delta"] for data in _by_name(events, "text"))
    assert results[0]["content"] not in streamed_text


def test_an_error_frame_carries_a_message(nano_registry, tmp_path):
    """The client renders `data.message`; a bare error would show as an empty bubble."""
    from adaptrna_agentic.api.events import stream_turn
    from api_helpers import parse_sse

    class Boom:
        def stream(self, *_args, **_kwargs):
            raise RuntimeError("the model exploded")

    frames = parse_sse("".join(stream_turn(Boom(), {}, {})))

    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["message"] == "the model exploded"


# ------------------------------------------------------------------ approval

def test_the_approval_request_carries_what_the_modal_renders(client):
    test_client, _ = client
    _events(test_client)                     # first scripted turn
    events = _events(test_client, "train it")

    request = events[-1]["data"]
    assert events[-1]["event"] == "approval_required"

    item = request["requests"][0]
    for field in ("id", "tool", "summary", "details"):
        assert field in item, f"the modal renders {field}"

    # The UI joins this on a single space; a string here would render character-spaced.
    assert isinstance(item["details"]["command"], list)


def test_the_browser_and_the_terminal_join_the_command_identically():
    """Gate 4 of the phase plan, pinned at the source level in both languages.

    What the modal shows must be byte-identical to what the terminal prints — a human
    approving a command they cannot see verbatim is the failure this gate exists to
    prevent. The browser suite checks the rendered result; this checks the expression.
    """
    terminal = inspect.getsource(chat_cli._prompt_approval)
    browser = (UI_DIR / "render.js").read_text()

    assert "' '.join(details['command'])" in terminal
    assert 'details.command.join(" ")' in browser


def test_the_modal_renders_every_detail_field_the_gate_can_carry():
    """A field the agent layer can emit but the modal ignores is invisible risk."""
    terminal = inspect.getsource(chat_cli._prompt_approval)
    browser = (UI_DIR / "render.js").read_text()

    fields = set(re.findall(r"details\.get\(\"(\w+)\"\)", terminal))
    # A check that silently stops finding anything is worse than no check: Phase 6 shipped
    # a harness that skipped its own data tests for exactly this reason.
    assert fields >= {"command", "output_dir", "files", "staging_path", "warnings"}, fields

    for field in fields:
        assert f"details.{field}" in browser, f"the modal never shows details.{field}"


def test_history_restores_a_pending_approval(client):
    """A browser refresh mid-approval must not strand the suspended turn."""
    test_client, _ = client
    _events(test_client)
    _events(test_client, "train it")

    body = test_client.get(f"/api/sessions/{SESSION}/history").json()

    assert body["pending_approval"]["requests"][0]["tool"] == "start_training"
    assert isinstance(body["messages"], list)


def test_history_messages_use_the_roles_the_client_switches_on(client):
    test_client, _ = client
    _events(test_client)

    messages = test_client.get(f"/api/sessions/{SESSION}/history").json()["messages"]
    roles = {message["role"] for message in messages}

    assert roles <= {"user", "assistant", "tool"}
    assert "user" in roles and "assistant" in roles

    for message in messages:
        assert "content" in message
        if message["role"] == "tool":
            assert "name" in message
        for call in message.get("tool_calls", []):
            assert "name" in call and "args" in call


# ------------------------------------------------------------------ panels

def test_the_tools_panel_fields(client):
    test_client, _ = client

    entry = test_client.get("/api/tools").json()[0]

    for field in ("name", "type", "state", "description"):
        assert field in entry
    assert entry["state"] in ("active", "disabled")


def test_the_health_badge_fields(client):
    test_client, _ = client

    body = test_client.get("/health").json()

    for field in ("status", "install", "failed_checks"):
        assert field in body

    # The badge counts this rather than testing it: an empty JS array is truthy, so a
    # healthy install once rendered as "install: ok ( failed)".
    assert isinstance(body["failed_checks"], list)


def test_the_session_rail_gets_rows_not_names(client):
    """Phase 10 changed this payload from bare strings to objects; `render.js::sessionRow`
    reads `id` and `updated_at` by name, and `app.js` sorts by nothing because the server
    already did."""
    test_client, _ = client
    _events(test_client)

    sessions = test_client.get("/api/sessions").json()

    assert SESSION in [s["id"] for s in sessions]
    for session in sessions:
        assert set(session) >= {"id", "updated_at", "checkpoints"}
        assert isinstance(session["id"], str)


def test_the_rail_can_create_rename_and_delete(client):
    """The three mutations the rail buttons make, in the order a user makes them."""
    test_client, _ = client

    created = test_client.post("/api/sessions", json={"id": "rail-one"})
    assert created.status_code == 201
    assert created.json()["id"] == "rail-one"

    renamed = test_client.patch("/api/sessions/rail-one", json={"id": "rail-two"})
    assert renamed.status_code == 200
    assert renamed.json()["id"] == "rail-two"

    listed = [s["id"] for s in test_client.get("/api/sessions").json()]
    assert "rail-two" in listed and "rail-one" not in listed

    deleted = test_client.delete("/api/sessions/rail-two")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": "rail-two"}
    assert "rail-two" not in [s["id"] for s in test_client.get("/api/sessions").json()]


def test_the_approval_modal_fields_for_a_tool_toggle(client):
    """`render.js::approvalBody` renders `details.tool` / `.current_state` /
    `.after_approval` for a gated activation, field for field with the terminal."""
    from adaptrna_agentic.agents.orchestrator import _details, _summarize

    call = {"name": "activate_tool", "args": {"name": "vienna_fold"}, "id": "c1"}

    assert _summarize(call) == "Enable the tool 'vienna_fold' (currently unknown)"
    assert set(_details(call)) == {"tool", "current_state", "after_approval"}


def test_the_jobs_panel_fields(client):
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    listed = test_client.get("/api/jobs").json()[0]
    for field in ("id", "task", "arm", "state"):
        assert field in listed

    status = test_client.get(f"/api/jobs/{job_id}").json()
    assert "progress" in status, "the monitor reads status.progress"
    for field in ("epoch", "step", "latest_metrics"):
        assert field in status["progress"]


def test_the_job_log_payload_fields(client):
    """Phase 11 put the log in the centre column, on a timer. It reads `body.log` by name."""
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    body = test_client.get(f"/api/jobs/{job_id}/logs?tail=200").json()

    for field in ("job_id", "tail", "log"):
        assert field in body
    assert body["tail"] == 200
    assert isinstance(body["log"], str)


def test_every_tail_the_client_offers_is_one_the_api_accepts(client):
    """The tail dropdown is a fixed list in `app.js`; the endpoint caps `tail` at 2000.

    A choice past the cap is a 422 the user sees as an empty log, so the list is read out
    of the client and every value in it is asked of the real server.
    """
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    source = (UI_DIR / "app.js").read_text()
    match = re.search(r"LOG_TAIL_CHOICES = \[([\d, ]+)\]", source)
    assert match, "app.js no longer declares LOG_TAIL_CHOICES as a literal list"

    choices = [int(value) for value in match.group(1).split(",")]
    assert choices, "the dropdown would render empty"

    for tail in choices:
        response = test_client.get(f"/api/jobs/{job_id}/logs?tail={tail}")
        assert response.status_code == 200, f"the client offers tail={tail}, the API refuses it"


def test_the_job_row_names_every_state_the_store_can_hold():
    """A state with no entry in the client's map draws a colourless dot.

    That is how `cancelled` shipped in Phase 9 and survived Phase 10: indistinguishable
    from a run that had not started.
    """
    from adaptrna_agentic.jobs.store import RUNNING_STATES, TERMINAL_STATES

    browser = (UI_DIR / "render.js").read_text()

    for job_state in (*RUNNING_STATES, *TERMINAL_STATES):
        assert f"{job_state}:" in browser, f"render.js never styles a '{job_state}' job"


def test_the_analysis_view_fields(client):
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    report = test_client.get(f"/api/jobs/{job_id}/analysis").json()

    for field in ("verdict", "checks", "suggestions", "truncated",
                  "primary_metric", "primary_value"):
        assert field in report


def _finished_job(tmp_path) -> str:
    script = tmp_path / "fake_train.py"
    script.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "m = out / 'metrics' / 'version_0'; m.mkdir(parents=True, exist_ok=True)\n"
        "(m / 'metrics.csv').write_text('epoch,step,train/loss,test/f1_score\\n"
        "0,50,0.4,\\n1,100,,96.5\\n')\n"
        "(out / 'exit_code').write_text('0')\n"
    )
    output_dir = tmp_path / "outputs" / "ui_job"
    runner = JobRunner()
    runner.start({
        "source": PLAN_SOURCE, "task": "splice_site", "arm": "lora",
        "output_dir": str(output_dir),
        "command": [sys.executable, str(script), str(output_dir)],
        "overrides": {}, "primary_metric": "test/f1_score",
    })

    for _ in range(200):
        if runner.status("ui_job")["state"] != "running":
            break
        time.sleep(0.1)

    return "ui_job"
