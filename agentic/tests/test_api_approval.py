"""The approval gate over HTTP.

The property that matters most: a gated action must not run before the human answers, and
must not run at all if they decline. Over HTTP that means the stream ends on
`approval_required` with nothing started, and the decision arrives in a separate request.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from adaptrna_agentic.jobs.store import JobStore
from adaptrna_agentic.profiling.recommender import PLAN_SOURCE
from api_helpers import build_test_app, event_names, stream
from scripted_model import tool_call

SESSION = "gate"


@pytest.fixture
def client(nano_registry, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)

    plan = {
        "source": PLAN_SOURCE, "task": "splice_site", "arm": "lora",
        "output_dir": str(tmp_path / "outputs" / "gated_run"),
        "command": ["/bin/echo", "trained"], "overrides": {},
        "estimated_wall_clock": "~7 min", "warnings": [],
    }
    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite", script=[
        AIMessage(content="", tool_calls=[tool_call("start_training", {"plan": plan})]),
        AIMessage(content="Done."),
    ])

    yield TestClient(app), tmp_path


def _jobs(tmp_path):
    return JobStore(tmp_path / "jobs_data").jobs


def test_the_stream_ends_on_approval_required_with_nothing_started(client):
    test_client, tmp_path = client

    events = stream(test_client, f"/api/sessions/{SESSION}/messages",
                    {"text": "train an acceptor model"})

    assert event_names(events)[-1] == "approval_required"
    assert "done" not in event_names(events)
    assert not _jobs(tmp_path), "a job ran BEFORE the human answered"


def test_the_request_carries_what_the_human_needs_to_decide(client):
    test_client, _ = client

    events = stream(test_client, f"/api/sessions/{SESSION}/messages", {"text": "train"})
    request = events[-1]["data"]["requests"][0]

    assert request["tool"] == "start_training"
    assert "splice_site" in request["summary"]
    assert request["details"]["command"] == ["/bin/echo", "trained"]   # the exact command


def test_history_reports_the_pending_approval(client):
    test_client, _ = client
    stream(test_client, f"/api/sessions/{SESSION}/messages", {"text": "train"})

    body = test_client.get(f"/api/sessions/{SESSION}/history").json()

    assert body["pending_approval"]["requests"][0]["tool"] == "start_training"


def test_declining_leaves_no_job(client):
    test_client, tmp_path = client
    stream(test_client, f"/api/sessions/{SESSION}/messages", {"text": "train"})

    events = stream(test_client, f"/api/sessions/{SESSION}/resume",
                    {"approved": False, "note": "not now"})

    assert event_names(events)[-1] == "done"
    blob = " ".join(str(e["data"]) for e in events if e["event"] == "tool_result")
    assert "declined" in blob and "not now" in blob
    assert not _jobs(tmp_path), "a job ran despite the refusal"


def test_approving_runs_it(client):
    test_client, tmp_path = client
    stream(test_client, f"/api/sessions/{SESSION}/messages", {"text": "train"})

    events = stream(test_client, f"/api/sessions/{SESSION}/resume", {"approved": True})

    assert event_names(events)[-1] == "done"
    assert "gated_run" in _jobs(tmp_path)


def test_a_new_message_is_refused_while_a_decision_is_pending(client):
    test_client, _ = client
    stream(test_client, f"/api/sessions/{SESSION}/messages", {"text": "train"})

    response = test_client.post(f"/api/sessions/{SESSION}/messages",
                                json={"text": "something else"})

    assert response.status_code == 409
    assert "resume" in response.json()["error"]


def test_resuming_with_nothing_pending_is_refused(client):
    test_client, _ = client

    response = test_client.post(f"/api/sessions/{SESSION}/resume", json={"approved": True})

    assert response.status_code == 409
    assert "nothing awaiting approval" in response.json()["error"]
