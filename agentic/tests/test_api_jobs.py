"""Job endpoints, and the error-mapping table.

Starting a job is deliberately absent from this surface: it happens through the chat,
behind the approval gate, so a human sees the exact command before GPU time is spent."""

import sys
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from adaptrna_agentic.jobs.runner import JobRunner
from adaptrna_agentic.jobs.store import JobRecord, JobStore
from adaptrna_agentic.profiling.recommender import PLAN_SOURCE
from api_helpers import build_test_app


@pytest.fixture
def client(nano_registry, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)

    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite",
                            script=[AIMessage(content="ok")])

    with TestClient(app) as test_client:
        yield test_client, tmp_path


def _finished_job(tmp_path) -> str:
    script = tmp_path / "fake_train.py"
    script.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "m = out / 'metrics' / 'version_0'; m.mkdir(parents=True, exist_ok=True)\n"
        "(m / 'metrics.csv').write_text('epoch,step,train/loss,test/f1_score\\n"
        "0,50,0.4,\\n1,100,,96.5\\n')\n"
        "print('training complete')\n"
        "(out / 'exit_code').write_text('0')\n"
    )
    output_dir = tmp_path / "outputs" / "api_job"
    runner = JobRunner()
    runner.start({
        "source": PLAN_SOURCE, "task": "splice_site", "arm": "lora",
        "output_dir": str(output_dir),
        "command": [sys.executable, str(script), str(output_dir)],
        "overrides": {}, "primary_metric": "test/f1_score",
    })

    for _ in range(200):
        if runner.status("api_job")["state"] != "running":
            break
        time.sleep(0.1)

    return "api_job"


def test_list_and_status(client):
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    jobs = test_client.get("/api/jobs").json()
    assert [j["id"] for j in jobs] == [job_id]

    status = test_client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "succeeded"
    assert status["progress"]["latest_metrics"]["test/f1_score"] == pytest.approx(96.5)


def test_logs(client):
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    body = test_client.get(f"/api/jobs/{job_id}/logs", params={"tail": 5}).json()

    assert "training complete" in body["log"]


def test_analysis(client):
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    report = test_client.get(f"/api/jobs/{job_id}/analysis").json()

    assert report["verdict"] == "ok"
    assert report["primary_value"] == pytest.approx(96.5)


def test_cancelling_a_finished_job_is_409(client):
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    response = test_client.post(f"/api/jobs/{job_id}/cancel")

    assert response.status_code == 409
    assert "not running" in response.json()["error"]


def test_recycled_pid_surfaces_as_409_not_500(client):
    """Phase 7's guard has to reach the client as a refusal, not an internal error."""
    test_client, tmp_path = client
    store = JobStore(tmp_path / "jobs_data")
    store.add(JobRecord(
        id="ghost", task="splice_site", arm="lora", command=["x"],
        output_dir=str(tmp_path / "outputs" / "ghost"),
        state="running", pid=999999, pid_starttime="1",
    ))

    response = test_client.post("/api/jobs/ghost/cancel")

    assert response.status_code == 409
    assert "may since have been reused" in response.json()["error"]


def test_unknown_job_is_404(client):
    test_client, _ = client

    response = test_client.get("/api/jobs/nope")

    assert response.status_code == 404
    assert "Known jobs" in response.json()["error"]


def test_logs_tail_is_validated(client):
    test_client, tmp_path = client
    job_id = _finished_job(tmp_path)

    assert test_client.get(f"/api/jobs/{job_id}/logs", params={"tail": 0}).status_code == 422


# ---------------------------------------------------------------- error mapping

def test_tool_refusal_is_409_with_the_cli_message(client, nano_registry, nano_splice_adapter):
    test_client, _ = client
    nano_registry.register(nano_splice_adapter)
    test_client.post("/api/tools/splice_site/deactivate")

    response = test_client.post("/api/tools/splice_site/predict", json={"sequences": ["ACGU"]})

    assert response.status_code == 409
    assert response.json()["type"] == "ToolHubError"


def test_concurrent_modification_is_409_and_marked_retryable(client, nano_registry):
    test_client, _ = client
    from adaptrna_agentic.toolhub.registry import Registry

    nano_registry.register_external("fixtures.dummy_external")
    # Another writer moves the manifest on behind the service's back.
    Registry(data_dir=nano_registry.data_dir).manifest.save()

    response = test_client.post("/api/tools/dummy_add/deactivate")

    assert response.status_code == 409
    assert response.json()["retryable"] is True
