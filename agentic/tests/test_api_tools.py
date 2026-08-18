"""Tool endpoints. Every one wraps CLI behaviour — including its refusals, which must
arrive as status codes carrying the CLI's own message."""

import pytest
from fastapi.testclient import TestClient

from api_helpers import build_test_app

SEQ = "GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC"


@pytest.fixture
def client(nano_registry, nano_splice_adapter, tmp_path):
    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external("fixtures.dummy_external")
    app, _ = build_test_app(nano_registry, tmp_path / "sessions.sqlite")

    with TestClient(app) as test_client:
        yield test_client


def test_list_shows_both_tool_kinds(client):
    tools = client.get("/api/tools").json()

    by_name = {t["name"]: t for t in tools}
    assert by_name["demo_binary"]["type"] == "adapter"
    assert by_name["dummy_add"]["type"] == "external"
    assert all(t["state"] == "active" for t in tools)


def test_info_returns_the_manifest_entry(client):
    entry = client.get("/api/tools/demo_binary").json()

    assert entry["task"] == "demo_binary"
    assert entry["provenance"]["source"]


def test_unknown_tool_is_404_listing_the_known_ones(client):
    response = client.get("/api/tools/banana")

    assert response.status_code == 404
    assert "Known tools" in response.json()["error"]


def test_predict_runs_on_the_nano_backbone(client):
    body = client.post("/api/tools/demo_binary/predict", json={"sequences": [SEQ]}).json()

    assert body["tool"] == "demo_binary"
    assert len(body["predictions"]) == 1
    assert 0.0 <= body["predictions"][0] <= 1.0


def test_disabled_tool_is_409_with_the_activate_hint(client):
    client.post("/api/tools/demo_binary/deactivate")

    response = client.post("/api/tools/demo_binary/predict", json={"sequences": [SEQ]})

    assert response.status_code == 409
    assert "disabled" in response.json()["error"]
    assert "activate" in response.json()["error"]


def test_activate_round_trip(client):
    assert client.post("/api/tools/demo_binary/deactivate").json()["state"] == "disabled"
    assert client.post("/api/tools/demo_binary/activate").json()["state"] == "active"
    assert client.post("/api/tools/demo_binary/predict",
                       json={"sequences": [SEQ]}).status_code == 200


def test_external_call(client):
    body = client.post("/api/tools/dummy_add/call", json={"args": {"a": 2, "b": 3}}).json()

    assert body["result"]["total"] == 5.0


def test_call_on_an_adapter_points_at_predict(client):
    response = client.post("/api/tools/demo_binary/call", json={"args": {}})

    assert response.status_code == 409
    assert "/predict" in response.json()["error"]


def test_predict_on_an_external_points_at_call(client):
    response = client.post("/api/tools/dummy_add/predict", json={"sequences": [SEQ]})

    assert response.status_code == 409
    assert "/call" in response.json()["error"]


def test_disabled_external_call_is_refused(client):
    client.post("/api/tools/dummy_add/deactivate")

    response = client.post("/api/tools/dummy_add/call", json={"args": {"a": 1, "b": 1}})

    assert response.status_code == 409
    assert "activate" in response.json()["error"]


def test_test_endpoint_dispatches_by_kind(client):
    adapter = client.post("/api/tools/demo_binary/test").json()
    external = client.post("/api/tools/dummy_add/test").json()

    assert adapter["ok"] is True and adapter["task"] == "demo_binary"
    assert external["ok"] is True


def test_health_and_doctor(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["install"] in ("ok", "warn", "fail")

    report = client.get("/api/doctor").json()
    assert {c["name"] for c in report["checks"]} >= {"engine", "artifacts"}


def test_predict_validation_error_is_422(client):
    """A missing required field is FastAPI's own validation, not ours."""
    assert client.post("/api/tools/demo_binary/predict", json={}).status_code == 422
