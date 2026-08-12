"""This API can start GPU jobs and write code into the repository.

Its defaults have to make the dangerous configuration unreachable by accident: loopback
only, and a refusal — not a warning — when asked to bind anywhere else without a token.
"""

import pytest
from fastapi.testclient import TestClient

from adaptrna_agentic.api.deps import is_loopback
from adaptrna_agentic.cli.serve import check_binding, main
from api_helpers import build_test_app


@pytest.fixture
def app_and_services(nano_registry, tmp_path):
    return build_test_app(nano_registry, tmp_path / "sessions.sqlite")


def test_no_token_configured_means_open_on_loopback(app_and_services):
    app, _ = app_and_services

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/tools").status_code == 200


def test_token_is_required_when_configured(nano_registry, tmp_path):
    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite", token="secret")

    with TestClient(app) as client:
        assert client.get("/api/tools").status_code == 401
        assert "bearer token" in client.get("/api/tools").json()["error"]

        ok = client.get("/api/tools", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200


def test_wrong_token_is_rejected(nano_registry, tmp_path):
    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite", token="secret")

    with TestClient(app) as client:
        response = client.get("/api/tools", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401


def test_health_stays_reachable_without_a_token(nano_registry, tmp_path):
    """A liveness probe should not need a credential."""
    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite", token="secret")

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------- binding refusal

@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_needs_no_token(host):
    assert is_loopback(host)
    assert check_binding(host, token=None) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10"])
def test_non_loopback_without_a_token_is_refused(host):
    problem = check_binding(host, token=None)

    assert problem is not None
    assert "Refusing to bind" in problem
    assert "ADAPTRNA_API_TOKEN" in problem


def test_non_loopback_with_a_token_is_allowed():
    assert check_binding("0.0.0.0", token="secret") is None


def test_serve_exits_rather_than_binding_unsafely(monkeypatch, capsys):
    monkeypatch.delenv("ADAPTRNA_API_TOKEN", raising=False)

    assert main(["--host", "0.0.0.0"]) == 1
    assert "Refusing to bind" in capsys.readouterr().err
