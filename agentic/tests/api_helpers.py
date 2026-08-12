"""Shared helpers for the HTTP tests: an app wired to a nano hub and a scripted model."""

from typing import List
import json

from fastapi.testclient import TestClient

from adaptrna_agentic.api.app import create_app
from adaptrna_agentic.api.deps import build_services
from scripted_model import scripted


def build_test_app(registry, db_path, script=None, model=None, token=None):
    """An app sharing one nano registry, one SQLite file, and a scripted model.

    Pass `model` when the test needs to inspect what the model was asked; `script`
    is the shorthand when it does not.
    """
    services = build_services(
        data_dir=registry.data_dir,
        db_path=db_path,
        model=model if model is not None else scripted(script or []),
        token=token,
    )
    # build_services makes its own Registry from the same directory; reuse the fixture's
    # so tests can mutate state and see it through the API.
    services.registry = registry
    services.runtime.registry = registry

    return create_app(services=services), services


def parse_sse(text: str) -> List[dict]:
    """SSE frames -> [{event, data}], in order."""
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event, data = None, {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event:
            events.append({"event": event, "data": data})

    return events


def event_names(events) -> List[str]:
    return [e["event"] for e in events]


def stream(client: TestClient, url: str, payload: dict) -> List[dict]:
    with client.stream("POST", url, json=payload) as response:
        assert response.status_code == 200, response.read()
        body = "".join(chunk for chunk in response.iter_text())

    return parse_sse(body)
