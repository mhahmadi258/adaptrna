"""The regression guard for the one genuinely unsafe piece of concurrency.

`RiNALMoHub.predict` calls `activate()`, which flips the active adapter on *every* tuner
layer of the shared backbone. Two overlapping predictions for different adapters could
therefore interleave that mutation and answer from the wrong one — a silent wrong answer,
which is the failure class this project treats most seriously.

The lock lives in `AdapterRuntime`, so these tests exercise the property rather than the
implementation: whatever the runtime does internally, concurrent callers must each get
their own tool's answer."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from api_helpers import build_test_app

SEQS = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]


@pytest.fixture
def client(nano_registry, nano_splice_adapter, nano_regression_adapter, tmp_path):
    nano_registry.register(nano_splice_adapter)
    nano_registry.register(nano_regression_adapter)
    app, services = build_test_app(nano_registry, tmp_path / "s.sqlite",
                                   script=[AIMessage(content="ok")])

    with TestClient(app) as test_client:
        yield test_client, services


def test_concurrent_predictions_for_different_adapters_stay_correct(client):
    test_client, _ = client

    def predict(name):
        response = test_client.post(f"/api/tools/{name}/predict", json={"sequences": SEQS})
        assert response.status_code == 200
        return name, response.json()

    # Establish each tool's answer serially first, then demand the same answers under
    # concurrency. A raced `set_adapter` shows up as one tool returning the other's.
    baseline = {
        "demo_binary": predict("demo_binary")[1]["predictions"],
        "demo_regression": predict("demo_regression")[1]["predictions"],
    }

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(predict, ["demo_binary", "demo_regression"] * 6))

    for name, body in results:
        assert body["tool"] == name
        assert body["predictions"] == pytest.approx(baseline[name]), (
            f"{name} returned another adapter's answer under concurrency"
        )


def test_predictions_are_serialised_by_the_runtime(nano_registry, nano_splice_adapter):
    """The lock is real: two threads never occupy the critical section at once."""
    nano_registry.register(nano_splice_adapter)
    runtime = AdapterRuntime(nano_registry)

    overlaps = []
    inside = threading.Lock()
    state = {"count": 0}

    original = runtime._ensure_tool

    def instrumented(name):
        with inside:
            state["count"] += 1
            overlaps.append(state["count"])
        time.sleep(0.01)
        result = original(name)
        with inside:
            state["count"] -= 1
        return result

    runtime._ensure_tool = instrumented

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: runtime.predict("demo_binary", SEQS[:1]), range(6)))

    assert max(overlaps) == 1, f"predictions overlapped: {overlaps}"


def test_a_slow_prediction_does_not_block_health(client, monkeypatch):
    """Handlers are `def`, so Starlette runs them off the event loop."""
    test_client, services = client

    real_predict = services.runtime.predict

    def slow(*args, **kwargs):
        time.sleep(0.5)
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(services.runtime, "predict", slow)

    with ThreadPoolExecutor(max_workers=2) as pool:
        prediction = pool.submit(
            test_client.post, "/api/tools/demo_binary/predict", json={"sequences": SEQS[:1]}
        )
        time.sleep(0.1)

        started = time.monotonic()
        health = test_client.get("/health")
        elapsed = time.monotonic() - started

        assert health.status_code == 200
        assert elapsed < 0.4, f"/health waited {elapsed:.2f}s behind the prediction"
        assert prediction.result().status_code == 200
