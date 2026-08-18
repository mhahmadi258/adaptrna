"""AdapterRuntime — lazy backbone, routing-level deactivation, smoke tests.

Runs a real forward pass on a randomly-initialised nano backbone (CPU, no weights)."""

import pytest

from adaptrna_agentic.toolhub.registry import ToolHubError
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

SEQUENCES = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]


@pytest.fixture
def runtime(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    return AdapterRuntime(nano_registry)


def test_registry_ops_never_load_the_backbone(runtime):
    runtime.registry.list()
    runtime.registry.get("demo_binary")
    runtime.registry.deactivate("demo_binary")
    runtime.registry.activate("demo_binary")

    assert runtime.loaded is False


def test_predict_loads_lazily_and_returns_probabilities(runtime):
    assert runtime.loaded is False

    outputs = runtime.predict("demo_binary", SEQUENCES)

    assert runtime.loaded is True
    values = [float(v) for v in outputs]
    assert len(values) == len(SEQUENCES)
    assert all(0.0 <= v <= 1.0 for v in values)

    # The second call reuses the resident hub.
    hub = runtime._hub
    runtime.predict("demo_binary", SEQUENCES[0])
    assert runtime._hub is hub


def test_two_tools_share_one_backbone(runtime, nano_regression_adapter):
    runtime.registry.register(nano_regression_adapter)

    splice = runtime.predict("demo_binary", SEQUENCES)
    hub = runtime._hub
    demo_regression = runtime.predict("demo_regression", SEQUENCES)

    assert runtime._hub is hub                      # same backbone serves both
    assert len(list(demo_regression)) == len(SEQUENCES)
    assert all(float(v) >= 0.0 for v in demo_regression)
    assert len(list(splice)) == len(SEQUENCES)


def test_disabled_tool_refuses_predict_while_resident(runtime):
    runtime.predict("demo_binary", SEQUENCES[0])    # make it resident
    runtime.registry.deactivate("demo_binary")

    with pytest.raises(ToolHubError, match="disabled.*activate demo_binary"):
        runtime.predict("demo_binary", SEQUENCES[0])

    runtime.registry.activate("demo_binary")
    runtime.predict("demo_binary", SEQUENCES[0])    # routing restored, no rebuild needed


def test_tool_registered_after_warmup_is_served_on_demand(runtime, nano_regression_adapter):
    runtime.warmup()
    assert runtime.loaded

    runtime.registry.register(nano_regression_adapter)
    outputs = runtime.predict("demo_regression", SEQUENCES[0])

    assert len(list(outputs)) == 1


def test_rebuild_drops_residency(runtime):
    runtime.predict("demo_binary", SEQUENCES[0])
    assert runtime.loaded

    runtime.rebuild()
    assert runtime.loaded is False

    runtime.predict("demo_binary", SEQUENCES[0])    # rebuilds cleanly
    assert runtime.loaded


def test_missing_weights_error_is_actionable(runtime):
    runtime.registry.configure_backbone(weights="does/not/exist.pt")

    with pytest.raises(ToolHubError, match="toolhub config --weights"):
        runtime.predict("demo_binary", SEQUENCES[0])


def test_smoke_test_reports_ok(runtime):
    report = runtime.smoke_test("demo_binary")

    assert report["ok"] is True
    assert report["task"] == "demo_binary"
    assert any("output form: ok" in check for check in report["checks"])
    assert len(report["outputs"]) == 2              # the default test sequences


def test_smoke_test_fails_on_wrong_expected_values(runtime):
    entry = runtime.registry.get("demo_binary")
    entry.test["expected"] = [999.0, 999.0]
    runtime.registry.manifest.save()

    report = runtime.smoke_test("demo_binary")

    assert report["ok"] is False
    assert any("FAIL" in check for check in report["checks"])


def test_warmup_skips_a_broken_tool_instead_of_blocking_startup(runtime, nano_regression_adapter):
    """One tool with a missing artifact must not stop a chat from starting; the same
    error still fires the moment that tool is actually used."""
    runtime.registry.register(nano_regression_adapter)
    runtime.registry.get("demo_regression").artifact_path().unlink()

    problems = runtime.warmup()

    assert len(problems) == 1 and "demo_regression" in problems[0]
    assert runtime.loaded
    assert "demo_binary" in runtime._resident        # the healthy tool is resident
    assert runtime.predict("demo_binary", SEQUENCES[:1])

    with pytest.raises(ToolHubError, match="does not exist"):
        runtime.predict("demo_regression", SEQUENCES[:1])
