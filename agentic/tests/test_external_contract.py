"""The wrapper contract: loading, golden runner, install helpers. No installs, no PyPI."""

import sys
import types

import pytest

from adaptrna_agentic.toolhub.external import contract
from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec,
    FunctionSpec,
    GoldenCase,
    PackageSpec,
    install_command,
    is_available,
    load_spec,
    run_golden,
)
from adaptrna_agentic.toolhub.manifest import ToolEntry
from adaptrna_agentic.toolhub.registry import ToolHubError

DUMMY = "fixtures.dummy_external"


@pytest.fixture
def injected_module(monkeypatch):
    """Create an importable throwaway module and return it."""
    module = types.ModuleType("throwaway_external")
    monkeypatch.setitem(sys.modules, "throwaway_external", module)
    return module


def _entry(function="echo", golden=None, state="active"):
    return ToolEntry(
        name=f"dummy_{function}", type="external", state=state, description="d",
        test={"golden": golden if golden is not None
              else [{"args": {"value": "hi"}, "expect": {"value": "hi"}}]},
        external={"module": DUMMY, "function": function,
                  "package": {"pip": "dummy-package", "import_name": "json"}},
    )


def test_load_spec_on_compliant_module():
    spec, module = load_spec(DUMMY)

    assert spec.name == "dummy"
    assert [fn.name for fn in spec.functions] == ["echo", "add"]
    assert callable(module.echo)


def test_module_without_spec_rejected(injected_module):
    with pytest.raises(ToolHubError, match="does not define `SPEC"):
        load_spec("throwaway_external")


def test_spec_with_missing_function_rejected(injected_module):
    injected_module.SPEC = ExternalToolSpec(
        name="x", description="d", package=PackageSpec(pip="p", import_name="json"),
        functions=(FunctionSpec(name="nope", description="d"),),
    )

    with pytest.raises(ToolHubError, match="declares function 'nope'"):
        load_spec("throwaway_external")


def test_spec_with_non_callable_rejected(injected_module):
    injected_module.SPEC = ExternalToolSpec(
        name="x", description="d", package=PackageSpec(pip="p", import_name="json"),
        functions=(FunctionSpec(name="thing", description="d"),),
    )
    injected_module.thing = 42

    with pytest.raises(ToolHubError, match="not callable"):
        load_spec("throwaway_external")


def test_unimportable_module_rejected():
    with pytest.raises(ToolHubError, match="Cannot import wrapper module"):
        load_spec("definitely.not.a.module")


def test_availability_check():
    assert is_available(PackageSpec(pip="p", import_name="json")) is True
    assert is_available(PackageSpec(pip="p", import_name="not_a_module_xyz")) is False


def test_install_command_is_built_not_run():
    command = install_command(PackageSpec(pip="SomePkg", import_name="x"))

    assert command[-3:] == ["pip", "install", "SomePkg"]
    assert command[0] == sys.executable


def test_golden_pass():
    report = run_golden(_entry())

    assert report["ok"] is True
    assert report["outputs"] == [{"value": "hi"}]


def test_golden_exact_mismatch_fails():
    report = run_golden(_entry(golden=[{"args": {"value": "hi"},
                                        "expect": {"value": "bye"}}]))

    assert report["ok"] is False
    assert any("expected 'bye'" in check for check in report["checks"])


def test_golden_approx_within_and_outside_tolerance():
    good = _entry("add", golden=[{"args": {"a": 2, "b": 3},
                                  "expect": {"total": {"approx": 5.0, "tol": 0.01}}}])
    assert run_golden(good)["ok"] is True

    bad = _entry("add", golden=[{"args": {"a": 2, "b": 3},
                                 "expect": {"total": {"approx": 9.0, "tol": 0.01}}}])
    report = run_golden(bad)
    assert report["ok"] is False
    assert any("differs from 9.0" in check for check in report["checks"])


def test_golden_missing_key_fails():
    report = run_golden(_entry(golden=[{"args": {"value": "hi"},
                                        "expect": {"nope": 1}}]))

    assert report["ok"] is False
    assert any("missing key 'nope'" in check for check in report["checks"])


def test_golden_call_failure_is_reported_not_raised():
    report = run_golden(_entry("add", golden=[{"args": {"a": "x", "b": 3},
                                               "expect": {"total": 3}}]))

    assert report["ok"] is False
    assert any("call failed" in check for check in report["checks"])


def test_golden_on_disabled_tool_fails_with_hint():
    report = run_golden(_entry(state="disabled"))

    assert report["ok"] is False
    assert any("activate" in check for check in report["checks"])
