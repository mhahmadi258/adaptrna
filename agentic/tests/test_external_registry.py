"""register_external lifecycle + manifest backward compatibility."""

import json

import pytest

from adaptrna_agentic.toolhub.manifest import Manifest
from adaptrna_agentic.toolhub.registry import Registry, ToolHubError

DUMMY = "fixtures.dummy_external"


def test_register_external_creates_one_entry_per_function(nano_registry):
    entries = nano_registry.register_external(DUMMY)

    assert [e.name for e in entries] == ["dummy_echo", "dummy_add"]
    for entry in entries:
        assert entry.type == "external"
        assert entry.active
        assert entry.task is None and entry.artifact is None
        assert entry.external["module"] == DUMMY
        assert entry.external["package"]["import_name"] == "json"
        assert entry.provenance["source"] == DUMMY
        assert entry.provenance["family"] == "dummy"
        assert entry.test["golden"]                 # copied from the SPEC


def test_only_subset(nano_registry):
    entries = nano_registry.register_external(DUMMY, only=["add"])

    assert [e.name for e in entries] == ["dummy_add"]
    assert "dummy_echo" not in nano_registry.manifest.tools


def test_only_with_unknown_function_rejected(nano_registry):
    with pytest.raises(ToolHubError, match=r"Unknown function\(s\) \['nope'\]"):
        nano_registry.register_external(DUMMY, only=["nope"])


def test_duplicate_rejected_before_anything_is_written(nano_registry):
    nano_registry.register_external(DUMMY, only=["echo"])

    with pytest.raises(ToolHubError, match="'dummy_echo' is already registered"):
        nano_registry.register_external(DUMMY)

    # The batch was refused atomically: add was not sneaked in.
    assert "dummy_add" not in nano_registry.manifest.tools


def test_unavailable_package_names_the_command(nano_registry, monkeypatch):
    from adaptrna_agentic.toolhub.external import contract

    monkeypatch.setattr(contract, "is_available", lambda package: False)

    with pytest.raises(ToolHubError, match="pip install dummy-package"):
        nano_registry.register_external(DUMMY)


def test_lifecycle_and_remove_without_artifact(nano_registry):
    nano_registry.register_external(DUMMY)

    nano_registry.deactivate("dummy_echo")
    assert Registry(data_dir=nano_registry.data_dir).get("dummy_echo").state == "disabled"

    nano_registry.activate("dummy_echo")
    nano_registry.remove("dummy_echo")          # artifact is None -- must not crash

    assert "dummy_echo" not in Registry(data_dir=nano_registry.data_dir).manifest.tools


def test_adapters_and_externals_coexist(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external(DUMMY)

    names = [e.name for e in nano_registry.list()]
    assert names == ["demo_binary", "dummy_add", "dummy_echo"]


def test_phase2_manifest_without_external_keys_still_loads(tmp_path):
    """A v1 manifest written before the Phase 3 additive fields must load unchanged."""
    (tmp_path / "tools.json").write_text(json.dumps({
        "format_version": 1,
        "backbone": {"lm_config": "nano", "weights": None, "device": "cpu", "dtype": "auto"},
        "tools": {
            "demo_binary": {
                "type": "adapter", "state": "active", "description": "d",
                "task": "demo_binary", "lm_config": "nano",
                "artifact": "toolhub_data/adapters/demo_binary.pt",
                "serving": {"batch_size": None},
                "test": {"sequences": ["ACGU"], "expected": None},
                "provenance": {},
                # note: no "external" key
            }
        },
    }))

    manifest = Manifest.load(tmp_path)

    entry = manifest.tools["demo_binary"]
    assert entry.external is None
    assert entry.task == "demo_binary"
