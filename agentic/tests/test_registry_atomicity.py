"""Registration touches two things — an artifact copy and the manifest — so a failure
between them must leave neither an orphan file nor an entry pointing at nothing."""

import pytest

from adaptrna_agentic.toolhub.errors import ConcurrentModificationError
from adaptrna_agentic.toolhub.registry import Registry


def test_failed_manifest_write_leaves_no_orphan_artifact(nano_registry, nano_splice_adapter):
    # Another writer moves the manifest on, so our save is refused mid-registration.
    other = Registry(data_dir=nano_registry.data_dir)
    other.manifest.save()

    with pytest.raises(ConcurrentModificationError):
        nano_registry.register(nano_splice_adapter)

    adapters = nano_registry.data_dir / "adapters"
    assert not list(adapters.glob("*.pt")) if adapters.exists() else True
    assert not list(adapters.glob("*.incoming")) if adapters.exists() else True
    assert Registry(data_dir=nano_registry.data_dir).list() == []


def test_successful_registration_leaves_no_temp_files(nano_registry, nano_splice_adapter):
    entry = nano_registry.register(nano_splice_adapter)

    adapters = nano_registry.data_dir / "adapters"
    assert entry.artifact_path().exists()
    assert list(adapters.glob("*.incoming")) == []


def test_verify_reports_a_missing_artifact(nano_registry, nano_splice_adapter):
    entry = nano_registry.register(nano_splice_adapter)
    entry.artifact_path().unlink()

    report = nano_registry.verify()

    assert [m["tool"] for m in report["missing_artifacts"]] == [entry.name]
    assert report["orphan_artifacts"] == []


def test_verify_reports_an_orphan_artifact(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    orphan = nano_registry.data_dir / "adapters" / "left_behind.pt"
    orphan.write_bytes(b"stale")

    report = nano_registry.verify()

    assert [o["artifact"] for o in report["orphan_artifacts"]] == [str(orphan)]
    assert report["missing_artifacts"] == []


def test_verify_is_clean_on_a_healthy_hub(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external("fixtures.dummy_external")

    report = nano_registry.verify()

    assert report == {"missing_artifacts": [], "orphan_artifacts": []}


def test_linked_artifacts_are_not_reported_as_orphans(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter, link=True)

    assert nano_registry.verify()["orphan_artifacts"] == []


def test_missing_artifact_fails_serving_with_an_actionable_message(
    nano_registry, nano_splice_adapter
):
    from adaptrna_agentic.toolhub.errors import ToolHubError
    from adaptrna_agentic.toolhub.runtime import AdapterRuntime

    entry = nano_registry.register(nano_splice_adapter)
    entry.artifact_path().unlink()

    with pytest.raises(ToolHubError, match="does not exist") as excinfo:
        AdapterRuntime(nano_registry).predict(entry.name, ["ACGUACGUACGUACGU"])

    message = str(excinfo.value)
    assert entry.name in message
    assert "toolhub remove" in message and "doctor" in message
