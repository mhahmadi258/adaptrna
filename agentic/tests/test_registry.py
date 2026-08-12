"""Registry lifecycle — engine used only to validate adapter files; no backbone loads."""

import pytest

from adaptrna_agentic.toolhub.registry import Registry, ToolHubError


def test_register_defaults(nano_registry, nano_splice_adapter):
    entry = nano_registry.register(nano_splice_adapter)

    assert entry.name == "splice_site"          # defaults to the adapter's task
    assert entry.state == "active"
    assert entry.task == "splice_site"
    assert entry.lm_config == "nano"
    assert entry.provenance["source"] == str(nano_splice_adapter)
    assert entry.provenance["adapter_metadata"]["arm"] == "lora"

    # The artifact was copied into registry-owned storage.
    copy = entry.artifact_path()
    assert copy.exists()
    assert copy != nano_splice_adapter
    assert copy.parent == nano_registry.data_dir / "adapters"


def test_register_with_name_override(nano_registry, nano_splice_adapter):
    entry = nano_registry.register(nano_splice_adapter, name="splice_site_donor")

    assert entry.name == "splice_site_donor"
    assert entry.artifact_path().name == "splice_site_donor.pt"


def test_duplicate_name_rejected(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)

    with pytest.raises(ToolHubError, match="already registered"):
        nano_registry.register(nano_splice_adapter)


def test_link_references_in_place(nano_registry, nano_splice_adapter):
    entry = nano_registry.register(nano_splice_adapter, link=True)

    assert entry.artifact_path() == nano_splice_adapter
    assert not (nano_registry.data_dir / "adapters").exists()


def test_lm_config_mismatch_rejected(tmp_path, nano_splice_adapter):
    registry = Registry(data_dir=tmp_path / "hub")  # default backbone: giga

    with pytest.raises(ToolHubError, match="not interchangeable"):
        registry.register(nano_splice_adapter)


def test_full_ft_export_rejected(nano_registry, full_ft_export):
    with pytest.raises(ToolHubError, match="full fine-tuning export"):
        nano_registry.register(full_ft_export)


def test_mrl_gets_pad_sensitive_serving_default(nano_registry, nano_mrl_adapter):
    entry = nano_registry.register(nano_mrl_adapter)

    assert entry.serving["batch_size"] == 1
    assert "pad-sensitive" in entry.description

    # An explicit batch size wins over the caveat default.
    entry2 = nano_registry.register(nano_mrl_adapter, name="mrl_batched", batch_size=4)
    assert entry2.serving["batch_size"] == 4


def test_state_changes_persist(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    nano_registry.deactivate("splice_site")

    reloaded = Registry(data_dir=nano_registry.data_dir)
    assert reloaded.get("splice_site").state == "disabled"

    reloaded.activate("splice_site")
    assert Registry(data_dir=nano_registry.data_dir).get("splice_site").active


def test_remove_deletes_owned_copy(nano_registry, nano_splice_adapter):
    entry = nano_registry.register(nano_splice_adapter)
    copy = entry.artifact_path()

    nano_registry.remove("splice_site")

    assert not copy.exists()
    assert nano_splice_adapter.exists()          # the source is never touched
    assert nano_registry.list() == []


def test_remove_keeps_linked_source(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter, link=True)

    nano_registry.remove("splice_site")

    assert nano_splice_adapter.exists()


def test_unknown_name_lists_known_tools(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)

    with pytest.raises(KeyError, match="splice_site"):
        nano_registry.get("banana")


def test_lm_config_change_blocked_while_tools_registered(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)

    with pytest.raises(ToolHubError, match="Remove them first"):
        nano_registry.configure_backbone(lm_config="giga")
