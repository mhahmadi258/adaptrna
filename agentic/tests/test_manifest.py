"""Manifest I/O — pure data layer, no engine, no torch model."""

import json

import pytest

from adaptrna_agentic.toolhub.manifest import (
    FORMAT_VERSION,
    BackboneConfig,
    Manifest,
    ToolEntry,
    resolve_data_dir,
)


def _entry(name="demo_binary"):
    return ToolEntry(
        name=name, type="adapter", state="active", description="d",
        task="demo_binary", lm_config="nano", artifact=f"toolhub_data/adapters/{name}.pt",
    )


def test_fresh_load_is_empty_with_defaults(tmp_path):
    manifest = Manifest.load(tmp_path)

    assert manifest.tools == {}
    assert manifest.backbone.lm_config == "giga"
    assert not manifest.path.exists()


def test_round_trip(tmp_path):
    manifest = Manifest.load(tmp_path)
    manifest.backbone = BackboneConfig(lm_config="nano", weights=None, device="cpu")
    manifest.tools["demo_binary"] = _entry()
    manifest.save()

    loaded = Manifest.load(tmp_path)

    assert loaded.backbone == manifest.backbone
    assert loaded.tools.keys() == {"demo_binary"}
    assert loaded.tools["demo_binary"] == manifest.tools["demo_binary"]


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    manifest = Manifest.load(tmp_path)
    manifest.tools["demo_binary"] = _entry()
    manifest.save()
    manifest.save()  # overwrite path too

    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert manifest.path.exists()


def test_unknown_format_version_rejected(tmp_path):
    (tmp_path / "tools.json").write_text(json.dumps({"format_version": 99, "tools": {}}))

    with pytest.raises(ValueError, match=f"format version 99.*{FORMAT_VERSION}"):
        Manifest.load(tmp_path)


def test_malformed_json_names_the_file(tmp_path):
    (tmp_path / "tools.json").write_text("{not json")

    with pytest.raises(ValueError, match="tools.json"):
        Manifest.load(tmp_path)


def test_non_manifest_json_rejected(tmp_path):
    (tmp_path / "tools.json").write_text(json.dumps({"something": "else"}))

    with pytest.raises(ValueError, match="not a ToolHub manifest"):
        Manifest.load(tmp_path)


def test_data_dir_resolution_order(tmp_path, monkeypatch):
    explicit = resolve_data_dir(tmp_path / "explicit")
    assert explicit == tmp_path / "explicit"

    monkeypatch.setenv("ADAPTRNA_TOOLHUB_DIR", str(tmp_path / "from_env"))
    assert resolve_data_dir() == tmp_path / "from_env"

    monkeypatch.delenv("ADAPTRNA_TOOLHUB_DIR")
    assert resolve_data_dir().name == "toolhub_data"


def test_auto_device_and_dtype_resolution():
    import torch

    assert BackboneConfig(device="auto").resolved_device() in ("cuda", "cpu")

    # auto keeps the model default (None) on every device -- non-autocast half-precision
    # trips an engine dtype promotion (see BackboneConfig.resolved_dtype).
    assert BackboneConfig(device="cpu", dtype="auto").resolved_dtype() is None
    assert BackboneConfig(device="cuda", dtype="auto").resolved_dtype() is None
    assert BackboneConfig(device="cpu", dtype="bfloat16").resolved_dtype() is torch.bfloat16
    assert BackboneConfig(device="cpu", dtype="float32").resolved_dtype() is torch.float32

    with pytest.raises(ValueError, match="Unknown dtype"):
        BackboneConfig(device="cpu", dtype="florp").resolved_dtype()
