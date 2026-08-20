"""`reset.py` returns the install to fresh, keeping `outputs/`.

The rules it must never break: a dry run by default, the manifest's `backbone` block
survives unless `--forget-backbone`, `outputs/` is never a candidate, and the four
git-tracked `adaptrna_custom/` skeleton files are never deleted.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reset as reset_module  # noqa: E402

from adaptrna_agentic.toolhub.manifest import BackboneConfig, Manifest, ToolEntry


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A throwaway install: toolhub/jobs/chat dirs plus a fake adaptrna_custom/ tree."""
    data_dir = tmp_path / "toolhub_data"
    jobs_dir = tmp_path / "jobs_data"
    chat_dir = tmp_path / "chat_data"
    data_dir.mkdir()
    jobs_dir.mkdir()
    chat_dir.mkdir()

    manifest = Manifest(data_dir=data_dir, backbone=BackboneConfig(weights="/abs/giga-v1.pt"))
    manifest.tools["demo_binary"] = ToolEntry(
        name="demo_binary", type="external", state="active", description="x",
    )
    manifest.save()

    (data_dir / "adapters").mkdir()
    (data_dir / "adapters" / "demo_binary.pt").write_bytes(b"weights")
    (data_dir / "staging" / "leftover-abcd1234").mkdir(parents=True)
    (data_dir / "staging" / "leftover-abcd1234" / "task.py").write_text("x = 1\n")

    (jobs_dir / "jobs.json").write_text('{"format_version": 1, "revision": 1, "jobs": {}}\n')

    (chat_dir / "sessions.sqlite").write_bytes(b"db")
    (chat_dir / "sessions.sqlite-wal").write_bytes(b"wal")

    custom_root = tmp_path / "adaptrna_custom"
    for keep in reset_module.CUSTOM_KEEP:
        path = custom_root / keep
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# tracked skeleton\n")
    (custom_root / "tasks" / "my_task").mkdir(parents=True)
    (custom_root / "tasks" / "my_task" / "task.py").write_text("x = 1\n")
    (custom_root / "tools" / "my_wrapper.py").write_text("y = 2\n")
    pycache = custom_root / "tasks" / "__pycache__"
    pycache.mkdir()
    (pycache / "__init__.cpython-313.pyc").write_bytes(b"\x00")

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "some_run").mkdir()
    (outputs / "some_run" / "run_summary.json").write_text("{}")

    monkeypatch.setattr(reset_module, "REPO_ROOT", tmp_path)

    return {
        "data_dir": data_dir, "jobs_dir": jobs_dir, "chat_dir": chat_dir,
        "custom_root": custom_root, "outputs": outputs,
    }


def _run(install, **kwargs):
    return reset_module.plan_reset(
        data_dir=install["data_dir"], jobs_dir=install["jobs_dir"],
        chat_dir=install["chat_dir"], **kwargs,
    )


# ---------------------------------------------------------------- dry run

def test_dry_run_deletes_nothing(install):
    report = _run(install)

    assert report["applied"] is False
    assert (install["data_dir"] / "tools.json").exists()
    assert (install["data_dir"] / "adapters" / "demo_binary.pt").exists()
    assert (install["jobs_dir"] / "jobs.json").exists()
    assert (install["chat_dir"] / "sessions.sqlite").exists()
    assert (install["custom_root"] / "tasks" / "my_task").exists()


def test_dry_run_lists_what_apply_would_remove(install):
    report = _run(install)
    labels = {c["label"] for group in report["groups"].values() for c in group}

    assert "demo_binary.pt" in labels
    assert "jobs.json" in labels
    assert "sessions.sqlite" in labels


# ---------------------------------------------------------------- apply

def test_apply_empties_manifest_but_keeps_backbone(install):
    _run(install, apply=True)

    man = Manifest.load(install["data_dir"])
    assert man.tools == {}
    assert man.backbone.weights == "/abs/giga-v1.pt"


def test_forget_backbone_deletes_the_manifest_file(install):
    _run(install, apply=True, forget_backbone=True)

    assert not (install["data_dir"] / "tools.json").exists()


def test_apply_removes_adapters_staging_jobs_chat(install):
    _run(install, apply=True)

    assert not (install["data_dir"] / "adapters" / "demo_binary.pt").exists()
    assert not (install["data_dir"] / "staging" / "leftover-abcd1234").exists()
    assert not (install["jobs_dir"] / "jobs.json").exists()
    assert not (install["chat_dir"] / "sessions.sqlite").exists()
    assert not (install["chat_dir"] / "sessions.sqlite-wal").exists()


def test_apply_removes_generated_code_but_keeps_the_tracked_skeleton(install):
    _run(install, apply=True)

    assert not (install["custom_root"] / "tasks" / "my_task").exists()
    assert not (install["custom_root"] / "tools" / "my_wrapper.py").exists()
    assert not (install["custom_root"] / "tasks" / "__pycache__").exists()

    for keep in reset_module.CUSTOM_KEEP:
        assert (install["custom_root"] / keep).exists(), f"{keep} must survive a reset"


def test_outputs_is_never_touched(install):
    _run(install, apply=True)

    assert (install["outputs"] / "some_run" / "run_summary.json").exists()
    assert "outputs" not in _run(install)["groups"]
