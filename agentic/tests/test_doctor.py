"""`doctor` must find every fault it claims to check for.

A health check that reports green on a broken install is worse than none, so each check
is exercised against a purpose-built *broken* install — the same discipline as the Phase 6
harness controls. And `doctor` must never change anything it inspects.
"""

import pytest

from adaptrna_agentic.toolhub import doctor
from adaptrna_agentic.toolhub.doctor import FAIL, OK, WARN, run_checks


def _check(report, name):
    return next((c for c in report["checks"] if c["name"] == name), None)


def _status(report, name):
    check = _check(report, name)
    return check["status"] if check else None


@pytest.fixture
def healthy(nano_registry, nano_splice_adapter, tmp_path, monkeypatch):
    """A hub with one adapter, one external family, a real checkpoint and no jobs."""
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    weights = tmp_path / "giga-v1.pt"
    weights.write_bytes(b"x" * 1024)
    nano_registry.configure_backbone(weights=str(weights))
    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external("fixtures.dummy_external")
    return nano_registry


def test_healthy_install_reports_ok(healthy):
    report = run_checks(healthy.data_dir)

    assert report["status"] == OK, report["failed"] + report["warned"]
    assert report["failed"] == [] and report["warned"] == []


def test_missing_artifact_is_a_failure(healthy):
    healthy.get("splice_site").artifact_path().unlink()

    report = run_checks(healthy.data_dir)

    assert report["status"] == FAIL
    check = _check(report, "artifact:splice_site")
    assert check["status"] == FAIL
    assert "toolhub remove splice_site" in check["remedy"]


def test_orphan_artifact_is_a_warning(healthy):
    (healthy.data_dir / "adapters" / "stale.pt").write_bytes(b"junk")

    report = run_checks(healthy.data_dir)

    assert _status(report, "orphan_artifacts") == WARN
    assert report["status"] == WARN


def test_missing_backbone_checkpoint_is_a_failure(healthy):
    healthy.configure_backbone(weights="nowhere/giga-v1.pt")

    report = run_checks(healthy.data_dir)

    assert _status(report, "backbone") == FAIL
    assert "toolhub config --weights" in _check(report, "backbone")["remedy"]


def test_unset_backbone_is_a_warning_not_a_failure(healthy):
    healthy.configure_backbone(weights="null")

    assert _status(run_checks(healthy.data_dir), "backbone") == WARN


def test_external_tool_with_a_missing_package_is_a_failure(healthy):
    entry = healthy.get("dummy_echo")
    entry.external["package"]["import_name"] = "definitely_not_installed_xyz"
    healthy.manifest.save()

    report = run_checks(healthy.data_dir)

    assert _status(report, "external_tools") == FAIL
    assert "dummy_echo" in _check(report, "external_tools")["detail"]


def test_stale_running_job_is_a_failure(healthy, tmp_path):
    from adaptrna_agentic.jobs.store import JobRecord, JobStore

    store = JobStore(tmp_path / "jobs_data")
    store.add(JobRecord(
        id="ghost", task="splice_site", arm="lora", command=["x"],
        output_dir=str(tmp_path / "outputs" / "ghost"),
        state="running", pid=999999, pid_starttime="1", started_at="2026-01-01T00:00:00+00:00",
    ))

    report = run_checks(healthy.data_dir, jobs_dir=tmp_path / "jobs_data")

    assert _status(report, "stale_jobs") == FAIL
    assert "ghost" in _check(report, "stale_jobs")["detail"]


def test_orphaned_stage_is_a_warning(healthy):
    from adaptrna_agentic.codegen import staging

    staging.stage_task(
        "leftover",
        {"task.py": "x = 1\n", "datamodule.py": "y = 2\n", "config.yaml": "task: leftover\n"},
        data_dir=healthy.data_dir,
    )

    report = run_checks(healthy.data_dir)

    assert _status(report, "staging") == WARN
    assert "prune staging" in _check(report, "staging")["remedy"]


def test_broken_custom_task_is_a_failure(healthy, monkeypatch):
    def boom(only=None):
        return [("broken_task", ImportError("no module named 'nowhere'"))]

    monkeypatch.setattr("adaptrna_agentic.codegen.discovery.load_all", boom)

    report = run_checks(healthy.data_dir)

    assert _status(report, "custom_tasks") == FAIL
    assert "broken_task" in _check(report, "custom_tasks")["detail"]


def test_template_version_check_ok_with_no_landed_tasks(healthy):
    report = run_checks(healthy.data_dir)

    assert _status(report, "template_version") == OK


def test_stale_template_version_is_a_warning(healthy, monkeypatch):
    from adaptrna_agentic.codegen.templates.render import TEMPLATE_VERSION

    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.custom_task_names", lambda: ["old_task"]
    )
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.landed_spec",
        lambda task: {"template_version": TEMPLATE_VERSION - 1},
    )

    report = run_checks(healthy.data_dir)

    assert _status(report, "template_version") == WARN
    check = _check(report, "template_version")
    assert "old_task" in check["detail"]
    assert check["data"]["stale"] == [
        {"task": "old_task", "template_version": TEMPLATE_VERSION - 1}
    ]


def test_current_template_version_is_not_flagged(healthy, monkeypatch):
    from adaptrna_agentic.codegen.templates.render import TEMPLATE_VERSION

    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.custom_task_names", lambda: ["fresh_task"]
    )
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.landed_spec",
        lambda task: {"template_version": TEMPLATE_VERSION},
    )

    report = run_checks(healthy.data_dir)

    assert _status(report, "template_version") == OK


def test_a_task_landed_by_the_llm_path_is_not_flagged(healthy, monkeypatch):
    """No template_version key at all -- generated, not rendered -- must not read as
    stale."""
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.custom_task_names", lambda: ["generated_task"]
    )
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.landed_spec",
        lambda task: {"task_name": "generated_task"},   # no template_version
    )

    report = run_checks(healthy.data_dir)

    assert _status(report, "template_version") == OK


def test_a_task_with_no_spec_json_is_not_flagged(healthy, monkeypatch):
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.custom_task_names", lambda: ["predates_this_build"]
    )
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.landed_spec", lambda task: None
    )

    report = run_checks(healthy.data_dir)

    assert _status(report, "template_version") == OK


def test_doctor_changes_nothing(healthy):
    """It is a diagnosis, not a repair."""
    before = healthy.manifest.path.read_text()
    artifacts_before = sorted(p.name for p in (healthy.data_dir / "adapters").iterdir())

    run_checks(healthy.data_dir)

    assert healthy.manifest.path.read_text() == before
    assert sorted(p.name for p in (healthy.data_dir / "adapters").iterdir()) == artifacts_before


def test_report_formats_with_remedies(healthy):
    healthy.get("splice_site").artifact_path().unlink()

    text = doctor.format_report(run_checks(healthy.data_dir))

    assert "FAIL" in text
    assert "->" in text            # the remedy line
