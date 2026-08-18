"""The seven Phase 5 tools, as the agent sees them."""

import json
import sys

import pytest

from adaptrna_agentic.agents.tool_factory import GATED_TOOLS, build_agent_tools
from adaptrna_agentic.profiling.recommender import PLAN_SOURCE
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

PIPELINE_TOOLS = (
    "profile_dataset", "recommend_training_config", "start_training",
    "job_status", "list_jobs", "analyze_run", "register_trained_adapter",
)


@pytest.fixture
def tools(nano_registry, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)

    runtime = AdapterRuntime(nano_registry)
    return {t.name: t for t in build_agent_tools(nano_registry, runtime)}, nano_registry


def test_all_pipeline_tools_are_bound(tools):
    built, _registry = tools

    for name in PIPELINE_TOOLS:
        assert name in built, f"{name} not bound"


def test_only_consequential_tools_are_gated():
    # GPU hours, a new servable tool, writing code into the repository, approving the
    # profiler's interpretation of a dataset — and, gated for authority rather than
    # cost, changing which tools the assistant may run at all.
    assert set(GATED_TOOLS) == {
        "confirm_data_profile", "start_training", "register_trained_adapter",
        "land_generated_code", "activate_tool", "deactivate_tool",
    }


def test_profile_confirm_and_recommend_round_trip(tools, tmp_path):
    # profile_dataset no longer matches a shipped task's on-disk layout (Phase 13) — it
    # proposes a DatasetSpec from one flat table, approved (here, unedited) through
    # confirm_data_profile. recommend_training_config takes the approved spec directly,
    # the same way it would for a task the human has not yet landed through
    # create_task_tool + land_generated_code.
    built, _registry = tools
    path = tmp_path / "data.csv"
    path.write_text(
        "sequence,label\n" + "\n".join(f"{'ACGT' * 100},{i % 2}" for i in range(40)) + "\n"
    )

    profile = built["profile_dataset"].invoke({"path": str(path)})
    assert profile["target_type"] == "binary"
    assert profile["sequence_column"] == "sequence"

    spec = built["confirm_data_profile"].invoke({"spec": profile})

    plan = built["recommend_training_config"].invoke({
        "task": spec["task_name"], "spec": spec,
    })
    assert plan["overrides"]["optim.lr"] == pytest.approx(3.0e-4)
    assert plan["overrides"]["data.root"] == spec["path"]
    assert plan["command"][1:3] == ["-m", "adaptrna_agentic.jobs.train_entrypoint"]


def test_recommend_with_no_task_returns_the_refusal_as_a_result(tools):
    built, _registry = tools

    result = built["recommend_training_config"].invoke({"task": ""})

    assert isinstance(result, str)                    # ToolException -> tool result
    assert "no task to train yet" in result


def test_recommend_for_an_unlanded_task_returns_the_refusal_as_a_result(tools):
    built, _registry = tools

    result = built["recommend_training_config"].invoke({"task": "never_landed"})

    assert isinstance(result, str)
    assert "No dataset spec found" in result


def _finished_job(tmp_path, built, adapter_source):
    """Run a fake job to completion and return its id."""
    import shutil
    import time

    from adaptrna_agentic.jobs.store import JobStore

    output_dir = tmp_path / "outputs" / "fake_run"
    script = tmp_path / "fake.py"
    script.write_text(
        "import pathlib,sys,shutil\n"
        "out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "m=out/'metrics'/'version_0'; m.mkdir(parents=True, exist_ok=True)\n"
        "(m/'metrics.csv').write_text('epoch,step,train/loss,test/f1_score\\n"
        "0,50,0.4,\\n1,100,,96.5\\n')\n"
        "shutil.copy(sys.argv[2], out/'demo_binary_adapter.pt')\n"
        "(out/'exit_code').write_text('0')\n"
    )
    plan = {
        "source": PLAN_SOURCE,
        "task": "demo_binary", "arm": "lora", "output_dir": str(output_dir),
        "command": [sys.executable, str(script), str(output_dir), str(adapter_source)],
        "overrides": {}, "estimated_wall_clock": "~7 min", "warnings": [],
        "primary_metric": "test/f1_score",
    }

    result = built["start_training"].invoke({"plan": plan})
    job_id = result["job_id"]

    for _ in range(200):
        if built["job_status"].invoke({"job_id": job_id})["state"] != "running":
            break
        time.sleep(0.1)

    return job_id


def test_start_status_analyze_register_flow(tools, tmp_path, nano_splice_adapter):
    built, registry = tools

    job_id = _finished_job(tmp_path, built, nano_splice_adapter)

    status = built["job_status"].invoke({"job_id": job_id})
    assert status["state"] == "succeeded"
    assert status["progress"]["latest_metrics"]["test/f1_score"] == pytest.approx(96.5)

    report = built["analyze_run"].invoke({"job_id": job_id})
    assert report["verdict"] == "ok"
    assert report["primary_value"] == pytest.approx(96.5)

    entry = built["register_trained_adapter"].invoke(
        {"job_id": job_id, "name": "demo_binary_acceptor",
         "description": "Acceptor splice sites"}
    )
    assert entry["name"] == "demo_binary_acceptor"
    assert entry["provenance"]["job_id"] == job_id
    assert entry["provenance"]["training_metrics"]["test/f1_score"] == pytest.approx(96.5)
    assert registry.get("demo_binary_acceptor").active


def test_list_jobs_reports_the_run(tools, tmp_path, nano_splice_adapter):
    built, _registry = tools
    job_id = _finished_job(tmp_path, built, nano_splice_adapter)

    jobs = built["list_jobs"].invoke({})

    assert [j["id"] for j in jobs] == [job_id]


def test_registering_an_unfinished_job_is_refused(tools, tmp_path):
    built, _registry = tools
    script = tmp_path / "slow.py"
    script.write_text("import time,pathlib,sys\n"
                      "pathlib.Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)\n"
                      "time.sleep(30)\n")
    output_dir = tmp_path / "outputs" / "slow_run"
    built["start_training"].invoke({"plan": {
        "source": PLAN_SOURCE,
        "task": "demo_binary", "arm": "lora", "output_dir": str(output_dir),
        "command": [sys.executable, str(script), str(output_dir)],
        "overrides": {}, "warnings": [],
    }})

    result = built["register_trained_adapter"].invoke({"job_id": "slow_run"})

    assert isinstance(result, str)
    assert "not succeeded" in result

    from adaptrna_agentic.jobs.runner import JobRunner
    JobRunner().cancel("slow_run")


def test_start_training_refuses_a_hand_assembled_plan(tools):
    """The 'never invent hyperparameters' rule, enforced mechanically rather than by
    prompt: a plan that did not come from the recommender is refused."""
    built, _registry = tools
    forged = {
        "task": "demo_binary", "arm": "lora", "output_dir": "outputs/forged",
        "command": ["echo", "hi"], "overrides": {"optim.lr": 1e-3},
    }

    result = built["start_training"].invoke({"plan": forged})

    assert isinstance(result, str)
    assert "recommend_training_config" in result
    assert "knowledge base" in result
