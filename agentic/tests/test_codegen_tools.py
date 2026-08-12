"""Codegen as the agent sees it: discovery, the three tools, and the approval payload."""

import shutil
import uuid

import pytest

from adaptrna_agentic.agents.tool_factory import GATED_TOOLS, build_agent_tools, staged
from adaptrna_agentic.codegen import staging
from adaptrna_agentic.codegen.discovery import (
    CUSTOM_ROOT,
    custom_task_names,
    describe_failures,
    load_all,
    task_module_path,
)
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from fixtures import broken_task_sources as sources


@pytest.fixture
def tools(nano_registry):
    return {t.name: t for t in build_agent_tools(nano_registry, AdapterRuntime(nano_registry))}


@pytest.fixture
def landed_task(tmp_path):
    """Land a generated task in the real tree, then clean it up."""
    name = f"gen_{uuid.uuid4().hex[:6]}"
    data = tmp_path / "data"
    data.mkdir()
    rows = ["sequence,label"] + [f"{'ACGU' * 8},{i % 2}" for i in range(6)]
    (data / "train.csv").write_text("\n".join(rows) + "\n")
    (data / "val.csv").write_text("\n".join(rows[:4]) + "\n")

    stage = staging.stage_task(name, {
        "task.py": sources.good_task(name),
        "datamodule.py": sources.GOOD_DATAMODULE,
        "config.yaml": sources.CONFIG_TEMPLATE.format(task_name=name, data_root=data),
    }, data_dir=tmp_path / "hub")
    staging.land(stage)

    yield name

    shutil.rmtree(CUSTOM_ROOT / "tasks" / name, ignore_errors=True)


# ---------------------------------------------------------------- discovery

def test_landed_task_is_discovered_and_registers(landed_task):
    assert landed_task in custom_task_names()
    assert load_all(only=[landed_task]) == []

    import rinalmo_hub.tasks  # noqa: F401
    from rinalmo_hub.registry import available_tasks

    assert landed_task in available_tasks()
    assert task_module_path(landed_task).endswith(f"{landed_task}.task")


def test_a_broken_task_is_reported_without_breaking_the_others(tmp_path, landed_task):
    broken = f"gen_{uuid.uuid4().hex[:6]}"
    package = CUSTOM_ROOT / "tasks" / broken
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "task.py").write_text("import nonexistent_module_xyz\n")

    try:
        failures = load_all()
        names = [name for name, _ in failures]

        assert broken in names
        assert landed_task not in names            # the good one still loaded
        assert "ModuleNotFoundError" in describe_failures(failures)
    finally:
        shutil.rmtree(package, ignore_errors=True)


def test_runtime_imports_custom_tasks_before_serving(nano_registry, landed_task, tmp_path):
    """Without this the engine's get_task() raises and a generated task trains but
    cannot be served."""
    import torch

    import rinalmo_hub.tasks  # noqa: F401
    from rinalmo_hub.registry import get_task

    load_all(only=[landed_task])
    torch.manual_seed(0)
    module = get_task(landed_task)(
        lm_config="nano", head_config={"hidden_dim": 32},
        lora={"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 3},
    )
    module.apply_lora(verbose=False)
    adapter = tmp_path / f"{landed_task}_adapter.pt"
    module.save_adapter(adapter)

    nano_registry.register(adapter, name=landed_task)
    runtime = AdapterRuntime(nano_registry)

    outputs = runtime.predict(landed_task, ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC"])

    assert len(list(outputs)) == 1


# ---------------------------------------------------------------- tools

def test_codegen_tools_are_bound_and_gated(tools):
    for name in ("create_task_tool", "create_external_tool", "land_generated_code"):
        assert name in tools

    assert "land_generated_code" in GATED_TOOLS
    assert "create_task_tool" not in GATED_TOOLS      # generating is cheap and reversible


def test_landing_an_unknown_stage_is_refused(tools):
    result = tools["land_generated_code"].invoke({"stage_id": "nope"})

    assert isinstance(result, str)
    assert "No staged artifact" in result


def test_approval_payload_shows_the_files_and_the_staging_path(tmp_path):
    from adaptrna_agentic.agents.orchestrator import _details, _summarize
    from adaptrna_agentic.agents.tool_factory import _STAGES

    name = f"gen_{uuid.uuid4().hex[:6]}"
    stage = staging.stage_task(name, {
        "task.py": "x = 1\n", "datamodule.py": "y = 2\n", "config.yaml": "task: x\n",
    }, data_dir=tmp_path / "hub")
    _STAGES[stage.id] = stage

    try:
        call = {"name": "land_generated_code", "args": {"stage_id": stage.id}, "id": "c1"}

        summary = _summarize(call)
        details = _details(call)

        assert name in summary and "task.py" in summary
        assert {f["path"] for f in details["files"]} == set(stage.files)
        assert details["staging_path"] == str(stage.package_dir)
        assert "x = 1" in details["diff"]           # the human can read the code itself
    finally:
        _STAGES.pop(stage.id, None)


def test_landing_makes_a_staged_task_available(tmp_path, tools):
    from adaptrna_agentic.agents.tool_factory import _STAGES

    name = f"gen_{uuid.uuid4().hex[:6]}"
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("sequence,label\nACGUACGUACGU,1\n")
    (data / "val.csv").write_text("sequence,label\nACGUACGUACGU,0\n")

    stage = staging.stage_task(name, {
        "task.py": sources.good_task(name),
        "datamodule.py": sources.GOOD_DATAMODULE,
        "config.yaml": sources.CONFIG_TEMPLATE.format(task_name=name, data_root=data),
    }, data_dir=tmp_path / "hub")
    _STAGES[stage.id] = stage

    try:
        result = tools["land_generated_code"].invoke({"stage_id": stage.id})

        assert result["name"] == name
        assert any("task.py" in path for path in result["landed"])
        assert name in custom_task_names()
        assert stage.id not in _STAGES              # consumed
    finally:
        _STAGES.pop(stage.id, None)
        shutil.rmtree(CUSTOM_ROOT / "tasks" / name, ignore_errors=True)
