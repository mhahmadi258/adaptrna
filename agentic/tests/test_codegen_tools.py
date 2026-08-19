"""Codegen as the agent sees it: discovery, the three tools, and the approval payload."""

import shutil
import uuid
from pathlib import Path

import pytest

from adaptrna_agentic.agents.tool_factory import GATED_TOOLS, build_agent_tools, staged
from adaptrna_agentic.codegen import staging
from adaptrna_agentic.codegen.discovery import (
    CUSTOM_ROOT,
    TOOLS_DIRNAME,
    custom_task_names,
    describe_failures,
    load_all,
    task_module_path,
)
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from fixtures import broken_task_sources as sources

# Minimal contract-compliant wrapper whose "package" is the stdlib json module.
# Uses a unique family name per test run to avoid name collisions across tests.
_WRAPPER_TEMPLATE = """\
from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec, FunctionSpec, GoldenCase, PackageSpec,
)

def ping(message: str) -> dict:
    import json
    return {{"echo": json.dumps(message)}}

SPEC = ExternalToolSpec(
    name="{family}",
    description="Test-only wrapper.",
    package=PackageSpec(pip="dummy-pkg", import_name="json"),
    functions=(
        FunctionSpec(
            name="ping",
            description="Echo a message.",
            golden=(GoldenCase(args={{"message": "hi"}}, expect={{"echo": '"hi"'}}),),
        ),
    ),
)
"""


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


def test_create_task_tool_refuses_a_spec_not_from_confirm_data_profile(tools):
    result = tools["create_task_tool"].invoke({"spec": {"task_name": "x", "source": "made_up"}})

    assert isinstance(result, str)
    assert "confirm_data_profile" in result


def test_create_task_tool_renders_an_approved_spec_via_the_template(tools, tmp_path):
    """The end-to-end round trip through the real tool wrapper: profile -> approve ->
    build. A realistic spec is covered by the template, so this makes no model call."""
    from adaptrna_agentic.agents.tool_factory import staged
    from adaptrna_agentic.profiling.profiler import confirm_profile, profile_dataset

    path = tmp_path / "data.csv"
    rows = ["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")

    proposed = profile_dataset(path)
    spec = confirm_profile(proposed)

    result = tools["create_task_tool"].invoke({"spec": spec})

    assert result["ok"] is True, result
    assert result["path"] == "template"
    assert result["stage_id"]

    stage = staged(result["stage_id"])
    assert stage is not None
    assert {Path(f).name for f in stage.files} == {
        "task.py", "datamodule.py", "config.yaml", "spec.json",
    }


def test_file_mode_spec_renders_via_the_template_through_the_real_tools(tools, tmp_path):
    """The full cold-start round trip for a two-file dataset, through the real
    profile_dataset/confirm_data_profile/create_task_tool LangChain wrappers -- not just
    the underlying Python functions."""
    # Not named "train.csv" / "val.csv": _default_task_name derives the task name from
    # the filename, and "train" collides with nn.Module's own .train() attribute during
    # serving -- a pre-existing landmine unrelated to split.mode, sidestepped here rather
    # than fixed (out of scope for this change).
    train_path = tmp_path / "main_data.csv"
    val_path = tmp_path / "val_data.csv"
    train_path.write_text(
        "\n".join(["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]) + "\n"
    )
    val_path.write_text(
        "\n".join(["sequence,label"] + [f"{'ACGC' * 10},{i % 2}" for i in range(10)]) + "\n"
    )

    proposed = tools["profile_dataset"].invoke(
        {"path": str(train_path), "validation_path": str(val_path)}
    )
    assert proposed["split"]["mode"] == "file"

    spec = tools["confirm_data_profile"].invoke({"spec": proposed})

    result = tools["create_task_tool"].invoke({"spec": spec})

    assert result["ok"] is True, result
    assert result["path"] == "template"


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
        # Each file entry now carries the source so the browser can show it inline.
        file_by_path = {f["path"]: f for f in details["files"]}
        py_key = next(k for k in file_by_path if k.endswith("task.py"))
        assert file_by_path[py_key]["content"] == "x = 1\n"
        assert "lines" in file_by_path[py_key]
        assert "diff" not in details           # diff string removed; content is per-file
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


# ---------------------------------------------------------------- external-tool land + register

def test_landing_an_external_tool_registers_it_automatically(tmp_path, nano_registry):
    """land_generated_code for a kind='tool' stage must write the file AND call
    register_external, leaving active manifest entries — no separate CLI step required."""
    from adaptrna_agentic.agents.tool_factory import _STAGES

    family = f"gen_{uuid.uuid4().hex[:6]}"
    content = _WRAPPER_TEMPLATE.format(family=family)
    stage = staging.stage_tool(family, content, data_dir=tmp_path / "hub")
    _STAGES[stage.id] = stage

    tools_map = {t.name: t for t in build_agent_tools(nano_registry, AdapterRuntime(nano_registry))}

    try:
        result = tools_map["land_generated_code"].invoke({"stage_id": stage.id})

        # The tool result must name the registered entries.
        assert "registered" in result
        assert f"{family}_ping" in result["registered"]

        # The manifest must have the entry.
        entry = nano_registry.get(f"{family}_ping")
        assert entry.type == "external"
        assert entry.active
        assert entry.external["module"] == stage.module_path

        # The file must be on disk.
        landed = CUSTOM_ROOT / TOOLS_DIRNAME / f"{family}.py"
        assert landed.exists()

        assert stage.id not in _STAGES            # consumed
    finally:
        _STAGES.pop(stage.id, None)
        landed = CUSTOM_ROOT / TOOLS_DIRNAME / f"{family}.py"
        landed.unlink(missing_ok=True)
        # Remove the manifest entry if it was registered.
        if f"{family}_ping" in nano_registry.manifest.tools:
            nano_registry.remove(f"{family}_ping")


def test_landing_an_external_tool_reports_package_not_installed(tmp_path, nano_registry, monkeypatch):
    """If the wrapped package is absent, land_generated_code surfaces the install hint
    (not a silent success or a crash)."""
    import sys
    from adaptrna_agentic.agents.tool_factory import _STAGES
    from adaptrna_agentic.toolhub.external import contract as ext_contract

    family = f"gen_{uuid.uuid4().hex[:6]}"
    content = _WRAPPER_TEMPLATE.format(family=family)
    stage = staging.stage_tool(family, content, data_dir=tmp_path / "hub")
    _STAGES[stage.id] = stage

    # Pretend the package is not importable.
    monkeypatch.setattr(ext_contract, "is_available", lambda _pkg: False)

    tools_map = {t.name: t for t in build_agent_tools(nano_registry, AdapterRuntime(nano_registry))}

    try:
        result = tools_map["land_generated_code"].invoke({"stage_id": stage.id})

        # ToolException is returned as a string by handle_tool_error=True.
        assert isinstance(result, str)
        assert "pip install" in result.lower() or "not installed" in result.lower()
    finally:
        _STAGES.pop(stage.id, None)
        (CUSTOM_ROOT / TOOLS_DIRNAME / f"{family}.py").unlink(missing_ok=True)
