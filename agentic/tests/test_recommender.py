"""ConfigRecommender: table-driven, no free-floating constants, derived (not invented)
values for whatever cannot transfer across datasets, and — the strongest cheap
guarantee — the command it materialises parses under the engine's own CLI.

Phase 13: there are no known tasks any more, so `recommend()` takes a task name plus the
DatasetSpec it derives batch size and epoch count from (either passed directly, or read
from the task's own landed `spec.json`) — never a profile matched against a shipped
task's layout.
"""

import json

import pytest

from adaptrna_agentic.knowledge import arm as arm_knowledge
from adaptrna_agentic.profiling.recommender import PLAN_SOURCE, build_command, recommend
from adaptrna_agentic.toolhub.errors import ToolHubError


@pytest.fixture
def registry(tmp_path):
    """A hub whose backbone points at an existing (fake) checkpoint."""
    from adaptrna_agentic.toolhub.registry import Registry

    weights = tmp_path / "giga-v1.pt"
    weights.write_bytes(b"fake-checkpoint")

    hub = Registry(data_dir=tmp_path / "toolhub_data")
    hub.configure_backbone(lm_config="giga", weights=str(weights), device="cpu")
    return hub


@pytest.fixture(autouse=True)
def _default_registry(registry, monkeypatch):
    """Every recommend() call in this module uses the fixture hub unless told otherwise."""
    import adaptrna_agentic.profiling.recommender as module

    monkeypatch.setattr(module, "_backbone_config",
                        lambda reg=None: (reg or registry).manifest.backbone)


def _spec(**overrides):
    spec = {
        "path": "/data/donor_sites.csv",
        "target_type": "binary",
        "length": {"min": 380, "median": 400, "max": 420},
        "split": {"row_counts": {"train": 19350, "val": 2419, "test": 2419}},
        "head": {"primary_metric": "test/f1_score"},
    }
    spec.update(overrides)
    return spec


def _overrides(plan):
    return plan["overrides"]


# ---------------------------------------------------------------------- arm settings

def test_lora_uses_validated_settings():
    plan = recommend("donor_sites", spec=_spec())
    overrides = _overrides(plan)

    assert plan["task"] == "donor_sites"
    assert plan["arm"] == "lora"
    assert overrides["optim.lr"] == pytest.approx(3.0e-4)
    assert overrides["trainer.gradient_clip_val"] == 1.0
    assert overrides["lora.layer_stride"] == 3
    assert overrides["trainer.precision"] == "bf16-mixed"
    assert plan["primary_metric"] == "test/f1_score"


def test_every_recommended_arm_number_comes_from_the_knowledge_base():
    """No free-floating constants: the arm's settings must match the YAML exactly."""
    overrides = _overrides(recommend("donor_sites", spec=_spec()))
    lora = arm_knowledge("lora")

    assert overrides["optim.lr"] == lora["optim"]["lr"]
    assert overrides["optim.name"] == lora["optim"]["name"]
    assert overrides["trainer.gradient_clip_val"] == lora["trainer"]["gradient_clip_val"]
    for key, value in lora["lora"].items():
        assert overrides[f"lora.{key}"] == value


def test_rationale_carries_the_collapse_story():
    rationale = " ".join(recommend("donor_sites", spec=_spec())["rationale"])

    assert "1e-3" in rationale
    assert "constant-output" in rationale


def test_full_ft_warns_that_it_cannot_become_a_tool():
    plan = recommend("donor_sites", spec=_spec(), arm="full_ft")

    assert _overrides(plan)["optim.lr"] == pytest.approx(1.0e-5)
    assert "--use_lora" not in plan["command"]
    assert any("cannot become served tools" in w for w in plan["warnings"])


# ---------------------------------------------------------------------- data.root

def test_data_root_is_the_spec_file_not_a_directory():
    plan = recommend("donor_sites", spec=_spec(path="/data/donor_sites.csv"))

    assert _overrides(plan)["data.root"] == "/data/donor_sites.csv"


# ---------------------------------------------------------------------- derived values

def test_derived_batch_size_is_wired_into_overrides_and_rationale():
    plan = recommend("donor_sites", spec=_spec())

    assert _overrides(plan)["data.batch_size"] == 32   # median 400 nt
    assert any("data.batch_size 32" in line for line in plan["rationale"])


def test_derived_max_epochs_is_wired_into_overrides_and_rationale():
    plan = recommend("donor_sites", spec=_spec())

    epochs = _overrides(plan)["trainer.max_epochs"]
    assert any(f"trainer.max_epochs {epochs}" in line for line in plan["rationale"])


def test_derived_num_workers():
    plan = recommend("donor_sites", spec=_spec())

    assert _overrides(plan)["data.num_workers"] == 8


def test_no_reference_band_and_baseline_caveat():
    """There are no known tasks any more; a band is a property of a task and a dataset
    and cannot be invented."""
    plan = recommend("donor_sites", spec=_spec())

    assert plan["reference"]["band"] is None
    assert any("baseline" in w for w in plan["warnings"])


def test_estimated_wall_clock_is_honest_about_being_unknown():
    plan = recommend("donor_sites", spec=_spec())

    assert "unknown" in plan["estimated_wall_clock"]


# ---------------------------------------------------------------------- refusals

def test_no_early_stopping_is_ever_proposed():
    """Nothing may select on validation (MASTER_PLAN §7): checked on the *config*, not
    the prose."""
    plan = recommend("donor_sites", spec=_spec())

    keys = set(_overrides(plan))
    command_keys = {
        plan["command"][i + 1].partition("=")[0]
        for i, token in enumerate(plan["command"]) if token == "--set"
    }
    for forbidden in ("early_stopping", "monitor"):
        assert not any(forbidden in key for key in keys | command_keys)


def test_quick_run_truncates_and_warns():
    plan = recommend("donor_sites", spec=_spec(), quick=True)

    assert plan["quick_run"] is True
    assert _overrides(plan)["trainer.max_steps"] == 200
    assert any("NOT comparable" in w for w in plan["warnings"])
    assert "truncated" in plan["estimated_wall_clock"]


def test_missing_task_refuses_with_the_new_flow():
    with pytest.raises(ToolHubError, match="Approve a dataset spec"):
        recommend("", spec=_spec())


def test_missing_spec_and_no_landed_task_refuses(tmp_path, monkeypatch):
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path)

    with pytest.raises(ToolHubError, match="No dataset spec found"):
        recommend("nonexistent_task")


# ---------------------------------------------------------------------- landed spec.json

def test_spec_is_read_from_the_landed_task_when_not_passed(tmp_path, monkeypatch):
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path)
    task_dir = tmp_path / "tasks" / "donor_sites"
    task_dir.mkdir(parents=True)
    (task_dir / "spec.json").write_text(json.dumps(_spec()))

    plan = recommend("donor_sites")

    assert _overrides(plan)["data.batch_size"] == 32


def test_a_spec_passed_explicitly_overrides_data_root_for_reuse(tmp_path, monkeypatch):
    """Reuse (plan §9): recommending against an already-landed task, pointed at a new
    file of the same shape — the code is not regenerated, only data.root changes."""
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path)
    task_dir = tmp_path / "tasks" / "donor_sites"
    task_dir.mkdir(parents=True)
    (task_dir / "spec.json").write_text(json.dumps(_spec(path="/data/original.csv")))

    plan = recommend("donor_sites", spec=_spec(path="/data/new_file.csv"))

    assert _overrides(plan)["data.root"] == "/data/new_file.csv"


def test_an_unreadable_landed_spec_is_treated_as_absent_not_an_error(tmp_path, monkeypatch):
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path)
    task_dir = tmp_path / "tasks" / "donor_sites"
    task_dir.mkdir(parents=True)
    (task_dir / "spec.json").write_text("not valid json{{{")

    with pytest.raises(ToolHubError, match="No dataset spec found"):
        recommend("donor_sites")


# ---------------------------------------------------------------------- executability

def test_command_parses_under_the_engine_cli():
    """The plan must be executable: its command line has to satisfy the engine's own
    argument parser before anyone is asked to approve it."""
    from rinalmo_hub.cli.train import build_parser

    command = recommend("donor_sites", spec=_spec())["command"]

    assert command[1:3] == ["-m", "adaptrna_agentic.jobs.train_entrypoint"]
    args = build_parser().parse_args(command[3:])       # drop python -m <module>

    assert args.task == "donor_sites"
    assert args.use_lora is True


def test_command_renders_overrides_the_engine_can_parse():
    from rinalmo_hub.config import parse_scalar

    plan = recommend("donor_sites", spec=_spec(), quick=True)
    rendered = {}
    for i, token in enumerate(plan["command"]):
        if token == "--set":
            key, _, value = plan["command"][i + 1].partition("=")
            rendered[key] = parse_scalar(value)

    # Scientific notation survives the round trip (the engine's own parse_scalar quirk).
    assert rendered["optim.lr"] == pytest.approx(3.0e-4)
    assert rendered["trainer.max_steps"] == 200
    assert rendered["lora.layer_stride"] == 3


def test_build_command_is_pure():
    plan = recommend("donor_sites", spec=_spec())

    assert build_command(plan) == plan["command"]


# ---------------------------------------------------------------------- backbone

def test_plan_trains_against_the_hub_backbone(registry, tmp_path):
    """The engine defaults to `weights/giga-v1.pt` relative to the CWD; the plan must
    instead name the checkpoint the ToolHub actually serves."""
    plan = recommend("donor_sites", spec=_spec(), registry=registry)

    assert _overrides(plan)["pretrained_weights"] == str(tmp_path / "giga-v1.pt")
    assert _overrides(plan)["lm_config"] == "giga"
    assert any("could not be served alongside" in line for line in plan["rationale"])


def test_missing_checkpoint_refuses_with_the_fix(registry):
    registry.configure_backbone(weights="nowhere/giga-v1.pt")

    with pytest.raises(ToolHubError, match="toolhub config --weights"):
        recommend("donor_sites", spec=_spec(), registry=registry)


def test_hub_without_weights_warns_about_a_random_backbone(registry):
    registry.configure_backbone(weights="null")

    plan = recommend("donor_sites", spec=_spec(), registry=registry)

    assert _overrides(plan)["pretrained_weights"] == "null"
    assert any("randomly initialised backbone" in w for w in plan["warnings"])


# ---------------------------------------------------------------------- config path

def test_config_path_points_at_the_landed_tasks_own_package(tmp_path, monkeypatch):
    import adaptrna_agentic.profiling.recommender as module

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    package = tmp_path / "adaptrna_custom" / "tasks" / "donor_sites"
    package.mkdir(parents=True)
    (package / "config.yaml").write_text("task: donor_sites\n")

    plan = recommend("donor_sites", spec=_spec())

    assert plan["config_path"] == "adaptrna_custom/tasks/donor_sites/config.yaml"


def test_every_plan_is_stamped_with_its_source():
    assert recommend("donor_sites", spec=_spec())["source"] == PLAN_SOURCE
