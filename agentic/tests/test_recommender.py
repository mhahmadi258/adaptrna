"""ConfigRecommender: table-driven, no free-floating constants, and — the strongest
cheap guarantee — the command it materialises parses under the engine's own CLI."""

import pytest

from adaptrna_agentic.knowledge import arm as arm_knowledge
from adaptrna_agentic.knowledge import task_knowledge
from adaptrna_agentic.profiling.recommender import build_command, recommend
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


@pytest.fixture
def splice_profile(tmp_path):
    root = tmp_path / "train_data"
    (root / "GS_1" / "db_1").mkdir(parents=True)
    (tmp_path / "test_data").mkdir()
    return {"path": str(root), "layout_match": "splice_site",
            "target_type": "binary", "length_median": 400}


@pytest.fixture
def mrl_profile(tmp_path):
    root = tmp_path / "mrl_data"
    root.mkdir()
    return {"path": str(root), "layout_match": "mrl",
            "target_type": "continuous", "length_median": 50}


def _overrides(plan):
    return plan["overrides"]


def test_splice_site_lora_uses_validated_settings(splice_profile):
    plan = recommend(splice_profile)
    overrides = _overrides(plan)

    assert plan["task"] == "splice_site"
    assert plan["arm"] == "lora"
    assert overrides["optim.lr"] == pytest.approx(3.0e-4)
    assert overrides["trainer.gradient_clip_val"] == 1.0
    assert overrides["lora.layer_stride"] == 3
    assert overrides["trainer.precision"] == "bf16-mixed"
    assert plan["primary_metric"] == "test/f1_score"


def test_acceptor_arm_is_honored(splice_profile):
    plan = recommend(splice_profile, task_options={"ss_type": "acceptor"})

    assert _overrides(plan)["data.ss_type"] == "acceptor"
    assert "acceptor" in plan["output_dir"]


def test_invalid_task_option_rejected(splice_profile):
    with pytest.raises(ToolHubError, match="not a valid data.ss_type"):
        recommend(splice_profile, task_options={"ss_type": "banana"})


def test_splice_test_root_points_at_the_benchmark_species(splice_profile):
    overrides = _overrides(recommend(splice_profile))

    assert overrides["data.test_root"].endswith("test_data")


def test_every_recommended_number_comes_from_the_knowledge_base(splice_profile):
    """No free-floating constants: the arm's settings must match the YAML exactly."""
    overrides = _overrides(recommend(splice_profile))
    lora = arm_knowledge("lora")

    assert overrides["optim.lr"] == lora["optim"]["lr"]
    assert overrides["optim.name"] == lora["optim"]["name"]
    assert overrides["trainer.gradient_clip_val"] == lora["trainer"]["gradient_clip_val"]
    for key, value in lora["lora"].items():
        assert overrides[f"lora.{key}"] == value


def test_rationale_carries_the_collapse_story(splice_profile):
    rationale = " ".join(recommend(splice_profile)["rationale"])

    assert "1e-3" in rationale
    assert "constant-output" in rationale


def test_mrl_random7600_warning_emitted(mrl_profile):
    warnings = " ".join(recommend(mrl_profile)["warnings"])

    assert "random7600" in warnings
    assert "holdout" in warnings


def test_no_early_stopping_is_ever_proposed(splice_profile, mrl_profile):
    """Nothing may select on validation (MASTER_PLAN §7): checked on the *config*, not
    the prose — the MRL caveat mentions early stopping precisely to warn against it."""
    for profile in (splice_profile, mrl_profile):
        plan = recommend(profile)

        # Config KEYS only — values carry filesystem paths, and pytest's tmp dirs are
        # named after the test, so scanning values would match this test's own name.
        keys = set(_overrides(plan))
        command_keys = {
            plan["command"][i + 1].partition("=")[0]
            for i, token in enumerate(plan["command"]) if token == "--set"
        }

        for forbidden in ("early_stopping", "monitor"):
            assert not any(forbidden in key for key in keys | command_keys)

        assert _overrides(plan).get("trainer.checkpoint_every_epoch") in (None, False)


def test_quick_run_truncates_and_warns(splice_profile):
    plan = recommend(splice_profile, quick=True)

    assert plan["quick_run"] is True
    assert _overrides(plan)["trainer.max_steps"] == 200
    assert any("NOT comparable" in w for w in plan["warnings"])
    assert "truncated" in plan["estimated_wall_clock"]


def test_full_ft_warns_that_it_cannot_become_a_tool(splice_profile):
    plan = recommend(splice_profile, arm="full_ft")

    assert _overrides(plan)["optim.lr"] == pytest.approx(1.0e-5)
    assert "--use_lora" not in plan["command"]
    assert any("cannot become served tools" in w for w in plan["warnings"])


def test_eta_comes_from_the_knowledge_base(splice_profile):
    plan = recommend(splice_profile)

    assert plan["estimated_wall_clock"] == task_knowledge("splice_site")["wall_clock"]["reference"]


def test_unmatched_profile_refuses_with_the_reason():
    profile = {"path": "/tmp", "layout_match": None,
               "layout_reason": "No shipped task reads this layout."}

    with pytest.raises(ToolHubError, match="nothing to train yet"):
        recommend(profile)


# ---------------------------------------------------------------- executability

def test_command_parses_under_the_engine_cli(splice_profile):
    """The plan must be executable: its command line has to satisfy the engine's own
    argument parser before anyone is asked to approve it."""
    from rinalmo_hub.cli.train import build_parser

    command = recommend(splice_profile, task_options={"ss_type": "acceptor"})["command"]

    assert command[1:3] == ["-m", "adaptrna_agentic.jobs.train_entrypoint"]
    args = build_parser().parse_args(command[3:])       # drop python -m <module>

    assert args.task == "splice_site"
    assert args.use_lora is True
    assert any(override.startswith("data.ss_type=") for override in args.overrides)


def test_command_renders_overrides_the_engine_can_parse(splice_profile):
    from rinalmo_hub.config import parse_scalar

    plan = recommend(splice_profile, quick=True)
    rendered = {}
    for i, token in enumerate(plan["command"]):
        if token == "--set":
            key, _, value = plan["command"][i + 1].partition("=")
            rendered[key] = parse_scalar(value)

    # Scientific notation survives the round trip (the engine's own parse_scalar quirk).
    assert rendered["optim.lr"] == pytest.approx(3.0e-4)
    assert rendered["trainer.max_steps"] == 200
    assert rendered["lora.layer_stride"] == 3


def test_build_command_is_pure(splice_profile):
    plan = recommend(splice_profile)

    assert build_command(plan) == plan["command"]


# ---------------------------------------------------------------- backbone

def test_plan_trains_against_the_hub_backbone(splice_profile, registry, tmp_path):
    """The engine defaults to `weights/giga-v1.pt` relative to the CWD; the plan must
    instead name the checkpoint the ToolHub actually serves."""
    plan = recommend(splice_profile, registry=registry)

    assert plan["overrides"]["pretrained_weights"] == str(tmp_path / "giga-v1.pt")
    assert plan["overrides"]["lm_config"] == "giga"
    assert any("could not be served alongside" in line for line in plan["rationale"])


def test_missing_checkpoint_refuses_with_the_fix(splice_profile, registry):
    registry.configure_backbone(weights="nowhere/giga-v1.pt")

    with pytest.raises(ToolHubError, match="toolhub config --weights"):
        recommend(splice_profile, registry=registry)


def test_hub_without_weights_warns_about_a_random_backbone(splice_profile, registry):
    registry.configure_backbone(weights="null")

    plan = recommend(splice_profile, registry=registry)

    assert plan["overrides"]["pretrained_weights"] == "null"
    assert any("randomly initialised backbone" in w for w in plan["warnings"])
