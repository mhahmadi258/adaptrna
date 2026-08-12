"""Config resolution and CLI wiring -- CPU only, no training.

The `--set optim.lr=3e-4` case is the one worth staring at: YAML 1.1 parses `3e-4` as the
*string* "3e-4", so a naive `yaml.safe_load` override would hand the optimizer a string and
either crash deep inside torch or, worse, silently coerce.
"""

from pathlib import Path

import pytest
import torch

from rinalmo_hub.cli import evaluate as evaluate_cli
from rinalmo_hub.cli import predict as predict_cli
from rinalmo_hub.cli import train as train_cli
from rinalmo_hub.cli.common import _multi_device, build_module, build_trainer, resolve_run_config
from rinalmo_hub.config import Config, deep_merge, parse_scalar, resolve_config, save_config
from rinalmo_hub.registry import available_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_CONFIGS = sorted((REPO_ROOT / "configs" / "tasks").glob("*.yaml"))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3e-4", 3e-4),
        ("1e-6", 1e-6),
        ("1.0e-4", 1.0e-4),
        ("42", 42),
        ("0.5", 0.5),
        ("true", True),
        ("False", False),
        ("null", None),
        ("", None),
        ("bf16-mixed", "bf16-mixed"),
        ("[a, b]", ["a", "b"]),
    ],
)
def test_parse_scalar(text, expected):
    assert parse_scalar(text) == expected


def test_scientific_notation_is_a_float_not_a_string():
    """The trap: `yaml.safe_load("3e-4")` returns the string "3e-4"."""
    assert isinstance(parse_scalar("3e-4"), float)


def test_deep_merge_is_recursive():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}


def test_config_supports_both_access_styles():
    cfg = Config({"data": {"root": "x"}})
    assert cfg.data.root == cfg["data"]["root"] == "x"

    with pytest.raises(AttributeError):
        cfg.nope


@pytest.mark.parametrize("path", TASK_CONFIGS, ids=lambda p: p.stem)
def test_shipped_task_configs_resolve(path):
    cfg = resolve_config(path)

    assert cfg.task in available_tasks()
    assert cfg.lm_config == "giga"
    assert cfg.trainer.precision == "bf16-mixed"      # inherited from base.yaml
    assert cfg.finetune.initial_denom_lr == 1.0       # not Lightning's 10.0
    assert set(cfg.lora) >= {"r", "alpha", "dropout", "layer_stride", "target_modules"}


def test_overrides_beat_the_task_yaml():
    path = REPO_ROOT / "configs" / "tasks" / "mrl.yaml"
    cfg = resolve_config(path, ["optim.lr=3e-4", "lora.layer_stride=1", "trainer.max_epochs=2"])

    assert cfg.optim.lr == 3e-4
    assert cfg.lora.layer_stride == 1
    assert cfg.trainer.max_epochs == 2
    assert cfg.data.batch_size == 64  # untouched by the overrides


def test_override_can_create_a_missing_key():
    cfg = resolve_config(None, ["data.brand_new=7"])
    assert cfg.data.brand_new == 7


def test_malformed_override_raises():
    with pytest.raises(ValueError, match="dotted.key=value"):
        resolve_config(None, ["optim.lr"])


def test_resolved_config_is_written_for_reproducibility(tmp_path):
    cfg = resolve_config(REPO_ROOT / "configs" / "tasks" / "mrl.yaml", ["optim.lr=3e-4"])
    path = save_config(cfg, tmp_path / "resolved_config.yaml")

    assert resolve_config(path).optim.lr == 3e-4


@pytest.mark.parametrize(
    "devices,expected",
    [("1", False), ("2", True), ("0,1", True), (1, False), (4, True), (-1, True), ([0, 1], True)],
)
def test_multi_device_resolution(devices, expected):
    assert _multi_device(devices) is expected


def test_devices_auto_is_resolved_not_compared():
    """The source's MRL script only checked explicit device counts and crashed on 'auto'."""
    assert _multi_device("auto") == (torch.cuda.device_count() > 1)


# ---------------------------------------------------------------- CLI wiring


def _train_args(argv):
    return train_cli.build_parser().parse_args(argv)


def test_train_cli_builds_a_lora_run_on_cpu(tmp_path):
    args = _train_args([
        "--task", "splice_site",
        "--use_lora",
        "--set", "lm_config=nano",
        "--set", "pretrained_weights=null",
        "--set", "head.head_embed_dim=16",
        "--set", "optim.lr=3e-4",
        "--set", "lora.layer_stride=3",
        "--set", "trainer.accelerator=cpu",
        "--set", "trainer.devices=1",
        "--set", "trainer.precision=32-true",
        "--output_dir", str(tmp_path),
    ])

    cfg, task = resolve_run_config(args)
    assert task == "splice_site"
    assert cfg.use_lora is True
    assert cfg.optim.lr == 3e-4

    module = build_module(cfg, task, args, verbose=False)
    assert module.use_lora
    assert module.lora_spec.layer_stride == 3

    optimizers = module.configure_optimizers()
    assert optimizers["optimizer"].param_groups[0]["lr"] == 3e-4
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    assert 0 < trainable < total

    trainer = build_trainer(cfg, str(tmp_path))
    callbacks = {type(c).__name__ for c in trainer.callbacks}
    # Rich buffers on a non-TTY, so a redirected log stays empty for the whole run.
    assert "TQDMProgressBar" in callbacks
    assert "LearningRateMonitor" in callbacks
    assert any(type(logger).__name__ == "CSVLogger" for logger in trainer.loggers)


def test_no_checkpoint_callback_unless_asked(tmp_path):
    """
    Lightning installs a *default* `ModelCheckpoint` when `enable_checkpointing` is left
    alone, so a full-FT run would quietly write 7.8 GB per epoch.
    """
    args = _train_args(["--task", "mrl", "--set", "lm_config=nano"])
    cfg, _ = resolve_run_config(args)

    trainer = build_trainer(cfg, str(tmp_path))
    assert not any(type(c).__name__ == "ModelCheckpoint" for c in trainer.callbacks)
    assert trainer.checkpoint_callback is None

    args = _train_args(["--task", "mrl", "--set", "lm_config=nano",
                        "--set", "trainer.checkpoint_every_epoch=true"])
    cfg, _ = resolve_run_config(args)

    trainer = build_trainer(cfg, str(tmp_path))
    assert any(type(c).__name__ == "ModelCheckpoint" for c in trainer.callbacks)


def test_full_ft_run_uses_every_parameter(tmp_path):
    args = _train_args([
        "--task", "splice_site",
        "--set", "lm_config=nano",
        "--set", "pretrained_weights=null",
        "--set", "head.head_embed_dim=16",
        "--output_dir", str(tmp_path),
    ])

    cfg, task = resolve_run_config(args)
    module = build_module(cfg, task, args, verbose=False)

    assert module.use_lora is False
    assert all(p.requires_grad for p in module.parameters())


def test_lora_and_a_finetune_schedule_are_mutually_exclusive(tmp_path):
    args = _train_args([
        "--task", "mrl", "--use_lora",
        "--set", "lm_config=nano",
        "--set", "finetune.schedule=ft_schedules/mrl_paper.yaml",
    ])
    cfg, _ = resolve_run_config(args)

    with pytest.raises(SystemExit, match="mutually exclusive"):
        build_trainer(cfg, str(tmp_path))


def test_missing_backbone_weights_fail_loudly(tmp_path):
    args = _train_args([
        "--task", "mrl",
        "--set", "lm_config=nano",
        "--set", "pretrained_weights=weights/does-not-exist.pt",
    ])
    cfg, task = resolve_run_config(args)

    with pytest.raises(SystemExit, match="not found"):
        build_module(cfg, task, args, verbose=False)


def test_unknown_task_is_rejected():
    args = _train_args(["--task", "not_a_task"])

    with pytest.raises(SystemExit, match="Unknown task"):
        resolve_run_config(args)


def test_adapter_supplies_task_and_geometry(tmp_path):
    """`--adapter` alone is enough: the file carries task, head config and LoRA geometry."""
    from tests.helpers import LORA_CONFIGS, build_module as build_test_module, randomise_trainable

    path = randomise_trainable(
        build_test_module("mrl", lora=LORA_CONFIGS["stride1"])
    ).save_adapter(tmp_path / "mrl.pt")

    args = _train_args(["--adapter", str(path), "--set", "pretrained_weights=null"])
    cfg, task = resolve_run_config(args)

    assert task == "mrl"
    assert cfg.lm_config == "nano"
    assert cfg.use_lora is True
    assert cfg.lora["layer_stride"] == 1
    assert cfg.head == {"head_embed_dim": 8, "num_blocks": 1}

    module = build_module(cfg, task, args, verbose=False)
    assert module.scaler._mean.shape == (1,)


def test_evaluate_cli_requires_weights_to_evaluate():
    args = evaluate_cli.build_parser().parse_args(["--task", "mrl"])

    with pytest.raises(SystemExit, match="Nothing to evaluate"):
        evaluate_cli.main(["--task", "mrl"])

    assert args.split == "test"


def test_predict_cli_reads_sequences_from_a_file(tmp_path):
    path = tmp_path / "seqs.txt"
    path.write_text("> header\nGGCAUUAC\n# comment\nAUGCAUGC\n\n")

    args = predict_cli.build_parser().parse_args(["--adapter", "x.pt", "--input", str(path)])
    assert predict_cli.read_sequences(args) == ["GGCAUUAC", "AUGCAUGC"]
