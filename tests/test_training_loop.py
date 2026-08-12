"""End-to-end `trainer.fit` on the CPU, with synthetic in-memory data.

Not one of the numbered checks in §9, but it is the cheapest way to find out that the
generic train/val/test steps, metric aggregation, the target-scaler fit, checkpoint
slimming and the gradual-unfreezing schedules actually work -- before an eight-hour GPU run
discovers it for you. `nano` backbone, a handful of fake sequences, seconds.
"""

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from rinalmo.utils.finetune_callback import GradualUnfreezing
from rinalmo_hub.config import load_yaml
from tests.helpers import LORA_CONFIGS, build_module

REPO_ROOT = Path(__file__).resolve().parent.parent
SEQ_LEN = 24
NUM_SAMPLES = 8


def _trainer(tmp_path, max_epochs=2, callbacks=None):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
        callbacks=callbacks or [],
    )


class _TokenDataset(Dataset):
    """`(tokens, target)` batches, the shape both splice site and MRL consume."""

    def __init__(self, target_fn, n=NUM_SAMPLES):
        generator = torch.Generator().manual_seed(0)
        # Token ids 5.. are the RNA nucleotides; 0-4 are the special tokens.
        self.tokens = torch.randint(5, 10, (n, SEQ_LEN), generator=generator)
        self.targets = target_fn(generator, n)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.targets[idx]


class _SimpleDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=4):
        super().__init__()
        self.train_dataset = dataset
        self.batch_size = batch_size

    def _loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    train_dataloader = val_dataloader = test_dataloader = _loader


def _splice_site_datamodule():
    return _SimpleDataModule(
        _TokenDataset(lambda g, n: (torch.rand(n, generator=g) > 0.5).float())
    )


def _mrl_datamodule():
    return _SimpleDataModule(
        _TokenDataset(lambda g, n: torch.rand(n, generator=g) * 4.0 + 3.0)
    )


# ---------------------------------------------------------------- LoRA training


def test_lora_fit_runs_and_only_touches_lora_and_head(tmp_path):
    module = build_module("splice_site", lora=LORA_CONFIGS["stride3"])
    frozen_before = {
        name: param.detach().clone()
        for name, param in module.backbone.named_parameters()
        if "lora_" not in name
    }
    lora_b_before = {
        name: param.detach().clone()
        for name, param in module.named_parameters()
        if "lora_B" in name
    }

    _trainer(tmp_path).fit(module, datamodule=_splice_site_datamodule())

    for name, before in frozen_before.items():
        after = dict(module.backbone.named_parameters())[name]
        assert torch.equal(before, after), f"frozen backbone parameter '{name}' moved"

    moved = [n for n, before in lora_b_before.items()
             if not torch.equal(before, dict(module.named_parameters())[n])]
    assert moved, "no lora_B parameter received an update"


def test_metrics_are_logged_per_stage(tmp_path):
    module = build_module("splice_site", lora=LORA_CONFIGS["stride3"])
    trainer = _trainer(tmp_path)
    datamodule = _splice_site_datamodule()

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)

    logged = set(trainer.callback_metrics)
    assert {"test/f1_score", "test/acc", "test/precision", "test/recall"} <= logged
    assert "test/loss" in logged


def test_lora_checkpoint_holds_only_the_adapter(tmp_path):
    """`strict_loading=False` plus `on_save_checkpoint` filtering: 7.8 GB -> 18 MB."""
    module = build_module("splice_site", lora=LORA_CONFIGS["stride3"])
    trainer = _trainer(tmp_path)
    trainer.fit(module, datamodule=_splice_site_datamodule())

    path = tmp_path / "ckpt.ckpt"
    trainer.save_checkpoint(path)

    state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    assert state_dict, "checkpoint is empty"
    for key in state_dict:
        assert "lora_" in key or key.startswith("head."), key

    assert module.strict_loading is False


def test_full_ft_checkpoint_keeps_everything(tmp_path):
    module = build_module("splice_site", lora=None)
    trainer = _trainer(tmp_path, max_epochs=1)
    trainer.fit(module, datamodule=_splice_site_datamodule())

    path = tmp_path / "full.ckpt"
    trainer.save_checkpoint(path)

    state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    assert any(k.startswith("backbone.transformer") for k in state_dict)
    assert module.strict_loading is True


# ---------------------------------------------------------------- MRL target scaler


def test_mrl_scaler_is_fitted_before_the_first_training_step(tmp_path):
    """
    The source fitted it inside `training_step` during epoch 0 and returned None, which made
    epoch 0 a non-training epoch and made the task unrunnable under DDP.
    """
    module = build_module("mrl", lora=LORA_CONFIGS["stride3"])
    datamodule = _mrl_datamodule()

    assert module.scaler._mean.item() == 0.0 and module.scaler._std.item() == 1.0

    trainer = _trainer(tmp_path, max_epochs=1)
    trainer.fit(module, datamodule=datamodule)

    expected = datamodule.train_dataset.targets
    assert module.scaler._mean.item() == pytest.approx(expected.mean().item(), abs=1e-4)
    assert module.scaler._std.item() == pytest.approx(expected.std(unbiased=False).item(), abs=1e-4)

    # Epoch 0 was a real training epoch: two batches of four samples.
    assert trainer.global_step == len(datamodule.train_dataset) // datamodule.batch_size


def test_mrl_training_step_never_returns_none(tmp_path):
    """
    What made the task impossible to run under DDP: Lightning raises "Skipping the
    training_step by returning None in distributed training is not supported".
    """
    outputs = []

    class _CaptureOutputs(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, out, batch, batch_idx):
            outputs.append(out)

    module = build_module("mrl", lora=LORA_CONFIGS["stride3"])
    datamodule = _mrl_datamodule()

    _trainer(tmp_path, max_epochs=2, callbacks=[_CaptureOutputs()]).fit(module, datamodule=datamodule)

    assert len(outputs) == 2 * (len(datamodule.train_dataset) // datamodule.batch_size)
    assert all(out is not None for out in outputs), "a training step returned None"


def test_mrl_predictions_come_back_on_the_original_scale(tmp_path):
    module = build_module("mrl", lora=LORA_CONFIGS["stride3"])
    datamodule = _mrl_datamodule()
    _trainer(tmp_path, max_epochs=1).fit(module, datamodule=datamodule)

    tokens, _ = next(iter(datamodule.train_dataloader()))
    module.eval()
    with torch.no_grad():
        scaled = module(tokens)
        unscaled = module.postprocess_predictions(scaled, tokens, None)

    assert not torch.allclose(scaled, unscaled)
    assert (unscaled >= 0).all()


# ---------------------------------------------------------------- schedules


@pytest.mark.parametrize(
    "schedule", sorted((REPO_ROOT / "ft_schedules").glob("*.yaml")), ids=lambda p: p.stem
)
def test_shipped_schedules_use_the_fixed_names(schedule):
    """
    §4.1: the backbone is `backbone`, the head is `head`, everywhere.

    The source repo called the backbone `lm` in four scripts and `rinalmo` in two, so its
    schedules were not interchangeable between tasks. Nothing here may mention either.
    """
    entries = load_yaml(schedule)

    assert 0 in entries, f"{schedule.name} has no epoch-0 phase"
    for epoch, patterns in entries.items():
        for pattern in patterns:
            assert pattern.startswith(("head.", "backbone.")), f"{schedule.name}[{epoch}]: {pattern}"


@pytest.mark.parametrize(
    "schedule,transformer_trains",
    [
        ("ft_schedules/splice_site.yaml", True),   # phase 0 unfreezes a backbone *subtree*
        ("ft_schedules/mrl.yaml", False),          # phase 0 is head-only
        ("ft_schedules/mrl_paper.yaml", False),
        ("ft_schedules/sec_struct.yaml", False),
    ],
)
def test_phase_zero_trains_exactly_what_the_schedule_says(schedule, transformer_trains):
    """
    `self.freeze()` walks a whole subtree, so freezing every non-matching module would
    freeze the *ancestors* of a matched one too: a phase 0 of `backbone.transformer.*`
    would freeze the transformer via the `backbone` root and silently train the head alone.
    """
    module = build_module("splice_site", lora=None)
    GradualUnfreezing(str(REPO_ROOT / schedule), initial_denom_lr=1.0).freeze_before_training(module)

    assert all(p.requires_grad for p in module.head.parameters())
    assert any(p.requires_grad for p in module.backbone.transformer.parameters()) is transformer_trains

    # The embedding and the unused masked-LM head never train in phase 0 of any schedule.
    assert not any(p.requires_grad for p in module.backbone.embedding.parameters())
    assert not any(p.requires_grad for p in module.backbone.lm_mask_head.parameters())


def test_schedule_freezes_everything_but_the_head_at_epoch_zero(tmp_path):
    schedule = tmp_path / "schedule.yaml"
    schedule.write_text("0:\n  - head.*\n1:\n  - backbone.transformer.*\n")

    module = build_module("splice_site", lora=None)
    callback = GradualUnfreezing(str(schedule), initial_denom_lr=1.0)
    trainer = _trainer(tmp_path, max_epochs=1, callbacks=[callback])

    trainer.fit(module, datamodule=_splice_site_datamodule())

    assert all(p.requires_grad for p in module.head.parameters())
    assert not any(p.requires_grad for p in module.backbone.embedding.parameters())


def test_initial_denom_lr_is_configurable(tmp_path):
    """
    Lightning's `BaseFinetuning` default of 10.0 is what trained the MRL backbone at 1e-6
    instead of the requested 1e-5 for 44 epochs.
    """
    schedule = tmp_path / "schedule.yaml"
    schedule.write_text("0:\n  - head.*\n1:\n  - backbone.transformer.*\n")

    lr = 1e-4
    groups = {}
    for denom in (1.0, 10.0):
        module = build_module("splice_site", lora=None)
        module.optim_config = {"name": "adam", "lr": lr}

        trainer = _trainer(tmp_path, max_epochs=2,
                           callbacks=[GradualUnfreezing(str(schedule), initial_denom_lr=denom)])
        trainer.fit(module, datamodule=_splice_site_datamodule())

        groups[denom] = trainer.optimizers[0].param_groups

    assert len(groups[1.0]) == 2, "the schedule never added a backbone param group"
    assert groups[1.0][-1]["lr"] == pytest.approx(lr)
    assert groups[10.0][-1]["lr"] == pytest.approx(lr / 10.0)


def test_scheduler_is_built_from_config(tmp_path):
    module = build_module("mrl", lora=LORA_CONFIGS["stride3"])
    module.optim_config = {
        "name": "adam", "lr": 1e-4,
        "scheduler": {"name": "linear", "start_factor": 1.0, "end_factor": 0.1,
                      "total_iters": 5000, "interval": "step"},
    }

    optimizers = module.configure_optimizers()
    scheduler = optimizers["lr_scheduler"]

    assert scheduler["interval"] == "step"
    assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.LinearLR)
    assert scheduler["scheduler"].total_iters == 5000


def test_unknown_scheduler_key_is_not_silently_ignored():
    module = build_module("mrl")
    module.optim_config = {"name": "adam", "lr": 1e-4,
                           "scheduler": {"name": "linear", "totalIters": 500}}

    with pytest.raises(ValueError, match="Unused scheduler config keys"):
        module.configure_optimizers()
