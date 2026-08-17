"""`CostProfiler` -- per-stage iteration time and GPU memory, CPU-only.

Mirrors `test_training_loop.py`'s synthetic-data harness: a `nano` backbone, a handful of
fake sequences, seconds. GPU memory fields are exercised for `None`-on-CPU behaviour only;
the real numbers need a GPU, see the "Real GPU check" in
`plans/TRAINING_COST_INSTRUMENTATION.md`.
"""

import json

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from rinalmo_hub.cli import train as train_cli
from rinalmo_hub.cli.common import build_trainer, resolve_run_config
from rinalmo_hub.cost import CostProfiler, build_cost_profiler, write_run_summary
from tests.helpers import LORA_CONFIGS, build_module

SEQ_LEN = 24
NUM_SAMPLES = 8
BATCH_SIZE = 4


class _TokenDataset(Dataset):
    def __init__(self, n=NUM_SAMPLES):
        generator = torch.Generator().manual_seed(0)
        self.tokens = torch.randint(5, 10, (n, SEQ_LEN), generator=generator)
        self.targets = (torch.rand(n, generator=generator) > 0.5).float()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.targets[idx]


class _SimpleDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = _TokenDataset()

    def _loader(self):
        return DataLoader(self.dataset, batch_size=BATCH_SIZE)

    train_dataloader = val_dataloader = test_dataloader = _loader


def _fit_trainer(tmp_path, profiler, max_epochs=2):
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=max_epochs, precision="32-true",
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, default_root_dir=str(tmp_path), callbacks=[profiler],
    )
    module = build_module("splice_site", lora=LORA_CONFIGS["stride3"])
    datamodule = _SimpleDataModule()
    trainer.fit(module, datamodule=datamodule)

    return trainer, module, datamodule


# ---------------------------------------------------------------- build_cost_profiler


def test_disabled_config_builds_no_profiler():
    assert build_cost_profiler({"cost": {"enabled": False}}) is None


def test_missing_cost_block_uses_defaults():
    profiler = build_cost_profiler({})
    assert isinstance(profiler, CostProfiler)
    assert profiler.warmup_steps == 10
    assert profiler.sync_cuda is True
    assert profiler.log_per_step is True


def test_cost_block_is_respected():
    profiler = build_cost_profiler(
        {"cost": {"warmup_steps": 3, "sync_cuda": False, "log_per_step": False}}
    )
    assert profiler.warmup_steps == 3
    assert profiler.sync_cuda is False
    assert profiler.log_per_step is False


def test_cost_disabled_via_cli_override():
    args = train_cli.build_parser().parse_args(
        ["--task", "mrl", "--set", "lm_config=nano", "--set", "cost.enabled=false"]
    )
    cfg, _ = resolve_run_config(args)
    assert build_cost_profiler(cfg) is None


def test_cost_profiler_wired_into_trainer_by_default(tmp_path):
    args = train_cli.build_parser().parse_args(["--task", "mrl", "--set", "lm_config=nano"])
    cfg, _ = resolve_run_config(args)

    profiler = build_cost_profiler(cfg)
    trainer = build_trainer(cfg, str(tmp_path), extra_callbacks=[profiler])

    assert any(isinstance(c, CostProfiler) for c in trainer.callbacks)


# ---------------------------------------------------------------- CostProfiler / fit


def test_warmup_steps_are_excluded_from_the_mean(tmp_path):
    profiler = CostProfiler(warmup_steps=1, sync_cuda=False)
    trainer, _, datamodule = _fit_trainer(tmp_path, profiler, max_epochs=2)

    batches_per_epoch = len(datamodule.dataset) // BATCH_SIZE
    train_acc = profiler.accumulators["train"]
    assert train_acc.batches == batches_per_epoch * 2
    assert len(train_acc.durations_ms) == train_acc.batches - 1


def test_sanity_check_batches_are_excluded(tmp_path):
    """Lightning's default 2-batch sanity check must not leak into the val accumulator."""
    profiler = CostProfiler(warmup_steps=0, sync_cuda=False)
    _, _, datamodule = _fit_trainer(tmp_path, profiler, max_epochs=2)

    batches_per_epoch = len(datamodule.dataset) // BATCH_SIZE
    val_acc = profiler.accumulators["val"]
    assert val_acc.batches == batches_per_epoch * 2  # 2 epochs, no sanity-check batches


def test_val_and_test_are_tracked_separately(tmp_path):
    profiler = CostProfiler(warmup_steps=0, sync_cuda=False)
    trainer, module, datamodule = _fit_trainer(tmp_path, profiler, max_epochs=1)
    trainer.test(module, datamodule=datamodule)

    assert profiler.accumulators["train"].batches > 0
    assert profiler.accumulators["val"].batches > 0
    assert profiler.accumulators["test"].batches > 0


def test_warmup_larger_than_batch_count_yields_null_mean(tmp_path):
    profiler = CostProfiler(warmup_steps=1000, sync_cuda=False)
    _fit_trainer(tmp_path, profiler, max_epochs=1)

    summary = profiler.stage_summary("train")
    assert summary["measured_batches"] == 0
    assert summary["iter_time_ms_mean"] is None
    assert summary["wall_seconds"] > 0  # wall time is still tracked, warm-up or not


def test_memory_fields_are_null_on_cpu(tmp_path):
    profiler = CostProfiler(warmup_steps=0, sync_cuda=False)
    _fit_trainer(tmp_path, profiler, max_epochs=1)

    summary = profiler.stage_summary("train")
    assert summary["mem_allocated_gb_mean"] is None
    assert summary["mem_allocated_gb_peak"] is None


def test_per_step_metrics_land_in_the_logged_columns(tmp_path):
    profiler = CostProfiler(warmup_steps=0, sync_cuda=False, log_per_step=True)
    trainer, _, _ = _fit_trainer(tmp_path, profiler, max_epochs=1)

    assert "cost/train/iter_time_ms" in trainer.callback_metrics


def test_log_per_step_false_logs_nothing(tmp_path):
    profiler = CostProfiler(warmup_steps=0, sync_cuda=False, log_per_step=False)
    trainer, _, _ = _fit_trainer(tmp_path, profiler, max_epochs=1)

    assert "cost/train/iter_time_ms" not in trainer.callback_metrics


# ---------------------------------------------------------------- write_run_summary


def test_run_summary_is_written_and_parses(tmp_path):
    profiler = CostProfiler(warmup_steps=0, sync_cuda=False)
    trainer, module, datamodule = _fit_trainer(tmp_path, profiler, max_epochs=1)
    trainer.test(module, datamodule=datamodule)

    path = write_run_summary(
        tmp_path, cfg={"use_lora": True, "seed": 7}, task="splice_site",
        trainer=trainer, profiler=profiler,
    )

    summary = json.loads(path.read_text())
    assert summary["task"] == "splice_site"
    assert summary["arm"] == "lora"
    assert summary["seed"] == 7
    assert "final_metrics" in summary
    assert set(summary["cost"]) >= {"train", "val", "test", "device", "world_size", "warmup_steps"}


def test_run_summary_without_a_profiler_has_no_cost_key(tmp_path):
    trainer, _, _ = _fit_trainer(tmp_path, CostProfiler(), max_epochs=1)

    path = write_run_summary(
        tmp_path, cfg={"use_lora": False, "seed": None}, task="mrl",
        trainer=trainer, profiler=None,
    )

    summary = json.loads(path.read_text())
    assert summary["arm"] == "full_ft"
    assert "cost" not in summary
