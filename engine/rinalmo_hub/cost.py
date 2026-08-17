"""Per-stage training cost: iteration time and GPU memory.

Answers the question a paper table needs and the `CSVLogger` does not: how long did an
average iteration take, and how much GPU memory did it use. See `CostProfiler` for why
timing is measured start-to-end of the same batch rather than between consecutive
`batch_end` calls, and why `sync_dist` never appears here.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import statistics
import time

import lightning.pytorch as pl
import torch

from rinalmo_hub.adapter import git_sha

STAGES = ("train", "val", "test")


class _StageAccumulator:
    """Per-stage running state, reset once at that stage's first batch."""

    def __init__(self):
        self.durations_ms: List[float] = []
        self.mem_sum_bytes = 0.0
        self.mem_count = 0
        self.mem_peak_bytes = 0.0
        self.wall_seconds = 0.0
        self.batches = 0
        self.started = False
        self.batch_start: Optional[float] = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CostProfiler(pl.Callback):
    """
    Tracks mean/peak GPU memory and mean iteration time per stage (train/val/test).

    Iteration time is measured start-to-end of the *same* batch, not between consecutive
    `batch_end` calls: Lightning interleaves the validation loop inside the training epoch
    loop, so a delta-between-ends measurement would record the first train batch after
    every validation pass as one enormous iteration. Pair-based timing keeps each stage
    independent of what the other stages are doing.

    The first `warmup_steps` batches of each stage are excluded from the timing and memory
    *means* -- cuDNN autotuning, allocator growth and lazy CUDA init inflate them -- but
    always count toward the peak-memory figure, which describes the stage's true
    high-water mark, warm-up included.

    Figures are accumulated per rank under DDP; only the caller writing the summary
    (typically gated on `trainer.is_global_zero`) decides whose numbers get persisted.
    """

    def __init__(self, warmup_steps: int = 10, sync_cuda: bool = True, log_per_step: bool = True):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.sync_cuda = sync_cuda
        self.log_per_step = log_per_step
        self.accumulators: Dict[str, _StageAccumulator] = {stage: _StageAccumulator() for stage in STAGES}

    # ------------------------------------------------------------------ shared timing

    def _batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if trainer.sanity_checking:
            return

        acc = self.accumulators[stage]
        device = pl_module.device

        if not acc.started:
            acc.started = True
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if self.sync_cuda:
            _sync(device)
        acc.batch_start = time.perf_counter()

    def _batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if trainer.sanity_checking:
            return

        acc = self.accumulators[stage]
        device = pl_module.device

        if self.sync_cuda:
            _sync(device)
        now = time.perf_counter()
        elapsed = now - acc.batch_start
        duration_ms = elapsed * 1000.0

        acc.wall_seconds += elapsed
        acc.batches += 1
        measured = acc.batches > self.warmup_steps
        if measured:
            acc.durations_ms.append(duration_ms)

        allocated_gb = None
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device)
            allocated_gb = allocated / 1e9
            if measured:
                acc.mem_sum_bytes += allocated
                acc.mem_count += 1
            acc.mem_peak_bytes = torch.cuda.max_memory_allocated(device)

        if self.log_per_step:
            log_kwargs = dict(on_step=True, on_epoch=False, add_dataloader_idx=False, rank_zero_only=True)
            pl_module.log(f"cost/{stage}/iter_time_ms", duration_ms, **log_kwargs)
            if allocated_gb is not None:
                pl_module.log(f"cost/{stage}/mem_allocated_gb", allocated_gb, **log_kwargs)

    # ------------------------------------------------------------------ Lightning hooks

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start(trainer, pl_module, "train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_end(trainer, pl_module, "train")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._batch_start(trainer, pl_module, "val")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._batch_end(trainer, pl_module, "val")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._batch_start(trainer, pl_module, "test")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._batch_end(trainer, pl_module, "test")

    # ------------------------------------------------------------------ summary

    def stage_summary(self, stage: str) -> Optional[Dict[str, Any]]:
        acc = self.accumulators[stage]
        if acc.batches == 0:
            return None

        durations = sorted(acc.durations_ms)
        return {
            "batches": acc.batches,
            "measured_batches": len(durations),
            "iter_time_ms_mean": statistics.mean(durations) if durations else None,
            "iter_time_ms_median": statistics.median(durations) if durations else None,
            "iter_time_ms_p90": durations[int(0.9 * (len(durations) - 1))] if durations else None,
            "wall_seconds": acc.wall_seconds,
            "wall_ms_per_batch": (acc.wall_seconds * 1000.0) / acc.batches,
            "mem_allocated_gb_mean": (acc.mem_sum_bytes / acc.mem_count / 1e9) if acc.mem_count else None,
            "mem_allocated_gb_peak": (acc.mem_peak_bytes / 1e9) if acc.mem_peak_bytes else None,
        }

    def summary(self, device_name: Optional[str] = None, world_size: int = 1) -> Dict[str, Any]:
        stages = {stage: self.stage_summary(stage) for stage in STAGES}
        return {
            "device": device_name,
            "world_size": world_size,
            "warmup_steps": self.warmup_steps,
            **{stage: data for stage, data in stages.items() if data is not None},
        }


def build_cost_profiler(cfg) -> Optional[CostProfiler]:
    """`None` when `cost.enabled` is false -- callers append it to `extra_callbacks` as-is."""
    cost_cfg = dict(cfg.get("cost") or {})
    if not cost_cfg.get("enabled", True):
        return None

    return CostProfiler(
        warmup_steps=int(cost_cfg.get("warmup_steps", 10)),
        sync_cuda=bool(cost_cfg.get("sync_cuda", True)),
        log_per_step=bool(cost_cfg.get("log_per_step", True)),
    )


def write_run_summary(
    output_dir: Union[str, Path],
    *,
    cfg,
    task: str,
    trainer: pl.Trainer,
    profiler: Optional[CostProfiler],
    filename: str = "run_summary.json",
) -> Path:
    """Write `<output_dir>/run_summary.json`: final task metrics plus per-stage cost."""
    device = trainer.strategy.root_device
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type

    summary = {
        "task": task,
        "arm": "lora" if cfg.get("use_lora") else "full_ft",
        "seed": cfg.get("seed"),
        "git_sha": git_sha(),
        "final_metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
    }
    if profiler is not None:
        summary["cost"] = profiler.summary(device_name=device_name, world_size=trainer.world_size)

    path = Path(output_dir) / filename
    path.write_text(json.dumps(summary, indent=2))
    _print_summary(summary)

    return path


def _print_summary(summary: Dict[str, Any]) -> None:
    print(f"\nFinal metrics ({summary['task']}, {summary['arm']}):")
    for key, value in summary["final_metrics"].items():
        print(f"  {key}: {value:.4f}")

    cost = summary.get("cost")
    if not cost:
        return

    print(f"\nTraining cost ({cost['device']}, world_size={cost['world_size']}):")
    for stage in STAGES:
        stats = cost.get(stage)
        if stats is None:
            continue

        mean_ms = stats["iter_time_ms_mean"]
        parts = [f"{mean_ms:.1f} ms/it" if mean_ms is not None else "no measured batches"]

        mem_mean, mem_peak = stats["mem_allocated_gb_mean"], stats["mem_allocated_gb_peak"]
        if mem_mean is not None:
            parts.append(f"{mem_mean:.2f}/{mem_peak:.2f} GB mem (mean/peak)")

        print(f"  {stage}: " + ", ".join(parts))
