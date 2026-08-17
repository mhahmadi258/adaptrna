# Training cost instrumentation — GPU memory and iteration time

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) · engine / fine-tuning pipeline
>
> **Definition of done:** every training run writes a `run_summary.json` next to its
> metrics carrying mean/peak GPU memory and mean iteration time for the train, val
> and test stages, alongside the final task metrics — enough to build a paper cost
> table by globbing run directories.
>
> **Status:** planned · not started

## Context

Paper experiments comparing LoRA against full fine-tuning need a *cost* column next
to the quality column. Today a run reports only task metrics: `self.log()` values
land in a `CSVLogger` ([common.py:155-159](../engine/rinalmo_hub/cli/common.py#L155-L159))
and `trainer.callback_metrics` is baked into the adapter's metadata
([train.py:57-60](../engine/rinalmo_hub/cli/train.py#L57-L60)).

Neither GPU memory nor per-iteration time is recorded anywhere in the training path —
confirmed by search: no `torch.cuda.*memory*`, `time.perf_counter()` or
`torch.cuda.synchronize()` in `module.py`, `common.py` or `train.py`. The only code
doing this is [benchmark_switching.py](../engine/scripts/benchmark_switching.py), which
measures *inference* adapter-switching latency and is unrelated to training.
Lightning's `TQDMProgressBar` shows a live it/s rate, but it is never persisted, so
there is nothing to average after the fact.

## 1. Decisions

| Question | Decision |
|---|---|
| Memory statistic | **Mean allocated** and **peak allocated** (`torch.cuda.memory_allocated` / `max_memory_allocated`). Reserved-memory figures skipped. |
| Warm-up | **Skip the first N steps** (`warmup_steps`, default 10) from the *means*. First iterations pay cuDNN autotuning, allocator growth and lazy CUDA init. |
| Where results go | **JSON summary file** in `--output_dir`, **printed** at end of run, and **per-step columns in `metrics.csv`**. Adapter metadata left alone. |
| Scope | **Train, val and test tracked separately.** |

### Naming

The new engine module is `rinalmo_hub/cost.py` and the config block is `cost:` —
**not** `profiling`. That name is already taken by
`agentic/adaptrna_agentic/profiling/`, which profiles *datasets* to produce a training
plan ([profiling-and-knowledge.md](../documents/modules/profiling-and-knowledge.md)).
Two unrelated "profiling" concepts would be a permanent source of confusion. The
Lightning callback class keeps the name `CostProfiler`, which is unambiguous in context.

## 2. Files to change

### Engine

| File | Change |
|---|---|
| [rinalmo_hub/cost.py](../engine/rinalmo_hub/cost.py) | **New.** `CostProfiler` callback, `build_cost_profiler(cfg)`, `write_run_summary(...)` |
| [cli/train.py](../engine/rinalmo_hub/cli/train.py) | Build the profiler, pass it via the existing `extra_callbacks` parameter, write the summary at the end of `main()` |
| [configs/base.yaml](../engine/configs/base.yaml) | New top-level `cost:` block |
| [tests/test_cost.py](../engine/tests/test_cost.py) | **New.** CPU-only tests |

`build_trainer()` is deliberately **not** modified — it already accepts
`extra_callbacks` ([common.py:150](../engine/rinalmo_hub/cli/common.py#L150)), so
`train.py` can inject the profiler with zero change to shared plumbing, leaving
`evaluate.py` and `predict.py` untouched.

A flat module rather than a `callbacks/` subpackage: `pyproject.toml` lines 39-52 list
packages explicitly, so a subpackage would also require a packaging change for no benefit.

### Agentic layer

| File | Change |
|---|---|
| [jobs/analysis.py](../agentic/adaptrna_agentic/jobs/analysis.py) | `_final_metrics` skips the `cost/` prefix — see §4 |

### Documents

See §6.

## 3. Design

### 3.1 `CostProfiler(pl.Callback)`

Style reference: [finetune_callback.py](../engine/rinalmo/utils/finetune_callback.py) —
module docstring, Google-style `Args:`, prose comments explaining *why*, `_`-prefixed
helpers. Mirror the existing `_step(batch, stage)` shape in
[module.py:241](../engine/rinalmo_hub/module.py#L241) with generic `_batch_start(stage)`
/ `_batch_end(stage, pl_module)` helpers driven by the six Lightning hooks
(`on_{train,validation,test}_batch_{start,end}`), so the three stages share one code path.

Per stage it keeps an accumulator holding: a list of per-batch durations, a running sum
and count of sampled `memory_allocated`, the stage wall-clock start, and total
accumulated wall seconds.

**Iteration timing — pair-based, not delta-based.** Duration is measured from
`on_*_batch_start` to `on_*_batch_end` for the same batch. Lightning calls
`on_train_batch_end` after the optimizer step, so this window covers the full forward +
backward + step.

> The trap this avoids: the obvious alternative — timing the delta between consecutive
> `batch_end` calls — breaks because Lightning interleaves the validation loop *inside*
> the training epoch loop. The first train batch after each val pass would be recorded
> as one enormous iteration. Pair-based timing is immune, and each stage stays
> independently clean.

Because pair-based timing excludes dataloader wait between iterations, the accumulator
*also* records total stage wall seconds and batch count. The JSON reports both:
`iter_time_ms_mean` (compute) and `wall_ms_per_batch` (inclusive). The gap between them
is the data-pipeline overhead, and their agreement is a sanity check on the measurement.

**CUDA synchronisation.** CUDA kernels are asynchronous, so a `perf_counter()` stamp at
`batch_end` measures *enqueue*, not completion. `torch.cuda.synchronize()` is called
before each stamp, guarded by a `sync_cuda` config flag (default on — these are
measurement runs). Reuse the `_sync(device)` idiom from `benchmark_switching.py` lines 44-46.

**Memory.** `torch.cuda.reset_peak_memory_stats()` once at each stage's first entry,
then at each `batch_end` sample `memory_allocated()` into the running mean. Peak is read
at stage end via `max_memory_allocated()`. Peak covers the whole stage including warm-up
(warm-up depresses timing, not the high-water mark); the mean honours `warmup_steps`.
The peak still includes resident model weights, which is what makes it the "does this
fit on the card" number.

Validation runs once per epoch, so `on_validation_start` fires repeatedly: reset peak
stats only on first entry, and accumulate wall seconds across all passes.

**Guards:**

- `trainer.sanity_checking` → skip entirely, matching the existing pattern at
  [module.py:260](../engine/rinalmo_hub/module.py#L260).
- Non-CUDA device → timing still works; memory fields emit `null` rather than `nan`, so
  the JSON stays clean on CPU.
- Zero measured batches (`warmup_steps` ≥ batch count) → `null` mean, never a
  `ZeroDivisionError`.
- DDP → figures are per-rank; only rank zero's are written. Recorded in the JSON as
  `world_size` so a multi-GPU run is never mistaken for a single-GPU one.

### 3.2 Per-step CSV logging

Guarded by `log_per_step`. From the callback, call
`pl_module.log(f"cost/{stage}/iter_time_ms", ...)` and `f"cost/{stage}/mem_allocated_gb"`
with `on_step=True, on_epoch=False`.

> **Do not pass `sync_dist=True` here**, unlike every other `self.log()` call in
> `module.py`. `sync_dist` inserts an all-reduce per logged value per step, which would
> directly inflate the very iteration time being measured.

The `cost/` prefix is load-bearing for §4, not cosmetic.

The config's `log_every_n_steps: 50` means the CSV holds a downsampled series. That is
fine and intended — the callback's own accumulators are full-fidelity, so the reported
averages are unaffected by CSV downsampling.

### 3.3 Run summary

`write_run_summary(path, cfg, task, trainer, profiler)` writes
`<output_dir>/run_summary.json` and prints a compact block to stdout:

```json
{
  "task": "mrl",
  "arm": "lora",
  "seed": 42,
  "git_sha": "...",
  "final_metrics": { "test/r2": 0.81 },
  "cost": {
    "device": "NVIDIA A100-SXM4-40GB",
    "world_size": 1,
    "warmup_steps": 10,
    "train": {
      "batches": 5000, "measured_batches": 4990,
      "iter_time_ms_mean": 412.7, "iter_time_ms_median": 410.2,
      "iter_time_ms_p90": 428.9,
      "wall_seconds": 2110.4, "wall_ms_per_batch": 422.1,
      "mem_allocated_gb_mean": 18.3, "mem_allocated_gb_peak": 21.7
    },
    "val": { "...": "same shape" },
    "test": { "...": "same shape" }
  }
}
```

Median and p90 come free from the retained duration list and are far more robust than
the mean for a paper table (100k steps ≈ 800 KB of floats — negligible).

Reuse `git_sha()` from [adapter.py](../engine/rinalmo_hub/adapter.py) rather than
re-implementing it. Pull `final_metrics` from `trainer.callback_metrics` exactly as
`_save_artifacts` already does (`train.py` line 58).

### 3.4 Wiring in `train.py`

In `main()` (`train.py` lines 85-104):

```python
profiler = build_cost_profiler(cfg)
trainer = build_trainer(cfg, args.output_dir,
                        extra_callbacks=[profiler] if profiler else None)
```

Call `write_run_summary(...)` at the **end** of `main()`, after `trainer.test(...)` at
line 102 and guarded by `trainer.is_global_zero`, so the summary captures test metrics
as well as training ones. `_save_artifacts` and its metadata are left exactly as they are.

### 3.5 Config

Add to [base.yaml](../engine/configs/base.yaml) as a **top-level** block:

```yaml
cost:
  enabled: true
  warmup_steps: 10      # first iterations pay cuDNN autotuning and allocator growth
  log_per_step: true    # emit cost/* columns into metrics.csv
  sync_cuda: true       # torch.cuda.synchronize() before each stamp; off = faster, wrong
```

> **This must not go under `trainer:`.** `build_trainer` pops the keys it knows and
> splats the remainder into `pl.Trainer(**trainer_cfg)` (`common.py` line 206), so any
> unrecognised key under `trainer:` raises a `TypeError` from Lightning. A top-level
> block sidesteps that.

No CLI flag is needed — `resolve_config` already supports `--set cost.enabled=false`,
and `test_override_can_create_a_missing_key` (`tests/test_config_and_cli.py` lines 83-85)
confirms unknown dotted keys are accepted.

## 4. Downstream dependency — the agentic job layer reads `metrics.csv`

`metrics.csv` is a documented interface, not a private artifact. Two consumers iterate
**every** column except `epoch`/`step`:

- `_final_metrics` ([analysis.py:246](../agentic/adaptrna_agentic/jobs/analysis.py#L246))
  — feeds the post-run analysis report and the comparison against reference metrics.
- `read_progress` ([runner.py:248](../agentic/adaptrna_agentic/jobs/runner.py#L248))
  — feeds the live job-status panel in the UI.

Without action, `cost/train/iter_time_ms` would show up as a *task metric* in both. The
deliberate split:

- **`_final_metrics`: filter out the `cost/` prefix.** A run's quality metrics should not
  silently acquire two timing columns; the analyzer compares this dict against reference
  metrics.
- **`read_progress`: leave unfiltered.** Live iteration time and memory in the job panel
  is a genuine improvement — exactly what someone watching a long run wants to see.

Two related functions need **no** change, but should be verified rather than assumed:

- `_nonfinite_loss_columns` (`analysis.py` line 262) filters on `"loss" in column`, so
  the new columns cannot trigger a false divergence report.
- `_looks_collapsed` (`analysis.py` line 278) only reads `train/loss`.

## 5. Verification

**Unit tests** — new `engine/tests/test_cost.py`, following `test_config_and_cli.py`
conventions (module docstring, `REPO_ROOT` constants, a `_train_args(argv)` helper
through `train_cli.build_parser()`), and `tests/helpers.py` for the CPU-only `nano`
backbone with `pretrained_weights=null`:

1. A short CPU `fit` records `len(durations) == batches - warmup_steps` for train, and
   separate non-empty accumulators for val and test.
2. `warmup_steps` larger than the batch count yields zero measured batches and a `null`
   mean rather than a `ZeroDivisionError`.
3. Sanity-check batches are excluded (run with `num_sanity_val_steps=2`).
4. `run_summary.json` is written into `tmp_path`, parses, and carries the expected
   `cost.train` / `cost.val` / `cost.test` keys plus `final_metrics`.
5. On CPU, memory fields are `null` and the run does not crash.
6. `cost.enabled=false` appends no callback — assert against `trainer.callbacks`,
   matching `test_no_checkpoint_callback_unless_asked` (`test_config_and_cli.py` lines 158-175).

**Agentic test** — extend the existing `agentic/tests/` job-analysis coverage: a
`metrics.csv` containing `cost/*` columns yields `final_metrics` without them, while
`read_progress` still surfaces them.

Default `pytest` runs stay CPU-only and fast (`addopts = "-ra -m 'not gpu and not
weights and not data'"`, `pyproject.toml` lines 58-68).

**End-to-end smoke, CPU:**

```bash
cd engine && python -m rinalmo_hub.cli.train \
  --task mrl --use_lora --set pretrained_weights=null --set lm_config=nano \
  --set trainer.accelerator=cpu --set trainer.devices=1 \
  --set trainer.precision=32-true --set trainer.max_steps=30 \
  --set cost.warmup_steps=5 --output_dir /tmp/cost_smoke
```

Check `run_summary.json` exists and that `metrics/version_0/metrics.csv` has the new
`cost/train/iter_time_ms` column.

**Real GPU check:** run one short real training job and confirm
`cost.train.mem_allocated_gb_peak` is in the same ballpark as `nvidia-smi` during the run
(expect `nvidia-smi` to read somewhat higher — it shows reserved, not allocated), and
that `iter_time_ms_mean` roughly matches the it/s the tqdm bar displayed.

## 6. Documents to update

| Document | Section | Change |
|---|---|---|
| [engine/README.md](../engine/README.md) | §3 "What a run writes" | Add `run_summary.json` to the output listing, with the JSON shape |
| | §3 "Hyperparameters" | Document `cost.*` — this is the reference `configuration.md` defers to for what each key means and what happens when it is wrong |
| | §2 "The pieces that matter" | Add `cost.py` |
| [configuration.md](../documents/configuration.md) | §2 key-groups table | New `cost.*` row alongside `trainer.*` |
| | §9 "Run output directory" | Add `run_summary.json` to the tree; note `cost/*` columns are sparse like every other column |
| [workflows/finetuning.md](../documents/workflows/finetuning.md) | Step 4 output tree | Add `run_summary.json` |
| | Step 5 "analyse" | Note that cost figures are excluded from analysed metrics but visible in job status |
| [project_structure.md](../documents/project_structure.md) | `rinalmo_hub/` tree | Add the `cost.py` line |
| | Module responsibility table | New `rinalmo_hub/cost.py` row |
| [modules/engine-hub.md](../documents/modules/engine-hub.md) | §1 "The pieces", §8 `cli/train.py` | New subsection for `cost.py`: the pair-based timing rationale and the `sync_dist` warning |
| [testing.md](../documents/testing.md) | Engine test inventory | New `test_cost.py` row |

[profiling-and-knowledge.md](../documents/modules/profiling-and-knowledge.md) is **not**
touched — the naming decision in §1 exists precisely so that document stays correct and
unambiguous.

## 7. Out of scope

- Reserved-memory figures (`max_memory_reserved`) — decided against.
- Cost numbers inside adapter metadata — decided against; the JSON is the record.
- Profiling `evaluate.py` / `predict.py`. The test stage is already covered because
  `train.py` calls `trainer.test()` itself. Trivial to extend later, as
  `build_cost_profiler` takes only `cfg`.
- Surfacing cost figures in the web UI beyond what `read_progress` already feeds.
- Any cross-run aggregation script for building the paper table. Worth a follow-up once
  several `run_summary.json` files exist.
