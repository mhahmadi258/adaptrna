# RiNALMo-Hub

A task-pluggable fine-tuning framework for the RiNALMo RNA language model. One pretrained
backbone, swappable LoRA adapters and task heads, one CLI.

---

## Contents

1. [Overview](#1-overview)
2. [Project structure](#2-project-structure)
3. [Training, validation and testing](#3-training-validation-and-testing)
4. [Adding a new task](#4-adding-a-new-task)
5. [The shipped tasks](#5-the-shipped-tasks)
6. [Pretrained weights](#6-pretrained-weights)

---

## 1. Overview

RiNALMo is a pretrained RNA language model. Using it for a downstream task normally means
writing a training script per task, each with its own head, its own data loading, its own
freezing logic and its own checkpoint format. This project replaces that with one framework:

- **One backbone, many tasks.** The RiNALMo transformer is loaded once. Each task adds a
  prediction head and, optionally, LoRA adapters.
- **Two training arms from the same command.** *Full fine-tuning* updates the backbone;
  *LoRA* freezes it and trains small low-rank adapters plus the head. The only difference
  between the two arms on the command line is the `--use_lora` flag and the learning rate —
  everything else lives in the task's YAML so it cannot drift between runs.
- **Small, self-describing artifacts.** A LoRA run writes a single adapter file containing
  the adapter weights, the head weights, the head config, the LoRA geometry and any extra
  task state. It is a few megabytes instead of the multi-gigabyte backbone, and it carries
  enough metadata to rebuild the trained model without its YAML.
- **Several tasks served from one loaded backbone.** `RiNALMoHub` keeps multiple adapters
  resident in one backbone and switches between them by name, so serving N tasks costs one
  backbone in memory instead of N.
- **New tasks need no framework changes.** A task is a registered subclass plus a config plus
  a datamodule. Nothing in the core package is edited to add one.

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./engine
pip install flash-attn --no-build-isolation    # CUDA only; match your torch build
pip install -e "./engine[dev]"                        # pytest, for the test suite
```

`flash_attn` is imported lazily. Without it the package still imports and the CPU test suite
still runs, but training on CUDA falls back to a much slower plain-PyTorch attention path,
so install it before any real run.

Verify the install with the default test suite, which is CPU-only and needs neither weights
nor datasets:

```bash
cd engine && pytest
```

---

## 2. Project structure

```
engine/                 everything below lives here; runtime artifacts (weights/, dataset/,
                        outputs/) stay at the repo root, and commands run from the repo root
 
rinalmo_hub/            the framework
├── registry.py         @register_task / get_task / available_tasks
├── module.py           BaseDownstreamModule: everything task-independent
├── lora.py             LoRA injection, freezing, multi-adapter support
├── adapter.py          the adapter file format: save / load / inspect
├── config.py           base.yaml -> task YAML -> --set overrides
├── hub.py              RiNALMoHub: multi-adapter inference
├── cost.py             CostProfiler: per-stage iteration time and GPU memory
├── tasks/              one module per task, each registered
├── data/               datamodules this project adds
└── cli/                train.py, evaluate.py, predict.py

rinalmo/                the vendored RiNALMo backbone: model, alphabet, data, downstream heads
configs/                base.yaml + tasks/<task>.yaml
ft_schedules/           gradual-unfreezing schedules for full fine-tuning
examples/               a complete worked example of adding a task
scripts/                benchmarking utilities
tests/                  CPU test suite plus opt-in suites needing weights or data
```

### The pieces that matter

| file | responsibility |
|---|---|
| `rinalmo_hub/module.py` | `BaseDownstreamModule` — backbone construction, LoRA injection and freezing, adapter I/O, optimizer and scheduler, the train/val/test steps. A task subclass implements only what is genuinely task-specific. |
| `rinalmo_hub/registry.py` | `@register_task("name")` makes a task visible to the CLI and the hub. |
| `rinalmo_hub/config.py` | Layered configuration and the `Config` object, which supports both `cfg["optim"]["lr"]` and `cfg.optim.lr`. |
| `rinalmo_hub/lora.py` | Where LoRA is injected into the backbone, what stays frozen, and how several adapters coexist in one backbone. |
| `rinalmo_hub/adapter.py` | The adapter file: a versioned dict holding the task name, backbone size, LoRA geometry, head config, tensors, non-tensor extras and run metadata. |
| `rinalmo_hub/hub.py` | `RiNALMoHub` — register adapters, list them, switch, predict, embed. |
| `rinalmo_hub/cost.py` | `CostProfiler` — per-stage (train/val/test) mean/peak GPU memory and mean iteration time, written to `run_summary.json` alongside the final task metrics. |
| `rinalmo_hub/cli/common.py` | Shared CLI plumbing, including the load-bearing construction order: build module → load backbone → inject LoRA → load adapter. |

Naming is fixed across the whole framework: the backbone is always `self.backbone`, the head
is always `self.head`. That is what makes the `ft_schedules/*.yaml` regexes interchangeable
between tasks.

### Configuration layering

Every run resolves its config in three stages, each overriding the last:

```
configs/base.yaml  ->  configs/tasks/<task>.yaml  ->  --set dotted.key=value
```

`configs/base.yaml` holds defaults for every key. The task YAML overrides what the task needs.
`--set` is for the things that change per run — the learning rate, the seed, a data split.
Every run writes the fully resolved config to `<output_dir>/resolved_config.yaml`; that file,
not your shell history, is the record of what was trained.

---

## 3. Training, validation and testing

### Three commands

```bash
# train (and test at the end)
python -m rinalmo_hub.cli.train    --task <task> --config engine/configs/tasks/<task>.yaml [--use_lora] ...

# evaluate a trained artifact on the test or validation split
python -m rinalmo_hub.cli.evaluate --adapter <adapter.pt> --config engine/configs/tasks/<task>.yaml

# run raw sequences through one or more adapters
python -m rinalmo_hub.cli.predict  --adapter <adapter.pt> --sequences ACGU... 
```

### Training

```bash
python -m rinalmo_hub.cli.train \
    --task <task> \
    --config engine/configs/tasks/<task>.yaml \
    --use_lora \
    --set optim.lr=3e-4 \
    --seed 42 \
    --output_dir outputs/<run_name>
```

Common flags, all defined in `rinalmo_hub/cli/common.py` and `cli/train.py`:

| flag | meaning |
|---|---|
| `--task` | Which registered task to run. Omit `--config` and the matching `configs/tasks/<task>.yaml` is picked up automatically. |
| `--config` | Task YAML, layered on top of `configs/base.yaml`. |
| `--set KEY=VALUE` | Dotted config override, repeatable: `--set optim.lr=3e-4 --set trainer.max_epochs=10`. |
| `--use_lora` | Train LoRA adapters + head with the backbone frozen. Without it, the run is full fine-tuning. |
| `--pretrained_weights` | Path to the backbone checkpoint. Defaults to `weights/giga-v1.pt`. |
| `--lm_config` | Backbone size: `nano`, `micro`, `mega`, `giga`. |
| `--seed` | Seeds Lightning, dataloader workers included. |
| `--output_dir` | Where logs, metrics, the resolved config and the artifact go. |
| `--prepare_data` | Download and lay out the dataset before training. Needed once per task. |
| `--adapter` | Resume from / start from an existing adapter. The file supplies the task, backbone size, LoRA geometry and head config. |
| `--init_params` | Start from a full state dict (a full fine-tuning export). |
| `--adapter_out` | Where to write the trained adapter. Defaults to `<output_dir>/<task>_adapter.pt`. |
| `--save_full_weights` | Full-FT runs: write the whole state dict to `<output_dir>/<task>_full.pt`. Off by default, because that file is the size of the backbone. |
| `--no_test` | Skip the test pass that otherwise follows training. |
| `--test_only` | Skip training; test the loaded weights. |

Validation runs inside `fit` on the schedule Lightning applies by default (once per epoch),
using the task's validation split. Testing runs automatically after training unless you pass
`--no_test`. Note that no metric-based checkpoint selection is configured: the model that gets
tested is the one at the end of training, so validation metrics inform you but do not select.

### What a run writes

| path | contents |
|---|---|
| `<output_dir>/resolved_config.yaml` | the exact, fully merged config the run used |
| `<output_dir>/metrics/version_N/metrics.csv` | every logged metric, written as the run goes — `tail -f` it |
| `<output_dir>/run_summary.json` | final task metrics plus per-stage training cost (see below) |
| `<output_dir>/<task>_adapter.pt` | the LoRA artifact (LoRA runs) |
| `<output_dir>/<task>_full.pt` | the full state dict (full-FT runs, with `--save_full_weights`) |
| stdout | Lightning's test table, then the same summary that lands in `run_summary.json` |

A `CSVLogger`, a `TQDMProgressBar`, a `LearningRateMonitor` and a `CostProfiler` are attached
to every run. tqdm rather than Lightning's default rich bar, because the rich bar buffers when
stdout is not a TTY and a redirected log stays empty for the whole run.

`run_summary.json` exists for paper-table experiments, where the quality metric alone is not
the whole story:

```json
{
  "task": "mrl", "arm": "lora", "seed": 42, "git_sha": "a1b2c3d",
  "final_metrics": { "test/r2": 0.81 },
  "cost": {
    "device": "NVIDIA A100-SXM4-40GB", "world_size": 1, "warmup_steps": 10,
    "train": {
      "batches": 5000, "measured_batches": 4990,
      "iter_time_ms_mean": 412.7, "iter_time_ms_median": 410.2, "iter_time_ms_p90": 428.9,
      "wall_seconds": 2110.4, "wall_ms_per_batch": 422.1,
      "mem_allocated_gb_mean": 18.3, "mem_allocated_gb_peak": 21.7
    },
    "val": { "...": "same shape" },
    "test": { "...": "same shape" }
  }
}
```

A stage with zero measured batches (a `max_steps` run cut off before the first validation
pass, say) is simply absent from `cost`, rather than present with null figures. See
`cost.*` below for the knobs, and `rinalmo_hub/cost.py` for how the numbers are measured.

Per-epoch checkpointing is off by default (`trainer.checkpoint_every_epoch`), since a full-FT
checkpoint is the size of the backbone times a few. Turn it on deliberately if you need it.

### Validating and testing a trained artifact

```bash
python -m rinalmo_hub.cli.evaluate \
    --adapter outputs/<run_name>/<task>_adapter.pt \
    --pretrained_weights weights/giga-v1.pt \
    --config engine/configs/tasks/<task>.yaml \
    --split test \
    --metrics_out outputs/<run_name>/test_metrics.json
```

The adapter file carries the task, the backbone size, the LoRA geometry and the head config,
so `--task`, `--use_lora` and every `lora.*` key are read from it. `--config` is still needed,
for the data paths. `--split validate` scores the validation split instead. For a full
fine-tuning export, pass `--init_params <task>_full.pt` in place of `--adapter`.

### Predicting on raw sequences

```bash
python -m rinalmo_hub.cli.predict \
    --pretrained_weights weights/giga-v1.pt --device cuda --dtype bfloat16 \
    --adapter outputs/run_a/<task_a>_adapter.pt \
    --adapter outputs/run_b/<task_b>_adapter.pt \
    --sequences ACGUACGUACGU GCGCAUAUGCGC \
    --output predictions.json
```

`--input seqs.txt` reads one sequence per line, skipping `>` and `#` lines. `--task` restricts
which registered adapters run. In Python the same thing is `RiNALMoHub`:

```python
import torch
from rinalmo_hub.hub import RiNALMoHub

hub = RiNALMoHub(backbone_weights="weights/giga-v1.pt", lm_config="giga",
                 device="cuda", dtype=torch.bfloat16)

hub.register("adapters/task_a.pt")        # the task name is read from the file
hub.register("adapters/task_b.pt")
hub.available()                           # registered task names

hub.predict("task_a", ["ACGUACGU...", ...])

with hub.no_adapter():                    # base-model baseline
    hub.predict("task_a", ["ACGUACGU...", ...])

hub.embed(["ACGUACGU...", ...])           # raw backbone representations
```

Adapters with different geometries coexist in one backbone, and switching between them is a
dictionary lookup rather than a reload. Registering a *full fine-tuning* export is refused:
only its head travels in the file, so serving it would silently pair a fine-tuned head with a
pretrained backbone. Evaluate those with `evaluate --init_params`.

### Inspecting an adapter

```bash
python -m rinalmo_hub.adapter <path>/<task>_adapter.pt
```

prints the format version, task, backbone size, head config, LoRA geometry, tensor and
parameter counts, any non-tensor extras and the run metadata (creation time, git SHA, seed,
arm, final train metrics).

### Hyperparameters

Everything below is a config key. Set it in the task YAML for anything that should be part of
the task's definition, and with `--set` for anything that varies per run.

#### Model and run

| key | meaning |
|---|---|
| `lm_config` | Backbone size: `nano`, `micro`, `mega`, `giga`. `nano` is what the CPU tests use; real runs use `giga`. |
| `pretrained_weights` | Path to the backbone checkpoint. `null` starts from a random backbone. |
| `use_lora` | Set by `--use_lora`. Chooses the arm. |
| `seed` | Seeds Lightning and the dataloader workers. |
| `head` | Free-form kwargs forwarded to the task's `build_head()`, and stored in the adapter file. |
| `task_config` | Task-specific knobs that are not head kwargs. |

#### LoRA (`lora.*`, used only when `use_lora` is on)

| key | default | meaning |
|---|---|---|
| `lora.r` | `16` | Rank of the low-rank update. Higher means more trainable parameters and a larger adapter file. |
| `lora.alpha` | `32` | LoRA scaling; the effective scale is `alpha / r`. |
| `lora.dropout` | `0.05` | Dropout on the LoRA branch. |
| `lora.layer_stride` | `3` | Adapt every n-th transformer block. `1` adapts every block. |
| `lora.layers` | `null` | Explicit block indices, e.g. `[0, 8, 16, 24]`. Overrides `layer_stride`. |
| `lora.target_modules` | attention projections | Which submodules inside a block get adapters. The default targets the fused QKV projection and the attention output projection; the fused projection means one adapter covers Q, K and V jointly. |

Rank and stride are the two knobs that trade adapter size against capacity. Stride is the
coarser of the two: it decides *how many* blocks are adapted at all.

#### Optimization (`optim.*`)

| key | default | meaning |
|---|---|---|
| `optim.name` | `adam` | `adam`, `adamw` or `sgd`. |
| `optim.lr` | `1e-4` | Learning rate. **This is the setting that differs most between the two arms** — see below. |
| `optim.weight_decay` | `null` | Passed through when set. |
| `optim.scheduler.name` | `none` | `none`, `linear`, `cosine` or `constant`. |
| `optim.scheduler.interval` | `step` | `step` or `epoch`. |
| `optim.scheduler.*` | | Per-scheduler arguments: `start_factor`, `end_factor`, `total_iters` for `linear`; `T_max`, `eta_min` for `cosine`; `factor`, `total_iters` for `constant`. Unknown keys are rejected rather than ignored. |

**Learning rate by arm.** LoRA trains a small number of fresh parameters and tolerates — needs —
a learning rate one to two orders of magnitude higher than full fine-tuning, which is updating
weights that already encode everything the pretrained model knows. Full fine-tuning at a
LoRA-sized learning rate destroys the pretrained backbone; LoRA at a full-FT-sized rate barely
moves. Each shipped task YAML carries the full-FT default, and its LoRA arm is documented with
the task in §5. Keep `trainer.gradient_clip_val` set for LoRA runs: a single gradient spike can
otherwise collapse the adapters into a constant-output state they never recover from.

#### Trainer (`trainer.*`)

| key | default | meaning |
|---|---|---|
| `trainer.max_epochs` / `trainer.max_steps` | `-1` | Training budget. `-1` means unbounded on that axis. `--set trainer.max_steps=0` downloads data without training. |
| `trainer.precision` | `bf16-mixed` | Prefer bf16 over fp16 here: fp16's range is a real risk at these learning rates. |
| `trainer.gradient_clip_val` | `null` | Global gradient-norm clip. |
| `trainer.accelerator` / `trainer.devices` | `auto` | `--set trainer.devices=2` for multi-GPU. DDP is configured with `find_unused_parameters=True` automatically, because the backbone always runs a masked-LM head whose output never reaches the loss. |
| `trainer.log_every_n_steps` | `50` | Logging cadence. |
| `trainer.progress_refresh_rate` | `50` | Progress-bar refresh. |
| `trainer.checkpoint_every_epoch` | `false` | Per-epoch `ModelCheckpoint`. Expensive for full FT. |

#### Gradual unfreezing (`finetune.*`, full fine-tuning only)

| key | default | meaning |
|---|---|---|
| `finetune.schedule` | `null` | Path to an `ft_schedules/*.yaml`. Each top-level key is an epoch index; its list holds regexes of parameter groups to unfreeze at that epoch. Everything not yet unfrozen is frozen. |
| `finetune.initial_denom_lr` | `1.0` | Newly unfrozen parameters join at `lr / initial_denom_lr`. Lightning's own default is `10.0`, which silently trains them an order of magnitude below the configured learning rate. |

`finetune.schedule` and `--use_lora` are mutually exclusive, and the CLI refuses the
combination: a schedule would unfreeze the backbone that LoRA exists to keep frozen.

The shipped schedules are `ft_schedules/head_only.yaml` (head only, forever) and per-task
schedules that train the head for the first few epochs and then unfreeze the backbone — either
all at once or a few blocks at a time from the output end downwards. A warm-up phase like this
is the usual alternative to simply lowering the full-FT learning rate.

#### Data (`data.*`)

`data.batch_size`, `data.num_workers`, `data.pin_memory` and `data.prepare` are universal.
`data.root` and everything else under `data.*` is task-defined; see §5.

#### Training cost (`cost.*`)

| key | default | meaning |
|---|---|---|
| `cost.enabled` | `true` | Attaches a `CostProfiler` callback that tracks iteration time and GPU memory per stage (train/val/test) and writes `run_summary.json`. Set `false` to skip it entirely. |
| `cost.warmup_steps` | `10` | Batches excluded from the *mean* timing and memory figures at the start of each stage — cuDNN autotuning, allocator growth and lazy CUDA init otherwise skew a short run badly. Peak memory still covers the whole stage, warm-up included. |
| `cost.log_per_step` | `true` | Also emit `cost/<stage>/iter_time_ms` and `cost/<stage>/mem_allocated_gb` into `metrics.csv`, at the same `trainer.log_every_n_steps` cadence as everything else. |
| `cost.sync_cuda` | `true` | Calls `torch.cuda.synchronize()` before each timing stamp. CUDA kernels are asynchronous, so without this the numbers measure kernel *enqueue*, not completion. Turning it off trades accuracy for a small speedup. |

This must live under top-level `cost:`, not `trainer:` — `build_trainer()` splats unrecognised
`trainer.*` keys straight into `pl.Trainer(**kwargs)`, so an unknown key there is a hard
`TypeError` from Lightning.

### A note on reproducibility

FlashAttention's backward pass is non-deterministic, so two runs of the same command with the
same seed do not produce identical weights. The forward pass *is* deterministic, so evaluating
a fixed artifact reproduces its metrics exactly. Treat small differences between two training
runs as noise, and compare arms across multiple seeds or multiple data splits rather than from
a single pair of runs.

---

## 4. Adding a new task

A task is three files and no edits to any core file. `examples/ncrna_classification/` is a
complete worked example — an ncRNA family classifier — and `tests/test_new_task_acceptance.py`
asserts that the abstraction holds, including that no core file so much as mentions the task.

### Step 1 — the task module

Create `engine/rinalmo_hub/tasks/my_task.py`:

```python
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


@register_task("my_task")
class MyTaskModule(BaseDownstreamModule):
    TASK_NAME = "my_task"

    # Extra *tensor* state to ship inside the adapter file. Leave empty unless the task owns
    # buffers (a target scaler, say) that predictions depend on.
    ADAPTER_EXTRA_PREFIXES = ()

    # Which metric the task is judged on.
    PRIMARY_METRIC = "test/acc"

    def build_head(self, embed_dim, hidden_dim: int = 128, **kwargs):
        # `embed_dim` is the backbone width, supplied by the base class. Everything else
        # comes from the YAML's `head:` block.
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")

        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def build_metrics(self, stage):            # optional; exposed as self.metrics[stage]
        return None if stage == "train" else BinaryAccuracy()

    def extract_features(self, representation, tokens):
        # Decide what the head consumes. CLS token here.
        return representation[:, 0]

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return nn.functional.binary_cross_entropy_with_logits(outputs, labels.unsqueeze(1))

    def update_metrics(self, outputs, batch, stage):
        if stage in self.metrics:
            _, labels = batch
            self.metrics[stage].update(torch.sigmoid(outputs), labels.unsqueeze(1).int())

    def compute_metrics(self, stage):
        return {f"{stage}/acc": self.metrics[stage].compute()} if stage in self.metrics else {}

    @staticmethod
    def build_datamodule(cfg):
        from rinalmo_hub.data.my_task import MyTaskDataModule

        return MyTaskDataModule(
            data_root=cfg.data.root,
            alphabet=Alphabet(),
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )
```

Then add one line to `engine/rinalmo_hub/tasks/__init__.py`, so the decorator fires on import:

```python
from . import my_task  # noqa: F401
```

### Step 2 — the config

Create `engine/configs/tasks/my_task.yaml`. Anything not set here is inherited from
`configs/base.yaml`:

```yaml
task: my_task
lm_config: giga

head: {hidden_dim: 128}

data: {root: dataset/my_task, batch_size: 32, num_workers: 8, pin_memory: true}

lora: {r: 16, alpha: 32, dropout: 0.05, layer_stride: 3}

optim: {name: adamw, lr: 1.0e-5}          # the full-FT default; the LoRA arm uses a higher lr

trainer: {max_epochs: 10, precision: bf16-mixed, gradient_clip_val: 1.0}
```

### Step 3 — the datamodule

A standard `lightning.pytorch.LightningDataModule` returning `(tokens, target)` batches.
`examples/ncrna_classification/datamodule.py` is a template. Tokenise with
`Alphabet().encode(seq, pad_to_len=...)` or `Alphabet().batch_tokenize(seqs)`. If your batches
are not `(tokens, target)`, override `batch_tokens(batch)` to say where the token tensor is.

### Then it just works

```bash
python -m rinalmo_hub.cli.train --task my_task --config engine/configs/tasks/my_task.yaml --use_lora \
    --prepare_data --output_dir outputs/my_task_lora
```

so do `evaluate`, `predict`, adapter save/load and hub registration.

### The subclass contract

| hook | required | purpose |
|---|---|---|
| `build_head(embed_dim, **head_config)` | ✅ | the prediction head |
| `extract_features(representation, tokens)` | ✅ | pooling / slicing; a returned tuple is splatted into the head as positional arguments |
| `compute_loss(outputs, batch)` | ✅ | scalar loss |
| `update_metrics(outputs, batch, stage)` | ✅ | accumulate |
| `compute_metrics(stage) -> dict` | ✅ | reduce and log at epoch end |
| `build_datamodule(cfg)` | ✅ | `@staticmethod` |
| `build_metrics(stage)` | | per-stage metric object, exposed as `self.metrics[stage]` |
| `batch_tokens(batch)` | | where the token tensor is in a batch; default `batch[0]` |
| `postprocess_predictions(outputs, tokens, sequences)` | | raw head output → the task's native prediction type, used by the hub |
| `adapter_extra_payload()` / `load_adapter_extra(extra)` | | non-tensor state to carry in the adapter file |
| `on_fit_start_hook()` | | setup that needs the datamodule, e.g. fitting a target scaler |

Class attributes worth setting: `ADAPTER_EXTRA_PREFIXES` (extra tensor prefixes for the
adapter file), `PRIMARY_METRIC` (the metric the task is judged on) and
`DEFAULT_PREDICT_BATCH_SIZE` (lower it for heads that are quadratic in sequence length).

### Two questions to ask before you ship a task

Both of these fail *silently* — the adapter loads without error and the numbers look plausible.

1. **Does the task own state that predictions depend on but that isn't a head weight?**
   A tensor or buffer → add its prefix to `ADAPTER_EXTRA_PREFIXES`. A plain Python value (a
   tuned threshold, a class mapping) → implement `adapter_extra_payload()` and
   `load_adapter_extra()`. Omit either and the state silently reverts to its default.
2. **Does the head need CLS, EOS or padded positions excluded?** `extract_features` is the only
   place that happens, which is why it is an explicit hook rather than a convention.

---

## 5. The shipped tasks

Three tasks ship with the framework, plus one worked example under `examples/`.

| task | kind | dataset | metrics logged | primary |
|---|---|---|---|---|
| `splice_site` | binary sequence classification | Spliceator (via the SpliceBERT release) | acc, precision, recall, specificity, f1_score | `test/f1_score` |
| `mrl` | regression | synthetic human 5'UTR library | r2, mse, mae | `test/r2` |
| `sec_struct` | per-pair binary classification | bpRNA (SPOT-RNA split) or ArchiveII family folds | precision, recall, f1 | `test/f1` |
| `ncrna_classification` *(example)* | multi-class classification | ncRNA families | acc, f1 | `test/f1` |

### Getting the data

Each dataset downloads itself. Pass `--prepare_data` once; add `--set trainer.max_steps=0` to
download without training:

```bash
python -m rinalmo_hub.cli.train --task splice_site --prepare_data --set trainer.max_steps=0
python -m rinalmo_hub.cli.train --task mrl         --prepare_data --set trainer.max_steps=0
python -m rinalmo_hub.cli.train --task sec_struct  --prepare_data --set trainer.max_steps=0
```

They land under `dataset/`, at the paths the task YAMLs point `data.root` at:

```
dataset/
├── train_data/     splice site: the training/validation folds
├── test_data/      splice site: the benchmark species
├── mrl_data/       MRL: the 5'UTR library CSV
└── sec_struct/     secondary structure: bpRNA and/or ArchiveII splits
weights/
└── giga-v1.pt      the backbone
```

---

### `splice_site` — splice site prediction

Binary classification of whether a sequence contains a donor or acceptor splice site, read off
the CLS token. This is a **cross-species benchmark**: training and validation come from the
Spliceator training set, and the test set is a different organism entirely.

| config key | options | meaning |
|---|---|---|
| `data.ss_type` | `donor`, `acceptor` | which splice site type to train on |
| `data.dataset_id` | `db_1` … `db_10` | which of the ten stratified train/validation folds |
| `data.species` | `Danio`, `Fly`, `Thaliana`, `Worm` | which species the test set comes from |
| `data.root` / `data.test_root` | | training folds / benchmark species |
| `head.head_embed_dim` | | width of the classification head |

Splits: `data.root` provides a real train/validation split (the fold), and `data.test_root`
provides the held-out species.

```bash
# LoRA
python -m rinalmo_hub.cli.train --task splice_site --config engine/configs/tasks/splice_site.yaml \
    --use_lora --set optim.lr=3e-4 --seed 42 \
    --output_dir outputs/splice_donor_lora

# full fine-tuning (the YAML's own lr; --save_full_weights to keep the result)
python -m rinalmo_hub.cli.train --task splice_site --config engine/configs/tasks/splice_site.yaml \
    --seed 42 --save_full_weights \
    --output_dir outputs/splice_donor_full_ft

# the acceptor arm, on a different fold and a different benchmark species
python -m rinalmo_hub.cli.train --task splice_site --config engine/configs/tasks/splice_site.yaml \
    --use_lora --set data.ss_type=acceptor --set data.dataset_id=db_2 --set data.species=Fly \
    --output_dir outputs/splice_acceptor_fly_lora

# test a trained adapter
python -m rinalmo_hub.cli.evaluate --adapter outputs/splice_donor_lora/splice_site_adapter.pt \
    --config engine/configs/tasks/splice_site.yaml --split test \
    --metrics_out outputs/splice_donor_lora/test_metrics.json
```

The ten folds are what makes this task the cheapest place to measure run-to-run variance:
sweep `data.dataset_id` and compare distributions rather than single numbers.

---

### `mrl` — mean ribosome load

Regression on a synthetic 5'UTR library. Targets are standardised by a `StandardScaler` fitted
on the training targets at `on_fit_start`, and predictions are un-scaled back before metrics
are computed. **The fitted mean and std travel inside the adapter file**
(`ADAPTER_EXTRA_PREFIXES = ("scaler.",)`) — without them the adapter would still load and would
return plausible-looking numbers on the wrong scale.

| config key | options | meaning |
|---|---|---|
| `data.val_split` | `random7600`, `holdout` | which validation set to use |
| `data.holdout_fraction` / `data.holdout_seed` | | size and seed of the holdout split |
| `head.head_embed_dim`, `head.num_blocks`, `head.dropout` | | the convolutional regression head |

**About `data.val_split`.** The default, `random7600`, is the dataset's own evaluation split —
it is also a *reported result*, so using it as the validation set means validating on a test
set. Nothing in the framework currently selects on validation metrics (the final epoch is what
gets tested; there is no monitored checkpoint), so this is safe as shipped and preserves
comparability with the published setup. But if you add early stopping or best-checkpoint
selection, switch to `data.val_split=holdout`, which carves a validation set out of the
training sequences and leaves the evaluation splits untouched.

```bash
# LoRA
python -m rinalmo_hub.cli.train --task mrl --config engine/configs/tasks/mrl.yaml \
    --use_lora --set optim.lr=3e-4 --seed 42 \
    --output_dir outputs/mrl_lora

# full fine-tuning with the head-only warm-up schedule
python -m rinalmo_hub.cli.train --task mrl --config engine/configs/tasks/mrl.yaml \
    --set finetune.schedule=engine/ft_schedules/mrl_paper.yaml --seed 42 --save_full_weights \
    --output_dir outputs/mrl_full_ft

# a clean validation split instead of the reported one
python -m rinalmo_hub.cli.train --task mrl --config engine/configs/tasks/mrl.yaml \
    --use_lora --set data.val_split=holdout --output_dir outputs/mrl_lora_holdout

# test a trained adapter
python -m rinalmo_hub.cli.evaluate --adapter outputs/mrl_lora/mrl_adapter.pt \
    --config engine/configs/tasks/mrl.yaml --split test \
    --metrics_out outputs/mrl_lora/test_metrics.json
```

Two schedules ship for the full-FT arm: `ft_schedules/mrl_paper.yaml` (head only, then the
entire backbone) and `ft_schedules/mrl.yaml` (head only, then the upper blocks and the final
layer norm, leaving the lower blocks and the embedding frozen throughout).

---

### `sec_struct` — RNA secondary structure

Predicts the base-pairing matrix: per-pair binary classification over the upper triangle. The
head is quadratic in sequence length and structures are variable length, so the datamodule
feeds **one structure at a time** — there is no batch size to set.

This task owns a tuned `threshold`: a plain Python float re-tuned on the validation set every
`task_config.tune_threshold_every_n_epoch` epochs, and used to binarise probabilities into a
structure. It travels in the adapter file through the non-tensor channel. This is the one
shipped task whose validation set is genuinely load-bearing.

| config key | options | meaning |
|---|---|---|
| `data.dataset` | `bpRNA`, `archiveII_<family>` | which benchmark. The ArchiveII variants are inter-family folds: `archiveII_5s`, `_16s`, `_23s`, `_grp1`, `_srp`, `_telomerase`, `_RNaseP`, `_tmRNA`, `_tRNA`. |
| `data.min_seq_len` / `data.max_seq_len` | | length filter, useful for capping memory |
| `task_config.tune_threshold_every_n_epoch` | | how often the threshold is re-tuned |
| `task_config.save_test_ct_files` | | write one `.ct` per test structure into the output dir |
| `head.num_resnet_blocks`, `head.conv_dim`, `head.kernel_size` | | the ResNet pairing head |

```bash
# LoRA on bpRNA
python -m rinalmo_hub.cli.train --task sec_struct --config engine/configs/tasks/sec_struct.yaml \
    --use_lora --set optim.lr=3e-4 --seed 42 \
    --output_dir outputs/sec_struct_lora

# full fine-tuning with the gradual-unfreezing schedule
python -m rinalmo_hub.cli.train --task sec_struct --config engine/configs/tasks/sec_struct.yaml \
    --set finetune.schedule=engine/ft_schedules/sec_struct.yaml --seed 42 --save_full_weights \
    --output_dir outputs/sec_struct_full_ft

# an ArchiveII inter-family fold, with a length cap
python -m rinalmo_hub.cli.train --task sec_struct --config engine/configs/tasks/sec_struct.yaml \
    --use_lora --set data.dataset=archiveII_tRNA --set data.max_seq_len=500 \
    --output_dir outputs/sec_struct_trna_lora

# test, writing .ct files for the predicted structures
python -m rinalmo_hub.cli.evaluate --adapter outputs/sec_struct_lora/sec_struct_adapter.pt \
    --config engine/configs/tasks/sec_struct.yaml --split test \
    --set task_config.save_test_ct_files=true --output_dir outputs/sec_struct_lora \
    --metrics_out outputs/sec_struct_lora/test_metrics.json
```

`ft_schedules/sec_struct.yaml` trains the head first, then unfreezes three transformer blocks
every three epochs from the output end downwards.

---

### `ncrna_classification` — the worked example

Multi-class ncRNA family classification, living in `examples/ncrna_classification/` rather than
in `rinalmo_hub/tasks/`, precisely to demonstrate that a task needs no framework changes. It is
registered by `@register_task`, which fires when its module is imported — so to run it from the
CLI, make it part of the import graph the way step 1 of §4 describes. Add one line to
`rinalmo_hub/tasks/__init__.py`:

```python
from examples.ncrna_classification import task  # noqa: F401
```

and then it is an ordinary task:

```bash
python -m rinalmo_hub.cli.train --task ncrna_classification \
    --config engine/examples/ncrna_classification/config.yaml --use_lora \
    --prepare_data --output_dir outputs/ncrna_lora
```

(For a real task, move the module into `rinalmo_hub/tasks/` instead and import it from there.
The example is left outside the package on purpose, so `tests/test_new_task_acceptance.py` can
assert that no core file mentions it.)

---

## 6. Pretrained weights

### The RiNALMo backbone

Every task needs the pretrained backbone. `giga-v1` is the full-size model and is what all the
shipped configs assume. Download it once:

```bash
python -c "from rinalmo.pretrained import get_pretrained_model; get_pretrained_model('giga-v1')"
mkdir -p weights && ln -s ~/.cache/rinalmo_pretrained/giga-v1.pt weights/giga-v1.pt
```

That caches the file in `~/.cache/rinalmo_pretrained/`. The direct link, if you would rather
download it by hand:

**`giga-v1.pt`** — https://drive.google.com/file/d/1-E2Ziu2VFDAgwCmQvVeAviGtsQQ94L3L/view

Every config defaults to `pretrained_weights: weights/giga-v1.pt`. Override it per run with
`--pretrained_weights /path/to/giga-v1.pt` or `--set pretrained_weights=...`, and set it to
`null` to start from a randomly initialised backbone (useful for smoke tests).

### Trained task adapters

<!-- Paste the download URL for the trained LoRA adapters here. -->

**Trained adapters:** _(link to be added)_

Drop the downloaded files anywhere and point the CLI at them — the adapter file names its own
task, so nothing else needs configuring:

```bash
python -m rinalmo_hub.cli.evaluate --adapter adapters/<task>_adapter.pt \
    --config engine/configs/tasks/<task>.yaml --split test
```
