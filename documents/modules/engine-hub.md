# `rinalmo_hub/` — the fine-tuning framework

`engine/rinalmo_hub/`

One pretrained backbone, swappable LoRA adapters and task heads, one CLI. **New tasks need
no framework changes**: a task is a registered subclass plus a config plus a datamodule, and
nothing in this package is edited to add one.

This document covers structure and the contracts. The engine's own
[README](../../engine/README.md) is the complete hyperparameter reference and the
task-authoring walkthrough — it is not duplicated here.

---

## Contents

1. [The pieces](#1-the-pieces)
2. [`registry.py`](#2-registrypy)
3. [`module.py` — `BaseDownstreamModule`](#3-modulepy--basedownstreammodule)
4. [`lora.py`](#4-lorapy)
5. [`adapter.py`](#5-adapterpy)
6. [`config.py`](#6-configpy)
7. [`hub.py` — `RiNALMoHub`](#7-hubpy--rinalmohub)
8. [`cli/`](#8-cli)
9. [`tasks/` — the three shipped tasks](#9-tasks--the-three-shipped-tasks)
10. [`data/mrl.py`](#10-datamrlpy)
11. [Assumptions and limitations](#11-assumptions-and-limitations)

---

## 1. The pieces

| File | Lines | Responsibility |
|---|---:|---|
| `registry.py` | 46 | Name → task class. `@register_task` is the whole extension mechanism. |
| `module.py` | 359 | `BaseDownstreamModule` — everything task-independent |
| `lora.py` | 383 | Injection, freezing, multi-adapter residency, adapter-key ownership |
| `adapter.py` | 183 | The adapter file format v2: save / load / describe |
| `config.py` | 139 | Three-layer config resolution and the `Config` object |
| `hub.py` | 206 | `RiNALMoHub`: N adapters in one backbone |
| `cost.py` | ~215 | `CostProfiler`: per-stage GPU memory + iteration time, `run_summary.json` |
| `cli/common.py` | 217 | The construction order, trainer assembly, config→CLI wiring |
| `cli/{train,evaluate,predict}.py` | 108/68/107 | The three entry points |
| `tasks/*.py` | 103/146/193 | splice_site · mrl · sec_struct |
| `data/mrl.py` | ~90 | `MRLDataModule` with a selectable validation split |

Naming is fixed framework-wide: the backbone is always `self.backbone`, the head is always
`self.head`. That is what makes the `ft_schedules/*.yaml` regexes interchangeable between
tasks — the source project called the backbone `self.rinalmo` in two scripts and `self.lm`
in four.

## 2. `registry.py`

```python
@register_task("my_task")          # sets cls.TASK_NAME and records the class
class MyTaskModule(BaseDownstreamModule): ...

get_task(name)        # KeyError listing available_tasks()
available_tasks()     # sorted names
is_registered(name)
```

Re-registering the same name with a *different* class raises, naming the module that already
owns it. The decorator only fires when the module is imported — which is why
`tasks/__init__.py` imports all three shipped tasks, and why the agentic layer's
[`discovery.load_all()`](codegen.md#6-discoverypy) exists for generated ones.

## 3. `module.py` — `BaseDownstreamModule`

`LoRAAdapterMixin + pl.LightningModule`. Handles backbone construction, LoRA injection and
freezing, adapter I/O, checkpoint slimming, optimizer and scheduler, and the generic
train/val/test steps.

```python
BaseDownstreamModule(lm_config="giga", head_config=None, lora=None, optim=None,
                     finetune=None, task_config=None, attach_backbone=True)
```

`attach_backbone=False` is how `RiNALMoHub` makes N tasks cost one backbone: the module is
built without one, and the hub assigns its shared instance afterwards. The `backbone`
attribute is registered as a child module either way (`add_module("backbone", … or None)`).

### The subclass contract

| Hook | Required | Purpose |
|---|:---:|---|
| `build_head(embed_dim, **head_config)` | ✅ | The prediction head. `embed_dim` comes from the base class; everything else from the YAML's `head:` block. Convention: raise `TypeError` on unexpected keys so config drift is loud. |
| `extract_features(representation, tokens)` | ✅ | Backbone output → what the head consumes. A returned **tuple is splatted** into the head as positional args. |
| `compute_loss(outputs, batch)` | ✅ | Scalar loss |
| `update_metrics(outputs, batch, stage)` | ✅ | Accumulate |
| `compute_metrics(stage) -> dict` | ✅ | Reduce and log at epoch end |
| `build_datamodule(cfg)` | ✅ | `@staticmethod` |
| `build_metrics(stage)` | | Per-stage metric object, exposed as `self.metrics[stage]` |
| `batch_tokens(batch)` | | Where the token tensor is; default `batch[0]` |
| `postprocess_predictions(outputs, tokens, sequences)` | | Raw head output → the task's native prediction type. **Used by the hub.** |
| `adapter_extra_payload()` / `load_adapter_extra(extra)` | | Non-tensor state to carry in the adapter file |
| `on_fit_start_hook()` | | Setup needing the datamodule, e.g. fitting a target scaler |

Class attributes: `TASK_NAME` (set by the decorator), `ADAPTER_EXTRA_PREFIXES` (extra
**tensor** prefixes), `PRIMARY_METRIC`, `DEFAULT_PREDICT_BATCH_SIZE` (default 8).

`extract_features` is an explicit hook rather than a convention precisely because getting it
wrong is silent: splice site wants the CLS token, MRL wants padded positions zeroed **plus**
the pad mask, secondary structure wants CLS and EOS dropped.

### The generic step

```python
forward(tokens):  representation = backbone(tokens)["representation"]
                  features = extract_features(representation, tokens)
                  return head(*features) if isinstance(features, tuple) else head(features)

_step(batch, stage): log f"{stage}/loss"; update_metrics(...)
on_{train,val,test}_epoch_end: compute_metrics + log_dict + reset
```

Sanity-check epochs are skipped for metric logging (`trainer.sanity_checking`), which
otherwise pollutes the first row of `metrics.csv`.

### Optimizer and scheduler

```python
params = [p for p in self.parameters() if p.requires_grad]
```

In LoRA mode only the adapters and head have `requires_grad` — handing the optimizer the
frozen backbone as well is what makes "LoRA" runs quietly cost full-FT memory. Optimizers:
`adam`, `adamw`, `sgd`. Schedulers: `linear`, `cosine`, `constant`, `none`, and **unknown
per-scheduler keys raise** rather than being silently ignored.

### Adapter I/O

```python
save_adapter(path, metadata=None)   # metadata defaults to build_metadata(); "arm" is stamped
load_adapter(path)                  # validates task name AND LoRA geometry before loading
```

The geometry check is worth knowing about: a mismatch raises with the fix — *"Build the
model from the adapter's own config (the CLI does this automatically when you pass
`--adapter`)."*

## 4. `lora.py`

### `LoRASpec`

```python
r=16, alpha=32, dropout=0.05, layer_stride=3, layers=None,
target_modules=["mh_attn.Wqkv", "mh_attn.out_proj"]
```

`from_dict` **rejects unknown keys** listing the known ones. `layer_indices(num_blocks)`
returns explicit `layers` if given (validated against the block count), else every
`layer_stride`-th block from 0 — stride 3 picks 0, 3, …, 30 out of `giga`'s 33.

The default targets are the fused QKV projection and the attention output projection. The
flash-attention path uses a **single fused `Wqkv`**, so one adapter covers q, k and v; there
are no separate q/k/v Linears to target. Both are bias-free, hence peft's `bias="none"`.

### `inject_lora(backbone, spec, adapter_name, verbose)`

Uses `inject_adapter_in_model`, **not** `get_peft_model`: the latter would wrap the backbone
and rename every state-dict key to `base_model.model.*`, breaking checkpoint compatibility
with plain RiNALMo weights.

Two failure modes are handled explicitly:

* peft only complains when **nothing** matched. A *partial* match is the dangerous one — it
  trains a handful of blocks plus the head and looks exactly like "LoRA doesn't work" — so
  the adapted-module count is checked against the requested count and a mismatch raises with
  the target list.
* Injecting a second adapter into the same backbone is the whole point of the hub, so peft's
  "already found a `peft_config` attribute" warning is filtered.

**Order is load-bearing**: injection moves each target's weight to
`<target>.base_layer.weight`, so the pretrained load must happen *first*, and an adapter
file's keys are the post-injection ones, so it must be loaded *last*.

### Multi-adapter support

```python
set_active_adapter(backbone, name)   # every tuner layer that carries `name`
resident_adapters(backbone)
adapted_module_names(backbone, name)
disable_adapters(backbone)           # contextmanager — the bare pretrained backbone
freeze_backbone_except_lora(model)   # requires_grad = "lora_" in name
```

peft keeps a per-adapter dict inside each `lora.Linear` and its forward **skips any active
adapter a layer does not hold** — layers adapted by a different task's geometry are left as
plain base layers. That is what lets adapters with different `layer_stride` values coexist in
one backbone.

### Adapter key ownership

peft bakes the adapter name into every LoRA state-dict key
(`<target>.lora_A.<adapter_name>.weight`). Training always writes `"default"`; the hub
injects under the **tool name** so several adapters can be resident, so keys are remapped on
the way in:

```python
adapter_names_in_state_dict(sd)          # which names appear
rename_adapter_in_state_dict(sd, new)    # rewrite lora_*.<old>.* → <new>
is_adapter_key(key, extra_prefixes, adapter_name)
```

`is_adapter_key` taking an `adapter_name` matters as soon as a backbone holds more than one
adapter — the normal case inside `RiNALMoHub`. Without it, one task's `adapter_state_dict()`
would sweep up every other resident task's LoRA weights.

### `LoRAAdapterMixin`

`apply_lora()`, `adapter_state_dict()`, `expected_adapter_keys()`,
`load_adapter_state_dict()`, plus two Lightning integration points:

```python
strict_loading -> not use_lora        # LoRA checkpoints carry no backbone keys
on_save_checkpoint(ckpt)              # slim to adapter keys: 7.8 GB → 18 MB per file
```

`load_adapter_state_dict` checks the key sets **in both directions**. `strict=False` is what
makes missing backbone keys acceptable, but it would just as happily accept a stale file and
leave the head randomly initialised — so neither `expected - got` nor `got - expected` may be
non-empty. A geometry mismatch is named as the usual cause.

## 5. `adapter.py`

Format v2; schema in [../configuration.md §7](../configuration.md#7-the-adapter-file-format).

```python
save_adapter(path, *, task, lm_config, lora, head_config, state_dict, extra, metadata)
load_adapter(path, lm_config=None)     # validates version, required fields, backbone size
describe_adapter(path) -> str
build_metadata(train_metrics=None, seed=None, **extra)   # created, git_sha, train_metrics, seed
git_sha(repo_root=None)                # short SHA, or None outside a checkout
```

`task` and `head_config` are the additions over the v1 format the source project used, and
both are needed so `RiNALMoHub.register(path)` can rebuild a task from the file alone.
Passing `lm_config` to `load_adapter` turns a silent shape mismatch deep inside
`load_state_dict` into a clear message.

```bash
python -m rinalmo_hub.adapter outputs/my_run/splice_site_adapter.pt
```

prints size, version, task, backbone size, head config, LoRA geometry, tensor and parameter
counts split by kind, any non-tensor extras, and the run metadata.

## 6. `config.py`

```python
resolve_config(task_config_path=None, overrides=None, base_config_path=BASE_CONFIG_PATH) -> Config
save_config(cfg, path)
```

Layering, the `Config` object and the `parse_scalar` rationale are documented in
[../configuration.md §§2–3](../configuration.md#2-engine-config-layering). The short version:
`base.yaml` → task YAML → `--set` dotted overrides, and `--set optim.lr=3e-4` works because
numeric parsing is tried before YAML (YAML 1.1 reads `3e-4` as a *string*).

`REPO_ROOT` here is `engine/`, so `BASE_CONFIG_PATH = engine/configs/base.yaml`.

## 7. `hub.py` — `RiNALMoHub`

```python
hub = RiNALMoHub(backbone_weights="weights/giga-v1.pt", lm_config="giga",
                 device="cuda", dtype=torch.bfloat16)
hub.register("adapters/task_a.pt")        # the task name is read from the file
name = hub.register(path, name="donor")   # or register under an explicit name
hub.available(); hub.module(name); hub.source(name)
hub.predict(name, ["ACGU…"], batch_size=None)
with hub.no_adapter(): ...                # base-model baseline
hub.embed(["ACGU…"], batch_size=8)        # raw backbone representations
```

`register()` in order: load and validate the file → **refuse full-FT exports** → build the
task module with `attach_backbone=False` → assign the shared backbone → set
`module.lora_adapter_name = name` → inject LoRA under that name → rename the state dict's
adapter keys → load tensors and extras → move to device/dtype, `eval()`, `requires_grad_(False)`.

`predict` calls `activate(task)` first, which flips the active adapter across the **whole**
backbone — the reason [`AdapterRuntime`](toolhub.md#4-runtimepy--one-backbone-all-adapters)
serialises inference. Batching uses `type(module).DEFAULT_PREDICT_BATCH_SIZE` unless
overridden, and `_concat` handles both tensor outputs and list outputs (e.g. sec-struct's
per-sequence matrices).

The full-FT refusal message is explicit about why: *"only its head travelled with the file,
and the backbone it was trained with is not this one… Load it with
`rinalmo_hub.cli.evaluate --init_params` instead."*

## 8. `cli/`

### `common.py` — the shared plumbing

| Function | Purpose |
|---|---|
| `add_common_arguments(parser)` | `--task --config --set --use_lora --pretrained_weights --adapter --init_params --lm_config --seed --output_dir` |
| `resolve_run_config(args)` | Layer configs, fold in flags, then let `--adapter` override `task`/`lm_config`/`head`/`lora`/`use_lora`. Refuses an unknown or missing task, listing what exists. |
| `build_module(cfg, task, args)` | **The construction order**, below |
| `build_trainer(cfg, output_dir, extra_callbacks)` | Loggers, callbacks, strategy |
| `prepare_run(cfg, output_dir)` | `seed_everything(workers=True)` + write `resolved_config.yaml` |

```python
# build_module — this order is load-bearing
module = task_cls(lm_config, head_config, lora, optim, finetune, task_config)
if weights: module.load_pretrained_backbone(weights)   # 1. backbone first
if module.use_lora: module.apply_lora()                # 2. then inject
if args.adapter: module.load_adapter(args.adapter)     # 3. then the adapter
if args.init_params: module.load_state_dict(...)       # (full-FT export)
```

`build_trainer` decisions worth knowing:

* **`TQDMProgressBar`, not Lightning's rich bar** — the rich bar detects a non-TTY and
  buffers, so a redirected log stays empty for the whole run. An 8-hour job once looked
  frozen at 3855 bytes. tqdm flushes on every refresh.
* **`CSVLogger`** writes `metrics/version_N/metrics.csv` as the run goes. That is the file
  the agentic layer's job monitoring and analysis both read.
* **Checkpointing is off by default** — a full-FT `giga` checkpoint is 7.8 GB per epoch.
  Note that Lightning installs a *default* `ModelCheckpoint` unless `enable_checkpointing`
  is explicitly false, so opting out has to be deliberate.
* **DDP gets `find_unused_parameters=True`** whenever the run is multi-device, because
  `RiNALMo.forward` always runs its masked-LM head whose output never reaches the loss.
  `'auto'` is resolved here too, or a multi-GPU box crashes on default arguments.
* **`finetune.schedule` + `--use_lora` is refused**: a schedule would unfreeze the backbone
  LoRA exists to keep frozen.

### `train.py`

`fit` → save artifacts (rank zero only) → `test`, unless `--no_test`/`--test_only`.

`_save_artifacts` encodes the LoRA-only principle at the source: a LoRA run writes
`<output_dir>/<task>_adapter.pt`; a full-FT run writes **nothing** unless
`--save_full_weights` (~2.6 GB), because *a full-FT run's adapter file would carry only the
head, and the backbone it was trained with would not travel with it*. Asking for
`--adapter_out` on a full-FT run does produce the file, with a printed warning.

After `test` (or `test_only`), `write_run_summary` (rank zero only) writes
`<output_dir>/run_summary.json`: final task metrics plus, when `cost.enabled`, per-stage
(train/val/test) mean/median/p90 iteration time and mean/peak GPU memory from `cost.py`'s
`CostProfiler`, built by `build_cost_profiler(cfg)` and threaded through `build_trainer`'s
existing `extra_callbacks` parameter — no change to shared trainer plumbing was needed.

### `cost.py` — `CostProfiler`

A `pl.Callback` timing every `training_step`/`validation_step`/`test_step` and sampling
`torch.cuda.memory_allocated` alongside it, kept in a per-stage accumulator.

Two decisions worth knowing if you touch this file:

* **Timing is pair-based** — `on_*_batch_start` to `on_*_batch_end` of the *same* batch —
  rather than the delta between consecutive `batch_end` calls. Lightning interleaves the
  validation loop inside the training epoch loop, so a delta measurement would record the
  first train batch after every validation pass as one enormous iteration. Pairing each
  batch's own start and end keeps train/val/test independent regardless of what the other
  stages are doing.
* **Per-step logging never sets `sync_dist=True`**, unlike every `self.log()` call in
  `module.py`. `sync_dist` inserts an all-reduce per logged value per step, which would
  directly inflate the very iteration time being measured.

The first `cost.warmup_steps` batches of each stage are excluded from the *mean* timing and
memory figures (cuDNN autotuning, allocator growth, lazy CUDA init), but always count toward
the peak-memory figure. `cost.sync_cuda` (default on) calls `torch.cuda.synchronize()` before
each timing stamp — CUDA kernels are asynchronous, so without it the numbers measure kernel
*enqueue*, not completion.

### `evaluate.py`

Requires `--adapter` or `--init_params`. `--split test|validate`, `--metrics_out FILE`.
Writes `resolved_config_eval.yaml` so an evaluation never overwrites the training record.

### `predict.py`

Builds a `RiNALMoHub`, registers every `--adapter` given (repeatable), and runs every
registered task — or just `--task`-listed ones — over `--sequences`/`--input`, writing JSON.
`--dtype` accepts `float32|bfloat16|float16`; FlashAttention needs half precision on CUDA.

## 9. `tasks/` — the three shipped tasks

| Task | Shape | Primary metric | Notable |
|---|---|---|---|
| `splice_site` | Binary classification from the CLS token; `SpliceSitePredictionHead` | `test/f1_score` | Metrics accumulate as a `BinaryConfusionMatrix` and are reduced through `rinalmo.utils.splice_site_metrics` **as percentages**, so numbers stay comparable with the published F1 of 95.82. Train batches are not scored. |
| `mrl` | Regression; pooled `RibosomeLoadingPredictionHead` over a padded sequence | `test/r2` | `ADAPTER_EXTRA_PREFIXES = ("scaler.",)`. The scaler is fitted in `on_fit_start_hook` from the training targets, read straight off the dataframe when possible. Predictions are un-scaled and clamped at 0. |
| `sec_struct` | Per-pair binary classification over the upper triangle; `SecStructPredictionHead` (outer concat + 2D ResNet) | `test/f1` | Owns a tuned `threshold` — a plain Python float re-fitted on the validation set every `tune_threshold_every_n_epoch` epochs — carried through `adapter_extra_payload()`. `DEFAULT_PREDICT_BATCH_SIZE = 1` because the head is O(L²) and structures are variable length. |

These three are the canonical illustrations of the **two silent-failure questions**:

1. *Does the task own state predictions depend on that is not a head weight?* MRL answers
   yes with a **tensor** (`ADAPTER_EXTRA_PREFIXES`); sec-struct answers yes with a **plain
   value** (`adapter_extra_payload`); splice-site answers no.
2. *Does the head need CLS/EOS/padded positions excluded?* splice-site takes
   `representation[:, 0]`; MRL zeroes padded positions and passes the mask; sec-struct drops
   CLS and EOS with `representation[..., 1:-1, :]`.

Two implementation notes carried over deliberately:

* MRL's scaler is fitted in `on_fit_start`, not inside `training_step`. The source fitted it
  during epoch 0 and returned `None`, which made epoch 0 a non-training epoch (shifting every
  fine-tuning schedule index by one) and made the task impossible to run under DDP.
* sec-struct's threshold travels through the typed non-tensor channel; the source
  hand-stashed it into `checkpoint['state_dict']['threshold']` and then did an unguarded
  lookup that `KeyError`ed on any dict lacking the key.

## 10. `data/mrl.py`

`MRLDataModule(RibosomeLoadingDataModule)` adds `val_split ∈ {random7600, holdout}` plus
`holdout_fraction` / `holdout_seed`.

The vendored datamodule uses `Random7600` as the validation set **and reports it as a
result**. Nothing currently selects on validation (the final epoch is tested; there is no
monitored checkpoint), so the published numbers are clean as shipped — but adding early
stopping or best-checkpoint selection on `val/r2` would silently contaminate a headline
metric. `holdout` carves a validation set out of the training sequences instead.

This is the concrete case behind the master plan's warning about metric-based checkpoint
selection.

## 11. Assumptions and limitations

* **No metric-based checkpoint selection.** The final-epoch model is what gets tested;
  validation informs but does not select.
* **`flash_attn` is imported lazily.** Without it, CUDA training falls back to a much slower
  plain-PyTorch attention path. Its **backward pass is non-deterministic**, so two runs of
  the same command with the same seed differ; the forward pass is deterministic, so
  evaluating a fixed artifact reproduces its metrics exactly.
* **Gradient checkpointing is unconditionally on**, and `need_attn_weights=True` forces the
  slow attention path.
* **Backbone sizes are not interchangeable.** An adapter records its `lm_config` and loading
  it onto a different size is refused.
* **`config.py`'s `REPO_ROOT` is `engine/`**, while the agentic layer's `REPO_ROOT` is the
  repository root. Both names, two meanings — check which module you are in.
