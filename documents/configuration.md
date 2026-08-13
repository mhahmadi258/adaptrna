# Configuration and Data Formats

Everything configurable, and the exact shape of everything written to disk.

---

## Contents

1. [Environment variables](#1-environment-variables)
2. [Engine config layering](#2-engine-config-layering)
3. [The `Config` object and `--set` parsing](#3-the-config-object-and---set-parsing)
4. [The knowledge base](#4-the-knowledge-base)
5. [`toolhub_data/tools.json` — the manifest](#5-toolhub_datatoolsjson--the-manifest)
6. [`jobs_data/jobs.json` — the job store](#6-jobs_datajobsjson--the-job-store)
7. [The adapter file format](#7-the-adapter-file-format)
8. [The training plan](#8-the-training-plan)
9. [Run output directory](#9-run-output-directory)
10. [Format versions and compatibility](#10-format-versions-and-compatibility)

---

## 1. Environment variables

None are required. Every one has a documented default.

| Variable | Default | Read by | Effect |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | — | [`settings.py`](../agentic/adaptrna_agentic/settings.py) | **Required for any LLM path.** Checked at model construction, not import. Loaded from `<repo>/.env` if not already in the environment. |
| `ADAPTRNA_MODEL` | `anthropic:claude-opus-5` | `settings.py` | Model spec for **all** roles |
| `ADAPTRNA_MODEL_ORCHESTRATOR` | ← global | `settings.py` | Per-role override; wins over `ADAPTRNA_MODEL` |
| `ADAPTRNA_MODEL_TOOLSMITH` | ← global | `settings.py` | Per-role override |
| `ADAPTRNA_MODEL_VERIFIER` | ← global | `settings.py` | Per-role override |
| `ADAPTRNA_MAX_TOKENS` | `8192` | `settings.py` | `max_tokens` passed to every chat model |
| `ADAPTRNA_TOOLHUB_DIR` | `<repo>/toolhub_data` | [`toolhub/manifest.py`](../agentic/adaptrna_agentic/toolhub/manifest.py) | Manifest, adapters, staging. `--data-dir` overrides it again. |
| `ADAPTRNA_JOBS_DIR` | `<repo>/jobs_data` | [`jobs/store.py`](../agentic/adaptrna_agentic/jobs/store.py) | Job records |
| `ADAPTRNA_CHAT_DIR` | `<repo>/chat_data` | [`cli/chat.py`](../agentic/adaptrna_agentic/cli/chat.py) | `sessions.sqlite` — shared by terminal and HTTP |
| `ADAPTRNA_API_TOKEN` | — | [`api/deps.py`](../agentic/adaptrna_agentic/api/deps.py) | Bearer token. **Required** to bind anywhere but loopback; `cli/serve.py` refuses otherwise. |

Precedence for models: `ADAPTRNA_MODEL_<ROLE>` → `ADAPTRNA_MODEL` → `DEFAULT_MODEL`. The
spec string is provider-prefixed and resolved by LangChain's `init_chat_model`; that string
is the **entire** provider abstraction, so switching providers is a config edit, never a
code change. No module imports `langchain_anthropic` directly.

The three roles are fixed: `ROLES = ("orchestrator", "toolsmith", "verifier")`. Everything
else in the platform is a deterministic service and never talks to a model.

Set inside the sandbox for generated code, not by you
([`codegen/sandbox.py`](../agentic/adaptrna_agentic/codegen/sandbox.py)):
`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`, `TOKENIZERS_PARALLELISM=false`,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` — the thread caps alone cut ~5 GiB of virtual
address space that 128 OpenMP threads would reserve for no benefit on a nano model.

## 2. Engine config layering

Three layers, each overriding the last:

```
engine/configs/base.yaml  →  engine/configs/tasks/<task>.yaml  →  --set dotted.key=value
```

Then, in [`cli/common.py::resolve_run_config`](../engine/rinalmo_hub/cli/common.py), CLI
flags are folded in on top (`--task`, `--lm_config`, `--pretrained_weights`, `--seed`,
`--use_lora`, `--prepare_data`), and finally — if `--adapter` was given — **the adapter file
overrides everything it knows about**: `task`, `lm_config`, `head`, `lora`, `use_lora`. The
file is the authority on what is needed to rebuild the trained model.

Every run writes the fully merged result to `<output_dir>/resolved_config.yaml`. That file,
not your shell history, is the record of what was trained.

### The key groups in `base.yaml`

| Group | Keys | Notes |
|---|---|---|
| top level | `task`, `lm_config`, `pretrained_weights`, `seed`, `use_lora`, `head`, `task_config` | `head` is free-form kwargs for `build_head()` and is stored in the adapter file; `task_config` is task knobs that are not head kwargs |
| `data.*` | `batch_size`, `num_workers`, `pin_memory`, `prepare` universally; everything else is task-defined | `data.root`, `data.ss_type`, `data.val_split`, `data.dataset` … |
| `lora.*` | `r` 16, `alpha` 32, `dropout` 0.05, `layer_stride` 3, `layers` null, `target_modules` | Only used when `use_lora`. `layer_stride 3` adapts 11 of `giga`'s 33 blocks |
| `optim.*` | `name` adam, `lr` 1e-4, `weight_decay` null, `scheduler.{name,interval,…}` | Unknown scheduler keys are **rejected**, not ignored |
| `trainer.*` | `accelerator`/`devices` auto, `max_epochs`/`max_steps` -1, `precision` bf16-mixed, `gradient_clip_val` null, `log_every_n_steps` 50, `progress_refresh_rate` 50, `checkpoint_every_epoch` false | |
| `finetune.*` | `schedule` null, `initial_denom_lr` 1.0 | Full FT only; mutually exclusive with `--use_lora` (the CLI refuses the combination) |

`initial_denom_lr: 1.0` is a deliberate departure from Lightning's default of `10.0`, which
silently trains newly unfrozen parameters an order of magnitude below the configured rate.

The engine README is the complete reference for what each key means and what happens when
it is wrong: [`../engine/README.md`](../engine/README.md#hyperparameters).

## 3. The `Config` object and `--set` parsing

[`rinalmo_hub/config.py`](../engine/rinalmo_hub/config.py) provides a nested `dict`
subclass supporting both access styles, so `cfg["optim"]["lr"]` and `cfg.optim.lr` are the
same thing. A missing key raises `AttributeError` listing what *is* available.

`parse_scalar` deliberately does **not** use `yaml.safe_load` alone:

> YAML 1.1 parses `3e-4` as the *string* `"3e-4"` (it wants `3.0e-4`), and
> `--set optim.lr=3e-4` is the single most common override there is.

Order tried: `null|none|~|""` → `None`; `true`/`false` → bool; `int`; `float`; then YAML for
lists and anything else; falling back to the raw string.

## 4. The knowledge base

Two YAML files under
[`agentic/adaptrna_agentic/knowledge/`](../agentic/adaptrna_agentic/knowledge/), merged and
`lru_cache`d by `load_knowledge()`. **Every hyperparameter the recommender proposes and
every rationale line it shows come from here**, which is what keeps the explanation and the
executed config from drifting apart.

### `hyperparameters.yaml`

```yaml
arms:
  lora:                    # optim.lr 3e-4 (adamw), trainer.gradient_clip_val 1.0,
                           # lora r16/alpha32/dropout0.05/layer_stride3
    why: [...]             # rendered verbatim into the plan's rationale
    failure_modes:         # each: {setting, symptom, remedy}
      - lr 1e-3 → collapses into constant output after a gradient spike at ~step 325
      - gradient_clip_val null → a single spike collapses the adapters irrecoverably
  full_ft:                 # optim.lr 1e-5 (adamw), clip 1.0, plus artifact_note
    failure_modes:
      - lr 1e-4 on an unfrozen backbone → destroys the backbone, MRL R² ≈ 0

universal:
  trainer: {precision: bf16-mixed}
  why: ["fp16's range is a real risk at these learning rates"]
  nondeterminism: {note, consequence}     # FlashAttention backward; F1 95.21 vs 95.82

tasks:
  <task>:
    primary_metric        # e.g. test/f1_score
    higher_is_better
    metric_scale          # percent | unit
    reference: {band: [lo, hi] | null, tolerance, sources[], tolerance_reason}
    wall_clock: {reference, measured}
    defaults: {trainer: {max_epochs: N}}
    caveats: [...]        # surfaced as plan warnings
```

Shipped entries: `splice_site` (band 95.8–97.5 F1, ±1.0, ~7 min), `mrl` (band 0.81–0.83 R²,
±0.02, ~5h45m), `sec_struct` (band `null` — no in-repo reference run yet).

A task with no entry gets `generic_task_knowledge()`: the *arm* settings still apply
(validated across tasks and they transfer), but the reference band is explicitly `null` and
a caveat says so — the first run on a new task is a **baseline**, not something to be judged
against somebody else's number.

### `task_templates.yaml`

One entry per shipped task shape: `match` (target type, typical length) → `shape` (head,
loss, metrics, `extract_features` pattern, prediction type) → `data_layout` (description,
`required_paths` used for exact layout matching, `key_options` used to validate
task-specific choices such as `data.ss_type: [donor, acceptor]`).

Consumed by the profiler (to name the matching or nearest task), the recommender (to
validate `task_options`) and the ToolSmith prompt (as "the closest known task shape").

`no_match_guidance` is the text shown when nothing matches — see
[README.md gap #3](README.md#known-documentation-gaps), it is stale.

`tests/test_knowledge.py` fails loudly if an edit drops one of the load-bearing numbers,
because the recommender has no other source for them.

## 5. `toolhub_data/tools.json` — the manifest

`format_version: 1`. Written atomically (`mkstemp` → `os.replace`) with an in-file
`revision` counter that must match what was read, or `ConcurrentModificationError` is
raised and nothing is written.

```jsonc
{
  "format_version": 1,
  "revision": 4,                       // monotonic; NOT an mtime — two identical writes
                                       // inside one filesystem tick are indistinguishable
  "backbone": {
    "lm_config": "giga",               // nano | micro | mega | giga
    "weights": "/abs/path/giga-v1.pt", // or repo-relative, or null (random backbone)
    "device": "auto",                  // auto → cuda if available else cpu
    "dtype":  "auto"                   // auto → None → the model default (fp32)
  },
  "tools": { "<name>": { /* ToolEntry, minus its name (that is the key) */ } }
}
```

### `ToolEntry`

| Field | Type | Adapter tools | External tools |
|---|---|---|---|
| `type` | `"adapter"` \| `"external"` | ✅ | ✅ |
| `state` | `"active"` \| `"disabled"` | ✅ | ✅ |
| `description` | str — shown to the user **and to the model** | ✅ | from the `FunctionSpec` |
| `task` | engine task name | ✅ | `null` |
| `lm_config` | backbone the adapter was trained on | ✅ | `null` |
| `artifact` | path, repo-relative or absolute | ✅ | `null` |
| `serving` | `{"batch_size": int\|null}` | ✅ (`null` = the task's own default) | `{}` |
| `test` | `{"sequences": [...], "expected": null}` | ✅ | `{"golden": [{args, expect}]}` |
| `provenance` | `{source, registered_at, adapter_metadata}` (+ `job_id`, `training_metrics` when registered from a run) | ✅ | `{source, registered_at, family, family_description}` |
| `external` | `{module, function, package: {pip, import_name, installed_version}}` | `null` | ✅ |

Golden cases are **copied into** the manifest entry at registration, so an entry stays
self-contained even if the wrapper module's `SPEC` changes later. `expect` values compare
exactly, unless written as `{"approx": x, "tol": t}`.

Registration rules enforced by [`Registry.register`](../agentic/adaptrna_agentic/toolhub/registry.py):
duplicate names refused; full-FT exports refused (only the head travels in such a file);
`lm_config` must match the hub's; `mrl` is forced to `serving.batch_size = 1` because its
head is pad-sensitive. The artifact is copied to `<name>.pt.incoming`, the manifest is
written, and only then is the copy moved into place — so a failure between the two leaves
neither an orphan file nor an entry pointing at nothing.

## 6. `jobs_data/jobs.json` — the job store

`format_version: 1`, same atomic-write and revision discipline.

```jsonc
{
  "format_version": 1,
  "revision": 179,
  "jobs": {
    "splice_simple_lora_20260813_101810": {   // id == the output directory's basename
      "task": "splice_simple",
      "arm": "lora",
      "command": ["/…/python", "-m", "adaptrna_agentic.jobs.train_entrypoint", "--task", …],
      "output_dir": "/abs/path/outputs/splice_simple_lora_20260813_101810",
      "state": "running",           // running | succeeded | failed | cancelled
      "pid": 20866,
      "pid_starttime": "2149151089", // /proc/<pid>/stat field 22 — a PID alone is NOT
                                     // an identity; the kernel recycles them
      "started_at": "2026-08-13T00:18:30+00:00",
      "ended_at":   "2026-08-13T00:21:14+00:00",
      "exit_code": 0,
      "adapter_path": "/…/splice_simple_adapter.pt",
      "plan": { /* the full plan the run came from */ }
    }
  }
}
```

Storing the whole `plan` is what lets a run stay analysable later — including its
`primary_metric` — even if its task can no longer be imported because it was deleted,
renamed or moved.

State is always **derived from disk** on read (`_refresh`): the `exit_code` file the
entrypoint writes wins; otherwise a dead-or-recycled PID means `failed`. Terminal states
are never revisited.

## 7. The adapter file format

`format_version: 2`, a `torch.save`d dict —
[`rinalmo_hub/adapter.py`](../engine/rinalmo_hub/adapter.py):

```python
{
  "format_version": 2,
  "task":        "mrl",      # so the hub can dispatch with no config file
  "lm_config":   "giga",     # guard: refuse to load onto a different backbone size
  "lora":        {...}|None, # None ⇒ head-only / full-FT export
  "head_config": {...},      # kwargs to rebuild the head with no YAML present
  "state_dict":  {...},      # lora_* + head.* + ADAPTER_EXTRA_PREFIXES tensors ONLY
  "extra":       {...},      # NON-tensor state (a tuned threshold, a class map)
  "metadata":    {created, git_sha, train_metrics, seed, arm},
}
```

`task` and `head_config` are what make `RiNALMoHub.register(path)` possible with nothing
but the path. Inspect any file with `python -m rinalmo_hub.adapter <path>`.

**The two silent-failure channels are `ADAPTER_EXTRA_PREFIXES` (tensors/buffers) and
`adapter_extra_payload()`/`load_adapter_extra()` (plain Python values).** State that
predictions depend on and that travels through neither will load without error and produce
plausible-looking wrong numbers. Harness check 6 exists to catch exactly this.

## 8. The training plan

Produced only by
[`recommender.recommend()`](../agentic/adaptrna_agentic/profiling/recommender.py); consumed
by the approval gate, the JobRunner and the analyzer.

| Key | Meaning |
|---|---|
| `source` | Always `"recommend_training_config"`. **`start_training` refuses any plan without it.** |
| `task`, `arm` | `lora` (default, the only arm that yields a servable tool) or `full_ft` |
| `config_path` | `adaptrna_custom/tasks/<t>/config.yaml` if it exists, else `engine/configs/tasks/<t>.yaml` |
| `overrides` | Flat dotted map → `--set key=value` pairs |
| `seed`, `output_dir`, `quick_run` | `outputs/<task>[_<option>…]_<arm>_<YYYYmmdd_HHMMSS>` |
| `primary_metric`, `reference` | Copied from the knowledge base so the analyzer needs nothing else |
| `estimated_wall_clock` | From `wall_clock.reference`; prefixed for a quick run |
| `rationale` | Generated from knowledge-base entries — the model narrates these, it does not write them |
| `warnings` | Arm notes, task caveats, quick-run truncation, missing backbone |
| `command` | **The exact argv** the JobRunner will execute, shown verbatim in the gate |

`quick_run` sets `trainer.max_steps=200` and `data.num_workers=8`, and its warning is
load-bearing: a truncated run is a smoke test and the analyzer will refuse to compare it to
reference metrics.

## 9. Run output directory

```
outputs/<run_name>/
├── resolved_config.yaml           the exact merged config (written before training starts)
├── metrics/version_N/
│   ├── metrics.csv                appended as the run goes — `tail -f` it
│   └── hparams.yaml
├── train.log                      stdout+stderr of the detached process
├── exit_code                      written by train_entrypoint in a `finally` block
└── <task>_adapter.pt              LoRA runs (or <task>_full.pt with --save_full_weights)
```

`metrics.csv` rows are **sparse** — each row carries only the metrics logged at that moment,
so every reader takes the last non-null value per column
(`runner.read_progress`, `analysis._final_metrics`). Because pandas reads both an empty cell
and a literal `"nan"` as NaN, divergence detection re-reads the file as text
(`analysis._nonfinite_loss_columns`) to tell "diverged" apart from "not logged here".

`latest_metrics_file()` picks the highest `version_N` that actually contains a `metrics.csv`.

## 10. Format versions and compatibility

| Artifact | Version | On mismatch |
|---|---|---|
| `tools.json` | 1 | `ValueError` naming both versions; refuses to read |
| `jobs.json` | 1 | same |
| adapter `.pt` | 2 (`SUPPORTED_FORMAT_VERSIONS = (2,)`) | `ValueError`; also validates required fields and `lm_config` |

`ToolEntry` gained its external-tool fields additively in Phase 3, with defaults, so v1
manifests written before that load unchanged — the pattern to follow for future additions.
