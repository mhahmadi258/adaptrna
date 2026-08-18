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
11. [The `DatasetSpec` schema](#11-the-datasetspec-schema)

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

### Browser-side storage

Not configuration you set, but state the web client keeps per browser:

| Key | Store | Holds |
|---|---|---|
| `adaptrna.token` | `sessionStorage` | The bearer token, when the server requires one. **`sessionStorage` on purpose** — it dies with the tab rather than outliving it on disk. |
| `adaptrna.rail.width` | `localStorage` | Session rail width, in `rem` (clamped 10–30) |
| `adaptrna.rail.collapsed` | `localStorage` | Whether the rail is hidden |

The split is deliberate: a credential that survives the tab is a hazard, a sidebar width that
resets on every tab is an annoyance.

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
| `cost.*` | `enabled` true, `warmup_steps` 10, `log_per_step` true, `sync_cuda` true | Drives the `CostProfiler` callback ([`rinalmo_hub/cost.py`](../engine/rinalmo_hub/cost.py)); a **top-level** block, not nested under `trainer:` — `build_trainer()` splats unrecognised `trainer.*` keys straight into `pl.Trainer(**kwargs)` |

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
every rationale line it shows come from here** — either verbatim (`arms:`, `universal:`) or
as the output of a rule executed against the approved `DatasetSpec` (`generic.derived`,
§11) — which is what keeps the explanation and the executed config from drifting apart.

The platform ships no task definitions any more, so there is no per-task knowledge left to
carry: no shipped task exists for an entry to describe.

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
      - lr 1e-4 on an unfrozen backbone → destroys the backbone

universal:
  trainer: {precision: bf16-mixed}
  why: ["fp16's range is a real risk at these learning rates"]
  nondeterminism: {note, consequence}     # FlashAttention backward; F1 95.21 vs 95.82

generic:
  label: An unseen task trained from a single sequence+label table
  reference:
    band: null              # ALWAYS null — there are no known tasks to have a band
    tolerance: 0.0
    sources: ["No validated reference run: this task did not exist before your data."]
  derived:                  # values that cannot transfer across datasets, computed by
                            # recommender.py against the approved spec
    batch_size:
      rule: piecewise_on_median_length
      table: [[128, 64], [512, 32], [1024, 16], [2048, 8]]     # median nt → batch
      fallback: 4
      why: "…"
    max_epochs:
      rule: step_budget
      target_steps: [1000, 10000]
      clamp: [1, 20]
      why: "…"
    num_workers:
      value: 8
      why: "…"
  wall_clock:
    reference: "unknown — no run of this task exists yet"
    estimate_rule: "…"
  caveats:
    - "This task has no reference band. The first successful run becomes the baseline …"
```

`arms:` and `universal:` are unchanged from before this build ships tasks — the *arm*
settings were validated across tasks and transfer, so they still apply to any new task
verbatim. `generic:` replaces what used to be a `tasks:` section keyed by shipped task name
carrying a per-task reference band, epoch default and caveats: there is no longer a task to
key that on, so `generic.reference.band` is always `null` — a real, permanent state, not a
placeholder waiting to be filled in — and the values that used to be per-task defaults
(`trainer.max_epochs`, `data.batch_size`) are instead **derived** from the approved spec by
the three rules under `generic.derived`, each executed by `recommender.py` and each carrying
the `why:` line that generates the rationale shown at the gate.

### `target_shapes.yaml`

Three entries, keyed by **target type** — `binary`, `multiclass`, `regression` — not by a
task shape or dataset layout; there is nothing left to match a file against.

```yaml
target_shapes:
  binary:
    label: Binary sequence classification
    head: "one linear layer on the CLS-token representation → a single logit"
    extract_features: "representation[:, 0]   # CLS only; EOS/padding never reach the head"
    loss: binary_cross_entropy_with_logits
    metrics: [acc, precision, recall, f1_score]
    primary_metric: test/f1_score
    predict_output: "one probability per sequence — of the spec's positive_class"
    pad_sensitive: false
    adapter_state: "…"     # the concrete silent-failure trap for this shape
  multiclass: {...}        # cross_entropy, test/macro_f1, pad_sensitive: false
  regression: {...}        # mse, test/r2, pad_sensitive: true (pooled head)
```

Each recipe's `loss`, `metrics` and `primary_metric` are filled straight into
`DatasetSpec.head` (§11) — never chosen by the model. `head`/`extract_features` are prose
read by the ToolSmith fallback prompt; the templates
(`codegen/templates/*.j2`) independently reflect the same recipe in code. `predict_output`
becomes the sentence a registered tool's description appends; `pad_sensitive` forces serving
batch size 1 when true.

`tests/test_knowledge.py` fails loudly if an edit drops one of these values, because the
recommender and the templates have no other source for them.

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
`lm_config` must match the hub's; a tool whose landed `spec.json` marks its head
`pad_sensitive` (regression's pooled head, by the recipe in `target_shapes.yaml`) is forced
to `serving.batch_size = 1` — read from the spec, not from a hardcoded task name. The
artifact is copied to `<name>.pt.incoming`, the manifest is written, and only then is the
copy moved into place — so a failure between the two leaves neither an orphan file nor an
entry pointing at nothing.

## 6. `jobs_data/jobs.json` — the job store

`format_version: 1`, same atomic-write and revision discipline.

```jsonc
{
  "format_version": 1,
  "revision": 179,
  "jobs": {
    "my_task_lora_20260813_101810": {   // id == the output directory's basename
      "task": "my_task",
      "arm": "lora",
      "command": ["/…/python", "-m", "adaptrna_agentic.jobs.train_entrypoint", "--task", …],
      "output_dir": "/abs/path/outputs/my_task_lora_20260813_101810",
      "state": "running",           // running | succeeded | failed | cancelled
      "pid": 20866,
      "pid_starttime": "2149151089", // /proc/<pid>/stat field 22 — a PID alone is NOT
                                     // an identity; the kernel recycles them
      "started_at": "2026-08-13T00:18:30+00:00",
      "ended_at":   "2026-08-13T00:21:14+00:00",
      "exit_code": 0,
      "adapter_path": "/…/my_task_adapter.pt",
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
  "task":        "my_task",  # so the hub can dispatch with no config file
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
| `warnings` | Arm notes, generic caveats, quick-run truncation, missing backbone |
| `command` | **The exact argv** the JobRunner will execute, shown verbatim in the gate — **rebuilt** if the human edits `overrides`/`seed`/`arm`/`quick_run` at that gate, so it can never go stale |
| `human_overrides` | Present only if the human changed a field at the gate: `{"optim.lr": {"recommended": 0.0003, "chosen": 0.001}}`. Written by `orchestrator._apply_edits`; the job record and `analyze_run`'s report both print it, so a result is never read as having been produced on the recommended settings |

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
├── run_summary.json               final task metrics + per-stage cost (below), written last
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

When `cost.enabled` is on (the default), `metrics.csv` also carries `cost/<stage>/iter_time_ms`
and `cost/<stage>/mem_allocated_gb` columns at the usual `trainer.log_every_n_steps` cadence —
sparse like every other column. `analysis._final_metrics` filters the `cost/` prefix out before
comparing a run's metrics to a reference (it is training instrumentation, not a task metric),
but `runner.read_progress` surfaces it unfiltered for the live job-status panel. The
full-fidelity numbers — mean, median, p90 iteration time and mean/peak GPU memory, per stage —
live in `run_summary.json`, not in the (possibly downsampled) CSV.

## 10. Format versions and compatibility

| Artifact | Version | On mismatch |
|---|---|---|
| `tools.json` | 1 | `ValueError` naming both versions; refuses to read |
| `jobs.json` | 1 | same |
| adapter `.pt` | 2 (`SUPPORTED_FORMAT_VERSIONS = (2,)`) | `ValueError`; also validates required fields and `lm_config` |

`ToolEntry` gained its external-tool fields additively in Phase 3, with defaults, so v1
manifests written before that load unchanged — the pattern to follow for future additions.

## 11. The `DatasetSpec` schema

Produced by
[`profiling/profiler.py`](../agentic/adaptrna_agentic/profiling/profiler.py), put through
gate 1, consumed by `codegen/`, and landed beside the generated code as
`adaptrna_custom/tasks/<task>/spec.json`. One object carries the user's approved
interpretation of their data from gate 1 all the way to the registered tool — nothing
downstream re-derives it from the CSV a second time. Full walkthrough:
[modules/profiling-and-knowledge.md](modules/profiling-and-knowledge.md).

```jsonc
{
  "spec_version": 1,
  "source": "confirm_data_profile",          // "profile_dataset" until gate 1 approves it
  "path": "/abs/path/to/data.csv",
  "format": {"separator": ",", "compression": null, "rows": 24188, "header": true},

  "sequence_column": "sequence",
  "label_column": "label",
  "ignored_columns": ["gene_id", "source"],  // present in the file, not used
  "on_invalid": "fail",                      // fail | drop — rows outside ACGTUN

  "target_type": "binary",                   // binary | multiclass | regression
  "classes": ["0", "1"],                     // classification only; display order only —
                                              // NOT what decides polarity (positive_class is)
  "positive_class": "1",                     // binary only; required, never inferred from
                                              // classes' ordering
  "class_counts": {"0": 12094, "1": 12094},
  "target_summary": null,                    // {min, max, mean} for regression instead

  "alphabet": "dna",                         // dna | rna | other | unknown
  "length": {"min": 400, "median": 400, "max": 400},

  "split": {
    "mode": "random",                        // random | column
    "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
    "seed": 42, "stratify": true,
    "column": null, "mapping": null,
    "row_counts": {"train": 19350, "val": 2419, "test": 2419},
    "dropped_rows": 0
  },
  "split_candidates": {"species": {"human": 20000, "mouse": 4188}},

  "task_name": "donor_sites",
  "tool_description": "binary target trained from donors.csv",
  "head": {
    "kind": "cls_classifier",                // display-only; target_shapes.yaml itself
                                              // carries no "kind" field
    "loss": "binary_cross_entropy_with_logits",
    "metrics": ["acc", "precision", "recall", "f1_score"],
    "primary_metric": "test/f1_score",
    "predict_output": "one probability per sequence — of the spec's positive_class",
    "pad_sensitive": false
  },

  "warnings": ["1,204 sequences (5.0%) appear more than once in the file; …"],
  "similar_tasks": []
}
```

Two more keys appear conditionally:

* `human_edits` — present only if the human changed a field at gate 1:
  `{field: {"recommended": x, "chosen": y}}`. Written by `orchestrator._apply_edits` onto
  the spec object itself (the training plan's equivalent field is `human_overrides`, §8).
* `template_version` — present only in the **landed** `spec.json` of a task the template
  path rendered (`codegen/templates/render.py::TEMPLATE_VERSION`, read from the
  `TEMPLATE_VERSION` file next to the templates); absent for a task the LLM fallback wrote.
  `toolhub doctor`'s `template_version` check flags a landed task whose stamped version no
  longer matches the current template — a template fix does not reach already-landed code by
  design, so this makes *stale* a visible state rather than an invisible one.

Three properties matter, mirroring the training plan (§8):

* **It is stamped twice.** `profile_dataset` stamps `source: "profile_dataset"`; gate 1's
  re-validation (`confirm_profile`, called by the `confirm_data_profile` tool) re-stamps
  `source: "confirm_data_profile"` only after recomputing `classes`, `class_counts`,
  `row_counts` and `head` against the real file. `create_task_tool` refuses a spec that does
  not carry the second stamp.
* **`head` is filled from `target_shape(target_type)` (§4), never by the model.** Changing
  `target_type` at the gate changes the recipe; the loss and metrics are never a free
  choice.
* **It is the single source of truth downstream.** The datamodule's columns, the split, the
  primary metric, serving's pad-sensitivity and output validator, and the reuse matcher all
  read this one object.

Where it lives: in-memory during the turn; written as a fourth staged file by
`codegen/staging.py`; landed with the other three into `adaptrna_custom/tasks/<name>/spec.json`;
copied into `provenance["spec"]` (§5) at registration.
