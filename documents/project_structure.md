# Project Structure

Every directory and significant file, what it contains, and who depends on it.

Legend: **[entry]** executable entry point · **[cfg]** configuration · **[lib]** library
code · **[test]** test · **[gen]** generated/runtime artifact (git-ignored unless noted)
· **[doc]** documentation.

---

## Top level

```
adaptrna/
├── engine/            [lib]  the fine-tuning framework + vendored RiNALMo backbone
├── agentic/           [lib]  the agent platform (package: adaptrna_agentic)
├── adaptrna_custom/   [gen]  generated tasks and wrappers — git-TRACKED source
├── ui/                [lib]  browser client, plain ES modules, no build step
├── plans/             [doc]  master plan + one detailed plan per phase
├── docs/              [doc]  two screenshots referenced by the root README
├── documents/         [doc]  this documentation set
├── outputs/           [gen]  one directory per training run
├── toolhub_data/      [gen]  manifest, registry-owned adapters, staging
├── jobs_data/         [gen]  job records
├── chat_data/         [gen]  LangGraph SQLite checkpointer (all sessions)
├── dod_data/          [gen]  demo CSVs for the codegen flow (regenerable)
├── weights/           [gen]  backbone checkpoint (absent here; hub points at ~/.cache)
├── dataset/           [gen]  downloaded datasets (absent in this checkout)
├── .venv/             [gen]  Python 3.12 virtualenv
├── .env               [cfg]  ANTHROPIC_API_KEY — git-ignored, never printed
├── .gitignore         [cfg]
└── README.md          [doc]  user-facing quick start
```

Runtime artifacts live at the repo root — not inside a layer — because the engine's configs
reference `weights/…` and `dataset/…` relative to the **working directory**, and every
command runs from the repo root.

---

## `engine/` — the fine-tuning framework

```
engine/
├── pyproject.toml            [cfg]  package `rinalmo-hub`; pytest markers gpu/weights/data
├── README.md                 [doc]  complete hyperparameter + task-authoring reference
├── rinalmo_hub/              [lib]  THE FRAMEWORK — see modules/engine-hub.md
│   ├── registry.py                  @register_task / get_task / available_tasks / is_registered
│   ├── module.py                    BaseDownstreamModule — everything task-independent
│   ├── lora.py                      LoRA injection, freezing, multi-adapter switching
│   ├── adapter.py            [entry] adapter file format v2; `python -m rinalmo_hub.adapter <f>`
│   ├── config.py                    base.yaml → task yaml → --set layering; Config object
│   ├── hub.py                       RiNALMoHub: N adapters resident in one backbone
│   ├── cost.py                      CostProfiler: per-stage GPU memory + iteration time
│   ├── tasks/                       one module per shipped task
│   │   ├── __init__.py              imports all three so @register_task fires
│   │   ├── splice_site.py           binary classification on the CLS token
│   │   ├── mrl.py                   regression + fitted target scaler (ADAPTER_EXTRA_PREFIXES)
│   │   └── sec_struct.py            per-pair classification + tuned threshold (extra payload)
│   ├── data/
│   │   └── mrl.py                   MRLDataModule: adds val_split=random7600|holdout
│   └── cli/
│       ├── common.py                shared plumbing; the load-bearing construction order
│       ├── train.py          [entry] python -m rinalmo_hub.cli.train
│       ├── evaluate.py       [entry] python -m rinalmo_hub.cli.evaluate
│       └── predict.py        [entry] python -m rinalmo_hub.cli.predict
├── rinalmo/                  [lib]  VENDORED backbone — see modules/engine-backbone.md
│   ├── config.py                    model_config("nano"|"micro"|"mega"|"giga")
│   ├── pretrained.py                downloads giga-v1.pt into ~/.cache/rinalmo_pretrained/
│   ├── model/
│   │   ├── model.py                 RiNALMo: embedding → token dropout → transformer → LM head
│   │   ├── modules.py               Transformer, TokenDropout, MaskedLanguageModelHead
│   │   ├── attention.py             flash-attn path + plain-PyTorch fallback
│   │   ├── rope.py                  rotary position embeddings
│   │   └── downstream.py            the shipped prediction heads
│   ├── data/
│   │   ├── alphabet.py              Alphabet.encode / batch_tokenize
│   │   ├── constants.py             token names (CLS/PAD/EOS/UNK/MASK), RNA_TOKENS
│   │   └── downstream/              per-benchmark datasets + datamodules
│   ├── utils/
│   │   ├── finetune_callback.py     GradualUnfreezing (drives ft_schedules/*.yaml)
│   │   ├── scaler.py                StandardScaler as an nn.Module (buffers travel in adapters)
│   │   ├── sec_struct.py            prob_mat_to_sec_struct, ss_f1/precision/recall, save_to_ct
│   │   ├── splice_site_metrics.py   accuracy/precision/recall/specificity/f1 as percentages
│   │   └── download.py              dataset acquisition
│   └── resources/            [cfg]  model2gdisk.json, remote_data.json — download URLs
├── configs/                  [cfg]
│   ├── base.yaml                    defaults for EVERY key; the bottom layer
│   └── tasks/{splice_site,mrl,sec_struct}.yaml
├── ft_schedules/             [cfg]  gradual-unfreezing schedules (full FT only)
│   ├── head_only.yaml               head forever
│   ├── splice_site.yaml             head + whole transformer from epoch 0
│   ├── mrl.yaml                     head 3 epochs, then final LN + blocks 6..32
│   ├── mrl_paper.yaml               head 5 epochs, then the entire backbone
│   └── sec_struct.yaml              head 3 epochs, then 3 blocks every 3 epochs from the top
├── examples/
│   └── ncrna_classification/        the worked "adding a task" example, deliberately
│       ├── task.py                  OUTSIDE rinalmo_hub/tasks/ so a test can assert
│       ├── datamodule.py            that no core file mentions it
│       └── config.yaml
├── scripts/
│   └── benchmark_switching.py [entry] adapter-switch vs full-reload timing
└── tests/                    [test]  135 CPU tests + 7 opt-in — see testing.md
```

### The engine files that matter most

| File | Responsibility | Depended on by |
|---|---|---|
| `rinalmo_hub/module.py` | `BaseDownstreamModule`: backbone construction, LoRA injection, adapter I/O, optimizer/scheduler, generic train/val/test steps. Defines the subclass contract. | Every task (shipped, example, generated); the hub; both CLIs |
| `rinalmo_hub/registry.py` | Name → task class. `@register_task` is what makes a task visible without editing any core file. | CLI, hub, harness, knowledge base, discovery |
| `rinalmo_hub/lora.py` | Where LoRA is injected, what stays frozen, how several adapters coexist, and which state-dict keys belong to which adapter. | `module.py`, `hub.py` |
| `rinalmo_hub/adapter.py` | The adapter file: a versioned dict carrying task, backbone size, LoRA geometry, head config, tensors, non-tensor extras, run metadata. | `module.py`, `hub.py`, ToolHub registry, `toolhub info` |
| `rinalmo_hub/hub.py` | `RiNALMoHub` — the inference contract the whole agentic layer sits on. | `AdapterRuntime`, `cli/predict.py`, harness check 7 |
| `rinalmo_hub/cost.py` | `CostProfiler`: per-stage (train/val/test) mean/peak GPU memory and mean iteration time, written to `run_summary.json`. | `cli/train.py` |
| `rinalmo_hub/cli/common.py` | The construction order (build → load backbone → inject LoRA → load adapter) plus trainer/callback assembly. Getting this order wrong is a silent failure. | `train.py`, `evaluate.py` |
| `configs/base.yaml` | Every configurable key with a default. The single source for "what can I set?". | All three CLIs, the harness, the recommender |

Naming is fixed framework-wide: the backbone is always `self.backbone`, the head is always
`self.head`. That is what makes the `ft_schedules/*.yaml` regexes interchangeable between
tasks.

---

## `agentic/` — the agent platform

```
agentic/
├── pyproject.toml              [cfg]  package `adaptrna-agentic`; `ui` marker opt-in
├── README.md                   [doc]  per-phase user guide
├── adaptrna_agentic/
│   ├── settings.py             [lib]  ROLES, per-role model resolution, .env loading,
│   │                                  REPO_ROOT, require_api_key()
│   ├── models.py               [lib]  build_chat_model(role) — the ONLY provider-aware seam
│   ├── toolhub/                [lib]  see modules/toolhub.md
│   │   ├── manifest.py                on-disk state: BackboneConfig, ToolEntry, Manifest
│   │   ├── registry.py                lifecycle: register / activate / deactivate / remove / verify
│   │   ├── runtime.py                 AdapterRuntime: one backbone, all adapters, one lock
│   │   ├── errors.py                  ToolHubError, ConcurrentModificationError
│   │   ├── doctor.py                  read-only health checks, each with a remedy
│   │   ├── prune.py                   the only destructive code path in the project
│   │   └── external/
│   │       ├── contract.py            ExternalToolSpec + loader + install + golden runner
│   │       └── vienna.py              the hand-written reference wrapper (ViennaRNA)
│   ├── agents/                 [lib]  see modules/agents.md
│   │   ├── orchestrator.py            the LangGraph chat graph + approval node
│   │   ├── tool_factory.py            ToolHub → LangChain bridge; all 16 management tools
│   │   ├── toolsmith.py               writes task files / wrapper modules (structured output)
│   │   ├── verifier.py                independent review in a fresh context
│   │   └── hello.py                   Phase-0 scaffold graph + gc_content demo tool
│   ├── codegen/                [lib]  see modules/codegen.md
│   │   ├── pipeline.py                the bounded ToolSmith ⇄ Verifier loop
│   │   ├── harness.py                 launches the runner, summarises, enforces requirements
│   │   ├── _harness_runner.py  [entry] the 7 checks, executed inside the sandbox
│   │   ├── sandbox.py                 subprocess + rlimits + timeout + result marker
│   │   ├── staging.py                 stage / list / load / land / discard
│   │   ├── discovery.py               import adaptrna_custom tasks so @register_task fires
│   │   └── prompts.py                 context assembly for both agents
│   ├── jobs/                   [lib]  see modules/jobs.md
│   │   ├── runner.py                  launch/track/cancel; PID identity; metrics.csv reader
│   │   ├── store.py                   jobs.json with atomic writes + revision counter
│   │   ├── analysis.py                RunAnalyzer: verdict + reasons + remedies
│   │   └── train_entrypoint.py [entry] the MASTER_PLAN §3.4 seam
│   ├── profiling/              [lib]  see modules/profiling-and-knowledge.md
│   │   ├── profiler.py                describe a dataset; name the task that can read it
│   │   └── recommender.py             profile → executable, stamped training plan
│   ├── knowledge/              [cfg]  the grounding corpus
│   │   ├── __init__.py                loaders (lru_cached) + generic fallback entry
│   │   ├── hyperparameters.yaml       validated arms, failure modes, per-task bands + ETAs
│   │   └── task_templates.yaml        data profile → task shape; no_match_guidance
│   ├── api/                    [lib]  see modules/api.md
│   │   ├── app.py                     create_app: error handlers, bearer auth, routers
│   │   ├── deps.py                    Services singletons; WAL checkpointer; is_loopback
│   │   ├── events.py                  graph run → SSE frames; history(); pending_approval()
│   │   ├── schemas.py                 request bodies (responses are the CLI's own dicts)
│   │   └── routers/{system,tools,jobs,sessions,ui}.py
│   └── cli/                    [lib]  see modules/cli.md
│       ├── chat.py             [entry] terminal REPL + the terminal approval renderer
│       ├── toolhub.py          [entry] management CLI (15 subcommands)
│       └── serve.py            [entry] uvicorn launcher + the binding refusal
├── scripts/
│   └── make_demo_data.py       [entry] regenerate dod_data/ from Spliceator folds
└── tests/                      [test]  381 tests + 11 opt-in — see testing.md
    ├── conftest.py                    nano adapter fixtures built through the public API only
    ├── scripted_model.py              the fake chat model every graph test uses
    ├── api_helpers.py                 SSE collection helpers for the HTTP tests
    ├── fixtures/
    │   ├── broken_task_sources.py     deliberately defective tasks the harness MUST fail
    │   └── dummy_external.py          a wrapper module with no real package behind it
    └── scenarios/*.yaml               recorded conversations, replayed against the fake model
```

### The agentic files that matter most

| File | Responsibility | Depended on by |
|---|---|---|
| `agents/tool_factory.py` | The single meeting point of the LLM and the deterministic services. Defines the 16 management tools, wraps every registered tool, and names the gated set — which since Phase 10 includes the two tool-state toggles. | orchestrator, every test of agent behaviour |
| `api/sessions_store.py` | Sessions as a managed resource: recency listing and rename, in plain SQL over the checkpointer's own tables. Owns no schema — a sidecar metadata table would be a second source of truth the terminal never writes. | `api/routers/sessions.py`, the web UI's session rail |
| `toolhub/manifest.py` | The on-disk truth about what tools exist. Pure data + JSON I/O; never imports the engine. | registry, runtime, doctor, prune, recommender, API |
| `toolhub/runtime.py` | The only place a backbone is ever loaded in this layer, and the only place inference is serialised. | chat, CLI, HTTP, harness-adjacent code |
| `profiling/recommender.py` | Turns a profile into an *executable, stamped* plan — including the exact argv. | `start_training`, the approval gate, tests that parse the command with the engine's own parser |
| `codegen/_harness_runner.py` | The seven checks that decide whether generated code is trustworthy. | pipeline, `tests/test_harness.py` (controls **and** catches) |
| `jobs/runner.py` | Process lifecycle for detached GPU work, including PID identity. | pipeline tools, HTTP job routes, doctor |
| `settings.py` | `REPO_ROOT` (used by nearly everything to resolve paths) and per-role model config. | almost every module |

---

## `adaptrna_custom/` — generated code (git-tracked)

```
adaptrna_custom/
├── README.md              [doc]  what lives here and the guarantee that nothing regenerates it
├── __init__.py
├── tasks/
│   ├── __init__.py
│   └── splice_simple/     the task the ToolSmith generated, verified and landed
│       ├── __init__.py
│       ├── task.py        @register_task("splice_simple"); threshold in the extra payload
│       ├── datamodule.py  reads dod_data/splice_simple_{train,val,test}.csv
│       └── config.yaml    task: splice_simple; data.root: dod_data
└── tools/
    └── __init__.py        a docstring only — no external wrapper has been generated yet
```

This directory is **source code, not runtime state**: git-tracked, hand-editable, and never
silently regenerated — `pipeline._reject_existing` refuses to generate a task whose name
already exists here.

`splice_simple` is worth reading as the reference for what generated code looks like when
it passes review: it declares its non-tensor `threshold` through
`adapter_extra_payload()`/`load_adapter_extra()` (silent-failure question 1) and documents
in a comment that its head consumes only the CLS token (question 2).

Discovery is automatic — [`codegen/discovery.py`](../agentic/adaptrna_agentic/codegen/discovery.py)
imports every `tasks/*/task.py` before training, serving or verification. A module that
fails to import is reported by name and does not break the others.

---

## `ui/` — the browser client

```
ui/
├── README.md     [doc]  why no build step, and the tripwire for changing that
├── index.html           the shell: layout + modal skeleton; loads /ui/app.js as a module
├── app.js        (821)  wiring: the activity bar, both rail views, job + log polling
├── sse.js        (105)  SSE over fetch — a pure frame parser plus a thin transport
├── api.js        (114)  endpoints as functions; bearer token in sessionStorage
├── render.js     (394)  messages, session/tool/job rows, the job-log header, the modal
├── md.js         (180)  small Markdown renderer (the model writes tables)
├── dom.js        (34)   the shared el() helper; nothing here assigns innerHTML
└── style.css     (652)  one stylesheet, light and dark
```

1,939 lines total, 1,329 of them JavaScript — past the thousand-line tripwire `README.md`
sets for reaching for a framework, and now carrying a little client-side state as well. See
that file for the accounting. Served by
[`api/routers/ui.py`](../agentic/adaptrna_agentic/api/routers/ui.py): the shell at `/` with
`Cache-Control: no-store`, the assets mounted at `/ui` (not `/`, which would shadow `/api`
and `/health`). If the package is installed away from its checkout, `/` returns a 503 page
explaining that the API itself is fine.

---

## `plans/` and `docs/`

`plans/` holds the design record: [`MASTER_PLAN.md`](../plans/MASTER_PLAN.md) is the map
(architecture, principles, engine constraints, phase roadmap, resolved decisions), and
`PHASE_0`…`PHASE_9` are the detailed blueprints written before each phase. They explain
*why*; this documentation describes *what*. `docs/` holds only the two PNG screenshots the
root README embeds.

---

## Generated / runtime directories

| Directory | Written by | Structure |
|---|---|---|
| `outputs/<run_name>/` | engine trainer + `train_entrypoint` | `resolved_config.yaml`, `metrics/version_N/{metrics.csv,hparams.yaml}`, `train.log`, `exit_code`, `<task>_adapter.pt` (LoRA) or `<task>_full.pt` (`--save_full_weights`) |
| `toolhub_data/` | `Registry`, `staging` | `tools.json`, `adapters/<tool>.pt`, `staging/<stage_id>/adaptrna_custom/…` |
| `jobs_data/` | `JobStore` | `jobs.json` |
| `chat_data/` | `SqliteSaver` | `sessions.sqlite` (+ `-wal`, `-shm` under WAL mode) |
| `dod_data/` | `agentic/scripts/make_demo_data.py` | `splice_simple_{train,val,test}.csv` — flat `sequence,label` CSVs no shipped task can read, which is exactly why the codegen flow exists |
| `weights/`, `dataset/` | engine downloaders | `giga-v1.pt`; per-task dataset roots |

Nine run directories exist in this checkout, spanning donor LoRA, donor full-FT,
head-only, acceptor LoRA and four `splice_simple` runs — useful as real fixtures when
working on the analyzer or the job store.

On-disk schemas for `tools.json` and `jobs.json`: [configuration.md](configuration.md).
