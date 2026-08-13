# AdaptRNA — Master Plan

From a fine-tuning framework to a conversational agent platform for RNA analysis.

This document is the map for the whole project. It is deliberately high-level: each phase in
§8 gets its own detailed plan when work on it starts, and the status column in §8 is updated
as phases land. Decisions already made are recorded here so they are not re-litigated;
decisions still open are tracked in §10.

---

## Contents

1. [Vision](#1-vision)
2. [Architecture](#2-architecture)
3. [Design principles](#3-design-principles)
4. [The Tool-Hub](#4-the-tool-hub)
5. [Agent topology](#5-agent-topology)
6. [The knowledge base](#6-the-knowledge-base)
7. [Engine constraints the agent layer must respect](#7-engine-constraints-the-agent-layer-must-respect)
8. [Phased roadmap](#8-phased-roadmap)
9. [Testing strategy](#9-testing-strategy)
10. [Open decisions](#10-open-decisions)
11. [How to use this plan](#11-how-to-use-this-plan)

---

## 1. Vision

### What exists today

The **engine** (RiNALMo-Hub): one frozen pretrained RiNALMo backbone, swappable LoRA
adapters and task heads, task-pluggable training / evaluation / prediction behind one CLI,
and `RiNALMoHub` serving several adapters from a single loaded backbone. Measured results:
LoRA matches or beats full fine-tuning at ~0.2% of the parameters, each task fits in a ~6 MB
self-describing adapter file, and adapter switching is a dictionary lookup rather than a
2.6 GB reload. The engine's own spec anticipated this project: *"an agent will sit on top of
this and, asked to perform a task, will activate only that task's weights."*

### What this project builds

A conversational **agent platform** on top of the engine, where:

- The foundation model's capabilities — the adapters — and classical non-neural
  bioinformatics tools (ViennaRNA, …) are uniform **tools** in a **Tool-Hub**.
- The backbone is loaded once; when a user request needs a model capability, the matching
  adapter is activated and inference runs — the user never touches a checkpoint.
- Users **create new tools at runtime**:
  - *New adapter tools* — the user provides data and a description, receives a recommended
    fine-tuning configuration, approves it, a LoRA run trains on the local GPU, the results
    are analyzed and presented, and on approval the adapter is registered as a tool.
  - *New external tools* — the user names a package (e.g. ViennaRNA); the system proposes
    the install, generates wrapper code, verifies it, and registers it on approval.
- Tools can be **listed, activated, deactivated and tested** from the conversation.
- Delivery happens in three stages: **terminal chat → HTTP API → web UI**.

### Decisions already made

| Decision | Choice |
|---|---|
| Agent orchestration | **LangGraph**, model kept provider-agnostic via LangChain's chat-model interface |
| Model behind the agents | **Claude API** (per-role model selection, swappable) |
| Where fine-tuning runs | **This machine's GPU**, as engine-CLI subprocesses |
| Repository shape | Monorepo, three top-level layers: `engine/`, `agentic/`, `ui/` |
| First interface | Non-GUI terminal chat; web UI comes last |
| Naming | New platform packages are **model-agnostic** (`adaptrna_*`): RiNALMo is the first backbone this platform serves, not its brand — other foundation models may back it later. Only the vendored engine packages (`rinalmo`, `rinalmo_hub`) keep their established names. |

---

## 2. Architecture

```
┌────────────────────────────── ui/  (Phase 9) ──────────────────────────────┐
│ web app: chat · tool dashboard · training monitor · approval dialogs       │
└─────────────────────────────────────┬───────────────────────────────────---┘
                                      │ HTTP + SSE  (FastAPI, Phase 8)
┌───────────────────────────────── agentic/ ─────────────────────────────────┐
│                                                                            │
│  Agents (LangGraph graphs, Claude API behind a pluggable model interface)  │
│  ┌──────────────┐   delegates   ┌───────────┐  bounded loop  ┌──────────┐  │
│  │ Orchestrator │──────────────▶│ ToolSmith │◀──────────────▶│ Verifier │  │
│  │ (user-facing)│               │ (codegen) │                │ (review) │  │
│  └──────┬───────┘               └───────────┘                └──────────┘  │
│         │ direct tool calls                                                │
│  Deterministic services (plain Python, no LLM)                             │
│  ┌─────────────────────┐  ┌───────────────────┐  ┌───────────────────────┐ │
│  │ ToolHub             │  │ JobRunner         │  │ DataProfiler +        │ │
│  │ registry + runtime  │  │ local GPU         │  │ ConfigRecommender     │ │
│  │ (one backbone, all  │  │ subprocesses,     │  │ (grounded by the      │ │
│  │  adapters resident) │  │ metrics.csv watch │  │  knowledge base)      │ │
│  └─────────────────────┘  └───────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────┬──────────────────────────────────---─┘
                                      │ RiNALMoHub Python API · CLI subprocess
                                      │ · adapter files (.pt)
┌───────────────────────────────── engine/ ──────────────────────────────────┐
│ RiNALMo-Hub exactly as it exists today (moved, import names unchanged)     │
└──────────────────────────────────────────────────────────────────────────---┘
```

### Target repository layout

```
adaptrna/
├── engine/                # the current code, moved as-is:
│   ├── rinalmo/           #   vendored backbone
│   ├── rinalmo_hub/       #   the fine-tuning framework
│   ├── configs/  ft_schedules/  examples/  scripts/  tests/
│   └── pyproject.toml
├── agentic/               # NEW — the agent system (python package: adaptrna_agentic)
│   ├── toolhub/           #   registry, AdapterTool, ExternalTool, runtime, mgmt CLI
│   ├── agents/            #   orchestrator, toolsmith, verifier graphs + prompts
│   ├── jobs/              #   JobRunner: launch/track/monitor engine training runs
│   ├── knowledge/         #   validated hyperparameters, task templates, checklists
│   ├── profiling/         #   DataProfiler + ConfigRecommender
│   ├── cli/               #   terminal chat REPL + toolhub management commands
│   ├── api/               #   FastAPI service (Phase 8)
│   ├── tests/
│   └── pyproject.toml
├── ui/                    # NEW — web frontend (Phase 9)
├── weights/  dataset/  outputs/  adapters/  toolhub_data/
│                          # runtime artifacts, shared by all layers, at repo root
└── MASTER_PLAN.md
```

Runtime artifacts stay at the repo root: the engine's configs reference `weights/…` and
`dataset/…` relative to the working directory, so commands keep running from the repo root
exactly as they do today.

---

## 3. Design principles

1. **Agents thin, logic deterministic.** Registry operations, job management, validation,
   data profiling and file handling are plain Python services with unit tests. The LLM is
   used only where judgment is genuinely needed: understanding intent, recommending a
   configuration, writing code, reviewing code, interpreting training results. A registry
   that "usually" works is worthless; a prompt that occasionally rephrases a summary is fine.

2. **Human approval gates.** LangGraph's `interrupt()` mechanism pauses the graph and waits
   for the user before every consequential action: installing a package, launching a
   training run, landing generated code, registering or removing a tool. Approval in one
   step never implies approval for the next.

3. **Everything testable on `nano`.** The engine's test philosophy carries over: the agentic
   suite runs on a randomly-initialised `nano` backbone with synthetic data — seconds on
   CPU, no pretrained weights, no datasets, no GPU, no API key for the deterministic parts.

4. **The engine is a stable substrate — zero engine edits.** The agent layer consumes the
   engine through three contracts: the `RiNALMoHub` Python API for inference, the CLI as a
   subprocess for training (`resolved_config.yaml` + `metrics.csv` are the interface), and
   adapter files as artifacts. The one seam — custom task modules must be imported before
   the engine CLI can see them — is handled agentic-side: a thin training entrypoint imports
   the registered custom-task packages, then delegates to `rinalmo_hub.cli.train.main()`.

5. **Tools are manifest entries, not code paths.** Both tool kinds share one lifecycle
   (register / activate / deactivate / test / remove / list) driven by one manifest store,
   so the orchestrator, the management CLI, the API and the UI all see the same picture.

6. **Adapter tools are LoRA-only.** The hub refuses full fine-tuning exports by design
   (only the head travels in such a file; serving it would silently pair a fine-tuned head
   with the pretrained backbone). Full FT remains available through the engine CLI for
   paper experiments — it just cannot become a served tool, and the fine-tuning pipeline
   defaults to LoRA.

---

## 4. The Tool-Hub

The deterministic heart of the platform: a **registry** (what tools exist and in what
state) plus a **runtime** (how they execute).

### Manifest

One store at `toolhub_data/` (JSON or SQLite — §10). Each entry:

```
name            unique tool name (e.g. "splice_site_donor", "vienna_fold")
type            adapter | external
state           active | disabled
description     what the tool does, shown to the user and to the orchestrator
artifacts       adapter: path to the .pt file · external: wrapper module + install spec
task            adapter tools: the engine task name the adapter serves
serving         per-tool inference policy (batch size, length bucketing — see §7)
test            smoke-test spec: known inputs + expected outputs/tolerances
provenance      created_by (user/agent), created_at, training run dir,
                resolved_config, final metrics, engine git SHA
```

### AdapterTool (foundation-model tools)

Wraps `RiNALMoHub`. The backbone is loaded **once** (eager at chat start or lazy on first
call — §10); every *active* adapter tool is `hub.register()`-ed into it; a call activates
the right adapter via `set_adapter` — a dictionary lookup, not a reload. Adapter files are
self-describing (task, backbone size, LoRA geometry, head config, extras), so registration
needs nothing but the file path. Deactivation is routing-level: the tool disappears from
the orchestrator's tool list and the management surfaces. peft cannot cleanly uninject an
adapter, but resident adapters cost megabytes, so this is fine; a hub-rebuild operation
exists for full cleanup.

### ExternalTool (non-neural tools)

A Python wrapper module exposing typed functions (e.g. ViennaRNA: `fold` → MFE structure +
energy), an install spec (pip package, approval-gated), and golden-pair smoke tests (known
hairpin sequence → expected dot-bracket string). The first external tool — ViennaRNA — is
**hand-written** as the reference implementation; it doubles as the pattern the ToolSmith
imitates when generating wrappers for packages the user introduces later.

### Lifecycle

`register / activate / deactivate / test / remove / list` — exposed twice: as a raw
management CLI (usable and testable before any agent exists, Phase 2) and as agent tools
bound to the orchestrator (Phase 4).

---

## 5. Agent topology

Three agents, matching the three genuinely distinct judgment roles. Everything else is a
service (§3.1).

| Agent | Role | Why it is separate |
|---|---|---|
| **Orchestrator** | The only agent the user talks to. Understands intent, answers questions, calls inference and management tools **directly**, walks the user through the creation flows, presents approval gates. | An executor-agent indirection for single tool calls adds latency and context loss with zero benefit. Delegation is reserved for long multi-step work. |
| **ToolSmith** | Generates code for new tools: the three files of a new engine task (task module, datamodule, config YAML — following `engine/examples/ncrna_classification/` as the template) or an external wrapper (following the ViennaRNA reference). | Codegen wants a focused, file-centric context and its own prompt discipline, not the chat history. |
| **Verifier** | Reviews ToolSmith output in a **fresh context**: static review against the checklists (§6), then executes CPU structural tests (module constructs on `nano`, forward pass runs, adapter round-trips) and sandboxed smoke tests, and reports findings. | An auditor that inherits the writer's context inherits the writer's blind spots. Independence is the point. |

The ToolSmith↔Verifier loop is bounded (≤3 iterations); whatever the outcome, the final
gate is the human reviewing the diff before code lands anywhere.

Model selection is per-role (a fast model for orchestration, the strongest available for
codegen and review) and provider-agnostic through LangChain. Conversation state lives in a
LangGraph SQLite checkpointer — which also gives the API and UI session resume for free.

### The five user flows

- **A — Inference.** "Does this sequence contain a donor splice site?" → orchestrator picks
  the tool, ToolHub activates the adapter, predicts, answers in task-native terms.
- **B — Tool management.** "What tools are available?" / "disable vienna" / "test the MRL
  tool" → registry operations, results narrated.
- **C — New adapter, existing task shape.** User provides data + description → DataProfiler
  inspects it (format, lengths, label type/distribution) → ConfigRecommender proposes a
  config + arm + expected wall-clock, grounded in §6 → **[approval]** → JobRunner launches
  the engine CLI on the local GPU, streams progress from `metrics.csv` → results analyzed
  against expectations → **[approval]** → adapter registered as a tool.
- **D — New task type.** As C, but no existing task class fits: ToolSmith generates the
  three files → Verifier loop → **[human diff approval]** → then flow C's training path.
- **E — New external tool.** User names a package → ToolSmith proposes install + wrapper →
  **[approval to install]** → Verifier smoke-tests → **[approval]** → registered.

---

## 6. The knowledge base

What makes recommendations *grounded* rather than hallucinated: a curated, versioned corpus
in `agentic/knowledge/`, distilled from the engine's spec and measured runs.

- **Validated hyperparameters and their failure modes.** LoRA: `lr 3e-4`,
  `gradient_clip_val 1.0`, `layer_stride 3` — and *why*: `1e-3` trained well to step ~325,
  then one gradient spike collapsed it into a constant-output state it never escaped.
  Full FT: `lr 1e-5` or a head-only warm-up schedule — `1e-4` on an unfrozen backbone
  destroys it (R² ≈ 0). Always `bf16-mixed`, never fp16. Per-task recipes (MRL paper
  schedule, sec-struct gradual unfreezing, splice-site folds).
- **Task-shape templates.** Data profile → task class: binary/multiclass sequence
  classification (CLS token + linear head), sequence regression (pooled + conv head,
  target scaler), per-pair prediction (outer-concat + ResNet2D head). Each template names
  the head, loss, metrics, `extract_features` pattern and predict batch policy.
- **The three-file walkthrough** for new tasks, and the **two silent-state questions** as a
  hard Verifier checklist: (1) does the task own state predictions depend on that is not a
  head weight? (tensor → `ADAPTER_EXTRA_PREFIXES`; plain value → `adapter_extra_payload`) —
  (2) does the head need CLS/EOS/pad positions excluded in `extract_features`? Both fail
  silently with plausible-looking numbers if missed.

---

## 7. Engine constraints the agent layer must respect

Known engine behaviors, rephrased as design consequences for this layer:

| Engine fact | Consequence up here |
|---|---|
| Construction order is load-bearing: build module → load backbone → inject LoRA → load adapter | The agent layer **reuses** engine code paths (`RiNALMoHub.register`, CLI plumbing) and never reimplements loading |
| Trainable state outside `head.*` / LoRA keys / declared extras silently does not persist in adapter files | Verifier checklist item for every generated task; the two questions in §6 are mandatory, not advisory |
| Pad-sensitive heads (MRL) make hub predictions depend on batch composition (padding + InstanceNorm) | Per-tool `serving` policy in the manifest (batch size 1 or length bucketing); documented per tool |
| The hub refuses full-FT exports | Adapter tools are LoRA-only (§3.6); the pipeline defaults to LoRA |
| No metric-based checkpoint selection exists; final-epoch weights are what gets tested | If the pipeline ever adds early stopping / best-checkpoint selection, MRL-style tasks must switch to `val_split=holdout` first — otherwise a reported result becomes the selection set |
| FlashAttention's backward is non-deterministic; the forward pass is deterministic | Training-result analysis uses tolerances (differences under ~1 F1 point are noise; multi-seed for claims); adapter smoke tests may assert tight values on a fixed device/dtype |
| Gradient checkpointing is unconditionally on; `need_attn_weights=True` forces the slow attention path | Do not "optimize" either; attention-map features would need explicit design |
| Non-autocast half-precision inference is broken: casting the model to bf16/fp16 trips a dtype promotion in `TokenDropout` (an fp32 scalar promotes activations to fp32, which then hit bf16 layer-norm weights) — found in Phase 2 (2026-08-12) | ToolHub serving runs fp32 (`dtype: auto`); half-precision serving needs an engine fix or an agentic-side autocast wrapper (revisit when serving throughput matters, Phase 4/8) |
| A verification harness that runs in a different working directory than the code will actually run in *skips* data-dependent checks instead of failing them — found in Phase 6 | The harness runs from the repo root, exactly as the JobRunner runs training, and generated code cannot be approved unless the datamodule/forward/metrics checks actually **ran** (a skipped check is not a pass) |
| `RiNALMoHub.predict` activates an adapter across the *whole* backbone before running, so two overlapping predictions for different tools can answer from the wrong adapter — harmless with one CLI caller, unsafe the moment a server has two (Phase 8) | Inference is serialised by a lock inside `AdapterRuntime`, so every caller (CLI, chat, HTTP) is covered rather than each entry point remembering. Throughput is one prediction at a time; correctness first |
| A stored PID is not a process identity: PIDs are recycled, so a dead job can look alive and `killpg` can signal a stranger — found in Phase 7 on this machine, where every recorded PID was already free | Jobs record `(pid, start time)`; liveness and cancel require both. A record without a start time is treated as gone rather than signalled |
| `(mtime, size)` is not a change stamp: two writes of identical content inside one filesystem timestamp tick are indistinguishable — found in Phase 7 | Both JSON stores carry a monotonic revision counter *inside* the file |
| A model whose deterministic tool errors will route around it — here, hand-assembling a training plan when the recommender refused an unknown task (Phase 6) | Plans are stamped by `recommend_training_config`; `start_training` refuses anything unstamped. Guardrails that matter are enforced in code, not in the prompt |
| `configs/base.yaml` defaults `pretrained_weights` to `weights/giga-v1.pt`, resolved against the *working directory* — a path that need not exist (here the checkpoint lives in `~/.cache/rinalmo_pretrained/`) | Every training plan sets `pretrained_weights` and `lm_config` from the **ToolHub manifest's** backbone, so a run trains against exactly the backbone the hub serves; a hub with no checkpoint configured warns instead of silently training from random weights (found in Phase 5) |
| Phase 7's optimistic concurrency answers **409 while a store is mid-write**, and the likeliest moment for that is just after a run starts — exactly when a client is watching most closely (found in Phase 9's live DoD, which crashed on it) | A 409 marked `retryable` is a *transient* answer, not a stop condition: polling clients keep their last good render **and keep polling**. The browser's job monitor re-arms its timer on failure; a version that returned early froze the panel permanently the first time a job started |

---

## 8. Phased roadmap

Each phase gets a detailed plan before work starts. A phase is done when its definition of
done is demonstrated, not when its code exists.

| # | Phase | Deliverable | Definition of done | Status |
|---|---|---|---|---|
| 0 | Scaffolding | `agentic/` package skeleton; LangGraph + langchain-anthropic deps; per-role model config; API-key handling | A hello-world graph answers via Claude from the terminal | ✅ done 2026-08-12 — 16 tests green; DoD demo verified (tool call → 0.583 answer, REPL incl. parallel tool calls) |
| 1 | Repo restructure | Existing code moved to `engine/`; `agentic/` and `ui/` created; paths fixed (pyproject packaging, `rinalmo_hub/config.py` REPO_ROOT, CLI default-config path, pytest pythonpath, README) | `cd engine && pytest` → 135 passed; a smoke train command still works from the repo root | ✅ done 2026-08-12 — 135+16 tests green post-move, zero engine code edits; stale editable install (pointing at ~/bio2/RiNALMo) replaced by `pip install -e ./engine` |
| 2 | ToolHub core (no LLM) | Manifest registry; AdapterTool over `RiNALMoHub`; lifecycle ops; management CLI | The existing splice-site adapter (`outputs/splice_donor_lora/`) registered and predicting from the CLI; nano tests green | ✅ done 2026-08-12 — 52 agentic tests green; real donor adapter registered, positives 0.9996/0.9998 vs negatives ~0.01–0.14 on Danio windows; smoke test + routing-level deactivation verified |
| 3 | External tools | ExternalTool interface; approval-gated install flow; hand-written ViennaRNA reference wrapper; golden smoke tests | `toolhub test vienna_fold` passes; enable/disable works | ✅ done 2026-08-12 — 91 agentic tests green; gated install demonstrated (decline → --yes → ViennaRNA 2.7.2); goldens captured & pinned (`((((....))))` @ −5.4); enable/disable verified; wrapper contract in `external/contract.py` is the Phase 6 template |
| 4 | Orchestrator MVP | LangGraph chat graph in a terminal REPL; ToolHub tools bound; sessions checkpointed (SQLite) | "What tools are available?" and "is this sequence a donor site?" answered end-to-end, adapter switching under the hood | ✅ done 2026-08-12 — 111 agentic tests green; live DoD: list_tools → real donor window scored 0.9996 → vienna fold → disable/refuse/re-enable-and-use-in-one-turn → session resumed across process restart |
| 5 | Fine-tuning pipeline | DataProfiler; ConfigRecommender (knowledge-base-grounded); JobRunner (local GPU subprocess, `metrics.csv` monitoring, job store); result analysis; registration step | The scenario works from chat: data in → recommended config → approval → LoRA run trains → results analyzed → registered → serving predictions | ✅ done 2026-08-12 — 191 agentic tests green. Live on splice-site **acceptor** (complete 2-epoch run, ~7 min, chosen over MRL's ~6 h): profiled → validated config → approval gate showing the exact command → trained → analyzed **test/f1 96.71, inside the 95.8–97.5 band** → approval → registered as `splice_site_acceptor` → donor **0.010** vs acceptor **0.998** on a real Danio acceptor window from **one** loaded backbone |
| 6 | ToolSmith + Verifier | Codegen for new task types (three files) and external wrappers; verification pipeline (checklists, CPU structural tests, sandboxed smoke tests); bounded loop; human diff approval; custom-task import seam | A new task type becomes a working, registered tool without hand-written code | ✅ done 2026-08-12 — `splice_simple` generated, verified (7/7 harness checks, first attempt), landed and trained; 240 agentic tests green |
| 7 | Hardening | Persistence polish (job history, provenance); error recovery (crashed runs, partial registrations, orphaned staging); scenario eval suite; user guide | Failure paths behave as documented; eval scenarios green | ✅ done 2026-08-12 — 297 agentic tests green; chaos drill passed (missing artifact, recycled PID, orphaned stage, prune guards); doctor clean afterwards |
| 8 | Service API | FastAPI app over the same graph + checkpointer: streaming chat (SSE), toolhub endpoints, job endpoints | Terminal and HTTP clients share sessions; the Phase-4 demo works over HTTP | ✅ done 2026-08-13 — 348 agentic tests green; live: donor window scored 0.9998 over SSE, sessions crossed terminal↔HTTP both ways, approval gate refused and then launched a run whose progress streamed back |
| 9 | Web UI | `ui/` frontend: chat, tool dashboard, live training monitor, approval dialogs. **Stack: plain ES modules served by the API, no build step** (§10) | The Phase-5 scenario driven end-to-end from the browser | ✅ done 2026-08-13 — 381 agentic tests green + 11 opt-in browser tests; live in Chromium against the real install: profiled `dod_data/` → validated config → **approval modal byte-identical to the terminal's** (621 chars, verified against `_prompt_approval` itself) → approved → monitor updated **9 times in place** (epoch 0/step 49 → epoch 4/step 200, val/f1 0.923 → 0.970) → analysis rendered `ok` + truncated caveat → real 400 nt windows scored **0.9998** / **0.0047**; sessions crossed browser↔terminal both ways; zero JS errors throughout |
| 10 | Session rail + user-owned tool state | Tool activation moves behind the approval gate; the thinking indicator tells the truth; sessions get a resizable, collapsible rail with create / rename / delete, and the API a matching session-management surface | A disabled tool cannot be re-enabled without the user's approval in either front end; the indicator is dark at rest; sessions are managed from the rail | ✅ done 2026-08-13 — 405 agentic tests green (from 381) + 15 opt-in browser tests; live: `vienna_fold` disabled → the assistant **reported it and stopped** rather than calling `activate_tool` at all → asked to proceed, the gate suspended with `tools.json` still `"disabled"` at the interrupt → **declined, and it stayed disabled** → approved, flipped, folded `GGGGAAAACCCC` to `((((....))))` @ −5.4; terminal prompt field-identical to the modal; rail created/renamed/deleted/filtered/resized with zero JS errors; sessions crossed browser↔terminal both ways |

**Dependencies:** 0 → 1 → 2 → 3 → 4 → 5 → 6, in order. Phase 7 runs continuously from
Phase 4 onward. Phase 8 can start after 4 (it is genuinely useful after 5). Phase 9 needs 8.
Phase 10 needs 9 and is the first phase driven by *using* the product rather than building
it — which is why it is the first to reverse earlier decisions (see its §1).

**Milestone framing:** Phase 4 is the first thing worth showing anyone; Phase 5 is the
user's headline scenario; Phase 6 is the platform's differentiator.

---

## 9. Testing strategy

- **Deterministic services** (registry, JobRunner, profiler, wrappers): ordinary unit tests
  on `nano` backbones and tiny synthetic datasets — no weights, no GPU, no API key, seconds
  on CPU. This is the bulk of the suite.
- **Agent graphs:** agents are kept thin (§3.1), so most behavior is testable below the
  LLM. Graph wiring and gates are tested with scripted/mocked model responses; a small
  scenario eval suite (recorded conversations with expected tool-call sequences) guards the
  prompts against regression.
- **End-to-end GPU tests** (a real miniature fine-tune → register → predict round trip):
  written, marked, documented, user-run — the engine's `gpu`/`weights`/`data` marker
  convention extends to the agentic suite.

---

## 10. Open decisions

Tracked here; each is decided in the detailed plan of the phase that first needs it.

| Decision | Options on the table | Decide in |
|---|---|---|
| Manifest store | **Decided (Phase 2):** JSON file (`toolhub_data/tools.json`, atomic writes, versioned) — single user/process until Phase 8; human-diffable. Phase 7 added optimistic concurrency (an in-file revision counter) so a second writer is refused rather than silently clobbering | revisit the store choice at Phase 8 rather than adding more locking |
| Where generated code lands | **Decided (Phase 6):** repo-root `adaptrna_custom/`, git-tracked — generated code is a reviewable deliverable, not runtime state | — |
| Sandbox depth for generated code | **Decided (Phase 6):** subprocess + rlimits + timeout. Accident-isolation, not adversarial sandboxing; the human diff gate is the real boundary, containers the upgrade path | revisit if generated code ever comes from an untrusted source |
| Model per agent role | **Decided (Phase 0, revisited Phase 4):** all roles default `anthropic:claude-opus-5`; per-role env overrides (`ADAPTRNA_MODEL`, `ADAPTRNA_MODEL_<ROLE>`) are the cost lever. Phase 4 kept opus-5 for the orchestrator MVP (quality-first while flows are new) | re-evaluate with real usage data at Phase 8 |
| Backbone load policy | **Decided (Phase 2):** lazy — registry ops never load; first forward-pass call does; explicit `warmup`/`rebuild` ops | Phase 4's chat may call `warmup` eagerly at startup |
| UI stack | **Decided (Phase 9) — neither default: plain ES modules served by the API, no build step.** Node 20 was available, so React was ruled out on fit rather than feasibility: the client is a renderer of server state (the checkpointer, manifest and job store hold everything), so a second dependency ecosystem in a 164-file Python repo bought nothing for four views. Gradio was worse — it wants to own the app loop, so the gate and monitor would have been reimplemented a third time. Tripwire: past ~1000 client lines, a second maintainer, or real client-side state, port to React; the API is the contract, so that costs the client alone | **Re-opened: Phase 10 trips the tripwire** — the session rail adds genuine client-side state (session list, filter, persisted geometry) on top of a client already at ~1,100 lines. Phase 10 does not port; it requires the reassessment be written down rather than assumed |

---

## 11. How to use this plan

1. Before starting a phase, write its detailed plan (files, interfaces, tests, commands) —
   this document stays the map, not the blueprint.
2. Update the status column in §8 as phases land; record any decision from §10 the moment
   it is made, with a one-line rationale.
3. If a phase reveals that a principle in §3 or a constraint in §7 is wrong, change this
   document first, then the code.

**Status (2026-08-13):** all ten phases (0–9) are landed and every §10 decision is
resolved. The roadmap in this document is complete: engine untouched at 135 tests, the
agent platform at 381 deterministic tests plus 11 opt-in browser tests, and the same
system reachable from a terminal REPL, an HTTP API and a browser — sharing one
checkpointer, one backbone and one set of approval gates.

**Where to take it next**, in rough order of value:

1. **Use it on a real question.** Everything after this point should be driven by what
   breaks when the platform meets a problem nobody designed it around.
2. **A second backbone.** The naming is model-agnostic and the manifest records which
   checkpoint a tool was built against, but nothing has ever served two. That is the
   assumption most worth testing.
3. **Half-precision serving**, if throughput starts to matter — §7 records what blocks it.
4. **Multi-user**, if it is ever more than one researcher: today's posture (loopback, one
   token, no delete surface) is a deliberate single-user design, not a starting point for
   a shared deployment.
