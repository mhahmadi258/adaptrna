# Phase 5 — Fine-tuning pipeline (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 5; user flow **C** in §5; knowledge
> base in §6; engine constraints in §7.
> **Definition of done:** the MRL scenario works from chat — data in → recommended config
> → approval → LoRA run trains → results analyzed → registered → serving predictions.
> Status: planned · not started

---

## 1. Context and goal

This is the platform's headline scenario and the first phase where the agent does
something **consequential**: it launches GPU training and writes new tools into the hub.
Everything it needs below the LLM now exists — the ToolHub registry and runtime
(Phase 2), the external-tool pattern (Phase 3), the orchestrator with a SQLite
checkpointer (Phase 4, the prerequisite for `interrupt()` gates).

Phase 5 adds five deterministic services and wires them in as agent tools:
**DataProfiler** (what is this data?), **ConfigRecommender** (what should we train, and
why?), **JobRunner** (launch and track the engine CLI on the local GPU), **RunAnalyzer**
(did it work?), and the **registration step** (reusing Phase 2's `Registry.register`).
The LLM narrates, asks, and sequences; it never invents a hyperparameter and never starts
a run without an approval gate.

Verified at plan time (2026-08-12): both H200s are idle; the MRL file the engine reads
(`GSM4084997_varying_length_25to100.csv.gz`, columns `utr`/`rl`/`set`/`total_reads`) is on
disk; and — the basis for the DoD — the **splice-site acceptor** data is complete: 19,961
training rows in `train_data/GS_1/db_1/Train_acceptor_400.csv` (2,218 validation) plus the
Danio acceptor benchmark. The existing donor LoRA run on this machine took **6 min 44 s**
end to end, so an acceptor run is a *complete, reference-comparable* ~7-minute job.
**The DoD is executable end to end on this machine.**

**Out of scope:** ToolSmith/Verifier codegen for *new task types* (Phase 6 — flow D), the
external-tool creation flow (Phase 6 — flow E), retention/cleanup policies (Phase 7), the
HTTP API (Phase 8).

## 2. Scope of "the user provides data" (the honest boundary)

Phase 5 handles data that **a shipped engine task can already read**:

| case | Phase 5 behavior |
|---|---|
| Canonical dataset already on disk (the MRL CSV, the splice folds) | profile it, point `data.root` at it, train |
| Canonical dataset not yet downloaded | the plan carries `--prepare_data`; the download (MRL: 431 MB) is disclosed in the approval payload |
| The user's own data **in a shipped task's documented on-disk layout** | same as above — the profiler validates the layout and says exactly which columns/files it found |
| The user's own data in an **arbitrary schema** | the profiler reports the shape it sees, names the nearest task template, and states plainly that a new datamodule is needed → **that is flow D, Phase 6** |

This is exactly the master plan's flow C / flow D split. The profiler's job at the
boundary is to be *clear*, not to improvise a converter: silently reshaping a user's CSV
into a hardcoded upstream filename is the kind of thing that produces confidently wrong
numbers later.

## 3. Decisions this plan fixes

| Decision | Choice | Rationale |
|---|---|---|
| **Recommendation logic** | **Deterministic**, table-driven from `knowledge/*.yaml`. The LLM explains the recommendation and handles the conversation; it never chooses hyperparameters | MASTER_PLAN §3.1. The validated numbers exist precisely because guessing them is how these runs die (§6 failure modes) |
| **Job execution** | Detached background subprocess; the chat never blocks. Job records in `jobs_data/jobs.json`, run artifacts in `outputs/<run_name>/` (the engine's own convention) | A real MRL run is ~6 h. Status is derived from files (`metrics.csv`, exit-code file) + PID liveness, so it survives a chat restart |
| **Launch path** | Always through `adaptrna_agentic.jobs.train_entrypoint`, a thin wrapper that imports registered custom-task packages and then delegates to `rinalmo_hub.cli.train.main()` | The MASTER_PLAN §3.4 seam, built now so Phase 6's generated tasks need **no** JobRunner change |
| **Approval gates** | A dedicated `approval` **node** in the graph (not an in-tool interrupt): it calls `interrupt()` once per turn for pending gated calls and records decisions in state; the tools node honors them | A node containing only `interrupt()` is idempotent on resume. Interrupting *inside* the tools node would re-execute already-completed tool calls when the turn resumes — a real footgun |
| **Gated operations** | `start_training` and `register_trained_adapter` | Both are consequential and hard to reverse (GPU hours; a new servable tool). Everything else — profiling, recommending, status, analysis — is read-only and ungated |
| **Concurrency** | One training job at a time by default; a second `start_training` is refused naming the running job (override flag exists) | Two giga runs on one GPU is how you get an OOM 40 minutes in |
| **Approval payload** | Carries the **fully materialized command line**, the output dir, the estimated wall-clock, and any download size — the same "Would run: …" discipline as Phase 3's install gate | The human approves what will actually execute, not a summary of it |
| **DoD training task** | **`splice_site`, acceptor arm** — a *complete* 2-epoch LoRA run (~7 min), not a truncated one | A finished run exercises the analyzer's reference-comparison path for real, and yields an adapter genuinely worth keeping: registered next to the existing donor tool, it also proves the "several adapters of one task, one loaded backbone, switched by name" claim (MASTER_PLAN §4) that has never been demonstrated live. MRL (~6 h) remains the documented long-run path |
| **Smoke runs** | The `quick_run` option (engine's own `trainer.max_steps`) stays as a capability for long tasks like MRL, with the analyzer **marking truncated runs as not comparable** to reference metrics — but the DoD no longer depends on it | Keeps a truncated run from ever being read as a result, while removing that caveat from the demo path entirely |

## 4. Components

```
agentic/adaptrna_agentic/
├── knowledge/
│   ├── __init__.py            # load_knowledge() -> dict (cached YAML load)
│   ├── hyperparameters.yaml   # validated settings, failure modes, references, caveats
│   └── task_templates.yaml    # data profile -> task shape mapping
├── profiling/
│   ├── profiler.py            # profile_dataset(path) -> profile dict
│   └── recommender.py         # recommend(profile, task=None, quick=False) -> plan dict
├── jobs/
│   ├── store.py               # JobStore: jobs_data/jobs.json (atomic, like the manifest)
│   ├── runner.py              # JobRunner: start / status / logs / cancel / list
│   ├── train_entrypoint.py    # imports custom tasks, delegates to the engine CLI
│   └── analysis.py            # analyze_run(output_dir) -> verdict report
└── agents/
    ├── tool_factory.py        # + 7 pipeline tools; GATED_TOOLS marker
    └── orchestrator.py        # + approval node, OrchestratorState.approvals
```

### 4.1 `knowledge/` — what grounds the recommendation

`hyperparameters.yaml` carries, per arm and per task, the numbers *and their failure
modes* (MASTER_PLAN §6): LoRA `lr 3e-4` + `gradient_clip_val 1.0` + `layer_stride 3`
(with the `1e-3` collapse story — trains to loss 0.126 by step ~325, then a gradient
spike drives it to constant output it never escapes); full FT `lr 1e-5` or head-only
warm-up (`1e-4` on an unfrozen backbone → R² ≈ 0); `bf16-mixed` always.

Per task: `primary_metric`, reference metrics as a **band** rather than a point, wall-clock
references, and caveats. For `splice_site` the band spans the published donor LoRA F1
(95.82) and this repo's own donor run (97.48 on Danio/db_1), with a **±1 F1 tolerance**
because FlashAttention's backward is non-deterministic — the same command and seed
produced 95.21 and 95.82 (MASTER_PLAN §7); wall-clock ≈7 min (measured). For `mrl`:
test R² 0.8268 (LoRA) / 0.8187 (full FT), 5h45m / 6h41m at batch 64 over 51 epochs, and
the caveat that `val_split: random7600` doubles as a reported result — safe **only**
because nothing selects on validation, so the recommender must never propose early
stopping without switching to `holdout`.

`task_templates.yaml` maps a data profile to a task shape: continuous target + short
RNA (≤100 nt) → `mrl`; binary label + ~400 nt windows → `splice_site`; structure
targets → `sec_struct`; each with the head/loss/metrics it implies and the
`extract_features` pattern, so Phase 6 can reuse the same table for generation.

A test asserts the load-bearing numbers are present, so a careless edit that drops them
fails loudly.

### 4.2 `profiling/profiler.py`

`profile_dataset(path)` → format (CSV/TSV/gzip/FASTA/dir), row count, detected sequence
column (by alphabet + name heuristics), alphabet (RNA/DNA/other), length stats
(min/median/max), candidate target column(s), target type (continuous / binary /
multiclass, by dtype + cardinality), class balance or target distribution, missing
values, and a `layout_match` field naming the shipped task whose on-disk layout it
satisfies (or `null` with the reason). Pure pandas; no engine import, no LLM.

### 4.3 `profiling/recommender.py`

`recommend(profile, task=None, quick=False, seed=42)` → a **plan dict**:

```python
{
  "task": "mrl", "arm": "lora",
  "config_path": "engine/configs/tasks/mrl.yaml",
  "overrides": {"data.root": "<abs>", "optim.lr": 3e-4,
                "trainer.gradient_clip_val": 1.0, "trainer.max_steps": 200, ...},
  "seed": 42, "output_dir": "outputs/mrl_lora_20260812_1530",
  "command": ["<python>", "-m", "adaptrna_agentic.jobs.train_entrypoint", "--task", "mrl", ...],
  "rationale": ["LoRA at 3e-4 with clip 1.0 — validated; 1e-3 collapses irrecoverably", ...],
  "estimated_wall_clock": "~4 min (truncated: 200 steps) | ~5h45m full (50 epochs)",
  "warnings": ["val_split=random7600 is also a reported result — safe here because "
               "nothing selects on validation"],
  "quick_run": True,
}
```

Every number traces to `knowledge/`; `rationale` is generated from the same table, so the
model's explanation and the actual config cannot drift apart.

### 4.4 `jobs/` — runner, store, entrypoint

- **`train_entrypoint.py`**: imports any registered custom-task packages (none yet;
  Phase 6 fills this), then `sys.exit(rinalmo_hub.cli.train.main())`. Writes
  `<output_dir>/exit_code` on completion so status survives a lost PID.
- **`store.py`**: job records — `{id, task, arm, command, output_dir, adapter_path, pid,
  state (running|succeeded|failed|cancelled), started_at, ended_at, plan}` — atomic JSON
  writes, same pattern as the ToolHub manifest.
- **`runner.py`**: `start(plan)` (refuses when another job is running), `status(job_id)`
  (parses `metrics.csv` for the latest epoch/step and metrics; PID liveness + exit-code
  file for state), `logs(job_id, tail)`, `cancel(job_id)` (SIGTERM), `list()`.
  Subprocess is detached (`start_new_session=True`), stdout/stderr → `<output_dir>/train.log`.

### 4.5 `jobs/analysis.py`

`analyze_run(output_dir)` → `{verdict: ok|suspicious|failed, metrics, checks[],
comparison, suggestions[], truncated: bool}`. Deterministic checks, each mapped to a
knowledge-base failure mode:

| check | fires when | mapped remedy |
|---|---|---|
| loss NaN/inf | any NaN in the loss columns | lower lr; check data |
| collapse | training loss flat after an early drop, or val metric variance ≈ 0 | the LoRA `1e-3` collapse story → retry at 3e-4 with clip 1.0 |
| destroyed backbone | R² ≤ 0 (regression) on a full-FT run | full-FT `1e-4` failure mode → 1e-5 or head-only warm-up |
| no improvement | final ≈ initial on the primary metric | budget too small / lr too low |
| below reference | primary metric falls outside the task+arm reference **band**, beyond the per-task noise tolerance (splice site: ±1 F1, for FlashAttention's non-deterministic backward) | report the gap; **suppressed when `truncated`**. Never flag a within-tolerance difference — MASTER_PLAN §7 |
| truncated | `max_steps` cut the run short of `max_epochs` | "smoke run — not comparable to reference metrics" |

### 4.6 Agent tools and the approval gate

New tools: `profile_dataset`, `recommend_training_config`, `start_training` **(gated)**,
`job_status`, `list_jobs`, `analyze_run`, `register_trained_adapter` **(gated)**.

Graph change in `orchestrator.py`: state gains `approvals: dict[tool_call_id, decision]`;
routing becomes model → (`approval` if any pending gated call) → `tools` → model. The
`approval` node calls `interrupt({"type": "approval_request", "requests": [...]})`; the
CLI renders the request (command, output dir, ETA, download size), asks `[y/N]` with an
optional note, and resumes with `Command(resume=...)`. Denied calls become a ToolMessage
("the user declined: …") so the model can respond rather than silently stalling. The
exact interrupt/resume API surface is confirmed against the installed LangGraph 1.2 at
implementation time (same discipline as Phase 0's import-path check).

## 5. Tests (deterministic — no GPU, no real training, no API key)

| test file | asserts |
|---|---|
| `test_knowledge.py` | YAML loads; load-bearing values present (LoRA 3e-4 / clip 1.0 / stride 3, full-FT 1e-5, bf16-mixed, MRL reference R² and the random7600 caveat); every task template names head/loss/metrics |
| `test_profiler.py` | synthetic CSV/gz fixtures: MRL-shaped (continuous target, ≤100 nt) → `layout_match: mrl`; splice-shaped (binary, 400 nt) → `splice_site`; unknown schema → `layout_match: null` **with a reason naming the nearest template**; sequence-column and alphabet detection; length stats; missing-value reporting |
| `test_recommender.py` | splice-site profile → task `splice_site`, arm `lora`, lr 3e-4, clip 1.0, stride 3, and `data.ss_type` honored when the user asks for the acceptor arm; MRL profile → task `mrl` with the `random7600` warning emitted; every recommended number appears in the knowledge base (no free-floating constants); `quick=True` adds `trainer.max_steps` and flags `quick_run`; the materialized `command` round-trips through the engine's own argument parser (`rinalmo_hub.cli.train.build_parser().parse_args`) — the strongest cheap guarantee the plan is executable; early stopping is never proposed |
| `test_job_runner.py` | start/status/complete against a **fake command** (a short `python -c` that writes a synthetic `metrics.csv` and exits 0): state transitions, log capture, exit-code file, `metrics.csv` progress parsing, failure path (non-zero exit → `failed`), cancel, concurrency refusal naming the running job, store round-trip across a fresh `JobStore` |
| `test_analysis.py` | synthetic `metrics.csv` fixtures → verdicts: healthy splice-site run at F1 96.5 (`ok`, inside the band); **a 1-point difference is *not* flagged** (non-determinism tolerance); F1 60 (`suspicious`, outside the band); R²≈0 full-FT (`failed` + the 1e-5 remedy); NaN loss (`failed`); collapsed LoRA (`suspicious` + the 3e-4/clip remedy); truncated run (`truncated: True`, below-reference check suppressed) |
| `test_approval_gate.py` | scripted model + fake resume: a gated call **interrupts before executing** (assert no job was started); resume-approve executes it; resume-deny produces the declined ToolMessage and no side effect; ungated calls never interrupt; approvals persist across the interrupt via the checkpointer |
| `test_pipeline_tools.py` | the 7 tools are built and bound; `start_training` is in `GATED_TOOLS`; `register_trained_adapter` on a finished fake job creates a manifest entry whose provenance carries the job id and training metrics |

Expected ~39 new tests (total ~150). Phase 0–4's 111 stay green; engine's 135 untouched.

## 6. Implementation order

1. `knowledge/` YAML + loader + `test_knowledge.py`.
2. `profiling/profiler.py` + tests.
3. `profiling/recommender.py` + tests (incl. the engine-argparse round-trip).
4. `jobs/train_entrypoint.py`, `store.py`, `runner.py` + tests (fake command).
5. `jobs/analysis.py` + tests (synthetic metrics fixtures).
6. Approval node + state change in `orchestrator.py`; the 7 tools in `tool_factory.py`;
   CLI interrupt rendering/resume in `cli/chat.py` + tests.
7. DoD run (§7); `.gitignore` += `jobs_data/`; README sections.
8. Close-out: MASTER_PLAN §8 tick; record any §7-constraint discoveries.

## 7. Verification / definition of done

**Gate 1 — deterministic:** `cd agentic && pytest` all green (111 + ~39); `cd engine &&
pytest` → 135; no engine changes in `git status`.

**Gate 2 — the live scenario: a complete splice-site *acceptor* adapter** (one chat
session, ~7 minutes of GPU). The hub already serves a donor adapter, so this both closes
the create-a-tool loop and proves multi-adapter residency:

```
python -m adaptrna_agentic.cli.chat --session acceptor

you> I want an acceptor splice-site model. My Spliceator data is at
     ~/bio2/RiNALMo/dataset/train_data — what's in it?
#   → profile_dataset: 10 stratified folds (db_1…db_10), binary labels, 400 nt windows,
#     19,961 acceptor training rows in db_1; layout matches the splice_site task

you> Recommend a fine-tuning setup for the acceptor arm.
#   → recommend_training_config: splice_site / LoRA, lr 3e-4, clip 1.0, stride 3,
#     data.ss_type=acceptor, 2 epochs, ETA ~7 min, rationale from the knowledge base

you> Run it.
#   → APPROVAL GATE: exact command + output dir + ETA → [y] → detached job started
#     (the chat stays responsive throughout)

you> How's it going?
#   → job_status: epoch/step, train loss, elapsed

…~7 minutes later…

you> Analyze the run.
#   → analyze_run: test/f1_score against the reference band (95.8–97.5, ±1 F1 for
#     FlashAttention non-determinism) → verdict ok, with the numbers   ← real comparison

you> Register it as splice_site_acceptor.
#   → APPROVAL GATE → registered; provenance carries the job id and final metrics

you> Score this window with both the donor and the acceptor tools: <400 nt Danio window>
#   → both adapters answer from ONE loaded backbone, switched by name   ← loop closed,
#     and multi-adapter residency (MASTER_PLAN §4) demonstrated live
```

No cleanup step: unlike a truncated smoke run, `splice_site_acceptor` is a genuine,
fully-trained tool that belongs in the hub next to the donor one.

**Gate 3 — the long-run path stays one command:** the same flow on the MRL data produces
the full 50-epoch plan (and `quick_run` produces a ~4-minute truncated variant, flagged
as not comparable). Launching the full MRL run is documented as user-run: ~5h45m,
expected test R² ≈ 0.83 per the knowledge base.

## 8. Risks and notes

- **Long runs vs. a live demo** — resolved by choosing the task: splice-site acceptor
  finishes completely in ~7 min, so the demo needs no truncation caveat at all. For
  genuinely long tasks (MRL, ~6 h) `quick_run` + the analyzer's `truncated` flag remain,
  because the failure mode to avoid is a truncated run being read as a result.
- **Reference comparison is now load-bearing** (the demo compares a real run to a band):
  the band must stay a *band* with a stated tolerance. Two identical splice-site commands
  with the same seed produced F1 95.21 and 95.82 — the analyzer must never report that as
  a regression.
- **Detached jobs outlive the chat** (by design). Status must therefore be derivable from
  disk alone — hence the exit-code file and `metrics.csv` parsing rather than a live
  handle.
- **GPU contention** — single-job default; a second request is refused with the running
  job's id and ETA.
- **MRL's `random7600` caveat** (MASTER_PLAN §7) — the recommender surfaces it as a
  warning and never proposes early stopping or best-checkpoint selection without
  switching to `val_split=holdout`. This is the one place the pipeline could quietly
  contaminate a headline metric.
- **Absolute data paths** — the plan records absolute `data.root` (matching the existing
  `outputs/*/resolved_config.yaml` convention) so a job is reproducible regardless of CWD.
- **`num_workers`** — the shipped MRL YAML uses 32; quick runs lower it to keep startup
  from dominating a 4-minute demo.
- **Interrupt semantics** — the approval node must contain *only* the interrupt; any tool
  execution inside it would re-run on resume. The test suite pins this (no job started
  before approval).
