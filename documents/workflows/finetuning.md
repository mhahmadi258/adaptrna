# Fine-tuning an Adapter Tool from One CSV (flow C)

One CSV in → your approved interpretation → a built and verified task → your approval → a
validated training plan → your approval → a GPU run → an analysed result → your approval →
a servable tool.

Four steps, four separate gates. Everything in it is deterministic except the narration, and
approval at one gate never implies the next — each step is its own request.

---

## Contents

1. [The conversation](#the-conversation)
2. [Step 1 — profile](#step-1--profile)
3. [Step 2 — build](#step-2--build)
4. [Step 3 — recommend and train](#step-3--recommend-and-train)
5. [Step 4 — analyse and serve](#step-4--analyse-and-serve)
6. [Driving it from the browser](#driving-it-from-the-browser)
7. [Doing it without the agent](#doing-it-without-the-agent)
8. [What can go wrong](#what-can-go-wrong)

---

## The conversation

```
you> Profile ~/data/my_data.csv.
                                    ← gate 1: columns, target type, split, warnings — edit or approve
you> Build the task.
                                    ← gate 2: the code (rendered or generated), harness result — approve
you> Recommend a training config and run it.
                                    ← gate 3: the exact command, editable hyperparameters — approve
you> How's it going?               ← the job runs detached; the chat stays responsive
you> Analyze the run.
you> Register it as my_task.       ← gate 4
```

```mermaid
sequenceDiagram
    autonumber
    participant U as You
    participant O as Orchestrator
    participant S as Deterministic services
    participant G as GPU (detached)

    U->>O: profile my_data.csv
    O->>S: profile_dataset(path)
    S-->>U: ⛔ GATE 1 — confirm_data_profile: columns, target type, split, warnings
    U->>O: approve (optionally with edits)
    O->>S: confirm_data_profile(spec)
    U->>O: build the task
    O->>S: create_task_tool(spec)   [renders, or falls back to generation; harness + review]
    S-->>U: ⛔ GATE 2 — land_generated_code: the files, the diff
    U->>O: approve
    O->>S: land_generated_code(stage_id)
    U->>O: recommend a config and run it
    O->>S: recommend_training_config(task)
    S-->>U: ⛔ GATE 3 — start_training: exact command, editable hyperparameters
    U->>O: approve
    O->>S: start_training(plan)   [refuses an unstamped plan]
    S->>G: detached subprocess
    loop while running
        U->>O: how's it going?
        O->>S: job_status → metrics.csv tail
    end
    U->>O: analyze it
    O->>S: analyze_run(job_id)
    S-->>U: verdict + reasons + remedies
    U->>O: register it
    O-->>U: ⛔ GATE 4 — register_trained_adapter: a new servable tool
    U->>O: approve
    O->>S: register_trained_adapter(job_id, name)
```

Each `⛔` is a separate turn: the orchestrator is instructed to report what a gate produced
and stop, not to chain into the next step on its own (D10, plan §2).

---

## Step 1 — profile

`profile_dataset(path)` →
[`profiling/profiler.py`](../modules/profiling-and-knowledge.md#3-profilerpy--one-table-in-a-datasetspec-out).

Accepts exactly one delimited table — `.csv`, `.tsv`, or either gzipped — with one sequence
column and one label column whose target is binary, multiclass (≤20 classes) or a continuous
regression value. A directory, a FASTA file, multiple files, per-position labels, or more
free-text classes than that are refused by name, with what is supported stated plainly:

```json
{
  "path": "/home/you/data/my_data.csv",
  "sequence_column": "sequence", "label_column": "label",
  "target_type": "binary", "classes": ["0", "1"], "positive_class": "1",
  "length": {"min": 400, "median": 400, "max": 400}, "alphabet": "dna",
  "split": {"mode": "random", "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "row_counts": {"train": 19350, "val": 2419, "test": 2419}},
  "task_name": "my_task",
  "warnings": ["1,204 sequences (5.0%) appear more than once in the file; …"],
  "similar_tasks": []
}
```

`confirm_data_profile` is gate 1 — a real gated tool on the interpretation, not a
conversational "shall I proceed?". The gate shows the file, the chosen columns, class
counts (or the regression target's range), the split's real row counts, and every quality
warning **as computed**, never summarised away:

```
  ┌─ approval required ──────────────────────────────────────────
  │ Use column 'sequence' as the sequence and 'label' as a binary target from my_data.csv
  │ (24,188 rows → 19,350/2,419/2,419 train/val/test), and build task 'my_task'
  │   file:      /home/you/data/my_data.csv  (24,188 rows, 5 columns)
  │   sequence:  sequence          400 nt (min 400, max 400), dna
  │   label:     label             binary — 0: 12,094 · 1: 12,094 (positive: '1')
  │   ignored:   gene_id, source, chrom
  │   split:     random 80/10/10, seed 42, stratified → 19,350 / 2,419 / 2,419
  │   head:      linear classifier on the CLS token · BCE-with-logits · f1_score
  │   task:      my_task
  │   ! 1,204 sequences (5.0%) appear more than once in the file; …
  └──────────────────────────────────────────────────────────────
  edit a field, or approve? [field=value … | y/N]
```

Any field can be corrected before approving — the sequence/label column, `target_type`,
`positive_class`, `task_name`, `tool_description`, or any part of the split (§5, "Approval
decisions that carry edits" in the phase plan). On approval the spec is **re-validated and
recomputed against the file**, not merely accepted: a user who switches `target_type` from
`regression` to `multiclass` gets recomputed `classes`, `class_counts`, `row_counts` and
`head` that actually follow the new choice, never a spec that says one thing and trains
another.

If a similarly-shaped task already exists (`spec["similar_tasks"]`), the gate mentions it as
a suggestion — never a branch. Reusing it means skipping straight to step 3 with the
existing task name; see
[new-task-codegen.md §6](new-task-codegen.md#6-reuse-skipping-this-flow-entirely).

## Step 2 — build

`create_task_tool(spec)` turns the approved spec into three files — a task module, a
datamodule, a config — plus a `spec.json` sidecar, and stages them for review.
**The common case renders them deterministically from a template, with no model call**;
generation is the fallback for a spec the template does not cover. Full explanation of both
paths: [new-task-codegen.md](new-task-codegen.md).

`land_generated_code` is gate 2 — file list, line counts, and the full diff, exactly as
today. The tool result and the gate both say **which path produced the code**
(`"path": "template"` or `"path": "generated"`), so you are never unsure whether a human
wrote the logic you are approving.

```
you> Build the task.
  → create_task_tool({spec: {...}})
    = {"ok": true, "path": "template", "iterations": 1,
       "harness": "task 'my_task': PASS\n  [ok  ] import: …",
       "stage_id": "my_task-ab12cd34", "files": [...]}

you> Land it.
                                     ← gate 2: file list, line counts, staging path, diff
```

The task is discovered automatically on next use — no restart, no separate registration
step. Its `spec.json` is what makes step 3's derived hyperparameters and reuse (§9 of the
phase plan) possible after the session ends.

## Step 3 — recommend and train

`recommend_training_config(task, spec=None, arm="lora", quick=False, seed=42)` →
[`profiling/recommender.py`](../modules/profiling-and-knowledge.md#7-recommenderpy--the-plan).

Every hyperparameter comes from
[`knowledge/hyperparameters.yaml`](../configuration.md#4-the-knowledge-base) — the *arm*
settings verbatim, batch size / epoch count / worker count **derived from the approved
spec** the task landed with — and so does every rationale line, so what you read and what
runs cannot drift apart. There is **no reference metric band** for a task this build has
never seen before (`plan["reference"]["band"]` is always `null`); that is a real, permanent
state, not something waiting to be filled in.

```python
{"source": "recommend_training_config",
 "task": "my_task", "arm": "lora",
 "config_path": "adaptrna_custom/tasks/my_task/config.yaml",
 "overrides": {"lm_config": "giga", "pretrained_weights": "/home/you/.cache/…/giga-v1.pt",
               "data.root": "/home/you/data/my_data.csv",
               "optim.lr": 0.0003, "optim.name": "adamw",
               "trainer.gradient_clip_val": 1.0, "trainer.precision": "bf16-mixed",
               "lora.r": 16, "lora.alpha": 32, "lora.dropout": 0.05, "lora.layer_stride": 3,
               "data.batch_size": 32, "trainer.max_epochs": 6, "data.num_workers": 8},
 "seed": 42,
 "output_dir": "outputs/my_task_lora_20260818_101810",
 "primary_metric": "test/f1_score",
 "reference": {"band": null, "tolerance": 0.0,
               "sources": ["No validated reference run: this task did not exist before "
                           "your data."]},
 "estimated_wall_clock": "unknown — no run of this task exists yet",
 "rationale": ["lr 3e-4 with gradient_clip_val 1.0 was stable across tasks…",
               "data.batch_size 32: activation memory on the giga backbone scales with "
               "sequence length; measured on this machine…",
               "trainer.max_epochs 6: 19,350 training rows at batch 32 is 605 steps per "
               "epoch; 6 epochs ≈ 3,630 optimiser steps, inside the 1k-10k budget."],
 "warnings": ["This task has no reference band. The first successful run becomes the "
              "baseline that later runs of the same task are compared against."],
 "command": ["/…/python", "-m", "adaptrna_agentic.jobs.train_entrypoint", "--task", …]}
```

### Where the backbone comes from

Not from `configs/base.yaml` (whose `weights/giga-v1.pt` default is relative to the working
directory and need not exist) but from **the ToolHub manifest**, because an adapter trained
against a different backbone than the hub serves could not be hosted alongside the existing
tools. A hub with no checkpoint configured produces a warning, not a silent random-weights
run.

### The approval gate

```
  ┌─ approval required ──────────────────────────────────────────
  │ Train my_task (lora) — ETA unknown, output outputs/my_task_lora_20260818_101810
  │   would run: /home/you/adaptrna/.venv/bin/python -m adaptrna_agentic.jobs.train_entrypoint
  │              --task my_task --config adaptrna_custom/tasks/my_task/config.yaml
  │              --output_dir outputs/my_task_lora_20260818_101810 --seed 42
  │              --use_lora --set lm_config=giga --set pretrained_weights=/… --set …
  │   output:    outputs/my_task_lora_20260818_101810
  │   ! This task has no reference band. The first successful run becomes the baseline …
  └──────────────────────────────────────────────────────────────
  edit a field, or approve? [field=value … | y/N]
```

The command is shown **verbatim** — byte for byte what will be executed. You may edit
`overrides.*`, `seed`, `arm` or `quick_run` before approving; an edit **rebuilds the
command** so the gate never shows one thing and runs another, records
`plan["human_overrides"]`, and — if your chosen value matches a recorded failure mode in the
knowledge base — appends a warning naming it, e.g. *"optim.lr = 1e-3 is a recorded failure
mode: trains well to loss 0.126 by step ~325, then a gradient spike collapses it into a
constant-output state."* The gate re-renders after an edit and asks again.

`start_training` then applies the rule that makes all of this meaningful:

```python
if plan.get("source") != "recommend_training_config":
    raise ToolHubError("This plan did not come from recommend_training_config. …")
```

so a model that hand-assembles a plan — which happens when the recommender refuses
something — is rejected rather than trusted. A human override is a different actor with
different authority: the stamp check is about the model, not about you.

Training then runs the same way it always has:
[`jobs/runner.py`](../modules/jobs.md) launches it **detached**
(`start_new_session=True`), so it survives the chat exiting, with the run directory, live
metrics tail, and job-status/cancel surfaces unchanged from before this phase:

```
outputs/<run_name>/
├── resolved_config.yaml     written before training starts
├── metrics/version_0/metrics.csv   appended as it goes — tail -f it
├── run_summary.json         final metrics + per-stage GPU memory / iteration time
├── train.log                stdout + stderr
├── exit_code                written by the entrypoint in a `finally` block
└── my_task_adapter.pt       the ~6 MB artifact
```

```
you> How's it going?
  → job_status({job_id: "my_task_lora_20260818_101810"})
    = {"state": "running", "progress": {"epoch": 1, "step": 400,
       "latest_metrics": {"train/loss": 0.08, "val/f1_score": 96.4}}}
```

## Step 4 — analyse and serve

`analyze_run(job_id)` → [`jobs/analysis.py`](../modules/jobs.md#7-analysispy--the-runanalyzer).

```python
{"task": "my_task", "arm": "lora", "truncated": False,
 "metrics": {"train/loss": 0.031, "val/f1_score": 96.9, "test/f1_score": 96.71, …},
 "primary_metric": "test/f1_score", "primary_value": 96.71,
 "checks": ["test/f1_score = 96.71 — no reference band for this task; treat this run as "
            "the baseline"],
 "suggestions": [],
 "verdict": "baseline"}
```

**A run truncated by `trainer.max_steps` is never compared to anything** — the report says
so explicitly, and the orchestrator is instructed never to present a truncated smoke run as
a real result. **Every run of a task this build has never seen before is a baseline**, since
`generic.reference.band` is always `null` (§8 of
[profiling-and-knowledge.md](../modules/profiling-and-knowledge.md#8-the-two-stamps)) — the
first successful run is what later runs of the same task get compared against, and the
report says so rather than judging it against someone else's number.

```
you> Register it as my_task.

  ┌─ approval required ──────────────────────────────────────────
  │ Register job 'my_task_lora_20260818_101810' as the servable tool 'my_task'
  └──────────────────────────────────────────────────────────────
  approve? [y/N] y
```

`register_trained_adapter(job_id, name=None, description=None)` refuses a job that is not
`succeeded`, and one that produced no adapter file (*"LoRA runs write one; full fine-tuning
runs do not"*). On success it copies the artifact into `toolhub_data/adapters/`, writes the
manifest entry — including the landed `spec.json` into `provenance["spec"]`, which is what
lets the tool describe its own output and validate its own predictions
([toolhub.md](../modules/toolhub.md)) — and stamps `provenance.job_id` and
`provenance.training_metrics`.

The tool becomes resident on next use — no restart needed:

```
you> Is this sequence positive?
  → my_task({sequences: […]})   = [0.998]
```

## Driving it from the browser

```bash
python -m adaptrna_agentic.cli.serve --open
```

The same four steps, with each gate as a modal showing the same exact command or diff, and
a form for editing spec fields (gate 1) or overrides (gate 3) before approving. The Jobs
panel updates in place while a run proceeds. Sessions are shared: start in the terminal,
continue in the browser, and back.

A refresh mid-approval restores the dialog from `GET /api/sessions/{id}/history`'s
`pending_approval` field — the suspended turn is in the checkpointer, not in the tab.

## Doing it without the agent

Every step has a plain-Python equivalent:

```python
from adaptrna_agentic.profiling.profiler import profile_dataset, confirm_profile
from adaptrna_agentic.profiling.recommender import recommend
from adaptrna_agentic.codegen import pipeline, staging
from adaptrna_agentic.jobs.runner import JobRunner
from adaptrna_agentic.jobs.analysis import analyze_run
from adaptrna_agentic.toolhub.registry import Registry

spec = confirm_profile(profile_dataset("~/data/my_data.csv"))
result = pipeline.create_task(spec)                     # renders, or falls back to generation
staging.land(result.stage)                               # no gate at this level

plan   = recommend("my_task")
record = JobRunner().start(plan)
report = analyze_run(record.output_path, plan=record.plan)
Registry().register(record.adapter_path, name="my_task")
```

The approval gates live in the **graph**, not in these services — a Python caller is assumed
to have made the decision by calling. The `spec["source"]`/`plan["source"]` checks are
likewise in `tool_factory`, not in `pipeline`/`JobRunner`.

## What can go wrong

| Symptom | Meaning | Fix |
|---|---|---|
| `This build trains from a single table (.csv/.tsv, optionally gzipped) …` | Not a single delimited table | Profile a `.csv`/`.tsv` (optionally `.gz`) with one sequence column and one label column |
| Refusal naming binary/multiclass/regression | The label column is free text, per-position, or has too many distinct values | This build supports exactly those three target types — nothing is converted for you |
| `This spec did not come from profile_dataset` / `…confirm_data_profile` | Intentional — the spec must pass through its gate | Call the tool that produced it and pass the result through unchanged |
| `This plan did not come from recommend_training_config` | Intentional. Hyperparameters must come from the knowledge base | Call `recommend_training_config` and pass its result through unchanged |
| `Job '…' is still running` | One training job at a time by default | Wait, or cancel it |
| `The ToolHub's backbone checkpoint '…' does not exist` | The manifest points at a missing file | `toolhub config --weights /path/to/giga-v1.pt` |
| Plan warns about random weights | No checkpoint configured at all | Same fix, before training for real |
| Job goes to `failed` with no `exit_code` | SIGKILL, OOM, or a hard crash | `job logs` / `train.log` — the last lines usually name it |
| `PID … may since have been reused — refusing to signal it` | The process is gone; the record was closed out and **nothing was killed** | None needed |
| Verdict `baseline` | Expected for every task this build has not seen before — there is no reference band to compare against | Not an error; the run itself becomes the reference for the next one |
| Verdict `failed`, primary metric ≈ 0 | Degenerate output | The report carries the arm's documented failure mode and its remedy |
| `is a full fine-tuning export` at registration | Only LoRA adapters can be served | Evaluate it with `rinalmo_hub.cli.evaluate --init_params` instead |
| A crashed run | **Cannot be resumed mid-flight** | Start it again |
