# Operations, Troubleshooting and Recovery

Diagnosing a broken install, reclaiming disk, and recovering from every failure the platform
knows how to describe.

---

## Contents

1. [`doctor` — start here](#1-doctor--start-here)
2. [`prune` — the only destructive command](#2-prune--the-only-destructive-command)
3. [Error message index](#3-error-message-index)
4. [Recovering from specific failures](#4-recovering-from-specific-failures)
5. [Running the service](#5-running-the-service)
6. [Sessions](#6-sessions)
7. [Disk usage](#7-disk-usage)
8. [Known limitations](#8-known-limitations)
9. [Clean slate: resetting to zero tools](#9-clean-slate-resetting-to-zero-tools)

---

## 1. `doctor` — start here

```bash
python -m adaptrna_agentic.cli.toolhub doctor          # human-readable
python -m adaptrna_agentic.cli.toolhub doctor --json   # machine-readable
curl -s localhost:8000/api/doctor | jq                 # over HTTP
curl -s localhost:8000/health | jq                     # status only, no token needed
```

**It changes nothing**, and every failure it reports names the command that fixes it. An
install with one problem looks like this (a healthy one reports `install status: OK` with
every line `[ok  ]`):

```
install status: WARN
  [ok  ] engine: importable from /home/you/adaptrna/engine/rinalmo_hub
  [ok  ] backbone: giga checkpoint at /home/you/.cache/rinalmo_pretrained/giga-v1.pt (2.61 GB)
  [ok  ] artifacts: 5 tool(s), all artifacts present
  [ok  ] external_tools: 2 external tool(s) importable
  [ok  ] custom_tasks: 1 generated task(s) import cleanly
  [ok  ] template_version: current template version: v2
  [ok  ] jobs: 6 job record(s), all consistent
  [WARN] staging: 2 staged artifact(s) never landed or cleaned (14 KB): [...]
           -> review with `list_staged_code`, then land it or `toolhub prune staging`
  [ok  ] disk:outputs: /home/you/adaptrna/outputs is 0.31 GB
```

| Check | FAIL means | WARN means | Remedy it prints |
|---|---|---|---|
| `engine` | `rinalmo_hub` cannot be imported | — | `pip install -e ./engine` |
| `backbone` | the configured checkpoint is gone | none configured at all | `toolhub config --weights /path/to/giga-v1.pt` |
| `artifact:<tool>` | a tool points at a missing file | — | restore it, or `toolhub remove <tool>` |
| `orphan_artifacts` | — | adapter files no tool references | `toolhub prune artifacts` |
| `external_tools` | a package is no longer importable | — | reinstall it, or remove the tool |
| `custom_tasks` | a generated task fails to import | — | fix `adaptrna_custom/tasks/…`, or delete the package |
| `template_version` | — | a landed task's `spec.json` was rendered by a template version older than the current one | not automatic by design (landed code is yours) — review the current template's diff and re-render by hand if you want the fix |
| `stale_jobs` | a record says `running` but the process is gone/recycled | — | ⚠️ the printed remedy names a command that does not exist — see below |
| `job_outputs` | — | a succeeded job's output directory is gone | harmless if pruned deliberately |
| `staging` | — | generated code never landed or cleaned | `list_staged_code`, then land or prune |
| `disk:*` | — | over 5 GB | `toolhub prune runs --older-than 7` |

**Exit code 1** if any check failed, so `doctor` works in a script.

> ⚠️ The `stale_jobs` remedy says *"run `toolhub job-status <id>` to close them out"*. **That
> subcommand does not exist.** Reconcile a stale record by reading its status through any
> path that calls `JobRunner.status()` — `GET /api/jobs/{id}`, the `job_status` tool in chat,
> or:
> ```bash
> python -c "from adaptrna_agentic.jobs.runner import JobRunner; print(JobRunner().status('<id>'))"
> ```
> Reading the status is what performs the reconciliation. See
> [../README.md gap #1](../README.md#known-documentation-gaps).

`doctor` is tested against a purpose-built **broken** install, on the rule that *a health
check that reports green on a broken install is worse than no health check*.

## 2. `prune` — the only destructive command

```bash
$toolhub prune staging|sessions|jobs|runs|artifacts [--older-than DAYS] [--yes]
```

Three rules, in this order:

1. **Never delete anything the manifest references** — a registered tool's artifact, or the
   output directory of the job that produced one. Those are hours of GPU time and reviewed
   code. Referenced items are listed as *kept*, with the reason.
2. **Dry run by default.** `--yes` performs it.
3. **`runs` always requires `--older-than`**, because it deletes training outputs.

```bash
$toolhub prune runs --older-than 7
```
```
prune runs: would remove 2 item(s), 41.3 MB
  - my_task_lora_20260812_223641 (20.6 MB, 8.4d old)
  - my_other_task_lora_20260812_184704 (20.7 MB, 8.4d old)
  · kept my_task_lora_20260813_101810 — produced the registered tool 'my_task'

This was a dry run. Add --yes to perform it.
```

| Target | Removes | Never touches |
|---|---|---|
| `staging` | `toolhub_data/staging/<id>/` | anything younger than the filter |
| `artifacts` | orphaned `.pt` files in `toolhub_data/adapters/` | anything a tool references |
| `sessions` | conversation rows from `sessions.sqlite` | — (see the caveat below) |
| `jobs` | job **records** (not their output directories) | running jobs; jobs that produced a registered tool |
| `runs` | `outputs/<run>/` directories | protected runs; directories a job is still writing to |

Two caveats worth knowing:

* **`sessions --older-than` compares the age of the whole SQLite file**, not of each session
  — it is all-or-nothing per store. The kept-reason string says *"store younger than…"*.
* Pruning `jobs` removes the record; the run's `outputs/` directory is a separate target.

`prune` is deliberately **not** an agent tool: deletion stays a human action at the CLI.

## 3. Error message index

Every message below is produced verbatim by the code, and is the same string in the
terminal, in a tool result, and in an HTTP body.

| Message | Meaning | Fix |
|---|---|---|
| `ANTHROPIC_API_KEY is not set. Export it in your shell or put it in '…/.env'` | No credential | Create `.env` at the repo root |
| `The engine package is not installed in this environment.` | `rinalmo_hub` missing | `pip install -e ./engine` |
| `Backbone weights '…' not found (resolved to '…')` | The manifest points at a missing checkpoint | `toolhub config --weights …` |
| `Tool 'x' points at '…', which does not exist.` | The adapter file is gone | Restore it, or `toolhub remove x`. `doctor` lists every case |
| `Tool 'x' is disabled. Enable it with 'toolhub activate x'.` | Routing-level deactivation | `toolhub activate x`, or ask in chat |
| `Tool 'x' is of type 'external'; only adapter tools run on the backbone runtime.` | Wrong call path | `toolhub call x …` |
| `'x' is already registered (from '…')` | Name collision | Remove it first, or use `--name` |
| `'…' is a full fine-tuning export` | Only the head travels in that file | Evaluate with `rinalmo_hub.cli.evaluate --init_params` |
| `Adapter '…' was trained on the 'giga' backbone, but this ToolHub serves 'nano'` | Sizes are not interchangeable | Register against a matching hub |
| `This plan did not come from recommend_training_config` | **Intentional** — hyperparameters must come from the knowledge base | Call `recommend_training_config` and pass its result unchanged |
| `Job '…' is still running` | One training job at a time | Wait, or cancel |
| `Job '…' is no longer running (… PID N may since have been reused — refusing to signal it)` | The process is gone; the record was closed out and **nothing was killed** | None needed |
| `Job '…' is failed, not succeeded — nothing to register.` | Registration guard | Analyse the run first |
| `Job '…' produced no adapter file.` | A full-FT run | LoRA runs write one; full-FT runs cannot become served tools |
| `'…/tools.json' changed on disk since it was read` | Another process wrote first. **Nothing was lost.** | Retry |
| `Package 'X' (import 'y') is not installed.` | The external-tool install gate | Run the printed command, or `--yes` |
| `A task named 'x' already exists in adaptrna_custom/tasks/` | Never overwrite landed code | Edit it, or choose another name |
| `Refusing to bind to '0.0.0.0' without a token` | The service can spend GPU hours and write code | Set `ADAPTRNA_API_TOKEN`, or bind loopback |
| `the script exceeded its time limit (possible infinite loop)` | The codegen sandbox timed out at 600 s | Usually an eager datamodule |

HTTP mapping: `ConcurrentModificationError` → **409** with `retryable: true`; other
`ToolHubError` → **409**; `KeyError` → **404**; `FileNotFoundError`/`ValueError` → **400**;
anything else → **500** with a `request_id` that matches a server log line.

## 4. Recovering from specific failures

### A training run crashed

```bash
python -c "from adaptrna_agentic.jobs.runner import JobRunner; \
           print(JobRunner().logs('<job_id>', tail=100))"
# or: tail -100 outputs/<job_id>/train.log
```

Reading the status reconciles the record: an `exit_code` file wins; otherwise a dead or
recycled PID marks it `failed`. **A crashed run cannot be resumed mid-flight** — start it
again. The `outputs/` directory stays, so `prune runs` is how you reclaim it.

### A tool's artifact went missing

`doctor` reports `artifact:<tool>` FAIL. Either restore the file at the recorded path
(`toolhub info <tool>` shows it) or `toolhub remove <tool>`. The runtime refuses to serve it
in the meantime with a message naming the tool — not a raw torch error.

### Registration was interrupted

It cannot leave you inconsistent: the artifact is copied to `<name>.pt.incoming`, the
manifest is written, and only then is the copy moved into place, with rollback on any
exception. So you get neither an orphaned file nor an entry pointing at nothing. If an
orphan does appear from some other path, `doctor` warns and `prune artifacts` clears it.

### Generated code was staged but never landed

```
you> What code is waiting for approval?
  → list_staged_code({})
```

Staging directories **outlive the session** deliberately, so you can read the files in an
editor and approve later. `doctor` warns about accumulation; `prune staging` clears it.

### A generated task stopped importing

`doctor` reports `custom_tasks` FAIL with the exception. One broken task does not break the
others (`discovery.load_all` collects failures rather than raising), but it cannot be trained
or served. Fix the code in `adaptrna_custom/tasks/<name>/`, or delete the package.

### A store says another process wrote first

```
'…/jobs.json' changed on disk since it was read — another process started or updated a job.
Nothing was written; retry so the change applies on top of theirs.
```

Exactly what it says. The stores **detect** concurrent writes rather than preventing them:
two chat processes are fine, the second to save is asked to retry. Over HTTP this is a 409
marked `retryable` — a *transient* answer, not a stop condition. Polling clients keep their
last good render **and keep polling**; the browser's job monitor re-arms its timer for this
reason, because the likeliest moment for a 409 is just after a run starts, exactly when
someone is watching.

## 5. Running the service

```bash
python -m adaptrna_agentic.cli.serve                       # 127.0.0.1:8000
python -m adaptrna_agentic.cli.serve --open --warmup       # browser + preloaded backbone
ADAPTRNA_API_TOKEN=secret python -m adaptrna_agentic.cli.serve --host 0.0.0.0
```

It binds loopback by default and **refuses to start** on any other address without a token
— checked before uvicorn is even imported, so the dangerous configuration is not reachable
by accident. A configured token is required on every path except `/health`.

The only delete surface in the API is `DELETE /api/sessions/{id}` (Phase 10). No tool
removal, no pruning, no job or artifact deletion — those stayed CLI actions, because a shell
prompt really is a better boundary for things that are expensive or hard to recreate. A
conversation is neither. Note that `prune` is still not an agent tool: the *model* cannot
delete anything.

Health:

```bash
curl -s localhost:8000/health | jq
# {"status":"ok","install":"warn","failed_checks":[],"backbone_loaded":true}
```

## 6. Sessions

One SQLite checkpointer at `chat_data/sessions.sqlite`, shared by the terminal and the HTTP
service — a conversation started in one continues in the other.

```bash
python -m adaptrna_agentic.cli.chat --list-sessions
curl -s localhost:8000/api/sessions | jq          # [{id, updated_at, checkpoints}], newest first
curl -s localhost:8000/api/sessions/<name>/history | jq

curl -s -X POST localhost:8000/api/sessions -H 'content-type: application/json' \
     -d '{"id":"work"}' | jq                      # 409 if it already exists
curl -s -X PATCH localhost:8000/api/sessions/work -H 'content-type: application/json' \
     -d '{"id":"work-2"}' | jq                    # carries the history with it
curl -s -X DELETE localhost:8000/api/sessions/work-2 | jq
```

Rename is an `UPDATE` of `thread_id`, which is part of the checkpointer's primary key — so
it is refused (409) when the target name is taken, or when the session is suspended at an
approval whose interrupt is addressed by that id. Delete is irreversible and `chat_data/` is
git-ignored; the browser asks for a confirmation and that is the whole safety net.

A session waiting on an approval refuses new messages (409) until `/resume` answers it. In
the browser, a refresh mid-approval restores the dialog from `history.pending_approval` —
the suspended turn lives in the checkpointer, not in the tab.

The API sets `PRAGMA journal_mode=WAL` and `busy_timeout=5000` explicitly; without WAL a
writer would block every reader, defeating the point of sharing the file.

## 7. Disk usage

| Directory | What accumulates | Reclaim with |
|---|---|---|
| `outputs/` | ~20 MB per LoRA run; ~2.6 GB per full-FT export | `prune runs --older-than N` |
| `toolhub_data/adapters/` | ~6 MB per registered tool | `prune artifacts` (orphans only) |
| `toolhub_data/staging/` | kilobytes per stage | `prune staging` |
| `chat_data/` | conversation history (a few MB) | `prune sessions` |
| `jobs_data/` | negligible | `prune jobs` |
| `~/.cache/rinalmo_pretrained/` | 2.6 GB for `giga-v1.pt` | manually, if you mean it |

`doctor` warns above 5 GB for `outputs/`, `toolhub_data/` and `chat_data/`.

## 8. Known limitations

Each of these is a deliberate design choice with a recorded rationale, not an oversight:

| Limitation | Why |
|---|---|
| **A crashed training run cannot be resumed** mid-flight | No checkpoint-resume path exists; the engine has no metric-based checkpoint selection either |
| **One training job at a time** by default | Two `giga` runs on one GPU is how you get an OOM forty minutes in |
| **Serving runs fp32** | Casting the engine to bf16 for non-autocast inference trips a dtype promotion in its `TokenDropout` |
| **Inference is serialised** | The hub activates an adapter across the whole backbone, so overlapping predictions could answer from the wrong one. Correctness over throughput. |
| **Generated code is accident-isolated, not sandboxed** | Time, memory and file-size limits catch runaway loops; the human diff gate is the real boundary |
| **The stores detect concurrent writes, they do not prevent them** | Detection is cheap; losing a registration is not. The second writer retries. |
| **Deactivation does not free memory** | peft cannot cleanly uninject an adapter; resident adapters cost megabytes. `rebuild()` is the full cleanup. |
| **Single-user posture** | Loopback, one token, and the only thing deletable over HTTP is your own conversation. A deliberate design, not a starting point for a shared deployment. |
| **Linux only** | `/proc/<pid>/stat` for PID identity; `resource.setrlimit` + `os.setsid` for the sandbox |

## 9. Clean slate: resetting to zero tools

A fresh install of this build should show **no registered tools** — that is a documented
fact elsewhere ([README.md](../README.md#10-verified-facts)), not just a default. It is not,
however, automatic: nothing on upgrade deletes a tool you already registered, because
deleting a user's registered tool is a human action at the CLI, the same reason there is no
delete endpoint in the API (§5 above).

If your install carries a tool registered in an earlier session and you want the platform to
actually start from zero — matching a documentation walkthrough, or preparing a demo — reset
it by hand, in this order:

```bash
python -m adaptrna_agentic.cli.toolhub remove <tool_name>       # once per registered tool
rm adaptrna_custom/tools/<wrapper>.py                            # if it was a landed wrapper
python -m adaptrna_agentic.cli.toolhub prune staging             # dry run
python -m adaptrna_agentic.cli.toolhub prune staging --yes       # drops anything never landed
python -m adaptrna_agentic.cli.toolhub list                      # expect: no tools
```

This is **not** something the platform runs on upgrade or on its own — it is written down so
the sequence is repeatable, not so it is automated. Two things it deliberately leaves alone:

* **The backbone configuration** (`toolhub_data/tools.json`'s `backbone` block) — that is
  what the hub serves, not a tool, and resetting the tool list should not make the platform
  forget where the checkpoint lives.
* **`jobs_data/jobs.json` and `outputs/*`** — historical records of training runs stay.
  `analyze_run` on an old job still works; since the knowledge base carries no reference band
  for any task any more, it reports a **baseline** verdict rather than a reference-band
  comparison, which is accurate — the platform no longer has special knowledge of any task.

`adaptrna_custom/tasks/` needs no equivalent reset in the same sense: a task package left
there after a tool reset simply has no tool pointing at it any more, and remains trainable
(`recommend_training_config` can still target it) until you delete the directory yourself.
