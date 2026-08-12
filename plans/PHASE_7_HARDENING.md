# Phase 7 — Hardening (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 7 ("runs continuously from Phase 4
> onward"); testing strategy §9.
> **Definition of done:** failure paths behave as documented; eval scenarios green.
> Status: planned · not started

---

## 1. Context and goal

Six phases built the happy path and proved it end to end. Phase 7 is about what happens
when things go wrong — and unlike the earlier plans, this one is grounded in **evidence
from the running system** rather than speculation. An audit of the live install
(2026-08-12) found:

| Observed | Why it matters |
|---|---|
| `toolhub_data/staging/splice_simple-cb41d441` still present after landing | Staging accumulates forever; nothing ever cleans it |
| 3 job records holding PIDs **3754577 / 3622360 / 3624613**, all long dead | Those PIDs are free for reuse — see the correctness bug below |
| 6 chat sessions in SQLite, none expirable | Phase 4 explicitly deferred retention to here |
| `outputs/` at **2.5 GB** (one full-FT export alone is 2.6 GB) | Unbounded growth, no pruning |
| 5 manifest tools / 3 artifacts, currently consistent | Nothing *checks* that they stay consistent |

Nothing here is exotic. It is the ordinary debt of a system that has only ever been asked
to succeed.

**Out of scope:** the HTTP API (Phase 8), the web UI (Phase 9), resuming a crashed
training run mid-flight (documented as a limitation instead — the engine has no
checkpoint-resume path this layer could drive safely).

## 2. Decisions this plan fixes

| Decision | Choice | Rationale |
|---|---|---|
| **Deletion policy** | Nothing is ever deleted automatically. `doctor` *reports*; `prune` deletes only what the user names, and refuses to touch anything referenced by the manifest | The artifacts here are hours of GPU time and generated code. An automatic cleaner that is wrong once costs more than the disk it saves |
| **Consistency model** | Optimistic concurrency on the manifest and job store: remember the file's mtime+size at load, refuse to save over a newer one with a message telling the user to retry | Phase 2 chose JSON on the "single user, single process" premise. That premise breaks in Phase 8 and is already shaky with two terminals. Detecting a clobber is cheap; silently losing a registration is not |
| **Process identity** | A job records `pid` **and the process start time**; liveness and cancel both require both to match | PID reuse is not theoretical here — all three recorded PIDs are already free (§1). Today `cancel` would `killpg` whatever now owns that PID |
| **No-reference baselines** | When a task has no knowledge-base band, the analyzer compares against **previous completed runs of the same task+arm** from the job store, and says so | Closes the gap Phase 6 exposed: the generated task's verdict was "nothing to compare". A first run becomes the baseline for the second, without inventing a band |
| **Prompt regression guard** | Scenario suite runs on **scripted models** (deterministic, in CI) and covers graph wiring + tool contracts; a small **live** suite is written but marker-gated and user-run | Same split the engine uses for `gpu`/`weights` tests. Testing prompts against a real model on every commit is flaky and expensive |

## 3. Correctness fixes (the actual bugs)

### 3.1 PID reuse — `jobs/runner.py`

`_pid_alive(pid)` and `cancel()` trust a bare PID. After a reboot or heavy churn that PID
belongs to something else, so a dead job can look alive forever, and — the serious half —
`os.killpg(os.getpgid(pid), SIGTERM)` can signal an **unrelated process group**.

Fix: capture `/proc/<pid>/stat` field 22 (start time in clock ticks) at launch and store
it on the record; `_pid_alive` and `cancel` both verify pid **and** start time. A record
whose start time no longer matches is treated as "process gone", never as ours to kill.
Records written before this change have no start time — treat them as gone rather than
signalling them (documented, and covered by a test).

### 3.2 Partial registration — `toolhub/registry.py`

`register()` copies the artifact, then writes the manifest. A failure between the two
leaves an orphaned `.pt`; the reverse ordering would leave a manifest entry with no file.
Fix: copy to a temp name, write the manifest, then move into place; on any failure, remove
the temp copy. Plus a `verify()` that reports both directions (entry without artifact,
artifact without entry) — consumed by `doctor`.

### 3.3 Concurrent manifest / job-store writes

Both stores read-modify-write whole JSON files. Two chat processes → last write wins,
silently. Fix: record `(mtime_ns, size)` at load; on save, re-stat and refuse if it
changed, with a message naming the file and telling the user to retry. Cheap, and it turns
a silent loss into a visible, retryable error.

### 3.4 Missing artifacts fail unhelpfully — `toolhub/runtime.py`

A registered adapter whose `.pt` was deleted surfaces a raw torch/OSError from deep inside
`hub.register`. Fix: check existence first and raise a ToolHubError naming the tool, the
missing path, and the two ways out (`toolhub remove <name>`, or restore the file).

## 4. `doctor` — one command that tells you what is wrong

`python -m adaptrna_agentic.cli.toolhub doctor [--json]`, also exposed as an agent tool
(read-only, ungated). Checks, each reporting `ok | warn | fail` with a concrete remedy:

| Check | Detects |
|---|---|
| engine importable, version/git SHA | a broken or stale editable install |
| backbone checkpoint present and readable | the Phase 5 failure that cost a training run |
| manifest ↔ artifacts, both directions | orphaned `.pt`s; entries whose file vanished |
| external tools importable, packages installed | a wrapper whose package was uninstalled |
| custom task packages import (`load_all` failures) | a generated task broken by a later edit |
| jobs: `running` records whose process is gone | crashed runs stuck in `running` |
| jobs: output dirs missing / adapters missing | a pruned run still referenced |
| staging: stages not landed, with age and size | the orphan the audit found |
| disk: `outputs/`, `toolhub_data/`, `chat_data/` sizes | the 2.5 GB nobody was watching |
| sessions: count and last-touched | unbounded checkpoint growth |

`doctor` never changes anything. Every `fail`/`warn` line names the exact command that
fixes it.

## 5. `prune` — deletion, only where the user points

`python -m adaptrna_agentic.cli.toolhub prune <what> [--older-than DAYS] [--yes]` for
`staging`, `sessions`, `jobs`, `runs`. Rules, in this order:

1. **Never** delete anything referenced by the manifest (a registered tool's artifact, or
   the output dir of the job that produced a registered tool). Referenced items are listed
   as skipped, with the reason.
2. Default to a dry run; `--yes` performs it; sizes are printed before and after.
3. `runs` deletes output directories (the GPU-hours artifacts) — always requires an
   explicit age filter, never a bare "delete everything".

Not exposed as an agent tool in this phase: deletion stays a human action at the CLI.

## 6. Analyzer baselines from run history

`analyze_run` currently reports "no in-repo reference for this task yet — nothing to
compare" for any task outside the knowledge base, which is exactly what the Phase 6
generated task got. Fix: when the knowledge base has no band, look up previous
**succeeded** runs of the same task+arm in the job store, take their primary metric, and
report the comparison as a *baseline*, explicitly labelled as such — never as a validated
reference. First run says "this is the baseline"; later runs say "0.977 → 0.981 vs the
baseline from job X". Non-determinism tolerance still applies.

## 7. Scenario evals and the user guide

**`agentic/tests/scenarios/*.yaml`** — recorded conversations, replayed against scripted
models by one parametrised test. Each scenario declares user turns, the scripted assistant
responses, and the expected tool-call sequence plus assertions on the resulting state.
Seed set, covering the five flows end to end:

| scenario | asserts |
|---|---|
| `inference.yaml` | list_tools → adapter call → answer; disabled tool refuses and recovers |
| `management.yaml` | activate/deactivate/test round trip leaves the manifest consistent |
| `training.yaml` | profile → recommend → **gate** → job started only after approval; deny leaves no job |
| `codegen.yaml` | create → verify → **gate** → nothing lands on deny; lands on approve |
| `failure_paths.yaml` | missing artifact, dead job, unknown tool, unstamped plan — each returns a message the model can act on, none crash the turn |

A second, marker-gated file (`-m live`) runs two of these against the real model, user-run,
for prompt regressions.

**The user guide** — the root [README.md](../README.md) grows from a pointer into a real
guide: install, the five things you can ask for, where state lives, what `doctor` and
`prune` do, the known limitations (no mid-run resume; one training job at a time;
half-precision serving disabled; generated code is accident-isolated, not sandboxed), and
a troubleshooting table keyed by the error messages this phase standardises.

## 8. Tests

| test file | asserts |
|---|---|
| `test_process_identity.py` | start time captured at launch; a record whose start time no longer matches is "gone"; `cancel` refuses a mismatched record **without signalling**; legacy records (no start time) are treated as gone |
| `test_store_concurrency.py` | a second writer over a changed file is refused with a retryable message; the first writer's data survives; both stores covered |
| `test_registry_atomicity.py` | a failure mid-register leaves neither an orphan artifact nor a half entry; `verify()` reports both mismatch directions |
| `test_doctor.py` | each check fires on a purpose-built broken install (deleted artifact, orphan `.pt`, stale `running` job, missing checkpoint, broken custom task, orphaned stage) and passes on a clean one; `doctor` mutates nothing |
| `test_prune.py` | dry run by default; referenced artifacts and their run dirs are skipped with reasons; age filter honoured; `runs` refuses without one |
| `test_analysis_baseline.py` | no band + no history → "baseline"; no band + history → comparison naming the earlier job; within tolerance is not a regression; a knowledge-base band still wins |
| `test_scenarios.py` | the YAML suite above |

~35 new tests (total ~275). Phases 0–6's 240 stay green; engine's 135 untouched.

## 9. Implementation order

1. §3 correctness fixes + their tests (bugs before features).
2. §6 analyzer baselines + tests.
3. §4 `doctor` + tests.
4. §5 `prune` + tests.
5. §7 scenario suite + the live-marked file.
6. User guide; close-out (MASTER_PLAN §8 tick).

## 10. Verification / definition of done

**Gate 1 — deterministic:** `cd agentic && pytest` green (240 + ~35); `cd engine && pytest`
→ 135.

**Gate 2 — the chaos drill.** Deliberately break the live install, then show each failure
is *reported* and *recoverable*, with the tools still serving throughout:

```bash
# 1. an artifact vanishes
mv toolhub_data/adapters/splice_simple.pt /tmp/
toolhub doctor            # -> FAIL: 'splice_simple' artifact missing; remove or restore
toolhub predict splice_site --sequences ...   # other tools unaffected
mv /tmp/splice_simple.pt toolhub_data/adapters/

# 2. a job "running" whose process is gone (and whose PID gets reused)
#    -> doctor reports it stale; cancel refuses to signal a foreign process
# 3. the orphaned stage from the audit
toolhub prune staging --older-than 0          # dry run, then --yes
# 4. disk
toolhub doctor            # -> outputs/ 2.5 GB, with the prune command to reclaim it
```

Then the positive control: after every repair, `toolhub doctor` is clean and the Phase 6
scenario (score a window with two adapters) still works.

**Gate 3 — the guide is true.** Every command in the root README runs as written on this
machine; every troubleshooting row names an error the code actually emits.

## 11. Risks and notes

- **`doctor` must never lie.** A health check that reports green on a broken install is
  worse than none — hence every check is tested against a purpose-built broken install,
  the same discipline as the Phase 6 harness controls.
- **Prune is the one destructive command.** Manifest-referenced artifacts are skipped
  unconditionally; `runs` requires an age filter; nothing is exposed to the agent.
- **Optimistic locking is not locking.** It detects a clobber; it does not prevent two
  writers from racing. That is the right amount of machinery for a JSON store, and Phase 8
  should revisit the store choice rather than bolt on more.
- **Scripted scenarios do not test prompts.** They pin wiring and contracts. The live suite
  is the prompt guard, and it stays user-run on purpose.
