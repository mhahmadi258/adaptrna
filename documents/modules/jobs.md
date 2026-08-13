# `jobs/` — training runs

`agentic/adaptrna_agentic/jobs/`

Launch the engine's training CLI on the local GPU, track it while it runs, and judge the
result. Four files, no LLM.

---

## Contents

1. [Design constraint: jobs are detached](#1-design-constraint-jobs-are-detached)
2. [`train_entrypoint.py` — the seam](#2-train_entrypointpy--the-seam)
3. [`store.py` — job records](#3-storepy--job-records)
4. [`runner.py` — launch, track, cancel](#4-runnerpy--launch-track-cancel)
5. [PID identity](#5-pid-identity)
6. [Reading progress](#6-reading-progress)
7. [`analysis.py` — the RunAnalyzer](#7-analysispy--the-runanalyzer)
8. [Typical usage](#8-typical-usage)
9. [Assumptions and limitations](#9-assumptions-and-limitations)

---

## 1. Design constraint: jobs are detached

A real run takes minutes to hours, the chat must stay responsive, and the run must survive
the chat process exiting. Everything in this package follows from that:

> State is derived from **disk** — the `exit_code` file the entrypoint writes, the engine's
> own `metrics.csv`, and PID liveness — never from a live process handle.

`subprocess.Popen(..., start_new_session=True)` detaches the child. The runner keeps the
`Popen` handle *if it started the job in this process*, purely so polling reaps the child
and yields a real exit code even when the run died before writing `exit_code`; a fresh
process reconstructs everything from disk instead.

## 2. `train_entrypoint.py` — the seam

```
python -m adaptrna_agentic.jobs.train_entrypoint --task splice_site --use_lora ...
```

Every training job is launched through here rather than through
`rinalmo_hub.cli.train` directly. It does exactly two things the engine cannot:

1. **Imports every generated task** (`discovery.load_all()`) so `@register_task` has fired
   before the engine resolves the task name. Import failures are printed as a warning and
   are *not* fatal — the engine will raise a clear "unknown task" error if the run actually
   needed one of them.
2. **Writes `<output_dir>/exit_code`** in a `finally` block, so job state survives a lost
   PID, a killed chat, or a machine reboot.

Between the two it calls `rinalmo_hub.cli.train.main(argv)` **unchanged**. That is the whole
file — the MASTER_PLAN §3.4 seam, kept as thin as it sounds.

## 3. `store.py` — job records

`jobs_data/jobs.json`, `format_version: 1`. Same atomic-write + in-file revision discipline
as the tool manifest; schema in
[../configuration.md §6](../configuration.md#6-jobs_datajobsjson--the-job-store).

```python
JobRecord(id, task, arm, command, output_dir, state="running", pid=None,
          pid_starttime=None, started_at=None, ended_at=None, exit_code=None,
          adapter_path=None, plan={})
    .output_path   # absolute, resolved against REPO_ROOT if relative
    .log_path      # <output_path>/train.log

JobStore(jobs_dir=None)
    .add(record) .get(id) .list()      # newest first, by started_at
    .running()                         # state == "running"
    .save()                            # revision-checked, atomic
```

The job **id is the output directory's basename**, which is why run names carry a timestamp
(`splice_site_acceptor_lora_20260812_185058`) — it makes ids unique and human-readable, and
lets `analyze_run` find the run from the id alone.

Storing the whole `plan` on the record is deliberate: it keeps a run analysable later —
including its `primary_metric` and reference band — even if its task can no longer be
imported because it was deleted, renamed or moved.

## 4. `runner.py` — launch, track, cancel

```python
JobRunner(store=None, jobs_dir=None)
    .start(plan, allow_concurrent=False) -> JobRecord
    .status(job_id)  -> dict            # record fields + progress
    .list()          -> [dict]          # id, task, arm, state, started_at, output_dir
    .logs(job_id, tail=40) -> str
    .cancel(job_id)  -> {"id", "state"}
```

### `start(plan)`

1. `_refresh_all()` — reconcile every record with disk first, so a finished-but-unreaped job
   does not block a new one.
2. Refuse a second job unless `allow_concurrent`:
   > *"Job '…' is still running — two giga runs on one GPU is how you get an out-of-memory
   > failure mid-run. Wait for it to finish, cancel it, or pass allow_concurrent."*
3. Create `output_dir` (resolved against `REPO_ROOT` when relative); the job id is its
   basename, and a collision is refused.
4. `Popen(plan["command"], cwd=REPO_ROOT, stdout=train.log, stderr=STDOUT,
   start_new_session=True)`.
5. Record `(pid, pid_starttime)` and the full plan; `store.add()`.

Note what `start` does **not** do: it does not validate the plan's contents. The
`plan["source"]` check lives one level up in
[`tool_factory.start_training`](agents.md#the-16-management-tools), so that a Python caller
constructing a plan by hand is not blocked while the *model* is.

### `_refresh(record)`

Terminal states are never revisited. For a `running` record:

```python
returncode = self._processes[id].poll() if we started it else None

if (output_dir/"exit_code").exists():
    code   = int(...)                    # unparseable → 1
    state  = "succeeded" if code == 0 else "failed"
    adapter_path = adapter_path or _find_adapter(record)      # glob *_adapter.pt
elif returncode is not None or not _is_our_process(pid, pid_starttime):
    state = "failed"                     # gone without an exit code: SIGKILL, OOM, hard crash
    exit_code = returncode
```

### `cancel(job_id)`

Identity is checked **before** the state check, so the specific reason wins:

```python
if record.state == "running" and not _is_our_process(record.pid, record.pid_starttime):
    record.state = "failed"; store.save()
    raise ToolHubError("Job '…' is no longer running (its process is gone, and PID N may "
                       "since have been reused — refusing to signal it). "
                       "The record has been marked failed.")
```

Only then does it `os.killpg(os.getpgid(pid), SIGTERM)` — the process group, which is why
`start_new_session=True` matters — and mark the record `cancelled`.

## 5. PID identity

> A stored PID is not a process identity: PIDs are recycled, so a dead job can look alive
> and `killpg` can signal a stranger.

```python
process_starttime(pid)   # /proc/<pid>/stat field 22, in clock ticks
_is_our_process(pid, starttime) -> bool
```

`(pid, starttime)` is a stable identity on Linux: the kernel recycles PIDs, but a recycled
PID always has a *later* start time. Parsing splits after the parenthesised `comm` field,
which may itself contain spaces.

Four ways the answer is "no", and all four matter:

| Case | Detected by |
|---|---|
| The PID is gone | `os.kill(pid, 0)` → `ProcessLookupError` |
| It belongs to another user | `PermissionError` — definitively not ours |
| It is a **zombie** — exited, awaiting reaping | `/proc/<pid>/stat` state `Z`. A zombie still answers signal 0. |
| It was recycled, **or** the record predates start-time capture | `process_starttime(pid) != stored`, and `starttime is None` returns `False` — without an identity we must assume the worst rather than signal a stranger |

`tests/test_process_identity.py` exercises every branch. This was found in practice, on a
machine where every recorded PID had already been freed.

## 6. Reading progress

```python
latest_metrics_file(output_dir)   # <dir>/metrics/version_N/metrics.csv, highest N that exists
read_progress(output_dir)         # {"progress": {"epoch", "step", "latest_metrics"} | None}
```

The engine writes `metrics.csv` as the run goes — that is the documented interface — and its
rows are **sparse**: each row carries only the metrics logged at that moment. So every
reader takes the **last non-null value per column**, and `epoch`/`step` are excluded from
the metrics map and reported separately. An empty or unreadable file yields
`{"progress": None}` rather than an error.

## 7. `analysis.py` — the RunAnalyzer

```python
analyze_run(output_dir, task=None, arm="lora", plan=None) -> report
```

Deterministic checks over `metrics.csv`, each mapped to a documented failure mode in the
knowledge base — so a bad run comes back with the **reason** and the **remedy**, not just a
number. The LLM narrates this report; it does not judge the run.

```python
{"output_dir", "task", "arm", "truncated": bool,
 "metrics": {...},                    # last logged value of every column
 "primary_metric", "primary_value",
 "checks": [str],                     # plain lines, plus "FAIL: …" / "WARN: …"
 "suggestions": [str],
 "baseline": {job_id, value, finished_at},   # only when there is no reference band
 "verdict": "ok" | "suspicious" | "failed"}
```

`verdict` = `failed` if any failure, else `suspicious` if any warning, else `ok`.

### The two rules this module exists to enforce

1. **A truncated run is never compared to reference metrics.** If the plan set
   `quick_run` or `trainer.max_steps`, the report says so explicitly — *"a smoke test, NOT
   comparable to the reference metrics for this task"* — and skips the band comparison
   entirely.
2. **A difference inside the task's tolerance is not a regression.** FlashAttention's
   backward pass is non-deterministic; the same splice-site command and seed produced F1
   95.21 and 95.82.

### The checks

| Check | Detection | Verdict effect |
|---|---|---|
| No `metrics.csv` | `latest_metrics_file` returns `None` | `failed` + *"check the run log for an early crash"* |
| Non-finite loss | `_nonfinite_loss_columns` | `failed` + *"lower the learning rate and set `trainer.gradient_clip_val`"* |
| Primary metric never logged | absent from the final metrics | warning + *"the run may not have reached its test phase"* |
| Primary metric ≤ 1e-6 | `_DEGENERATE` | `failed` + the arm's first known failure mode, verbatim from the knowledge base |
| Constant-output collapse | `_looks_collapsed`: `train/loss` std over the second half < 1e-9 (needs ≥ 4 points) | warning + the arm remedy |
| Below the reference band | `value < low - tolerance` | warning + *"confirm with a second seed or data split before concluding anything"* |
| Above the reference band | `value > high + tolerance` | note — *"good, but worth confirming there is no leakage between splits"* |
| No band recorded | `band is None` | compare against `previous_best(...)` instead |

### Divergence vs "not logged here"

```python
raw = pd.read_csv(metrics_file, dtype=str, keep_default_na=False)
# a column named *loss* containing a literal "nan"/"inf"/… as TEXT
```

Because the CSV is sparse, pandas reads both an empty cell and a literal `"nan"` as `NaN`.
Reading the cells as text is the only way to tell an actually-diverged run from a metric
that simply was not logged on that row.

### Baselines for unknown tasks

`previous_best(task, arm, metric, exclude=…)` scans this project's own **succeeded**,
non-truncated runs of the same task and arm and returns the best value with its job id.
Used only when the knowledge base has no band — a generated task, say.

The wording is deliberate and enforced in the strings: a baseline is *"a comparison with an
earlier run of this project, not a validated reference"*. The first run of a new task is
reported as *"this run is the baseline for future ones"*. `tests/test_analysis_baseline.py`
pins that distinction.

## 8. Typical usage

```python
from adaptrna_agentic.jobs.runner import JobRunner
from adaptrna_agentic.jobs.analysis import analyze_run

runner = JobRunner()
record = runner.start(plan)                    # plan from recommender.recommend()

runner.status(record.id)
# {'id': ..., 'state': 'running', 'progress': {'epoch': 3, 'step': 150,
#  'latest_metrics': {'train/loss': 0.11, 'val/f1_score': 0.96}}, ...}

print(runner.logs(record.id, tail=100))
runner.cancel(record.id)                       # refuses if it cannot prove the PID is ours

report = analyze_run(record.output_path, plan=record.plan)
print(report["verdict"], report["checks"])
```

Tests drive this with a **fake command** that writes a `metrics.csv` the way the engine does
and exits with a chosen code — which is exactly the interface the runner consumes, so no GPU
and no engine training are involved (`tests/test_job_runner.py`).

## 9. Assumptions and limitations

* **Linux only** — `/proc/<pid>/stat` has no portable equivalent.
* **A crashed run cannot be resumed** mid-flight; start it again.
* **One job at a time** by default.
* **`JobRunner` instances are cheap but not shared.** `tool_factory` builds one per
  `_pipeline_tools()` call and the HTTP job router builds a fresh one per request — fine,
  because all state is on disk, but it means the `_processes` handle cache only helps the
  process that launched the job.
* **`_find_adapter` takes the first `*_adapter.pt` alphabetically** in the output directory.
  One file per run in practice.
* **The analyzer reads the final logged value, not the best.** That matches the engine,
  which has no metric-based checkpoint selection: the final-epoch model is what gets tested.
