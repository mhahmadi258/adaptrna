# Clean-slate reset — one script that returns the install to fresh, keeping `outputs/`

## Context

A fresh AdaptRNA install is *defined* as shipping zero task definitions and zero adapters —
`README.md` states it as a verified fact, and `documents/workflows/operations.md` §9 ("Clean
slate: resetting to zero tools", lines ~291–325) writes down a **manual** five-step sequence to
get back there. That sequence is deliberately narrow: it resets the *tool list* only, and leaves
`chat_data/`, `jobs_data/` and landed `adaptrna_custom/` code in place. It was also last executed
by hand in commit `cea931d` ("Phase 13 §15: the clean-slate reset").

The install has drifted well past fresh again: three registered adapters (`mrl`,
`splice_site_donor`, `splice_site_acceptor`, 18 MB of registry-owned copies), one landed task
package `adaptrna_custom/tasks/train_acceptor_400/`, one landed wrapper
`adaptrna_custom/tools/rna_secondary_structure.py`, a leftover staging directory
(`toolhub_data/staging/train_acceptor_400-4e27ea49`), a 6.8 MB conversation store and a populated
job store.

**Goal:** one script that returns the install to the state a `git clone` + `pip install -e` would
leave it in — *except* that `outputs/` is preserved untouched, because training results live
there. Datasets, the downloaded backbone, `.env` and the manifest's backbone configuration are
preserved too; the reset is dry-run by default.

Scope decisions taken with the user before writing this plan:

| Question | Decision |
|---|---|
| How far does the reset go? | App state only. Wipe `toolhub_data/`, `jobs_data/`, `chat_data/` and generated `adaptrna_custom/` code. **Keep** `outputs/`, `dataset/`, `weights/`, `.env` and the `backbone` block. |
| Where does it live? | `agentic/scripts/reset.py`, beside `make_demo_data.py` — not a `toolhub` CLI verb. |
| Safety | Dry run by default, `--yes` applies — the `prune` convention. |
| Docs | Rewrite `documents/workflows/operations.md` §9. |

---

## 1. `agentic/scripts/reset.py`

New standalone script, beside the existing `make_demo_data.py` / `update_template_golden.py`.
Kept out of the shipped `toolhub` CLI on purpose: `prune.py:1` documents `prune` as "the one
destructive command, and the only one", and a whole-install wipe is a bigger hammer than that CLI
surface should advertise.

### Invocation

```bash
python agentic/scripts/reset.py                 # dry run — prints the plan, deletes nothing
python agentic/scripts/reset.py --yes           # applies it
```

| Flag | Effect |
|---|---|
| `--yes` | Actually delete. Without it nothing is touched — mirrors `prune.py:74`. |
| `--cancel-running` | Cancel live training jobs first (see the guard below). |
| `--forget-backbone` | Also drop the `backbone` block — deletes `tools.json` outright instead of rewriting it. Off by default. |
| `--data-dir` / `--jobs-dir` / `--chat-dir` | Test seams, matching `cli/toolhub.py:30` and `cli/chat.py:59`. |

### Guard, before anything else

`JobStore.running()` (`agentic/adaptrna_agentic/jobs/store.py:156`) — job state is re-derived from
disk on read, so this is authoritative. If any job is running, refuse: print the job ids and the
fix, exit 1. With `--cancel-running`, call `JobRunner.cancel(job_id)`
(`agentic/adaptrna_agentic/jobs/runner.py:172`) for each instead — it verifies PID identity via
`pid_starttime` against `/proc/<pid>/stat` before `killpg`, which a hand-rolled kill would not.

### What it removes

Order is load-bearing only for 1 → 2 (an artifact still referenced by the manifest must not be
deleted while its entry exists); the rest is independent.

| # | Target | How |
|---|---|---|
| 1 | Every entry in `toolhub_data/tools.json` | `Manifest.load()`, `manifest.tools.clear()`, `manifest.save()` — preserves the `backbone` block and keeps the atomic-write + revision discipline (`toolhub/manifest.py:165`). With `--forget-backbone`, unlink the file instead. |
| 2 | `toolhub_data/adapters/*` | Registry-owned adapter copies, plus any `*.pt.incoming` left by an interrupted registration (`toolhub/registry.py:123`). |
| 3 | `toolhub_data/staging/*` | Never-landed generated code (`codegen/staging.py:33`). |
| 4 | `jobs_data/jobs.json` | Recreated empty on next write; `JobStore._load` (`jobs/store.py:84`) treats a missing file as empty state. |
| 5 | `chat_data/sessions.sqlite` + `-wal` + `-shm` | Path from `chat_db_path()` (`cli/chat.py:37`). LangGraph's `SqliteSaver.setup()` recreates the file and both tables lazily — there is no migration step to run and no init command to call. |
| 6 | `adaptrna_custom/tasks/<name>/` and `adaptrna_custom/tools/*.py` | Everything **except** the four git-tracked skeleton files: `README.md`, `__init__.py`, `tasks/__init__.py`, `tools/__init__.py`. Roots from `codegen/discovery.py:25-30`. |
| 7 | Stale temp state | `toolhub_data/.tools.*.tmp`, `jobs_data/.jobs.*.tmp`, and `adaptrna-sandbox-*` / `adaptrna-harness-*` under `tempfile.gettempdir()` — sandbox leftovers that only appear after a SIGKILL (`codegen/sandbox.py:107`, `codegen/harness.py:50`). |
| 8 | `__pycache__/` under `adaptrna_custom/` | Stale bytecode for modules that no longer exist — there are ~20 `gen_*.pyc` files there now. |

Step 6 carries a trap worth a comment in the code: `adaptrna_custom/README.md` calls that
directory "deliberately git-tracked", but `.gitignore:16` ignores the whole thing and only those
four files are actually tracked. Deleting them would **not** be recoverable with
`git checkout` — hence an explicit keep-set constant rather than a blanket `rmtree`.

### Explicitly preserved — printed as a footer in both modes

`outputs/` (**the user's training results — the reason this is not a `git clean -Xdf`**),
`dataset/`, `weights/`, `.venv/`, `.env` (holds the live `ANTHROPIC_API_KEY`; never deleted, never
printed), `~/.cache/rinalmo_pretrained/giga-v1.pt`, and the manifest's `backbone` block.

Because `jobs_data/jobs.json` goes but `outputs/` stays, the run directories survive as plain files
with no job records pointing at them. That is intended, and the footer must say so: `analyze_run`
on an old job id will no longer resolve, the metrics on disk are untouched, and `toolhub doctor`'s
missing-job-output check has nothing to complain about (it walks jobs, not outputs).

### Structure and reuse

Mirror `prune.py`'s shape rather than inventing one — the report format is already familiar:

- Reuse `prune.Candidate` (label / path / bytes / skip_reason) and `prune._dir_size`
  (`toolhub/prune.py:99`); import them rather than re-declaring. Reuse `prune._remove`
  (`toolhub/prune.py:235`) for file-or-directory deletion.
- `plan_reset(...) -> dict` collects candidates and does nothing; `apply=True` performs them.
  Keeps the whole thing unit-testable without a filesystem fixture per step.
- Anchor everything on `settings.REPO_ROOT`, `manifest.resolve_data_dir`,
  `store.resolve_jobs_dir`, `chat.chat_db_path` — never on `Path.cwd()`, so the script works from
  any directory and honours the `ADAPTRNA_*_DIR` env vars.
- Print a per-item table with sizes and a reclaimed total, then either
  `Dry run — nothing was deleted. Re-run with --yes.` or the applied summary, then the
  preserved-items footer and the two verification commands.
- Exit codes: `0` done (or dry run), `1` refused because jobs are running.

### Optional test — `agentic/tests/test_reset.py`

Build a throwaway install under `tmp_path` (data-dir, jobs-dir, chat-dir, plus a fake
`adaptrna_custom`), then assert: dry run deletes nothing; `--yes` empties the manifest while
`backbone` survives byte-for-byte; the four skeleton files survive; `outputs/` never appears as a
candidate. Follows the seam-injection style already used across `agentic/tests/`.

---

## 2. Documentation

**Rewrite `documents/workflows/operations.md` §9**, retitled from "Clean slate: resetting to zero
tools" to cover the whole install. Keep the section's existing rationale — that part is worth
preserving — and update it honestly:

- Lead with `python agentic/scripts/reset.py` (dry run) → `--yes`.
- Keep the manual `toolhub remove` / `prune staging` sequence as the narrower alternative for
  removing *one* tool without touching conversations or job history.
- Rewrite the "two things it deliberately leaves alone" list: the backbone block still stands; the
  claim that `jobs_data/` and `outputs/*` are left alone is now **half true** — the script clears
  job records and keeps `outputs/`. Say so plainly, with the orphaning consequence.
- Replace the "`adaptrna_custom/tasks/` needs no equivalent reset" paragraph — the script does
  delete landed task packages now, keeping only the tracked skeleton.
- "Nothing automates this" becomes "nothing automates this *on upgrade*": the reset is still a
  deliberate human invocation, and still dry-run by default.

Two smaller edits, per the docs' own convention that a change updates the files it invalidates:

- `documents/workflows/operations.md` §7 "Disk usage" — add the script to the reclaim-command
  table.
- `documents/project_structure.md` — list `agentic/scripts/reset.py` under the `agentic/scripts/`
  entry.

---

## 3. Verification

```bash
# 1. Preconditions — record what should survive
du -sh outputs/ dataset/                                   # note both sizes
ls adaptrna_custom/tasks/ adaptrna_custom/tools/
python -m adaptrna_agentic.cli.toolhub list                # expect the 3 current tools
python -m adaptrna_agentic.cli.toolhub config              # note lm_config + weights path

# 2. Dry run — must change nothing
python agentic/scripts/reset.py
python -m adaptrna_agentic.cli.toolhub list                # STILL the 3 tools
git status --short                                         # unchanged

# 3. Apply
python agentic/scripts/reset.py --yes

# 4. Postconditions
python -m adaptrna_agentic.cli.toolhub list                # "a fresh install has nothing to list"
python -m adaptrna_agentic.cli.toolhub config              # backbone UNCHANGED from step 1
python -m adaptrna_agentic.cli.toolhub doctor              # no staging leftovers, no orphans
du -sh outputs/ dataset/                                   # identical to step 1
git status --short                                         # clean: the 4 skeleton files survived
ls toolhub_data/adapters/ toolhub_data/staging/            # empty
ls chat_data/ jobs_data/                                   # sessions.sqlite / jobs.json gone

# 5. The system still works from zero
python -m adaptrna_agentic.cli.chat --once "What tools are available?"   # answers "none"
cd agentic && python -m pytest && cd ..
cd engine  && python -m pytest && cd ..
```

Step 4's `toolhub config` check is the one that catches the most likely bug: rewriting
`tools.json` through a fresh `Manifest()` instead of a loaded one would silently reset the backbone
to the `BackboneConfig` defaults (`weights: weights/giga-v1.pt`, `manifest.py:49`), pointing the
hub at a path that does not exist on this machine — the live manifest points at
`~/.cache/rinalmo_pretrained/giga-v1.pt`.
