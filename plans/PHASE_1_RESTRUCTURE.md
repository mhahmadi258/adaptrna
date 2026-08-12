# Phase 1 — Repo restructure into `engine/` + `agentic/` + `ui/` (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 1.
> **Definition of done:** `cd engine && pytest` → 135 passed; a smoke command still works
> from the repo root.
> Status: planned · not started

---

## 1. Context and goal

The repo currently has the engine at its root. Phase 1 moves it — unchanged — into
`engine/`, creates the `ui/` placeholder, and leaves runtime artifacts at the repo root so
every path convention keeps working. Baseline is clean: working tree committed
(`d2bbfb5`), engine suite at 135 passed, agentic suite at 16 passed.

**Key finding (verified 2026-08-12), which makes this phase lower-risk than expected:**
every path the engine computes in code is `__file__`-relative and therefore moves with the
package — **zero engine code edits are required**:

| file:line | resolves | after the move |
|---|---|---|
| `rinalmo_hub/config.py:14-15` | `REPO_ROOT` = two dirs above `config.py` → `BASE_CONFIG_PATH` | `engine/configs/base.yaml` ✓ (configs move too) |
| `rinalmo_hub/cli/common.py:61` | default task YAML three dirs above `common.py` | `engine/configs/tasks/<task>.yaml` ✓ |
| `rinalmo_hub/adapter.py:37` | `git -C <two dirs above adapter.py>` for the SHA | `git -C engine` — works anywhere inside the checkout ✓ |
| `scripts/benchmark_switching.py:30` | `sys.path` insert of the dir above `scripts/` | `engine/` — where the packages are ✓ |

**Second finding — a latent environment bug Phase 1 must fix:** the venv's editable
install is a stale `rinalmo-1.0.0` dist pointing at `/home/mh/bio2/RiNALMo` (the old
repo); this repo's `rinalmo_hub` was never properly installed. Everything has worked only
because commands run from the repo root, where the CWD lands on `sys.path`. After the move
that accident stops working, so a correct `pip install -e ./engine` becomes **load-bearing**
for running `python -m rinalmo_hub.cli.*` from the repo root.

## 2. Target layout (from MASTER_PLAN §2)

```
adaptrna/
├── engine/          # ← rinalmo/, rinalmo_hub/, configs/, ft_schedules/, examples/,
│                    #   scripts/, tests/, pyproject.toml, README.md — moved as-is
├── agentic/         # unchanged (Phase 0)
├── ui/              # NEW placeholder (README stub, filled in Phase 9)
├── plans/           # unchanged
├── outputs/  weights/  dataset/  adapters/   # runtime artifacts, stay at repo root
├── README.md        # NEW short root readme pointing at the three layers
├── .gitignore  .env  .venv/                  # unchanged
```

**Moves (all tracked → use `git mv`, staged as renames, history preserved):**
`rinalmo/`, `rinalmo_hub/`, `configs/`, `ft_schedules/`, `examples/`, `scripts/`,
`tests/`, `pyproject.toml`, `README.md` → `engine/`.

**Stays at root:** `outputs/` (holds the 2.6 GB full-FT export and the 6 MB donor adapter —
git-ignored, referenced by CWD-relative `--output_dir`), `agentic/`, `plans/`, `.env`,
`.gitignore`, `.venv/`. The `.gitignore` patterns (`dataset/`, `weights/`, `outputs/`,
`*.egg-info`, …) are unanchored, so they keep matching at any depth — no edits needed.

## 3. Why the CWD conventions survive

Commands keep running **from the repo root**, exactly as today (MASTER_PLAN §2):
`pretrained_weights: weights/giga-v1.pt`, `data.root: dataset/…` and `--output_dir
outputs/…` are CWD-relative and keep resolving to the root-level runtime dirs. Real
training runs on this machine already pass absolute paths for data
(`/home/mh/bio2/RiNALMo/dataset/...`, per `outputs/*/resolved_config.yaml`), so they are
unaffected either way.

Two references change spelling **in docs only** (they are plain file paths opened at
runtime, so what matters is the CWD of the command):

| in a command run from repo root | before | after |
|---|---|---|
| explicit task YAML | `--config configs/tasks/mrl.yaml` | `--config engine/configs/tasks/mrl.yaml` (or omit `--config`: the package-relative default finds it) |
| unfreezing schedule | `--set finetune.schedule=ft_schedules/mrl_paper.yaml` | `--set finetune.schedule=engine/ft_schedules/mrl_paper.yaml` |

The engine test suite's CWD-relative strings (`tests/test_training_loop.py:233-236`
`"ft_schedules/…"`, `tests/test_config_and_cli.py:198`, `tests/test_user_run.py:153`)
resolve against pytest's CWD — correct once the suite runs from `engine/`, which is
exactly the new DoD invocation. `tests/test_training_loop.py:213` already uses the
package-relative `REPO_ROOT / "ft_schedules"`. No test edits expected.

## 4. Steps, in order

1. **Pre-flight.** `git status` clean; record baselines (`pytest` at root → 135;
   `cd agentic && pytest` → 16).
2. **Move.** `mkdir engine`, then one `git mv` per item listed in §2. Clean stray
   `__pycache__`/`.pytest_cache` dirs left behind at the root (ignored files, delete).
3. **Fix the environment (the latent-bug fix).**
   - `pip uninstall -y rinalmo` — removes the stale editable dist pointing at
     `~/bio2/RiNALMo`.
   - `pip install -e ./engine` — installs `rinalmo-hub 0.1.0` properly (its explicit
     `[tool.setuptools] packages` list is project-root-relative, so it needs no changes).
   - Prove it from a neutral CWD: `cd /tmp && python -c "import rinalmo_hub, rinalmo;
     print(rinalmo_hub.__file__)"` → a path under `<repo>/engine/`.
4. **Engine suite (DoD #1).** `cd engine && pytest` → **135 passed, 7 deselected**. Any
   failure here is a CWD-dependent reference §3 missed — fix at the test, never by adding
   path hacks to core files.
5. **Agentic unaffected.** `cd agentic && pytest` → 16 passed (it imports nothing from the
   engine; this is a regression tripwire only).
6. **Root smokes (DoD #2).** From the repo root:
   - `python -m rinalmo_hub.cli.train --help` — entrypoint importable without CWD tricks;
   - `python -m rinalmo_hub.adapter outputs/splice_donor_lora/splice_site_adapter.pt` —
     package + root-level artifact IO in one shot (expects the familiar 6.08 MB summary);
   - `python -m rinalmo_hub.cli.evaluate --help` and `...predict --help`.
   A real GPU training smoke stays **user-run** (weights/data live outside this repo);
   its command shape is unchanged apart from §3's two doc-level spellings.
7. **Docs.**
   - `engine/README.md`: install becomes `pip install -e ./engine` (+ dev extra), test
     invocation becomes `cd engine && pytest`, `--config configs/…` →
     `engine/configs/…`, `ft_schedules/…` → `engine/ft_schedules/…` in the run-from-root
     command examples, and the §2 project-structure diagram gains the `engine/` root.
   - New root `README.md` (~15 lines): the three layers, where runtime artifacts live,
     pointers to `engine/README.md`, `agentic/README.md`, `plans/MASTER_PLAN.md`.
   - `ui/README.md` stub: "web UI — Phase 9; see plans/MASTER_PLAN.md".
   - `agentic/README.md`: no changes needed (its install line already says
     `pip install -e ./agentic`).
8. **Close-out.** Tick Phase 1 in MASTER_PLAN §8. Everything is staged (moves as renames
   R, plus the new/edited docs); committing is the user's call — suggested message:
   `Phase 1: restructure into engine/ + agentic/ + ui/`.

## 5. Verification (definition of done)

| # | check | expected |
|---|---|---|
| 1 | `cd engine && pytest` | 135 passed, 7 deselected |
| 2 | `cd agentic && pytest` | 16 passed |
| 3 | `cd /tmp && python -c "import rinalmo_hub; print(rinalmo_hub.__file__)"` | path under `<repo>/engine/` (stale-install fix proven) |
| 4 | repo root: `python -m rinalmo_hub.cli.train --help` | usage text, no ImportError |
| 5 | repo root: `python -m rinalmo_hub.adapter outputs/splice_donor_lora/splice_site_adapter.pt` | the 6.08 MB adapter summary |
| 6 | `git status` | renames (`R`) for every moved path — no delete+add pairs; new files: root `README.md`, `ui/README.md` |
| 7 | grep of `engine/README.md` | no remaining run-from-root example says bare `configs/` or `ft_schedules/` |

## 6. Risks and rollback

- **Rollback is trivial** until the commit: `git reset --hard HEAD` restores the old
  layout (runtime dirs are untouched throughout), then `pip install -e .` from the root
  if the environment step already ran. No data, weights or outputs are moved or modified
  at any point.
- **Stale bytecode/caches**: delete leftover `__pycache__`/`.pytest_cache` at the old
  locations so nothing shadows the moved packages.
- **The old-repo coupling**: uninstalling the stale `rinalmo` dist could in principle
  affect workflows in `~/bio2/RiNALMo` that relied on this venv — that venv is this
  repo's `.venv`, and this repo's tests/CLI are its only known consumers, but flag it to
  the user in the close-out summary.
- **Hidden CWD dependence**: §3's analysis found none outside the listed doc/test
  strings; step 4 is the safety net, and the fix policy is "adjust the test/doc, never
  patch core files".

## 7. Explicitly out of scope

Any engine code change (none is needed — §1), the ToolHub (Phase 2), moving or renaming
`outputs/` contents, engine dependency upgrades, and committing/pushing without the
user's go-ahead.
