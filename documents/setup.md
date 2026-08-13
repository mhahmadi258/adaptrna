# Setup

Runtime requirements, installation, environment, and how to prove the install works.

---

## Contents

1. [Requirements](#1-requirements)
2. [Install](#2-install)
3. [Credentials](#3-credentials)
4. [Point the hub at a backbone](#4-point-the-hub-at-a-backbone)
5. [Verify](#5-verify)
6. [Dependencies, and which layer needs which](#6-dependencies-and-which-layer-needs-which)
7. [Optional components](#7-optional-components)
8. [Hardware](#8-hardware)
9. [What you need for which task](#9-what-you-need-for-which-task)
10. [Packaging caveat](#packaging-caveat)

---

## 1. Requirements

| | Required | Notes |
|---|---|---|
| Python | **≥ 3.10** (both packages declare this) | This checkout runs **3.12.3** |
| OS | Linux | `jobs/runner.py` reads `/proc/<pid>/stat` for PID identity, and `codegen/sandbox.py` uses `resource.setrlimit` + `os.setsid`. Neither has a portable fallback. |
| Disk | ~10 GB | 2.6 GB backbone + datasets + run outputs |
| GPU | Only for training and fast inference | Everything else, including both test suites, runs on CPU |
| Network | Only for the Anthropic API, dataset/weight downloads and gated package installs | The web UI itself is fully offline — no CDN assets |

## 2. Install

```bash
cd /path/to/adaptrna
python -m venv .venv && source .venv/bin/activate

# Both layers, editable, from the repo root. Order does not matter.
python -m pip install -e ./engine -e ./agentic

# CUDA only — must match your torch build. Imported lazily; without it training on
# CUDA falls back to a much slower plain-PyTorch attention path.
python -m pip install flash-attn --no-build-isolation

# For the test suites
python -m pip install -e "./engine[dev]" -e "./agentic[dev]"
```

**Editable installs are the supported path**, and the only one — see
[Packaging caveat](#packaging-caveat). Both packages must be installed: the agentic layer
declares no dependency on the engine (deliberately, so it imports in milliseconds), but
resolves it lazily at call time.

Confirm both are linked to the checkout:

```bash
python -m pip list | grep -E "adaptrna-agentic|rinalmo-hub"
# adaptrna-agentic  0.1.0  /path/to/adaptrna/agentic
# rinalmo-hub       0.1.0  /path/to/adaptrna/engine
```

## 3. Credentials

Only the LLM-backed paths need a key. The ToolHub CLI, both test suites and every
deterministic service run without one — the check happens at **model construction**, never
at import.

```bash
# Repo root, git-ignored:
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

`Settings.from_env()` loads `<repo>/.env` when it exists, with `override=False` — **real
environment variables always win over the file**. Missing key produces one actionable
message naming the file to put it in.

## 4. Point the hub at a backbone

The engine's `configs/base.yaml` defaults `pretrained_weights` to `weights/giga-v1.pt`
relative to the working directory — a path that need not exist. The ToolHub carries the
real location, and every training plan takes the checkpoint from **the manifest**, so a run
always trains against exactly the backbone the hub serves.

```bash
# Download once (caches in ~/.cache/rinalmo_pretrained/)
python -c "from rinalmo.pretrained import get_pretrained_model; get_pretrained_model('giga-v1')"

# Tell the hub where it is
python -m adaptrna_agentic.cli.toolhub config --weights ~/.cache/rinalmo_pretrained/giga-v1.pt
```

Direct download, if you prefer:
[`giga-v1.pt`](https://drive.google.com/file/d/1-E2Ziu2VFDAgwCmQvVeAviGtsQQ94L3L/view).

A hub with no checkpoint configured does not fail silently: `doctor` warns, and the
recommender emits a warning that training would start from randomly initialised weights.

## 5. Verify

```bash
# 1. The install, top to bottom — read-only, changes nothing
python -m adaptrna_agentic.cli.toolhub doctor

# 2. Engine suite: CPU, no weights, no datasets, ~18 s
cd engine && python -m pytest && cd ..

# 3. Agentic suite: CPU, no network, no API key, ~2m40s
cd agentic && python -m pytest && cd ..
```

Expected in a healthy checkout (measured here on 2026-08-13):

```
engine:  135 passed, 7 deselected      (gpu / weights / data markers)
agentic: 381 passed, 11 deselected     (ui marker)
```

`doctor` output is a list of `ok | WARN | FAIL` lines, each failure followed by the command
that fixes it. It is the first thing to run whenever anything looks off. See
[workflows/operations.md](workflows/operations.md#1-doctor--start-here).

## 6. Dependencies, and which layer needs which

### `engine/` (`rinalmo-hub`)

| Package | Why |
|---|---|
| `torch>=2.0.0` | everything |
| `lightning>=2.0.0` | `LightningModule`, `Trainer`, datamodules, callbacks |
| `peft>=0.11.0` | LoRA injection (`inject_adapter_in_model`, `BaseTunerLayer`) |
| `torchmetrics>=1.0.0` | task metrics |
| `numpy`, `pandas`, `scikit-learn` | data handling, splits, the MRL scaler's neighbourhood |
| `ml_collections` | the backbone's `ConfigDict` |
| `einops` | attention/head reshapes |
| `PyYAML` | config layering, ft schedules |
| `gdown`, `requests`, `tqdm` | weight and dataset downloads |
| *(extra)* `dev` → `pytest>=7.0` | the test suite |

### `agentic/` (`adaptrna-agentic`)

| Package | Why |
|---|---|
| `langgraph>=1.0` | the graph, `interrupt()`, checkpointing |
| `langgraph-checkpoint-sqlite>=3.0` | `SqliteSaver` — sessions shared across front ends |
| `langchain>=1.0` | `init_chat_model`, `StructuredTool`, message types |
| `langchain-anthropic` | resolved by `init_chat_model` from the `anthropic:` prefix; **never imported directly** anywhere in the codebase |
| `python-dotenv` | `.env` loading |
| `fastapi>=0.110`, `uvicorn>=0.27` | the HTTP service |
| *(extra)* `dev` → `pytest>=7.0` | the test suite |

**Deliberately absent from `agentic/pyproject.toml`:** any dependency on the engine,
`torch`, `lightning` or `pandas`. That keeps the package importable in milliseconds, and
every heavy import is inside a function. The practical consequence: modules such as
[`profiling/profiler.py`](../agentic/adaptrna_agentic/profiling/profiler.py),
[`jobs/runner.py`](../agentic/adaptrna_agentic/jobs/runner.py) and
[`jobs/analysis.py`](../agentic/adaptrna_agentic/jobs/analysis.py) call `import pandas`
at runtime, and [`toolhub/runtime.py`](../agentic/adaptrna_agentic/toolhub/runtime.py)
calls `from rinalmo_hub.hub import RiNALMoHub`. Installing the engine satisfies all of
them; installing only the agentic package produces a clear `ToolHubError` naming
`pip install -e ./engine`.

### Versions resolved in this checkout

| | |
|---|---|
| `torch` 2.10.0 · `lightning` 2.6.1 · `peft` 0.20.0 · `torchmetrics` 1.9.0 | `numpy` 2.4.1 · `pandas` 3.0.2 · `scikit-learn` 1.9.0 |
| `langchain` 1.3.15 · `langgraph` 1.2.11 · `langchain-core` 1.5.4 | `langchain-anthropic` 1.5.5 · `anthropic` 0.121.0 |
| `langgraph-checkpoint-sqlite` 3.1.1 | `fastapi` 0.136.0 · `uvicorn` 0.45.0 |
| `flash_attn` 2.3.2 | `ViennaRNA` 2.7.2 · `playwright` 1.62.0 · `pytest` 9.1.1 |

## 7. Optional components

| Component | Install | Needed for |
|---|---|---|
| `flash-attn` | `pip install flash-attn --no-build-isolation` | Fast CUDA attention. Lazy import; without it CPU tests still pass and CUDA training uses a slow fallback. **Its backward pass is non-deterministic** — see [architecture.md §11](architecture.md#11-engine-constraints-that-shaped-this-layer). |
| `ViennaRNA` | Through the gated flow: `toolhub register-external adaptrna_agentic.toolhub.external.vienna` (shows the exact `pip` command, then asks) | The `vienna_fold` / `vienna_cofold` tools. Wrapped packages are **tool** dependencies and are deliberately absent from every `pyproject.toml`. |
| `playwright` + Chromium | `pip install playwright && playwright install chromium` (~150 MB) | The 11 opt-in browser tests (`pytest -m ui`) |
| Datasets | `python -m rinalmo_hub.cli.train --task <t> --prepare_data --set trainer.max_steps=0` | Real training on the shipped tasks. Each dataset downloads itself into `dataset/`. |

## 8. Hardware

| Activity | Needs |
|---|---|
| Both test suites, `doctor`, registry operations, profiling, the recommender | CPU only |
| CPU inference on a `nano` backbone | CPU (this is how the tests exercise real forward passes) |
| Serving `giga` adapters | ~3 GB RAM/VRAM for the backbone, plus megabytes per resident adapter. Serving runs **fp32** — `dtype: auto` resolves to the model default, because non-autocast bf16 trips a dtype promotion in the engine's `TokenDropout`. |
| Real fine-tuning | One CUDA GPU. Reference timings in the knowledge base come from an H200: splice-site LoRA **~7 min** (batch 32, 2 epochs, ~19.8k sequences); MRL **~5h45m** LoRA / ~6h41m full FT (batch 64, 51 epochs). |

One training job runs at a time by default — `JobRunner.start` refuses a second unless
`allow_concurrent` is passed, because two `giga` runs on one GPU is how you get an
out-of-memory failure forty minutes in.

## 9. What you need for which task

| I want to… | API key | Engine installed | GPU | Backbone checkpoint | Dataset |
|---|:---:|:---:|:---:|:---:|:---:|
| Run either test suite | – | ✅ | – | – | – |
| `toolhub list` / `config` / `doctor` / `prune` | – | ✅ | – | – | – |
| `toolhub predict` / `test` on a registered adapter | – | ✅ | optional | ✅ | – |
| `toolhub call vienna_fold …` | – | – | – | – | – |
| Terminal chat, HTTP API, web UI | ✅ | ✅ | optional | ✅ | – |
| Profile data / get a recommendation | ✅ (for chat) | ✅ | – | ✅ | your own |
| Fine-tune | ✅ | ✅ | ✅ | ✅ | ✅ |
| Generate a new task (flow D) | ✅ | ✅ | – (verification is CPU) | – | ✅ your own |

## Packaging caveat

[`agentic/pyproject.toml`](../agentic/pyproject.toml) lists its packages **explicitly**, and
the list is incomplete:

```toml
packages = [
    "adaptrna_agentic", "adaptrna_agentic.agents", "adaptrna_agentic.api",
    "adaptrna_agentic.api.routers", "adaptrna_agentic.cli", "adaptrna_agentic.jobs",
    "adaptrna_agentic.knowledge", "adaptrna_agentic.profiling", "adaptrna_agentic.toolhub",
]
```

Missing: **`adaptrna_agentic.codegen`** and **`adaptrna_agentic.toolhub.external`**. There
is also no `[tool.setuptools.package-data]` entry, so `knowledge/*.yaml` — which the
ConfigRecommender cannot function without — would not be included in a wheel either.

Editable installs are unaffected (setuptools' editable finder maps the top-level package to
the source tree, so every subpackage resolves; verified in this checkout), which is why
this has never surfaced. But `pip install ./agentic` or any wheel build would produce a
package that fails on the first `create_task_tool` call, the first external-tool
registration, and the first recommendation.

Fix, if you ever need a distributable build:

```toml
packages = [
    ..., "adaptrna_agentic.codegen", "adaptrna_agentic.toolhub.external",
]
include-package-data = true

[tool.setuptools.package-data]
"adaptrna_agentic.knowledge" = ["*.yaml"]
```

(The engine's `pyproject.toml` does this correctly, including `package-data` for
`rinalmo/resources/*.json`.)
