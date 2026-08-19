# Testing

Strategy, how to run each suite, and what every test file actually guards.

Measured in this checkout: **engine 135 passed / 7 deselected** (2026-08-13, unchanged by
Phase 13 — `engine/` is untouched, D1), **agentic 633 tests** collected
(`pytest tests/ --collect-only`, 2026-08-19), up from 381 before Phase 13.

---

## Contents

1. [Strategy](#1-strategy)
2. [Running the suites](#2-running-the-suites)
3. [Markers](#3-markers)
4. [Shared fixtures and test doubles](#4-shared-fixtures-and-test-doubles)
5. [The engine suite](#5-the-engine-suite)
6. [The agentic suite](#6-the-agentic-suite)
7. [Scenario tests](#7-scenario-tests)
8. [Controls and catches](#8-controls-and-catches)
9. [Writing a new test](#9-writing-a-new-test)

---

## 1. Strategy

Three tiers, and the bulk of the value is in the first:

| Tier | What | Cost |
|---|---|---|
| **Deterministic services** — registry, JobRunner, profiler, recommender, analyzer, wrappers, harness, stores | Ordinary unit tests on a randomly-initialised `nano` backbone and tiny synthetic data | Seconds on CPU. **No weights, no GPU, no datasets, no API key, no network.** |
| **Agent graphs** | Agents are kept thin, so most behaviour is testable below the LLM. Wiring and gates are tested with a **scripted** chat model; a scenario suite of recorded conversations guards the flows. | Seconds |
| **End-to-end** — a real miniature fine-tune, a GPU forward/backward, a browser | Written, marked, documented, **user-run** | Minutes to hours |

The rule that makes tier 1 possible: **every heavy import is inside a function**, and the
API key is required at model *construction*, never at import. So `pytest` never needs a
credential, and importing the agentic package never pulls in torch.

Everything runs on `nano` — 6 blocks, width 320, random initialisation. It is not a small
version of the real model; it is a structurally identical one, which is exactly what
structural tests need.

## 2. Running the suites

```bash
# Engine — CPU only, ~18 s
cd engine && ../.venv/bin/python -m pytest

# Agentic — CPU only, no network, no API key, ~2m40s
cd agentic && ../.venv/bin/python -m pytest

# One file, verbosely
../.venv/bin/python -m pytest tests/test_registry.py -v

# The opt-in browser suite (~150 MB Chromium)
../.venv/bin/python -m pip install playwright
../.venv/bin/python -m playwright install chromium
../.venv/bin/python -m pytest -m ui
```

Both `pyproject.toml`s configure `testpaths = ["tests"]`, so run `pytest` **from the layer
directory**, not from the repo root. The engine also sets `pythonpath = ["."]`.

## 3. Markers

Configured as default *exclusions* in `addopts`, so opting in is explicit.

### Engine — `addopts = "-ra -m 'not gpu and not weights and not data'"`

| Marker | Needs | Run with |
|---|---|---|
| `gpu` | A CUDA device | `pytest -m gpu` |
| `weights` | `giga-v1.pt` | `RINALMO_WEIGHTS=… pytest -m weights` |
| `data` | A downloaded dataset | `RINALMO_SPLICE_TEST_DATA=… pytest -m data` |

`tests/test_user_run.py` holds all seven of these and documents each environment variable in
its own docstring. The most valuable is the **regression test** (`weights and data`): it
scores a known donor adapter on the real benchmark and is the strongest guarantee the port
preserved upstream behaviour.

### Agentic — `addopts = "-ra -m 'not ui'"`

| Marker | Needs | Run with |
|---|---|---|
| `ui` | Playwright + Chromium and a live server | `pytest -m ui` |

Browser tests get the same treatment as the engine's GPU tests: *a ~150 MB download has no
business being a prerequisite for the default suite*.

## 4. Shared fixtures and test doubles

### `engine/tests/helpers.py` + `conftest.py`

`LM_CONFIG = "nano"`, `NUM_BLOCKS = 6`, per-task `HEAD_CONFIGS` small enough that a full
forward pass on CPU stays instant, and `LORA_CONFIGS` covering several geometries (stride 1
vs 3, different ranks) — which is what lets the hub tests prove that adapters with
*different* geometries coexist. Session-scoped `alphabet`, `sequences` and `tokens` fixtures.

### `agentic/tests/conftest.py`

Builds real `nano` adapter files **through the engine's public API only** — mirroring the
engine's own test philosophy. Since Phase 13, D1 forbids the agentic test suite from
importing the engine's shipped tasks (`splice_site`, `mrl`, `sec_struct` — even as test
doubles), so `conftest.py` defines its own tiny, neutral `BaseDownstreamModule` subclasses
(`_DemoBinaryModule`, `_DemoRegressionModule`) purely for the fixtures below to build real
adapters from:

| Fixture | Provides |
|---|---|
| `nano_splice_adapter` | A real `.pt` on `nano`, built from a locally-defined binary-classification module (no engine task import) |
| `nano_regression_adapter` | The pad-sensitive counterpart, built from a locally-defined regression module (exercises `ADAPTER_EXTRA_PREFIXES`) |
| `full_ft_export` | A head-only export — **what the registry must refuse** |
| `nano_registry` | A fresh `Registry` on a `tmp_path`, configured for a random `nano` backbone |

This swap is safe precisely because nothing about *what* the fixtures exercise changed:
the demo modules go through the same `BaseDownstreamModule` construction order, the same
LoRA injection, the same adapter save/load path as any real task — they are structurally
identical to `splice_site`/`mrl`, just written locally instead of imported from `engine/`.
No engine code changed, and the test suite gets *more* honest, not less: it now proves the
adapter mechanics work for an arbitrary subclass, rather than only for the three the engine
happens to ship.

`_build_nano_module` deliberately moves every trainable tensor off its initialised value
(`lora_B` starts at zero), so saved adapters carry distinguishable state and a round-trip
test can actually fail.

### `agentic/tests/scripted_model.py`

```python
class ScriptedChatModel(BaseChatModel):   # honours bind_tools; records calls
scripted([AIMessage(...), ...])
tool_call(name, args, call_id="call_1")
```

The single test double behind every graph test. It is why 381 tests run with no network and
no credential.

### `agentic/tests/fixtures/`

* `broken_task_sources.py` — deliberately defective task modules, **one per failure mode**,
  which the harness must fail.
* `target_type_tasks.py` — hand-written, known-good fixture tasks, one per supported target
  type (binary/multiclass/regression). These are `test_harness.py`'s **PASS controls**
  since Phase 13 — the shipped engine tasks can no longer serve that role (D1 forbids
  importing them from `agentic/tests/`), so a control needed a replacement that does not
  depend on the engine's own task set. `codegen/prompts.py`'s worked example is also built
  from this same rendered shape, so the fallback prompt's one example and the harness's
  controls stay consistent with each other by construction.
* `template_specs.py` — synthetic `DatasetSpec`s used by the golden-file and coverage tests
  for `codegen/templates/`.
* `dummy_external.py` / `validating_external.py` — wrapper modules with no real package
  behind them, for contract and golden-runner tests (the fixture wrapper
  `test_external_fixture_wrapper.py` exercises in place of the deleted `vienna.py`).

### `agentic/tests/api_helpers.py`

SSE collection helpers, so HTTP tests can assert on frame sequences.

## 5. The engine suite

| File | Guards |
|---|---|
| `test_registry.py` | The registry sees exactly the three shipped tasks, and each constructs |
| `test_lora_injection.py` | Injection adapts **exactly** the intended modules and freezes everything else. A silent zero-match would train only the head and look exactly like "LoRA doesn't work"; a missing freeze would quietly train the whole backbone at LoRA learning rates |
| `test_adapter_roundtrip.py` | Per-task adapter round trips, plus the guards that stop a bad load. Both failure modes covered produce *plausible* numbers rather than errors |
| `test_hub.py` | Three adapters resident in one hub, each task's output **bit-identical** to that task loaded alone; and that adapters with different geometries coexist — the assumption the whole hub design rests on |
| `test_config_and_cli.py` | Config resolution and CLI wiring. The `--set optim.lr=3e-4` case is the one to stare at: YAML 1.1 parses it as a *string* |
| `test_training_loop.py` | End-to-end `trainer.fit` on CPU with synthetic data: the generic steps, metric aggregation, the MRL scaler fit, checkpoint slimming, gradual unfreezing, `initial_denom_lr` |
| `test_cost.py` | `CostProfiler` on CPU with synthetic data: warm-up exclusion, Lightning's sanity-check batches never leaking into the val accumulator, `null` (not a crash) when a stage has zero measured batches, `run_summary.json` round-trips |
| `test_new_task_acceptance.py` | **The acceptance test for the abstraction**: a fourth task is added by three files under `examples/`, no core file is edited, and a check asserts that no core file so much as *mentions* the task name |
| `test_user_run.py` | The seven opt-in tests (GPU / weights / data), including the regression test |

## 6. The agentic suite

### Tool-Hub

| File | Guards |
|---|---|
| `test_manifest.py` | Pure data layer — no engine, no torch |
| `test_registry.py` | Lifecycle; the engine is used only to validate adapter files, no backbone loads |
| `test_registry_atomicity.py` | Registration touches an artifact copy **and** the manifest, so a failure between them must leave neither an orphan file nor a dangling entry |
| `test_runtime.py` | Lazy backbone, routing-level deactivation, smoke tests — with a real forward pass on a random `nano` |
| `test_store_concurrency.py` | Both JSON stores refuse a blind second write. The guard is a revision counter *inside* the file, not an mtime |
| `test_process_identity.py` | Every branch of "is this PID still ours?" — gone, permission-denied, zombie, recycled, and a record with no start time |
| `test_doctor.py` | Every check, against a purpose-built **broken** install |
| `test_prune.py` | Dry run by default; nothing the manifest references is ever deleted |
| `test_toolhub_cli.py` | The management CLI end to end against tmp data dirs on `nano` |
| `test_external_contract.py` / `test_external_registry.py` / `test_external_cli.py` | The wrapper contract, the golden runner, install helpers (no installs, no PyPI), manifest backward compatibility, and the approval-gated install — all against fixture wrappers, never a shipped one |
| `test_external_fixture_wrapper.py` | Replaces the deleted `test_vienna_wrapper.py` (D2 — the wrapper it tested, `vienna.py`, no longer exists): `contract.load_spec`, validation-before-import, and `run_golden` exercised end to end against a minimal fixture wrapper under `tests/fixtures/` |

### Agents and graphs

| File | Guards |
|---|---|
| `test_graph_wiring.py` | The load-bearing Phase-0 test: a scripted model drives the full loop — tool binding, `ToolNode` execution, loop termination |
| `test_orchestrator_graph.py` | Tool loops, the activate-then-use-in-one-turn policy, turn survival |
| `test_approval_gate.py` | **The absence of a side effect at the interrupt** — if the gate were inside the tools node, the tool would already have executed |
| `test_tool_factory.py` | Names, schemas, shared-state mutation, refusals |
| `test_chat_sessions.py` | The checkpointer owns history: a thread accumulates, threads are isolated, and a fresh graph on the same file resumes (the process-restart proof without a process restart) |
| `test_settings.py` | Settings resolution and API-key gating |
| `test_hello_tool.py` | `gc_content` validation |

### Pipeline

| File | Guards |
|---|---|
| `test_profiler.py` | The one-table reader — column/target-type detection, delimiter sniffing, headerless-file detection, split-column detection, `mode: "file"` proposal from a `validation_path`, the quality/leakage warnings, `similar_tasks` — synthetic fixtures, deterministic, no engine, no real datasets. No layout matching: there is nothing left to match against |
| `test_dataset_spec.py` | `DatasetSpec` validation: bad fractions, an unknown column, an unsupported target type, a `task_name` collision, mode switching (including into/out of `file` mode), an edited `format.header`/`format.separator` taking effect on re-validation, and that `confirm_data_profile`'s approval recomputes `row_counts`/`classes`/`head` from the file rather than trusting the proposal |
| `test_profile_gate.py` | `confirm_data_profile` refuses a spec that is not stamped `source: "profile_dataset"`; the interrupt fires before any spec is (re-)stamped; a decline leaves nothing behind |
| `test_approval_edits.py` | `_apply_edits`: whitelist (`EDITABLE_ARGS`) enforcement, type checking, unknown-path refusal; an edited training plan rebuilds `command`; `human_edits`/`human_overrides` recorded; a recorded failure mode produces its warning when the human's chosen value matches one |
| `test_recommender.py` | Spec-driven, table-driven, no free-floating constants, and **the command it materialises parses under the engine's own CLI parser** — no `layout_match`, `data.root` is the CSV's own file path |
| `test_generic_recommender.py` | The `generic.derived` rules: batch size at each length band, the step-budget epoch rule at small/medium/large row counts, clamping, and that the rationale lines shown at the gate are generated from the same `why:` strings |
| `test_similar_tasks.py` | The reuse matcher (D9): scores landed `spec.json`s against a proposed spec, the "columns and target type both match" threshold, and that a missing/unreadable `spec.json` simply never matches rather than erroring |
| `test_knowledge.py` | `arms:`/`universal:` still carry their load-bearing numbers; `tasks:` is *absent* (there are no known tasks any more); `generic:` and `target_shapes.yaml` are read correctly |
| `test_job_runner.py` | Driven by a **fake command** that writes a `metrics.csv` the way the engine does and exits with a chosen code — exactly the interface the runner consumes |
| `test_analysis.py` | The two rules: a truncated run is never compared to a reference, and a within-tolerance difference is never called a regression |
| `test_analysis_baseline.py` | A task with no band is compared against this project's own earlier runs — and the wording distinguishes a *baseline* from a validated reference. Since Phase 13 this is the **only** path: `generic.reference.band` is always `null`, so every run is a baseline until its own task has a history |
| `test_pipeline_tools.py` | The eight pipeline tools as the agent sees them, including gate 1 (`confirm_data_profile`) |

### Codegen

| File | Guards |
|---|---|
| `test_harness.py` | **Controls and catches** — see below |
| `test_templates_render.py` | Golden-file tests: each of the three target types × two split modes, plus one `file`-mode and one headerless case, renders byte-for-byte expected output from `codegen/templates/`. Cheap, fast, no model — what makes a template change reviewable as a diff (`scripts/update_template_golden.py` regenerates the goldens) |
| `test_templates_cover.py` | `covers(spec)` accepts every spec gate 1 can produce and rejects a spec carrying a field or value the template does not declare it handles — the predicate is never allowed to claim coverage it lacks |
| `test_codegen_paths.py` | The template path renders and passes the harness with **no model call at all** (asserts the model is never invoked); a spec `covers()` rejects goes straight to the LLM path; a harness failure on rendered code **falls through** rather than retrying, recording `fell_back_from_template` with a reason |
| `test_runtime_validators.py` | Serving validators keyed by `target_type` (probabilities in [0,1]; a recorded class label; one finite float), not by task name — every generated task gets coverage now, not just the ones with a hand-written entry |
| `test_pipeline.py` | The bounded ToolSmith⇄Verifier fallback loop, driven by a fake structured-output model; the *real* harness verifies the replayed files for real |
| `test_sandbox.py` | Hangs, memory runaways and crashes all reported as data |
| `test_prompts.py` | What goes into the fallback prompt: the contract, the silent-failure rules, the approved spec, and **exactly one** `target_shapes.yaml` recipe — and asserts the prompt contains **no shipped task's name**, on the reasoning that a generator which never sees the contract writes code that fails check 1 |
| `test_codegen_tools.py` | Discovery, the codegen tools, the approval payload |
| `test_no_shipped_task_knowledge.py` | Greps every **file** under `agentic/` — source, docstrings, help text, comments, YAML, and the test suite itself — for the shipped task/dataset-source names (D1/D2), with no exemptions. This is the one test that keeps the "the system starts empty" claim true as the code evolves |

### HTTP and UI

| File | Guards |
|---|---|
| `test_api_sessions.py` | Streaming; session management (create / rename / delete, and their 404s and 409s); and the property the demo turns on: a session written by the terminal continues correctly through the API against the same file |
| `test_api_approval.py` | Over HTTP a gated action must not run before the human answers, and not at all if they decline |
| `test_approval_gate.py` | The gate itself, including tool-state changes: an activation must not touch the manifest at the interrupt, and **a decline must leave the tool disabled** |
| `test_api_tools.py` | Every endpoint wraps CLI behaviour — **including its refusals**, which must arrive as status codes carrying the CLI's own message |
| `test_api_jobs.py` | Job endpoints and the error-mapping table |
| `test_api_concurrency.py` | Overlapping predictions for different adapters cannot answer from the wrong one |
| `test_api_security.py` | Loopback default, the refusal to bind elsewhere without a token, the 401, the `/health` exemption |
| `test_ui_serving.py` | The client is served and is genuinely **self-contained** — the offline check is the one with teeth |
| `test_ui_contract.py` | **The compiler this pair of languages does not have**: every field and event `ui/*.js` reads by name, so a server-side rename fails in `pytest` naming the client file — including gate 1's `spec`/`warnings`/`similar_tasks` detail fields and the `edits` payload the approval modal's spec/plan form sends back. Also pins two client-side literals against the server — every `LOG_TAIL_CHOICES` value the tail dropdown offers, and every job state `render.js` must style |
| `test_ui_browser.py` | Opt-in. The only tests that prove the JavaScript actually runs — streaming, the modal, the session rail's create/rename/delete, its resize surviving a reload, the thinking dots dark at rest, the activity bar swapping the rail and the centre column, and the grip resizing from the rail's own left edge |

## 7. Scenario tests

`tests/test_scenarios.py` replays `tests/scenarios/*.yaml` against a scripted model.
Scenarios are **data**, so a new flow is a new file rather than new code:

| Scenario | Covers |
|---|---|
| `inference.yaml` | Flow A: list tools, then a real prediction through a demo adapter tool |
| `management.yaml` | Flow B: lifecycle operations |
| `training_gate.yaml` | The approval gate around `start_training` |
| `failure_paths.yaml` | The documented refusals |

Since Phase 13, the tools these scenarios script are neutral fixture tools
(`demo_binary`, `dummy_add`, …) rather than the shipped tasks — D11 covers test data too,
so a scenario naming a shipped task would fail `test_no_shipped_task_knowledge.py`:

```yaml
turns:
  - user: What tools are available?
    script:
      - tool_calls: [{name: list_tools, args: {}}]
      - text: You have a demo binary-classification adapter and two dummy tools.
    expect_tools: [list_tools]
    expect_tool_result_contains: [demo_binary, dummy_add]
```

They pin **wiring and contracts, not prompts** — the model is scripted, so what is asserted
is what the graph and the tools do with a given sequence of model outputs. Prompt regressions
are a different problem, and a live suite's job.

## 8. Controls and catches

The most transferable idea in this suite: any component that *judges* other code must be
tested in **both** directions.

| Component | Control (must pass) | Catch (must fail) |
|---|---|---|
| Verification harness | `fixtures/target_type_tasks.py` — one hand-written, known-good fixture task per supported target type runs through the **full** harness (not just structural checks, since these fixtures ship real data) in CI. *A harness that fails a known-good task is broken.* Replaces the shipped-task control the harness used before Phase 13: D1 forbids importing `engine/rinalmo_hub/tasks` from `agentic/tests/`, so the control had to become fixtures the agentic suite owns itself — checked against the same `broken_task_sources.py` catches below before the swap was trusted (plan §17, Stage 0) | `fixtures/broken_task_sources.py` — one deliberately defective task per failure mode, each of which must fail its specific check |
| `doctor` | A healthy install reports `ok` | A purpose-built broken install, one fault per check. *A health check that reports green on a broken install is worse than no health check.* |
| `prune` | Removes what it should | Never removes anything the manifest references, and defaults to a dry run |

## 9. Writing a new test

| You are testing… | Do this |
|---|---|
| A deterministic service | Plain unit test. Use `nano_registry` / `tmp_path`; never load `giga`. |
| Anything needing an adapter file | `nano_splice_adapter` / `nano_mrl_adapter`, never a real artifact |
| Graph behaviour | `scripted([...])` from `scripted_model.py` |
| A new conversational flow | A YAML file in `tests/scenarios/` — no new Python |
| An HTTP endpoint | `create_app(services=build_services(model=scripted(...), data_dir=tmp_path))`; `api_helpers.py` for SSE |
| Something needing a GPU, weights or a dataset | Mark it `gpu` / `weights` / `data`, read its inputs from environment variables, and document them in the module docstring |
| A field the browser reads | Add it to `test_ui_contract.py` **as well as** the server test — that file is the only thing standing between a rename and a blank panel |

Two habits worth copying from the existing suite:

* **Assert on the message, not just the exception type.** Almost every refusal in this
  codebase carries the fix in its text, and that text is a contract with three front ends.
* **Say why the test exists in its docstring.** Nearly every file here opens with the failure
  it was written to prevent, which is what makes the suite readable as documentation.
