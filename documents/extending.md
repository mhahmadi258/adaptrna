# Extending the Project

"I want to change X — where do I edit?" Answers, in the order a new developer usually asks
them.

---

## Quick index

| I want to… | Edit | Also update |
|---|---|---|
| [Add a new engine task](#add-a-new-engine-task) | `engine/rinalmo_hub/tasks/<t>.py` + `tasks/__init__.py` + `engine/configs/tasks/<t>.yaml` | `knowledge/*.yaml`, engine tests |
| [Add a task without touching the engine](#add-a-task-without-touching-the-engine) | `adaptrna_custom/tasks/<t>/` | — (discovery is automatic) |
| [Add an agent tool](#add-an-agent-tool) | `agents/tool_factory.py` | `MANAGEMENT_TOOL_NAMES`, the system prompt, tests |
| [Gate a new action behind approval](#gate-a-new-action-behind-approval) | `GATED_TOOLS`, `_summarize`, `_details` | `cli/chat.py::_prompt_approval`, `ui/render.js`, `test_ui_contract.py` |
| [Change a hyperparameter recommendation](#change-a-hyperparameter-recommendation) | `knowledge/hyperparameters.yaml` | `test_knowledge.py` |
| [Teach the profiler a new layout](#teach-the-profiler-a-new-layout) | `knowledge/task_templates.yaml`, `profiling/profiler.py` | `test_profiler.py` |
| [Add an HTTP endpoint](#add-an-http-endpoint) | `api/routers/<r>.py` | `api/schemas.py`, tests; `test_ui_contract.py` if the UI reads it |
| [Add a UI panel](#add-a-ui-panel) | `ui/index.html`, `render.js`, `app.js` | `test_ui_contract.py` |
| [Wrap a classical package](#wrap-a-classical-package) | a new module following `external/vienna.py` | register it; goldens |
| [Add a harness check](#add-a-harness-check) | `codegen/_harness_runner.py` | `STRUCTURAL_CHECKS` / `REQUIRED_FOR_GENERATED`, `fixtures/broken_task_sources.py` |
| [Add a doctor check](#add-a-doctor-check) | `toolhub/doctor.py` | `test_doctor.py` — with a broken install |
| [Support a second backbone](#support-a-second-backbone) | `manifest.py`, `runtime.py` | a lot; see below |
| [Swap the model provider](#swap-the-model-provider) | `settings.py` (or just an env var) | nothing else |

---

## Add a new engine task

Three files, and **no edits to any core file**. `engine/examples/ncrna_classification/` is a
complete worked example, and `engine/tests/test_new_task_acceptance.py` asserts the
abstraction holds — including that no core file so much as mentions the task's name.

1. **`engine/rinalmo_hub/tasks/my_task.py`** — a `@register_task("my_task")` subclass of
   `BaseDownstreamModule` implementing `build_head`, `extract_features`, `compute_loss`,
   `update_metrics`, `compute_metrics` and the `build_datamodule` staticmethod.
2. **One line in `engine/rinalmo_hub/tasks/__init__.py`** so the decorator fires on import.
3. **`engine/configs/tasks/my_task.yaml`** — anything unset is inherited from `base.yaml`.
4. A `LightningDataModule` yielding `(tokens, target)` batches (override `batch_tokens` if
   your batch is shaped differently).

Full walkthrough with code: [`../engine/README.md` §4](../engine/README.md#4-adding-a-new-task).
The contract table: [modules/engine-hub.md](modules/engine-hub.md#the-subclass-contract).

### The two questions to answer before you ship it

Both fail **silently** — the adapter loads without error and the numbers look plausible:

1. **Does the task own state that predictions depend on but that is not a head weight?**
   A tensor or buffer → add its prefix to `ADAPTER_EXTRA_PREFIXES` (see `mrl`'s
   `("scaler.",)`). A plain Python value → implement `adapter_extra_payload()` and
   `load_adapter_extra()` (see `sec_struct`'s threshold).
2. **Does the head need CLS, EOS or padded positions excluded?** `extract_features` is the
   only place that happens, which is why it is an explicit hook rather than a convention.

Prove your answer to question 1 rather than asserting it:

```python
from adaptrna_agentic.codegen.harness import verify_task, summarize
print(summarize(verify_task("my_task", config_path="engine/configs/tasks/my_task.yaml")))
```

Check 6 (`adapter_roundtrip`) fails if any prediction-affecting state is missing from the
file.

### Then make the platform aware of it

| Add to | Why |
|---|---|
| `knowledge/hyperparameters.yaml` → `tasks.my_task` | Otherwise the recommender falls back to the generic entry: arm settings still apply, but there is no reference band and the analyzer treats every run as a baseline. Add `primary_metric`, `reference.band`, `wall_clock`, `defaults`, `caveats`. |
| `knowledge/task_templates.yaml` → `templates` | So `profile_dataset` can **match** your layout (`data_layout.required_paths`) and validate task options (`key_options`) |
| `registry.PAD_SENSITIVE_TASKS` | Only if the head sees padding, so serving is forced to batch size 1 |
| `tool_factory._TASK_OUTPUT_NOTES` | So the model knows what the tool returns |
| `runtime._VALIDATORS` | A per-task smoke-test output-form validator |

## Add a task without touching the engine

Generated tasks land in `adaptrna_custom/tasks/<name>/` and are discovered automatically —
[`discovery.load_all()`](modules/codegen.md#6-discoverypy) imports every `tasks/*/task.py`
before training, serving or verification. **No registration step, no restart.**

You can also write one there by hand: same three files, plus a `__init__.py`. The
recommender prefers `adaptrna_custom/tasks/<t>/config.yaml` over
`engine/configs/tasks/<t>.yaml`, so it is picked up for training too.

Use this when the task is specific to your data. Use `engine/rinalmo_hub/tasks/` when it is
a general capability the framework should ship.

## Add an agent tool

In [`agents/tool_factory.py`](../agentic/adaptrna_agentic/agents/tool_factory.py), add a
plain function inside the appropriate group (`_management_tools`, `_pipeline_tools` or
`_codegen_tools`) and list it in that group's `StructuredTool.from_function` loop:

```python
def my_operation(name: str, count: int = 1) -> dict:
    """One-line description — this is what the MODEL reads to decide whether to call it.

    Longer detail goes here; it is part of the tool description.
    """
    return registry.do_something(name, count)
```

Then:

1. Add the name to `MANAGEMENT_TOOL_NAMES` — it is the collision guard against a registered
   tool of the same name.
2. Mention it in `orchestrator.SYSTEM_PROMPT` if the model needs to know *when* to use it.
3. Add a test in `test_tool_factory.py` (schema, name) and, if it is part of a flow, a
   scenario YAML.

Conventions that matter:

* **The docstring is the tool description.** Write it for the model.
* Argument types drive the inferred schema; keep them JSON-scalar.
* Raise `ToolHubError` / `ValueError` / `KeyError` — `_surface_errors` turns them into
  `ToolException`s that come back as actionable results rather than killing the turn.
* Put the **fix** in the error message.

## Gate a new action behind approval

Anything that spends money, changes the world outside the process, or adds a servable
capability should be gated.

1. Add the tool name to `GATED_TOOLS` in `tool_factory.py`.
2. Add a branch to `orchestrator._summarize` — one line saying what approving it will do.
3. Add a branch to `orchestrator._details` — **everything the human needs to judge it**: the
   exact command, the file list, the diff. Not a paraphrase.
4. Render the new detail fields in `cli/chat.py::_prompt_approval` and in
   `ui/render.js::approvalBody`.
5. Pin the field names in `test_ui_contract.py`, and assert the **absence of the side
   effect** at the interrupt in `test_approval_gate.py`.

Do not put `interrupt()` inside the tool. The tools node runs every call of a turn in one
pass, so a resume would re-execute the ones that already completed; the dedicated node is
idempotent because it contains only the interrupt.

## Change a hyperparameter recommendation

Edit [`knowledge/hyperparameters.yaml`](../agentic/adaptrna_agentic/knowledge/hyperparameters.yaml).
**Do not touch `recommender.py`** — it is table-driven precisely so that a change here is a
change everywhere, including the rationale shown to the user.

Every setting should carry *why*, which in practice means *what went wrong when it was
different*:

```yaml
failure_modes:
  - setting: "optim.lr = 1e-3"
    symptom: "Trains well to loss 0.126 by step ~325, then a gradient spike (norm 3.3e+02)
              collapses it into a constant-output state it never escapes."
    remedy: "Use lr 3e-4 with trainer.gradient_clip_val 1.0."
```

Those three fields are rendered into the plan's rationale *and* reused by the analyzer as
the remedy when a run shows that signature. Then update `test_knowledge.py`, which exists to
fail loudly if a load-bearing number is dropped.

The file is `lru_cache`d — restart the process after editing.

## Teach the profiler a new layout

1. Add a template to
   [`knowledge/task_templates.yaml`](../agentic/adaptrna_agentic/knowledge/task_templates.yaml)
   with `match` (target type, typical length), `shape` (head/loss/metrics/`extract_features`
   /prediction type) and `data_layout` (`description`, `required_paths`, `key_options`).
   `required_paths` is what `_match_layout` checks — exact existence under the root.
2. If the layout needs bespoke inspection (like the Spliceator `;`-separated headerless
   folds), add a branch to `profiler._profile_directory`.
3. Add a fixture-based test to `test_profiler.py`.

`shape` is not consumed by the profiler at all — it goes into the **ToolSmith prompt** as
"the closest known task shape", so write it as guidance for a code generator.

## Add an HTTP endpoint

1. Add the handler to the right router in `api/routers/`. Use `def`, **not `async def`** —
   Starlette runs sync handlers in its threadpool, which is what keeps a GPU-bound call from
   freezing the event loop.
2. Request bodies go in `api/schemas.py`; responses stay plain dicts (the same ones the CLI
   prints).
3. Let exceptions propagate — `app._install_error_handlers` already maps `ToolHubError` →
   409, `KeyError` → 404, and so on. Do not invent new wording; the whole point is that the
   browser shows what the terminal prints.
4. Add a test. If the UI will read it, add the field names to `test_ui_contract.py` too.

**Do not add a delete endpoint.** The absence of a delete surface is deliberate.

## Add a UI panel

Four small edits, in this order:

1. `ui/index.html` — the container element.
2. `ui/render.js` — a function returning the DOM for a row/section. Build nodes with `el()`;
   **never assign `innerHTML`**.
3. `ui/app.js` — a `refreshX()` that fetches and renders, wired into `refreshPanels()` or the
   boot sequence.
4. `ui/api.js` — the endpoint as a function, if it is new.

Then add the field names to `test_ui_contract.py`. That file is the only thing between a
server-side rename and a silently blank panel.

Watch the polling pattern: if your panel polls, **re-arm the timer in the failure branch
too**. A 409 while a store is mid-write is transient, and returning early froze the job
monitor permanently the first time a run started.

If you find yourself adding real client-side state, that is the tripwire the UI README names
— re-open the framework question rather than assuming.

## Wrap a classical package

Follow [`toolhub/external/vienna.py`](../agentic/adaptrna_agentic/toolhub/external/vienna.py)
exactly: module-level `SPEC`, one callable per declared function, and **validation before
importing the wrapped package**. Full guide: [workflows/external-tools.md](workflows/external-tools.md#5-writing-a-wrapper-by-hand).

Capture goldens against the installed version — never guess them — and prefer at least one
*a priori* case whose value is true of any correct implementation.

## Add a harness check

In [`codegen/_harness_runner.py`](../agentic/adaptrna_agentic/codegen/_harness_runner.py):

1. Add a `check_<name>` method raising `CheckFailed` (understood failure) or `CheckSkipped`
   (could not run). Anything else is caught and reported with a truncated traceback.
2. Register it in the ordered tuple inside `run()`.
3. Decide its tier: `STRUCTURAL_CHECKS` (no dataset needed — usable as a control over the
   shipped tasks) and/or `REQUIRED_FOR_GENERATED` (**must actually pass** before generated
   code may be approved; a `skip` counts as a failure).
4. **Add a broken fixture to `tests/fixtures/broken_task_sources.py`** that your check
   catches, and a control asserting the shipped tasks still pass.

Step 4 is not optional. A check with no catch test is a check nobody knows works.

Failure messages should name the remedy, the way check 6 does — *"Tensor state belongs in
`ADAPTER_EXTRA_PREFIXES`; plain Python values belong in `adapter_extra_payload()`…"*.

## Add a doctor check

Add a `_x_check()` returning a `Check(name, status, detail, remedy, data)` and call it from
`run_checks`. **The remedy must be a command that exists** — one currently is not (see
[README.md gap #1](README.md#known-documentation-gaps)).

Then test it against a *broken* install in `test_doctor.py`. A check that has only ever been
run against a healthy install has not been tested.

## Support a second backbone

Nothing in this project has ever served two, and that is the assumption most worth testing.
The naming is already model-agnostic and the manifest already records which checkpoint a tool
was built against, but you would need at minimum:

* `BackboneConfig` to become a collection, keyed by name, with `ToolEntry.lm_config` (and a
  new backbone id) selecting between them;
* `AdapterRuntime` to hold a hub **per backbone**, and `inference_lock` to become per hub —
  or stay global, if simplicity beats throughput;
* `Registry.register`'s `lm_config` equality check and `configure_backbone`'s
  "remove them first" refusal to become per-backbone;
* `recommender._backbone_config` to choose which backbone a plan trains against;
* `doctor._backbone_check` to iterate.

The engine needs no change: `RiNALMoHub` is already per-backbone, and adapters already
record their `lm_config`.

## Swap the model provider

Usually zero code:

```bash
export ADAPTRNA_MODEL=<provider>:<model>              # all roles
export ADAPTRNA_MODEL_TOOLSMITH=<provider>:<model>    # or per role
```

Model specs are provider-prefixed strings resolved by LangChain's `init_chat_model`, and
[`models.py`](../agentic/adaptrna_agentic/models.py) is the **only** provider-aware module —
nothing imports `langchain_anthropic` anywhere. A new provider needs its LangChain
integration installed, and one line in `build_chat_model` if it requires a different
credential check than `require_api_key()`.

---

## Things not to do

| Don't | Because |
|---|---|
| Edit anything under `engine/` to make the agent layer work | The engine is a stable substrate consumed through three contracts. The one seam that genuinely needed handling — importing generated tasks — is handled agentic-side by `discovery`. |
| Put hyperparameters in code | They belong in `knowledge/*.yaml`, where the rationale lives with them and cannot drift |
| Add an `interrupt()` inside a tool | The tools node runs every call in one pass; a resume would re-execute the completed ones |
| Import torch or the engine at module level in `agentic/` | It is what keeps the package importable in milliseconds and the test suite fast |
| Import a provider integration outside `models.py` | It is the single provider seam |
| Assign `innerHTML` in `ui/` | Escaping is a property of how nodes are made, not something each call site remembers |
| Raise from a report-producing function (harness, sandbox, smoke test, doctor) | A crash, a hang and a failed check are all *results* the report needs to describe |
| Add a delete endpoint to the API | Deletion stays a human action at the CLI; an HTTP endpoint is a weaker boundary than a shell prompt |
| Write an error message without the fix in it | That string is the contract with three front ends |
