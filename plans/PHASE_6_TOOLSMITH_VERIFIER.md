# Phase 6 — ToolSmith + Verifier (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 6; agent topology §5 (flows **D** and
> **E**); the two silent-failure questions §6; engine constraints §7.
> **Definition of done:** a new task type becomes a working, registered tool without
> hand-written code.
> Status: planned · not started

---

## 1. Context and goal

Phase 5 drew an explicit boundary: data in a shipped task's layout can be trained today,
and *"an arbitrary schema needs a new datamodule — that is flow D, Phase 6."* Phase 6
closes exactly that boundary, and the same machinery closes flow E (a new external tool
for a package the user names).

The pieces are already in place. Phase 3's `external/contract.py` was written as the
template a generated wrapper must satisfy; Phase 5's `knowledge/task_templates.yaml`
records what each task shape implies (head, loss, metrics, `extract_features` pattern);
Phase 5's `train_entrypoint.py` already carries the `CUSTOM_TASK_MODULES` import seam; and
the engine's own `examples/ncrna_classification/` (96 + 92 + 34 lines) is a complete,
tested worked example of "a task is three files and no core edits".

What Phase 6 adds is the pair of agents that *write* those files, and — the part that
actually decides whether this is trustworthy — a **deterministic verification harness**
that catches the failure modes generated code will realistically have.

**Out of scope:** retention/cleanup of generated artifacts (Phase 7), the HTTP API
(Phase 8), and any engine edit (still forbidden — generated tasks use the engine's
public `@register_task` extension mechanism, which is what it exists for).

## 2. Decisions this plan fixes (including both open §10 items)

| Decision | Choice | Rationale |
|---|---|---|
| **Where generated code lands** (§10) | Repo-root **`adaptrna_custom/`**, **git-tracked**, with `tasks/<name>/` and `tools/<family>.py` | Generated code is a *reviewable deliverable*, not runtime state: it must be diffable, committable and hand-editable later, which rules out git-ignored `toolhub_data/`. Keeping it out of `agentic/` keeps the pip package free of generated content. The `adaptrna_` prefix follows the project's naming convention and avoids a generic top-level `custom` colliding on `sys.path` |
| **Sandbox depth** (§10) | **Subprocess + timeout + `resource` limits (address space, CPU, file size) + a temp working directory.** No container | Honest threat model: this code is written by our own model, from the user's description, on the user's machine, and **the human reviews the diff before it lands**. The realistic risks are accidents — an infinite loop, runaway memory, writing outside the intended directory — not adversarial escape. Documented as accident-isolation, with containers named as the upgrade path if generated code ever arrives from an untrusted source |
| **Custom-task discovery** | Auto-discovery: `adaptrna_custom.tasks.load_all()` imports every `tasks/<name>/task.py`, collecting per-module failures instead of raising | No second registry to drift out of sync with the filesystem. A broken generated task reports itself rather than breaking every other task's import |
| **Agent split** | ToolSmith and Verifier are **one LLM call each per iteration**; the loop, the staging, and the test harness are plain Python | MASTER_PLAN §3.1. It also makes the whole pipeline testable with scripted models |
| **What the Verifier is for** | The **harness** proves the code runs and round-trips; the **reviewer** judges what tests cannot — intent, and the CLS/EOS question. Both must pass | The two silent-failure questions split cleanly: one is mechanically checkable (below), one is a judgement call |
| **Structured codegen** | ToolSmith returns `{files: [{path, content}], notes}` via LangChain structured output | Reliable to parse, trivial to diff, and testable with a scripted model |
| **Approval** | Two tools: `create_task_tool` / `create_external_tool` run the bounded pipeline unattended and return a report + staged diff; **`land_generated_code` is gated** and copies staging → `adaptrna_custom/` | Same shape as Phase 5: cheap work is ungated, the consequential step shows exactly what will happen and waits. The staging directory is a real path the user can open in their editor |
| **Loop bound** | ≤3 ToolSmith↔Verifier iterations, then report the failure honestly and stop | MASTER_PLAN §5. A fourth attempt on the same failing idea is rarely the fix |

## 3. Generated-code layout and the import seam

```
adaptrna_custom/                     # repo root, git-TRACKED
├── README.md                        # what this is, and that it is generated-then-reviewed
├── __init__.py
├── tasks/
│   ├── __init__.py                  # load_all() -> list[(name, exception)]
│   └── <task_name>/
│       ├── __init__.py
│       ├── task.py                  # @register_task subclass of BaseDownstreamModule
│       ├── datamodule.py            # LightningDataModule returning (tokens, target)
│       └── config.yaml              # task YAML, layered on engine/configs/base.yaml
└── tools/
    ├── __init__.py
    └── <family>.py                  # SPEC + typed functions (the Phase 3 contract)
```

Three consumers import it, all through `load_all()`:

1. **`jobs/train_entrypoint.py`** — already the seam; `CUSTOM_TASK_MODULES` is replaced by
   a `load_all()` call, so a generated task is trainable with **no** change to the
   JobRunner, the plan format, or the engine.
2. **`toolhub/runtime.py`** — before `hub.register()`, so an adapter trained on a custom
   task can be *served*. (Phase 2's runtime calls the engine's `get_task(name)`, which
   raises unless the module has been imported — this is the integration point that makes
   generated tasks first-class tools rather than train-only artifacts.)
3. **The verification harness** — in its subprocess.

`REPO_ROOT` is inserted on `sys.path` by each entrypoint rather than relying on the CWD.

## 4. Components

```
agentic/adaptrna_agentic/
├── agents/
│   ├── toolsmith.py       # NEW — generate files (one structured LLM call)
│   └── verifier.py        # NEW — review the diff (one LLM call) + run the harness
├── codegen/
│   ├── pipeline.py        # NEW — the bounded loop, staging, diffing, landing
│   ├── harness.py         # NEW — the deterministic test harness (below)
│   ├── prompts.py         # NEW — context assembly: templates, contract, profile
│   └── sandbox.py         # NEW — subprocess + rlimits + timeout, JSON in/out
└── agents/tool_factory.py # + create_task_tool, create_external_tool, land_generated_code
```

`codegen/prompts.py` assembles what ToolSmith actually needs, all of it already written:
the matching entry from `knowledge/task_templates.yaml`, the engine's subclass contract
(the hook table), `examples/ncrna_classification/` as a worked example, the data profile,
and — for flow E — `external/contract.py`'s docstring plus `vienna.py`.

## 5. The verification harness (the part that makes this trustworthy)

Run in the sandbox subprocess against a **nano** backbone and a small slice of the user's
real data, emitting JSON. Every check maps to a way generated code actually breaks:

| # | Check | Catches |
|---|---|---|
| 1 | The module imports and `@register_task` fires under the expected name | Decorator missing, name mismatch, import errors |
| 2 | The config resolves through the engine's own `resolve_config`, and its `head:` block builds a head on nano | Config/head-kwarg drift — the `**kwargs` TypeError the shipped tasks raise deliberately |
| 3 | The datamodule constructs from the config, `setup()` runs, and a batch is drawn from the real data | The most likely thing to be wrong: a datamodule that does not match the user's file |
| 4 | Forward pass on that batch; loss is finite; `backward()` populates grads on head/LoRA params and on nothing frozen | Shape errors in `extract_features`, detached graphs, wrong loss reduction |
| 5 | `update_metrics` + `compute_metrics` return a dict of finite scalars | Metric wiring, per-stage guards |
| 6 | **Adapter round-trip prediction equivalence** (see below) | **Any** task-owned state missing from the adapter file |
| 7 | The task serves through `RiNALMoHub` — register the saved adapter and predict | The Phase 2 serving path, including `postprocess_predictions` |

**Check 6 is the important one.** The spec's number-one silent failure is task state that
predictions depend on but that never reaches the adapter file (`ADAPTER_EXTRA_PREFIXES`
for tensors, `adapter_extra_payload()` for plain values). Rather than asking the reviewer
"did you remember?", the harness proves it mechanically:

1. build the module on nano, **randomise every parameter and buffer** (so no state sits at
   its default), and let the task set any state it owns;
2. predict on fixed sequences → **A**;
3. `save_adapter()` → load into a **fresh** module → predict → **B**;
4. assert `A == B` exactly.

If any prediction-affecting tensor state is missing from the file, A ≠ B and the check
fails with the diverging outputs. (Non-tensor state — a tuned threshold, a class mapping —
stays a reviewer checklist item, since randomising arbitrary Python attributes is not
mechanically safe. That asymmetry is stated in the checklist rather than papered over.)

The harness runs against the **shipped** tasks in the test suite as its own control: if
`splice_site`/`mrl`/`sec_struct` do not pass it, the harness is wrong, not the task.

## 6. The two flows

**Flow D — a new task type** (the DoD):
profile says no layout match → user asks for a task → ToolSmith writes the three files into
`toolhub_data/staging/<id>/` → Verifier reviews + harness runs → ≤3 iterations →
**[gated] land** into `adaptrna_custom/tasks/<name>/` → then Phase 5's flow C unchanged
(recommend → **[gated]** train → analyze → **[gated]** register → serve).

**Flow E — a new external tool:** user names a package → ToolSmith writes the wrapper
against the Phase 3 contract → **[gated]** install (Phase 3's flow) → Verifier runs
`load_spec` + the golden cases → **[gated]** land → `register_external` (Phase 3, unchanged).

## 7. Tests (deterministic: scripted models, nano backbone, no API key)

| test file | asserts |
|---|---|
| `test_harness_controls.py` | the harness **passes** on the shipped `splice_site` and `mrl` tasks (the control), and its round-trip check is exact for both |
| `test_harness_catches.py` | deliberately broken fixture tasks are caught, one per failure mode: a task owning a buffer outside `ADAPTER_EXTRA_PREFIXES` → check 6 fails with diverging outputs; a head that ignores its config kwargs → check 2; a datamodule whose columns do not match the data → check 3; an `extract_features` with the wrong shape → check 4 |
| `test_sandbox.py` | timeout kills a hanging script; a memory hog hits the address-space limit; the working directory is temporary; results come back as JSON; a crash is reported, not raised |
| `test_pipeline.py` | scripted ToolSmith returns broken code, then fixed code → the loop converges in 2 iterations and stages a diff; three consecutive failures → gives up with the reports, stages nothing; staged files never land without the landing step; landing writes `adaptrna_custom/tasks/<name>/` and `load_all()` then registers the task |
| `test_prompts.py` | the assembled ToolSmith context contains the task template, the engine hook contract, the worked example, and the user's data profile (a prompt missing the contract is how you get a task that fails check 1) |
| `test_codegen_tools.py` | the three new agent tools; `land_generated_code` is in `GATED_TOOLS`; the approval payload lists the files with line counts and the staging path |
| `test_custom_discovery.py` | `load_all()` imports a good task; a broken task is reported as `(name, exception)` without breaking the others; the ToolHub runtime imports custom tasks before registering, so an adapter from a generated task **serves** |

Expected ~35 new tests (total ~225). Phases 0–5's 191 stay green; engine's 135 untouched.

## 8. Implementation order

1. `adaptrna_custom/` skeleton + `load_all()` + `test_custom_discovery.py`; wire the
   runtime and `train_entrypoint` to it.
2. `codegen/sandbox.py` + `test_sandbox.py`.
3. `codegen/harness.py` + `test_harness_controls.py` (shipped tasks as the control) +
   `test_harness_catches.py` (broken fixtures).
4. `codegen/prompts.py` + `test_prompts.py`.
5. `agents/toolsmith.py`, `agents/verifier.py`, `codegen/pipeline.py` + `test_pipeline.py`.
6. Agent tools + approval payload + `test_codegen_tools.py`.
7. DoD run (§9); README sections; `adaptrna_custom/README.md`.
8. Close-out: MASTER_PLAN §8 tick; §10 rows "Where generated code lands" and "Sandbox
   depth" marked decided.

## 9. Verification / definition of done

**Gate 1 — deterministic:** `cd agentic && pytest` green (191 + ~35); `cd engine && pytest`
→ 135; no engine changes.

**Gate 2 — the live flow-D scenario.** Its premise is the boundary Phase 5 declared, so the
DoD data is deliberately a schema **no shipped task can read**: derive a plain
`sequence,label` CSV (train/val/test) from the Spliceator donor folds — same 400 nt windows
and labels, none of the `GS_1/db_N` layout. Real data, genuinely unsupported shape,
~3-minute training run.

```
python -m adaptrna_agentic.cli.chat --session newtask

you> I have labelled sequences at dod_data/splice_simple_{train,val,test}.csv.
     Can you train on this?
#   → profile_dataset: binary target, 400 nt, layout_match: null,
#     "no shipped task reads this layout" + the nearest template

you> Then build me a task for it.
#   → create_task_tool: ToolSmith writes 3 files → Verifier reviews + harness runs
#     → report + staged diff (files, line counts, staging path)

you> Show me what it wrote, then land it.
#   → APPROVAL GATE: file list + contents → [y] → adaptrna_custom/tasks/<name>/

you> Now train it on that data.
#   → recommend_training_config on the NEW task → APPROVAL GATE → ~3 min run

you> Analyze and register it as splice_simple.
#   → analyze_run → APPROVAL GATE → registered

you> Score this donor window with both splice_site and splice_simple.
#   → both serve from one backbone; the generated task's tool agrees with the
#     hand-written donor tool   ← end-to-end proof, cross-checked against a known-good tool
```

The cross-check is the point: a generated task that trains and serves but disagrees with
the hand-written tool on the same underlying problem would be a *quiet* failure, and the
DoD is designed to expose it.

**Gate 3 — flow E, live:** ask for a wrapper around an installed pure-Python package
(no new install needed), verify + land + `register_external` + `test`.

## 10. Risks and notes

- **The harness is the trust boundary, not the reviewer.** If check 6 is weak, silent
  wrong-scale predictions ship. Hence the shipped tasks are run through it as a control in
  CI — a harness that passes everything is worse than no harness.
- **Generated datamodules are the likeliest failure**, which is why check 3 uses the
  user's real data rather than a synthetic fixture.
- **Accident-isolation, not adversarial sandboxing** — stated plainly in the code and the
  README, with the human diff gate as the real boundary and containers as the upgrade path.
- **Landed code is the user's**, not the platform's: it is git-tracked, hand-editable, and
  never silently regenerated. Re-running codegen for an existing task stages a diff against
  what is there rather than overwriting.
- **A generated task changes what "the engine" means for serving** — the ToolHub runtime
  must import custom tasks before registering, or an adapter from a generated task trains
  fine and then fails to serve. Covered by `test_custom_discovery.py`.
- **Loop cost**: each iteration is two model calls with a sizeable context (templates +
  example + profile). Bounded at 3, and the prompt assembly is static enough to be
  prompt-cache friendly.
