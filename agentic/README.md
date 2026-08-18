# adaptrna-agentic

The agent layer of the AdaptRNA platform: LangGraph agents and (from Phase 2) the
Tool-Hub, sitting on top of the fine-tuning engine. Model-agnostic by design — RiNALMo is
the first backbone this platform serves, not its brand.

Phase 0 ships the scaffold: per-role model configuration, API-key handling, and a
hello-world graph with one real tool call (`gc_content`).

## Install

```bash
# from the repo root, into the shared .venv
.venv/bin/python -m pip install -e ./agentic
```

Resolved versions at first install (2026-08-12): `langchain 1.3.15`, `langgraph 1.2.11`,
`langchain-anthropic 1.5.5`, `langchain-core 1.5.4`, `anthropic 0.121.0`,
`python-dotenv 1.2.2`.

## API key

Export `ANTHROPIC_API_KEY`, or put it in a repo-root `.env` (git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Chat (Phase 4)

The orchestrator: every ToolHub tool (adapters + external) plus lifecycle operations
(list/info/activate/deactivate/test) bound as agent tools. Sessions persist in
`chat_data/sessions.sqlite` (`ADAPTRNA_CHAT_DIR` overrides; `langgraph-checkpoint-sqlite
3.1.1`). One chat process holds one runtime — the backbone loads once, on the first
foundation-model call (`--warmup` for eager).

```bash
python -m adaptrna_agentic.cli.chat                      # REPL, session 'default'
python -m adaptrna_agentic.cli.chat --session paper      # named persistent session
python -m adaptrna_agentic.cli.chat --once "Is GGC...ACU positive under my_tool?"
python -m adaptrna_agentic.cli.chat --list-sessions
```

Per-role models default to `anthropic:claude-opus-5`; override with `ADAPTRNA_MODEL`
(all roles) or `ADAPTRNA_MODEL_ORCHESTRATOR` / `_TOOLSMITH` / `_VERIFIER`.

## From one CSV to a servable tool (Phases 5-6, rebuilt in Phase 13)

The platform starts with **no tools and no task definitions**. Everything servable begins
from one file you provide — `.csv`/`.tsv`, optionally gzipped — with one sequence column
and one label column holding a binary, multiclass, or continuous target. Four steps take
it from there to a registered tool, each its own turn in the conversation, each ending in
your approval — nothing auto-chains to the next step:

```
you> Profile ~/data/my_data.csv.
#    → profile_dataset reads the file and proposes an interpretation: which column is
#      the sequence, which is the label, the target type, a split, and any data-quality
#      warnings (duplicate sequences, leakage across a column split, class imbalance)
#    → confirm_data_profile — approval gate: edit any field (column, target type,
#      split, task name, positive class), or approve as proposed
you> Build the task.
#    → create_task_tool takes the approved spec and renders task.py / datamodule.py /
#      config.yaml straight from a reviewed template, deterministically and with no
#      model call, whenever the spec is one of the three supported shapes; an unusual
#      spec falls back to writing the code with a model instead — either way it then
#      runs the same 7-check harness against your real file and an independent review
#    → land_generated_code — approval gate: file list, line counts, the diff, and which
#      path produced it (rendered or generated)
you> Recommend a training config.
#    → hyperparameters DERIVED from the approved spec (batch size from sequence length,
#      epochs from a step budget) plus validated arm settings — never invented, and
#      there is no reference metric band, because there is no run of this task yet
you> Run it.                     ← approval gate: shows the exact command, waits for [y/N]
you> How's it going?             ← the job runs detached; the chat stays responsive
you> Analyze the run.            ← reported as a baseline: the first successful run
                                    becomes what later runs of the same task compare against
you> Register it as my_tool.     ← approval gate
```

Generated code lands in [`adaptrna_custom/`](../adaptrna_custom/) — git-tracked, yours to
edit, never silently regenerated — alongside `spec.json`, the approved dataset
interpretation the task landed with. Staged code survives the session, so you can open it
in an editor and approve it later (`list_staged_code`).

Training is **deterministic**: every hyperparameter comes from
`adaptrna_agentic/knowledge/*.yaml` (validated arm settings and their failure modes, plus
rules derived from your approved spec), and the rationale shown to you is generated from
the same entries — the model narrates, it never invents a number. Jobs are recorded in
`jobs_data/`, run artifacts land in `outputs/<run_name>/`, and one training job runs at a
time by default.

Two rules the analyzer enforces: a run truncated by `trainer.max_steps` is never compared
to a reference, and a difference inside tolerance is never called a regression
(FlashAttention's non-deterministic backward gave F1 95.21 vs 95.82 for the same command
and seed on one task). Hyperparameters can only come from the knowledge base:
`start_training` refuses any plan that did not come out of `recommend_training_config`, so
the rule is enforced by code rather than by asking the model to behave.

The harness is the trust boundary, so it is itself controlled: known-good fixture tasks
run through it in CI (one per supported target type), and deliberately broken fixtures
must fail it. Its sharpest check is the adapter **round-trip prediction equivalence** —
randomise everything the adapter should carry, predict, save, reload into a fresh module,
predict again, and require identical outputs. That turns this project's worst silent
failure (task state that never reaches the adapter file) into a hard test rather than a
question on a checklist — and on the template path, a property proven once for the
template and inherited by every task rendered from it.

Generated code runs under a subprocess with time, memory and file-size limits. That is
**accident-isolation, not adversarial sandboxing** — the human diff gate is the real
boundary.

## ToolHub (Phase 2)

Adapters served as tools from one shared backbone. Registry operations are instant;
`predict`/`test`/`warmup` load the backbone lazily (per process). State lives in
`toolhub_data/` at the repo root (git-ignored).

```bash
python -m adaptrna_agentic.cli.toolhub config --weights ~/.cache/rinalmo_pretrained/giga-v1.pt
python -m adaptrna_agentic.cli.toolhub register outputs/<run>/<task>_adapter.pt
python -m adaptrna_agentic.cli.toolhub list
python -m adaptrna_agentic.cli.toolhub predict <name> --sequences ACGU... [--input file]
python -m adaptrna_agentic.cli.toolhub test <name>        # smoke test
python -m adaptrna_agentic.cli.toolhub deactivate <name>  # routing-level; activate restores
```

Adapter tools are LoRA-only (a full-FT export would pair its head with the pretrained
backbone). Serving runs fp32 (`dtype: auto`): non-autocast bf16 inference trips a dtype
promotion inside the engine's TokenDropout — see MASTER_PLAN §7.

## External tools (Phase 3)

Classical packages wrapped as typed functions, sharing the same manifest and lifecycle.
The wrapper contract lives in `toolhub/external/contract.py`: a module-level `SPEC`, plain
typed functions, and input validation that runs before the wrapped package is imported —
the same contract Phase 6's ToolSmith imitates when it writes a new wrapper.

```bash
# install is approval-gated: the exact pip command is shown; confirm or pass --yes
python -m adaptrna_agentic.cli.toolhub register-external adaptrna_custom.tools.<your_tool>
python -m adaptrna_agentic.cli.toolhub call <your_tool>_<function> arg=value
python -m adaptrna_agentic.cli.toolhub test <your_tool>_<function>      # golden-pair tests
```

Wrapped packages are *tool* dependencies: installed into the venv via the gated flow,
recorded in the manifest provenance, and deliberately absent from any `pyproject.toml`.

## HTTP service (Phase 8)

The same platform, over HTTP — so a browser can drive it and a session started in the
terminal continues in the browser (and back).

```bash
python -m adaptrna_agentic.cli.serve                 # 127.0.0.1:8000
python -m adaptrna_agentic.cli.serve --port 8077 --warmup
ADAPTRNA_API_TOKEN=secret python -m adaptrna_agentic.cli.serve --host 0.0.0.0
```

It binds loopback by default and **refuses to start** on any other address without a
token: this API can start GPU jobs and write code into the repository. Interactive docs
at `/docs`.

| | |
|---|---|
| `GET /health`, `GET /api/doctor` | liveness; the full Phase 7 report |
| `GET/POST /api/tools/...` | list, info, activate/deactivate, test, predict, call |
| `GET /api/jobs/...` | list, status, logs, analysis; `POST .../cancel` |
| `GET /api/sessions` | `[{id, updated_at, checkpoints}]`, newest first |
| `POST /api/sessions` | create one; `PATCH`/`DELETE /api/sessions/{id}` rename and remove it |
| `POST /api/sessions/{id}/messages` | one turn, streamed as SSE |
| `POST /api/sessions/{id}/resume` | answer a pending approval; the turn continues |

Turns stream `text` (token by token), `tool_call`, `tool_result`, then `done`. When the
agent reaches an approval gate the stream emits `approval_required` — carrying the exact
command or file list — and **ends**; the client decides and calls `/resume`, which returns
a new stream continuing the same turn. A gate can wait minutes for a human, and a
suspended turn survives in the checkpointer where a held-open connection would not.

Adapter inference is serialised inside `AdapterRuntime`: the engine's hub activates an
adapter across the whole backbone before predicting, so overlapping requests for different
tools would otherwise answer from the wrong one.

## Web UI (Phase 9)

```bash
python -m adaptrna_agentic.cli.serve --open
```

Served at `/` by the same process: streamed chat, a tool dashboard, a live training
monitor, and the approval gate as a modal showing the exact command — byte for byte what
the terminal prints. It is a pure client of the endpoints above and holds no state of its
own; a browser refresh mid-approval restores the dialog from `/history`, because the
suspended turn is in the checkpointer, not in the tab.

Plain ES modules, no build step — see [ui/README.md](../ui/README.md) for why, and for the
tripwire that says when to reach for a framework instead.

## Tests (no network, no API key)

```bash
cd agentic && ../.venv/bin/python -m pytest
```

The browser suite is opt-in, like the engine's GPU tests — it needs a ~150 MB Chromium:

```bash
../.venv/bin/python -m pip install playwright
../.venv/bin/python -m playwright install chromium
../.venv/bin/python -m pytest -m ui
```
