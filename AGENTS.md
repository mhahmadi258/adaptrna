# AdaptRNA

Conversational agent platform for RNA analysis, built on a task-pluggable fine-tuning
engine for the RiNALMo RNA language model. Every task-specific adaptation of the frozen
backbone is a LoRA adapter (~6 MB) + task head, docked onto one resident backbone — the
agent's tool registry *is* an adapter registry. A fresh install ships **no task
definitions and no adapters**; every tool comes from a user's own labelled CSV/TSV, taken
through gated steps: profile → build → train → register.

Full technical documentation already exists in this repo — read it instead of re-deriving
from source:

- [`documents/README.md`](documents/README.md) — start here: layers, architecture, the five
  user flows, entry points, where state lives, verified facts
- [`documents/project_structure.md`](documents/project_structure.md) — every directory/file
  and its responsibility
- [`documents/modules/`](documents/modules/README.md) — one doc per package
- [`documents/workflows/`](documents/workflows/README.md) — one doc per end-to-end flow
- [`documents/setup.md`](documents/setup.md), [`configuration.md`](documents/configuration.md),
  [`testing.md`](documents/testing.md), [`extending.md`](documents/extending.md) — install,
  config schemas, test strategy, "I want to change X — where do I edit?"
- [`plans/MASTER_PLAN.md`](plans/MASTER_PLAN.md) — design rationale (*why*), phase history;
  `plans/PHASE_*.md` are per-phase blueprints, useful as archaeology

For any non-trivial question about how something works, check `documents/` and `plans/`
first — they were written by reading the implementation and are kept accurate — before
reading source directly.

## Layout

```
engine/            fine-tuning framework + vendored RiNALMo backbone (packages: rinalmo_hub, rinalmo)
agentic/            the agent platform (package: adaptrna_agentic) — agents, ToolHub, codegen, jobs, api, cli
adaptrna_custom/    generated tasks/tools — git-tracked, human-approved, never auto-regenerated
ui/                 browser client, plain ES modules, no build step
documents/          technical documentation (see above)
plans/              design rationale + per-phase plans
```

Runtime state (git-ignored, repo root): `weights/`, `dataset/`, `outputs/`, `toolhub_data/`,
`chat_data/`, `jobs_data/`. Everything runs **from the repo root**.

## Two rules enforced in code (not by prompt)

1. Hyperparameters come only from the knowledge base — `start_training` refuses any plan not
   stamped by `recommend_training_config`.
2. Nothing consequential (GPU spend, registering a tool, landing generated code) happens
   without an explicit human approval gate.

## Environment

- Run Python via `/home/mh/adaptrna/.venv/bin/python` — the shell starts in conda base, and
  bashrc auto-activation only fires on an interactive prompt.
- `.env` (git-ignored) holds `ANTHROPIC_API_KEY`, loaded at runtime only. Never read or print
  its contents.

## Verify an install

```bash
python -m adaptrna_agentic.cli.toolhub doctor     # read-only health check — run first when anything looks off
cd engine  && python -m pytest && cd ..           # CPU, no weights/datasets
cd agentic && python -m pytest && cd ..           # CPU, no network/API key
```

## Naming convention

New platform packages/modules are model-agnostic: `adaptrna_*`, never `rinalmo`-branded —
RiNALMo is just the first backbone this platform serves. Only the vendored engine packages
(`rinalmo`, `rinalmo_hub`) keep their existing names.

## Working conventions for Claude

- **Plans.** When asked to create a plan — in plan mode or a normal session — write it as a
  new file under `plans/` (follow the existing `PHASE_*.md` naming/structure), and update
  [`plans/MASTER_PLAN.md`](plans/MASTER_PLAN.md) to reflect it (roadmap/phase list, and any
  resolved decisions the plan makes).
- **Answering questions about the project.** Start from `documents/` (see the doc map above)
  before reading source. Only fall back to reading the code itself when the documentation
  doesn't have the answer.
- **After making code changes.** Update the relevant files under `documents/` (and
  `plans/MASTER_PLAN.md` if a resolved decision or phase status changed) so the documentation
  stays accurate — don't leave it to drift from the implementation.
