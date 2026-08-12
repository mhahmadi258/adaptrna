# AdaptRNA

A conversational agent platform for RNA analysis, built on a task-pluggable fine-tuning
engine. Ask for a prediction and it activates the right adapter; ask for a capability that
does not exist yet and it fine-tunes one, or writes the task from scratch.

```
engine/           the fine-tuning engine: one frozen backbone, swappable LoRA adapters
agentic/          the agent platform: orchestrator, Tool-Hub, training pipeline, codegen
adaptrna_custom/  tools built for this project — generated, reviewed by you, then yours
ui/               web frontend (Phase 9)
plans/            the master plan and one detailed plan per phase
```

Runtime state lives at the repo root and is git-ignored: `weights/` `dataset/` `outputs/`
`toolhub_data/` `chat_data/` `jobs_data/`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ./engine -e ./agentic
python -m pip install flash-attn --no-build-isolation    # CUDA only; match your torch build
```

Put your Anthropic key in `.env` at the repo root (`ANTHROPIC_API_KEY=…`; git-ignored),
point the hub at a backbone checkpoint, and check the install:

```bash
python -m adaptrna_agentic.cli.toolhub config --weights ~/.cache/rinalmo_pretrained/giga-v1.pt
python -m adaptrna_agentic.cli.toolhub doctor
```

## The five things you can ask for

```bash
python -m adaptrna_agentic.cli.chat --session work
```

| Ask | What happens |
|---|---|
| *"What tools are available?"* | The registry, with each tool's state and purpose |
| *"Is this sequence a donor splice site?"* | The matching adapter is activated on the shared backbone and predicts |
| *"Disable the fold tool"* / *"test it"* | Lifecycle operations; a disabled tool refuses with the fix in the message |
| *"Fine-tune this data into a tool"* | Profile → **validated** config → **approval** → GPU run → analysis → **approval** → registered |
| *"No task can read my data — build one"* | Three files written, verified against your data, reviewed → **approval** → landed → trainable |

Two rules the platform enforces in code rather than by prompt: hyperparameters only ever
come from the knowledge base of validated runs, and nothing consequential (GPU hours, a
new servable tool, code written into your repo) happens without your approval.

## Managing it

```bash
toolhub=  "python -m adaptrna_agentic.cli.toolhub"

$toolhub list                       # every tool, both kinds
$toolhub predict <tool> --sequences ACGU...     # adapters
$toolhub call <tool> sequence=GGGG              # external tools (ViennaRNA, …)
$toolhub test <tool>                # smoke / golden tests
$toolhub doctor                     # what is wrong with this install (changes nothing)
$toolhub prune staging|artifacts|jobs|runs|sessions [--older-than N] [--yes]
```

`doctor` is the first thing to run when something looks off: every failure it reports
names the command that fixes it. `prune` is the only command that deletes anything — it
is a dry run unless you pass `--yes`, and it never touches an artifact a registered tool
depends on.

## Troubleshooting

| Message | Meaning and fix |
|---|---|
| `ANTHROPIC_API_KEY is not set` | Put it in `.env` at the repo root, or export it |
| `The engine package is not installed` | `pip install -e ./engine` |
| `checkpoint '…' does not exist` | `toolhub config --weights /path/to/giga-v1.pt` |
| `Tool 'x' points at '…', which is gone` | Restore the file, or `toolhub remove x`. `doctor` lists every such case |
| `Tool 'x' is disabled` | `toolhub activate x` (in chat, just ask) |
| `This plan did not come from recommend_training_config` | Intentional: hyperparameters must come from the knowledge base |
| `Job '…' is still running` | One training job at a time; wait, or cancel it |
| `PID … may since have been reused — refusing to signal it` | The job's process is gone; the record was closed out and nothing was killed |
| `'…' changed on disk since it was read` | Another process wrote first. Nothing was lost — retry |
| `is a full fine-tuning export` | Only LoRA adapters can be served; evaluate full-FT exports with the engine CLI |

## Known limitations

- **A crashed training run cannot be resumed** mid-flight; start it again.
- **One training job at a time** by default — two giga runs on one GPU is how you get an
  out-of-memory failure forty minutes in.
- **Serving runs in fp32.** Casting the engine to bf16 for non-autocast inference trips a
  dtype promotion in its `TokenDropout` (see plans/MASTER_PLAN.md §7).
- **Generated code is accident-isolated, not sandboxed.** Time, memory and file-size
  limits catch runaway loops; the human diff gate is the real boundary.
- **The stores detect concurrent writes, they do not prevent them.** Two chat processes
  are fine; the second to save is asked to retry.

## Docs

[plans/MASTER_PLAN.md](plans/MASTER_PLAN.md) is the map — architecture, design principles,
the engine constraints this layer respects, and the phase roadmap. Each phase has its own
detailed plan beside it. Layer specifics live in [engine/README.md](engine/README.md) and
[agentic/README.md](agentic/README.md).
