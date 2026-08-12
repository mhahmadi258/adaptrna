# AdaptRNA

An RNA foundation-model platform in three layers: a fine-tuning **engine**, a conversational
**agent** system on top of it, and (later) a web **UI**. RiNALMo is the first backbone the
platform serves — naming stays model-agnostic on purpose.

| layer | what it is | docs |
|---|---|---|
| [`engine/`](engine/) | RiNALMo-Hub: one frozen backbone, swappable LoRA adapters and task heads, one CLI | [engine/README.md](engine/README.md) |
| [`agentic/`](agentic/) | LangGraph agents + Tool-Hub (adapters and classical tools as uniform tools) | [agentic/README.md](agentic/README.md) |
| [`ui/`](ui/) | web UI — placeholder until Phase 9 | [plans/MASTER_PLAN.md](plans/MASTER_PLAN.md) |
| [`plans/`](plans/) | the master plan and per-phase detailed plans | [plans/MASTER_PLAN.md](plans/MASTER_PLAN.md) |

Runtime artifacts (`weights/`, `dataset/`, `outputs/`, `adapters/`) live at the repo root,
and engine commands are run from here — e.g.:

```bash
python -m rinalmo_hub.cli.train --task splice_site --use_lora --set optim.lr=3e-4 \
    --output_dir outputs/splice_donor_lora
python -m adaptrna_agentic.cli.chat
```

Setup: `pip install -e ./engine` and `pip install -e ./agentic` into one virtualenv;
secrets (e.g. `ANTHROPIC_API_KEY`) go in a git-ignored `.env` at this root.
