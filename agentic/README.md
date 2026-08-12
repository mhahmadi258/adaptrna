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

## Try it

```bash
python -m adaptrna_agentic.cli.chat --once "What is the GC content of GGCAUUACGGCU?"
python -m adaptrna_agentic.cli.chat            # REPL; 'quit' to exit
```

Per-role models default to `anthropic:claude-opus-5`; override with `ADAPTRNA_MODEL`
(all roles) or `ADAPTRNA_MODEL_ORCHESTRATOR` / `_TOOLSMITH` / `_VERIFIER`.

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
The wrapper contract lives in `toolhub/external/contract.py`; `toolhub/external/vienna.py`
(ViennaRNA) is the hand-written reference — and the template Phase 6's ToolSmith imitates.

```bash
# install is approval-gated: the exact pip command is shown; confirm or pass --yes
python -m adaptrna_agentic.cli.toolhub register-external adaptrna_agentic.toolhub.external.vienna
python -m adaptrna_agentic.cli.toolhub call vienna_fold sequence=GGGGAAAACCCC
python -m adaptrna_agentic.cli.toolhub test vienna_fold      # golden-pair tests
```

Wrapped packages (e.g. `ViennaRNA`) are *tool* dependencies: installed into the venv via
the gated flow, recorded in the manifest provenance, and deliberately absent from any
`pyproject.toml`.

## Tests (no network, no API key)

```bash
cd agentic && ../.venv/bin/python -m pytest
```
