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

## Tests (no network, no API key)

```bash
cd agentic && ../.venv/bin/python -m pytest
```
