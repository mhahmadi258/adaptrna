# Phase 0 — Agentic Scaffolding (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 0.
> **Definition of done:** a hello-world LangGraph graph answers via Claude from the terminal.
> Status: planned · not started

---

## 1. Context and goal

Phase 0 creates the `agentic/` layer's skeleton and proves the entire LLM plumbing end to
end — package, configuration, per-role model selection, API-key handling, a real LangGraph
graph with a real tool call — before any RiNALMo-specific work exists. Everything later
(ToolHub tools in Phase 2–3, the orchestrator in Phase 4) plugs into the seams built here,
so the point of this phase is to get those seams right while they are still one file each.

**Deliberately out of scope** (each has its own phase): any import of `rinalmo`/
`rinalmo_hub` or torch (Phase 2), the repo restructure into `engine/` (Phase 1), the
ToolHub, checkpointer/session persistence (Phase 4), FastAPI, and the real agents. Phase 0
must leave the existing engine tree completely untouched — the only repo-root change is one
`.gitignore` entry.

---

## 2. Decisions this plan fixes (from MASTER_PLAN §10)

| Decision | Choice | Rationale |
|---|---|---|
| **Model per agent role** | Default **`claude-opus-5`** for all three roles (orchestrator, toolsmith, verifier), each independently overridable via config/env | Opus 5 is the current default recommendation for capability-sensitive work ($5/$25 per MTok) and codegen/review quality is the platform's bottleneck. Cost levers stay in the user's hands: set e.g. the orchestrator to `claude-sonnet-5` ($3/$15; intro $2/$10 through 2026-08-31) or `claude-haiku-4-5` ($1/$5) per role once flows stabilize. Revisit per phase as §10 says. |
| **Provider-agnostic seam** | Model specs are strings of the form `"anthropic:claude-opus-5"`, resolved through LangChain's `init_chat_model()` | Exactly the decision recorded in MASTER_PLAN §1: LangGraph + Claude API now, provider swap = config-string edit, zero code changes. |
| **Package name / layout** | Top-level dir `agentic/`, python package **`adaptrna_agentic`**, own `pyproject.toml` | Model-agnostic on purpose: RiNALMo is only the first backbone this platform serves, so no new package carries its name (the vendored engine packages `rinalmo`/`rinalmo_hub` keep theirs). Refines the master-plan tree: subpackages live under `agentic/adaptrna_agentic/`, tests under `agentic/tests/`. |
| **Virtualenv** | One shared `.venv` at repo root; `pip install -e ./agentic` alongside the engine | Single-machine research setup; the LangChain deps are lightweight and conflict-free with torch/lightning. |
| **Settings mechanism** | Plain dataclass + env vars + optional repo-root `.env` (python-dotenv) | No pydantic-settings dependency; trivially testable; YAML config can be added later if settings outgrow this. |

---

## 3. Deliverables — file tree

```
agentic/
├── pyproject.toml                  # package adaptrna-agentic + deps + pytest config
├── README.md                       # ~20 lines: what this layer is, install, API key, hello chat
├── adaptrna_agentic/
│   ├── __init__.py
│   ├── settings.py                 # Settings dataclass, from_env(), .env loading, key check
│   ├── models.py                   # build_chat_model(role) -> BaseChatModel
│   ├── agents/
│   │   ├── __init__.py
│   │   └── hello.py                # hello graph: model node ⇄ tool node + gc_content tool
│   ├── cli/
│   │   ├── __init__.py
│   │   └── chat.py                 # python -m adaptrna_agentic.cli.chat  (REPL / --once)
│   ├── toolhub/__init__.py         # placeholder — docstring: "filled in Phase 2/3"
│   ├── jobs/__init__.py            # placeholder — Phase 5
│   ├── profiling/__init__.py       # placeholder — Phase 5
│   ├── knowledge/__init__.py       # placeholder — Phase 5/6
│   └── api/__init__.py             # placeholder — Phase 8
└── tests/
    ├── test_settings.py
    ├── test_hello_tool.py
    └── test_graph_wiring.py        # fake model drives the tool loop — no network, no key
```

Placeholder subpackages are one-line docstrings naming the phase that fills them — the
skeleton stays self-documenting without inviting dead code.

Repo root: add `.env` to `.gitignore` (the only change outside `agentic/` and `plans/`).

---

## 4. Component specs

### 4.1 `settings.py`

```python
ROLES = ("orchestrator", "toolsmith", "verifier")

@dataclass(frozen=True)
class Settings:
    models: dict          # role -> model spec, e.g. {"orchestrator": "anthropic:claude-opus-5", ...}
    max_tokens: int       # default 8192 — LangChain's Anthropic default (1024) truncates real answers
    @classmethod
    def from_env(cls) -> "Settings": ...
    def model_for(self, role: str) -> str: ...   # KeyError with the valid roles on a bad role
```

- `from_env()` loads a repo-root `.env` first if present (`dotenv.load_dotenv`, never
  overriding already-set env vars), then reads:
  - `ADAPTRNA_MODEL` — override for **all** roles;
  - `ADAPTRNA_MODEL_ORCHESTRATOR` / `_TOOLSMITH` / `_VERIFIER` — per-role overrides
    (win over the global one);
  - default when nothing is set: `anthropic:claude-opus-5` per role.
- `require_api_key()` helper: raises with an actionable message
  (*"Set ANTHROPIC_API_KEY in the environment or in <repo-root>/.env"*) when the key is
  absent. Called at model-construction time, **never at import time**, so the deterministic
  tests and later ToolHub code never need a key. The key value itself is never logged.

### 4.2 `models.py`

```python
def build_chat_model(role: str, settings: Settings | None = None, **overrides) -> BaseChatModel:
```

- Resolves the role's spec via `Settings`, calls `require_api_key()`, then
  `init_chat_model(spec, max_tokens=settings.max_tokens, **overrides)`.
- `init_chat_model` (LangChain) is the *entire* provider abstraction — no other file in the
  agentic layer may import `langchain_anthropic` directly. Verify the exact import path
  (`langchain.chat_models.init_chat_model` in LangChain 1.x) against the installed version
  at implementation time.

### 4.3 `agents/hello.py` — the hello graph

Hand-built `StateGraph(MessagesState)`, **not** the prebuilt agent helper: Phase 4's
orchestrator needs custom wiring (approval-gate `interrupt()`s), so Phase 0 proves the
exact graph style the project will actually use.

- One demo tool, RNA-flavored on purpose:
  `@tool gc_content(sequence: str) -> str` — fraction of G/C in an RNA/DNA sequence,
  validated against the alphabet, returned with 3 decimals. Pure python, no engine imports.
- Graph wiring: `model` node (chat model bound to `[gc_content]`) → conditional edge
  (`tool_calls` present → `tools` node (`ToolNode`), else `END`) → `tools` → `model`.
- `build_hello_graph(model: BaseChatModel | None = None)` — the injectable `model`
  parameter is the test seam; production callers pass nothing and get
  `build_chat_model("orchestrator")`.
- Includes a short system message ("You are the AdaptRNA assistant scaffold; use tools when
  they apply") so the DoD demo reliably triggers the tool call.

### 4.4 `cli/chat.py`

- `python -m adaptrna_agentic.cli.chat` → REPL (exit on `quit`/EOF);
  `--once "PROMPT"` → single exchange and exit; `--model SPEC` → override every role for
  this run (plumbed through `Settings`).
- Streams the graph (`graph.stream(..., stream_mode="values")`) and prints tool calls as
  they happen (`→ gc_content({'sequence': ...}) = 0.583`) followed by the final assistant
  text — this output style previews Phase 4's UX and makes the DoD demo self-evidencing.
- Multi-turn REPL state is a plain in-process message list this phase (checkpointer arrives
  in Phase 4).

### 4.5 `pyproject.toml`

- Package `adaptrna-agentic`, version 0.1.0, `requires-python >= 3.10` (matches engine).
- Dependencies: `langgraph>=1.0`, `langchain>=1.0`, `langchain-anthropic`,
  `python-dotenv`. Bounds are floors — record the exact resolved versions in
  `agentic/README.md` after the first install. Dev extra: `pytest>=7.0`.
- **Deliberately absent:** any dependency on the engine, torch, or lightning. The agentic
  package must import and test in milliseconds; the engine dependency is added in Phase 2
  when the ToolHub actually needs `RiNALMoHub`.
- Pytest config scoped to this package (`testpaths = ["tests"]` relative to `agentic/`).
  Until the Phase 1 restructure, the repo-root `pytest` still runs the engine suite only;
  agentic tests run as `cd agentic && pytest` (or `pytest agentic/tests` from the root).

---

## 5. Tests (all CPU, no network, no API key)

| test | asserts |
|---|---|
| `test_settings.py` | defaults are `anthropic:claude-opus-5` for all three roles; global env override applies to every role; per-role override beats global; unknown role raises listing valid roles; `require_api_key()` raises the actionable message when unset and passes when set (monkeypatched env) |
| `test_hello_tool.py` | `gc_content` on known sequences (e.g. `GGCC` → 1.000, `AUAU` → 0.000, mixed case); invalid characters rejected with a clear error |
| `test_graph_wiring.py` | build the graph with a **scripted fake chat model** (`langchain_core`'s fake chat model, or a minimal local stub honoring `bind_tools`) that first returns an `AIMessage` carrying a `gc_content` tool call, then a final text `AIMessage`. Assert: the resulting message history contains a `ToolMessage` with the correct computed value, the loop terminates, and the final message is the fake's second response. Also: graph compiles with no model argument **without** touching the network (construction must be lazy or injected). |

The fake-model test is the load-bearing one: it proves graph wiring, tool binding, tool
execution and loop termination — everything except Anthropic's API — which is exactly what
CI can verify without a key.

---

## 6. Implementation order

1. `agentic/pyproject.toml` + package skeleton + placeholder subpackages; `pip install -e ./agentic`; record resolved dependency versions in `agentic/README.md`.
2. `settings.py` + `test_settings.py` (green before any LangChain import matters).
3. `agents/hello.py` (tool + graph) + `test_hello_tool.py` + `test_graph_wiring.py`.
4. `cli/chat.py`; `.gitignore` entry for `.env`; `agentic/README.md`.
5. Verification (below), then update MASTER_PLAN.md.

---

## 7. Verification / definition of done

1. **Deterministic (CI-grade, no key):** `cd agentic && pytest` → all green, seconds.
   Engine untouched: `pytest` from the repo root still reports **135 passed**.
2. **DoD demo (user-run, needs `ANTHROPIC_API_KEY`):**
   ```bash
   python -m adaptrna_agentic.cli.chat --once "What is the GC content of GGCAUUACGGCU?"
   ```
   Expected: a printed `gc_content` tool call followed by an answer containing `0.583`
   (7 G/C of 12). Then one interactive REPL exchange. Cost: negligible (single short
   Opus 5 call).
3. `git status`: only `agentic/`, `plans/PHASE_0_SCAFFOLDING.md`, and the `.gitignore`
   line — nothing under the engine tree.
4. Close-out: tick Phase 0 in [MASTER_PLAN.md](MASTER_PLAN.md) §8 and record the per-role
   model decision in §10 with a one-line rationale.

## 8. Risks / notes

- **LangChain/LangGraph API drift** — the 1.x import paths (`init_chat_model`, `ToolNode`,
  `MessagesState`) must be confirmed against the versions actually installed; if an import
  moved, fix at the `models.py` / `hello.py` seam only.
- **Key handling discipline** — construction-time key checks keep every deterministic code
  path key-free; this property is load-bearing for all later phases' test suites and must
  not regress.
- **Opus 5 default cost** — fine for Phase 0's single calls; before Phase 4 makes the
  orchestrator chatty, revisit whether it should default to `claude-sonnet-5` (that
  revisit is already scheduled in MASTER_PLAN §10).
