# Phase 4 — Orchestrator MVP (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 4; agent topology in §5.
> **Definition of done:** "What tools are available?" and "is this sequence a donor site?"
> answered end-to-end from the terminal chat, adapter switching under the hood; sessions
> checkpointed (SQLite).
> Status: planned · not started

---

## 1. Context and goal

Everything below the LLM already exists and returns plain data: the ToolHub registry and
`AdapterRuntime` (Phase 2), external tools (Phase 3), the LangGraph wiring style and chat
REPL (Phase 0). Phase 4 connects them: the **orchestrator** — the one agent the user talks
to (MASTER_PLAN §5) — gets the ToolHub bound as agent tools, a persistent SQLite
checkpointer for sessions, and takes over the terminal chat. Flows **A** (inference) and
**B** (tool management) become conversational; flows C–E stay in Phases 5–6.

**Out of scope:** ToolSmith/Verifier and any codegen (Phase 6), training flows and
`interrupt()` approval gates (Phase 5 — nothing Phase 4's tools do is expensive or
irreversible; the checkpointer this phase installs is the prerequisite those gates need),
registering *new* tools from chat (Phase 5/6 flows own that), the HTTP API (Phase 8).

## 2. Decisions this plan fixes

| Decision | Choice | Rationale |
|---|---|---|
| **Orchestrator model (the §10 revisit)** | Keep `anthropic:claude-opus-5` as the default for the MVP; re-evaluate with real usage data at Phase 8 | Quality-first while the flows are new; per-role env overrides already exist as the cost lever. Recorded in §10 at close-out |
| **Tool binding policy** | Bind **every registered tool** (active *and* disabled — state noted in the description); execution enforces state at call time via the shared Registry/Runtime, refusing disabled tools with the activate hint | Makes "activate X, then use X" work within a single turn with no rebinding machinery, and teaches the model the lifecycle through the refusal message. Tool set is rebuilt at each user turn, so lifecycle changes are visible next turn at the latest |
| **One process, one runtime** | The chat process holds one `Registry` + one `AdapterRuntime` for its lifetime; management tools mutate exactly these instances | The backbone loads once per chat session (lazy on first FM call; `--warmup` for eager — the Phase 2 §10 note) |
| **Checkpointer** | `langgraph-checkpoint-sqlite` (3.1.1 verified on PyPI; new dependency) with `SqliteSaver` on `<repo>/chat_data/sessions.sqlite` (env `ADAPTRNA_CHAT_DIR`); session = LangGraph `thread_id`, chosen by `--session NAME` (default `default`) | Sessions survive process restarts — the DoD resume proof — and the same store serves Phase 8. `chat_data/` is git-ignored runtime state, sibling of `toolhub_data/` |
| **History ownership** | The checkpointer owns it: each turn invokes the graph with **only the new HumanMessage**; the `add_messages` reducer appends to the persisted thread | This is the semantic change vs Phase 0's in-process list — resuming a session needs no replay logic |
| **Layering** | The toolhub package stays LangChain-free; the bridge lives in `agents/tool_factory.py` | ToolHub remains usable (and testable) without any agent stack |

## 3. Components

```
agentic/adaptrna_agentic/agents/
├── orchestrator.py     # NEW — SYSTEM_PROMPT + build_orchestrator_graph(...)
├── tool_factory.py     # NEW — Registry/Runtime -> list[BaseTool]
└── hello.py            # unchanged (Phase 0 artifact; its tests keep passing)
agentic/adaptrna_agentic/cli/chat.py    # switches to the orchestrator; session flags
agentic/pyproject.toml                  # + langgraph-checkpoint-sqlite
.gitignore                              # + chat_data/
```

### 3.1 `tool_factory.py` — the ToolHub→LangChain bridge

`build_agent_tools(registry, runtime) -> list[BaseTool]`, two groups:

**Management tools** (closures over the shared instances, exactly the CLI's operations):

| tool | returns |
|---|---|
| `list_tools()` | `[{name, type, state, task, description}]` |
| `tool_info(name)` | the manifest entry as a dict |
| `activate_tool(name)` / `deactivate_tool(name)` | the new state |
| `test_tool(name)` | the smoke/golden report (type-dispatched like the CLI) |

**Capability tools**, one per manifest entry, named exactly like the entry:

- *Adapter entries* → `StructuredTool` with args `{sequences: list[str]}` calling
  `runtime.predict(name, sequences)` and returning JSON-able outputs. Description =
  manifest description + a task-output sentence (splice_site → "returns one probability
  per sequence") + serving notes; disabled entries get "(currently DISABLED — activate
  first)" appended.
- *External entries* → a state-checking wrapper around the wrapper-module function
  (`functools.wraps` preserves the signature and docstring, so `StructuredTool.from_function`
  infers the args schema from the *real* function — the contract's typed kwargs pay off
  here). Description from the FunctionSpec.

All capability tools set `handle_tool_error=True` so refusals and validation errors come
back as ToolMessages the model can act on (activate first, fix the sequence, …), instead
of crashing the turn. Name collisions between a manifest entry and a management tool are
refused at build time with a rename hint (edge case, but silent shadowing would be worse).

### 3.2 `orchestrator.py`

Same hand-built wiring as `hello.py` (model ⇄ ToolNode loop, conditional edge on
`tool_calls`) — that was the point of Phase 0 — with:

- `build_orchestrator_graph(model=None, registry=None, runtime=None, checkpointer=None)`
  — every argument injectable (the test seams); defaults: orchestrator model (lazy),
  default-data-dir Registry, its AdapterRuntime, no checkpointer.
- Tools are (re)built via `build_agent_tools` and bound **at each model-node call**, so a
  mid-turn `activate_tool` is honored by execution and the next turn's binding.
- `SYSTEM_PROMPT` (static, cache-friendly — specifics live in tool descriptions): the
  AdaptRNA assistant over one shared RNA foundation-model backbone plus classical tools;
  use tools for any prediction rather than guessing; report tool numbers faithfully
  (probabilities are probabilities); mention when a needed tool is disabled and offer to
  activate it; sequences arrive as plain ACGU/T strings.

### 3.3 `cli/chat.py` (rewired)

- Builds Registry + AdapterRuntime + `SqliteSaver` once; `--warmup` preloads the backbone.
- Flags: `--session NAME` (default `default`), `--new-session` (timestamped name, printed),
  `--list-sessions` (distinct `thread_id`s from the SQLite store), `--once`, `--model`
  (all-roles override, as today), `--data-dir` (ToolHub override, for demos/tests).
- Turn loop: send only the new HumanMessage with
  `config={"configurable": {"thread_id": session}}`; keep Phase 0's progress narration
  (`→ tool(args)` / `= result`, results truncated to ~200 chars) and final-answer print.
- The hello graph remains importable but the CLI now speaks orchestrator.

## 4. Tests (scripted fake model — no API key; nano backbone — no weights)

Extends the Phase 0 `ScriptedChatModel` pattern; ToolHub fixtures from Phases 2–3 reused
(`nano_registry`, `nano_splice_adapter`, dummy external module).

| test file | asserts |
|---|---|
| `test_tool_factory.py` | management + capability tools built with expected names/descriptions; disabled entry's description carries the DISABLED note; adapter tool schema takes `sequences: list[str]`; external tool schema mirrors the wrapper signature (`dummy_add(a, b)`); management tools mutate the *shared* registry; capability tool on a disabled entry returns the refusal as a tool result (not an exception); name-collision with a management tool refused |
| `test_orchestrator_graph.py` | scripted "list tools" turn: model calls `list_tools`, ToolMessage carries the names, loop terminates on the final answer; scripted FM turn: `splice_site` tool call runs a real nano prediction, ToolMessage has per-sequence probabilities in [0,1]; **activate-then-use in one turn** (3 scripted steps: `activate_tool` → capability call → answer) succeeds; system prompt injected once |
| `test_chat_sessions.py` | with `SqliteSaver` on a tmp path: two invocations with the same `thread_id` see accumulated history (the scripted model's second call receives the first turn's messages); different `thread_id`s are isolated; state survives a *new* graph instance on the same DB file (the restart-resume proof, no process restart needed) |
| CLI | `--list-sessions` output; `--once` exits cleanly (fake model injected via a test hook or covered implicitly by the graph tests — keep CLI assertions light) |

Expected: ~15–18 new tests; the 91 existing stay green; engine 135 untouched.

## 5. Implementation order

1. Add `langgraph-checkpoint-sqlite` to `agentic/pyproject.toml`; `pip install -e ./agentic`;
   record the version in `agentic/README.md`.
2. `tool_factory.py` + `test_tool_factory.py`.
3. `orchestrator.py` + `test_orchestrator_graph.py`.
4. Checkpointer wiring + `test_chat_sessions.py`; rewire `cli/chat.py`; `.gitignore` +
   `chat_data/`.
5. DoD conversation (below); README updates (agentic README chat section; root README
   line already mentions the chat CLI).
6. Close-out: MASTER_PLAN §8 tick; §10 "Model per agent role" row updated with the
   revisit outcome ("kept opus-5 for MVP; re-evaluate with usage data at Phase 8").

## 6. Verification / definition of done (live, this machine: key in `.env`, GPU, weights)

One scripted terminal session (plus a restart) demonstrating flows A and B:

```bash
python -m adaptrna_agentic.cli.chat --session dod
you> What tools are available?
# → list_tools; answer names splice_site, vienna_fold, vienna_cofold with one-liners
you> Is this sequence a donor splice site? <a real 400 nt Danio positive window>
# → splice_site tool; backbone loads (one-time pause); answer reports p ≈ 0.99+  ← DoD
you> And fold GGGGAAAACCCC for me.
# → vienna_fold; answer shows ((((....)))) at −5.4 kcal/mol
you> Disable the vienna fold tool.
# → deactivate_tool; confirmed
you> Fold AAAA please.
# → tool refuses (disabled) → model explains and offers to re-activate  ← flow B
you> Re-enable it and fold AAAA.
# → activate_tool + vienna_fold in one turn                              ← bind policy proof
quit

python -m adaptrna_agentic.cli.chat --session dod --once "What was the first thing I asked you?"
# → answers from checkpointed history                                    ← session DoD
```

Deterministic gate first: `cd agentic && pytest` all green (existing 91 + new), engine 135
untouched, `git status` clean of engine changes.

## 7. Risks and notes

- **Token cost per turn**: opus-5 + tool results in a growing thread. Mitigations now:
  truncated tool-result *display* (context still carries full results), small tool
  outputs by design, static system prompt (prompt-cache-friendly). Measure before Phase 8;
  the per-role override is the immediate lever.
- **Tool-result size**: fine for splice/mrl/vienna (scalars, short strings); a future
  sec_struct adapter returns L×L matrices — the adapter tool wrapper should cap and
  summarize anything huge (one sentence + shape) rather than dump it into context; note
  left in `tool_factory.py`.
- **Loop safety**: LangGraph's default recursion limit (25) kept; the scripted tests pin
  termination behavior.
- **Stale bindings**: registration of *new* tools mid-chat isn't a Phase 4 flow; if done
  via a second terminal, the per-turn rebind picks it up next turn — acceptable, noted in
  the CLI help.
- **Checkpointed threads accumulate** in SQLite; no retention policy yet — Phase 7
  (hardening) owns cleanup/expiry.
