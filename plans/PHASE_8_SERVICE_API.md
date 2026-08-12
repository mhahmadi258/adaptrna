# Phase 8 — Service API (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 8; architecture §2 (the HTTP + SSE
> layer between `ui/` and `agentic/`).
> **Definition of done:** terminal and HTTP clients share sessions; the Phase-4 demo works
> over HTTP.
> Status: planned · not started

---

## 1. Context and goal

Everything the platform does is already reachable from Python and the terminal. Phase 8
puts an HTTP surface on exactly that — no new capability, no second implementation — so
Phase 9's browser UI has something to talk to, and so a session started in the terminal
can be continued in a browser and vice versa.

The pieces line up: the orchestrator graph is already injectable and checkpointed
(Phase 4), every ToolHub and job operation already returns plain data (Phases 2–5), the
approval gate is already a graph interrupt that suspends and resumes (Phase 5), and
`doctor` already reports install health as JSON (Phase 7).

Verified at plan time (2026-08-13): **fastapi 0.136, uvicorn 0.45, starlette 1.0 and
httpx 0.28 are already in the venv** (transitively), and LangGraph's `stream` accepts
`values | updates | messages | custom`, so token-level streaming needs no new dependency.
`chat_data/sessions.sqlite` is **already in WAL mode with a 5 s busy timeout**, which is
what makes two processes sharing sessions viable.

**Out of scope:** the web UI itself (Phase 9), multi-user auth or accounts, remote/cluster
execution, and anything that would introduce a *second* way to do something the CLI
already does.

## 2. Decisions this plan fixes

| Decision | Choice | Rationale |
|---|---|---|
| **Approval over HTTP** | The SSE stream emits an `approval_required` event and **ends**. The client decides, then `POST /sessions/{id}/resume`, which returns a *new* SSE stream continuing the same turn | The gate can pause for minutes. Holding an SSE connection open across a human decision strands the turn on any dropped connection; ending the stream keeps the whole thing resumable from the checkpointer, which is exactly what it is for |
| **One runtime, one lock** | The app holds a single `Registry` + `AdapterRuntime` for its lifetime, and **adapter inference is serialised behind a lock** | `RiNALMoHub.predict` calls `activate()` → `set_active_adapter`, which mutates *every* tuner layer in the shared backbone. Two concurrent predictions on different adapters would interleave that mutation and silently answer with the wrong adapter. This is the one place concurrency is genuinely unsafe |
| **Sync work off the event loop** | Route handlers that touch the graph, the GPU or the job store are `def` (not `async def`), so Starlette runs them in its threadpool | A blocking graph invocation inside the event loop would freeze every other request, including `/health` |
| **Binding and auth** | `127.0.0.1` by default; a bearer token required only if `ADAPTRNA_API_TOKEN` is set; binding to a non-loopback address **requires** the token and refuses to start without it | This API can start GPU jobs and write code into the repo. Localhost-only is the safe default, and the refusal makes the dangerous configuration impossible to reach by accident |
| **What is *not* exposed** | No `prune`, no tool removal, no arbitrary file paths for training data | Phase 7 deliberately kept deletion a human action at the CLI. An HTTP endpoint is a weaker boundary than a shell prompt, and nothing in the UI needs it yet |
| **Session sharing** | Both processes open the same SQLite file; the API sets WAL + `busy_timeout` explicitly rather than relying on what the file happens to have | The DoD depends on it, and a fresh `chat_data/` would otherwise start in rollback-journal mode where concurrent readers and a writer collide |
| **Streaming granularity** | `stream_mode=["updates", "messages"]`: `messages` gives token-level text for a responsive UI, `updates` gives the tool calls and results | One stream carries everything Phase 9 needs to render; no polling |

## 3. Surface

```
GET    /health                          liveness + doctor status only (no detail)
GET    /api/doctor                      the full Phase 7 report
GET    /api/tools                       list
GET    /api/tools/{name}                manifest entry
POST   /api/tools/{name}/activate       -> new state
POST   /api/tools/{name}/deactivate
POST   /api/tools/{name}/test           smoke (adapter) / golden (external) report
POST   /api/tools/{name}/predict        {sequences: [...]} -> predictions      [locked]
POST   /api/tools/{name}/call           {args: {...}} -> result
GET    /api/jobs                        list, newest first
GET    /api/jobs/{id}                   status + progress
GET    /api/jobs/{id}/logs?tail=N       log tail
POST   /api/jobs/{id}/cancel
GET    /api/sessions                    session ids
GET    /api/sessions/{id}/history       messages so far (for a reconnecting UI)
POST   /api/sessions/{id}/messages      {text} -> SSE stream of the turn
POST   /api/sessions/{id}/resume        {approved, note?} -> SSE stream continuing it
```

### SSE event schema (the contract Phase 9 renders)

One `event:` type per line-block, `data:` a JSON object:

| event | payload | meaning |
|---|---|---|
| `text` | `{delta}` | assistant text, token by token |
| `tool_call` | `{name, args}` | a tool is about to run |
| `tool_result` | `{name, content}` | its result (truncated for display, full in history) |
| `approval_required` | `{requests: [{id, tool, summary, details}]}` | the turn is suspended; the stream ends after this |
| `done` | `{answer}` | the turn finished |
| `error` | `{message, type}` | something failed; the stream ends |

`approval_required` carries the same payload the terminal renders — including the exact
command line for `start_training` and the file list for `land_generated_code` — so the
browser can show the user precisely what the CLI would have.

### Error mapping

| Raised | HTTP | Body |
|---|---|---|
| `KeyError` (unknown tool/job/session) | 404 | `{error, known}` |
| `ToolHubError` (refusal: disabled tool, unstamped plan, LoRA-only, …) | 409 | `{error, type}` |
| `ConcurrentModificationError` | 409 | `{error, retryable: true}` |
| `FileNotFoundError`, validation | 400 | `{error}` |
| anything else | 500 | `{error: "internal error", request_id}` |

The bodies carry the same actionable messages the CLI prints — Phase 7 standardised them
precisely so a second front end would not need its own wording.

## 4. Components

```
agentic/adaptrna_agentic/api/
├── app.py            # create_app(): lifespan-owned singletons, error handlers, routers
├── deps.py           # Registry / AdapterRuntime / graph / checkpointer + the inference lock
├── events.py         # SSE encoding; the turn generator (stream -> events)
├── schemas.py        # pydantic request/response models
└── routers/{system,tools,jobs,sessions}.py
agentic/adaptrna_agentic/cli/serve.py     # python -m adaptrna_agentic.cli.serve
```

`deps.py` is the whole concurrency story in one file: one process-wide `AdapterRuntime`,
one `SqliteSaver` (WAL, `check_same_thread=False`), one compiled graph, and the
`inference_lock` that `predict` and the chat's adapter tools both take. Nothing else in
the API is stateful.

`events.py` holds the only genuinely new logic: turn a LangGraph stream into SSE events,
detect the interrupt, and stop cleanly. It is a plain generator over the graph, so it is
testable without a server.

## 5. Tests (deterministic: scripted models, nano backbone, no key, no network)

FastAPI's `TestClient` (starlette, already installed) drives everything in-process.

| test file | asserts |
|---|---|
| `test_api_tools.py` | list/info/activate/deactivate/test/predict/call round trips; a disabled tool returns **409** with the activate hint; unknown tool **404** listing known ones; external `call` on an adapter (and vice versa) is refused with the pointer to the right endpoint |
| `test_api_jobs.py` | list/status/logs against a fake job; cancel on a finished job → 409; unknown id → 404; a job whose PID was recycled → the Phase 7 refusal surfaces as 409, not 500 |
| `test_api_sessions.py` | `POST /messages` streams `tool_call` → `tool_result` → `text` → `done`; history endpoint returns the turn; **a session written by the terminal is visible to the API and continues correctly** (the DoD, exercised in-process against one SQLite file) |
| `test_api_approval.py` | a gated call streams `approval_required` and **the stream ends with no job started**; `POST /resume {approved:false}` yields the declined tool result and still no job; `{approved:true}` runs it; resuming a session with nothing pending → 409 |
| `test_api_concurrency.py` | two concurrent `predict` calls for **different** adapters return each adapter's own answer (with the lock; the test is the regression guard for the `set_adapter` hazard); a slow prediction does not block `/health` |
| `test_api_security.py` | no token configured → localhost requests work; token configured → 401 without it, 200 with it; binding to a non-loopback host without a token **refuses to start** |
| `test_api_errors.py` | the mapping table above, end to end |

~40 new tests (total ~337). Phases 0–7's 297 stay green; engine's 135 untouched.

## 6. Implementation order

1. `deps.py` (singletons, lock, WAL) + `app.py` skeleton + `/health`; `test_api_security.py`.
2. `routers/tools.py`, `routers/jobs.py`, `routers/system.py` + their tests.
3. `events.py` + `routers/sessions.py` (streaming) + `test_api_sessions.py`.
4. Approval flow (`/resume`) + `test_api_approval.py`.
5. `test_api_concurrency.py` (the lock's regression guard).
6. `cli/serve.py`; deps into `pyproject.toml` (fastapi, uvicorn — both already resolved);
   README section; close-out (MASTER_PLAN §8 tick).

## 7. Verification / definition of done

**Gate 1 — deterministic:** `cd agentic && pytest` green (297 + ~40); `cd engine && pytest`
→ 135.

**Gate 2 — the Phase-4 demo over HTTP**, against the live install (real backbone, real
adapters), driven with `curl`:

```bash
python -m adaptrna_agentic.cli.serve &          # 127.0.0.1:8000

curl -s localhost:8000/api/tools | jq '.[].name'
#   -> splice_simple, splice_site, splice_site_acceptor, vienna_cofold, vienna_fold

curl -sN -X POST localhost:8000/api/sessions/http-demo/messages \
     -d '{"text":"What tools are available?"}' -H 'content-type: application/json'
#   -> event: text ... event: done          (streamed)

curl -sN -X POST localhost:8000/api/sessions/http-demo/messages \
     -d '{"text":"Is <400nt window> a donor splice site?"}' ...
#   -> event: tool_call {splice_site} / tool_result / text / done   ← the Phase-4 demo
```

**Gate 3 — sessions are genuinely shared** (the DoD's own words):

```bash
python -m adaptrna_agentic.cli.chat --session shared --once "Remember the number 42."
curl -sN -X POST localhost:8000/api/sessions/shared/messages \
     -d '{"text":"What number did I ask you to remember?"}' ...
#   -> answers 42 from the terminal-written checkpoint
python -m adaptrna_agentic.cli.chat --session shared --once "And what did I ask you over HTTP?"
#   -> sees the HTTP turn
```

**Gate 4 — the approval gate over HTTP**: ask for a training run; the stream ends on
`approval_required` with the exact command; confirm no job exists; `POST /resume` with
`approved:false`; confirm still no job; then approve a real (short) run and watch
`/api/jobs/{id}` progress.

## 8. Risks and notes

- **The `set_adapter` hazard is the real one.** `predict` mutates the shared backbone's
  active adapter on every layer, so without the lock two simultaneous requests can answer
  from the wrong adapter — a *silent* wrong answer, the failure class this project treats
  most seriously. The concurrency test exists to keep the lock honest.
- **A lock means serialised inference.** Throughput is one prediction at a time. That is
  correct before it is fast; batching or a second backbone is a later question, and the
  master plan's fp32 note (§7) already caps throughput anyway.
- **SSE through proxies** buffers unless disabled. Documented for Phase 9; irrelevant on
  loopback.
- **This API can spend GPU hours and write code.** Loopback default, token required for
  any other binding, no delete surface. Phase 9 is same-origin, so nothing here needs CORS
  by default.
- **Two writers, one manifest.** Phase 7's revision counter turns a terminal/API collision
  into a retryable 409 rather than a silent loss — the API surfaces it as such rather than
  retrying automatically, because the caller may want to re-read first.
