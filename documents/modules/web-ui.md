# `ui/` — the browser client

`ui/` (repo root, per the master plan's three-directory split)

Streamed chat, a tool dashboard, a live training monitor, and the approval gate as a modal.
Served by the API at `/` — open a running server, or launch one with
`python -m adaptrna_agentic.cli.serve --open`.

---

## Contents

1. [No build step, and why](#1-no-build-step-and-why)
2. [File map](#2-file-map)
3. [`sse.js` — the only real client logic](#3-ssejs--the-only-real-client-logic)
4. [`api.js`](#4-apijs)
5. [`app.js` — wiring and state](#5-appjs--wiring-and-state)
6. [`render.js`, `md.js`, `dom.js`](#6-renderjs-mdjs-domjs)
7. [Two behaviours worth knowing before editing](#7-two-behaviours-worth-knowing-before-editing)
8. [How the client is tested](#8-how-the-client-is-tested)
9. [Assumptions and limitations](#9-assumptions-and-limitations)

---

## 1. No build step, and why

Plain ES modules and one stylesheet, loaded straight by the browser. No `package.json`, no
bundler, no `node_modules`. The UI ships with `pip install -e ./agentic` and works offline,
because nothing is fetched from a CDN.

This was a deliberate reversal of the master plan's React default, decided on **fit rather
than feasibility** (Node 20 was available):

> This client is a *renderer of server state* — conversations live in the checkpointer,
> tools in the manifest, jobs in the job store — so there is no client-side domain model to
> justify a second dependency ecosystem in a 164-file Python repo, for four views.

Gradio was rejected for a different reason: it wants to own the app loop, so the approval
gate and the training monitor would have been reimplemented a third time.

**The tripwire is explicit**: past ~1000 client lines, a second maintainer, or real
client-side state, port to React. At 1,109 lines of JS this sits *at* that threshold, not
comfortably under it. It stays justified because none of those lines are client-side state —
but the next feature that adds some should re-open the question. The API is the contract, so
switching costs the client alone.

## 2. File map

| File | Lines | What it does |
|---|---:|---|
| `index.html` | 70 | The shell: topbar (brand, health badge, session picker), chat column, tools/jobs/inspector sidebar, and the approval modal skeleton. Loads `/ui/app.js` as a module. |
| `app.js` | 418 | Wiring — panels, session picker, job polling, the event dispatch for a turn |
| `sse.js` | 105 | SSE over `fetch`: a pure frame parser plus a thin transport |
| `api.js` | 109 | The endpoints as functions, and the bearer token when one is configured |
| `render.js` | 263 | Messages, tool rows, job rows, test/analysis reports, the approval modal body |
| `md.js` | 180 | A small Markdown renderer — the model writes tables, and raw pipes are unreadable |
| `dom.js` | 34 | The shared `el()` / `clear()` helpers |
| `style.css` | 431 | One stylesheet, light and dark |

## 3. `sse.js` — the only real client logic

**`EventSource` cannot POST.** A chat turn is a POST carrying a JSON body, so frames are
parsed here instead. That is this layer's one piece of genuine logic, which is why the
parser is a pure function of text and the transport is a thin wrapper around it — the hard
part is testable without a network.

```js
parseFrame(block) -> {event, data} | null      // one SSE block; null if it named no event
createFrameParser() -> push(chunk) -> frames[] // stateful buffer across chunk boundaries
streamEvents(url, options, onEvent)            // POST + consume + dispatch
```

Frames arrive split across arbitrary chunk boundaries, so the parser buffers and splits on
`\n\n` rather than doing a `split()` over one string. It handles `\r\n` line endings, skips
blank lines and `:` keep-alive comments, strips one leading space after the field colon, and
joins multiple `data:` lines with newlines per the spec — the server emits single-line JSON,
but a payload that ever grows a newline should not silently become unparseable.

A failed request is reported as an `error` **frame** rather than thrown, so callers have one
place to handle "the turn went wrong" instead of two.

## 4. `api.js`

Same-origin, so on loopback there is nothing to authenticate. When the server was started
with `ADAPTRNA_API_TOKEN`, the UI asks for it once and keeps it in **`sessionStorage`** —
not `localStorage`, so it dies with the tab rather than outliving the session on disk.

```js
api.health() api.doctor()
api.tools() api.tool(n) api.activate(n) api.deactivate(n) api.testTool(n) api.predict(n, seqs)
api.jobs() api.job(id) api.jobLogs(id, tail) api.jobAnalysis(id) api.cancelJob(id)
api.sessions() api.history(id)
sendMessage(session, text, onEvent)              // SSE
resumeSession(session, approved, note, onEvent)  // SSE
```

Errors carry `status` and `retryable` onto the thrown `Error`, and the message is
`payload.error` — the server's own text:

> Phase 7 standardised these messages so a second front end would not need its own wording:
> what the browser shows is what the terminal prints.

Path segments go through `encodeURIComponent`.

## 5. `app.js` — wiring and state

The server holds all the domain state, so this file keeps only what is genuinely about *this
browser tab*:

```js
const state = { session, pending, streaming, jobsTimer };
```

### `consume(run)` — the event dispatch

Used for **both** a new turn and a resumed one, because `/resume` answers with a new stream
continuing the same turn:

| Event | Action |
|---|---|
| `text` | Open a bubble if needed, accumulate raw Markdown, repaint on a timer |
| `tool_call` | Close the current bubble (text after a tool call belongs in a new one), append a call row |
| `tool_result` | Close the bubble, append a result row, mark `touchedTools` |
| `approval_required` | Close the bubble, store `state.pending`, show the modal |
| `done` | Close the bubble, clear `pending` |
| `error` | Close the bubble, append an error message |

Afterwards `closeBubble()` runs again — a stream that ended without `done` still leaves
rendered text — and the panels refresh if any tool was touched.

**Markdown is re-parsed from the whole reply on every repaint**, so tokens are coalesced into
frames rather than repainting per delta (`MARKDOWN_REPAINT_MS = 80`), with a forced repaint
when the block closes. `painted = 0` on a new bubble so its first token paints immediately.

### Job polling

`JOB_POLL_MS = 3000`. `list` is cheap and carries no progress, so only **running** jobs get a
second call to `/api/jobs/{id}`. Polling re-arms only while at least one job is running.

The `catch` branch is load-bearing and carries the reason in a comment:

```js
} catch {
  // Keep the last good render, but KEEP POLLING. Phase 7's optimistic concurrency answers
  // 409 while the job store is mid-write — most likely just after a run starts, exactly
  // when someone is watching this panel. Returning without re-arming froze the monitor.
  state.jobsTimer = setTimeout(refreshJobs, JOB_POLL_MS);
  return;
}
```

### Boot sequence

`refreshHealth()` → `ensureAuth()` (prompts for a token only on a 401, then retries) →
pick the session from `?session=`, else the first existing, else `default` →
`refreshSessions` → `loadSession` → `refreshPanels`.

`refreshHealth` counts `failed_checks` rather than testing it, because **an empty array is
truthy in JS**. Clicking the badge opens the full doctor report in the inspector.

## 6. `render.js`, `md.js`, `dom.js`

`dom.js` exports `el(tag, props, ...children)` and `clear(node)`. **Nothing in the client
assigns `innerHTML`** — escaping is a property of how nodes are made rather than something
each call site has to remember. Model output, tool results and job logs all flow through
here.

`render.js` exports one function per visual element: `userMessage`, `assistantMessage`
(returning `{node, set(markdown)}`), `toolCallRow`, `toolResultRow`, `errorMessage`,
`noticeMessage`, `toolRow(entry, handlers)`, `testResult(report)`, `jobRow(job, status,
handlers)`, `analysisReport(report)`, `approvalBody(request)`.

`assistantMessage().set()` **re-renders rather than appends**, and that is what makes
streaming Markdown work at all: a table or a code fence only becomes meaningful once its
later lines have arrived, so there is nothing to append *to* until the block is complete.

`md.js` is a deliberately small Markdown renderer — headings, emphasis, code, lists, links
and **tables**, which is the reason it exists (the model writes tables, and raw pipes are
unreadable).

## 7. Two behaviours worth knowing before editing

### The approval round trip is three steps

The stream ends on `approval_required`; the modal collects a decision; `/resume` returns a
**new** stream continuing the same turn. There is no separate resume path in `app.js` —
`consume()` is simply called again with a different `run`.

### A refresh mid-approval must not strand the turn

```js
if (body.pending_approval) { state.pending = …; showApproval(…); }
```

`GET /api/sessions/{id}/history` carries `pending_approval` precisely so the dialog can come
back after a reload. The suspended turn lives in the checkpointer, not in the tab.

## 8. How the client is tested

Three suites, in increasing cost:

| Suite | What it proves |
|---|---|
| `agentic/tests/test_ui_serving.py` | The shell is served, the assets are mounted, and the client is genuinely **self-contained** — the offline check is the one with teeth, since a single CDN `<script>` would break a workstation with no internet while passing everything else |
| `agentic/tests/test_ui_contract.py` | **The compiler this pair of languages does not have.** Every assertion names a field or event that `ui/*.js` reads *by name*, so a server-side rename fails in `pytest` — naming the client file — instead of silently blanking a panel in someone's browser |
| `agentic/tests/test_ui_browser.py` | Opt-in (`pytest -m ui`, needs Playwright + Chromium). The only tests that prove the JavaScript actually runs: that the fetch-based SSE reader assembles frames, that the modal collects a decision, that the monitor updates in place |

```bash
cd agentic
../.venv/bin/python -m pytest tests/test_ui_serving.py tests/test_ui_contract.py
../.venv/bin/python -m pip install playwright && ../.venv/bin/python -m playwright install chromium
../.venv/bin/python -m pytest -m ui
```

## 9. Assumptions and limitations

* **Same-origin only.** No CORS configuration; the client assumes it is served by the API.
* **One session at a time per tab.** Switching sessions reloads the log from `/history`.
* **No client-side domain state** — by design. Everything rendered comes from the server on
  each refresh.
* **The token prompt is a `window.prompt`.** Adequate for a single-user loopback tool; not a
  login flow.
* **No delete affordances**, because the API has no delete surface. Removing a tool or
  pruning is a CLI action.
* **`md.js` is not a complete Markdown implementation.** It covers what the model actually
  writes; unusual constructs render as text.
* **Job logs are fetched with `tail=200`** from the inspector, not streamed.
