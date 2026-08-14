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
7. [The activity bar and the rail](#7-the-activity-bar-and-the-rail)
8. [Two behaviours worth knowing before editing](#8-two-behaviours-worth-knowing-before-editing)
9. [How the client is tested](#9-how-the-client-is-tested)
10. [Assumptions and limitations](#10-assumptions-and-limitations)

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

**The tripwire has been restated.** Phase 9 set it as a line count — past ~1000 client
lines, a second maintainer, or real client-side state, port to React — and by Phase 11 that
number (now **1,648 lines of JS**) had been renegotiated twice without anything changing.
A threshold that moves every phase is not a threshold, so the count was replaced with two
conditions that can actually fire:

> Port when the client first holds state the server does not — an optimistic update, a local
> edit buffer, an undo stack — or when a third mode joins chat and job-log in the centre
> column. Until then, **extract rather than port**: the next module out is the job-log
> controller (`ui/jobs.js`), which is already self-contained.

Neither has fired. `view`, `job` and `logFollow` are *which pane is on screen*; `jobs` and
`jobStatus` are a render cache of a list the server owns, exactly like `sessions`. None of it
can disagree with the server. The API is the contract, so switching still costs the client
alone.

## 2. File map

| File | Lines | What it does |
|---|---:|---|
| `index.html` | 90 | The shell: topbar (rail toggle, brand, health badge), activity bar, the rail, the centre column (chat **or** job log), tools/inspector sidebar, and the approval modal skeleton. Loads `/ui/app.js` as a module. |
| `app.js` | 821 | Wiring — the activity bar, both rail views, job and log polling, the event dispatch for a turn |
| `sse.js` | 105 | SSE over `fetch`: a pure frame parser plus a thin transport |
| `api.js` | 114 | The endpoints as functions, and the bearer token when one is configured |
| `render.js` | 394 | Messages, session rows, tool rows, job rows, the job-log header, test/analysis reports, the approval modal body |
| `md.js` | 180 | A small Markdown renderer — the model writes tables, and raw pipes are unreadable |
| `dom.js` | 34 | The shared `el()` / `clear()` helpers |
| `style.css` | 652 | One stylesheet, light and dark |

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
api.createSession(id) api.renameSession(id, next) api.deleteSession(id)
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
const state = {
  session, pending, streaming, sessions,          // the conversation half
  view, job, jobs, jobStatus,                     // which pane is on screen, and the run cache
  filter: { sessions, jobs },                     // per view, so a needle does not follow you
  jobsTimer, logTimer, logTail, logFollow, logError,
};
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

Two timers, both owned by `state`:

| Timer | Cadence | Stops when |
|---|---|---|
| `jobsTimer` (`refreshJobs`) | `JOB_POLL_MS = 3000` while any run is running, `JOB_IDLE_POLL_MS = 15000` otherwise | The document is hidden. `visibilitychange` starts it again. |
| `logTimer` (`refreshJobLog`) | `JOB_POLL_MS` while the open run is running | The run reaches a terminal state, the log 404s, or the log is closed. |

`refreshJobs` runs **whichever view is open** — a run started from the chat has to appear in
the rail on its own, and the dot on the Jobs icon is only honest if something is counting.
Before Phase 11 it stopped re-arming once nothing was running, so a new run stayed invisible
until something else forced a refresh.

The expensive half is gated on visibility rather than on liveness: `list` is cheap and
carries no progress, so the second call to `/api/jobs/{id}` fires **only when the Jobs rail
is on screen**. Idle cost is therefore one request per 15 s per tab.

The `catch` branch is load-bearing and carries the reason in a comment:

```js
} catch {
  // Keep the last good render, but KEEP POLLING. Phase 7's optimistic concurrency answers
  // 409 while the job store is mid-write — most likely just after a run starts, exactly
  // when someone is watching. Returning without re-arming froze the monitor.
  state.jobsTimer = setTimeout(refreshJobs, JOB_POLL_MS);
  return;
}
```

`refreshJobLog` follows the same rule with one exception: a **404** stops it for good,
because the run's output directory is gone and there is nothing to come back for.

**The log is a polled tail, never a stream.** `train.log` is a plain file a detached process
appends to, so `GET /api/jobs/{id}/logs?tail=N` on a timer is the whole mechanism.
`LOG_TAIL_CHOICES = [200, 500, 2000]` is read by `test_ui_contract.py`, which asks the real
server to honour every value in it — the endpoint caps `tail` at 2000, and a choice past the
cap would reach the user as an empty log.

Following the tail is a consequence of being at the bottom, not a mode: scrolling up turns it
off, scrolling back turns it on, and the checkbox in the header mirrors it.

### Boot sequence

`installRail()` (restores width/collapsed from `localStorage`) → `installActivity()` →
`refreshHealth()` → `ensureAuth()` (prompts for a token only on a 401, then retries) → pick
the session from `?session=`, else the newest existing, else `default` → `loadSession` →
`refreshSessions` → `refreshPanels` → `setView` (restores the open view).

`loadSession` runs **before** `refreshSessions`: `state.session` is what marks the current
row in the rail, so rendering the list first would leave nothing highlighted. `setView` runs
**last**, so booting straight into the Jobs view still leaves a populated session list behind
it.

`refreshHealth` counts `failed_checks` rather than testing it, because **an empty array is
truthy in JS**. Clicking the badge opens the full doctor report in the inspector.

## 6. `render.js`, `md.js`, `dom.js`

`dom.js` exports `el(tag, props, ...children)` and `clear(node)`. **Nothing in the client
assigns `innerHTML`** — escaping is a property of how nodes are made rather than something
each call site has to remember. Model output, tool results and job logs all flow through
here.

`render.js` exports one function per visual element: `userMessage`, `assistantMessage`
(returning `{node, set(markdown)}`), `toolCallRow`, `toolResultRow`, `errorMessage`,
`noticeMessage`, `toolRow(entry, handlers)`, `testResult(report)`, `progressLines(progress)`,
`jobRow(job, status, handlers, isCurrent)`, `jobLogHead(job, status, handlers, options)`,
`analysisReport(report)`, `approvalBody(request)`.

`progressLines` is shared by the rail row and the log header so epoch/step and the metric
chips render identically in both.

`assistantMessage().set()` **re-renders rather than appends**, and that is what makes
streaming Markdown work at all: a table or a code fence only becomes meaningful once its
later lines have arrived, so there is nothing to append *to* until the block is complete.

`md.js` is a deliberately small Markdown renderer — headings, emphasis, code, lists, links
and **tables**, which is the reason it exists (the model writes tables, and raw pipes are
unreadable).

## 7. The activity bar and the rail

A static 3 rem icon bar decides what the rail beside it lists. Phase 10 built the rail for
sessions; Phase 11 gave runs the same treatment and took the icon-bar idea from VS Code.

```
nav#activity      💬 sessions · ⏱ jobs (+ #activity-dot when a run is live)
aside#rail
├── .rail-head    #rail-new (＋ New session — sessions view only) · #rail-filter
├── #rail-list    .session-row × n   — name, relative time, ✎ rename, 🗑 delete
│                 .job-row × n       — dot, id, state, epoch/step, metrics, analysis, cancel
└── #rail-grip    4px drag handle on the right edge
```

**Both views render into `#rail-list`**, and each render function returns early if it is not
the open view, so neither can clobber the other. `RAIL_VIEWS` is an object map rather than a
`switch` — `test_ui_contract.py` reads every `case "…"` in `app.js` and asserts the set is
exactly the six streamed event names, so a `switch (state.view)` would fail a test about
streaming.

**There is no "＋ New run".** Starting a run is a gated action in the chat, so the human sees
the exact argv before GPU time is spent; `#rail-new` is hidden in the Jobs view and the empty
state says why rather than leaving someone hunting for the button.

**Clicking the open view collapses the rail**, the way VS Code does. The activity bar itself
never collapses — it is the only way back.

**The grip measures from the rail's own left edge.** It used to divide `clientX` by the root
font size directly, which was correct only while the rail started at viewport x = 0; the
activity bar made every drag report a width one bar too wide.
`test_the_grip_resizes_from_the_rails_own_left_edge` is the guard.

**Filtering is local, and per view.** `state.sessions` / `state.jobs` hold the last fetched
lists and the render functions are pure over them plus `state.filter[view]`, so typing does
not hit the server — and a needle typed in one view does not follow you to the other.

**Geometry and the open view live in `localStorage`** (`adaptrna.rail.width`,
`adaptrna.rail.collapsed`, `adaptrna.rail.view`),
deliberately unlike the auth token's `sessionStorage`. A token that outlives the tab is a
hazard; a sidebar width that resets on every tab is just an annoyance. The width is written
as a `--rail-w` custom property on `:root`, which `.layout`'s grid reads — so the drag handle
touches one property and the browser does the layout.

**Mutations are refused mid-turn.** Rename and delete move or destroy the rows a streaming
turn is writing to, so all three handlers bail out through `busyWithATurn()` when
`state.streaming` or `state.pending` is set.

**A new session is created server-side before it is opened**, because a thread only appears
in the listing once it has a checkpoint. `POST /api/sessions` seeds an empty one; without
that, a session created in the rail would vanish on the next refresh.

## 8. Two behaviours worth knowing before editing

### `[hidden]` loses to an author `display`

The thinking dots animated permanently for the whole of Phase 9. `app.js` set
`$("thinking").hidden` correctly the entire time — but `style.css` said
`.thinking { display: flex }`, and an **author-origin** `display` beats the user agent's
`[hidden] { display: none }` whatever the specificity. The element was painted regardless of
the attribute. Any element this stylesheet gives a `display` to and JS hides needs an
explicit `[hidden]` rule. Four exist: `.thinking[hidden]`, `.overlay[hidden]`, and — added
with the centre-column swap — `.chat[hidden]`, `.joblog[hidden]` and `.activity-dot[hidden]`.
A browser test asserts `#composer-input` is *not visible* while a log is open, which is what
catches the next one if it is missed.

The dots now also mean something narrower: "waiting on the model", not "a stream is open".
`consume()` turns them off on the first `text` delta and back on in `closeBubble()`, so they
are dark while an answer is streaming and lit while a tool call is in flight.

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

## 9. How the client is tested

Three suites, in increasing cost:

| Suite | What it proves |
|---|---|
| `agentic/tests/test_ui_serving.py` | The shell is served, the assets are mounted, and the client is genuinely **self-contained** — the offline check is the one with teeth, since a single CDN `<script>` would break a workstation with no internet while passing everything else |
| `agentic/tests/test_ui_contract.py` | **The compiler this pair of languages does not have.** Every assertion names a field or event that `ui/*.js` reads *by name*, so a server-side rename fails in `pytest` — naming the client file — instead of silently blanking a panel in someone's browser |
| `agentic/tests/test_ui_browser.py` | Opt-in (`pytest -m ui`, needs Playwright + Chromium). The only tests that prove the JavaScript actually runs: that the fetch-based SSE reader assembles frames, that the modal collects a decision, that the monitor updates in place, that the activity bar swaps the centre column, and that the grip resizes from the right edge |

```bash
cd agentic
../.venv/bin/python -m pytest tests/test_ui_serving.py tests/test_ui_contract.py
../.venv/bin/python -m pip install playwright && ../.venv/bin/python -m playwright install chromium
../.venv/bin/python -m pytest -m ui
```

## 10. Assumptions and limitations

* **Same-origin only.** No CORS configuration; the client assumes it is served by the API.
* **One session at a time per tab.** Switching sessions reloads the log from `/history`.
* **Almost no client-side domain state.** Everything rendered comes from the server. The
  exceptions are caches and chrome — `state.sessions` / `state.jobs` (so filtering is local),
  the rail's persisted geometry, and which view and run are on screen. None of it can
  disagree with the server, which is the condition §1's tripwire now turns on.
* **The job log is a polled tail, not a stream.** `GET /api/jobs/{id}/logs?tail=N` every 3 s
  while the run is running; there is no follow endpoint, no SSE for jobs, and `tail` is
  capped at 2000 lines server-side. A log longer than the chosen tail loses its head.
* **Jobs are read-only plus cancel.** No start (that is the gated path through the chat) and
  no delete (runs stay a CLI-only deletion — `toolhub prune`).
* **The token prompt is a `window.prompt`.** Adequate for a single-user loopback tool; not a
  login flow.
* **The only delete affordance is for sessions** (Phase 10), behind a `window.confirm` and
  irreversible. Removing a tool, pruning artifacts or deleting a job is still a CLI action.
* **`md.js` is not a complete Markdown implementation.** It covers what the model actually
  writes; unusual constructs render as text.
* **Job logs are fetched with `tail=200`** from the inspector, not streamed.
