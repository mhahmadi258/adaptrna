# ui/

The AdaptRNA web client: streamed chat, a tool dashboard, a live training monitor, and the
approval gate as a dialog. Served by the Phase 8 API — open `/` on a running server.

```bash
python -m adaptrna_agentic.cli.serve --open
```

## No build step

Plain ES modules and one stylesheet, loaded straight by the browser. There is no
`package.json`, no bundler and no `node_modules`: the UI ships with
`pip install -e ./agentic` and works offline, because nothing here is fetched from a CDN.

That was a deliberate reversal of the master plan's React default. This client is
overwhelmingly a *renderer of server state* — conversations live in the checkpointer, tools
in the manifest, jobs in the job store — so there was no client-side domain model to justify
a second dependency ecosystem in a Python repo.
[plans/PHASE_9_WEB_UI.md](../plans/PHASE_9_WEB_UI.md) §2 has the full comparison.

### The framework tripwire, restated

Phase 9 set the signal as a line count: *past roughly a thousand lines, a second maintainer,
or real client-side state, port to React.* It recorded 1,109 lines and held, because none of
them were client-side state. Phase 10 recorded 1,329 and held again, on the narrower ground
that `state.sessions` and a persisted rail width are derived or cosmetic. Phase 11 brings the
client to **1,648 lines**.

A threshold renegotiated every phase is not a threshold. So the count is retired, and the
condition is now something that can actually fire:

> **Port** when the client first holds state the server does not — an optimistic update, a
> local edit buffer, an undo stack — or when a third mode joins chat and job-log in the
> centre column.
> **Until then, extract rather than port.** The next module out is the job-log controller
> (`ui/jobs.js`), which is already self-contained.

Neither condition has fired. `state.view`, `state.job` and `state.logFollow` are *which pane
is on screen*; `state.jobs` and `state.jobStatus` are a render cache of a list the server
owns, exactly like `state.sessions`. Nothing here can disagree with the server. The API is
the contract, so switching still costs the client alone.

| file | what it does |
|---|---|
| `index.html` | the shell: the activity bar, the rail, the centre column, the modal skeleton |
| `app.js` | wiring — the activity bar, both rail views, job and log polling, the event dispatch for a turn |
| `sse.js` | SSE over `fetch`: a pure frame parser plus a thin transport |
| `api.js` | the endpoints as functions, and the bearer token when one is configured |
| `render.js` | messages, session rows, tool rows, job rows, the job-log header, the approval modal |
| `md.js` | a small Markdown renderer — the model writes tables, and raw pipes are unreadable |
| `dom.js` | the shared `el()` helper; nothing in the client assigns `innerHTML` |
| `style.css` | one stylesheet, light and dark |

## Three things worth knowing before editing

**`EventSource` cannot POST.** A chat turn is a POST with a JSON body, so `sse.js` reads
the response stream and parses frames itself. That is the only real logic in the client,
which is why the parser is a pure function of text — `createFrameParser()` is fed arbitrary
chunks and returns whole frames.

**The approval round trip is three steps.** The stream ends on `approval_required`; the
modal collects a decision; `/resume` returns a *new* stream continuing the same turn. There
is no separate resume path in `app.js` — `consume()` is just called again.

**An author-origin `display` beats `[hidden]`.** The thinking dots animated permanently for
the whole of Phase 9: `app.js` set `.hidden` correctly, but `.thinking { display: flex }` in
the stylesheet overrode the user agent's `[hidden] { display: none }` regardless of
specificity. Anything this stylesheet gives a `display` to and JS hides needs its own
`[hidden]` rule — there are five now, and the centre-column swap (`.chat` / `.joblog`) added
two of them.

**The centre column has two modes and one composer.** `#chat` and `#joblog` are siblings, and
opening a run's log hides the chat rather than tearing it down — a turn keeps streaming into
`#chat-log` while you read a log, and is whole when you come back. The composer goes with the
chat, because it belongs to a session and not to a run.

## Tests

```bash
cd ../agentic
../.venv/bin/python -m pytest tests/test_ui_serving.py tests/test_ui_contract.py

../.venv/bin/python -m pip install playwright && ../.venv/bin/python -m playwright install chromium
../.venv/bin/python -m pytest -m ui        # opt-in: drives a real browser
```

`test_ui_contract.py` is the compiler this pair of languages does not have: it pins every
event name and response field that `ui/*.js` reads, so a server-side rename fails in
`pytest` rather than silently blanking a panel.
