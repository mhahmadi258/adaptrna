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

That was a deliberate reversal of the master plan's React default. This client is a
*renderer of server state* — conversations live in the checkpointer, tools in the manifest,
jobs in the job store — so there is no client-side domain model to justify a second
dependency ecosystem in a Python repo. [plans/PHASE_9_WEB_UI.md](../plans/PHASE_9_WEB_UI.md)
§2 has the full comparison, and §10 the honest accounting: at 1,109 lines of JS this sits
*at* the threshold that plan set for reaching for a framework, not comfortably under it.
It stays justified because none of those lines are client-side state — but the next feature
that adds some should re-open the question rather than assume. The API is the contract, so
switching costs the client alone.

| file | what it does |
|---|---|
| `index.html` | the shell: layout and the modal skeleton |
| `app.js` | wiring — panels, session picker, job polling, the event dispatch for a turn |
| `sse.js` | SSE over `fetch`: a pure frame parser plus a thin transport |
| `api.js` | the endpoints as functions, and the bearer token when one is configured |
| `render.js` | messages, tool rows, job rows, the approval modal |
| `md.js` | a small Markdown renderer — the model writes tables, and raw pipes are unreadable |
| `dom.js` | the shared `el()` helper; nothing in the client assigns `innerHTML` |
| `style.css` | one stylesheet, light and dark |

## Two things worth knowing before editing

**`EventSource` cannot POST.** A chat turn is a POST with a JSON body, so `sse.js` reads
the response stream and parses frames itself. That is the only real logic in the client,
which is why the parser is a pure function of text — `createFrameParser()` is fed arbitrary
chunks and returns whole frames.

**The approval round trip is three steps.** The stream ends on `approval_required`; the
modal collects a decision; `/resume` returns a *new* stream continuing the same turn. There
is no separate resume path in `app.js` — `consume()` is just called again.

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
