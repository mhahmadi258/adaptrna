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

### The framework tripwire has been tripped

Phase 9 set an explicit signal for when to stop hand-rolling this: *past roughly a thousand
lines, a second maintainer, or real client-side state, port to React.* It recorded 1,109
lines and argued the client stayed justified **because none of those lines were client-side
state**.

Phase 10 crossed both halves. The session rail brings the client to **1,329 lines of JS**,
and it introduces exactly the thing that argument leaned on: `state.sessions` (so filtering
is local rather than a round trip) and rail geometry persisted in `localStorage`.

The port has not been made, and the reasoning for holding is narrower than before: the new
state is two fields and a width, all of it derived or cosmetic, none of it a domain model
that can disagree with the server. That is a weaker argument than "there is none", and it is
recorded here rather than left implicit so the next feature to add state meets a stated
position instead of an assumption. The API is the contract, so switching still costs the
client alone.

| file | what it does |
|---|---|
| `index.html` | the shell: layout and the modal skeleton |
| `app.js` | wiring — the session rail, panels, job polling, the event dispatch for a turn |
| `sse.js` | SSE over `fetch`: a pure frame parser plus a thin transport |
| `api.js` | the endpoints as functions, and the bearer token when one is configured |
| `render.js` | messages, session rows, tool rows, job rows, the approval modal |
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
`[hidden]` rule.

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
