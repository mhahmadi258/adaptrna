# Phase 9 — Web UI (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 9; the `ui/` layer in §2.
> **Definition of done:** the Phase-5 scenario driven end-to-end from the browser.
> Status: ✅ **done 2026-08-13** — all four gates passed; see §10 for what the plan got
> wrong and what the live run found.

---

## 1. Context and goal

The last phase, and the smallest in new logic: Phase 8 already exposes everything over
HTTP, streams turns as SSE, and carries the approval payload — including the exact command
line — in a form a browser can render. Phase 9 writes the client.

That framing matters for scoping. The server holds *all* the state: conversations in the
checkpointer, tools in the manifest, jobs in the job store. The browser is a renderer of
server state plus an SSE consumer. There is no client-side domain model to build, no
optimistic updates to reconcile, no offline story.

**Out of scope:** anything that would add capability the CLI and API do not have. In
particular no `prune`/delete surface — Phase 7 kept deletion a human action at the CLI and
Phase 8 declined to weaken that; a button is not a better boundary than a shell prompt.

## 2. Decisions this plan fixes (including the last open §10 item)

### The UI stack: **vanilla ES modules served by FastAPI — no build step**

The master plan's stated default was React + TypeScript, with Gradio as the "quick path".
I am recommending neither, and the reasoning is worth stating because it reverses a
default:

| | Fit here |
|---|---|
| **React + TS + Vite** | Node 20 *is* on this machine, so it is feasible. But it puts a second ecosystem into a repo that is 164 Python files and zero `package.json`: `npm install`, a build step, ~200 MB of `node_modules`, a dev server needing a proxy, and two dependency trees to keep current — for four views whose entire job is rendering server state |
| **Gradio** | Pure Python and no build, but it wants to *own* the app loop. Consuming an external SSE API from inside Gradio is awkward, and the approval gate and live job monitor would end up re-implemented Gradio-side — a third front end with its own logic, exactly the drift Phase 8 avoided by wrapping the CLI rather than reimplementing it |
| **Vanilla ES modules + CSS, served by the existing app** | No toolchain, no build, no second language ecosystem. `pip install -e ./agentic && serve` gives you the UI. Same-origin, so no CORS and no dev-server proxy. The whole client is a few hundred readable lines a researcher can modify without learning the project's build system |

**Tripwire for revisiting:** if the client grows past roughly a thousand lines, gains a
second contributor, or starts needing real client-side state, port it to React. The Phase 8
API is the contract, so that is a rewrite of the client alone — nothing server-side moves.

### The rest

| Decision | Choice | Rationale |
|---|---|---|
| **Serving** | The FastAPI app mounts `ui/` and serves `index.html` at `/` | One process, one port, same origin. `serve --open` launches a browser |
| **Auth in the browser** | Same-origin means no token on loopback. If `ADAPTRNA_API_TOKEN` is set, the UI asks once and keeps it in `sessionStorage`, sending `Authorization` on every request | Matches Phase 8's model; `sessionStorage` (not `localStorage`) so it dies with the tab |
| **Approval dialogs show the *exact* command** | The modal renders `details.command` verbatim, plus file lists and warnings — never a summary | The entire value of the gate is that the human sees what will actually run. A UI that paraphrased it would quietly undo Phases 5–8 |
| **Refresh safety** | On load the UI calls `GET /history`, which returns messages **and** `pending_approval`, and restores the modal if one is open | A browser refresh mid-approval must not strand a suspended turn. The API already supports this; the UI has to use it |
| **Job monitoring** | Poll `GET /api/jobs` every 3 s while any job is running, stop when none is | A training run is minutes-to-hours; polling is the right amount of machinery, and the API is already cheap to call |
| **Browser tests** | Contract tests always; Playwright marked `ui` and opt-in | Same split as the engine's `gpu`/`weights` markers. Playwright pulls ~150 MB of browser, which should not be a prerequisite for `pytest` |

## 3. What the UI shows

```
┌──────────────────────────────────────────────────────────────────┐
│ AdaptRNA              install: ok            session: [work  ▾]  │
├────────────────────────────────────┬─────────────────────────────┤
│ Chat                               │ Tools                       │
│                                    │  splice_site      ●  test   │
│  you  Is this a donor site? …      │  splice_simple    ●  test   │
│  ⚙   splice_site({sequences:…})    │  vienna_fold      ○  test   │
│  =   [0.9998]                      │                             │
│  ai  The model gives 0.9998 …      ├─────────────────────────────┤
│                                    │ Jobs                        │
│                                    │  acceptor_lora  running     │
│                                    │    epoch 1 · step 99        │
│                                    │    val/f1 0.923      [log]  │
│  [ask something…            ] [↵]  │  splice_simple  succeeded   │
└────────────────────────────────────┴─────────────────────────────┘

        ┌─ Approval required ────────────────────────────┐
        │ Train splice_site (lora) — ETA ~7 min          │
        │ would run:                                     │
        │   …/python -m adaptrna_agentic.jobs.train_…    │
        │   --task splice_site --use_lora --set …        │
        │ ! Cross-species benchmark: test is a different │
        │   organism                                     │
        │                        [ Decline ]  [ Approve ]│
        └────────────────────────────────────────────────┘
```

Four views, one modal:

1. **Chat** — streamed assistant text; tool calls and results rendered inline, results
   collapsed by default (they can be a manifest entry or a matrix).
2. **Tools** — every tool with its state; toggle active/disabled; run its smoke/golden
   test and show the report.
3. **Jobs** — list with live progress from `/api/jobs`; expand for the log tail and, when
   finished, the analysis verdict.
4. **Approval modal** — blocking, verbatim command, approve/decline with an optional note.

Plus a header showing `install:` from `/health`, and a session picker backed by
`GET /api/sessions` (so a conversation started in the terminal appears in the dropdown).

## 4. The non-obvious parts

**`EventSource` cannot POST.** The browser's SSE API is GET-only, and our chat endpoint
takes a JSON body. So the client consumes the stream with `fetch()` + a `ReadableStream`
reader and parses SSE frames itself (split on `\n\n`, read `event:` / `data:`). This is
the single piece of real client logic in the phase, so it lives in one module
(`sse.js`) with the parser separated from the transport for testability.

**The approval round trip has three steps, not two.** Stream ends on `approval_required`
→ modal → `POST /resume` returns a *new* stream that must be consumed the same way. The
client needs one function that consumes a stream, called twice, rather than a special
"resume" path.

**A tool result can be large.** The API already truncates to 2000 chars for display; the
UI collapses them further and does not attempt to pretty-print matrices.

**Static files without a new dependency.** Starlette's `StaticFiles` may want `aiofiles`
(not currently installed). If it does, three `FileResponse` routes serve the three assets
instead — the UI is small enough that this is not a compromise.

## 5. Components

```
ui/
├── index.html          layout + the modal skeleton
├── app.js              wiring: panels, session picker, polling, event dispatch
├── sse.js              fetch-based SSE: parseFrames() + streamTurn()
├── api.js              thin typed-ish wrapper over the Phase 8 endpoints + auth header
├── render.js           message / tool / job / approval rendering
└── style.css           one small stylesheet; no framework, no CDN
agentic/adaptrna_agentic/api/routers/ui.py    mounts ui/ and serves index.html at /
agentic/adaptrna_agentic/cli/serve.py         + --open to launch a browser
```

No bundler, no transpile: the browser loads `app.js` as `<script type="module">` and it
imports the rest. Nothing is fetched from a CDN, so the UI works offline.

## 6. Tests

| test file | asserts |
|---|---|
| `test_ui_serving.py` | `/` returns the app shell; `app.js`, `sse.js`, `style.css` are served with sensible content types; assets reference no external origin (offline-safe); the shell loads `app.js` as a module |
| `test_ui_contract.py` | **the API surface the UI depends on, pinned**: the SSE event names (`text`, `tool_call`, `tool_result`, `approval_required`, `done`, `error`), the fields the modal needs (`requests[].summary`, `.details.command`), `/history` returning `pending_approval`, and the job-status shape the monitor renders. A server change that would break the UI fails here rather than in a browser |
| `test_ui_browser.py` (`-m ui`, opt-in) | Playwright headless: send a message and see streamed text; toggle a tool and see the badge change; trigger an approval and see the modal with the exact command; decline → no job; approve → job appears in the monitor; refresh mid-approval → the modal is still there |

~20 deterministic tests (total ~368) plus the opt-in browser suite. Phases 0–8's 348 stay
green; engine's 135 untouched.

## 7. Implementation order

1. `routers/ui.py` + the shell (`index.html`, `style.css`) + `test_ui_serving.py`.
2. `api.js` + `sse.js` + `render.js` + the chat panel; `test_ui_contract.py`.
3. Tools panel and jobs monitor.
4. Approval modal + refresh restoration.
5. `serve --open`; Playwright suite (opt-in) + the DoD run.
6. README section; close-out: MASTER_PLAN §8 tick and the §10 UI-stack row marked decided.

## 8. Verification / definition of done

**Gate 1 — deterministic:** `cd agentic && pytest` green (348 + ~20); `cd engine && pytest`
→ 135.

**Gate 2 — the Phase-5 scenario, entirely in the browser.** Playwright headless against
the live install, driving the same flow Phase 5 proved in the terminal:

1. open `http://127.0.0.1:8077/`, pick a new session;
2. ask it to profile `dod_data/` and recommend a config for `splice_simple`;
3. the **approval modal** appears showing the exact command — screenshot it, assert no job
   exists;
4. approve; the job appears in the monitor and its progress updates live;
5. when it finishes, ask for the analysis and see the verdict (with the truncated-run
   caveat if it was a quick run);
6. ask it to score a real 400 nt window; the prediction renders inline.

If Playwright is unavailable, the same walkthrough is performed manually and recorded in
the README — but the automated version is the goal, because "it worked when I clicked it"
is not a regression test.

**Gate 3 — the two front ends really are one system:** a session created in the browser
appears in `chat --list-sessions` and continues correctly from the terminal, and vice
versa. (Phase 8 proved the API half; this proves the UI uses it properly.)

**Gate 4 — the gate is honest:** the command shown in the modal is byte-identical to the
one the terminal prints for the same request. A screenshot goes in the README.

## 9. Risks and notes

- **The modal is the safety boundary made visible.** If it ever summarises instead of
  showing the command, the human is approving something they cannot see. Gate 4 exists to
  keep that true.
- **No build step is a decision with a shelf life.** It is right for four views and one
  maintainer; the tripwire in §2 says when to change course, and the API contract means
  changing course costs only the client.
- **Polling, not websockets.** Simpler, and the job endpoint is cheap. If a future UI needs
  push, the SSE plumbing already exists to carry it.
- **Playwright is ~150 MB.** Opt-in, marked, never a prerequisite for the default suite —
  the same treatment the engine gives GPU tests.
- **This is still a local, single-user tool.** The UI inherits Phase 8's posture: loopback
  by default, token only if configured, no delete surface. Nothing here makes it
  multi-tenant, and it should not be exposed as if it were.

---

## 10. Outcome (2026-08-13)

**381 deterministic agentic tests** (348 before + 33), **11 opt-in browser tests**, engine
untouched at 135. All four gates passed against the real install in Chromium.

| Gate | Result |
|---|---|
| 1 — deterministic | agentic 381 ✅, engine 135 ✅ |
| 2 — the Phase-5 scenario in a browser | profiled `dod_data/` → validated config → gate → approved → **monitor updated 9 times in place** (epoch 0/step 49 → epoch 4/step 200; train/loss 0.390→0.029, val/f1 0.923→0.970) → analysis `ok` + truncated caveat (test/f1 0.9686) → real 400 nt windows scored **0.9998** and **0.0047**. Zero JS errors throughout |
| 3 — one system | browser session recalled verbatim from `chat --once`; the terminal's turn appeared in the browser on reload; both directions in the picker |
| 4 — the gate is honest | the modal's command compared against `cli/chat.py::_prompt_approval` rendering the *same* suspended request: **621 chars, byte-identical** |

### What the plan got wrong

- **Markdown was not in scope, and should have been.** §3 said "streamed assistant text";
  the DoD screenshots showed the orchestrator answers in Markdown and reaches for a table
  whenever it reports metrics, so plain text rendered the primary reading surface as raw
  pipes. Added `md.js` — a ~150-line renderer covering what the model actually writes,
  building DOM nodes rather than parsing HTML, so it cannot inject markup. `dom.js` was
  split out so it and `render.js` need not import each other.
- **`aiofiles` was a non-issue.** §4 hedged about `StaticFiles`; it works without it.

### What the live run found that no test would have

- **A transient 409 froze the job monitor permanently.** Phase 7's optimistic concurrency
  answers 409 while the job store is mid-write — likeliest *just after a run starts*,
  exactly when someone is watching. `refreshJobs` returned without re-arming its timer, so
  the panel stopped updating for good. The live DoD script crashed on the same 409, which
  is how it surfaced. Fixed, recorded as a platform constraint in MASTER_PLAN §7, and
  covered by a browser test that fails without the fix.
- **`failed_checks` is a list, and `[] || 0` is truthy in JS**, so a healthy install
  rendered as "install: ok ( failed)". Found by looking at the real `/health`, not a
  fixture. The contract test now pins the type.

Both are the same lesson: the contract tests pin *names*, and these were bugs about
*values* and *timing*. That is the seam the opt-in browser suite and a live run cover.

### The tripwire, honestly

The client landed at **1,109 lines of JavaScript** (1,610 with CSS and HTML) — right at the
"roughly a thousand lines" threshold §2 set for reaching for React. Where it actually
stands:

| | lines |
|---|---|
| app logic — `app.js`, `render.js`, `api.js`, `sse.js` | 895 |
| `md.js` — a self-contained Markdown renderer | 180 |
| `dom.js` — the `el()` helper | 34 |

The threshold was a proxy for "is this complex enough that a framework would help", and
the answer is still no: there is no client-side state to speak of, `md.js` is a leaf
nobody else touches, and `app.js` is mostly a switch over six event names. But it is at
the line, not comfortably under it. **The next feature that adds real client-side state
should re-open §2 rather than assume the answer.**
