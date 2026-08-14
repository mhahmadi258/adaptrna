# Phase 11 — Activity bar, jobs rail, and the job log view

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8; the `ui/` layer in §2.
> Follows [PHASE_10_SESSION_RAIL_AND_TOOL_GATE.md](PHASE_10_SESSION_RAIL_AND_TOOL_GATE.md).
> **Definition of done:** a static icon bar on the far left switches the rail between
> Sessions and Jobs; selecting a job replaces the chat column with that run's live log;
> the right-hand panel no longer carries a Jobs list.
> Status: ✅ **done 2026-08-14** — all six gates passed; see §10 for what the plan got
> wrong and what the live run found.

---

## Contents

1. [Context and goal](#1-context-and-goal)
2. [Decisions this plan fixes](#2-decisions-this-plan-fixes)
3. [Surface](#3-surface)
4. [Components](#4-components)
5. [Documentation](#5-documentation)
6. [Tests](#6-tests)
7. [Implementation order](#7-implementation-order)
8. [Verification / definition of done](#8-verification--definition-of-done)
9. [Risks and notes](#9-risks-and-notes)
10. [What actually happened](#10-what-actually-happened)

---

## 1. Context and goal

Phase 10 gave sessions a real home: a resizable, filterable left rail. Training runs never
got the same treatment. They live in a 22 rem fixed column on the *right*, sharing it with
Tools and the Inspector, and the only way to read a run's log is a 200-line tail squeezed
into `#inspector-body` under a `max-height: 20rem` — for the single artefact a fine-tune
produces that a human actually watches for minutes at a time.

The asymmetry is the bug. A session and a run are the two long-lived things this platform
makes, and only one of them has a browsable list and a full-width reading surface.

This phase borrows VS Code's answer: a **static icon bar** on the far left that decides
what the adjacent rail lists, and a **middle column that shows either a conversation or a
run's log** depending on what is selected. Nothing about the backend changes — every
endpoint this needs already exists and already ships
([routers/jobs.py](../agentic/adaptrna_agentic/api/routers/jobs.py)).

**Out of scope.**

- **No route to start a job.** [jobs.py:3-4](../agentic/adaptrna_agentic/api/routers/jobs.py#L3)
  is explicit: starting a run happens through the chat, behind the approval gate, so the
  human sees the exact argv before GPU time is spent. The Jobs rail gets **no `＋ New`
  button** — deliberately, and the empty state says why.
- **No log streaming.** No SSE, no websocket, no `follow` endpoint. The client polls
  `GET /api/jobs/{id}/logs?tail=N`, which is what the panel already does, just on a timer.
- **No widening of the delete surface** ([PHASE_10 §1](PHASE_10_SESSION_RAIL_AND_TOOL_GATE.md)).
  Jobs stay CLI-only deletions; `cancel` is the only mutation this view offers.
- **No new backend code.** Zero Python changes outside `agentic/tests/`.
- No deep link to a job (`?job=…`). `?session=` stays the only one.

---

## 2. Decisions this plan fixes

| Decision | Choice | Rationale |
|---|---|---|
| **Icon bar height** | A fourth grid column *inside* `main.layout`, below the topbar | The full-height VS Code look needs `<body>` restructured into a flex row wrapping both the topbar and the grid. The topbar is one line high and carries the brand and the health badge; putting a nav bar beside them buys nothing and costs the whole layout skeleton, the `☰` toggle's position, and the mobile media query. |
| **Fate of the right-hand Jobs panel** | Removed; the rail is the only job list | Two lists of the same store means two render paths, two polls, and a guaranteed drift. The rail is resizable (10–30 rem), so job rows finally get room for their id, state, epoch/step and metric chips without ellipsis. The right panel keeps Tools + Inspector. |
| **How the middle column swaps** | `section.chat` and `section#joblog` are siblings inside a `div.center` wrapper; exactly one is `hidden` | Keeps every id the Playwright suite pins (`#chat-log`, `#composer-input`, `#composer-send`) and keeps the streaming code entirely unaware of the swap — a turn writes to `#chat-log` whether or not it is on screen. Explicit `grid-column` numbers on two overlapping children would break the moment a column is added. |
| **Composer while a log is open** | Hidden with the rest of the chat | It belongs to a session, not a run. A composer under a log invites the question "does this message go to the job?" — which has no good answer. Picking a session, or `×` on the log header, brings it back. |
| **Log freshness** | Poll `logs?tail=N` every 3 s **while the run is running**, stop on a terminal state after one final fetch | Same cadence and same reasoning as the existing job poll. `train.log` is a plain file written by a detached process ([runner.py:110-118](../agentic/adaptrna_agentic/jobs/runner.py#L110)); a tail read is cheap and a streaming endpoint would be new backend surface for a file that already sits on disk. |
| **Scroll behaviour** | Follow the tail only while the reader is already at the bottom | The reason to open a log is often to read something that scrolled past. Anchoring unconditionally to the bottom makes that impossible. Scrolling up turns follow off; scrolling back to the bottom turns it on. |
| **Jobs poll lifecycle** | One poll, always running: 3 s while any job is running, 15 s idle, paused while `document.visibilityState === "hidden"` | Today `refreshJobs` stops re-arming once nothing is running ([app.js:279-281](../ui/app.js#L279)), so a run started from chat never appears until something else forces a refresh. The always-on poll also feeds a running-count dot on the Jobs icon, which is the whole point of an activity bar. The per-job `status` call (the N+1) fires **only** when the Jobs rail is on screen, so the idle cost is one request per 15 s. |
| **Which view is open** | `localStorage`, key `adaptrna.rail.view` | Chrome for this browser, exactly like the rail's width and collapsed state ([app.js:21-27](../ui/app.js#L21)). The auth token's `sessionStorage` split is unchanged. |
| **Selected job** | Not persisted, not in the URL | A job id in `localStorage` outlives the run's output directory; restoring it on boot means booting into a 404. The view is restored, the selection is not. |
| **Clicking the already-active icon** | Collapses / expands the rail | VS Code's behaviour, and it is what the icon bar makes people expect. The `☰` topbar toggle keeps working unchanged. |
| **An approval arriving while a log is open** | Returns the middle column to chat, then shows the modal | The modal is `position: fixed` and would render fine over the log, but the gate's value is that the human sees *what led here*. Approving a command with the conversation hidden behind a log is the wrong default. |
| **No new JS module** | Wiring goes in `app.js`, DOM construction in `render.js` | The existing division of labour already has a slot for each half — `refreshTools`/`refreshJobs`/`refreshSessions` are all wiring in `app.js`, and `jobRow`/`sessionRow` are both already in `render.js`. A `ui/jobs.js` would need `state` and the error surface threaded through it. See §9 on the framework tripwire. |

---

## 3. Surface

### 3.1 Sessions selected — chat as it is today

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ ☰  AdaptRNA                                              install: ok           │
├───┬──────────────┬─────────────────────────────────────┬───────────────────────┤
│💬▌│ ＋ New       ┃ Chat                                │ Tools                 │
│   │ [filter…   ] ┃                                     │  splice_site    ●  ⋯  │
│⏱  │              ┃  you  fold this hairpin             │  vienna_fold    ○  ⋯  │
│ ● │ ▸ acceptor   ┃  ai   vienna_fold is disabled —     ├───────────────────────┤
│   │   2h ago  ⋯  ┃       shall I ask to enable it?     │ Inspector             │
│   │ ▸ dod9live   ┃                                     │                       │
│   │   yesterday  ┃  [ask something…              ] [↵] │                       │
└───┴──────────────┴─────────────────────────────────────┴───────────────────────┘
  ↑ activity bar     ↑ resizable       ↑ grip
     3rem, static    the ● under ⏱ is the running-run dot
```

### 3.2 Jobs selected, a run open — the log replaces the chat

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ ☰  AdaptRNA                                              install: ok           │
├───┬──────────────┬─────────────────────────────────────┬───────────────────────┤
│💬 │ [filter…   ] ┃ ● splice_simple_lora_20260813_101810 │ Tools                │
│   │              ┃   running · epoch 2 · step 340       │  splice_site    ●  ⋯ │
│⏱▌│ ▸ splice_…   ┃   train/loss 0.41  val/auroc 0.912   │  vienna_fold    ○  ⋯ │
│ ● │   ● running  ┃  [tail 200 ▾] [follow ✓] [cancel] [×]├──────────────────────┤
│   │   ep 2 · 340 ┠──────────────────────────────────────│ Inspector            │
│   │ ▸ acceptor_… ┃ Epoch 2:  38%|███▏      | 340/900    │                      │
│   │   ○ succeeded┃ val_loss=0.4113  val_auroc=0.9121    │                      │
│   │ ▸ crossover… ┃ …                                    │                      │
│   │   ○ failed   ┃                                      │                      │
└───┴──────────────┴─────────────────────────────────────┴───────────────────────┘
                     ↑ no composer — the middle column is the log
```

**Empty Jobs rail** reads: *"No training runs yet. Runs start from the chat, behind the
approval gate."* — the constraint from §1, stated where someone looks for the missing
button.

### 3.3 HTTP

```
GET  /api/jobs                       ← unchanged
GET  /api/jobs/{id}                  ← unchanged   (state + progress)
GET  /api/jobs/{id}/logs?tail=N      ← unchanged   (N ∈ {200, 500, 2000}; API caps at 2000)
GET  /api/jobs/{id}/analysis         ← unchanged
POST /api/jobs/{id}/cancel           ← unchanged
```

**No backend change.** `api.js` already wraps all five
([api.js:60-85](../ui/api.js#L60)); this phase adds no wrapper and no route.

---

## 4. Components

### 4.1 `ui/index.html`

| Change | Detail |
|---|---|
| New `<nav id="activity" class="activity">` | First child of `main.layout`. `role="tablist"`, `aria-label="Views"`. Two `<button role="tab">`: `#activity-sessions` (glyph `💬`, `aria-controls="rail"`, `aria-selected`) and `#activity-jobs` (glyph `⏱`, plus a `<span class="activity-dot" hidden>` for the running-run badge). Unicode glyphs only — the repo has no icon font and no SVG sprite, and `test_ui_serving.py` enforces that nothing is fetched from another host. |
| Rail head | `#rail-new` keeps its markup; JS sets `hidden` in Jobs view. `#rail-filter` keeps its id; JS rewrites `placeholder` and `aria-label` per view. `aside#rail`'s `aria-label` becomes JS-managed (`"Sessions"` / `"Jobs"`). |
| Middle column wrapped | `<div class="center">` wraps `<section id="chat" class="chat" aria-label="Chat">` (gains an id) and the new `<section id="joblog" class="joblog" aria-label="Job log" hidden>`. |
| `#joblog` skeleton | `<header id="joblog-head" class="joblog-head">` (populated by `render.js`) + `<pre id="joblog-body" class="joblog-body">`. |
| Right panel | The `<section class="panel">` holding `<h2>Jobs</h2>` and `#jobs-list` is **deleted**. `aside.side`'s `aria-label` becomes `"Tools"`. |

### 4.2 `ui/style.css`

| Change | Detail |
|---|---|
| `:root` | Add `--activity-w: 3rem;` |
| `.layout` (L101) | `grid-template-columns: var(--activity-w) var(--rail-w, 15rem) minmax(0, 1fr) 22rem;` |
| `.activity` | `display: flex; flex-direction: column; align-items: center; gap: .25rem; padding: .4rem 0; background: var(--panel); border-right: 1px solid var(--line);` — never collapses, never resizes. |
| `.activity-btn` | Square (`2.2rem`), transparent, large glyph. `[aria-selected="true"]` gets `box-shadow: inset 2px 0 0 var(--accent)` — the same current-item idiom as `.session-row.is-current` (L168). |
| `.activity-dot` | Small `var(--warn)` dot, `animation: pulse 1.4s infinite` — reuses the `@keyframes pulse` already defined at L333 for `.job-dot.is-running`. Needs its own `[hidden] { display: none }`. |
| `.center` | `display: flex; flex-direction: column; min-width: 0; min-height: 0;` |
| **`[hidden]` rules** | `.chat[hidden]`, `.joblog[hidden]`, `.activity-dot[hidden] { display: none; }` — the trap documented at [style.css:322-325](../ui/style.css#L322) and [ui/README.md:62-66](../ui/README.md#L62). An author `display` beats the UA's `[hidden]`; every one of these three gets a `display`. |
| `.joblog*` | `.joblog { flex: 1; display: flex; flex-direction: column; min-height: 0; }` · `.joblog-head { flex: none; padding: .75rem 1.25rem; border-bottom: 1px solid var(--line); background: var(--panel); }` · `.joblog-body { flex: 1; margin: 0; overflow: auto; padding: 1rem 1.25rem; font-family: var(--mono); font-size: .78rem; white-space: pre; }` — `white-space: pre`, not `pre-wrap`: training logs are progress bars and aligned columns, and wrapping them is what makes them unreadable. It scrolls in its own box, so the page never scrolls sideways. |
| `@media (max-width: 900px)` (L107-122) | `grid-template-columns: var(--activity-w) minmax(0, 1fr);` · `.activity { grid-row: 1 / -1; }` · `.center`, `.side { grid-column: 2; }` · **`.rail { left: var(--activity-w); }`** — today it is `left: 0`, which would sit on top of the icon bar. |

### 4.3 `ui/render.js`

| Function | Change |
|---|---|
| `jobRow(job, status, handlers, isCurrent)` (L187) | Gains `isCurrent`; the id becomes a `<button class="job-open">` calling `handlers.select(job.id)`, mirroring `sessionRow`'s `.session-open` (L102-126). The `log` button is **dropped** — selecting the row *is* opening the log. `analysis` and `cancel` stay as row actions. `is-current` styling reuses the `.session-row.is-current` rule via a shared `.is-current` selector. |
| `JOB_STATES` (L185) | Add `cancelled: "is-fail"`. Today a cancelled run renders a colourless dot because the map has no entry for it — a small pre-existing bug this phase is in a position to fix. |
| **New** `jobLogHead(job, status, handlers)` | Returns the header node: state dot + id + state tag, the progress line and metric chips (extract the `progress` block currently inlined in `jobRow` L199-219 into a shared `progressLines(progress)` so both callers render metrics identically), then the action row — a tail `<select>` (200 / 500 / 2000), a follow toggle, `analysis` (terminal states only), `cancel` (running only), and `×` close. |
| `analysisReport` (L238) | Unchanged; still rendered into `showInspector`. |

### 4.4 `ui/app.js`

**State** (L29-36) gains:

```js
view: "sessions",              // which rail view: "sessions" | "jobs"
job: null,                     // the job whose log fills the middle column, or null
jobs: [],                      // last fetched list, so switching views is instant
filter: { sessions: "", jobs: "" },   // was a bare string; per-view so a needle does not leak
logTimer: null,
logTail: 200,
logFollow: true,
```

Plus constants `JOB_IDLE_POLL_MS = 15000`, `LOG_TAIL_CHOICES = [200, 500, 2000]`,
`RAIL_VIEW_KEY = "adaptrna.rail.view"`.

| Function | Change |
|---|---|
| **New** `setView(next)` | Persists to `localStorage`, flips `aria-selected` on both activity buttons, sets `#rail`'s `aria-label`, toggles `#rail-new.hidden`, rewrites the filter placeholder and restores that view's needle, then calls `renderSessions()` or `renderJobsRail()`. Uses an **object map, not a `switch`** — see §9. |
| **New** `renderJobsRail()` | Pure render of `state.jobs` + `state.filter.jobs` into `#rail-list`, exactly parallel to `renderSessions` (L355-371). |
| `refreshJobs()` (L226-282) | Renders into `#rail-list` (via `renderJobsRail`) instead of `#jobs-list`; caches into `state.jobs`; fires the per-job `status` N+1 **only** when the Jobs view is on screen; always re-arms — 3 s if anything is running, `JOB_IDLE_POLL_MS` otherwise — and skips the fetch entirely while the document is hidden. Keeps the existing catch-and-re-arm for the retryable 409 ([app.js:232-239](../ui/app.js#L232)) verbatim; that comment is a scar and stays. Updates `#activity-jobs`'s dot from the running count. |
| **New** `openJob(id)` | Sets `state.job`, `state.logFollow = true`, hides `#chat`, shows `#joblog`, then `refreshJobLog()`. Also re-renders the rail so the row highlights. |
| **New** `closeJob()` | Clears `state.job` and `state.logTimer`, shows `#chat`, hides `#joblog`, re-renders the rail, focuses the composer. |
| **New** `refreshJobLog()` | `clearTimeout(state.logTimer)`, then `Promise.all([api.job(id), api.jobLogs(id, state.logTail)])`. Rebuilds `#joblog-head`; writes `#joblog-body.textContent`; if `state.logFollow`, scrolls to bottom. Re-arms at 3 s while the state is `running`. On a **404** stop and render the error in the head (the run's directory is gone); on any other failure keep the last good body, show the message in the head, and re-arm — the retryable-409 rule from [MASTER_PLAN §7](MASTER_PLAN.md). |
| **New** scroll handler on `#joblog-body` | `state.logFollow = body.scrollHeight - body.scrollTop - body.clientHeight < 40`, and the follow toggle mirrors it. |
| `sessionHandlers.select` (L383) | Also calls `closeJob()` — picking a conversation is the other way back to the chat. |
| `consume()`, `approval_required` branch (L125-129) | Calls `closeJob()` before `showApproval(data)`. |
| `refreshPanels()` (L284) | Unchanged in shape; `refreshJobs` now knows where to render. |
| `installRail()` grip drag (L473-487) | **Bug fix, required.** `setRailWidth(move.clientX / rootFontSize)` assumes the rail starts at viewport x = 0; with a 3 rem bar to its left every drag over-reports by 3 rem. Capture `const railLeft = $("rail").getBoundingClientRect().left` at `pointerdown` and use `(move.clientX - railLeft) / rootFontSize`. |
| **New** `installActivity()` | Wires both buttons; clicking the **already-selected** one toggles `rail-collapsed` instead of re-rendering. |
| `boot()` (L540-578) | Calls `installActivity()`; restores the view from `localStorage` **after** the first `refreshSessions`, so a boot into Jobs view still has sessions cached behind it. `visibilitychange` listener resumes `refreshJobs` / `refreshJobLog`. |

Projected size: `app.js` ≈ 580 → ~730, `render.js` ≈ 316 → ~380. Client total ≈ 1,470 lines.

---

## 5. Documentation

Per the repo convention, the exact claims that become false:

| File | Line / section | What changes |
|---|---|---|
| [documents/modules/web-ui.md](../documents/modules/web-ui.md) | §5 `app.js` state, **:107** | The `state` shape gains `view`, `job`, `jobs`, `logTimer`, `logTail`, `logFollow`; `filter` becomes per-view. |
| " | §5 **Job polling, :137** | Rewritten: two cadences, the visibility pause, the N+1 gated on the view, and the log follower as a second timer. |
| " | §7 **The session rail, :187** | Retitled *The activity bar and the rail*; documents the two views, the persisted key, and the collapse-on-reclick. |
| " | §8 `[hidden]` trap, **:218** | Add `.chat`, `.joblog`, `.activity-dot` to the list of elements that need their own `[hidden]` rule. |
| " | §2 file map :48 · §4 `api.js` :83 · §9 tests :246 · §10 limitations :263 | Descriptions of `app.js`/`render.js`; `jobLogs` gains a documented polling caller; the new browser tests; "the log is a polled tail, never a stream". |
| [documents/modules/api.md](../documents/modules/api.md) | §5 **Jobs, :180** | No route changes. Extend the "starting a job is not here" note to say the browser's Jobs view is read-only plus cancel, for the same reason. |
| [documents/modules/jobs.md](../documents/modules/jobs.md) | §9 limitations | The browser now tails `train.log` on a timer; there is still no streaming endpoint and `tail` is still capped at 2000. |
| [documents/architecture.md](../documents/architecture.md) | §8 front ends | The browser now displays two long-lived things — conversations and runs — and still owns no state store of its own. |
| [documents/project_structure.md](../documents/project_structure.md) | the `ui/` list | File *set* is unchanged; descriptions of `app.js` and `render.js` are not. |
| [documents/testing.md](../documents/testing.md) | `test_ui_browser.py`, `test_ui_contract.py` rows | The new assertions. |
| [ui/README.md](../ui/README.md) | file table :40-49 and **§"The framework tripwire has been tripped" :22-38** | Mandatory: that section says *"the next feature to add state meets a stated position instead of an assumption."* This is that feature. Update the line count and restate the position — see §9. |
| [README.md](../README.md), `docs/web-ui.png` | the screenshot | The current shot shows the old three-column layout with Jobs on the right. **Re-shoot it** (Gate 6) and add `docs/job-log.png`. |
| [plans/MASTER_PLAN.md](MASTER_PLAN.md) | §8 status table (:316 area), §"Status (2026-08-13)" :367 | Add the Phase 11 row; the "all ten phases landed" line becomes eleven. |

**Grep gate** — the old claims must survive only as history:

```bash
cd /home/mh/adaptrna
grep -rn "jobs-list" ui/ documents/ agentic/tests/     # expect: no hits
grep -rn "Tools and jobs" ui/ documents/               # expect: no hits
# `index.html` keeps a static aria-label as the pre-JS initial value; `setView` owns it
# from boot onward. Expect exactly one hit, on the `aside#rail` element.
grep -rn 'aria-label="Sessions"' ui/index.html
grep -rn "1,329 lines" ui/README.md documents/         # expect: no hits (count updated)
grep -rn "activity" ui/index.html ui/style.css ui/app.js | head   # expect: the new bar
```

---

## 6. Tests

| File | Asserts |
|---|---|
| `agentic/tests/test_ui_contract.py` | **New** `test_the_job_log_payload_fields` — `GET /api/jobs/{id}/logs?tail=200` returns `job_id`, `tail`, `log`; the view reads `body.log` by name. |
| " | **New** `test_every_tail_the_client_offers_is_one_the_api_accepts` — regex `LOG_TAIL_CHOICES = \[([\d, ]+)\]` out of `app.js`, then `GET …/logs?tail=N` for each and assert `200`. Pins the client against the `Query(ge=1, le=2000)` cap in [jobs.py:34](../agentic/adaptrna_agentic/api/routers/jobs.py#L34) — the same "missing compiler" idea as the rest of that file. |
| " | **New** `test_the_job_row_reads_the_states_the_store_can_hold` — assert `render.js` names every member of `store.TERMINAL_STATES + RUNNING_STATES`. Catches the `cancelled` gap directly. |
| " | ⚠️ `test_every_streamed_event_is_one_the_client_handles` (:72-80) greps `app.js` for `case "(\w+):"` and asserts **set equality** with the six SSE events. A `switch (view)` in `app.js` breaks it. The view dispatch must be an object map or `if/else`. Unchanged test, hard constraint. |
| `agentic/tests/test_ui_browser.py` (`-m ui`) | **New** `test_the_activity_bar_switches_the_rail_between_sessions_and_jobs` — click `#activity-jobs`, `#rail-list` holds `.job-row` not `.session-row`; `#rail-new` hidden. |
| " | **New** `test_selecting_a_job_replaces_the_chat_with_its_log` — `#chat` hidden, `#joblog` visible, `#composer-input` not visible, `#joblog-body` contains a known line from the fixture's `train.log`. |
| " | **New** `test_closing_the_job_log_returns_to_the_chat` — via `×` **and** via clicking a session row. |
| " | **New** `test_the_open_view_survives_a_reload` — mirrors the existing rail-width localStorage test (:357-376). |
| " | **New** `test_the_grip_still_resizes_from_the_rails_own_left_edge` — regression guard for the `clientX` fix in §4.4; drag to a known x and assert the width equals `x − activity-bar width`. |
| " | **Update** the three tests that pin `#jobs-list` (:169, :196-197, :263) to click `#activity-jobs` and read `#rail-list`. |
| `agentic/tests/test_ui_serving.py` | Unchanged — no new asset file, and the external-host scan covers the edited CSS/JS automatically. |

Everything else in the 405-test suite must stay green; no Python outside `tests/` is touched.

---

## 7. Implementation order

1. **Register the phase** — add the §8 row in `MASTER_PLAN.md` and flip this file's status line.
2. **Layout skeleton** — `index.html` (`nav#activity`, `div.center`, `section#joblog`, delete the right-hand Jobs panel) + `style.css` (`--activity-w`, the four-column grid, `.activity*`, `.center`, `.joblog*`, the three `[hidden]` rules, the media-query fixes). Load `/` and confirm the bar renders and the chat still works before any JS.
3. **Grip fix** — `installRail()`'s `clientX` offset. Do it now, while the cause is visible; it is a two-line change that a later drag would otherwise silently mis-measure.
4. **`render.js`** — `progressLines()` extraction, `jobRow` gains `select`/`isCurrent` and loses `log`, `JOB_STATES.cancelled`, new `jobLogHead`.
5. **View switching** — `state.view`/`state.filter` reshape, `setView`, `installActivity`, `renderJobsRail`, `refreshJobs` retargeted to `#rail-list`, the two cadences, the visibility pause, the activity dot.
6. **The log view** — `openJob` / `closeJob` / `refreshJobLog`, follow-on-scroll, tail select, `analysis` / `cancel` wiring, the `closeJob()` calls in `sessionHandlers.select` and the `approval_required` branch.
7. **Tests** — contract tests first (they run without a browser), then the Playwright additions and the `#jobs-list` updates.
8. **Docs + screenshots** — §5, then the grep gate.

---

## 8. Verification / definition of done

**Gate 1 — the suite is green.**
```bash
cd /home/mh/adaptrna/agentic
../.venv/bin/python -m pytest -q          # expect: 405+ passed, 0 failed
../.venv/bin/python -m pytest tests/test_ui_serving.py tests/test_ui_contract.py -q
```

**Gate 2 — the browser suite is green.**
```bash
cd /home/mh/adaptrna/agentic
../.venv/bin/python -m pytest -m ui -q    # expect: 15 existing + 5 new, 0 failed
```

**Gate 3 — the bar switches, by hand.**
```bash
cd /home/mh/adaptrna && python -m adaptrna_agentic.cli.serve --open
```
Click `⏱`: the rail lists the seven runs under `outputs/`, `＋ New session` is gone, the
filter says "Filter jobs…". Click `💬`: sessions are back, and the needle typed in Jobs
view did not follow. Click the already-selected icon: the rail collapses; click again: it
returns. Reload: the last view is still open.

**Gate 4 — a real log, live.** Start a run from chat (approve the gate), open it from the
Jobs rail. The middle column is the log, the composer is gone, the header shows
`running · epoch N · step N` with metric chips, and both the header and the tail advance
without a reload. Scroll up mid-run: it stops following. Scroll to the bottom: it resumes.
Let the run finish: the dot goes green, polling stops, `analysis` appears, `cancel` does not.

**Gate 5 — the rail still resizes correctly.** Drag the grip to roughly a third of the
window and confirm the rail's right edge lands under the pointer, not 3 rem past it.
Collapse and reload: the width survives, and the activity bar never collapses.

**Gate 6 — the docs match.** The §5 grep gate returns no hits for the stale claims;
`docs/web-ui.png` is re-shot on the new layout and `docs/job-log.png` exists.

---

## 9. Risks and notes

**The framework tripwire, again.** [ui/README.md:22-38](../ui/README.md#L22) records that
Phase 10 crossed both halves of Phase 9's signal (1,329 lines, real client state) and held
the port anyway, on the narrow ground that the new state was *"two fields and a width, all
of it derived or cosmetic, none of it a domain model that can disagree with the server."*

This phase must answer the same question in writing. The recommendation is **hold again**,
and the ground is the same one, still standing: `view`, `job` and `logFollow` are *which
pane is on screen* — pure chrome, unable to disagree with the server — and `jobs` is a
render cache of a list the server owns, exactly like `sessions`. What has changed is size:
~1,470 lines, and `app.js` alone near 730.

So the tripwire should be **restated in terms that can actually fire**, not left as a line
count that gets renegotiated each phase. Proposal, to be written into `ui/README.md`:

> Port when the client first holds state the server does not — an optimistic update, a
> local edit buffer, an undo stack — or when a third mode joins chat and job-log in the
> middle column. Until then, extract rather than port: the next module out is the job-log
> controller (`ui/jobs.js`), which is already self-contained.

**Two lists become one, and one of them was load-bearing for tests.** Three Playwright
tests read `#jobs-list`. Deleting that node without updating them turns a deliberate
removal into three confusing failures; §7 step 7 sequences the update with the deletion.

**`case "…":` set equality.** The contract test that pins the SSE dispatch reads *all*
`case "…"` strings in `app.js`. A `switch (state.view)` would silently add `"sessions"`
and `"jobs"` to that set and fail a test about streaming events. Object map or `if/else`.

**The `[hidden]` trap is the single most likely bug in this phase.** Three new elements
get an author `display` *and* are hidden from JS. `.thinking` cost Phase 9 an entire cycle
of a permanently animating indicator for exactly this reason. The rules are listed in
§4.2; a browser test asserting `#composer-input` is *not visible* in job view is what
catches it if they are missed.

**A turn streaming behind a hidden chat.** `consume()` keeps appending to `#chat-log` and
toggling `#composer-*` while the log is on screen. That is intended — the conversation is
whole when you come back — but `setBusy(false)` calls `$("composer-input").focus()`
(app.js:56) on a hidden input, which is a no-op in every browser and not worth guarding.
The one case that *is* worth handling is the approval gate, and §2 handles it by returning
to chat first.

**Polling cost.** Idle is one `GET /api/jobs` per 15 s per open tab, paused when hidden.
With the concurrency guard in [runner.py:91-99](../agentic/adaptrna_agentic/jobs/runner.py#L91)
allowing at most one running job, the N+1 status call is at most one extra request, and
only while the Jobs rail is on screen.

**Long job ids.** `splice_simple_lora_20260813_101810` is 34 characters; at the 10 rem
minimum rail width it ellipsizes. `.panel-name` already handles this and the row carries a
`title`, so the full id is one hover away — and the log header shows it in full.

---

## 10. What actually happened

**Gates.** 408 deterministic tests (from 405) and 21 opt-in browser tests (from 15). Zero
Python changed outside `tests/`, as planned.

### The three tests that failed first were the interesting part

| Test | Why it failed | What it meant |
|---|---|---|
| `test_the_monitor_keeps_polling_after_a_transient_conflict` | Reached the empty state after **two** calls, not three | Real regression in the *test*, caused by a real change in the code: the jobs rail now paints "No training runs yet" from an empty cache the moment you switch views, so arriving at the empty state stopped being evidence that a poll had succeeded. The 409 test had been silently weakened. Rewritten so the third response carries a job that can only have come over the wire. |
| `test_selecting_a_job_replaces_the_chat_with_its_log` | `#joblog-head` was `''` | Not just a timing bug in the test. `openJob` unhid the pane and *then* awaited two fetches, so the log pane was genuinely blank — no id, no state, nothing — for a full round trip. Fixed in `app.js`: paint the header from the rail's cached record first, then let the poll replace it. |
| `test_closing_the_job_log_returns_to_the_chat` | Timed out waiting for `#joblog[hidden]` | Ordinary test bug: Playwright's default `wait_for_selector` state is *visible*, which a hidden element never reaches. Needs `state="hidden"`. |

The first two are the pattern this project keeps hitting: a UI change that looks additive
quietly invalidates an assertion elsewhere, and the test that breaks is the one worth reading
rather than the one worth silencing.

### What the plan got wrong

* **The grep gate was wrong about `aria-label="Sessions"`.** It demanded no hits in
  `index.html`, but the rail needs a sensible label *before* `setView` runs. The static
  attribute is the correct initial value and JS overwrites it per view; the gate expectation
  was deleted, not the attribute.
* **Line estimates were low by ~12%.** Projected `app.js` ≈ 730 / `render.js` ≈ 380 / client
  ≈ 1,470; actual **821 / 394 / 1,648**. The gap is almost entirely error handling and the
  header rebuild, both of which the plan described in one line each.
* **`state.jobStatus` was not in the plan's state list.** The plan said `jobs` cached the
  list, forgetting that progress comes from a *separate* per-job call and therefore needs its
  own map. It also needs invalidating: a finished run's cached status still said `running`,
  and `jobRow` prefers `status.state` over `job.state`, so a completed run would have shown
  as running forever. Three lines, and nothing would have caught it but eyes.
* **`.center` needed `.chat { flex: 1 }` as well.** The plan listed the `[hidden]` rules
  (correctly — they were the predicted trap and cost nothing because of it) but not the fact
  that `.chat` becomes a flex *child* as well as a flex container.

### What the live run found

Against the real install, real registry and the six real runs under `outputs/`:

* The rail listed **6 runs**; opening `splice_simple_lora_20260813_101810` put its real
  `train.log` in the centre column and its final metrics in the header — `test/f1_score
  0.968599`, `test/acc 0.9675`, `epoch 4 · step 200`.
* `#chat` and `#composer-input` both correctly not visible; **zero JS errors** across the
  whole session.
* Geometry, checked rather than eyeballed: activity bar at `x=0 w=48` full height, rail at
  `x=48`, **no horizontal overflow**; collapsed rail keeps the bar at `w=48`; at a 700 px
  viewport the fixed rail sits at `x=96` — clear of the bar, which was the specific thing the
  media-query change existed to prevent.
* The grip fix is real and guarded: the drag now lands the rail's edge under the pointer,
  which it did not with the bar in place.

### Notes for whoever is next

* **The tripwire is no longer a line count.** `ui/README.md` now says: port when the client
  holds state the server does not, or when a *third* mode joins chat and job-log in the
  centre column — and until then extract rather than port, starting with `ui/jobs.js`. A
  third mode is the likely trigger, and it is close: an artifacts view or a diff view would
  do it.
* **The two blank-until-loaded moments are now one.** `openJob` paints from cache; the log
  *body* still arrives a round trip late. If that ever matters, the rail could cache the last
  tail alongside the status.
* **`cancelled` had never been styled** since Phase 9 — it drew a colourless dot,
  indistinguishable from a run that had not started. `test_the_job_row_names_every_state_the_store_can_hold`
  now reads the state tuples straight out of `jobs/store.py`, so the next state added to the
  store fails in `pytest` rather than rendering as nothing.
