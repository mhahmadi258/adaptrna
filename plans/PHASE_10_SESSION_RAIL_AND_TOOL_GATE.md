# Phase 10 — Session rail, honest thinking indicator, user-owned tool state

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8; the `ui/` layer in §2; the approval gate in
> §3.2. Follows [PHASE_9_WEB_UI.md](PHASE_9_WEB_UI.md).
> **Definition of done:** a disabled tool cannot be re-enabled without the user's approval
> in either front end; the thinking indicator is dark at rest; sessions are created,
> renamed, deleted and switched from a resizable, collapsible left rail.
> Status: ✅ **done 2026-08-13** — all six gates passed; see §10 for what the plan got
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

Phases 0–9 built the platform and put a browser in front of it. This phase is the first
driven entirely by *using* it. Three defects and one gap came out of a live session:

1. **The assistant enables tools by itself.** With `vienna_fold` disabled, a request for a
   secondary-structure prediction produced a silent `activate_tool` call followed by the
   prediction. The switch the user had just flipped was undone by the thing it was meant to
   constrain.
2. **The three-dot thinking indicator never stops.** It animates from page load onward, so
   it carries no information about whether a turn is in flight.
3. **Sessions live in a `<select>`.** There is no way to see conversations at a glance, and
   no way to create, rename or remove one from the browser.

This phase is small in new machinery and unusually large in *reversals*. Two of the three
fixes undo decisions earlier phases made deliberately and wrote down with reasons. That is
the interesting part of the phase, and §2 states both reversals explicitly rather than
letting them arrive as diffs.

**Out of scope.** No change to how tool state is stored, no change to the streaming
protocol, no framework for the client, and no widening of the delete surface beyond
sessions — tools, artifacts, jobs and staging stay CLI-only deletions
([PHASE_7_HARDENING.md](PHASE_7_HARDENING.md)).

### Reversal 1 — "activate and use within the same turn"

[PHASE_4_ORCHESTRATOR.md](PHASE_4_ORCHESTRATOR.md) §2 fixed the binding policy: bind every
registered tool, mark disabled ones in their description, and enforce state at *call* time.
The stated benefit was that the model could activate a tool and use it in one turn, and the
refusal message would teach it the lifecycle. The Phase 4 definition of done demonstrates
exactly that: *"disable/refuse/re-enable-and-use-in-one-turn"*.

That was a coherent design when the question was "can the agent recover from a refusal?"
It is the wrong design once the question is "who owns the tool switches?" A disabled tool is
a statement of intent by the user — that the wrapper is broken, or the output is untrusted,
or the environment is not ready. An agent that routes around it is not being helpful.

**The binding policy itself survives.** Everything stays bound, and enforcement stays at
call time. What changes is that `activate_tool` and `deactivate_tool` become *requests*: the
model may propose the change, and the existing approval gate suspends the turn until a human
answers. The model can still say "vienna_fold is off — want me to turn it on?" It can no
longer answer that question itself.

### Reversal 2 — "no delete surface in the API"

[PHASE_8_SERVICE_API.md](PHASE_8_SERVICE_API.md) §2 declined to expose deletion over HTTP,
and [PHASE_9_WEB_UI.md](PHASE_9_WEB_UI.md) §1 restated it: *"a button is not a better
boundary than a shell prompt."* That reasoning holds for artifacts, jobs, runs and staged
code — expensive, hard-to-reproduce things, where a deliberate trip to the terminal is a
feature.

It does not hold for a conversation. A session is the user's own text, cheap to recreate,
and the client that displays a list of them is precisely where a person expects to be able
to remove one. Requiring `toolhub prune sessions --older-than N --yes` to drop a session
named `gate2` is friction without safety: `--older-than` is computed from the mtime of the
whole SQLite file ([prune.py:158-168](../agentic/adaptrna_agentic/toolhub/prune.py#L158)),
so it cannot even target one session precisely.

**Narrowed claim:** sessions are deletable over HTTP and from the UI. Nothing else is. The
*agent* still cannot delete anything — `prune` remains outside the tool set, and the new
endpoints are not bound as agent tools.

---

## 2. Decisions this plan fixes

| Decision | Choice | Rationale |
|---|---|---|
| **How to take tool state away from the agent** | Keep `activate_tool`/`deactivate_tool` bound, add them to `GATED_TOOLS` | Removing them entirely would also remove the model's ability to *offer*, and the user would lose a useful conversational path ("this needs vienna_fold, shall I ask to enable it?"). Gating gives the same guarantee — nothing flips without a human — while keeping the affordance. It also reuses machinery that already exists and is already tested in both front ends. |
| **What the gate shows for a tool toggle** | Tool name, current state, target state | The gate's whole value is that the human sees what will actually happen (Phase 5 §2). For training that means the exact argv; for a toggle it means naming the tool and both states, so "Approve" is never ambiguous about direction. |
| **Session identity** | The `thread_id` stays the display name; rename is an `UPDATE` | The alternative — a `title` column in a sidecar table — would add a second source of truth that the terminal front end does not write, so a session renamed in the browser would still show its id in `chat --list-sessions`. Rewriting the key keeps one name everywhere at the cost of one guarded transaction. |
| **Session list payload** | `[{id, updated_at, checkpoints}]`, newest first | A rail sorted alphabetically is a worse rail. `checkpoint_id` is a UUIDv6, so `MAX(checkpoint_id)` per thread is genuinely the newest checkpoint and its embedded timestamp decodes without deserializing a single BLOB — recency for the cost of one `GROUP BY`. |
| **Session titles** | Not derived from the first message | Would require loading each session's earliest checkpoint on every list call. Users already name sessions (`--session work`); auto-titling can be revisited if that stops being true. |
| **Rail geometry storage** | `localStorage` | Deliberately different from the auth token's `sessionStorage` ([PHASE_9_WEB_UI.md](PHASE_9_WEB_UI.md) §2). A token that outlives the tab is a hazard; a sidebar width that resets every tab is an annoyance. |
| **New-session persistence** | Seed an empty checkpoint on create | Otherwise a session created in the rail vanishes on refresh before its first turn, which is exactly the kind of thing a rail makes visible and a dropdown hid. See §4 for the fallback if the checkpointer declines to write it. |
| **Thinking indicator semantics** | "Waiting on the model", not "a stream is open" | The dots currently span the whole turn including the streaming of text, which is when it is most obvious that the model is *not* being waited on. Hide them once text starts; bring them back before a tool call. |

---

## 3. Surface

### 3.1 What the browser shows

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ☰  AdaptRNA           install: ok                                        │
├──────────────┬───────────────────────────────────┬───────────────────────┤
│ ＋ New       ┃ Chat                              │ Tools                 │
│ [filter…   ] ┃                                   │  splice_site    ●  ⋯  │
│              ┃  you  fold this hairpin           │  vienna_fold    ○  ⋯  │
│ ▸ acceptor   ┃  ai   vienna_fold is disabled —   ├───────────────────────┤
│   2h ago  ⋯  ┃       shall I ask to enable it?   │ Jobs                  │
│ ▸ dod9live   ┃  you  yes                         │  acceptor_lora  ok    │
│   yesterday  ┃  ⚙   activate_tool({vienna_fold}) │                       │
│ ▸ crossover  ┃                                   │                       │
│   2d ago     ┃  [ask something…            ] [↵] │                       │
└──────────────┸───────────────────────────────────┴───────────────────────┘
   ↑ resizable                ↑ grip
        ┌─ Approval required ────────────────────────────┐
        │ Enable the tool 'vienna_fold'                  │
        │   tool           vienna_fold                   │
        │   current state  disabled                      │
        │   after approval active                        │
        │                        [ Decline ]  [ Approve ]│
        └────────────────────────────────────────────────┘
```

The top-bar session `<select>` is **removed**; the rail replaces it. The `☰` toggle sits
leftmost in the top bar so it stays reachable when the rail is collapsed.

### 3.2 HTTP

```
GET    /api/sessions            → [{id, updated_at, checkpoints}]   ← shape change
POST   /api/sessions            {id}       → the new row            ← new
PATCH  /api/sessions/{id}       {id: new}  → the renamed row        ← new
DELETE /api/sessions/{id}                  → {deleted: id}          ← new
```

Errors follow the Phase 7 mapping already installed in
[api/app.py:54-92](../agentic/adaptrna_agentic/api/app.py#L54): `KeyError` → 404 for an
unknown session, `ToolHubError` → 409 for a duplicate name or a rename blocked by a pending
approval, `ValueError` → 400 for a blank name.

The four `/tools/{name}/activate|deactivate` endpoints are unchanged — they are the *user's*
surface and always were.

---

## 4. Components

### 4.1 The gate (`agents/`)

| File | Change |
|---|---|
| `agents/tool_factory.py` | `GATED_TOOLS` gains `"activate_tool"`, `"deactivate_tool"`. `_DISABLED_NOTE` and `_check_active`'s refusal drop the imperative "call `activate_tool` first" and say the tool is enabled by the user; mirror the wording already in [runtime.py:138-141](../agentic/adaptrna_agentic/toolhub/runtime.py#L138). The two docstrings become "Ask the user to …; requires their approval" — the docstring *is* the description the model reads. The module docstring (lines 5-9) states the old Phase 4 policy and must be rewritten to §1's version: bound so the model can *ask*, enforced so it cannot *act*. |
| `agents/orchestrator.py` | `SYSTEM_PROMPT` lines 35-38 currently instruct the model to offer and use `activate_tool`; rewrite so tool state is stated as the user's. Add both names to the "pause for the user's approval … do not retry" sentence at lines 47-48. `_summarize` gains a case producing `Enable the tool 'vienna_fold' (currently disabled)`. `_details` gains `{tool, current_state, after_approval}` — it needs the `Registry`, which is only in `build_orchestrator_graph`'s closure, so give both helpers an optional `registry` parameter and pass it from `request_approval`. |

`request_approval`, `run_tools` and the router are untouched: the gate is generic over
`GATED_TOOLS` and both front ends already render whatever `summary`/`details` contain.

### 4.2 The indicator (`ui/`)

The JS was already right. [style.css:224](../ui/style.css#L224) declares
`.thinking { display: flex }` — an author-origin rule, so it beats the user agent's
`[hidden] { display: none }` regardless of specificity, and the element has been painted
since page load. Add `.thinking[hidden] { display: none; }`.

Then a `setThinking(bool)` helper in `app.js`, called from `consume()`: off on the first
`text` delta, on again in `closeBubble()` before a tool call, off at the end and while the
approval modal is up. `setBusy()` keeps owning `state.streaming` and the composer.

### 4.3 Sessions (`api/`)

New module **`api/sessions_store.py`** — plain `sqlite3`, no new tables, no new dependency:

```python
list_sessions(db_path)  -> list[dict]   # GROUP BY thread_id; MAX(checkpoint_id) → updated_at
session_exists(db_path, thread_id) -> bool
rename_session(db_path, old, new)      # UPDATE thread_id, one transaction
```

`rename_session` walks `("checkpoints", "writes", "checkpoint_blobs")` with a per-table
`try/except sqlite3.OperationalError`, exactly as
[prune.py:252-266](../agentic/adaptrna_agentic/toolhub/prune.py#L252) does — the table set
varies with the checkpointer version, and that function is the project's existing precedent
for coping with it.

Deletion **reuses** that function: promote `prune._delete_session` to a public
`delete_session(thread_id, db_path=None)` and call it from the router. One deletion path,
already covered by `test_prune.py`.

`api/routers/sessions.py` gains the three routes from §3.2 and switches `list_sessions` to
the store. `api/schemas.py` gains `SessionCreateRequest` / `SessionRenameRequest` beside the
existing bodies. The "Deliberately absent: any delete surface" comment at
[api/app.py:8-9](../agentic/adaptrna_agentic/api/app.py#L8) is narrowed to §1's version.

> **Resolved during implementation:** LangGraph 1.2.11 *does* write a checkpoint for
> `update_state(config, {"messages": []})` — verified against a scratch database before the
> route was written. `POST` seeds, and the fallback was not needed.

### 4.4 The rail (`ui/`)

```
ui/index.html   +aside.rail (head: new + filter · list · grip), +#rail-toggle, −.session
ui/style.css    .layout → three columns; --rail-w; body.rail-collapsed; .rail-grip
ui/render.js    +sessionRow(session, handlers, isCurrent), +relativeTime()
ui/api.js       +createSession / renameSession / deleteSession
ui/app.js       state.sessions, state.filter; refreshSessions() renders rows not <option>s;
                new/rename/delete handlers; toggle + drag-resize with localStorage
```

`.layout` becomes `grid-template-columns: var(--rail-w, 15rem) minmax(0,1fr) 22rem`;
`body.rail-collapsed` sets `--rail-w: 0`. The grip is a 4 px right-edge strip
(`cursor: col-resize`) whose `pointermove` writes `--rail-w` on the root element, clamped to
10–30 rem. The `max-width: 900px` block ([style.css:104-107](../ui/style.css#L104)) gains a
rule making the rail overlay rather than take a column.

`sessionRow` is modelled on `toolRow` ([render.js:81-114](../ui/render.js#L81)) and built
with `el()` — nothing in this client assigns `innerHTML`, and that stays true. Rename and
delete use `window.prompt` / `window.confirm`, matching the existing `newSession()` idiom
rather than introducing a second dialog system for two operations. All three mutations are
guarded on `state.streaming` / `state.pending`, the way `send()` is at
[app.js:128-133](../ui/app.js#L128).

---

## 5. Documentation

Three phases wrote the *old* behaviour into the docs as considered design, with reasons.
Those statements become false when this lands, and the reasons are the part that matters —
each edit changes the claim and records the reversal, rather than quietly rewording.

**The tool-state reversal.**
[documents/modules/agents.md](../documents/modules/agents.md) restates `GATED_TOOLS`
verbatim (`:122`) with prose explaining why each member is gated — that prose needs a fourth
reason, *changes what the assistant is allowed to run*, which is unlike the other three
(they are gated for cost and blast radius; this one for authority). Also `:98-100` (the
system-prompt paraphrase), `:169` (`_DISABLED_NOTE` quoted verbatim), and `:186`/`:190`
(the tool table, where **bold** means gated).
[documents/architecture.md:162](../documents/architecture.md#L162) repeats the tuple.
[documents/workflows/inference-and-tools.md:145-160](../documents/workflows/inference-and-tools.md#L145)
is the worst offender: a worked transcript of the agent disabling `vienna_fold`, being
refused, and re-activating itself — the reported bug, documented as a feature. Replace the
transcript; keep and re-argue the binding explanation at `:150`.
[documents/extending.md:123-125](../documents/extending.md#L123) is the recipe this change
follows — verify it end to end and add `activate_tool` as its worked example.

**The delete surface.** Four live claims to narrow:
[modules/api.md:41](../documents/modules/api.md#L41),
[modules/api.md:245](../documents/modules/api.md#L245),
[workflows/operations.md:218-219](../documents/workflows/operations.md#L218),
[workflows/operations.md:272](../documents/workflows/operations.md#L272). Plus the route
table at [modules/api.md:188-191](../documents/modules/api.md#L188) and the curl recipes at
[operations.md:235-236](../documents/workflows/operations.md#L235).
`operations.md:116` — "`prune` is deliberately not an agent tool" — is untouched and still
true; say so, because it is the sentence that keeps the reversal narrow.

**The UI.** [documents/modules/web-ui.md](../documents/modules/web-ui.md): file map
(`:51-52`), the `api.js` surface (`:92-94`), the `state` shape (`:111`), the boot sequence
(`:155`), and `:223` — "**No delete affordances**, because the API has no delete surface" —
now false. Add a numbered section for the rail, and a note in §7 on the `[hidden]` trap,
since §7 exists precisely to warn about this class of thing.
[ui/README.md](../ui/README.md) lines 20-24 argue the no-framework choice is justified *at*
1,109 lines "because none of those lines are client-side state", and name client-side state
as the tripwire for porting to React. The rail adds `state.sessions`, `state.filter` and
persisted geometry — the tripwire is tripped. Record the new count, say so, and either
re-argue it or leave it flagged. Do not delete the paragraph.
Also: [agentic/README.md:164-165](../agentic/README.md#L164) (endpoint table),
[project_structure.md:195](../documents/project_structure.md#L195) (add `sessions_store.py`;
the "16 management tools" count is unchanged — none are removed),
[configuration.md](../documents/configuration.md) (the two new `localStorage` keys beside
the existing `sessionStorage` note), and [testing.md](../documents/testing.md) (the test
tables).

**Screenshots.** `docs/web-ui.png` shows the old picker and no rail. Re-take it. Capture the
tool-activation modal during Gate 2 if it is cheap — `docs/approval-gate.png` is the
precedent.

**Grep gate — the docs are not done until this returns only historical references:**

```bash
grep -rn "call activate_tool\|no delete surface\|no delete endpoint\|session picker" \
  documents/ ui/README.md agentic/README.md README.md
```

---

## 6. Tests

Run from the layer directory; `testpaths` is relative. Baseline **381 green**
(`-m 'not ui'`) plus 11 opt-in browser tests.

| File | Asserts |
|---|---|
| `test_approval_gate.py` | `activate_tool` on a disabled tool interrupts **before the registry is touched** — the entry is still `disabled` at the interrupt; approve → `active`; **decline → still `disabled`, and the model is told not to retry.** The last one is the reported bug, pinned. |
| `test_tool_factory.py` | both names are in `GATED_TOOLS`; the disabled-tool description and the `_check_active` refusal no longer instruct the model to self-activate |
| `test_prompts.py` | the system prompt does not tell the model to activate tools on its own |
| `test_api_sessions.py` | create → listed; duplicate → 409; blank → 400; rename carries the history and the old id is gone; rename onto an existing id → 409; rename with an approval pending → 409; delete removes `checkpoints` **and** `writes` rows; unknown id → 404; the list is newest-first with a parseable `updated_at` |
| `test_ui_contract.py` | **update** `test_the_session_picker_gets_a_list_of_names` ([:215-222](../agentic/tests/test_ui_contract.py#L215)) — it pins `isinstance(name, str)` and fails on the new shape. Re-pin as objects with `id`/`updated_at`, and add the create/rename/delete round trip the rail depends on |
| `test_ui_serving.py` | unchanged; confirm still green (offline-safe assets, module shell) |
| `test_ui_browser.py` (`-m ui`) | rail lists sessions; new/rename/delete; toggle hides and restores; the grip resizes and the width survives a reload; **the dots are hidden at rest and visible mid-turn** |
| `test_prune.py` | stays green after `_delete_session` is promoted |
| `test_scenarios.py`, `tests/scenarios/` | audit for any scenario that relied on the agent self-activating — those now suspend on an interrupt |

~15 new deterministic tests (≈396 total) plus browser cases. Engine's 135 untouched.

---

## 7. Implementation order

1. **Part B first** — one CSS line plus `setThinking()`. It is independent, it is the
   fastest thing to confirm by eye, and it makes every subsequent manual check readable.
2. **Part A** — `GATED_TOOLS`, the wordings, `_summarize`/`_details`, the prompt; then
   `test_approval_gate.py` / `test_tool_factory.py` / `test_prompts.py`; then the CLI check
   that both front ends gate identically.
3. **`sessions_store.py` + `prune.delete_session`** with `test_api_sessions.py`, before any
   client work — including the `update_state` question in §4.3, whose answer decides how
   `POST` behaves.
4. **The routes**, then `test_ui_contract.py` re-pinned.
5. **The rail**: markup and CSS, then `render.js`/`api.js`, then `app.js` wiring, then
   toggle and resize persistence.
6. **Docs (§5)** and the grep gate; re-take `docs/web-ui.png`.
7. **Browser suite** and the gates in §8; close out `MASTER_PLAN.md` §8.

---

## 8. Verification / definition of done

**Gate 1 — deterministic.** `cd agentic && ../.venv/bin/python -m pytest` green
(381 + ~15); `cd engine && ../.venv/bin/python -m pytest` → 135.

**Gate 2 — the reported bug, in both front ends.** With a tool disabled (the manifest
currently has `vienna_cofold` and `splice_site_acceptor` disabled; or toggle `vienna_fold`
off in the Tools panel):

```bash
.venv/bin/python -m adaptrna_agentic.cli.serve --open
```

Ask for a secondary-structure prediction. The assistant reports the tool is disabled and
offers to ask; the approval modal names the tool, its current state and the state after
approval. **Decline → `toolhub_data/tools.json` still reads `"state": "disabled"`, and the
assistant does not retry.** Approve → it flips and the prediction runs. Repeat at
`python -m adaptrna_agentic.cli.chat` and confirm the prompt text matches the modal, the way
Phase 9 Gate 4 required for training.

**Gate 3 — the dots.** Dark at rest, including immediately after page load. During a turn:
visible while waiting, gone once text streams, back before a tool call, gone at `done`, and
gone while the approval modal is open.

**Gate 4 — the rail.** Create a session, send a turn, rename it, reload — new name and full
history. Delete it; confirm at the shell:

```bash
sqlite3 chat_data/sessions.sqlite "SELECT DISTINCT thread_id FROM checkpoints"
```

Filter narrows the list; drag the grip, collapse, reload — geometry persists.

**Gate 5 — the docs match the code.** §5's grep returns only historical references, and
`docs/web-ui.png` shows the rail.

**Gate 6 — the two front ends are still one system** (Phase 8 §8 Gate 3, Phase 9 Gate 3): a
session created in the rail appears in `chat --list-sessions` and continues from the
terminal; one created in the terminal appears in the rail with a sensible timestamp.

---

## 9. Risks and notes

- **`GET /api/sessions` changes shape.** Consumers are `ui/api.js` and
  `test_ui_contract.py` only — the CLI has its own `_list_sessions`
  ([cli/chat.py:170-190](../agentic/adaptrna_agentic/cli/chat.py#L170)). Contained, but
  change them in the same commit or the UI blanks silently.
- **Rename rewrites a primary key.** `thread_id` is part of the PK on both tables, so this
  is an `UPDATE`, not a metadata edit. Guarded against collisions and against a pending
  approval. A turn streaming from *another* process mid-rename would write under the old
  id — acceptable for a single-user local tool, worth a comment at the call site.
- **Deletion is irreversible and now one click away.** `window.confirm` and the fact that
  `chat_data/` is git-ignored are the entire safety net. That is the trade §1 accepted.
- **Gating a formerly silent path changes turn shapes.** Anything that assumed
  `activate_tool` executes inline now suspends; audit `tests/scenarios/` (§6) rather than
  discovering it in a browser run.
- **`Registry` never re-reads `tools.json` after construction**
  ([registry.py:58](../agentic/adaptrna_agentic/toolhub/registry.py#L58)), so a long-lived
  server and a concurrent CLI `toolhub activate` can diverge until the `revision` guard
  refuses a save. Pre-existing and out of scope — but it is why Gate 2 reads the JSON from
  disk instead of trusting the panel.
- **The client is now past its own tripwire.** [PHASE_9_WEB_UI.md](PHASE_9_WEB_UI.md) §2 set
  "roughly a thousand lines, or real client-side state" as the signal to port to React. The
  rail crosses both. This plan does not port it — the API is still the contract and the
  switch still costs the client alone — but §5 requires the reassessment be written down in
  `ui/README.md` rather than left implicit, so the next feature confronts a stated position
  instead of an assumption.


---

## 10. What actually happened

All six gates passed on 2026-08-13. **405 agentic tests** (from 381) and **15 opt-in browser
tests** (from 11) green; engine's 135 untouched.

### The five tests that failed first were the interesting part

Adding two names to `GATED_TOOLS` broke exactly five tests, and every one of them was a test
of the behaviour being removed rather than a test the change had damaged:

| Test | What it pinned |
|---|---|
| `test_orchestrator_graph.py::test_activate_then_use_in_one_turn` | Phase 4's headline: activate and use in a single turn |
| `test_orchestrator_graph.py::test_disabled_tool_refusal_reaches_the_model` | the refusal text that said "Call activate_tool" |
| `test_tool_factory.py::test_disabled_capability_returns_refusal_as_result` | the same string, one layer down |
| `test_pipeline_tools.py::test_only_consequential_tools_are_gated` | the gated set as a literal |
| `test_scenarios.py::test_scenario[management]` | a recorded conversation of the model re-enabling a tool and using it |

The one-turn test was kept under a new name — same script, opposite assertion — because a
test that records a reversal is worth more than a deleted one. `management.yaml` grew a third
turn so the scenario now covers decline *and* approve; the decline turn is the bug report.

### What the plan got wrong

* **`_summarize` needed a guard the plan did not anticipate.** It reads the registry to name
  a tool's current state, so a hallucinated tool name would have raised *inside the gate* —
  turning a recoverable bad tool call into a broken turn. `_tool_state` returns `"unknown"`
  instead, and `test_the_gate_survives_an_unknown_tool_name` pins it.
* **The prompt test went to the wrong file.** §6 put it in `test_prompts.py`, which is about
  codegen prompts; it belongs in `test_orchestrator_graph.py`, which already imports
  `SYSTEM_PROMPT`.
* **`test_approval_gate.py`'s `stack` fixture registers no tools** — it exists for the
  training gate, which needs none. The toggle tests needed a `stack_with_tool` fixture.
* **The rail's boot order matters.** `state.session` is what marks the current row, so
  `loadSession` has to run *before* `refreshSessions`, not after as §4.4's ordering implied.
* **A browser assertion was wrong on its own terms.** The filter test assumed the list had
  more than one row after creating one session; a session with no checkpoint is a client-side
  draft that never reaches the server, so the count was 1. The test now creates two.

### What the live run found

Gate 2 was better than the plan's success criterion. The plan expected the assistant to call
`activate_tool` and be stopped by the gate. With the rewritten prompt it did not call it at
all unprompted:

> The tool for this — `vienna_fold` — is currently disabled, so I can't fold that sequence.
> I won't guess a structure for it, and I won't substitute another tool. If you'd like it
> back on, I can put the request to you to enable `vienna_fold` — just say the word, and it
> takes effect only once you approve.

Asked to proceed, it called `activate_tool`, the gate suspended, and `tools.json` still read
`"state": "disabled"` at the interrupt. Declining left it disabled (revision unchanged at 5).
Approving flipped it and folded `GGGGAAAACCCC` to `((((....))))` at −5.4 — the golden pinned
in Phase 3. The terminal's `_prompt_approval` renders the same request field-for-field.

That is the distinction §2 drew, confirmed in practice: **the gate stops the action, the
prompt stops the intent.** Gating alone would have produced an approval modal on every
disabled tool the assistant wanted; the prompt is what makes the modal rare and deliberate.

### Notes for whoever is next

* `docs/tool-gate.png` was captured from a *deactivation* request that was then declined, so
  taking that screenshot changed no state. Worth copying as a technique.
* The client is now **1,329 lines of JS**, past the thousand-line tripwire Phase 9 set, and
  carries its first client-side state. Not ported; the reasoning for holding is recorded in
  `ui/README.md` and it is weaker than it was. The next feature that adds state should treat
  that as a decision to make rather than a default to inherit.
