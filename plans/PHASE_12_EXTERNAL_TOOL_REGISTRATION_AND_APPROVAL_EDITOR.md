# Phase 12 — Fix external-tool registration + a code editor in the approval window

## Context

Two independent problems surfaced while working with **non-adapter (external) tools** —
Python wrappers around classical packages (e.g. ViennaRNA), as opposed to LoRA **adapter**
tools trained on the backbone.

1. **External tools can never actually be registered.** The agent can *generate* and
   *land* an external wrapper (e.g. the untracked
   [rna_secondary_structure.py](../adaptrna_custom/tools/rna_secondary_structure.py), whose
   `SPEC.name` is `rnafold`), but nothing ever calls `Registry.register_external()`, so the
   tool never appears in the manifest and is never servable. `register_external` exists but
   is wired **only** into the CLI, never into the agent flow. Landing a *task* auto-registers
   on next use; landing a *tool* is a dead end. See root cause below.

2. **The verification/approval window shows no code.** Before code is written and (now)
   registered, the `land_generated_code` approval modal only lists file paths + line counts
   and tells the user to "open it in your editor". The user wants the code shown **inside**
   the window in an embedded **Monaco** editor (read-only for now, possibly editable later),
   with **browser-style tabs** when multiple files are involved so they can switch between
   files.

Decisions already made with the user:
- Editor: **Monaco (VS Code editor)**, **vendored locally** under `ui/vendor/monaco/` and
  served by the API like every other asset — *not* from a CDN. The UI is deliberately
  offline/self-contained ([documents/modules/web-ui.md](../documents/modules/web-ui.md) §1, §9),
  and [test_ui_serving.py](../agentic/tests/test_ui_serving.py) asserts this with teeth (a single
  CDN `<script>` would break an offline workstation while passing everything else). Vendoring
  keeps that guarantee and the test green.
- Registration: **auto-register on approval** — `land_generated_code` registers the external
  tool itself, symmetric with how tasks become usable after landing.

---

## Problem 1 — External tool registration

### Root cause
- [tool_factory.py](../agentic/adaptrna_agentic/agents/tool_factory.py) `land_generated_code`
  (lines ~305-327) calls `staging.land(stage)` and returns, with a misleading `"next"`
  message ("Register its functions as tools with the external-tool flow") pointing at a flow
  that does not exist for the agent.
- [registry.py](../agentic/adaptrna_agentic/toolhub/registry.py) `register_external`
  (lines 176-255) is only reachable from the CLI.
- `register_external` → `contract.load_spec(module_path)` →
  `importlib.import_module("adaptrna_custom.tools.<name>")`, but nothing puts the repo root
  on `sys.path` in the server/agent process, so even a manually-registered custom tool would
  fail to import at runtime (built-in `vienna.py` lives inside the installed package, so it
  never exposed this gap).

### Changes

1. **`land_generated_code` auto-registers external tools** —
   [tool_factory.py](../agentic/adaptrna_agentic/agents/tool_factory.py), inside
   `_codegen_tools`. After `written = staging.land(stage)`:
   - If `stage.kind == "tool"`:
     - Call `discovery.ensure_importable()`.
     - **Evict stale staging modules from `sys.modules`.** `pipeline._verify_wrapper`
       imported `adaptrna_custom.tools.<name>` (and its parent packages) from the *staging*
       directory during generation; those cached modules would shadow the just-landed file.
       Pop `stage.module_path`, `adaptrna_custom.tools`, and `adaptrna_custom` from
       `sys.modules`, then `importlib.invalidate_caches()`.
     - Call `entries = registry.register_external(stage.module_path)` and include the
       registered tool names in the returned dict.
   - Update the `"next"` message to reflect what actually happened (e.g. "Registered tools:
     `rnafold_mfe_structure` — active and usable now.").
   - Errors surface normally through `_surface_errors` → `ToolException`, so if the wrapped
     package is not installed, the agent gets the exact install hint from
     `register_external` (see caveat).

2. **`register_external` resolves `adaptrna_custom` imports** —
   [registry.py](../agentic/adaptrna_agentic/toolhub/registry.py) `register_external`: call
   `discovery.ensure_importable()` (from `adaptrna_agentic.codegen.discovery`, lazy import to
   avoid any cycle — `discovery` only imports `settings`) before `contract.load_spec(...)`.

3. **Custom external tools import at runtime** —
   [tool_factory.py](../agentic/adaptrna_agentic/agents/tool_factory.py) `build_agent_tools`
   (lines ~412-427): call `discovery.ensure_importable()` at the top so `_external_tool`'s
   `importlib.import_module(entry.external["module"])` resolves landed
   `adaptrna_custom.tools.*` wrappers on a fresh server start.

### Caveat (surfaced, not silently fixed)
`register_external` refuses if the wrapped package (e.g. ViennaRNA / `import RNA`) is not
installed — and `create_external_tool` verification *skips* (does not fail) golden cases when
the package is absent, so a wrapper can be landed for an uninstalled package. In that case the
auto-register step returns the exact `pip install` hint as the tool result. Wiring an
approval-gated auto-install into the agent (the CLI's `--yes` path) is **out of scope** here;
the plan just makes the failure legible. Confirm ViennaRNA is installed to exercise the happy
path end-to-end.

---

## Problem 2 — Monaco code editor with file tabs in the approval window

### Server: send file contents
[orchestrator.py](../agentic/adaptrna_agentic/agents/orchestrator.py) `_details` (lines 134-141),
`land_generated_code` branch. Today it sends `files` (`[{path, lines}]`), `staging_path`, and
an unused `diff`. Add the source per file so the client can render it:

```python
details["files"] = [
    {"path": d, "lines": len(c.splitlines()), "content": c}
    for d, c in sorted(stage.files.items())
]
```

`stage.files` (repo-relative path → content) already carries everything; the `diff` string can
be dropped since the client now renders structured content. This keeps the "show exactly what
will be written" guarantee documented in [render.js](../ui/render.js) (lines 348-353).

### Vendor Monaco locally
Download `monaco-editor` (e.g. v0.52.x) and copy its `min/vs/` tree into
`ui/vendor/monaco/vs/`. The API already serves `ui/` statically; confirm the static mount
covers `ui/vendor/**` (check `api/` static-file routing / `test_ui_serving.py` for the mount
path) so `/ui/vendor/monaco/vs/loader.js` resolves. This is the one place `pip`-shipped
vendored assets enter the repo — note it in the commit; it is a few MB.

### Client: tabbed Monaco viewer
Vanilla ES-module UI ([ui/](../ui/), no build step). Monaco initialises imperatively via its AMD
loader, so `render.js` builds the container/tabs and `app.js` drives Monaco after insertion.

1. **[ui/index.html](../ui/index.html)** — load Monaco's AMD loader from the **local** vendor
   path with a `<script src="/ui/vendor/monaco/vs/loader.js">` before `app.js`. Configure
   `require.config({ paths: { vs: "/ui/vendor/monaco/vs" } })` and set
   `self.MonacoEnvironment.getWorkerUrl` to a **local** blob/worker proxy pointing at the
   vendored `vs/base/worker/workerMain.js` (read-only highlighting runs on the main thread;
   workers only power language services, so a proxy avoids 404s while staying offline). No
   change to the `#approval` modal skeleton is required — the code panel is injected into
   `#approval-body`.

2. **[ui/render.js](../ui/render.js)** `approvalBody` — when an item's `details.files` entries
   carry `content`, replace the current per-file `"write ... (N lines)"` lines with a code
   panel: a tab bar (`el("button", …)` per file, labelled by path basename, first active) and
   an empty editor mount `<div class="approval-code-editor">`. Stash the file list (path +
   content) on the mount element (e.g. `dataset`/a WeakMap) so `app.js` can pick it up. Keep
   the `staging_path` line; drop the "open it in your editor" hint (now redundant).

3. **[ui/app.js](../ui/app.js)** `showApproval` (lines ~189-203) — after rendering the body,
   if a code mount exists: lazy-load Monaco via `require(["vs/editor/editor.main"], …)`,
   create **one** read-only editor and **one model per file** (`monaco.editor.createModel`,
   language inferred from extension: `.py`→`python`, `.yaml`/`.yml`→`yaml`, else
   `plaintext`). Tab clicks call `editor.setModel(model)` and toggle an `active` class. Call
   `editor.layout()` on show. In `decide()` / on close, `editor.dispose()` and dispose models
   to avoid leaks across successive approvals. Guard for Monaco load failure (offline) by
   falling back to a read-only `<pre>` of the content.

4. **[ui/style.css](../ui/style.css)** — style `.approval-tabs` (horizontal, browser-like,
   active tab highlighted) and `.approval-code-editor` (fixed height, e.g. `min(50vh, 30rem)`,
   bordered) within the existing `.modal` (max-height 85vh, `min(46rem, 100%)` wide — likely
   widen slightly for code). Reuse existing CSS custom properties (`--panel`, `--ink`,
   `--mono`, `--accent`) for light/dark parity; pick Monaco theme `vs`/`vs-dark` from the
   active color scheme.

### Read-only now, editable later
Editor created with `readOnly: true`. Making it editable later is a one-flag change plus
sending the edited buffer back on approve — noted, not built now.

---

## Files to change (summary)

| File | Change |
|------|--------|
| [agentic/adaptrna_agentic/agents/tool_factory.py](../agentic/adaptrna_agentic/agents/tool_factory.py) | `land_generated_code` auto-registers `kind=="tool"` stages (evict stale staging modules, `register_external`, update `next`); `build_agent_tools` calls `ensure_importable()` |
| [agentic/adaptrna_agentic/toolhub/registry.py](../agentic/adaptrna_agentic/toolhub/registry.py) | `register_external` calls `discovery.ensure_importable()` before `load_spec` |
| [agentic/adaptrna_agentic/agents/orchestrator.py](../agentic/adaptrna_agentic/agents/orchestrator.py) | `_details` sends per-file `content` for `land_generated_code` |
| `ui/vendor/monaco/vs/**` (new) | Locally vendored Monaco `min/vs` tree |
| [ui/index.html](../ui/index.html) | Local Monaco loader + `require.config` + `MonacoEnvironment` |
| [ui/render.js](../ui/render.js) | `approvalBody` builds tab bar + editor mount |
| [ui/app.js](../ui/app.js) | `showApproval`/`decide` create/dispose Monaco, per-file models, tab switching |
| [ui/style.css](../ui/style.css) | `.approval-tabs`, `.approval-code-editor`, wider modal |
| `agentic/tests/…` | New assertions (external-tool land→register; UI serves vendored Monaco) |

No new agent tool, no new approval gate, no changes to `staging.land` or the verification
harness.

---

## Documentation to update

Keep the docs consistent with the new behaviour (the repo treats docs as first-class):

- **[documents/workflows/external-tools.md](../documents/workflows/external-tools.md)** — §6
  ("Generating one") currently ends by telling the user to run
  `toolhub register-external adaptrna_custom.tools.<name>` manually and quotes the misleading
  *"Register its functions as tools with the external-tool flow."* Rewrite: landing an external
  wrapper now **auto-registers** it (one approval, tool live immediately), symmetric with tasks.
  Update the "next step" wording. Note the caveat: the wrapped package must be installed or the
  register step returns the install hint.
- **[documents/modules/toolhub.md](../documents/modules/toolhub.md)** — wherever it describes
  `register_external` as CLI-only, add that the agent's `land_generated_code` now calls it for
  `kind=="tool"` stages, and that registration/runtime call `discovery.ensure_importable()` so
  `adaptrna_custom.tools.*` wrappers resolve.
- **[documents/modules/codegen.md](../documents/modules/codegen.md)** — the landing description
  should state that a landed tool is registered (not just written), and mention the
  `sys.modules` eviction of the stale staging copy before re-import.
- **[documents/modules/web-ui.md](../documents/modules/web-ui.md)** — this doc asserts the client
  is fully self-contained / no CDN. Update §1, §2 (file map), §6 (`render.js` element list),
  and §10 to record: Monaco is **vendored locally** under `ui/vendor/monaco/` (still offline,
  still no CDN), the approval modal now embeds a read-only tabbed code editor, and add a line
  to §2's file map for the vendored tree. Update the JS line counts if they gate anything.
- **[ui/README.md](../ui/README.md)** and **[documents/modules/agents.md](../documents/modules/agents.md)**
  — light touch: mention the embedded editor in the approval gate and the auto-register on land,
  if they describe those flows.

---

## Verification (end-to-end)

Registration (Problem 1):
1. Ensure ViennaRNA is installed (`python -c "import RNA"`); if not, `pip install ViennaRNA`.
2. In the chat, ask for an RNA secondary-structure / MFE-folding tool so the agent runs
   `create_external_tool` → `land_generated_code`; approve the land.
3. Confirm the tool is now registered and active: `toolhub list` (or the agent's
   `list_tools`) shows `rnafold_mfe_structure`; `toolhub test rnafold_mfe_structure` (or
   `test_tool`) passes its golden cases.
4. Restart the server and confirm the tool still loads (validates
   `build_agent_tools` → `ensure_importable`) and can fold a sequence, e.g. `GGGGAAAACCCC`
   → `((((....))))`.
5. Add/adjust a unit test asserting `land_generated_code` on a `kind=="tool"` stage produces
   a manifest entry (mirroring existing codegen/toolhub tests).

Editor (Problem 2):
6. Trigger any `land_generated_code` approval (task = 3 files, tool = 1 file). Confirm the
   modal shows a Monaco editor with the file content, read-only, correct syntax highlighting.
7. With a multi-file task, confirm the tab bar switches files (browser-like) and each tab
   shows the right content; approve/decline and reopen to confirm the editor is disposed and
   recreated cleanly (no leak, no duplicate editors).
8. Confirm light/dark theme parity, and that Monaco loads with **no network** (vendored
   locally) — disconnect and reload to prove it.

Tests:
9. `test_ui_serving.py` (self-contained/offline assertion) must stay **green** — Monaco is
   vendored, not CDN. Extend it to assert the vendored `loader.js` is actually served.
10. `test_ui_contract.py` reads UI field/event names by name — if `_details` renames anything
    (e.g. adds `content` to `files`), update the contract test accordingly.
11. Optionally add a `pytest -m ui` (Playwright) check that the approval modal renders the
    editor and switches tabs, mirroring the existing modal test.
