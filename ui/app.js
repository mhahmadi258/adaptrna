/**
 * Wiring: panels, the session picker, job polling, and the event dispatch for a turn.
 *
 * The server holds all the state — conversations in the checkpointer, tools in the
 * manifest, jobs in the job store — so this file keeps only what is genuinely about
 * *this browser tab*: which session is open, whether a turn is in flight, and whether an
 * approval is on screen.
 */

import { api, resumeSession, sendMessage, token } from "./api.js";
import * as ui from "./render.js";

// ------------------------------------------------------------------ monaco
// One editor instance lives for the lifetime of a single approval dialog.
// It is created in showApproval and disposed in _closeApprovalEditor.
let _monacoEditor = null;
let _monacoModels = [];

function _langFromPath(path) {
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".yaml") || path.endsWith(".yml")) return "yaml";
  return "plaintext";
}

function _closeApprovalEditor() {
  for (const m of _monacoModels) m.dispose();
  _monacoModels = [];
  if (_monacoEditor) { _monacoEditor.dispose(); _monacoEditor = null; }
}

function _initMonacoEditor(mount, files) {
  _closeApprovalEditor();

  const theme = matchMedia("(prefers-color-scheme: dark)").matches ? "vs-dark" : "vs";

  // Lazy-load Monaco via the AMD loader injected in index.html.
  require(["vs/editor/editor.main"], function (monaco) {
    if (!mount.isConnected) return; // modal closed before Monaco loaded

    const models = files.map((f) =>
      monaco.editor.createModel(f.content, _langFromPath(f.path))
    );
    _monacoModels = models;

    const editor = monaco.editor.create(mount, {
      model: models[0],
      readOnly: true,
      theme,
      automaticLayout: false,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      fontSize: 13,
      lineNumbers: "on",
      wordWrap: "off",
    });
    _monacoEditor = editor;

    // Wire the tab bar that render.js built alongside the mount.
    const panel = mount.parentElement;
    if (panel) {
      panel.addEventListener("click", (e) => {
        const btn = e.target.closest(".approval-tab");
        if (!btn) return;
        const idx = Number(btn.dataset.index);
        if (!Number.isFinite(idx) || !models[idx]) return;
        panel.querySelectorAll(".approval-tab").forEach((b) =>
          b.classList.toggle("approval-tab-active", b === btn)
        );
        editor.setModel(models[idx]);
      });
    }

    // layout() is required when the container was hidden during create().
    editor.layout();
  });

  // Fallback: if require is not available (Monaco failed to load), show plain text.
  if (typeof require === "undefined") {
    _monacoFallback(mount, files[0]);
  }
}

function _monacoFallback(mount, file) {
  if (!file) return;
  mount.className = "approval-code-editor approval-code-editor-fallback";
  const pre = document.createElement("pre");
  pre.textContent = file.content;
  mount.appendChild(pre);
}

const $ = (id) => document.getElementById(id);

const JOB_POLL_MS = 3000;
//: Nothing is running, so nothing is changing fast — but a run started from the chat still
//: has to show up in the rail without a reload, which the old "stop polling when idle"
//: never did.
const JOB_IDLE_POLL_MS = 15000;
//: The API caps `tail` at 2000 (`routers/jobs.py`); `test_ui_contract.py` reads this list
//: and asks the server to honour every value in it.
const LOG_TAIL_CHOICES = [200, 500, 2000];

//: Markdown is re-parsed from the whole reply on each repaint, so tokens are coalesced
//: into frames rather than repainting per delta.
const MARKDOWN_REPAINT_MS = 80;

//: Rail geometry is chrome for *this browser*, not server state, so it belongs in
//: localStorage — deliberately unlike the auth token, which uses sessionStorage so it dies
//: with the tab. A width that reset on every tab would just be an annoyance.
const RAIL_WIDTH_KEY = "adaptrna.rail.width";
const RAIL_COLLAPSED_KEY = "adaptrna.rail.collapsed";
const RAIL_VIEW_KEY = "adaptrna.rail.view";
const RAIL_MIN_REM = 10;
const RAIL_MAX_REM = 30;

const state = {
  session: null,
  pending: null, // the approval request the graph is suspended on, if any
  streaming: false,
  jobsTimer: null,
  sessions: [], // the last fetched list, so filtering does not need a round trip

  //: `view` and `job` are *which pane is on screen* — chrome, like the rail's width, and
  //: nothing the server could disagree with. `jobs` / `jobStatus` are a render cache of a
  //: list the server owns, exactly like `sessions`.
  view: "sessions", // "sessions" | "jobs"
  job: null, // the run whose log has replaced the chat, if any
  jobs: [],
  jobStatus: {}, // id → the per-job status call, which is the only source of progress
  filter: { sessions: "", jobs: "" }, // per view, so a needle does not follow you across
  logTimer: null,
  logTail: LOG_TAIL_CHOICES[0],
  logFollow: true,
  logError: null,
};

// ------------------------------------------------------------------ chat

const append = (node) => {
  $("chat-log").append(node);
  scrollChat();
  return node;
};

function scrollChat() {
  const log = $("chat-log");
  log.scrollTop = log.scrollHeight;
}

function setBusy(busy) {
  state.streaming = busy;
  $("composer-send").disabled = busy;
  $("composer-input").disabled = busy;
  setThinking(busy);
  if (!busy) $("composer-input").focus();
}

/**
 * The dots mean "waiting on the model", not "a stream is open".
 *
 * Spanning the whole turn made them least informative exactly when the answer was already
 * arriving, so `consume()` turns them off on the first token and back on before a tool
 * call. `setBusy` still owns `state.streaming` and the composer.
 */
function setThinking(thinking) {
  $("thinking").hidden = !thinking;
}

/**
 * Consume one stream.
 *
 * Used for both a new turn and a resumed one: `/resume` answers with a *new* stream
 * continuing the same turn, so there is no separate resume path — just this, called
 * again.
 */
async function consume(run) {
  setBusy(true);
  let bubble = null; // the bubble currently accumulating deltas
  let markdown = ""; // its raw text, re-rendered rather than appended to
  let painted = 0;
  let touchedTools = false;

  const paint = (force) => {
    // Markdown has to be re-parsed from the whole reply on every repaint, so repaint on a
    // timer rather than per token — and always once at the end of the block.
    const now = performance.now();
    if (!force && now - painted < MARKDOWN_REPAINT_MS) return;
    painted = now;
    bubble.set(markdown);
    scrollChat();
  };

  const closeBubble = () => {
    if (bubble) paint(true);
    bubble = null;
    setThinking(true); // whatever comes next, we are waiting on the model again
  };

  await run(({ event, data }) => {
    switch (event) {
      case "text":
        if (!bubble) {
          bubble = ui.assistantMessage();
          markdown = "";
          painted = 0; // so the first token of a new bubble paints immediately
          append(bubble.node);
          setThinking(false); // the answer is arriving; the dots have nothing left to say
        }
        markdown += data.delta || "";
        paint(false);
        break;

      case "tool_call":
        closeBubble(); // text after a tool call belongs in a new bubble
        append(ui.toolCallRow(data.name, data.args));
        break;

      case "tool_result":
        closeBubble();
        touchedTools = true;
        append(ui.toolResultRow(data.name, data.content));
        break;

      case "approval_required":
        closeBubble();
        state.pending = data;
        // The modal would render fine over a log, but the gate's whole value is that the
        // human sees what led here. Put the conversation back first.
        closeJob();
        showApproval(data);
        break;

      case "done":
        closeBubble();
        state.pending = null;
        break;

      case "error":
        closeBubble();
        append(ui.errorMessage(data.message || "the turn failed"));
        break;
    }
  });

  closeBubble(); // a stream that ended without `done` still leaves rendered text
  setBusy(false);
  if (touchedTools) refreshPanels();
}

async function send() {
  const input = $("composer-input");
  const text = input.value.trim();
  if (!text || state.streaming) return;

  if (state.pending) {
    append(ui.noticeMessage("Answer the approval request first."));
    return;
  }

  input.value = "";
  append(ui.userMessage(text));

  await consume((onEvent) => sendMessage(state.session, text, onEvent));
}

// ------------------------------------------------------------------ approval

function showApproval(request) {
  ui.clear($("approval-body")).append(ui.approvalBody(request));
  $("approval-note").value = "";
  $("approval").hidden = false;
  $("approval-approve").focus();

  const mount = $("approval-body").querySelector(".approval-code-editor");
  if (mount) {
    const files = ui.editorMountFiles(mount);
    if (files && files.length > 0) _initMonacoEditor(mount, files);
  }
}

/** Values changed from what the gate proposed, keyed by their dotted `data-edit-path`
 * (Phase 13 §5) — untouched fields (`input.value === input.defaultValue`) are not sent,
 * so an edits object only ever names what the human actually changed. */
function _collectEdits() {
  const edits = {};
  for (const input of $("approval-body").querySelectorAll("[data-edit-path]")) {
    if (input.value === input.defaultValue) continue;

    const path = input.dataset.editPath;
    if (input.dataset.editJson === "true") {
      try {
        edits[path] = JSON.parse(input.value);
      } catch {
        continue; // malformed JSON: leave it out rather than send garbage
      }
      continue;
    }

    const raw = input.value;
    if (raw.trim() === "") {
      edits[path] = raw;
    } else if (raw === "true" || raw === "false") {
      edits[path] = raw === "true";
    } else if (!Number.isNaN(Number(raw))) {
      edits[path] = Number(raw);
    } else {
      edits[path] = raw;
    }
  }
  return edits;
}

async function decide(approved) {
  _closeApprovalEditor();
  const note = $("approval-note").value.trim();
  const edits = approved ? _collectEdits() : null;
  $("approval").hidden = true;
  state.pending = null;

  await consume((onEvent) => resumeSession(state.session, approved, note || null, edits, onEvent));
  refreshPanels();
}

// ------------------------------------------------------------------ panels

async function refreshTools() {
  let tools;
  try {
    tools = await api.tools();
  } catch (error) {
    return showInspector("tools", error.message);
  }

  const list = ui.clear($("tools-list"));
  if (!tools.length) {
    list.append(ui.noticeMessage("No tools registered yet."));
    return;
  }

  const handlers = {
    toggle: async (entry) => {
      try {
        await (entry.state === "active" ? api.deactivate(entry.name) : api.activate(entry.name));
      } catch (error) {
        showInspector(entry.name, error.message);
      }
      refreshTools();
    },
    test: async (entry, button) => {
      const label = button.textContent;
      button.textContent = "…";
      button.disabled = true;
      try {
        const report = await api.testTool(entry.name);
        showInspector(`${entry.name} — test`, null, ui.testResult(report));
      } catch (error) {
        showInspector(`${entry.name} — test`, error.message);
      } finally {
        button.textContent = label;
        button.disabled = false;
      }
    },
  };

  for (const entry of tools) list.append(ui.toolRow(entry, handlers));
}

/**
 * Poll the job store, whichever view is open.
 *
 * This runs all the time now, not only while something is running: a run started from the
 * chat has to appear in the rail on its own, and the dot on the Jobs icon is only honest if
 * something is counting. The expensive half — the per-job status call, which is the only
 * source of progress — fires only when something on screen would render it.
 */
async function refreshJobs() {
  clearTimeout(state.jobsTimer);

  // Polling a job store from a tab nobody is looking at is pure load; `visibilitychange`
  // starts it again, so nothing is lost by simply not re-arming here.
  if (document.visibilityState === "hidden") return;

  let jobs;
  try {
    jobs = await api.jobs();
  } catch {
    // The rail keeps its last good render rather than blanking — but it must keep
    // polling. Phase 7's optimistic concurrency answers 409 while the job store is
    // mid-write, which is most likely *just after a run starts*: exactly when someone is
    // watching. Returning without re-arming froze the monitor for good.
    state.jobsTimer = setTimeout(refreshJobs, JOB_POLL_MS);
    return;
  }

  state.jobs = jobs;
  const running = jobs.filter((job) => job.state === "running");

  // A finished run's cached status still says "running"; drop it so the row falls back to
  // the state the list just reported.
  for (const job of jobs) {
    if (job.state !== "running") delete state.jobStatus[job.id];
  }

  if (state.view === "jobs") {
    const statuses = await Promise.all(running.map((job) => api.job(job.id).catch(() => null)));
    running.forEach((job, index) => {
      if (statuses[index]) state.jobStatus[job.id] = statuses[index];
    });
  }

  $("activity-dot").hidden = running.length === 0;
  renderJobsRail();

  state.jobsTimer = setTimeout(refreshJobs, running.length ? JOB_POLL_MS : JOB_IDLE_POLL_MS);
}

const jobHandlers = {
  select: (id) => openJob(id),

  close: () => closeJob(),

  tail: (lines) => {
    state.logTail = lines;
    state.logFollow = true; // asking for more lines means asking to see them
    refreshJobLog();
  },

  follow: (on) => {
    state.logFollow = on;
    if (on) scrollLog();
  },

  analysis: async (id) => {
    try {
      showInspector(`${id} — analysis`, null, ui.analysisReport(await api.jobAnalysis(id)));
    } catch (error) {
      showInspector(`${id} — analysis`, error.message);
    }
  },

  cancel: async (id) => {
    try {
      await api.cancelJob(id);
    } catch (error) {
      showInspector(`${id} — cancel`, error.message);
    }
    refreshJobs();
    if (state.job === id) refreshJobLog();
  },
};

function refreshPanels() {
  refreshTools();
  refreshJobs();
}

function showInspector(title, text, node) {
  $("inspector").hidden = false;
  $("inspector-title").textContent = title;
  const body = ui.clear($("inspector-body"));
  body.append(node || ui.el("pre", { class: "inspector-pre", text: text ?? "" }));
}

// ------------------------------------------------------------------ job log

const scrollLog = () => {
  const pre = $("joblog-body");
  pre.scrollTop = pre.scrollHeight;
};

/** Open `id`'s log in place of the chat. The chat is hidden, never torn down. */
function openJob(id) {
  state.job = id;
  state.logFollow = true;
  state.logError = null;
  $("chat").hidden = true;
  $("joblog").hidden = false;
  $("joblog-body").textContent = "";
  renderJobsRail();

  // Paint the header from what the rail already knows before the first poll returns,
  // otherwise the pane is entirely blank for a round trip.
  renderJobLogHead(state.jobStatus[id] || null);
  refreshJobLog();
}

function closeJob() {
  if (!state.job) return;
  clearTimeout(state.logTimer);
  state.job = null;
  state.logError = null;
  $("joblog").hidden = true;
  $("chat").hidden = false;
  renderJobsRail();
  scrollChat();
  $("composer-input").focus();
}

function renderJobLogHead(status) {
  const job = state.jobs.find((entry) => entry.id === state.job) || { id: state.job, state: "unknown" };
  ui.clear($("joblog-head")).append(
    ui.jobLogHead(job, status, jobHandlers, {
      tail: state.logTail,
      tailChoices: LOG_TAIL_CHOICES,
      follow: state.logFollow,
      error: state.logError,
    }),
  );
}

/**
 * One poll of the open run: its status (for state and progress) and its log tail.
 *
 * There is no streaming endpoint and this phase did not add one — `train.log` is a plain
 * file a detached process appends to, so a tail read on a timer is the whole mechanism.
 */
async function refreshJobLog() {
  clearTimeout(state.logTimer);

  const id = state.job;
  if (!id || document.visibilityState === "hidden") return;

  let status;
  let body;
  try {
    [status, body] = await Promise.all([api.job(id), api.jobLogs(id, state.logTail)]);
  } catch (error) {
    if (state.job !== id) return;
    state.logError = error.message;
    renderJobLogHead(null);
    // A 404 means the run's directory is gone; there is nothing to come back for. Anything
    // else — the retryable 409 the job store answers mid-write, a dropped connection —
    // keeps the last good body on screen and tries again.
    if (error.status !== 404) state.logTimer = setTimeout(refreshJobLog, JOB_POLL_MS);
    return;
  }

  if (state.job !== id) return; // the reader moved on while this was in flight

  state.logError = null;
  renderJobLogHead(status);

  $("joblog-body").textContent = body.log || "(nothing in the log yet)";
  if (state.logFollow) scrollLog();

  if (status.state === "running") state.logTimer = setTimeout(refreshJobLog, JOB_POLL_MS);
}

// ------------------------------------------------------------------ sessions

async function loadSession(name) {
  state.session = name;
  state.pending = null;
  $("approval").hidden = true;

  const log = ui.clear($("chat-log"));

  let body;
  try {
    body = await api.history(name);
  } catch (error) {
    log.append(ui.errorMessage(error.message));
    return;
  }

  for (const message of body.messages || []) {
    if (message.role === "user") {
      log.append(ui.userMessage(message.content));
    } else if (message.role === "tool") {
      log.append(ui.toolResultRow(message.name, message.content));
    } else if (message.role === "assistant") {
      if (message.content) log.append(ui.assistantMessage(message.content).node);
      for (const call of message.tool_calls || []) log.append(ui.toolCallRow(call.name, call.args));
    }
  }

  if (!(body.messages || []).length) {
    log.append(ui.noticeMessage(`New session '${name}'. Ask about your data, your tools, or a fine-tune.`));
  }

  // A refresh in the middle of an approval must not strand the suspended turn: the
  // history carries the pending request precisely so the dialog can come back.
  if (body.pending_approval) {
    state.pending = body.pending_approval;
    showApproval(body.pending_approval);
  }

  scrollChat();
}

async function refreshSessions(select = null) {
  let sessions = [];
  try {
    sessions = await api.sessions();
  } catch {
    /* an empty rail is survivable; the open session still works */
  }

  const current = select || state.session;
  if (current && !sessions.some((s) => s.id === current)) {
    sessions = [{ id: current, updated_at: null, checkpoints: 0 }, ...sessions];
  }

  state.sessions = sessions;
  renderSessions();
}

/** Pure render from `state.sessions` + the needle, so typing in the filter is local. */
function renderSessions() {
  if (state.view !== "sessions") return; // the rail is showing runs; do not clobber it

  const needle = state.filter.sessions.trim().toLowerCase();
  const shown = needle
    ? state.sessions.filter((s) => s.id.toLowerCase().includes(needle))
    : state.sessions;

  const list = ui.clear($("rail-list"));
  if (!shown.length) {
    list.append(ui.noticeMessage(needle ? "No matching sessions." : "No sessions yet."));
    return;
  }

  for (const session of shown) {
    list.append(ui.sessionRow(session, sessionHandlers, session.id === state.session));
  }
}

/** The same, for runs. Both views render into `#rail-list`; only one owns it at a time. */
function renderJobsRail() {
  if (state.view !== "jobs") return;

  const needle = state.filter.jobs.trim().toLowerCase();
  const shown = needle
    ? state.jobs.filter((job) => job.id.toLowerCase().includes(needle))
    : state.jobs;

  const list = ui.clear($("rail-list"));
  if (!shown.length) {
    // There is no "＋ New run" button and never will be — starting a run is a gated action
    // in the chat, so the human sees the exact command first. Say so here, where someone
    // would otherwise go looking for the button.
    list.append(ui.noticeMessage(
      needle
        ? "No matching runs."
        : "No training runs yet. Runs start from the chat, behind the approval gate.",
    ));
    return;
  }

  for (const job of shown) {
    list.append(ui.jobRow(job, state.jobStatus[job.id], jobHandlers, job.id === state.job));
  }
}

/** A mutation mid-turn would race the stream that is writing to the same thread. */
function busyWithATurn() {
  if (state.streaming || state.pending) {
    showInspector("sessions", "Finish the current turn first.");
    return true;
  }
  return false;
}

const sessionHandlers = {
  select: (id) => {
    closeJob(); // picking a conversation is the other way back from a log
    if (id === state.session || busyWithATurn()) return;
    loadSession(id).then(renderSessions);
  },

  rename: async (id) => {
    if (busyWithATurn()) return;

    const next = (window.prompt(`Rename '${id}' to:`, id) || "").trim();
    if (!next || next === id) return;

    try {
      await api.renameSession(id, next);
    } catch (error) {
      return showInspector(`rename ${id}`, error.message);
    }

    // The messages did not move, only the key they are under — so re-point rather than
    // reloading the log from /history.
    if (state.session === id) state.session = next;
    await refreshSessions();
  },

  remove: async (id) => {
    if (busyWithATurn()) return;
    if (!window.confirm(`Delete session '${id}'? This cannot be undone.`)) return;

    try {
      await api.deleteSession(id);
    } catch (error) {
      return showInspector(`delete ${id}`, error.message);
    }

    const wasOpen = state.session === id;
    state.sessions = state.sessions.filter((s) => s.id !== id);

    if (wasOpen) {
      await loadSession(state.sessions[0]?.id || "default");
    }
    await refreshSessions();
  },
};

async function newSession() {
  if (busyWithATurn()) return;

  const name = (window.prompt("Name for the new session:", suggestName()) || "").trim();
  if (!name) return;

  try {
    await api.createSession(name);
  } catch (error) {
    return showInspector("new session", error.message);
  }

  await loadSession(name);
  await refreshSessions(name);
}

// ------------------------------------------------------------------ rail chrome

//: What each activity-bar button does to the rail. An object rather than a `switch`,
//: because `test_ui_contract.py` reads every `case "…"` in this file and asserts the set is
//: exactly the six streamed event names.
const RAIL_VIEWS = {
  sessions: { label: "Sessions", placeholder: "Filter sessions…", render: () => renderSessions() },
  jobs: { label: "Jobs", placeholder: "Filter runs…", render: () => renderJobsRail() },
};

function setView(next) {
  const view = RAIL_VIEWS[next] ? next : "sessions";
  const changed = state.view !== view;
  const chrome = RAIL_VIEWS[view];

  state.view = view;
  localStorage.setItem(RAIL_VIEW_KEY, view);

  $("activity-sessions").setAttribute("aria-selected", String(view === "sessions"));
  $("activity-jobs").setAttribute("aria-selected", String(view === "jobs"));
  $("rail").setAttribute("aria-label", chrome.label);
  $("rail-new").hidden = view !== "sessions"; // there is no "new run" — see `renderJobsRail`

  const filter = $("rail-filter");
  filter.placeholder = chrome.placeholder;
  filter.setAttribute("aria-label", chrome.placeholder);
  filter.value = state.filter[view];

  chrome.render();

  // Switching *to* runs should not show a list up to `JOB_IDLE_POLL_MS` old.
  if (changed && view === "jobs") refreshJobs();
}

function installActivity() {
  const pick = (view) => () => {
    // VS Code's behaviour, and what an icon bar makes people expect: clicking the view you
    // are already on toggles the rail rather than re-rendering it.
    if (state.view === view) {
      setRailCollapsed(!document.body.classList.contains("rail-collapsed"));
      return;
    }
    setRailCollapsed(false); // switching views with the rail shut would show nothing
    setView(view);
  };

  $("activity-sessions").addEventListener("click", pick("sessions"));
  $("activity-jobs").addEventListener("click", pick("jobs"));
}

function setRailCollapsed(collapsed) {
  document.body.classList.toggle("rail-collapsed", collapsed);
  $("rail-toggle").setAttribute("aria-expanded", String(!collapsed));
  localStorage.setItem(RAIL_COLLAPSED_KEY, collapsed ? "1" : "0");
}

function setRailWidth(rem) {
  const clamped = Math.min(RAIL_MAX_REM, Math.max(RAIL_MIN_REM, rem));
  document.documentElement.style.setProperty("--rail-w", `${clamped}rem`);
  localStorage.setItem(RAIL_WIDTH_KEY, String(clamped));
}

function installRail() {
  const stored = parseFloat(localStorage.getItem(RAIL_WIDTH_KEY));
  if (!Number.isNaN(stored)) setRailWidth(stored);
  setRailCollapsed(localStorage.getItem(RAIL_COLLAPSED_KEY) === "1");

  $("rail-toggle").addEventListener("click", () =>
    setRailCollapsed(!document.body.classList.contains("rail-collapsed")));

  $("rail-new").addEventListener("click", newSession);
  $("rail-filter").addEventListener("input", (event) => {
    state.filter[state.view] = event.target.value;
    RAIL_VIEWS[state.view].render();
  });

  const grip = $("rail-grip");
  const rootFontSize = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

  grip.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    grip.setPointerCapture(event.pointerId);
    document.body.classList.add("rail-resizing");

    // Measured from the rail's own left edge, not the viewport's: the activity bar sits to
    // its left, and `clientX` alone reported every width 3rem too wide. The edge cannot
    // move during a drag, so reading it once here is enough.
    const railLeft = $("rail").getBoundingClientRect().left;

    const onMove = (move) => setRailWidth((move.clientX - railLeft) / rootFontSize);
    const onUp = () => {
      grip.removeEventListener("pointermove", onMove);
      grip.removeEventListener("pointerup", onUp);
      document.body.classList.remove("rail-resizing");
    };

    grip.addEventListener("pointermove", onMove);
    grip.addEventListener("pointerup", onUp);
  });

  // Keyboard-reachable too: the grip is focusable, so arrows resize it.
  grip.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const current = parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue("--rail-w")
    ) || 15;
    setRailWidth(current + step);
  });
}

function suggestName() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `web-${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
}

// ------------------------------------------------------------------ boot

async function ensureAuth() {
  try {
    await api.sessions();
    return true;
  } catch (error) {
    if (error.status !== 401) return true; // a different failure; let it surface in context

    const supplied = window.prompt("This server requires an API token (ADAPTRNA_API_TOKEN):");
    if (!supplied) return false;

    token.set(supplied.trim());
    return ensureAuth();
  }
}

/**
 * The health badge, which is silent when there is nothing to say.
 *
 * A permanent "install: ok" is a line of chrome that reports the expected case forever; it
 * earns its place in the topbar only when the install is degraded or the server has gone
 * away — which is also exactly when the doctor report behind it is worth reaching.
 */
async function refreshHealth() {
  const badge = $("health");
  badge.title = "Click for the full doctor report";

  try {
    const body = await api.health();
    // `failed_checks` is the doctor's list of failed check names — and an empty array is
    // truthy in JS, so it has to be counted rather than tested.
    const failed = (body.failed_checks || []).length;

    badge.hidden = body.install === "ok" && !failed;
    if (badge.hidden) return;

    badge.textContent = failed ? `install: ${body.install} (${failed} failed)` : `install: ${body.install}`;
    badge.className = "badge badge-warn";
  } catch {
    badge.hidden = false;
    badge.textContent = "server unreachable";
    badge.className = "badge badge-fail";
  }
}

async function boot() {
  $("composer-send").addEventListener("click", send);
  $("composer-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  $("approval-approve").addEventListener("click", () => decide(true));
  $("approval-decline").addEventListener("click", () => decide(false));
  $("inspector-close").addEventListener("click", () => ($("inspector").hidden = true));
  installRail();
  installActivity();

  // Reading a log usually means reading something that already scrolled past, so following
  // the tail is a consequence of being at the bottom rather than a mode you are stuck in.
  $("joblog-body").addEventListener("scroll", () => {
    const pre = $("joblog-body");
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    if (atBottom === state.logFollow) return;

    state.logFollow = atBottom;
    // Tick the box directly rather than rebuilding the header on a scroll event.
    const box = $("joblog-head").querySelector(".joblog-follow input");
    if (box) box.checked = atBottom;
  });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") return;
    refreshJobs();
    if (state.job) refreshJobLog();
  });

  $("health").addEventListener("click", async () => {
    try {
      const report = await api.doctor();
      showInspector("doctor", JSON.stringify(report, null, 2));
    } catch (error) {
      showInspector("doctor", error.message);
    }
  });

  await refreshHealth();
  if (!(await ensureAuth())) {
    $("chat-log").append(ui.errorMessage("A token is required to use this server."));
    return;
  }

  const sessions = await api.sessions().catch(() => []);
  const initial =
    new URLSearchParams(location.search).get("session") || sessions[0]?.id || "default";

  // loadSession first: `state.session` is what marks the current row in the rail.
  await loadSession(initial);
  await refreshSessions(initial);
  refreshPanels();
  // After the sessions are cached, so booting straight into the Jobs view still leaves a
  // populated rail behind it.
  setView(localStorage.getItem(RAIL_VIEW_KEY));

  $("composer-input").focus();
}

boot();
