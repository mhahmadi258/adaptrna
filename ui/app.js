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

const $ = (id) => document.getElementById(id);

const JOB_POLL_MS = 3000;

//: Markdown is re-parsed from the whole reply on each repaint, so tokens are coalesced
//: into frames rather than repainting per delta.
const MARKDOWN_REPAINT_MS = 80;

//: Rail geometry is chrome for *this browser*, not server state, so it belongs in
//: localStorage — deliberately unlike the auth token, which uses sessionStorage so it dies
//: with the tab. A width that reset on every tab would just be an annoyance.
const RAIL_WIDTH_KEY = "adaptrna.rail.width";
const RAIL_COLLAPSED_KEY = "adaptrna.rail.collapsed";
const RAIL_MIN_REM = 10;
const RAIL_MAX_REM = 30;

const state = {
  session: null,
  pending: null, // the approval request the graph is suspended on, if any
  streaming: false,
  jobsTimer: null,
  sessions: [], // the last fetched list, so filtering does not need a round trip
  filter: "",
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
}

async function decide(approved) {
  const note = $("approval-note").value.trim();
  $("approval").hidden = true;
  state.pending = null;

  await consume((onEvent) => resumeSession(state.session, approved, note || null, onEvent));
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

async function refreshJobs() {
  clearTimeout(state.jobsTimer);

  let jobs;
  try {
    jobs = await api.jobs();
  } catch {
    // The panel keeps its last good render rather than blanking — but it must keep
    // polling. Phase 7's optimistic concurrency answers 409 while the job store is
    // mid-write, which is most likely *just after a run starts*: exactly when someone is
    // watching this panel. Returning without re-arming froze the monitor for good.
    state.jobsTimer = setTimeout(refreshJobs, JOB_POLL_MS);
    return;
  }

  // `list` is cheap and has no progress; only running jobs are worth a second call.
  const statuses = await Promise.all(
    jobs.map((job) => (job.state === "running" ? api.job(job.id).catch(() => null) : null)),
  );

  const list = ui.clear($("jobs-list"));
  if (!jobs.length) {
    list.append(ui.noticeMessage("No training runs yet."));
  }

  const handlers = {
    logs: async (id) => {
      try {
        const body = await api.jobLogs(id, 200);
        showInspector(`${id} — log`, body.log || "(empty)");
      } catch (error) {
        showInspector(`${id} — log`, error.message);
      }
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
    },
  };

  jobs.forEach((job, index) => list.append(ui.jobRow(job, statuses[index], handlers)));

  if (jobs.some((job) => job.state === "running")) {
    state.jobsTimer = setTimeout(refreshJobs, JOB_POLL_MS);
  }
}

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

/** Pure render from `state.sessions` + `state.filter`, so typing in the filter is local. */
function renderSessions() {
  const needle = state.filter.trim().toLowerCase();
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
    state.filter = event.target.value;
    renderSessions();
  });

  const grip = $("rail-grip");
  const rootFontSize = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

  grip.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    grip.setPointerCapture(event.pointerId);
    document.body.classList.add("rail-resizing");

    const onMove = (move) => setRailWidth(move.clientX / rootFontSize);
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

async function refreshHealth() {
  const badge = $("health");
  try {
    const body = await api.health();
    // `failed_checks` is the doctor's list of failed check names — and an empty array is
    // truthy in JS, so it has to be counted rather than tested.
    const failed = (body.failed_checks || []).length;
    badge.textContent = failed ? `install: ${body.install} (${failed} failed)` : `install: ${body.install}`;
    badge.className = `badge ${body.install === "ok" ? "badge-ok" : "badge-warn"}`;
    badge.title = failed ? "Click for the full doctor report" : "Install healthy";
  } catch {
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

  $("composer-input").focus();
}

boot();
