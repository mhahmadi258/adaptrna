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

const state = {
  session: null,
  pending: null, // the approval request the graph is suspended on, if any
  streaming: false,
  jobsTimer: null,
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
  $("thinking").hidden = !busy;
  if (!busy) $("composer-input").focus();
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
  };

  await run(({ event, data }) => {
    switch (event) {
      case "text":
        if (!bubble) {
          bubble = ui.assistantMessage();
          markdown = "";
          painted = 0; // so the first token of a new bubble paints immediately
          append(bubble.node);
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
    /* an empty picker is survivable; the session still works */
  }

  const current = select || state.session;
  if (current && !sessions.includes(current)) sessions = [current, ...sessions];

  const picker = ui.clear($("session-picker"));
  for (const name of sessions) {
    picker.append(ui.el("option", { value: name, selected: name === current }, name));
  }
}

async function newSession() {
  const name = (window.prompt("Name for the new session:", suggestName()) || "").trim();
  if (!name) return;

  await refreshSessions(name);
  await loadSession(name);
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
  $("session-new").addEventListener("click", newSession);
  $("session-picker").addEventListener("change", (event) => loadSession(event.target.value));
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
  const initial = new URLSearchParams(location.search).get("session") || sessions[0] || "default";

  await refreshSessions(initial);
  await loadSession(initial);
  refreshPanels();

  $("composer-input").focus();
}

boot();
