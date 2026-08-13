/**
 * DOM construction for every part of the app that is not Markdown.
 *
 * Model output, tool results and job logs all flow through here, and `el` never assigns
 * `innerHTML`, so escaping is a property of how the nodes are made rather than something
 * each call site has to remember.
 */

import { clear, el } from "./dom.js";
import { renderMarkdown } from "./md.js";

export { clear, el };

const number = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? String(Math.round(value * 1e6) / 1e6)
    : String(value);

// ------------------------------------------------------------------ chat

export function userMessage(text) {
  return el("div", { class: "msg msg-user" }, el("div", { class: "msg-body", text }));
}

/**
 * An assistant bubble. `set(markdown)` re-renders it from the accumulated text.
 *
 * Re-rendering rather than appending is what makes streaming Markdown work at all: a
 * table or a code fence only becomes meaningful once its later lines have arrived, so
 * there is nothing to append *to* until the block is complete.
 */
export function assistantMessage(text = "") {
  const body = el("div", { class: "msg-body md" });
  const node = el("div", { class: "msg msg-assistant" }, body);
  node.dataset.role = "assistant";

  const set = (markdown) => body.replaceChildren(renderMarkdown(markdown));
  if (text) set(text);

  return { node, body, set };
}

export function toolCallRow(name, args) {
  return el(
    "details",
    { class: "tool-line tool-call" },
    el("summary", {}, el("span", { class: "tool-glyph", text: "⚙" }), el("code", { text: name })),
    el("pre", { class: "tool-payload", text: JSON.stringify(args ?? {}, null, 2) }),
  );
}

export function toolResultRow(name, content) {
  const text = String(content ?? "");
  const oneLine = text.replace(/\s+/g, " ").trim();
  const preview = oneLine.length > 90 ? `${oneLine.slice(0, 90)}…` : oneLine;

  return el(
    "details",
    { class: "tool-line tool-result" },
    el(
      "summary",
      {},
      el("span", { class: "tool-glyph", text: "=" }),
      el("code", { text: name || "result" }),
      el("span", { class: "tool-preview", text: preview }),
    ),
    el("pre", { class: "tool-payload", text }),
  );
}

export function errorMessage(text) {
  return el("div", { class: "msg msg-error" }, el("div", { class: "msg-body", text }));
}

export function noticeMessage(text) {
  return el("div", { class: "msg msg-notice" }, el("div", { class: "msg-body", text }));
}

// ------------------------------------------------------------------ sessions

const MINUTE = 60000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** Coarse on purpose: the rail needs "which of these is recent", not a duration. */
export function relativeTime(iso) {
  if (!iso) return "";

  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";

  const ago = Date.now() - then;
  if (ago < MINUTE) return "just now";
  if (ago < HOUR) return `${Math.floor(ago / MINUTE)}m ago`;
  if (ago < DAY) return `${Math.floor(ago / HOUR)}h ago`;
  if (ago < 2 * DAY) return "yesterday";
  if (ago < 7 * DAY) return `${Math.floor(ago / DAY)}d ago`;

  return new Date(then).toLocaleDateString();
}

export function sessionRow(session, handlers, isCurrent) {
  return el(
    "div",
    { class: `session-row ${isCurrent ? "is-current" : ""}` },
    el(
      "button",
      { class: "session-open", type: "button", title: session.id,
        onclick: () => handlers.select(session.id) },
      el("span", { class: "session-name", text: session.id }),
      el("span", { class: "session-when", text: relativeTime(session.updated_at) }),
    ),
    el(
      "div",
      { class: "session-actions" },
      el("button", {
        class: "btn btn-quiet", text: "✎", title: `Rename '${session.id}'`,
        onclick: () => handlers.rename(session.id),
      }),
      el("button", {
        class: "btn btn-quiet", text: "🗑", title: `Delete '${session.id}'`,
        onclick: () => handlers.remove(session.id),
      }),
    ),
  );
}

// ------------------------------------------------------------------ tools

export function toolRow(entry, handlers) {
  const active = entry.state === "active";

  const row = el(
    "div",
    { class: `panel-row tool-row ${active ? "is-active" : "is-disabled"}` },
    el(
      "div",
      { class: "panel-row-main" },
      el("span", { class: `dot ${active ? "dot-on" : "dot-off"}`, title: entry.state }),
      el("span", { class: "panel-name", text: entry.name }),
      el("span", { class: "panel-tag", text: entry.type }),
    ),
    el(
      "div",
      { class: "panel-row-actions" },
      el("button", {
        class: "btn btn-quiet",
        text: active ? "disable" : "enable",
        title: active ? "Stop routing to this tool" : "Route to this tool again",
        onclick: () => handlers.toggle(entry),
      }),
      el("button", {
        class: "btn btn-quiet",
        text: "test",
        title: "Run this tool's smoke / golden test",
        onclick: (event) => handlers.test(entry, event.target),
      }),
    ),
  );

  if (entry.description) row.title = entry.description;
  return row;
}

export function testResult(report) {
  const passed = report.passed ?? report.ok ?? report.status === "ok";
  return el(
    "div",
    { class: `test-result ${passed ? "is-pass" : "is-fail"}` },
    el("strong", { text: passed ? "pass" : "fail" }),
    " ",
    el("span", { text: summarizeReport(report) }),
  );
}

function summarizeReport(report) {
  for (const key of ["message", "detail", "error", "summary"]) {
    if (typeof report[key] === "string" && report[key]) return report[key];
  }
  return JSON.stringify(report);
}

// ------------------------------------------------------------------ jobs

const JOB_STATES = { running: "is-running", succeeded: "is-ok", failed: "is-fail" };

export function jobRow(job, status, handlers) {
  const state = (status && status.state) || job.state;
  const progress = status && status.progress;

  const body = el(
    "div",
    { class: "panel-row-main" },
    el("span", { class: `dot job-dot ${JOB_STATES[state] || ""}`, title: state }),
    el("span", { class: "panel-name", text: job.id }),
    el("span", { class: "panel-tag", text: state }),
  );

  const lines = [];
  if (progress) {
    const marks = [];
    if (progress.epoch !== null && progress.epoch !== undefined) marks.push(`epoch ${progress.epoch}`);
    if (progress.step !== null && progress.step !== undefined) marks.push(`step ${progress.step}`);
    if (marks.length) lines.push(el("div", { class: "job-progress", text: marks.join(" · ") }));

    const metrics = Object.entries(progress.latest_metrics || {});
    if (metrics.length) {
      lines.push(
        el(
          "div",
          { class: "job-metrics" },
          metrics.map(([key, value]) =>
            el("span", { class: "metric" }, el("span", { class: "metric-key", text: key }), " ",
               el("span", { class: "metric-value", text: number(value) })),
          ),
        ),
      );
    }
  }

  const actions = el(
    "div",
    { class: "panel-row-actions" },
    el("button", { class: "btn btn-quiet", text: "log", onclick: () => handlers.logs(job.id) }),
    state === "succeeded" || state === "failed"
      ? el("button", { class: "btn btn-quiet", text: "analysis",
                       onclick: () => handlers.analysis(job.id) })
      : null,
    state === "running"
      ? el("button", { class: "btn btn-quiet btn-danger", text: "cancel",
                       onclick: () => handlers.cancel(job.id) })
      : null,
  );

  return el("div", { class: "panel-row job-row" }, body, ...lines, actions);
}

export function analysisReport(report) {
  const verdict = report.verdict || "unknown";

  return el(
    "div",
    { class: "analysis" },
    el(
      "div",
      { class: "analysis-head" },
      el("span", { class: `verdict verdict-${verdict}`, text: verdict }),
      report.primary_metric
        ? el("span", { class: "analysis-primary",
                       text: `${report.primary_metric} = ${number(report.primary_value)}` })
        : null,
      report.truncated ? el("span", { class: "chip chip-warn", text: "truncated" }) : null,
    ),
    (report.checks || []).length
      ? el("ul", { class: "analysis-list" },
           (report.checks || []).map((line) => el("li", { text: line })))
      : null,
    (report.suggestions || []).length
      ? el("ul", { class: "analysis-list analysis-suggestions" },
           (report.suggestions || []).map((line) => el("li", { text: line })))
      : null,
  );
}

// ------------------------------------------------------------------ approval

/**
 * The gate, rendered.
 *
 * Field for field and in the same order as the terminal's prompt (`cli/chat.py`
 * `_prompt_approval`) — including joining `command` on a single space, so what the
 * browser shows is byte-identical to what the terminal shows. The whole value of this
 * dialog is that the human sees what will actually run; a summary here would quietly
 * undo the gates built in Phases 5–8.
 */
export function approvalBody(request) {
  const items = (request.requests || []).map((item) => {
    const details = item.details || {};
    const rows = [];

    const line = (label, value, extra = {}) =>
      el("div", { class: "approval-line", ...extra },
         el("span", { class: "approval-label", text: label }),
         el("code", { class: "approval-value", text: value }));

    if (details.tool) {
      rows.push(line("tool", details.tool));
      rows.push(line("state", `${details.current_state} → ${details.after_approval}`));
    }
    if (details.command) rows.push(line("would run", details.command.join(" "), { class: "approval-line approval-command" }));
    if (details.output_dir) rows.push(line("output", details.output_dir));
    if (details.downloads) rows.push(line("note", details.downloads));

    for (const file of details.files || []) {
      rows.push(line("write", `${file.path}  (${file.lines} lines)`));
    }

    if (details.staging_path) {
      rows.push(line("staged in", details.staging_path));
      rows.push(el("div", { class: "approval-hint",
                            text: "open it in your editor to read the code before approving" }));
    }

    for (const warning of details.warnings || []) {
      rows.push(el("div", { class: "approval-warning" },
                   el("span", { class: "approval-bang", text: "!" }),
                   el("span", { text: warning })));
    }

    return el("div", { class: "approval-item" },
              el("div", { class: "approval-summary", text: item.summary || item.tool }),
              ...rows);
  });

  return el("div", { class: "approval-items" }, items);
}
