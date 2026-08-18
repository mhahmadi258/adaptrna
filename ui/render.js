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

//: Every state `jobs/store.py` can hold. `cancelled` was missing until Phase 11, so a
//: cancelled run drew a colourless dot and looked indistinguishable from one still queued.
const JOB_STATES = {
  running: "is-running",
  succeeded: "is-ok",
  failed: "is-fail",
  cancelled: "is-fail",
};

const isTerminal = (state) => state === "succeeded" || state === "failed" || state === "cancelled";

/** Epoch/step and the latest metric chips — shared so the rail row and the log header agree. */
export function progressLines(progress) {
  if (!progress) return [];

  const lines = [];
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

  return lines;
}

export function jobRow(job, status, handlers, isCurrent) {
  const state = (status && status.state) || job.state;

  const body = el(
    "div",
    { class: "panel-row-main" },
    el("span", { class: `dot job-dot ${JOB_STATES[state] || ""}`, title: state }),
    // Opening the log *is* selecting the row, so the id is the button — the same shape as
    // `sessionRow`'s `.session-open`.
    el("button", { class: "job-open", type: "button", title: job.id, text: job.id,
                   onclick: () => handlers.select(job.id) }),
    el("span", { class: "panel-tag", text: state }),
  );

  const actions = el(
    "div",
    { class: "panel-row-actions" },
    isTerminal(state)
      ? el("button", { class: "btn btn-quiet", text: "analysis",
                       onclick: () => handlers.analysis(job.id) })
      : null,
    state === "running"
      ? el("button", { class: "btn btn-quiet btn-danger", text: "cancel",
                       onclick: () => handlers.cancel(job.id) })
      : null,
  );

  return el(
    "div",
    { class: `panel-row job-row ${isCurrent ? "is-current" : ""}` },
    body,
    ...progressLines(status && status.progress),
    actions,
  );
}

/**
 * The header above a run's log: what it is, how far it got, and what you can do to it.
 *
 * Rebuilt on every poll rather than patched, for the same reason the assistant bubble is
 * re-rendered rather than appended to — the state, the progress and which actions are legal
 * all change together, and a half-updated header is worse than a rebuilt one.
 */
export function jobLogHead(job, status, handlers, options = {}) {
  const state = (status && status.state) || job.state;

  const tail = el(
    "select",
    { class: "joblog-tail", "aria-label": "How many lines to show",
      onchange: (event) => handlers.tail(Number(event.target.value)) },
    (options.tailChoices || []).map((lines) =>
      el("option", { value: String(lines), selected: lines === options.tail, text: `tail ${lines}` })),
  );

  const follow = el(
    "label",
    { class: "joblog-follow", title: "Keep the newest line in view" },
    el("input", { type: "checkbox", checked: options.follow,
                  onchange: (event) => handlers.follow(event.target.checked) }),
    "follow",
  );

  return el(
    "div",
    {},
    el(
      "div",
      { class: "joblog-title" },
      el("span", { class: `dot job-dot ${JOB_STATES[state] || ""}`, title: state }),
      el("span", { class: "joblog-name", text: job.id, title: job.id }),
      el("span", { class: "joblog-state", text: state }),
    ),
    ...progressLines(status && status.progress),
    el(
      "div",
      { class: "joblog-actions" },
      tail,
      follow,
      isTerminal(state)
        ? el("button", { class: "btn btn-quiet", text: "analysis",
                         onclick: () => handlers.analysis(job.id) })
        : null,
      state === "running"
        ? el("button", { class: "btn btn-quiet btn-danger", text: "cancel",
                         onclick: () => handlers.cancel(job.id) })
        : null,
      el("button", { class: "btn btn-quiet btn-close", text: "×", title: "Back to the chat",
                     onclick: () => handlers.close() }),
    ),
    options.error ? el("div", { class: "joblog-error", text: options.error }) : null,
  );
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
// WeakMap so file data survives without polluting DOM attributes with large strings.
const _editorFiles = new WeakMap();

/** `details.spec` (gate 1, Phase 13 §4), field by field — same content and order as the
 * terminal's `_print_spec_block` (`cli/chat.py`). */
function specSummary(spec) {
  const line = (label, value) =>
    el("div", { class: "approval-line" },
       el("span", { class: "approval-label", text: label }),
       el("code", { class: "approval-value", text: value }));

  const rows = [];
  const fmt = spec.format || {};
  rows.push(line("file", `${spec.path} (${fmt.rows ?? "?"} rows)`));

  const length = spec.length || {};
  rows.push(line(
    "sequence",
    `${spec.sequence_column} — ${length.median ?? "?"} nt `
    + `(min ${length.min ?? "?"}, max ${length.max ?? "?"}), ${spec.alphabet}`,
  ));

  let labelValue = `${spec.label_column} — ${spec.target_type}`;
  if (spec.class_counts) {
    labelValue += " — " + Object.entries(spec.class_counts).map(([k, v]) => `${k}: ${v}`).join(" · ");
    if (spec.positive_class) labelValue += ` (positive: '${spec.positive_class}')`;
  } else if (spec.target_summary) {
    const s = spec.target_summary;
    labelValue += ` — min ${s.min}, max ${s.max}, mean ${s.mean}`;
  }
  rows.push(line("label", labelValue));

  if (spec.ignored_columns && spec.ignored_columns.length) {
    rows.push(line("ignored", spec.ignored_columns.join(", ")));
  }

  const split = spec.split || {};
  const counts = split.row_counts || {};
  const countsStr = ["train", "val", "test"].map((k) => counts[k] ?? "?").join(" / ");
  if (split.mode === "random") {
    const fractions = split.fractions || {};
    const pct = (k) => Math.round((fractions[k] || 0) * 100);
    rows.push(line(
      "split",
      `random ${pct("train")}/${pct("val")}/${pct("test")}, seed ${split.seed} → ${countsStr}`,
    ));
  } else if (split.mode === "column") {
    rows.push(line("split", `column '${split.column}' → ${countsStr}`));
  }

  const head = spec.head || {};
  rows.push(line("head", `${head.kind ?? "?"} · ${head.loss ?? "?"} · ${head.primary_metric ?? "?"}`));
  rows.push(line("task", spec.task_name));

  return el("div", { class: "approval-spec-summary" }, rows);
}

/** One editable input, its `data-edit-path` matching a whitelisted entry in
 * `tool_factory.py`'s `EDITABLE_ARGS["confirm_data_profile"]`. */
function specField(path, label, value, { json = false } = {}) {
  const stringValue = value === null || value === undefined ? "" : String(value);
  const tag = json ? "textarea" : "input";
  const props = {
    class: "approval-edit-input" + (json ? " approval-edit-json" : ""),
    "data-edit-path": path,
  };
  if (json) props["data-edit-json"] = "true";

  const control = json
    ? el(tag, { ...props, text: value ? JSON.stringify(value) : "" })
    : el(tag, { ...props, value: stringValue });

  return el("div", { class: "approval-edit-row" },
            el("span", { class: "approval-label", text: label }), control);
}

/** The spec/plan edit form (Phase 13 §5): one row per editable field, pre-filled with the
 * profiler's proposal. `app.js::_collectEdits` reads back whichever inputs the human
 * actually changed. */
function specForm(spec) {
  const split = spec.split || {};
  const fractions = split.fractions || {};

  const rows = [
    specField("spec.sequence_column", "sequence_column", spec.sequence_column),
    specField("spec.label_column", "label_column", spec.label_column),
    specField("spec.target_type", "target_type", spec.target_type),
    specField("spec.task_name", "task_name", spec.task_name),
    specField("spec.tool_description", "tool_description", spec.tool_description),
    specField("spec.positive_class", "positive_class", spec.positive_class),
    specField("spec.on_invalid", "on_invalid", spec.on_invalid),
    specField("spec.split.mode", "split.mode", split.mode),
    specField("spec.split.column", "split.column", split.column),
    specField("spec.split.seed", "split.seed", split.seed),
    specField("spec.split.stratify", "split.stratify", split.stratify),
    specField("spec.split.fractions.train", "split.fractions.train", fractions.train),
    specField("spec.split.fractions.val", "split.fractions.val", fractions.val),
    specField("spec.split.fractions.test", "split.fractions.test", fractions.test),
    specField("spec.split.mapping", "split.mapping (JSON)", split.mapping, { json: true }),
  ];

  return el("details", { class: "approval-spec-form" },
            el("summary", { text: "edit fields" }), ...rows);
}

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
    if (details.spec) {
      rows.push(specSummary(details.spec));
      rows.push(specForm(details.spec));
    }
    if (details.command) rows.push(line("would run", details.command.join(" "), { class: "approval-line approval-command" }));
    if (details.output_dir) rows.push(line("output", details.output_dir));
    if (details.downloads) rows.push(line("note", details.downloads));

    const files = details.files || [];
    const hasContent = files.length > 0 && files[0].content != null;

    if (hasContent) {
      // Show a tabbed code viewer instead of plain "write …" lines.
      const editorMount = el("div", { class: "approval-code-editor" });
      _editorFiles.set(editorMount, files);

      const tabBar = el("div", { class: "approval-tabs" },
        ...files.map((f, i) => {
          const name = f.path.split("/").pop();
          return el("button", {
            class: "approval-tab" + (i === 0 ? " approval-tab-active" : ""),
            type: "button",
            "data-index": String(i),
          }, name);
        })
      );

      rows.push(el("div", { class: "approval-code-panel" }, tabBar, editorMount));
    } else {
      for (const file of files) {
        rows.push(line("write", `${file.path}  (${file.lines} lines)`));
      }
    }

    if (details.staging_path) {
      rows.push(line("staged in", details.staging_path));
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

/** Retrieve the file list stashed on an editor mount element. */
export function editorMountFiles(mount) {
  return _editorFiles.get(mount) || null;
}
