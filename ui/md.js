/**
 * A small Markdown renderer — enough for what the assistant actually writes.
 *
 * The model answers in Markdown, and it leans on tables hard (an analysis verdict or a
 * config recommendation is nearly always one), so rendering replies as plain text made
 * the main reading surface noticeably worse. A library would mean either a CDN — which
 * breaks the offline promise — or vendoring a bundle into a repo that deliberately has no
 * build step, so this covers the constructs that show up and ignores the rest.
 *
 * Everything is built as DOM nodes and text; no HTML is ever parsed from model output, so
 * this cannot inject markup no matter what comes back.
 *
 * Supported: headings, fenced and inline code, bold/italic, links (as text), unordered and
 * ordered lists, tables, blockquotes, horizontal rules. Anything else falls through as a
 * paragraph, which is the right failure mode for a renderer you cannot fully trust.
 */

import { el } from "./render.js";

export function renderMarkdown(source) {
  const fragment = document.createDocumentFragment();
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    // fenced code
    const fence = line.match(/^\s*```(\w*)\s*$/);
    if (fence) {
      const body = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1; // closing fence (or end of input, mid-stream)
      fragment.append(el("pre", { class: "md-code" },
                         el("code", { text: body.join("\n") })));
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const tag = `h${Math.min(heading[1].length + 2, 6)}`; // h1 in a bubble is too loud
      fragment.append(inlineInto(el(tag, { class: "md-h" }), heading[2]));
      index += 1;
      continue;
    }

    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      fragment.append(el("hr", { class: "md-hr" }));
      index += 1;
      continue;
    }

    // table: a header row followed by a |---|---| separator
    if (line.includes("|") && index + 1 < lines.length && isSeparator(lines[index + 1])) {
      const rows = [];
      const header = splitRow(line);
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(splitRow(lines[index]));
        index += 1;
      }
      fragment.append(table(header, rows));
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const body = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        body.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      fragment.append(inlineInto(el("blockquote", { class: "md-quote" }), body.join(" ")));
      continue;
    }

    const bullet = /^\s*[-*+]\s+/;
    const numbered = /^\s*\d+[.)]\s+/;
    if (bullet.test(line) || numbered.test(line)) {
      const ordered = numbered.test(line) && !bullet.test(line);
      const pattern = ordered ? numbered : bullet;
      const list = el(ordered ? "ol" : "ul", { class: "md-list" });

      while (index < lines.length && pattern.test(lines[index])) {
        let item = lines[index].replace(pattern, "");
        index += 1;
        // continuation lines belong to the item they are indented under
        while (index < lines.length && lines[index].trim()
               && !pattern.test(lines[index]) && /^\s{2,}/.test(lines[index])) {
          item += ` ${lines[index].trim()}`;
          index += 1;
        }
        list.append(inlineInto(el("li"), item));
      }

      fragment.append(list);
      continue;
    }

    // paragraph: consecutive plain lines
    const body = [];
    while (index < lines.length && lines[index].trim()
           && !/^\s*```/.test(lines[index])
           && !/^(#{1,6})\s/.test(lines[index])
           && !bullet.test(lines[index]) && !numbered.test(lines[index])
           && !/^\s*>\s?/.test(lines[index])
           && !(lines[index].includes("|") && index + 1 < lines.length
                && isSeparator(lines[index + 1]))) {
      body.push(lines[index]);
      index += 1;
    }
    fragment.append(inlineInto(el("p", { class: "md-p" }), body.join("\n")));
  }

  return fragment;
}

const isSeparator = (line) => /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(line) && line.includes("-");

function splitRow(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

function table(header, rows) {
  const head = el("tr", {}, header.map((cell) => inlineInto(el("th"), cell)));
  const body = rows.map((row) =>
    el("tr", {}, header.map((_, column) => inlineInto(el("td"), row[column] ?? ""))));

  // Wide tables scroll inside their own box rather than stretching the chat column.
  return el("div", { class: "md-table-wrap" },
            el("table", { class: "md-table" },
               el("thead", {}, head), el("tbody", {}, body)));
}

/**
 * Inline spans: `code`, **bold**, *italic*, [text](url), ~~strike~~.
 *
 * Code is taken first and its contents are never re-scanned, so `**not bold**` inside
 * backticks stays literal.
 */
function inlineInto(node, text) {
  const pattern = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(~~[^~]+~~)|(\*[^*\n]+\*)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) node.append(text.slice(last, match.index));
    const token = match[0];

    if (token.startsWith("`")) {
      node.append(el("code", { class: "md-inline-code", text: token.slice(1, -1) }));
    } else if (token.startsWith("**") || token.startsWith("__")) {
      node.append(el("strong", { text: token.slice(2, -2) }));
    } else if (token.startsWith("~~")) {
      node.append(el("del", { text: token.slice(2, -2) }));
    } else if (token.startsWith("[")) {
      // Rendered as text, not a live link: nothing in a model reply should be one click
      // away from navigating this page somewhere.
      const label = token.slice(1, token.indexOf("]"));
      const href = token.slice(token.indexOf("](") + 2, -1);
      node.append(el("span", { class: "md-link", title: href, text: label }));
    } else {
      node.append(el("em", { text: token.slice(1, -1) }));
    }

    last = match.index + token.length;
  }

  if (last < text.length) node.append(text.slice(last));
  return node;
}
