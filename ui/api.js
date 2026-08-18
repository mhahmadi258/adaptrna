/**
 * The Phase 8 endpoints, as functions.
 *
 * Same-origin, so on loopback there is nothing to authenticate. When the server was
 * started with ADAPTRNA_API_TOKEN the UI asks for it once and keeps it in
 * `sessionStorage` — not `localStorage`, so it dies with the tab rather than outliving
 * the session on disk.
 */

import { streamEvents } from "./sse.js";

const TOKEN_KEY = "adaptrna.token";

export const token = {
  get: () => sessionStorage.getItem(TOKEN_KEY) || "",
  set: (value) => sessionStorage.setItem(TOKEN_KEY, value),
  clear: () => sessionStorage.removeItem(TOKEN_KEY),
};

function authHeaders(extra = {}) {
  const headers = { ...extra };
  const value = token.get();
  if (value) headers["Authorization"] = `Bearer ${value}`;
  return headers;
}

async function request(path, { method = "GET", body } = {}) {
  const sending = body !== undefined;

  const response = await fetch(path, {
    method,
    headers: authHeaders(sending ? { "Content-Type": "application/json" } : {}),
    body: sending ? JSON.stringify(body) : undefined,
  });

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { error: text };
    }
  }

  if (!response.ok) {
    // Phase 7 standardised these messages so a second front end would not need its own
    // wording: what the browser shows is what the terminal prints.
    const error = new Error((payload && payload.error) || `HTTP ${response.status}`);
    error.status = response.status;
    error.retryable = Boolean(payload && payload.retryable);
    throw error;
  }

  return payload;
}

const path = (...parts) => parts.map(encodeURIComponent).join("/");

export const api = {
  health: () => request("/health"),
  doctor: () => request("/api/doctor"),

  tools: () => request("/api/tools"),
  tool: (name) => request(`/api/tools/${path(name)}`),
  activate: (name) => request(`/api/tools/${path(name)}/activate`, { method: "POST" }),
  deactivate: (name) => request(`/api/tools/${path(name)}/deactivate`, { method: "POST" }),
  testTool: (name) => request(`/api/tools/${path(name)}/test`, { method: "POST" }),
  predict: (name, sequences) =>
    request(`/api/tools/${path(name)}/predict`, { method: "POST", body: { sequences } }),

  jobs: () => request("/api/jobs"),
  job: (id) => request(`/api/jobs/${path(id)}`),
  jobLogs: (id, tail = 60) => request(`/api/jobs/${path(id)}/logs?tail=${tail}`),
  jobAnalysis: (id) => request(`/api/jobs/${path(id)}/analysis`),
  cancelJob: (id) => request(`/api/jobs/${path(id)}/cancel`, { method: "POST" }),

  // `[{id, updated_at, checkpoints}]`, newest first — the rail renders it in that order.
  sessions: () => request("/api/sessions"),
  history: (id) => request(`/api/sessions/${path(id)}/history`),
  createSession: (id) => request("/api/sessions", { method: "POST", body: { id } }),
  renameSession: (id, next) =>
    request(`/api/sessions/${path(id)}`, { method: "PATCH", body: { id: next } }),
  deleteSession: (id) => request(`/api/sessions/${path(id)}`, { method: "DELETE" }),
};

/** One turn. Ends on `done`, or on `approval_required` — which suspends it. */
export function sendMessage(session, text, onEvent) {
  return streamEvents(
    `/api/sessions/${path(session)}/messages`,
    {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text }),
    },
    onEvent,
  );
}

/**
 * Answer a pending approval. The reply is a *new* stream continuing the same turn, which
 * is why callers consume it with the same handler they used for `sendMessage`.
 *
 * `edits` (Phase 13 §5) is a dotted-path -> value object collected from the approval
 * modal's spec/plan form — absent or `{}` means "as proposed", same as no `edits` key at
 * all in the request body.
 */
export function resumeSession(session, approved, note, edits, onEvent) {
  const body = { approved };
  if (note) body.note = note;
  if (edits && Object.keys(edits).length) body.edits = edits;

  return streamEvents(
    `/api/sessions/${path(session)}/resume`,
    {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    },
    onEvent,
  );
}
