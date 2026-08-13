/**
 * Server-Sent Events, read off a `fetch` body.
 *
 * The browser's own `EventSource` is GET-only, and a chat turn is a POST carrying a JSON
 * body — so the frames get parsed here instead. That is this phase's one piece of real
 * client logic, which is why the parser is a pure function of text and the transport is a
 * thin wrapper around it: the hard part is testable without a network.
 *
 * Frames arrive split across arbitrary chunk boundaries, so the parser is a stateful
 * buffer rather than a `split("\n\n")` over one string.
 */

/** One SSE block -> {event, data}, or null if it carried no event name. */
export function parseFrame(block) {
  let event = null;
  const dataLines = [];

  for (const raw of block.split("\n")) {
    const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (line === "" || line.startsWith(":")) continue; // blank, or a keep-alive comment

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }

  if (event === null) return null;

  // The spec joins multiple data lines with newlines. Our server emits single-line JSON,
  // but a payload that ever grows a newline should not silently become unparseable.
  const text = dataLines.join("\n");
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }
  }

  return { event, data };
}

/** A parser that survives chunk boundaries: feed it text, get whole frames back. */
export function createFrameParser() {
  let buffer = "";

  return function push(chunk) {
    buffer += chunk;

    const frames = [];
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const frame = parseFrame(buffer.slice(0, index));
      buffer = buffer.slice(index + 2);
      if (frame) frames.push(frame);
    }

    return frames;
  };
}

/**
 * POST and consume the SSE response, calling `onEvent({event, data})` per frame.
 *
 * A failed request is reported as an `error` frame rather than thrown, so callers have
 * one place to handle "the turn went wrong" instead of two.
 */
export async function streamEvents(url, options, onEvent) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (cause) {
    onEvent({ event: "error", data: { message: `cannot reach the server (${cause.message})` } });
    return;
  }

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.error) message = body.error;
    } catch {
      /* a non-JSON error body tells us nothing more than the status */
    }
    onEvent({ event: "error", data: { message, status: response.status } });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const push = createFrameParser();

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const frame of push(decoder.decode(value, { stream: true }))) onEvent(frame);
  }

  for (const frame of push(decoder.decode())) onEvent(frame);
}
