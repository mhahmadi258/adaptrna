# Fix: tool responses printed twice in the chat

## Context

In the browser chat, when the model calls a tool, the tool's result is rendered
**twice** during streaming:

1. first as a **plain text** bubble (raw tool output rendered as assistant Markdown), then
2. as the proper **tool-result row** (the collapsible `<details>` element).

After a page refresh, only the tool-result row remains — the plain bubble disappears. So
the persisted state is correct; the duplication only exists in the live SSE stream. The
result should be rendered exactly once (as the tool-result row).

## Root cause

`stream_turn` in `agentic/adaptrna_agentic/api/events.py` (lines 37-53) consumes the graph
with two stream modes:

```python
for mode, chunk in graph.stream(payload, config, stream_mode=["updates", "messages"]):
    if mode == "messages":
        message, _meta = chunk
        text = _text_of(message)          # <-- runs on EVERY message, incl. ToolMessage
        if text:
            yield sse("text", {"delta": text})
    elif mode == "updates":
        yield from _update_events(chunk)  # <-- emits tool_result for the ToolMessage
```

The `"messages"` stream mode emits **all** messages produced inside nodes, not just LLM
token chunks. When the `tools` node appends a `ToolMessage` to state, that message is
delivered through the `"messages"` stream too. `_text_of()` happily returns its content,
so it is emitted as a `text` delta → the client's `text` handler paints it as a plain
Markdown assistant bubble (`ui/app.js` lines 200-210).

Separately, the `"updates"` branch runs `_update_events()` (`events.py` lines 66-79), which
emits the same `ToolMessage` as a `tool_result` frame → the client's `tool_result` handler
(`ui/app.js` lines 217-221) appends the `<details>` row.

Hence two DOM nodes for one logical result. On refresh, `history()` (`events.py` lines
107-127) rebuilds from the checkpointer, which holds the `ToolMessage` once and maps it to a
single tool-result row — so the plain bubble is gone.

**Why fix on the backend:** the SSE stream is the single source of the duplication, and the
persisted/`history()` path already behaves correctly (one row). Fixing the emission keeps
the live stream consistent with what a refresh shows, and matches the existing design where
`tool_result` is the sole channel for tool output. A frontend-only dedup would be a
band-aid over a backend that emits contradictory frames.

## The fix

In `stream_turn`, restrict the `"messages"` branch to **AI** messages only, so tool (and
human) messages never become `text` deltas. LLM token streaming is unaffected because LLM
tokens arrive as `AIMessage`/`AIMessageChunk` (a subclass of `AIMessage`, already imported
at `events.py` line 16).

```python
if mode == "messages":
    message, _meta = chunk
    if isinstance(message, AIMessage):
        text = _text_of(message)
        if text:
            yield sse("text", {"delta": text})
elif mode == "updates":
    yield from _update_events(chunk)
```

`ToolMessage` output continues to reach the client exactly once, via the `tool_result`
frame from `_update_events()`. No frontend change is required.

### Files to modify

- `agentic/adaptrna_agentic/api/events.py` — the guard above in `stream_turn` (the only
  code change).

## Tests

Add a regression test in `agentic/tests/test_ui_contract.py` (alongside
`test_a_turn_carries_the_fields_the_client_reads`, which already drives a turn that calls
`list_tools`). Assert that the tool's output appears **only** as a `tool_result` frame and
never leaks into a `text` frame — e.g. no `text` delta equals the tool's content, and there
is exactly one `tool_result` per tool call. This turns the bug into a failing-then-passing
test and guards against re-adding tool text to the stream.

The existing `test_a_turn_carries_the_fields_the_client_reads` and
`test_every_streamed_event_is_one_the_client_handles` should continue to pass unchanged
(the event set and field shapes are untouched).

## Documentation

Update the SSE-contract description in `documents/modules/api.md` (section
"4. `events.py` — the SSE contract", lines ~138-146). The current line reads:

> `messages` chunks become `text` frames (token by token), `updates` chunks become
> `tool_call` and `tool_result` frames.

Tighten it to state that **only assistant (AI) message tokens** become `text` frames, and
that tool output is carried solely by `tool_result` frames — so the same result is never
emitted as both plain text and a tool-result row.

## Verification

1. Run the contract tests: `cd agentic && python -m pytest tests/test_ui_contract.py -q`
   (or the repo's usual test command). The new regression test passes; existing ones stay
   green.
2. Manual end-to-end: start the API (`serve`), open the browser UI, send a message that
   triggers a tool (e.g. "what tools are there?"). Confirm the tool result appears once, as
   the collapsible tool-result row, with no preceding plain-text copy.
3. Refresh the page and confirm the conversation is unchanged (still one tool-result row) —
   i.e. the live view now matches the reloaded view.
