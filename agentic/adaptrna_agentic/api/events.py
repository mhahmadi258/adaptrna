"""Turning a LangGraph run into Server-Sent Events.

The only genuinely new logic in this layer, and deliberately a plain generator over the
graph so it is testable without a server.

The approval contract (plans/PHASE_8_SERVICE_API.md §2): when the graph suspends at a
gate, the stream emits `approval_required` and **ends**. A gate can wait minutes for a
human; holding an SSE connection open across that would strand the turn on any dropped
connection, whereas ending it leaves the turn suspended in the checkpointer — which is
exactly what the checkpointer is for. The client resumes with a separate request.
"""

from typing import Any, Dict, Iterator, Optional
import json

from langchain_core.messages import AIMessage, ToolMessage

#: Tool results can be large (a base-pairing matrix, a full manifest entry). The stream
#: carries a display-sized preview; the full value stays in the session history.
RESULT_PREVIEW_CHARS = 2000


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def pending_approval(graph, config) -> Optional[Dict[str, Any]]:
    """The approval request the graph is suspended on, if any."""
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        for item in task.interrupts:
            return item.value

    return None


def stream_turn(graph, config, payload) -> Iterator[str]:
    """Run one turn (or resume one) and yield SSE frames.

    `payload` is either `{"messages": [HumanMessage(...)]}` for a new turn or a
    `Command(resume=...)` for a suspended one.
    """
    try:
        for mode, chunk in graph.stream(
            payload, config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                message, _meta = chunk
                text = _text_of(message)
                if text:
                    yield sse("text", {"delta": text})
            elif mode == "updates":
                yield from _update_events(chunk)

        request = pending_approval(graph, config)
        if request is not None:
            yield sse("approval_required", request)
            return

        yield sse("done", {"answer": _final_answer(graph, config)})

    except Exception as exc:  # noqa: BLE001 -- the stream is the error channel
        yield sse("error", {"message": str(exc), "type": type(exc).__name__})


def _update_events(chunk: Dict[str, Any]) -> Iterator[str]:
    for node, update in (chunk or {}).items():
        if not isinstance(update, dict):
            continue

        for message in update.get("messages") or []:
            if isinstance(message, AIMessage) and message.tool_calls:
                for call in message.tool_calls:
                    yield sse("tool_call", {"name": call["name"], "args": call.get("args", {})})
            elif isinstance(message, ToolMessage):
                yield sse("tool_result", {
                    "name": message.name,
                    "content": _preview(str(message.content)),
                })


def _text_of(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    return "".join(
        block.get("text", "")
        for block in content or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _preview(text: str) -> str:
    return text if len(text) <= RESULT_PREVIEW_CHARS else text[:RESULT_PREVIEW_CHARS] + " …"


def _final_answer(graph, config) -> str:
    messages = (graph.get_state(config).values or {}).get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _text_of(message)

    return ""


def history(graph, config) -> list:
    """The session's messages, for a UI that reconnects mid-conversation."""
    messages = (graph.get_state(config).values or {}).get("messages") or []

    out = []
    for message in messages:
        role = getattr(message, "type", "")
        if role == "human":
            out.append({"role": "user", "content": _text_of(message)})
        elif isinstance(message, ToolMessage):
            out.append({"role": "tool", "name": message.name,
                        "content": _preview(str(message.content))})
        elif isinstance(message, AIMessage):
            entry = {"role": "assistant", "content": _text_of(message)}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"name": c["name"], "args": c.get("args", {})} for c in message.tool_calls
                ]
            out.append(entry)

    return out
