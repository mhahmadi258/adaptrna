"""Chat sessions: the streaming surface, and the approval round trip.

Sessions are LangGraph threads in the SQLite checkpointer the terminal also uses, so a
conversation started in one front end continues in the other.
"""

import sqlite3

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from adaptrna_agentic.api.app import get_services
from adaptrna_agentic.api.deps import Services
from adaptrna_agentic.api.events import history, pending_approval, stream_turn
from adaptrna_agentic.api.schemas import MessageRequest, ResumeRequest
from adaptrna_agentic.toolhub.errors import ToolHubError

router = APIRouter(prefix="/sessions", tags=["sessions"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",   # proxies otherwise buffer SSE into uselessness
}


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


@router.get("")
def list_sessions(services: Services = Depends(get_services)) -> list:
    if not services.db_path.exists():
        return []

    connection = sqlite3.connect(services.db_path)
    try:
        rows = connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        connection.close()

    return [thread_id for (thread_id,) in rows]


@router.get("/{session_id}/history")
def session_history(session_id: str, services: Services = Depends(get_services)) -> dict:
    """Everything said so far — what a UI reconnecting mid-conversation needs."""
    config = _config(session_id)

    return {
        "session": session_id,
        "messages": history(services.graph, config),
        "pending_approval": pending_approval(services.graph, config),
    }


@router.post("/{session_id}/messages")
def send_message(
    session_id: str, body: MessageRequest, services: Services = Depends(get_services)
) -> StreamingResponse:
    """One turn, streamed. Ends on `done` — or on `approval_required`, which suspends it."""
    config = _config(session_id)

    if pending_approval(services.graph, config) is not None:
        raise ToolHubError(
            f"Session '{session_id}' is waiting for an approval decision. "
            f"POST /api/sessions/{session_id}/resume first."
        )

    payload = {"messages": [HumanMessage(content=body.text)]}

    return StreamingResponse(
        stream_turn(services.graph, config, payload),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/{session_id}/resume")
def resume_session(
    session_id: str, body: ResumeRequest, services: Services = Depends(get_services)
) -> StreamingResponse:
    """Answer a pending approval and continue the same turn, as a new stream."""
    config = _config(session_id)

    if pending_approval(services.graph, config) is None:
        raise ToolHubError(
            f"Session '{session_id}' has nothing awaiting approval."
        )

    decision = {"approved": body.approved}
    if body.note:
        decision["note"] = body.note
    elif not body.approved:
        decision["note"] = "the user declined"

    return StreamingResponse(
        stream_turn(services.graph, config, Command(resume=decision)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
