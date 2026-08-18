"""Chat sessions: management, the streaming surface, and the approval round trip.

Sessions are LangGraph threads in the SQLite checkpointer the terminal also uses, so a
conversation started in one front end continues in the other.

Phase 8 declined to expose any delete surface over HTTP, and Phase 9 agreed. Phase 10
narrowed that rather than dropping it: a session is the user's own conversation, cheap to
recreate, and the client that lists them is where a person expects to manage them.
Everything else — tools, artifacts, jobs, staged code — is still CLI-only, and the *agent*
still cannot delete anything (plans/PHASE_10_SESSION_RAIL_AND_TOOL_GATE.md §1).
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from adaptrna_agentic.api import sessions_store
from adaptrna_agentic.api.app import get_services
from adaptrna_agentic.api.deps import Services
from adaptrna_agentic.api.events import history, pending_approval, stream_turn
from adaptrna_agentic.api.schemas import MessageRequest, ResumeRequest, SessionRequest
from adaptrna_agentic.toolhub.errors import ToolHubError
from adaptrna_agentic.toolhub.prune import delete_session

router = APIRouter(prefix="/sessions", tags=["sessions"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",   # proxies otherwise buffer SSE into uselessness
}


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _clean_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        raise ValueError("A session needs a name.")
    return name


def _require(services: Services, session_id: str) -> None:
    if not sessions_store.session_exists(services.db_path, session_id):
        raise KeyError(f"No session '{session_id}'.")


@router.get("")
def list_sessions(services: Services = Depends(get_services)) -> list:
    """Newest first — a rail sorted alphabetically is a worse rail."""
    return sessions_store.list_sessions(services.db_path)


@router.post("", status_code=201)
def create_session(
    body: SessionRequest, services: Services = Depends(get_services)
) -> dict:
    """Claim a session name and make it real.

    A thread only appears in the listing once it has a checkpoint, so an empty state update
    is written here — otherwise a session created in the rail would vanish on refresh,
    before its first turn.
    """
    name = _clean_name(body.id)
    if sessions_store.session_exists(services.db_path, name):
        raise ToolHubError(f"Session '{name}' already exists.")

    services.graph.update_state(_config(name), {"messages": []})

    return sessions_store.get_session(services.db_path, name) or {
        "id": name, "updated_at": None, "checkpoints": 0
    }


@router.patch("/{session_id}")
def rename_session(
    session_id: str, body: SessionRequest, services: Services = Depends(get_services)
) -> dict:
    """Rename a session, carrying its history with it."""
    _require(services, session_id)
    name = _clean_name(body.id)

    if name == session_id:
        return sessions_store.get_session(services.db_path, session_id)
    if sessions_store.session_exists(services.db_path, name):
        raise ToolHubError(f"Session '{name}' already exists.")

    # A suspended turn is addressed by thread id. Moving the rows out from under it would
    # strand the interrupt with no way for either front end to answer it.
    if pending_approval(services.graph, _config(session_id)) is not None:
        raise ToolHubError(
            f"Session '{session_id}' is waiting for an approval decision. "
            f"Answer it before renaming."
        )

    sessions_store.rename_session(services.db_path, session_id, name)
    return sessions_store.get_session(services.db_path, name)


@router.delete("/{session_id}")
def remove_session(session_id: str, services: Services = Depends(get_services)) -> dict:
    """Delete a session and everything in it. Irreversible."""
    _require(services, session_id)
    delete_session(session_id, db_path=services.db_path)

    return {"deleted": session_id}


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
    if body.edits:
        decision["edits"] = body.edits

    return StreamingResponse(
        stream_turn(services.graph, config, Command(resume=decision)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
