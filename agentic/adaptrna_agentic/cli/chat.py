"""Terminal chat with the AdaptRNA orchestrator.

    python -m adaptrna_agentic.cli.chat                       # REPL, session 'default'
    python -m adaptrna_agentic.cli.chat --session paper       # named persistent session
    python -m adaptrna_agentic.cli.chat --once "PROMPT"       # single exchange
    python -m adaptrna_agentic.cli.chat --list-sessions
    python -m adaptrna_agentic.cli.chat --warmup              # preload the backbone

Sessions persist in `chat_data/sessions.sqlite` (override the directory with
ADAPTRNA_CHAT_DIR). The chat process holds one ToolHub runtime: the backbone loads once,
on the first foundation-model tool call (or at startup with --warmup). Tool lifecycle
changes made outside this process are picked up at the next turn.
"""

from datetime import datetime
from pathlib import Path
from typing import List
import argparse
import dataclasses
import os
import sqlite3

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from adaptrna_agentic.agents.orchestrator import build_orchestrator_graph
from adaptrna_agentic.models import build_chat_model
from adaptrna_agentic.settings import REPO_ROOT, ROLES, Settings
from adaptrna_agentic.toolhub.registry import Registry
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

CHAT_DIR_VAR = "ADAPTRNA_CHAT_DIR"

_RESULT_PREVIEW_CHARS = 200


def chat_db_path() -> Path:
    root = Path(os.environ.get(CHAT_DIR_VAR) or REPO_ROOT / "chat_data").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / "sessions.sqlite"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaptrna_agentic.cli.chat",
        description="Chat with the AdaptRNA orchestrator (tools from the ToolHub).",
    )
    parser.add_argument("--session", type=str, default="default",
                        help="Session name; history persists per session (default: 'default')")
    parser.add_argument("--new-session", action="store_true",
                        help="Start a fresh, timestamp-named session")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List existing session names and exit")
    parser.add_argument("--once", type=str, default=None,
                        help="Send one prompt, print the answer, exit")
    parser.add_argument("--model", type=str, default=None,
                        help="Model spec overriding every role for this run")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="ToolHub state dir (default: $ADAPTRNA_TOOLHUB_DIR or <repo>/toolhub_data)")
    parser.add_argument("--warmup", action="store_true",
                        help="Load the backbone and active adapters at startup")

    return parser


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content

    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts) or str(content)


def _preview(text: str) -> str:
    text = " ".join(str(text).split())
    if len(text) > _RESULT_PREVIEW_CHARS:
        return text[:_RESULT_PREVIEW_CHARS] + " …"
    return text


def _print_progress(messages: List[BaseMessage], seen: int) -> int:
    for message in messages[seen:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  → {call['name']}({_preview(call['args'])})")
        elif isinstance(message, ToolMessage):
            print(f"    = {_preview(message.content)}")

    return len(messages)


def run_turn(graph, config, user_text: str) -> str:
    """One user turn. History lives in the checkpointer: only the new message is sent."""
    seen = None
    final: List[BaseMessage] = []

    for state in graph.stream(
        {"messages": [HumanMessage(content=user_text)]}, config, stream_mode="values"
    ):
        final = state["messages"]
        if seen is None:
            seen = len(final)          # checkpointed history + the new human message
        else:
            seen = _print_progress(final, seen)

    answer = _message_text(final[-1])
    print(answer)
    return answer


def _list_sessions() -> int:
    db = chat_db_path()
    if not db.exists():
        print("No sessions yet.")
        return 0

    connection = sqlite3.connect(db)
    try:
        rows = connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        connection.close()

    if not rows:
        print("No sessions yet.")
    for (thread_id,) in rows:
        print(thread_id)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_sessions:
        return _list_sessions()

    settings = Settings.from_env()
    if args.model:
        settings = dataclasses.replace(
            settings, models={role: args.model for role in ROLES}
        )

    # Built eagerly so a missing API key fails here, with the actionable message.
    model = build_chat_model("orchestrator", settings)

    registry = Registry(args.data_dir)
    runtime = AdapterRuntime(registry)
    if args.warmup:
        print("Loading backbone …", flush=True)
        runtime.warmup()
        print(f"Resident adapters: {sorted(runtime._resident) or 'none'}")

    session = args.session
    if args.new_session:
        session = datetime.now().strftime("session-%Y%m%d-%H%M%S")

    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(chat_db_path(), check_same_thread=False)
    checkpointer = SqliteSaver(connection)

    graph = build_orchestrator_graph(
        model=model, registry=registry, runtime=runtime, checkpointer=checkpointer
    )
    config = {"configurable": {"thread_id": session}}

    if args.once is not None:
        run_turn(graph, config, args.once)
        return 0

    print(f"AdaptRNA chat — session '{session}' — 'quit' or Ctrl-D to exit.")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.lower() in ("quit", "exit"):
            break
        if not user_text:
            continue

        run_turn(graph, config, user_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
