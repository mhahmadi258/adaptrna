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
from typing import Any, List, Optional
import argparse
import dataclasses
import os
import sqlite3

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

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


def _pending_interrupt(graph, config):
    """The approval request the graph is suspended on, if any."""
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        for item in task.interrupts:
            return item.value

    return None


def _print_spec_block(spec: dict) -> None:
    """`details["spec"]` (gate 1, Phase 13 §4), field by field."""
    fmt = spec.get("format") or {}
    rows = fmt.get("rows")
    columns = len(spec.get("ignored_columns") or []) + 2
    rows_str = f"{rows:,}" if isinstance(rows, int) else "?"
    print(f"  │   file:      {spec.get('path', '?')}  ({rows_str} rows, {columns} columns)")

    length = spec.get("length") or {}
    print(
        f"  │   sequence:  {spec.get('sequence_column', '?')}          "
        f"{length.get('median', '?')} nt (min {length.get('min', '?')}, "
        f"max {length.get('max', '?')}), {spec.get('alphabet', '?')}"
    )

    label_line = f"  │   label:     {spec.get('label_column', '?')}             {spec.get('target_type', '?')}"
    if spec.get("class_counts"):
        counts = " · ".join(f"{k}: {v:,}" for k, v in spec["class_counts"].items())
        label_line += f" — {counts}"
        if spec.get("positive_class"):
            label_line += f" (positive: '{spec['positive_class']}')"
    elif spec.get("target_summary"):
        summary = spec["target_summary"]
        label_line += (
            f" — min {summary.get('min')}, max {summary.get('max')}, mean {summary.get('mean')}"
        )
    print(label_line)

    if spec.get("ignored_columns"):
        print(f"  │   ignored:   {', '.join(spec['ignored_columns'])}")

    split = spec.get("split") or {}
    counts = split.get("row_counts") or {}
    counts_str = " / ".join(str(counts.get(k, "?")) for k in ("train", "val", "test"))
    if split.get("mode") == "random":
        fractions = split.get("fractions") or {}
        pct = lambda k: int(round((fractions.get(k) or 0) * 100))  # noqa: E731
        stratified = "stratified, " if split.get("stratify") else ""
        print(
            f"  │   split:     random {pct('train')}/{pct('val')}/{pct('test')}, "
            f"seed {split.get('seed')}, {stratified}→ {counts_str}"
        )
    elif split.get("mode") == "column":
        print(f"  │   split:     column '{split.get('column')}' → {counts_str}")

    head = spec.get("head") or {}
    print(
        f"  │   head:      {head.get('kind', '?')} · {head.get('loss', '?')} · "
        f"{head.get('primary_metric', '?')}"
    )
    print(f"  │   task:      {spec.get('task_name', '?')}")


def _parse_edit_value(text: str):
    """`field=value` values: numbers and JSON where they parse, plain text otherwise —
    so `positive_class=1` stays the string '1' (a class label) while
    `split.mapping={"train": ["human"]}` parses as the dict it needs to be."""
    import json

    for parser in (int, float):
        try:
            return parser(text)
        except ValueError:
            pass

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _prompt_approval(request) -> dict:
    """Show exactly what would happen, then ask. This is the human gate.

    Accepts `field=value` lines before the final y/N — each stages an edit (dotted path
    into the gate's own argument object) and re-prompts; typing 'y'/'yes' approves with
    whatever edits were staged, anything else declines.
    """
    print()
    print("  ┌─ approval required " + "─" * 46)
    for item in request.get("requests", []):
        print(f"  │ {item['summary']}")
        details = item.get("details") or {}
        if details.get("tool"):
            print(f"  │   tool:      {details['tool']}")
            print(f"  │   state:     {details['current_state']} → {details['after_approval']}")
        if details.get("spec"):
            _print_spec_block(details["spec"])
        if details.get("command"):
            print(f"  │   would run: {' '.join(details['command'])}")
        if details.get("output_dir"):
            print(f"  │   output:    {details['output_dir']}")
        if details.get("downloads"):
            print(f"  │   note:      {details['downloads']}")
        for entry in details.get("files") or []:
            print(f"  │   write:     {entry['path']}  ({entry['lines']} lines)")
        if details.get("staging_path"):
            print(f"  │   staged in: {details['staging_path']}")
            print("  │   (open it in your editor to read the code before approving)")
        for warning in details.get("warnings") or []:
            print(f"  │   ! {warning}")
    print("  └" + "─" * 66)

    edits: dict = {}
    while True:
        try:
            reply = input("  edit a field, or approve? [field=value … | y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return {"approved": False, "note": "the user declined at the approval prompt"}

        lowered = reply.lower()
        if lowered in ("y", "yes"):
            decision = {"approved": True}
            if edits:
                decision["edits"] = edits
            return decision

        if "=" in reply and lowered not in ("n", "no"):
            field, _, value = reply.partition("=")
            field = field.strip()
            if field:
                edits[field] = _parse_edit_value(value.strip())
                print(f"  │   staged: {field} = {edits[field]!r}")
                continue

        return {"approved": False, "note": "the user declined at the approval prompt"}


def run_turn(graph, config, user_text: str) -> str:
    """One user turn. History lives in the checkpointer: only the new message is sent.

    A turn may suspend at an approval gate; we render the request, ask, and resume.
    """
    payload: Any = {"messages": [HumanMessage(content=user_text)]}
    seen: Optional[int] = None
    final: List[BaseMessage] = []

    while True:
        for state in graph.stream(payload, config, stream_mode="values"):
            final = state["messages"]
            if seen is None:
                seen = len(final)      # checkpointed history + the new human message
            else:
                seen = _print_progress(final, seen)

        request = _pending_interrupt(graph, config)
        if request is None:
            break

        payload = Command(resume=_prompt_approval(request))
        seen = len(final)

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
        for problem in runtime.warmup():
            print(f"  skipped: {problem}")
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
