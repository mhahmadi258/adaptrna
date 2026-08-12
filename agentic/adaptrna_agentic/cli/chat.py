"""Terminal chat with the Phase 0 hello graph.

    python -m adaptrna_agentic.cli.chat                    # REPL ('quit' or EOF to exit)
    python -m adaptrna_agentic.cli.chat --once "PROMPT"    # single exchange, then exit
    python -m adaptrna_agentic.cli.chat --model anthropic:claude-sonnet-5 ...

Tool calls are printed as they happen — this output style previews the Phase 4
orchestrator's UX. Multi-turn state is a plain in-process message list; the checkpointer
arrives with the Phase 4 orchestrator.
"""

from typing import List
import argparse
import dataclasses

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from adaptrna_agentic.agents.hello import build_hello_graph
from adaptrna_agentic.models import build_chat_model
from adaptrna_agentic.settings import ROLES, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaptrna_agentic.cli.chat",
        description="Chat with the AdaptRNA hello graph (Phase 0 scaffold).",
    )
    parser.add_argument("--once", type=str, default=None,
                        help="Send one prompt, print the answer, exit")
    parser.add_argument("--model", type=str, default=None,
                        help="Model spec overriding every role for this run, "
                             "e.g. anthropic:claude-sonnet-5")

    return parser


def _message_text(message: BaseMessage) -> str:
    """Assistant text, tolerant of both plain-string and content-block responses."""
    content = message.content
    if isinstance(content, str):
        return content

    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts) or str(content)


def _print_progress(messages: List[BaseMessage], seen: int) -> int:
    """Print tool activity from messages we have not shown yet; return the new cursor."""
    for message in messages[seen:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  → {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"    = {message.content}")

    return len(messages)


def run_turn(graph, history: List[BaseMessage], user_text: str) -> str:
    """One user turn: stream the graph, narrate tool calls, print and return the answer."""
    history.append(HumanMessage(content=user_text))

    seen = len(history)
    final = history
    for state in graph.stream({"messages": history}, stream_mode="values"):
        final = state["messages"]
        seen = _print_progress(final, seen)

    history[:] = final
    answer = _message_text(history[-1])
    print(answer)

    return answer


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    settings = Settings.from_env()
    if args.model:
        settings = dataclasses.replace(
            settings, models={role: args.model for role in ROLES}
        )

    # Built eagerly so a missing API key fails here, with the actionable message,
    # rather than mid-conversation.
    model = build_chat_model("orchestrator", settings)
    graph = build_hello_graph(model)

    history: List[BaseMessage] = []

    if args.once is not None:
        run_turn(graph, history, args.once)
        return 0

    print("AdaptRNA scaffold chat — 'quit' or Ctrl-D to exit.")
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

        run_turn(graph, history, user_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
