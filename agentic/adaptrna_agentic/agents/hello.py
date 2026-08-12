"""Phase 0 hello graph: proves the exact LangGraph wiring style later phases build on.

Hand-built `StateGraph` rather than the prebuilt agent helper on purpose — the Phase 4
orchestrator needs custom wiring (approval-gate interrupts), so this file establishes the
model-node ⇄ tool-node loop the project will actually use.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

SYSTEM_PROMPT = "You are the AdaptRNA assistant scaffold; use tools when they apply."

_VALID_BASES = set("ACGTU")


@tool
def gc_content(sequence: str) -> str:
    """Fraction of G/C bases in an RNA or DNA sequence, as a number with 3 decimals.

    Args:
        sequence: The sequence to analyse. Only A, C, G, U or T (case-insensitive).
    """
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Empty sequence: provide at least one base (A, C, G, U or T).")

    invalid = sorted(set(seq) - _VALID_BASES)
    if invalid:
        raise ValueError(
            f"Invalid characters {invalid} in sequence; expected only A, C, G, U or T."
        )

    return f"{sum(base in 'GC' for base in seq) / len(seq):.3f}"


TOOLS = [gc_content]


def _route_after_model(state: MessagesState):
    """Model asked for tools -> run them; otherwise the turn is finished."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_hello_graph(model: Optional[BaseChatModel] = None):
    """
    Compile the hello graph.

    `model` is the test seam: tests inject a scripted fake. With no argument the
    orchestrator model is resolved *lazily on first use*, so compiling the graph needs
    neither a network call nor an API key.
    """
    bound = model.bind_tools(TOOLS) if model is not None else None

    def call_model(state: MessagesState):
        nonlocal bound
        if bound is None:
            from adaptrna_agentic.models import build_chat_model

            bound = build_chat_model("orchestrator").bind_tools(TOOLS)

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        return {"messages": [bound.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", _route_after_model, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    return graph.compile()
