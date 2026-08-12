"""The AdaptRNA orchestrator — the one agent the user talks to (MASTER_PLAN §5).

Same hand-built wiring as the Phase 0 hello graph (model ⇄ tools loop), with the ToolHub
bound as agent tools and an optional checkpointer for persistent sessions. Tools are
rebuilt at every model call from the shared Registry/Runtime, so lifecycle changes are
honored immediately; the tools node executes dynamically for the same reason.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from adaptrna_agentic.agents.tool_factory import build_agent_tools, stringify_tool_output
from adaptrna_agentic.toolhub.registry import Registry
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

SYSTEM_PROMPT = """\
You are the AdaptRNA assistant: a conversational interface to RNA analysis tools built on
one shared RNA foundation-model backbone (adapter tools) plus classical bioinformatics
tools (like ViennaRNA).

Use the tools for any prediction or computation — never guess sequence properties
yourself. Report tool outputs faithfully: probabilities are probabilities, not
certainties. If a tool you need is disabled, say so and offer to activate it with
activate_tool before using it. Use list_tools / tool_info / test_tool when the user asks
about capabilities, and activate_tool / deactivate_tool when they want tools switched on
or off.

Sequences arrive as plain ACGU/T strings. Keep answers concise and grounded in the tool
results you actually received."""


def build_orchestrator_graph(
    model: Optional[BaseChatModel] = None,
    registry: Optional[Registry] = None,
    runtime: Optional[AdapterRuntime] = None,
    checkpointer=None,
):
    """
    Compile the orchestrator graph.

    All arguments are injectable (the test seams). Defaults: the configured orchestrator
    model (resolved lazily, so compiling needs no API key), the default-data-dir
    Registry, and its AdapterRuntime. Pass a LangGraph checkpointer for persistent
    sessions (invoke with `config={"configurable": {"thread_id": ...}}`).
    """
    registry = registry or Registry()
    runtime = runtime or AdapterRuntime(registry)
    injected_model = model

    def call_model(state: MessagesState):
        nonlocal injected_model
        if injected_model is None:
            from adaptrna_agentic.models import build_chat_model

            injected_model = build_chat_model("orchestrator")

        # Rebuilt every call: descriptions reflect current tool states, and tools
        # registered/toggled since the last turn are picked up.
        bound = injected_model.bind_tools(build_agent_tools(registry, runtime))

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        return {"messages": [bound.invoke(messages)]}

    def run_tools(state: MessagesState):
        tools = {tool.name: tool for tool in build_agent_tools(registry, runtime)}
        last = state["messages"][-1]

        results = []
        for call in last.tool_calls:
            tool = tools.get(call["name"])
            if tool is None:
                output = f"Unknown tool '{call['name']}'. Use list_tools to see what exists."
            else:
                try:
                    # handle_tool_error=True turns ToolExceptions (refusals, validation)
                    # into result strings; anything else is caught here so one bad call
                    # never kills the turn.
                    output = tool.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001
                    output = f"Tool '{call['name']}' failed: {exc}"

            results.append(ToolMessage(
                content=stringify_tool_output(output),
                tool_call_id=call["id"],
                name=call["name"],
            ))

        return {"messages": results}

    def route_after_model(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    return graph.compile(checkpointer=checkpointer)
