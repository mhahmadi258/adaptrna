"""The load-bearing Phase 0 test: a scripted fake chat model drives the full graph loop —
tool binding, ToolNode execution, loop termination — with no network and no API key.
This is everything the hello graph does except Anthropic's API itself."""

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from adaptrna_agentic.agents.hello import SYSTEM_PROMPT, build_hello_graph


class ScriptedChatModel(BaseChatModel):
    """Minimal fake chat model honoring `bind_tools`: replays a fixed list of AIMessages."""

    script: List[AIMessage]
    calls: List[List[BaseMessage]]

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Any = None, **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        message = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def _scripted(script: List[AIMessage]) -> ScriptedChatModel:
    return ScriptedChatModel(script=script, calls=[])


def test_tool_loop_executes_and_terminates():
    model = _scripted([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "gc_content",
                "args": {"sequence": "GGCAUUACGGCU"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="The GC content is 0.583."),
    ])
    graph = build_hello_graph(model)

    state = graph.invoke(
        {"messages": [HumanMessage(content="GC content of GGCAUUACGGCU?")]}
    )
    messages = state["messages"]

    # The tool actually ran, with the correct computed value.
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "0.583"

    # The loop terminated on the tool-call-free response, which is the final message.
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "The GC content is 0.583."
    assert not messages[-1].tool_calls

    # model -> tools -> model -> END: exactly two model invocations.
    assert len(model.calls) == 2


def test_system_prompt_is_injected_once():
    model = _scripted([AIMessage(content="hi")])
    graph = build_hello_graph(model)

    graph.invoke({"messages": [HumanMessage(content="hello")]})

    first_call = model.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert first_call[0].content == SYSTEM_PROMPT
    assert sum(isinstance(m, SystemMessage) for m in first_call) == 1


def test_graph_compiles_without_model_and_without_key(monkeypatch):
    # Model construction must be lazy: compiling with no injected model touches neither
    # the network nor the credential check.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    graph = build_hello_graph()

    assert graph is not None
