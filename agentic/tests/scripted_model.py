"""A scripted fake chat model honoring `bind_tools` — the shared test double for every
graph test (no network, no API key). Replays a fixed list of AIMessages and records the
messages of every invocation in `calls`."""

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    script: List[AIMessage]
    calls: List[List[BaseMessage]]
    bound_tools: List[Any] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        self.bound_tools = list(tools)
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


def scripted(script: List[AIMessage]) -> ScriptedChatModel:
    return ScriptedChatModel(script=script, calls=[])


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}
