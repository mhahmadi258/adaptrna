"""Recorded conversations, replayed against a scripted model.

These pin the *wiring and the contracts* — which tools get called, what comes back, what
the approval gate refuses to do — for the five user flows. They deliberately do not test
prompts: the model is scripted, so what is asserted is what the graph and the tools do
with a given sequence of model outputs. Prompt regressions are the job of the live suite
(`-m live`), which is user-run.

Scenarios are data (`tests/scenarios/*.yaml`) so a new flow is a new file, not new code.
"""

import json
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from adaptrna_agentic.agents.orchestrator import build_orchestrator_graph
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from scripted_model import scripted, tool_call

SCENARIO_DIR = Path(__file__).parent / "scenarios"
SCENARIOS = sorted(SCENARIO_DIR.glob("*.yaml"))


def _load(path):
    return yaml.safe_load(path.read_text())


def _build_script(turn):
    """Turn a scenario's `script` into AIMessages the fake model replays."""
    messages = []
    for step in turn["script"]:
        if "tool_calls" in step:
            calls = [
                tool_call(c["name"], c.get("args", {}), c.get("id", f"call_{i}"))
                for i, c in enumerate(step["tool_calls"])
            ]
            messages.append(AIMessage(content="", tool_calls=calls))
        else:
            messages.append(AIMessage(content=step["text"]))

    return messages


@pytest.fixture
def stack(nano_registry, nano_splice_adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)

    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external("fixtures.dummy_external")

    return nano_registry, AdapterRuntime(nano_registry), tmp_path


def _jobs(tmp_path):
    from adaptrna_agentic.jobs.store import JobStore

    return JobStore(tmp_path / "jobs_data").jobs


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario(path, stack):
    registry, runtime, tmp_path = stack
    scenario = _load(path)
    config = {"configurable": {"thread_id": scenario["name"]}}
    saver = InMemorySaver()

    for index, turn in enumerate(scenario["turns"]):
        model = scripted(_build_script(turn))
        graph = build_orchestrator_graph(
            model=model, registry=registry, runtime=runtime, checkpointer=saver
        )

        state = graph.invoke({"messages": [HumanMessage(content=turn["user"])]}, config)
        where = f"{scenario['name']} turn {index}"

        if turn.get("expect_interrupt"):
            snapshot = graph.get_state(config)
            assert snapshot.next, f"{where}: expected the approval gate to suspend"
            if turn.get("expect_no_jobs"):
                assert not _jobs(tmp_path), f"{where}: a job ran BEFORE approval"

            state = graph.invoke(Command(resume=turn["resume"]), config)

        messages = state["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        if "expect_tools" in turn:
            called = [m.name for m in tool_messages][-len(turn["expect_tools"]):]
            assert called == turn["expect_tools"], f"{where}: tools called {called}"

        blob = " ".join(m.content for m in tool_messages)
        for needle in turn.get("expect_tool_result_contains", []):
            assert needle in blob, f"{where}: {needle!r} not in tool results"

        if turn.get("expect_probabilities"):
            values = json.loads(tool_messages[-1].content)
            assert all(0.0 <= float(v) <= 1.0 for v in values), f"{where}: {values}"

        for name, expected in (turn.get("expect_state") or {}).items():
            assert registry.get(name).state == expected, f"{where}: {name} state"

        if turn.get("expect_no_jobs_after"):
            assert not _jobs(tmp_path), f"{where}: a job exists after a refusal"

        # Every turn must end with the model speaking, not mid-tool-loop.
        assert isinstance(messages[-1], AIMessage), f"{where}: turn did not finish"
        assert not messages[-1].tool_calls, f"{where}: unresolved tool calls"


def test_every_flow_has_a_scenario():
    """A flow without a recorded conversation is a flow nobody notices breaking."""
    names = {_load(p)["name"] for p in SCENARIOS}

    assert {"inference", "management", "training_gate", "failure_paths"} <= names
