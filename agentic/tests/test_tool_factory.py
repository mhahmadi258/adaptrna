"""The ToolHub → LangChain bridge: names, schemas, shared-state mutation, refusals."""

import pytest

from adaptrna_agentic.agents.tool_factory import MANAGEMENT_TOOL_NAMES, build_agent_tools
from adaptrna_agentic.toolhub.manifest import ToolEntry
from adaptrna_agentic.toolhub.registry import ToolHubError
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

DUMMY = "fixtures.dummy_external"
SEQ = "GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC"


@pytest.fixture
def loaded(nano_registry, nano_splice_adapter):
    nano_registry.register(nano_splice_adapter)
    nano_registry.register_external(DUMMY)
    return nano_registry, AdapterRuntime(nano_registry)


def _by_name(registry, runtime):
    return {tool.name: tool for tool in build_agent_tools(registry, runtime)}


def test_all_expected_tools_built(loaded):
    names = set(_by_name(*loaded))

    assert set(MANAGEMENT_TOOL_NAMES) <= names
    assert {"splice_site", "dummy_echo", "dummy_add"} <= names


def test_adapter_schema_takes_sequences(loaded):
    tool = _by_name(*loaded)["splice_site"]

    assert set(tool.args) == {"sequences"}
    assert "probability" in tool.description


def test_external_schema_mirrors_wrapper_signature(loaded):
    tool = _by_name(*loaded)["dummy_add"]

    assert set(tool.args) == {"a", "b"}


def test_disabled_entry_description_carries_the_note(loaded):
    registry, runtime = loaded
    registry.deactivate("dummy_echo")

    tool = _by_name(registry, runtime)["dummy_echo"]

    assert "DISABLED" in tool.description


def test_management_tools_mutate_the_shared_registry(loaded):
    registry, runtime = loaded
    tools = _by_name(registry, runtime)

    assert "disabled" in tools["deactivate_tool"].invoke({"name": "dummy_echo"})
    assert registry.get("dummy_echo").state == "disabled"

    tools["activate_tool"].invoke({"name": "dummy_echo"})
    assert registry.get("dummy_echo").active


def test_disabled_capability_returns_refusal_as_result(loaded):
    registry, runtime = loaded
    registry.deactivate("dummy_echo")
    tools = _by_name(registry, runtime)

    result = tools["dummy_echo"].invoke({"value": "hi"})

    assert isinstance(result, str)
    assert "disabled" in result
    # Phase 10: the refusal must not read as an instruction to fix it yourself. It used to
    # say "Call activate_tool('dummy_echo') first", which is exactly what the model then did.
    assert "Only the user can enable it" in result
    assert "Call activate_tool" not in result


def test_the_disabled_note_does_not_tell_the_model_to_self_activate(loaded):
    """The description a disabled tool carries is the other half of the same lesson."""
    registry, runtime = loaded
    registry.deactivate("dummy_echo")
    tools = _by_name(registry, runtime)

    description = tools["dummy_echo"].description

    assert "DISABLED" in description
    assert "only the user can enable it" in description.lower()
    assert "call activate_tool" not in description.lower()


def test_tool_state_changes_are_gated(loaded):
    """Both toggles are approval-gated — the enforcement lives in the graph, but the
    membership is what every front end reads to know a modal is coming."""
    from adaptrna_agentic.agents.tool_factory import GATED_TOOLS

    assert "activate_tool" in GATED_TOOLS
    assert "deactivate_tool" in GATED_TOOLS


def test_unknown_name_comes_back_as_message_not_exception(loaded):
    tools = _by_name(*loaded)

    result = tools["tool_info"].invoke({"name": "banana"})

    assert isinstance(result, str)
    assert "Known tools" in result


def test_list_tools_and_test_tool_dispatch(loaded):
    tools = _by_name(*loaded)

    entries = tools["list_tools"].invoke({})
    assert {"splice_site", "dummy_echo"} <= {e["name"] for e in entries}

    report = tools["test_tool"].invoke({"name": "dummy_echo"})
    assert report["ok"] is True


def test_collision_with_management_name_refused(loaded):
    registry, runtime = loaded
    registry.manifest.tools["list_tools"] = ToolEntry(
        name="list_tools", type="external", state="active", description="rogue",
        external={"module": DUMMY, "function": "echo", "package": {}},
    )

    with pytest.raises(ToolHubError, match="collides"):
        build_agent_tools(registry, runtime)


def test_adapter_tool_executes_on_nano(loaded):
    tools = _by_name(*loaded)

    outputs = tools["splice_site"].invoke({"sequences": [SEQ]})

    assert len(outputs) == 1
    assert 0.0 <= outputs[0] <= 1.0
