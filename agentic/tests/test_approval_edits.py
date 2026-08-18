"""The edits mechanism: a human may correct a gate's proposed arguments before approving.

Phase 13 §5 — `_apply_edits` is whitelist-only and type-checked, and every edit actually
applied is recorded on the object the tool returns (`human_edits` / `human_overrides`),
so no later surface can present an edited value as if the system had recommended it.
"""

import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from adaptrna_agentic.agents.orchestrator import _apply_edits, build_orchestrator_graph
from adaptrna_agentic.profiling.profiler import profile_dataset
from adaptrna_agentic.toolhub.runtime import AdapterRuntime
from scripted_model import scripted, tool_call

CONFIG = {"configurable": {"thread_id": "edits"}}


# ---------------------------------------------------------------- _apply_edits, in isolation

def test_no_edits_returns_args_unchanged():
    args = {"spec": {"task_name": "x"}}
    assert _apply_edits(args, None, "confirm_data_profile") is args


def test_empty_edits_dict_returns_args_unchanged():
    args = {"spec": {"task_name": "x"}}
    assert _apply_edits(args, {}, "confirm_data_profile") is args


def test_unwhitelisted_path_is_refused():
    args = {"spec": {"task_name": "x"}}
    with pytest.raises(ValueError, match="not editable"):
        _apply_edits(args, {"spec.path": "/tmp/other.csv"}, "confirm_data_profile")


def test_tool_with_no_editable_fields_refuses_any_edit():
    with pytest.raises(ValueError, match="no editable fields"):
        _apply_edits({"stage_id": "x"}, {"anything": 1}, "land_generated_code")


def test_type_change_is_refused():
    # A list is not the int/float-to-string case _apply_edits deliberately tolerates
    # (see test_numeric_looking_edit_to_a_string_field_is_coerced_back_to_a_string).
    args = {"spec": {"positive_class": "1"}}
    with pytest.raises(ValueError, match="cannot change type"):
        _apply_edits(args, {"spec.positive_class": ["1"]}, "confirm_data_profile")


def test_numeric_looking_edit_to_a_string_field_is_coerced_back_to_a_string():
    """A CLI/UI edit arrives as untyped text: 'positive_class=1' means the class label
    '1', not the number 1, because the field it targets is a string even when the data's
    own labels look like integers."""
    args = {"spec": {"positive_class": "0"}}
    result = _apply_edits(args, {"spec.positive_class": 1}, "confirm_data_profile")

    assert result["spec"]["positive_class"] == "1"


def _minimal_plan(**overrides):
    """A plan with just enough fields for `build_command` to succeed -- the rebuild
    `_apply_edits` triggers on any plan.* edit (see test_editing_plan_overrides_rebuilds_
    the_command)."""
    plan = {
        "task": "x", "arm": "lora", "config_path": "adaptrna_custom/tasks/x/config.yaml",
        "output_dir": "outputs/x", "seed": 42, "overrides": {},
    }
    plan.update(overrides)
    return plan


def test_numeric_cross_between_int_and_float_is_allowed():
    args = {"plan": _minimal_plan()}
    result = _apply_edits(args, {"plan.seed": 7.0}, "start_training")
    assert result["plan"]["seed"] == 7.0


def test_wildcard_path_reaches_nested_fields():
    args = {"spec": {"split": {"fractions": {"train": 0.8, "val": 0.1, "test": 0.1}}}}
    result = _apply_edits(args, {"spec.split.fractions.train": 0.7}, "confirm_data_profile")
    assert result["spec"]["split"]["fractions"]["train"] == 0.7


def test_original_args_are_not_mutated():
    args = {"spec": {"task_name": "x"}}
    result = _apply_edits(args, {"spec.task_name": "y"}, "confirm_data_profile")

    assert args["spec"]["task_name"] == "x"
    assert result["spec"]["task_name"] == "y"


def test_spec_edits_are_recorded_as_human_edits():
    args = {"spec": {"task_name": "x"}}
    result = _apply_edits(args, {"spec.task_name": "y"}, "confirm_data_profile")

    assert result["spec"]["human_edits"] == {
        "spec.task_name": {"recommended": "x", "chosen": "y"},
    }


def test_plan_edits_are_recorded_as_human_overrides():
    args = {"plan": _minimal_plan()}
    result = _apply_edits(args, {"plan.seed": 7}, "start_training")

    assert result["plan"]["human_overrides"] == {
        "plan.seed": {"recommended": 42, "chosen": 7},
    }


def test_plan_overrides_is_a_flat_dotted_key_not_a_nested_structure():
    """plan['overrides'] is keyed by dotted strings ('optim.lr') the engine's CLI reads
    with --set; 'plan.overrides.optim.lr' must set that one flat key, not build a nested
    {'optim': {'lr': ...}} the engine would never look at."""
    args = {"plan": _minimal_plan(overrides={"optim.lr": 0.0003})}
    result = _apply_edits(args, {"plan.overrides.optim.lr": 0.001}, "start_training")

    assert result["plan"]["overrides"] == {"optim.lr": 0.001}
    assert result["plan"]["human_overrides"] == {
        "plan.overrides.optim.lr": {"recommended": 0.0003, "chosen": 0.001},
    }


def test_editing_plan_overrides_rebuilds_the_command():
    """The gate shows plan['command'] verbatim -- an edited override with a stale command
    would mean approving one thing and running another."""
    from adaptrna_agentic.profiling.recommender import build_command

    plan = {
        "task": "x", "arm": "lora", "config_path": "adaptrna_custom/tasks/x/config.yaml",
        "output_dir": "outputs/x", "seed": 42, "overrides": {"optim.lr": 0.0003},
    }
    plan["command"] = build_command(plan)
    args = {"plan": plan}

    result = _apply_edits(args, {"plan.overrides.optim.lr": 0.001}, "start_training")

    assert "optim.lr=0.001" in result["plan"]["command"]
    assert "optim.lr=0.0003" not in result["plan"]["command"]


def test_a_field_with_no_prior_value_may_be_set():
    args = {"spec": {"split": {"mode": "column"}}}
    result = _apply_edits(args, {"spec.split.column": "grp"}, "confirm_data_profile")

    assert result["spec"]["split"]["column"] == "grp"


# ---------------------------------------------------------------- end to end, through the gate

@pytest.fixture
def stack(nano_registry):
    return nano_registry, AdapterRuntime(nano_registry)


@pytest.fixture
def saver(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "sessions.sqlite", check_same_thread=False))


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")
    return path


def _graph(stack, script, saver):
    registry, runtime = stack
    model = scripted(script)
    return model, build_orchestrator_graph(
        model=model, registry=registry, runtime=runtime, checkpointer=saver
    )


def test_an_edit_at_the_gate_changes_the_approved_spec(stack, saver, csv_path):
    spec = profile_dataset(csv_path)
    new_name = "renamed_by_the_user"
    script = [
        AIMessage(content="", tool_calls=[tool_call("confirm_data_profile", {"spec": spec})]),
        AIMessage(content="Approved."),
    ]
    _model, graph = _graph(stack, script, saver)

    graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)
    final = graph.invoke(
        Command(resume={"approved": True, "edits": {"spec.task_name": new_name}}), CONFIG,
    )

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    approved = json.loads(tool_messages[0].content)
    assert approved["task_name"] == new_name
    assert approved["human_edits"] == {
        "spec.task_name": {"recommended": spec["task_name"], "chosen": new_name},
    }


def test_an_edit_that_switches_target_type_recomputes_downstream_fields(stack, saver, csv_path):
    spec = profile_dataset(csv_path)
    script = [
        AIMessage(content="", tool_calls=[tool_call("confirm_data_profile", {"spec": spec})]),
        AIMessage(content="Approved."),
    ]
    _model, graph = _graph(stack, script, saver)

    graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)
    final = graph.invoke(
        Command(resume={"approved": True, "edits": {"spec.target_type": "multiclass"}}), CONFIG,
    )

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    approved = json.loads(tool_messages[0].content)
    assert approved["target_type"] == "multiclass"
    assert approved["head"]["primary_metric"] == "test/macro_f1"  # recomputed, not carried over


def test_an_unwhitelisted_edit_surfaces_as_a_tool_failure(stack, saver, csv_path):
    spec = profile_dataset(csv_path)
    script = [
        AIMessage(content="", tool_calls=[tool_call("confirm_data_profile", {"spec": spec})]),
        AIMessage(content="Approved."),
    ]
    _model, graph = _graph(stack, script, saver)

    graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)
    final = graph.invoke(
        Command(resume={"approved": True, "edits": {"spec.path": "/tmp/somewhere-else.csv"}}),
        CONFIG,
    )

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert "not editable" in tool_messages[0].content


def test_no_edits_key_behaves_exactly_as_before(stack, saver, csv_path):
    """Absent or `{}` edits means 'as proposed' — every existing gate keeps working."""
    spec = profile_dataset(csv_path)
    script = [
        AIMessage(content="", tool_calls=[tool_call("confirm_data_profile", {"spec": spec})]),
        AIMessage(content="Approved."),
    ]
    _model, graph = _graph(stack, script, saver)

    graph.invoke({"messages": [HumanMessage(content="use it")]}, CONFIG)
    final = graph.invoke(Command(resume={"approved": True}), CONFIG)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    approved = json.loads(tool_messages[0].content)
    assert approved["task_name"] == spec["task_name"]
    assert "human_edits" not in approved
