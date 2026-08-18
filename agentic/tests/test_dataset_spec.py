"""`DatasetSpec` validation: `confirm_profile` re-validates and recomputes on approval.

Phase 13 §3/§4 — the load-bearing part of gate 1: a spec the human edited must be
recomputed against the real file, never trusted as given. `profile_dataset`'s proposal is
exercised in `test_profiler.py`; this file is about what happens to it at the gate.
"""

import pytest

from adaptrna_agentic.profiling.profiler import (
    SPEC_SOURCE, confirm_profile, profile_dataset,
)
from adaptrna_agentic.toolhub.errors import ToolHubError


@pytest.fixture
def binary_csv(tmp_path):
    path = tmp_path / "binary.csv"
    rows = ["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_confirm_profile_refuses_an_unstamped_spec(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["source"] = "made_up"

    with pytest.raises(ToolHubError, match="profile_dataset"):
        confirm_profile(spec)


def test_confirm_profile_refuses_a_spec_with_no_source(binary_csv):
    spec = profile_dataset(binary_csv)
    del spec["source"]

    with pytest.raises(ToolHubError, match="profile_dataset"):
        confirm_profile(spec)


def test_confirm_profile_restamps_on_approval(binary_csv):
    spec = profile_dataset(binary_csv)

    approved = confirm_profile(spec)

    assert approved["source"] == SPEC_SOURCE


def test_unknown_sequence_column_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["sequence_column"] = "no_such_column"

    with pytest.raises(ToolHubError, match="no_such_column"):
        confirm_profile(spec)


def test_unknown_label_column_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["label_column"] = "no_such_column"

    with pytest.raises(ToolHubError, match="no_such_column"):
        confirm_profile(spec)


def test_sequence_and_label_column_must_differ(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["label_column"] = spec["sequence_column"]

    with pytest.raises(ToolHubError, match="must be different"):
        confirm_profile(spec)


def test_unsupported_target_type_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["target_type"] = "per_position"

    with pytest.raises(ToolHubError, match="not a supported target type"):
        confirm_profile(spec)


def test_task_name_collision_is_rejected(binary_csv, monkeypatch):
    spec = profile_dataset(binary_csv)
    monkeypatch.setattr(
        "adaptrna_agentic.codegen.discovery.custom_task_names",
        lambda: [spec["task_name"]],
    )

    with pytest.raises(ToolHubError, match="already exists"):
        confirm_profile(spec)


def test_invalid_task_name_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["task_name"] = "Not Valid!"

    with pytest.raises(ToolHubError, match="not a valid task name"):
        confirm_profile(spec)


def test_bad_fractions_are_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"]["fractions"] = {"train": 0.8, "val": 0.1, "test": 0.2}

    with pytest.raises(ToolHubError, match="sum to"):
        confirm_profile(spec)


def test_fractions_missing_a_key_are_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"]["fractions"] = {"train": 0.9, "val": 0.1}

    with pytest.raises(ToolHubError, match="train, val and test"):
        confirm_profile(spec)


def test_negative_fraction_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"]["fractions"] = {"train": 1.1, "val": 0.1, "test": -0.2}

    with pytest.raises(ToolHubError, match="non-negative"):
        confirm_profile(spec)


def test_column_split_with_no_matching_values_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"] = {"mode": "column", "column": "sequence", "mapping": {}}

    with pytest.raises(ToolHubError, match="mapping must name"):
        confirm_profile(spec)


def test_unknown_split_mode_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"] = {"mode": "kfold"}

    with pytest.raises(ToolHubError, match="'random' or 'column'"):
        confirm_profile(spec)


def test_switching_target_type_recomputes_classes_and_head(binary_csv):
    """A user who switches target_type gets a spec whose head/classes follow, rather
    than a spec that says one thing and trains another."""
    spec = profile_dataset(binary_csv)
    spec["target_type"] = "multiclass"

    approved = confirm_profile(spec)

    assert approved["target_type"] == "multiclass"
    assert approved["head"]["primary_metric"] == "test/macro_f1"
    assert approved["classes"] == ["0", "1"]
    assert approved["positive_class"] is None


def test_binary_with_wrong_class_count_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["target_type"] = "binary"
    spec["classes"] = ["0", "1", "2"]

    with pytest.raises(ToolHubError, match="exactly two classes"):
        confirm_profile(spec)


def test_switching_split_mode_recomputes_row_counts(tmp_path):
    path = tmp_path / "grouped.csv"
    rows = ["sequence,label,grp"]
    for i in range(40):
        grp = "human" if i < 30 else "fly"
        rows.append(f"{'ACGU' * 10},{i % 2},{grp}")
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)
    spec["split"] = {
        "mode": "column", "column": "grp",
        "mapping": {"train": ["human"], "test": ["fly"]},
    }

    approved = confirm_profile(spec)

    assert approved["split"]["mode"] == "column"
    assert approved["split"]["row_counts"]["train"] == 30
    assert approved["split"]["row_counts"]["test"] == 10
    assert approved["split"]["row_counts"]["val"] == 0


def test_row_counts_recomputed_for_edited_fractions(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"]["fractions"] = {"train": 0.7, "val": 0.15, "test": 0.15}

    approved = confirm_profile(spec)

    counts = approved["split"]["row_counts"]
    assert counts["train"] + counts["val"] + counts["test"] == 40


def test_test_fraction_zero_is_allowed_and_warned(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"]["fractions"] = {"train": 0.9, "val": 0.1, "test": 0.0}

    approved = confirm_profile(spec)

    assert approved["split"]["row_counts"]["test"] == 0
    assert any("no held-out number" in w for w in approved["warnings"])


def test_edited_positive_class_is_honoured(binary_csv):
    spec = profile_dataset(binary_csv)
    other = next(c for c in spec["classes"] if c != spec["positive_class"])
    spec["positive_class"] = other

    approved = confirm_profile(spec)

    assert approved["positive_class"] == other


def test_positive_class_not_in_classes_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["positive_class"] = "not_a_real_class"

    with pytest.raises(ToolHubError, match="positive_class"):
        confirm_profile(spec)
