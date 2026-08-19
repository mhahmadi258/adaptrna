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

    with pytest.raises(ToolHubError, match="'random', 'column' or 'file'"):
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


# ---------------------------------------------------------------------- format.header / separator

def test_editing_format_header_is_honoured_on_reconfirm(tmp_path):
    """A human who disagrees with the auto-detected header-ness can correct it at the
    gate, and confirm_profile must re-read the file that way rather than trusting the
    original detection. The header row is literally named "0","1" so the column
    identifiers stay valid under either reading -- isolating the header edit from any
    column-name edit -- and proven by row count, which shifts by exactly one depending
    on whether row 1 is treated as data or consumed as a header."""
    path = tmp_path / "data.csv"
    rows = ["0,1"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)
    assert spec["format"]["header"] is True
    assert spec["sequence_column"] == "0" and spec["label_column"] == "1"
    assert sum(spec["class_counts"].values()) == 40

    spec["format"]["header"] = False  # the human insists row 1 ("0,1") is data

    approved = confirm_profile(spec)

    assert sum(approved["class_counts"].values()) == 41


def test_editing_format_separator_is_honoured_on_reconfirm(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)
    assert spec["format"]["separator"] == ","

    # A malicious/incorrect edit to the wrong separator must fail loudly, not silently
    # misparse -- proving the edit really does take effect on re-validation.
    spec["format"]["separator"] = ";"

    with pytest.raises(ToolHubError):
        confirm_profile(spec)


# ---------------------------------------------------------------------- split.mode == "file"

@pytest.fixture
def train_and_val_csv(tmp_path):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train_path.write_text(
        "\n".join(["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(40)]) + "\n"
    )
    val_path.write_text(
        "\n".join(["sequence,label"] + [f"{'ACGC' * 10},{i % 2}" for i in range(10)]) + "\n"
    )
    return train_path, val_path


def test_file_mode_is_accepted_and_row_counts_recomputed(train_and_val_csv):
    train_path, val_path = train_and_val_csv
    spec = profile_dataset(train_path, validation_path=val_path)

    approved = confirm_profile(spec)

    assert approved["split"]["mode"] == "file"
    assert approved["split"]["row_counts"] == {"train": 40, "val": 10, "test": 0}


def test_file_mode_missing_validation_path_is_rejected(binary_csv):
    spec = profile_dataset(binary_csv)
    spec["split"] = {
        "mode": "file", "validation_path": None, "test_fraction": 0.0,
        "fractions": None, "seed": None, "stratify": None, "column": None, "mapping": None,
    }

    with pytest.raises(ToolHubError, match="validation_path"):
        confirm_profile(spec)


def test_file_mode_unreadable_validation_path_is_rejected(binary_csv, tmp_path):
    spec = profile_dataset(binary_csv)
    spec["split"] = {
        "mode": "file", "validation_path": str(tmp_path / "nowhere.csv"), "test_fraction": 0.0,
        "fractions": None, "seed": None, "stratify": None, "column": None, "mapping": None,
    }

    with pytest.raises(ToolHubError, match="does not exist"):
        confirm_profile(spec)


def test_file_mode_cross_file_leakage_is_warned_on_reconfirm(tmp_path):
    shared = "ACGU" * 10
    bases = "ACGU"

    def _seq(i):
        return "".join(bases[(i + j) % 4] for j in range(32))

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train_path.write_text(
        "\n".join(["sequence,label", f"{shared},0"]
                   + [f"{_seq(i)},{i % 2}" for i in range(39)]) + "\n"
    )
    val_path.write_text(
        "\n".join(["sequence,label", f"{shared},1"]
                   + [f"{_seq(i + 100)},{i % 2}" for i in range(9)]) + "\n"
    )

    spec = profile_dataset(train_path, validation_path=val_path)
    approved = confirm_profile(spec)

    assert any("train and val" in w for w in approved["warnings"])
