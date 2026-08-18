"""`profile_dataset`'s `similar_tasks`: the replacement for `layout_match` (Phase 13, D9).

A suggestion presented at gate 1, built entirely from the user's own landed tasks' own
spec.json files — never from a shipped template. Weak matches are not reported at all: a
wrong suggestion here costs a wrong training run.
"""

import json

import pytest

from adaptrna_agentic.profiling.profiler import profile_dataset


def _land(tmp_path, monkeypatch, task_name, spec):
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path / "adaptrna_custom")
    task_dir = tmp_path / "adaptrna_custom" / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.py").write_text("# landed")
    (task_dir / "spec.json").write_text(json.dumps(spec))


def _csv(tmp_path, name="new_data.csv", n=40):
    path = tmp_path / name
    rows = ["sequence,label"] + [f"{'ACGU' * 10},{i % 2}" for i in range(n)]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_no_landed_tasks_means_no_suggestions(tmp_path, monkeypatch):
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path / "adaptrna_custom")

    spec = profile_dataset(_csv(tmp_path))

    assert spec["similar_tasks"] == []


def test_matching_columns_and_target_type_is_offered(tmp_path, monkeypatch):
    _land(tmp_path, monkeypatch, "donor_sites", {
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "binary", "classes": ["0", "1"],
        "length": {"median": 400},
    })

    spec = profile_dataset(_csv(tmp_path))

    assert len(spec["similar_tasks"]) == 1
    match = spec["similar_tasks"][0]
    assert match["task"] == "donor_sites"
    assert match["spec_path"] == "adaptrna_custom/tasks/donor_sites/spec.json"
    assert "same columns" in match["why"]
    assert "same target type" in match["why"]


def test_different_columns_is_not_offered(tmp_path, monkeypatch):
    _land(tmp_path, monkeypatch, "other_task", {
        "sequence_column": "seq", "label_column": "y",
        "target_type": "binary", "classes": ["0", "1"],
    })

    spec = profile_dataset(_csv(tmp_path))

    assert spec["similar_tasks"] == []


def test_different_target_type_is_not_offered(tmp_path, monkeypatch):
    _land(tmp_path, monkeypatch, "other_task", {
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "regression",
    })

    spec = profile_dataset(_csv(tmp_path))

    assert spec["similar_tasks"] == []


def test_a_landed_task_with_no_spec_json_never_matches_and_is_not_an_error(tmp_path, monkeypatch):
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path / "adaptrna_custom")
    task_dir = tmp_path / "adaptrna_custom" / "tasks" / "predates_phase_13"
    task_dir.mkdir(parents=True)
    (task_dir / "task.py").write_text("# landed before spec.json existed")

    spec = profile_dataset(_csv(tmp_path))

    assert spec["similar_tasks"] == []


def test_an_unreadable_spec_json_never_matches_and_is_not_an_error(tmp_path, monkeypatch):
    from adaptrna_agentic.codegen import discovery

    monkeypatch.setattr(discovery, "CUSTOM_ROOT", tmp_path / "adaptrna_custom")
    task_dir = tmp_path / "adaptrna_custom" / "tasks" / "corrupted"
    task_dir.mkdir(parents=True)
    (task_dir / "task.py").write_text("# landed")
    (task_dir / "spec.json").write_text("not valid json{{{")

    spec = profile_dataset(_csv(tmp_path))

    assert spec["similar_tasks"] == []


def test_matching_class_count_and_median_length_score_higher(tmp_path, monkeypatch):
    _land(tmp_path, monkeypatch, "close_match", {
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "binary", "classes": ["0", "1"],
        "length": {"median": 32},  # the fixture CSV's own sequences are 40 nt (ACGU * 10)
    })
    _land(tmp_path, monkeypatch, "columns_and_type_only", {
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "binary", "classes": ["0", "1", "2"],  # different class count
        "length": {"median": 9999},  # wildly different length
    })

    spec = profile_dataset(_csv(tmp_path))

    scores = {m["task"]: m["score"] for m in spec["similar_tasks"]}
    assert scores["close_match"] > scores["columns_and_type_only"]


def test_results_are_sorted_best_match_first(tmp_path, monkeypatch):
    _land(tmp_path, monkeypatch, "weaker", {
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "binary", "classes": ["0", "1", "2"],
    })
    _land(tmp_path, monkeypatch, "stronger", {
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "binary", "classes": ["0", "1"],
        "length": {"median": 40},
    })

    spec = profile_dataset(_csv(tmp_path))

    assert [m["task"] for m in spec["similar_tasks"]] == ["stronger", "weaker"]


def test_similar_tasks_survives_confirm_profile_unchanged(tmp_path, monkeypatch):
    from adaptrna_agentic.profiling.profiler import confirm_profile

    _land(tmp_path, monkeypatch, "donor_sites", {
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "binary", "classes": ["0", "1"],
    })

    proposed = profile_dataset(_csv(tmp_path))
    approved = confirm_profile(proposed)

    assert approved["similar_tasks"] == proposed["similar_tasks"]
