"""Canonical `DatasetSpec` fixtures for the template renderer's tests (Phase 13, §7.3).

One spec per (target type × split mode) combination the template declares coverage for —
shared between the golden-file render tests and the `covers()` tests, so both exercise
exactly the same specs the template is meant to handle.
"""

RANDOM_SPLIT = {
    "mode": "random",
    "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
    "seed": 42,
    "stratify": True,
}

COLUMN_SPLIT = {
    "mode": "column",
    "column": "source",
    "mapping": {"train": ["human", "mouse"], "val": ["fly"], "test": ["zebrafish"]},
}

_BASE = {
    "sequence_column": "sequence",
    "label_column": "label",
    "path": "/home/user/data/example.csv",
    "format": {"separator": ",", "compression": None},
}


def _spec(target_type, split, classes, metric, task_name):
    spec = dict(_BASE)
    spec["target_type"] = target_type
    spec["classes"] = classes
    spec["head"] = {"primary_metric": metric}
    spec["split"] = split
    spec["task_name"] = task_name
    spec["tool_description"] = f"{target_type} target from a flat sequence/label table"
    return spec


TEMPLATE_SPECS = {
    "binary_random": _spec("binary", RANDOM_SPLIT, ["0", "1"], "test/f1_score", "binary_random"),
    "binary_column": _spec("binary", COLUMN_SPLIT, ["0", "1"], "test/f1_score", "binary_column"),
    "multiclass_random": _spec(
        "multiclass", RANDOM_SPLIT, ["0", "1", "2"], "test/macro_f1", "multiclass_random"
    ),
    "multiclass_column": _spec(
        "multiclass", COLUMN_SPLIT, ["0", "1", "2"], "test/macro_f1", "multiclass_column"
    ),
    "regression_random": _spec("regression", RANDOM_SPLIT, None, "test/mse", "regression_random"),
    "regression_column": _spec("regression", COLUMN_SPLIT, None, "test/mse", "regression_column"),
}
