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

FILE_SPLIT = {
    "mode": "file",
    "validation_path": "/home/user/data/example_val.csv",
    "test_fraction": 0.0,
    "fractions": None, "seed": None, "stratify": None, "column": None, "mapping": None,
}

_BASE = {
    "sequence_column": "sequence",
    "label_column": "label",
    "path": "/home/user/data/example.csv",
    "format": {"separator": ",", "compression": None},
}

_HEADERLESS_BASE = {
    "sequence_column": "0",
    "label_column": "1",
    "path": "/home/user/data/example.csv",
    "format": {"separator": ",", "compression": None, "header": False},
}


def _spec(target_type, split, classes, metric, task_name, positive_class=None, base=None):
    spec = dict(base or _BASE)
    spec["target_type"] = target_type
    spec["classes"] = classes
    spec["head"] = {"primary_metric": metric}
    spec["split"] = split
    spec["task_name"] = task_name
    spec["tool_description"] = f"{target_type} target from a flat sequence/label table"
    if positive_class is not None:
        spec["positive_class"] = positive_class
    return spec


# Positive class is deliberately '0' -- first in the `classes` list, not last. If polarity
# ever regressed to following list order instead of `positive_class`, these specs would
# catch it (see test_binary_positive_class_is_independent_of_class_order below).
TEMPLATE_SPECS = {
    "binary_random": _spec(
        "binary", RANDOM_SPLIT, ["0", "1"], "test/f1_score", "binary_random", positive_class="0"
    ),
    "binary_column": _spec(
        "binary", COLUMN_SPLIT, ["0", "1"], "test/f1_score", "binary_column", positive_class="0"
    ),
    "multiclass_random": _spec(
        "multiclass", RANDOM_SPLIT, ["0", "1", "2"], "test/macro_f1", "multiclass_random"
    ),
    "multiclass_column": _spec(
        "multiclass", COLUMN_SPLIT, ["0", "1", "2"], "test/macro_f1", "multiclass_column"
    ),
    "regression_random": _spec("regression", RANDOM_SPLIT, None, "test/mse", "regression_random"),
    "regression_column": _spec("regression", COLUMN_SPLIT, None, "test/mse", "regression_column"),
    "binary_file": _spec(
        "binary", FILE_SPLIT, ["0", "1"], "test/f1_score", "binary_file", positive_class="0"
    ),
    "binary_headerless": _spec(
        "binary", RANDOM_SPLIT, ["0", "1"], "test/f1_score", "binary_headerless",
        positive_class="0", base=_HEADERLESS_BASE,
    ),
}
