"""`AdapterRuntime`'s output validators, keyed by target type (Phase 13 §10) rather than a
hardcoded task-name set — every generated task gets one, not just one shipped example did.
"""

from adaptrna_agentic.toolhub.manifest import ToolEntry
from adaptrna_agentic.toolhub.runtime import (
    _validate_binary,
    _validate_generic,
    _validate_multiclass,
    _validate_regression,
    _validator_for,
)


def _entry(target_type=None, classes=None, pad_sensitive=False):
    provenance = {}
    if target_type is not None:
        head = {"pad_sensitive": pad_sensitive}
        if classes is not None:
            provenance["spec"] = {"target_type": target_type, "classes": classes, "head": head}
        else:
            provenance["spec"] = {"target_type": target_type, "head": head}

    return ToolEntry(name="t", type="adapter", state="active", description="d",
                     task="t", provenance=provenance)


# ---------------------------------------------------------------------- selection

def test_no_spec_falls_back_to_generic():
    assert _validator_for(_entry(target_type=None)) is _validate_generic


def test_binary_spec_selects_the_binary_validator():
    assert _validator_for(_entry(target_type="binary")) is _validate_binary


def test_regression_spec_selects_the_regression_validator():
    assert _validator_for(_entry(target_type="regression")) is _validate_regression


def test_multiclass_spec_selects_a_multiclass_validator():
    validator = _validator_for(_entry(target_type="multiclass", classes=["exon", "intron"]))

    assert validator(
        [{"label": "exon", "probabilities": {"exon": 0.9, "intron": 0.1}}], ["ACGU"]
    ) == []


def test_unknown_target_type_falls_back_to_generic():
    assert _validator_for(_entry(target_type="per_position")) is _validate_generic


# ---------------------------------------------------------------------- binary

def test_binary_accepts_valid_probabilities():
    assert _validate_binary([0.0, 0.5, 1.0], ["a", "b", "c"]) == []


def test_binary_rejects_out_of_range_probabilities():
    problems = _validate_binary([1.5], ["a"])
    assert any("outside [0, 1]" in p for p in problems)


def test_binary_rejects_non_finite_values():
    problems = _validate_binary([float("nan")], ["a"])
    assert any("non-finite" in p for p in problems)


def test_binary_rejects_non_numeric_output():
    problems = _validate_binary([{"not": "a number"}], ["a"])
    assert any("expected one probability" in p for p in problems)


# ---------------------------------------------------------------------- regression

def test_regression_accepts_any_finite_scalar():
    assert _validate_regression([-100.0, 0.0, 5000.0], ["a", "b", "c"]) == []


def test_regression_rejects_non_finite_predictions():
    problems = _validate_regression([float("inf")], ["a"])
    assert any("non-finite" in p for p in problems)


# ---------------------------------------------------------------------- multiclass

def test_multiclass_accepts_a_recorded_label():
    problems = _validate_multiclass(
        [{"label": "intron", "probabilities": {"exon": 0.1, "intron": 0.9}}],
        ["ACGU"], classes=["exon", "intron"],
    )
    assert problems == []


def test_multiclass_rejects_a_label_outside_the_recorded_classes():
    problems = _validate_multiclass(
        [{"label": "nonsense", "probabilities": {}}], ["ACGU"], classes=["exon", "intron"],
    )
    assert any("not one of the recorded classes" in p for p in problems)


def test_multiclass_rejects_output_with_no_label():
    problems = _validate_multiclass([0.5], ["ACGU"], classes=["exon", "intron"])
    assert any("expected a class label" in p for p in problems)


# ---------------------------------------------------------------------- generic fallback

def test_generic_only_checks_finiteness():
    assert _validate_generic([-5.0, 1e9], ["a", "b"]) == []
    assert _validate_generic([float("nan")], ["a"]) != []
