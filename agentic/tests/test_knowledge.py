"""The knowledge base must keep carrying the load-bearing numbers: this suite fails
loudly if an edit drops one, because the recommender has no other source for them.

Phase 13: `arms:`/`universal:` are unchanged (validated across tasks, they transfer);
`tasks:` is gone (there are no known tasks any more); `generic:` and `target_shapes.yaml`
replace it.
"""

import pytest

from adaptrna_agentic.knowledge import (
    arm,
    derived,
    generic_knowledge,
    load_knowledge,
    target_shape,
    universal,
)


def test_lora_validated_settings_present():
    lora = arm("lora")

    assert lora["optim"]["lr"] == pytest.approx(3.0e-4)
    assert lora["trainer"]["gradient_clip_val"] == 1.0
    assert lora["lora"]["layer_stride"] == 3
    assert lora["lora"]["r"] == 16
    assert lora["lora"]["alpha"] == 32


def test_lora_collapse_failure_mode_documented():
    modes = arm("lora")["failure_modes"]
    collapse = next(m for m in modes if "1e-3" in m["setting"])

    assert "constant-output" in collapse["symptom"]
    assert "3e-4" in collapse["remedy"]


def test_full_ft_settings_and_failure_mode():
    full_ft = arm("full_ft")

    assert full_ft["optim"]["lr"] == pytest.approx(1.0e-5)
    destroyed = next(m for m in full_ft["failure_modes"] if "1e-4" in m["setting"])
    assert "R^2" in destroyed["symptom"]


def test_unknown_arm_raises():
    with pytest.raises(KeyError, match="Unknown training arm"):
        arm("no_such_arm")


def test_precision_and_nondeterminism_note():
    assert universal()["trainer"]["precision"] == "bf16-mixed"

    note = universal()["nondeterminism"]["note"]
    assert "95.21" in note and "95.82" in note


# ---------------------------------------------------------------------- generic:

def test_generic_has_no_reference_band():
    """There are no known tasks any more; a band is a property of a task and a dataset
    and cannot be invented."""
    reference = generic_knowledge()["reference"]

    assert reference["band"] is None
    assert reference["tolerance"] == 0.0
    assert reference["sources"]


def test_generic_caveats_state_the_first_run_is_a_baseline():
    caveats = " ".join(generic_knowledge()["caveats"])

    assert "baseline" in caveats


def test_derived_batch_size_rule():
    rule = derived("batch_size")

    assert rule["rule"] == "piecewise_on_median_length"
    assert rule["table"] == [[128, 64], [512, 32], [1024, 16], [2048, 8]]
    assert rule["fallback"] == 4
    assert rule["why"]


def test_derived_max_epochs_rule():
    rule = derived("max_epochs")

    assert rule["target_steps"] == [1000, 10000]
    assert rule["clamp"] == [1, 20]
    assert rule["why"]


def test_derived_num_workers_rule():
    rule = derived("num_workers")

    assert rule["value"] == 8
    assert rule["why"]


def test_unknown_derivation_rule_raises():
    with pytest.raises(KeyError, match="No derivation rule"):
        derived("no_such_rule")


# ---------------------------------------------------------------------- target_shapes.yaml

@pytest.mark.parametrize("target_type", ["binary", "multiclass", "regression"])
def test_every_target_shape_names_head_loss_metrics(target_type):
    shape = target_shape(target_type)

    for key in ("head", "loss", "metrics", "extract_features", "predict_output",
                "primary_metric", "adapter_state"):
        assert shape.get(key), f"{target_type} shape missing {key}"
    assert isinstance(shape["pad_sensitive"], bool)


def test_regression_is_pad_sensitive_and_binary_is_not():
    assert target_shape("regression")["pad_sensitive"] is True
    assert target_shape("binary")["pad_sensitive"] is False
    assert target_shape("multiclass")["pad_sensitive"] is False


def test_binary_adapter_state_names_positive_class():
    assert "positive_class" in target_shape("binary")["adapter_state"]


def test_unknown_target_type_raises_naming_the_supported_ones():
    with pytest.raises(KeyError, match="binary"):
        target_shape("per_position")


def test_target_shapes_cover_exactly_the_three_supported_types():
    assert set(load_knowledge()["target_shapes"]) == {"binary", "multiclass", "regression"}
