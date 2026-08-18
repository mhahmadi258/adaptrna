"""The `generic.derived` rules, exercised directly: batch size at each length band, the
step-budget epoch rule across a wide range of dataset sizes, and clamping at both ends.

Phase 13 §6/§8 — these are DERIVED from the approved spec, not invented, and every
derived value's rationale line comes from the same `why:` the knowledge base carries.
`test_recommender.py` proves they are actually wired into `recommend()`'s output; this
file is about the arithmetic itself.
"""

import pytest

from adaptrna_agentic.knowledge import derived
from adaptrna_agentic.profiling.recommender import _piecewise_on_median_length, _step_budget


# ---------------------------------------------------------------------- batch size

@pytest.mark.parametrize("median,expected_batch", [
    (25, 64),      # short: <=128
    (100, 64),
    (128, 64),     # exactly on the boundary
    (129, 32),     # just over -> next band
    (400, 32),     # <=512
    (512, 32),     # boundary
    (600, 16),     # <=1024
    (1024, 16),    # boundary
    (1500, 8),     # <=2048
    (2048, 8),     # boundary
    (3000, 4),     # beyond every band -> fallback
])
def test_batch_size_bands(median, expected_batch):
    rule = derived("batch_size")

    assert _piecewise_on_median_length(median, rule["table"], rule["fallback"]) == expected_batch


def test_batch_size_uses_the_knowledge_bases_own_table_and_fallback():
    rule = derived("batch_size")

    assert _piecewise_on_median_length(rule["table"][0][0], rule["table"], rule["fallback"]) \
        == rule["table"][0][1]
    assert _piecewise_on_median_length(rule["table"][-1][0] + 1, rule["table"], rule["fallback"]) \
        == rule["fallback"]


# ---------------------------------------------------------------------- step budget / epochs

@pytest.mark.parametrize("rows_train,batch_size", [
    (500, 32),
    (20_000, 32),
    (500_000, 32),
])
def test_step_budget_lands_inside_the_target_range_or_hits_a_clamp(rows_train, batch_size):
    import math

    rule = derived("max_epochs")
    steps_per_epoch = max(1, math.ceil(rows_train / batch_size))
    epochs = _step_budget(steps_per_epoch, rule["target_steps"], rule["clamp"])

    lo_clamp, hi_clamp = rule["clamp"]
    low, high = rule["target_steps"]
    total_steps = epochs * steps_per_epoch

    assert lo_clamp <= epochs <= hi_clamp
    # Either the step count lands inside the budget, or a clamp bound was hit trying.
    assert (low <= total_steps <= high) or epochs in (lo_clamp, hi_clamp)


def test_step_budget_at_500_rows():
    """A small dataset: few steps per epoch, so many epochs are needed to reach the
    budget's low end -- clamped at the upper bound rather than training forever."""
    rule = derived("max_epochs")
    steps_per_epoch = 16  # 500 rows / batch 32, rounded up

    epochs = _step_budget(steps_per_epoch, rule["target_steps"], rule["clamp"])

    assert epochs == rule["clamp"][1]  # 20: 16 * 63 would be needed to reach 1000


def test_step_budget_at_20k_rows():
    rule = derived("max_epochs")
    steps_per_epoch = 625  # 20,000 rows / batch 32

    epochs = _step_budget(steps_per_epoch, rule["target_steps"], rule["clamp"])
    total_steps = epochs * steps_per_epoch

    assert rule["clamp"][0] <= epochs <= rule["clamp"][1]
    assert rule["target_steps"][0] <= total_steps <= rule["target_steps"][1]


def test_step_budget_at_500k_rows():
    """A huge dataset: one epoch alone vastly exceeds the budget's high end, but epochs
    cannot go below 1 -- the lower clamp is what actually bounds this case."""
    rule = derived("max_epochs")
    steps_per_epoch = 15_625  # 500,000 rows / batch 32

    epochs = _step_budget(steps_per_epoch, rule["target_steps"], rule["clamp"])

    assert epochs == rule["clamp"][0]  # 1


def test_step_budget_clamped_low():
    rule = derived("max_epochs")

    epochs = _step_budget(1, rule["target_steps"], rule["clamp"])

    assert epochs == rule["clamp"][1]


def test_step_budget_clamped_high():
    rule = derived("max_epochs")

    epochs = _step_budget(1_000_000, rule["target_steps"], rule["clamp"])

    assert epochs == rule["clamp"][0]


def test_step_budget_never_returns_zero_or_negative_epochs():
    rule = derived("max_epochs")

    for steps_per_epoch in (1, 10, 1000, 1_000_000):
        assert _step_budget(steps_per_epoch, rule["target_steps"], rule["clamp"]) >= 1
