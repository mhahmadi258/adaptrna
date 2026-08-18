"""`covers(spec)` is a whitelist predicate, never allowed to claim coverage it lacks
(Phase 13 §7.2/§13): it must accept every spec the gate can produce and reject a spec
carrying a field or value it does not handle.
"""

import copy

import pytest

from adaptrna_agentic.codegen.templates import render as templates
from fixtures.template_specs import TEMPLATE_SPECS


@pytest.mark.parametrize("case", sorted(TEMPLATE_SPECS))
def test_covers_every_spec_the_template_declares_support_for(case):
    assert templates.covers(TEMPLATE_SPECS[case])


def test_rejects_unsupported_target_type():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["target_type"] = "per_position"

    assert not templates.covers(spec)


def test_rejects_multiclass_with_too_many_classes():
    spec = copy.deepcopy(TEMPLATE_SPECS["multiclass_random"])
    spec["classes"] = [str(i) for i in range(25)]

    assert not templates.covers(spec)


def test_rejects_binary_with_wrong_class_count():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["classes"] = ["0", "1", "2"]

    assert not templates.covers(spec)


def test_rejects_unsupported_split_mode():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["split"] = {"mode": "kfold"}

    assert not templates.covers(spec)


def test_rejects_random_split_fractions_that_do_not_sum_to_one():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["split"]["fractions"] = {"train": 0.8, "val": 0.1, "test": 0.2}

    assert not templates.covers(spec)


def test_rejects_random_split_missing_a_fraction_key():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["split"]["fractions"] = {"train": 0.9, "val": 0.1}

    assert not templates.covers(spec)


def test_rejects_column_split_without_a_mapping():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_column"])
    spec["split"] = {"mode": "column", "column": "source", "mapping": {}}

    assert not templates.covers(spec)


def test_rejects_column_split_with_an_empty_split_bucket():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_column"])
    spec["split"]["mapping"] = {"train": ["human"], "val": [], "test": ["fly"]}

    assert not templates.covers(spec)


def test_rejects_unsupported_separator():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["format"] = {"separator": "|", "compression": None}

    assert not templates.covers(spec)


def test_rejects_missing_primary_metric():
    spec = copy.deepcopy(TEMPLATE_SPECS["regression_random"])
    spec["head"] = {}

    assert not templates.covers(spec)


def test_rejects_task_name_that_is_not_a_valid_identifier():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["task_name"] = "not a valid name"

    assert not templates.covers(spec)


def test_render_raises_rather_than_silently_emitting_code_for_an_uncovered_spec():
    spec = copy.deepcopy(TEMPLATE_SPECS["binary_random"])
    spec["target_type"] = "per_position"

    with pytest.raises(ValueError):
        templates.render(spec)
