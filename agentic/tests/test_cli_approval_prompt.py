"""`_prompt_approval`: the terminal's approval gate, including `field=value` edits
(Phase 13 §5) staged before the final y/N.
"""

import builtins

import pytest

from adaptrna_agentic.cli.chat import _parse_edit_value, _prompt_approval


def _replies(monkeypatch, *values):
    it = iter(values)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


def _request(spec=None, **details):
    if spec is not None:
        details = {"spec": spec, **details}
    return {"requests": [{"summary": "do the thing", "details": details}]}


def test_plain_yes_approves_with_no_edits(monkeypatch):
    _replies(monkeypatch, "y")

    assert _prompt_approval(_request()) == {"approved": True}


def test_empty_reply_declines(monkeypatch):
    _replies(monkeypatch, "")

    decision = _prompt_approval(_request())

    assert decision["approved"] is False


def test_n_declines(monkeypatch):
    _replies(monkeypatch, "n")

    decision = _prompt_approval(_request())

    assert decision["approved"] is False


def test_field_value_stages_an_edit_and_reprompts(monkeypatch):
    _replies(monkeypatch, "spec.task_name=donor_sites", "y")

    decision = _prompt_approval(_request())

    assert decision == {"approved": True, "edits": {"spec.task_name": "donor_sites"}}


def test_multiple_edits_accumulate(monkeypatch):
    # _prompt_approval parses raw text without knowing field semantics -- '1' parses as
    # the number 1 here; _apply_edits (orchestrator.py) is what coerces a numeric-looking
    # edit back to a string when the field it targets is itself a string (positive_class
    # is a class label, not a number). See test_approval_edits.py for that coercion.
    _replies(monkeypatch, "spec.task_name=donor_sites", "spec.positive_class=1", "y")

    decision = _prompt_approval(_request())

    assert decision["edits"] == {
        "spec.task_name": "donor_sites", "spec.positive_class": 1,
    }


def test_declining_after_staged_edits_drops_them(monkeypatch):
    _replies(monkeypatch, "spec.task_name=x", "n")

    decision = _prompt_approval(_request())

    assert decision["approved"] is False
    assert "edits" not in decision


def test_eof_declines_immediately(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _raise)

    decision = _prompt_approval(_request())

    assert decision["approved"] is False


def test_spec_block_is_printed(monkeypatch, capsys):
    _replies(monkeypatch, "y")
    spec = {
        "path": "/tmp/data.csv", "format": {"rows": 40},
        "sequence_column": "sequence", "label_column": "label",
        "target_type": "binary", "classes": ["0", "1"], "class_counts": {"0": 20, "1": 20},
        "positive_class": "1", "alphabet": "rna",
        "length": {"min": 32, "median": 32, "max": 32},
        "ignored_columns": ["source"],
        "split": {
            "mode": "random", "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42, "stratify": True,
            "row_counts": {"train": 32, "val": 4, "test": 4},
        },
        "head": {
            "kind": "cls_classifier", "loss": "binary_cross_entropy_with_logits",
            "primary_metric": "test/f1_score",
        },
        "task_name": "data",
    }

    _prompt_approval(_request(spec=spec))

    out = capsys.readouterr().out
    assert "sequence:  sequence" in out
    assert "positive: '1'" in out
    assert "ignored:   source" in out
    assert "random 80/10/10" in out
    assert "task:      data" in out


@pytest.mark.parametrize("text,expected", [
    ("42", 42),
    ("0.7", 0.7),
    ("true", True),
    ("false", False),
    ("multiclass", "multiclass"),
    ('{"train": ["human"]}', {"train": ["human"]}),
    ('["a", "b"]', ["a", "b"]),
])
def test_parse_edit_value(text, expected):
    assert _parse_edit_value(text) == expected
