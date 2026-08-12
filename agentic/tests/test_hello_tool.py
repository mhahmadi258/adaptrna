"""The gc_content demo tool — pure python, no model involved."""

import pytest

from adaptrna_agentic.agents.hello import gc_content


def _run(sequence: str) -> str:
    return gc_content.invoke({"sequence": sequence})


def test_all_gc():
    assert _run("GGCC") == "1.000"


def test_no_gc():
    assert _run("AUAU") == "0.000"


def test_mixed_case_dod_sequence():
    # The Definition-of-Done demo sequence: 7 G/C of 12 bases.
    assert _run("ggcAUUACGGcu") == "0.583"


def test_dna_thymine_accepted():
    assert _run("GCTA") == "0.500"


def test_invalid_characters_rejected():
    with pytest.raises(Exception, match="Invalid characters"):
        _run("GGXX")


def test_empty_sequence_rejected():
    with pytest.raises(Exception, match="Empty sequence"):
        _run("   ")
