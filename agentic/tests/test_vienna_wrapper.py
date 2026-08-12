"""The ViennaRNA reference wrapper.

Validation tests run everywhere (validation precedes `import RNA` by contract). The
behavior tests are skipped until ViennaRNA is installed — they switch on automatically
after the Phase 3 DoD run, and assert the pinned goldens stay truthful.
"""

import importlib.util

import pytest

from adaptrna_agentic.toolhub.external.contract import _compare
from adaptrna_agentic.toolhub.external.vienna import SPEC, _clean, cofold, fold

vienna_installed = importlib.util.find_spec("RNA") is not None
needs_vienna = pytest.mark.skipif(not vienna_installed, reason="ViennaRNA not installed")


# ------------------------------------------------------------- validation (no RNA needed)

def test_clean_normalises_case_and_thymine():
    assert _clean("gcta") == "GCUA"


def test_invalid_characters_rejected_before_import():
    with pytest.raises(ValueError, match="Invalid characters"):
        fold("GGXX")


def test_empty_sequence_rejected_before_import():
    with pytest.raises(ValueError, match="Empty"):
        fold("   ")


def test_cofold_validates_both_strands():
    with pytest.raises(ValueError, match="sequence_b"):
        cofold("GGGG", "")


def test_spec_contract_shape():
    assert SPEC.name == "vienna"
    assert SPEC.package.import_name == "RNA"
    assert {fn.name for fn in SPEC.functions} == {"fold", "cofold"}
    assert all(fn.golden for fn in SPEC.functions)


# ------------------------------------------------------------- behavior (needs ViennaRNA)

def _balanced(structure: str) -> bool:
    depth = 0
    for char in structure:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


@needs_vienna
def test_fold_output_shape():
    result = fold("GGGGAAAACCCC")

    assert set(result) == {"structure", "mfe"}
    assert len(result["structure"]) == 12
    assert _balanced(result["structure"])
    assert result["mfe"] <= 0.0


@needs_vienna
def test_homopolymer_has_no_structure():
    result = fold("AAAAAAAAAAAA")

    assert result["structure"] == "." * 12
    assert result["mfe"] == 0.0


@needs_vienna
def test_pinned_goldens_stay_truthful():
    functions = {"fold": fold, "cofold": cofold}
    for spec in SPEC.functions:
        for case in spec.golden:
            result = functions[spec.name](**case.args)
            problems = _compare(result, case.expect)
            assert not problems, f"{spec.name} {case.args}: {problems}"
