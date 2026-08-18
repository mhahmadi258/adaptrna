"""The external-tool wrapper contract, exercised end-to-end against a real fixture wrapper
(`fixtures/validating_external.py`) rather than only through mocked entries — validation
runs before the wrapped package is ever imported, the goldens are truthful, and the SPEC
shape is what `contract.load_spec` requires.

`test_external_contract.py` covers `load_spec`/`run_golden` generically against
`dummy_external`; this file is about one wrapper's own behaviour end to end, and about
proving the validate-before-import ordering directly rather than assuming it.
"""

import sys

import pytest

from adaptrna_agentic.toolhub.external.contract import _compare
from fixtures.validating_external import SPEC, _clean, checksum, checksum_pair


def test_clean_normalises_whitespace():
    assert _clean("  hello  ") == "hello"


def test_empty_text_rejected_before_import():
    with pytest.raises(ValueError, match="Empty"):
        checksum("   ")


def test_non_printable_text_rejected_before_import():
    with pytest.raises(ValueError, match="Non-printable"):
        checksum("hi\x00there")


def test_checksum_pair_validates_both_strands():
    with pytest.raises(ValueError, match="text_b"):
        checksum_pair("hi", "")


def test_validation_runs_before_the_package_import(monkeypatch):
    """The defining property of the contract's rule 3: a bad argument must never reach
    the `import zlib` line. Proven by poisoning the import, not merely by observing that
    validation happens to run first for one already-working case."""

    class _Poison:
        def __getattr__(self, name):
            raise AssertionError("zlib was imported before validation ran")

    monkeypatch.setitem(sys.modules, "zlib", _Poison())

    with pytest.raises(ValueError, match="Empty"):
        checksum("")


def test_spec_contract_shape():
    assert SPEC.name == "checksum"
    assert SPEC.package.import_name == "zlib"
    assert {fn.name for fn in SPEC.functions} == {"checksum", "checksum_pair"}
    assert all(fn.golden for fn in SPEC.functions)


def test_pinned_goldens_stay_truthful():
    functions = {"checksum": checksum, "checksum_pair": checksum_pair}
    for spec in SPEC.functions:
        for case in spec.golden:
            result = functions[spec.name](**case.args)
            problems = _compare(result, case.expect)
            assert not problems, f"{spec.name} {case.args}: {problems}"
