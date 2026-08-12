"""ViennaRNA wrapper — the hand-written reference external tool, and the template the
Phase 6 ToolSmith imitates.

Contract compliance (see contract.py): a module-level `SPEC`, plain typed functions with
JSON-scalar kwargs and dict returns, and input validation that runs *before* `import RNA`
so a missing package fails at the call boundary and validation tests run everywhere.

Golden cases are captured against the installed ViennaRNA version (recorded in the
manifest provenance at registration) — the `AAAA…` no-structure case is a priori, the
rest were pinned from a live capture at implementation time.
"""

from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec,
    FunctionSpec,
    GoldenCase,
    PackageSpec,
)

_VALID_BASES = set("ACGU")


def _clean(sequence: str, label: str = "sequence") -> str:
    """Validate + normalise (upper-case, T→U). Runs before any ViennaRNA import."""
    seq = (sequence or "").strip().upper().replace("T", "U")
    if not seq:
        raise ValueError(f"Empty {label}: provide at least one base (A, C, G, U or T).")

    invalid = sorted(set(seq) - _VALID_BASES)
    if invalid:
        raise ValueError(
            f"Invalid characters {invalid} in {label}; expected only A, C, G, U or T."
        )

    return seq


def fold(sequence: str) -> dict:
    """Minimum-free-energy secondary structure of one RNA sequence."""
    seq = _clean(sequence)

    import RNA

    structure, mfe = RNA.fold(seq)
    return {"structure": structure, "mfe": round(float(mfe), 2)}


def cofold(sequence_a: str, sequence_b: str) -> dict:
    """Minimum-free-energy structure of two RNA strands hybridising as a dimer."""
    a = _clean(sequence_a, "sequence_a")
    b = _clean(sequence_b, "sequence_b")

    import RNA

    structure, mfe = RNA.cofold(f"{a}&{b}")
    return {"structure": structure, "mfe": round(float(mfe), 2)}


SPEC = ExternalToolSpec(
    name="vienna",
    description="Thermodynamic RNA secondary-structure prediction via ViennaRNA.",
    package=PackageSpec(pip="ViennaRNA", import_name="RNA"),
    functions=(
        FunctionSpec(
            name="fold",
            description=(
                "Predict the minimum-free-energy secondary structure of one RNA sequence. "
                "Returns the dot-bracket structure and its free energy in kcal/mol."
            ),
            golden=(
                # A priori: a homopolymer cannot pair with itself.
                GoldenCase(
                    args={"sequence": "AAAAAAAAAAAA"},
                    expect={"structure": "............", "mfe": {"approx": 0.0, "tol": 0.01}},
                ),
                # Captured against ViennaRNA 2.7.2 (2026-08-12); the classic hairpin.
                GoldenCase(
                    args={"sequence": "GGGGAAAACCCC"},
                    expect={"structure": "((((....))))",
                            "mfe": {"approx": -5.4, "tol": 0.5}},
                ),
            ),
        ),
        FunctionSpec(
            name="cofold",
            description=(
                "Predict the minimum-free-energy structure of two RNA strands hybridising "
                "as a dimer. Returns the dot-bracket structure and its energy in kcal/mol."
            ),
            golden=(
                # Captured against ViennaRNA 2.7.2 (2026-08-12). Note the returned
                # structure spans both strands with the '&' dropped (16 chars for 8+8).
                GoldenCase(
                    args={"sequence_a": "GGGGGGGG", "sequence_b": "CCCCCCCC"},
                    expect={"structure": "(((((((())))))))",
                            "mfe": {"approx": -19.0, "tol": 0.5}},
                ),
            ),
        ),
    ),
)
