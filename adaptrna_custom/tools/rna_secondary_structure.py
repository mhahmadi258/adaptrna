"""ViennaRNA MFE-folding wrapper — predicts RNA secondary structure from an ACGU sequence.

Contract compliance (see toolhub/external/contract.py): a module-level `SPEC`, one plain
module-level callable per declared FunctionSpec taking JSON-scalar keyword arguments and
returning a JSON-serializable dict, and input validation that runs *before* `import RNA`
so a missing package fails at the call boundary (with the install hint) and the validation
tests run even where ViennaRNA is absent.

Golden cases are restricted to values that are certain a priori:
  * homopolymers (poly-A, poly-C) cannot form any base pair, so the MFE structure is all
    dots and the free energy is exactly 0.00 kcal/mol;
  * a single base likewise cannot pair;
  * a GC-clamped tetraloop hairpin `GGGGAAAACCCC` folds to `((((....))))` — the structure
    is unambiguous, and its energy is checked only with a loose tolerance around the value
    captured from a live ViennaRNA 2.x run (the same case pinned by the reference wrapper).
Anything less certain is deliberately left out rather than guessed.
"""

from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec,
    FunctionSpec,
    GoldenCase,
    PackageSpec,
)

_VALID_BASES = set("ACGU")

# Guard rail: ViennaRNA folds in O(n^3) time / O(n^2) memory, so refuse absurd inputs
# at the call boundary instead of hanging an agent turn.
_MAX_LENGTH = 10_000


def _clean(sequence: str, label: str = "sequence") -> str:
    """Validate + normalise an RNA sequence (strip, upper-case, T→U).

    Runs entirely before any ViennaRNA import so that bad inputs raise ValueError even
    when the package is not installed.
    """
    if sequence is None or not isinstance(sequence, str):
        raise ValueError(
            f"Invalid {label}: expected a string of bases, got {type(sequence).__name__}."
        )

    seq = "".join(sequence.split()).upper().replace("T", "U")
    if not seq:
        raise ValueError(f"Empty {label}: provide at least one base (A, C, G, U or T).")

    invalid = sorted(set(seq) - _VALID_BASES)
    if invalid:
        raise ValueError(
            f"Invalid characters {invalid} in {label}; expected only A, C, G, U or T."
        )

    if len(seq) > _MAX_LENGTH:
        raise ValueError(
            f"{label} is {len(seq)} nt, which exceeds the {_MAX_LENGTH} nt limit for "
            f"MFE folding."
        )

    return seq


def predict_structure(sequence: str) -> dict:
    """Predict the MFE secondary structure of one RNA sequence.

    Returns the dot-bracket structure, its free energy in kcal/mol, the normalised
    sequence that was folded, and its length.
    """
    seq = _clean(sequence)

    import RNA

    structure, mfe = RNA.fold(seq)
    return {
        "sequence": seq,
        "length": len(seq),
        "structure": structure,
        "mfe": round(float(mfe), 2),
    }


SPEC = ExternalToolSpec(
    name="viennafold",
    description=(
        "RNA secondary-structure prediction from an ACGU sequence via ViennaRNA: the "
        "minimum-free-energy structure in dot-bracket notation plus its free energy."
    ),
    package=PackageSpec(pip="ViennaRNA", import_name="RNA"),
    functions=(
        FunctionSpec(
            name="predict_structure",
            description=(
                "Predict the minimum-free-energy (MFE) secondary structure of a single "
                "RNA sequence (A/C/G/U; T is accepted and read as U). Returns the "
                "dot-bracket structure, the free energy in kcal/mol, the normalised "
                "sequence and its length."
            ),
            golden=(
                # A priori: adenines cannot pair with each other — no structure, 0 energy.
                GoldenCase(
                    args={"sequence": "AAAAAAAAAAAA"},
                    expect={
                        "sequence": "AAAAAAAAAAAA",
                        "length": 12,
                        "structure": "............",
                        "mfe": {"approx": 0.0, "tol": 0.01},
                    },
                ),
                # A priori: likewise for a cytosine homopolymer.
                GoldenCase(
                    args={"sequence": "CCCCCCCCCC"},
                    expect={
                        "sequence": "CCCCCCCCCC",
                        "length": 10,
                        "structure": "..........",
                        "mfe": {"approx": 0.0, "tol": 0.01},
                    },
                ),
                # A priori: a single base has no partner; also pins the T→U normalisation.
                GoldenCase(
                    args={"sequence": "t"},
                    expect={
                        "sequence": "U",
                        "length": 1,
                        "structure": ".",
                        "mfe": {"approx": 0.0, "tol": 0.01},
                    },
                ),
                # Classic GC-clamped tetraloop hairpin: the structure is unambiguous, and
                # the energy is the value pinned by the reference ViennaRNA wrapper
                # (ViennaRNA 2.x) with a loose tolerance.
                GoldenCase(
                    args={"sequence": "GGGGAAAACCCC"},
                    expect={
                        "sequence": "GGGGAAAACCCC",
                        "length": 12,
                        "structure": "((((....))))",
                        "mfe": {"approx": -5.4, "tol": 0.5},
                    },
                ),
            ),
        ),
    ),
)
