"""D1/D2, made mechanical: the agentic layer must have no idea a shipped engine task or
the ViennaRNA wrapper ever existed — not in code, not in a docstring, not in help text,
not in a comment, not in a test.

Plain substring search, not word-boundary regex: an identifier that merely *contains* a
forbidden string (e.g. a variable named `..._mrl_...`) is exactly as much a leak as the
string standing alone, since it is still legible to anyone reading the code.

Two files are excluded, and only these two: this test's own source (it must name the
strings it searches for) and `forbidden_strings.py` (the one place those strings are
written down as the canonical list, imported here rather than re-embedded). Nothing under
`engine/` is in scope — the three shipped tasks staying on disk there is D1's explicit
exception; this sweep only ever walks `agentic/`.
"""

from pathlib import Path

from forbidden_strings import FORBIDDEN

AGENTIC_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "adaptrna_agentic.egg-info"}
EXCLUDE_FILES = {Path(__file__).resolve(), AGENTIC_ROOT / "tests" / "forbidden_strings.py"}
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".toml", ""}


def _files():
    for path in AGENTIC_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path in EXCLUDE_FILES:
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix not in SCANNED_SUFFIXES:
            continue
        yield path


def test_no_file_under_agentic_names_a_shipped_task_or_vienna():
    hits = []
    for path in _files():
        text = path.read_text(errors="ignore")
        for word in FORBIDDEN:
            if word in text:
                hits.append(f"{path.relative_to(AGENTIC_ROOT)}: {word!r}")

    assert not hits, "forbidden strings found:\n" + "\n".join(sorted(hits))


def test_the_sweep_actually_scans_something():
    """A sweep that silently walks zero files is worse than no sweep."""
    assert sum(1 for _ in _files()) > 100
