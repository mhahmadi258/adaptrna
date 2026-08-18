"""A contract-compliant external wrapper that demonstrates validate-before-import: input
is checked and normalised *before* the wrapped package is imported, so a bad argument
raises `ValueError` at the call boundary rather than failing deep inside the package (or,
if the package isn't installed, surfacing as an `ImportError` that has nothing to do with
what the caller got wrong). The wrapped package (`zlib`) is stdlib, so these tests need no
optional install — the ordering itself is what's under test, proven directly in
`test_external_fixture_wrapper.py` by poisoning the import and checking validation still
runs first.
"""

from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec,
    FunctionSpec,
    GoldenCase,
    PackageSpec,
)


def _clean(text: str, label: str = "text") -> str:
    """Validate + normalise. Runs before any `zlib` import."""
    value = (text or "").strip()
    if not value:
        raise ValueError(f"Empty {label}: provide at least one character.")
    if not value.isprintable():
        raise ValueError(f"Non-printable characters in {label}.")
    return value


def checksum(text: str) -> dict:
    """CRC32 checksum of one string."""
    value = _clean(text)

    import zlib

    return {"crc32": zlib.crc32(value.encode("utf-8"))}


def checksum_pair(text_a: str, text_b: str) -> dict:
    """CRC32 checksum of two strings concatenated."""
    a = _clean(text_a, "text_a")
    b = _clean(text_b, "text_b")

    import zlib

    return {"crc32": zlib.crc32((a + b).encode("utf-8"))}


SPEC = ExternalToolSpec(
    name="checksum",
    description="CRC32 checksums over text, via the stdlib zlib module.",
    package=PackageSpec(pip="stdlib", import_name="zlib"),
    functions=(
        FunctionSpec(
            name="checksum",
            description="CRC32 checksum of one string.",
            golden=(
                GoldenCase(args={"text": "hello"}, expect={"crc32": 907060870}),
            ),
        ),
        FunctionSpec(
            name="checksum_pair",
            description="CRC32 checksum of two strings concatenated.",
            golden=(
                GoldenCase(args={"text_a": "hel", "text_b": "lo"}, expect={"crc32": 907060870}),
            ),
        ),
    ),
)
