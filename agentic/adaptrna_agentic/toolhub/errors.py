"""ToolHub errors — a separate module so contract, registry and runtime can all share
them without import cycles."""


class ToolHubError(RuntimeError):
    """Anything the ToolHub refuses to do, with the reason and the fix in the message."""


class ConcurrentModificationError(ToolHubError):
    """A JSON store changed on disk since it was read.

    These stores are whole-file read-modify-write, so a blind save would silently discard
    whatever the other writer did. Detecting that is cheap; losing a registration or a job
    record is not. This is *detection*, not locking — see plans/PHASE_7_HARDENING.md §2.
    """
