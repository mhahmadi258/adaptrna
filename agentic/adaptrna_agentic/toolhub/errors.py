"""ToolHub errors — a separate module so contract, registry and runtime can all share
them without import cycles."""


class ToolHubError(RuntimeError):
    """Anything the ToolHub refuses to do, with the reason and the fix in the message."""
