"""Request bodies. Responses are the same plain dicts the CLI prints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequences: List[str] = Field(description="RNA/DNA sequences as plain ACGU/T strings")
    batch_size: Optional[int] = None


class CallRequest(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict,
                                 description="Keyword arguments for the wrapped function")


class MessageRequest(BaseModel):
    text: str = Field(description="What the user said")


class ResumeRequest(BaseModel):
    approved: bool = Field(description="Whether the user approved the pending action")
    note: Optional[str] = Field(default=None, description="Why, if they declined")
    edits: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dotted-path -> value corrections to the gate's own arguments, "
                     "e.g. {'spec.task_name': 'donor_sites'}. Absent or {} means "
                     "'as proposed'.",
    )


class SessionRequest(BaseModel):
    """Create or rename: both carry exactly one thing, the name the session should have."""

    id: str = Field(description="The session name, which is also its storage key")
