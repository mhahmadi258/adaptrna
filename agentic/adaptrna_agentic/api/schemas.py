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
