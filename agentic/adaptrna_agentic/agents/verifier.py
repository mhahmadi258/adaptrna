"""Verifier: an independent review of what ToolSmith wrote.

Deliberately a *fresh context* (MASTER_PLAN §5): an auditor that inherits the writer's
context inherits the writer's blind spots.

The division of labour matters. The harness has already proved mechanically whether the
code imports, trains a step, round-trips through an adapter file and serves — including
the tensor half of the silent-state trap. This agent judges what a test cannot: whether
the code does what the user asked, and whether *non-tensor* task state or a CLS/EOS
mistake would quietly produce plausible-looking wrong numbers.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from adaptrna_agentic.codegen import prompts


class Review(BaseModel):
    approved: bool = Field(
        description="True only if you would be comfortable with these numbers in a paper"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Specific, actionable problems. Empty when approving.",
    )
    owns_unsaved_state: Optional[str] = Field(
        default=None,
        description="Silent-failure question 1: state predictions depend on that is not "
                    "carried in the adapter file, or null if none.",
    )
    boundary_tokens: Optional[str] = Field(
        default=None,
        description="Silent-failure question 2: how CLS/EOS/padding are handled in "
                    "extract_features, and whether that is right for this head.",
    )
    notes: str = Field(default="")


def _model(model=None):
    if model is not None:
        return model

    from adaptrna_agentic.models import build_chat_model

    return build_chat_model("verifier")


def review_task(
    description: str,
    spec: Dict[str, object],
    files: Dict[str, str],
    harness_summary: str,
    model=None,
    rendered: bool = False,
) -> Review:
    """`rendered=True` on the template path (plan §7.4): there is no author to judge, so
    the prompt asks the narrower question — does this code do what the spec says?"""
    structured = _model(model).with_structured_output(Review)

    return structured.invoke([
        {"role": "system", "content": prompts.verifier_system_prompt()},
        {"role": "user", "content": prompts.verifier_user_prompt(
            description, spec, files, harness_summary, rendered=rendered
        )},
    ])


def format_feedback(harness_summary: str, review: Optional[Review]) -> str:
    """What ToolSmith gets to work from on the next attempt."""
    blocks = [f"Automated verification:\n{harness_summary}"]

    if review is not None and review.findings:
        blocks.append("Reviewer findings:\n" + "\n".join(f"- {f}" for f in review.findings))
    if review is not None and review.owns_unsaved_state:
        blocks.append(f"Unsaved task state: {review.owns_unsaved_state}")
    if review is not None and review.boundary_tokens:
        blocks.append(f"CLS/EOS/padding: {review.boundary_tokens}")

    return "\n\n".join(blocks)
