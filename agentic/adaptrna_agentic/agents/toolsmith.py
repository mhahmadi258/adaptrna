"""ToolSmith: writes the files for a new task or external wrapper.

One structured model call per attempt. The loop that calls it, the staging, and every
check applied to what it returns are plain Python (`codegen/pipeline.py`) — the agent's
only job is the part that needs judgement: writing the code.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from adaptrna_agentic.codegen import prompts

TASK_FILENAMES = ("task.py", "datamodule.py", "config.yaml")


class GeneratedFile(BaseModel):
    filename: str = Field(description="One of: task.py, datamodule.py, config.yaml")
    content: str = Field(description="The complete file content — no placeholders")


class GeneratedTask(BaseModel):
    """The three files a new engine task consists of."""

    files: List[GeneratedFile] = Field(description="Exactly the three task files")
    notes: str = Field(default="", description="Anything the reviewer should know")


class GeneratedTool(BaseModel):
    """A single external-wrapper module."""

    content: str = Field(description="The complete wrapper module")
    notes: str = Field(default="", description="Anything the reviewer should know")


def _model(model=None):
    if model is not None:
        return model

    from adaptrna_agentic.models import build_chat_model

    return build_chat_model("toolsmith")


def generate_task(
    task_name: str,
    description: str,
    profile: Dict[str, object],
    feedback: Optional[str] = None,
    model=None,
) -> Dict[str, str]:
    """Generate `{filename: content}` for a new task."""
    structured = _model(model).with_structured_output(GeneratedTask)

    result = structured.invoke([
        {"role": "system", "content": prompts.task_system_prompt()},
        {"role": "user", "content": prompts.task_user_prompt(
            task_name, description, profile, feedback=feedback
        )},
    ])

    files = {f.filename.strip(): f.content for f in result.files}

    # Normalise stray paths ("adaptrna_custom/tasks/x/task.py" -> "task.py") so staging
    # owns the layout rather than trusting the model with it.
    return {name.rsplit("/", 1)[-1]: content for name, content in files.items()}


def generate_external_tool(
    package: str,
    description: str,
    feedback: Optional[str] = None,
    model=None,
) -> str:
    """Generate the module source for an external-tool wrapper."""
    structured = _model(model).with_structured_output(GeneratedTool)

    result = structured.invoke([
        {"role": "system", "content": prompts.task_system_prompt()},
        {"role": "user", "content": prompts.external_tool_prompt(
            package, description, feedback=feedback
        )},
    ])

    return result.content
