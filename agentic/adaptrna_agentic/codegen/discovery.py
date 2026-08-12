"""Discovery of project-local (generated) code in `<repo>/adaptrna_custom/`.

The generated code lives at the repo root — reviewable and git-tracked — while this
loader lives in the installed package, so it is the one place that knows how to put the
repo root on `sys.path` and import what is there.

Three callers rely on it, and all three matter:

* the training entrypoint, so a generated task is trainable (`@register_task` must have
  fired before the engine CLI resolves the task name);
* the ToolHub runtime, so an adapter trained on a generated task can be **served** — the
  engine's `get_task()` raises otherwise, and a task that trains but cannot serve is a
  silent dead end;
* the verification harness, inside its sandbox.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import importlib
import sys

from adaptrna_agentic.settings import REPO_ROOT

CUSTOM_PACKAGE = "adaptrna_custom"
CUSTOM_ROOT = REPO_ROOT / CUSTOM_PACKAGE

TASKS_DIRNAME = "tasks"
TOOLS_DIRNAME = "tools"


def ensure_importable() -> None:
    """Put the repo root on `sys.path` so `adaptrna_custom.*` imports resolve."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def custom_task_names() -> List[str]:
    """Names of every generated task package that has a `task.py`."""
    tasks_dir = CUSTOM_ROOT / TASKS_DIRNAME
    if not tasks_dir.is_dir():
        return []

    return sorted(
        entry.name
        for entry in tasks_dir.iterdir()
        if entry.is_dir() and (entry / "task.py").exists()
    )


def task_module_path(name: str) -> str:
    return f"{CUSTOM_PACKAGE}.{TASKS_DIRNAME}.{name}.task"


def tool_module_path(name: str) -> str:
    return f"{CUSTOM_PACKAGE}.{TOOLS_DIRNAME}.{name}"


def load_all(only: Optional[List[str]] = None) -> List[Tuple[str, Exception]]:
    """
    Import every generated task module so `@register_task` fires.

    Returns the (name, exception) pairs that failed rather than raising: one broken
    generated task must not make every other task unavailable.
    """
    ensure_importable()

    failures: List[Tuple[str, Exception]] = []
    for name in custom_task_names():
        if only is not None and name not in only:
            continue
        try:
            importlib.import_module(task_module_path(name))
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            failures.append((name, exc))

    return failures


def describe_failures(failures: List[Tuple[str, Exception]]) -> str:
    return "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in failures)
