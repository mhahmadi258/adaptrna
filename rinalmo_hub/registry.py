"""Task registry.

A task is a `BaseDownstreamModule` subclass decorated with `@register_task("name")`. The
decorator is what makes a task visible to the CLI and to `RiNALMoHub`, so a new task never
requires touching any core file -- it only has to be imported once, which
`rinalmo_hub/tasks/__init__.py` does.
"""

from typing import Dict, List, Type

_TASKS: Dict[str, type] = {}


def register_task(name: str):
    """Class decorator registering a downstream task module under `name`."""

    def decorator(cls: type) -> type:
        if name in _TASKS and _TASKS[name] is not cls:
            raise ValueError(
                f"Task '{name}' is already registered to {_TASKS[name].__module__}."
                f"{_TASKS[name].__qualname__}"
            )

        cls.TASK_NAME = name
        _TASKS[name] = cls
        return cls

    return decorator


def get_task(name: str) -> Type:
    """Look up a registered task class by name."""
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available_tasks()}")

    return _TASKS[name]


def available_tasks() -> List[str]:
    """Names of every registered task, sorted."""
    return sorted(_TASKS)


def is_registered(name: str) -> bool:
    return name in _TASKS
