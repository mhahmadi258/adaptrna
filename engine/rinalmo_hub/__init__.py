"""RiNALMo-Hub: one frozen RiNALMo backbone, swappable LoRA adapters and task heads."""

from rinalmo_hub.registry import available_tasks, get_task, register_task

__all__ = ["available_tasks", "get_task", "register_task"]
