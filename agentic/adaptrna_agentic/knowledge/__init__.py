"""The knowledge base: validated hyperparameters, failure modes and task templates.

Everything the ConfigRecommender proposes is read from here, so a recommendation and the
rationale shown to the user always come from the same source (MASTER_PLAN §6).
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

KNOWLEDGE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_knowledge() -> Dict[str, Any]:
    """Both knowledge files, parsed and merged into one dict (cached)."""
    with open(KNOWLEDGE_DIR / "hyperparameters.yaml") as handle:
        knowledge = yaml.safe_load(handle)

    with open(KNOWLEDGE_DIR / "task_templates.yaml") as handle:
        knowledge.update(yaml.safe_load(handle))

    return knowledge


def arm(name: str) -> Dict[str, Any]:
    arms = load_knowledge()["arms"]
    if name not in arms:
        raise KeyError(f"Unknown training arm '{name}'. Known: {sorted(arms)}")
    return arms[name]


def task_knowledge(name: str) -> Dict[str, Any]:
    tasks = load_knowledge()["tasks"]
    if name not in tasks:
        raise KeyError(f"No knowledge for task '{name}'. Known: {sorted(tasks)}")
    return tasks[name]


def generic_task_knowledge(name: str, primary_metric: Optional[str] = None) -> Dict[str, Any]:
    """Knowledge for a task the base has never seen — a generated one, say.

    The *arm* settings (LoRA lr 3e-4 + clip 1.0 + stride 3, bf16-mixed) were validated
    across tasks and transfer; a *reference metric band* is a property of a specific task
    and dataset and cannot be transferred. Saying so plainly is the point of this entry:
    the first run on a new task is a baseline, not something to be judged against
    somebody else's number.
    """
    return {
        "primary_metric": primary_metric,
        "higher_is_better": True,
        "reference": {
            "band": None,
            "tolerance": 0.0,
            "sources": [f"No validated reference run for '{name}' yet."],
            "tolerance_reason": "No reference band recorded for this task.",
        },
        "wall_clock": {"reference": "unknown (no reference run for this task)"},
        "defaults": {},
        "caveats": [
            f"'{name}' is not in the knowledge base. The arm settings below are the "
            f"project's validated values, which transfer across tasks, but there is no "
            f"reference band to judge the result against — treat this run as the baseline."
        ],
        "known": False,
    }


def task_knowledge_or_generic(name: str) -> Dict[str, Any]:
    """Task knowledge, falling back to a generic entry for unknown (generated) tasks."""
    try:
        known = dict(task_knowledge(name))
        known["known"] = True
        return known
    except KeyError:
        return generic_task_knowledge(name, primary_metric=primary_metric_of(name))


def primary_metric_of(name: str) -> Optional[str]:
    """A registered task's own PRIMARY_METRIC, if the task can be imported."""
    try:
        from adaptrna_agentic.codegen.discovery import load_all

        load_all()

        import rinalmo_hub.tasks  # noqa: F401  -- registers the shipped tasks
        from rinalmo_hub.registry import get_task, is_registered

        if is_registered(name):
            return getattr(get_task(name), "PRIMARY_METRIC", None)
    except Exception:  # noqa: BLE001 -- best effort; absence is reported, not raised
        return None

    return None


def templates() -> List[Dict[str, Any]]:
    return load_knowledge()["templates"]


def template_for(task: str) -> Optional[Dict[str, Any]]:
    return next((t for t in templates() if t["task"] == task), None)


def universal() -> Dict[str, Any]:
    return load_knowledge()["universal"]
