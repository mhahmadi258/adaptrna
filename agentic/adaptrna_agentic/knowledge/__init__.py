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


def templates() -> List[Dict[str, Any]]:
    return load_knowledge()["templates"]


def template_for(task: str) -> Optional[Dict[str, Any]]:
    return next((t for t in templates() if t["task"] == task), None)


def universal() -> Dict[str, Any]:
    return load_knowledge()["universal"]
