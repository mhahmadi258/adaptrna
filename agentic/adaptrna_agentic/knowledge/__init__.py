"""The knowledge base: validated hyperparameters, failure modes and target-type recipes.

Everything the ConfigRecommender proposes is read from here, so a recommendation and the
rationale shown to the user always come from the same source (MASTER_PLAN §6).

Phase 13: the platform ships no task definitions, so there is no per-task knowledge left
to carry — `arms:` and `universal:` transfer across tasks and stay; a reference metric
band does not transfer and is never invented (`generic:`); and `target_shapes.yaml`
replaces `task_templates.yaml`, keyed by target type rather than by a shipped task name.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

KNOWLEDGE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_knowledge() -> Dict[str, Any]:
    """Both knowledge files, parsed and merged into one dict (cached)."""
    with open(KNOWLEDGE_DIR / "hyperparameters.yaml") as handle:
        knowledge = yaml.safe_load(handle)

    with open(KNOWLEDGE_DIR / "target_shapes.yaml") as handle:
        knowledge.update(yaml.safe_load(handle))

    return knowledge


def arm(name: str) -> Dict[str, Any]:
    arms = load_knowledge()["arms"]
    if name not in arms:
        raise KeyError(f"Unknown training arm '{name}'. Known: {sorted(arms)}")
    return arms[name]


def universal() -> Dict[str, Any]:
    return load_knowledge()["universal"]


def generic_knowledge() -> Dict[str, Any]:
    """The `generic:` section: no reference band (there are no known tasks any more),
    plus the derivation rules `recommender.py` executes against an approved spec."""
    return load_knowledge()["generic"]


def derived(rule_name: str) -> Dict[str, Any]:
    """One derivation rule from `generic.derived` (batch_size / max_epochs / num_workers).

    `recommender.py` executes these against the approved spec; the `why:` line each rule
    carries is what generates the rationale shown at the gate, so the explanation and the
    config can never drift apart.
    """
    rules = generic_knowledge().get("derived") or {}
    if rule_name not in rules:
        raise KeyError(f"No derivation rule '{rule_name}'. Known: {sorted(rules)}")
    return rules[rule_name]


def target_shape(target_type: str) -> Dict[str, Any]:
    """The recipe for one of the three supported target types."""
    shapes = load_knowledge()["target_shapes"]
    if target_type not in shapes:
        raise KeyError(
            f"'{target_type}' is not a supported target type. Known: {sorted(shapes)}"
        )
    return shapes[target_type]
