"""ConfigRecommender: an approved dataset spec in, an executable training plan out.

Deterministic and table-driven (MASTER_PLAN §3.1): every number comes from
`adaptrna_agentic.knowledge`, and every rationale line is generated from the same
entries — so what the model tells the user and what the run actually does cannot drift
apart. The LLM narrates this plan; it never invents a hyperparameter.

Phase 13: there are no known tasks any more, so nothing here derives from a task name.
The `arm` settings (LoRA lr 3e-4, clip 1.0, stride 3) transfer across tasks and stay
table-driven; the values that cannot transfer — batch size, epoch count, worker count —
are DERIVED from the approved `DatasetSpec` a task landed with, by rules in
`knowledge.derived()`, so the recommender still invents nothing.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

from adaptrna_agentic.knowledge import arm as arm_knowledge
from adaptrna_agentic.knowledge import derived, generic_knowledge, universal
from adaptrna_agentic.settings import REPO_ROOT
from adaptrna_agentic.toolhub.errors import ToolHubError

#: Steps for a `quick_run` — enough to show the loss moving, short enough to demo.
QUICK_RUN_MAX_STEPS = 200
QUICK_RUN_NUM_WORKERS = 8

TRAIN_ENTRYPOINT = "adaptrna_agentic.jobs.train_entrypoint"

#: Stamped on every plan this module produces. `start_training` refuses plans without it,
#: so a hand-assembled plan cannot smuggle invented hyperparameters past the knowledge
#: base — the rule is enforced mechanically rather than by asking the model nicely.
PLAN_SOURCE = "recommend_training_config"


def recommend(
    task: str,
    spec: Optional[Dict[str, Any]] = None,
    arm: str = "lora",
    quick: bool = False,
    seed: int = 42,
    run_name: Optional[str] = None,
    registry=None,
) -> Dict[str, Any]:
    """
    Build a training plan for a landed task.

    Args:
        task: the landed task's name (`adaptrna_custom/tasks/<task>/`).
        spec: the DatasetSpec to derive batch size/epochs from. Defaults to the spec the
            task itself landed with (`adaptrna_custom/tasks/<task>/spec.json`) — pass one
            explicitly only to point an *existing* task at a new file of the same shape
            (reuse, plan §9), which overrides `data.root` without regenerating any code.
        arm: "lora" (default, the only arm that yields a servable tool) or "full_ft".
        quick: truncate the run with the engine's own `trainer.max_steps`.
        registry: ToolHub registry supplying the backbone to train against (defaults to
            the configured hub). Training must use the backbone the hub serves, or the
            resulting adapter could not be served next to the existing tools.
    """
    if not task:
        raise ToolHubError(
            "There is no task to train yet. Approve a dataset spec with "
            "confirm_data_profile and build one with create_task_tool."
        )

    spec = spec or _landed_spec(task)
    if spec is None:
        raise ToolHubError(
            f"No dataset spec found for '{task}'. This build derives batch size and "
            f"epoch count from the spec.json a task lands with — land it through "
            f"confirm_data_profile, create_task_tool and land_generated_code first, or "
            f"pass spec explicitly."
        )

    arm_spec = arm_knowledge(arm)

    overrides: Dict[str, Any] = {}
    rationale: List[str] = []
    warnings: List[str] = []

    # --- backbone: whatever the ToolHub serves --------------------------------
    # The engine's own default is `weights/giga-v1.pt` relative to the working
    # directory; the hub knows where the checkpoint actually lives, and training
    # against a different backbone than the hub serves would produce an adapter it
    # could not host.
    backbone = _backbone_config(registry)
    overrides["lm_config"] = backbone.lm_config
    if backbone.weights:
        weights_path = _resolve_weights(backbone.weights)
        overrides["pretrained_weights"] = str(weights_path)
        rationale.append(
            f"Backbone: the '{backbone.lm_config}' checkpoint this ToolHub serves "
            f"({weights_path}) — an adapter trained on any other backbone could not be "
            f"served alongside the existing tools."
        )
    else:
        warnings.append(
            "The ToolHub has no backbone checkpoint configured, so this run would start "
            "from a randomly initialised backbone. Set one with "
            "`toolhub config --weights /path/to/giga-v1.pt` before training for real."
        )
        overrides["pretrained_weights"] = "null"

    # --- data --------------------------------------------------------------------
    # A single table now, not a directory: the config points straight at the file the
    # spec names.
    overrides["data.root"] = spec["path"]

    # --- arm settings, straight from the knowledge base -------------------------
    for section in ("optim", "trainer"):
        for key, value in (arm_spec.get(section) or {}).items():
            overrides[f"{section}.{key}"] = value

    if arm == "lora":
        for key, value in arm_spec["lora"].items():
            overrides[f"lora.{key}"] = value

    for key, value in universal()["trainer"].items():
        overrides[f"trainer.{key}"] = value

    rationale.extend(arm_spec["why"])
    rationale.extend(universal()["why"])
    for mode in arm_spec.get("failure_modes", []):
        rationale.append(
            f"Not {mode['setting']}: {mode['symptom']} Remedy: {mode['remedy']}"
        )

    if arm == "full_ft":
        warnings.append(arm_spec["artifact_note"])

    # --- derived values: batch size, epochs, workers -----------------------------
    # Values that cannot transfer across datasets, derived from the approved spec by
    # rules in the knowledge base rather than invented here.
    median_length = (spec.get("length") or {}).get("median") or 0
    batch_rule = derived("batch_size")
    batch_size = _piecewise_on_median_length(
        median_length, batch_rule["table"], batch_rule["fallback"]
    )
    overrides["data.batch_size"] = batch_size
    rationale.append(f"data.batch_size {batch_size}: {batch_rule['why']}")

    rows_train = (spec.get("split") or {}).get("row_counts", {}).get("train", 0)
    epoch_rule = derived("max_epochs")
    steps_per_epoch = max(1, math.ceil(rows_train / max(batch_size, 1)))
    max_epochs = _step_budget(
        steps_per_epoch, epoch_rule["target_steps"], epoch_rule["clamp"]
    )
    overrides["trainer.max_epochs"] = max_epochs
    low, high = epoch_rule["target_steps"]
    total_steps = max_epochs * steps_per_epoch
    rationale.append(
        f"trainer.max_epochs {max_epochs}: {rows_train:,} training rows at batch "
        f"{batch_size} is {steps_per_epoch:,} steps per epoch; {max_epochs} epochs ≈ "
        f"{total_steps:,} optimiser steps, inside the {low:,}-{high:,} budget."
    )

    workers_rule = derived("num_workers")
    overrides["data.num_workers"] = workers_rule["value"]
    rationale.append(f"data.num_workers {workers_rule['value']}: {workers_rule['why']}")

    # --- quick run --------------------------------------------------------------
    if quick:
        overrides["trainer.max_steps"] = QUICK_RUN_MAX_STEPS
        overrides["data.num_workers"] = QUICK_RUN_NUM_WORKERS
        warnings.append(
            f"Quick run: capped at {QUICK_RUN_MAX_STEPS} steps. The result is a smoke "
            f"test, NOT comparable to the reference metrics for this task."
        )

    # --- task caveats -----------------------------------------------------------
    generic = generic_knowledge()
    warnings.extend(generic.get("caveats", []))

    # --- assemble ---------------------------------------------------------------
    run_name = run_name or _run_name(task, arm)
    output_dir = f"outputs/{run_name}"
    config_path = _config_path(task)

    plan = {
        "source": PLAN_SOURCE,
        "task": task,
        "arm": arm,
        "config_path": config_path,
        "overrides": overrides,
        "seed": seed,
        "output_dir": output_dir,
        "quick_run": quick,
        "primary_metric": spec["head"]["primary_metric"],
        "reference": generic["reference"],
        "estimated_wall_clock": _eta(generic, quick),
        "rationale": rationale,
        "warnings": warnings,
    }
    plan["command"] = build_command(plan)

    return plan


def build_command(plan: Dict[str, Any]) -> List[str]:
    """Materialise the exact command line the JobRunner will execute.

    Shown verbatim in the approval gate — the same "Would run: …" discipline as the
    Phase 3 package-install gate.
    """
    command = [
        sys.executable, "-m", TRAIN_ENTRYPOINT,
        "--task", plan["task"],
        "--config", plan["config_path"],
        "--output_dir", plan["output_dir"],
        "--seed", str(plan["seed"]),
    ]

    if plan["arm"] == "lora":
        command.append("--use_lora")

    for key, value in plan["overrides"].items():
        command += ["--set", f"{key}={_render(value)}"]

    return command


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _piecewise_on_median_length(median: float, table: List[List[float]], fallback: int) -> int:
    """`generic.derived.batch_size`: the largest batch a sequence this long trains
    comfortably at, per the measured table (shorter sequences -> more headroom -> a
    larger batch)."""
    for threshold, batch in table:
        if median <= threshold:
            return int(batch)
    return int(fallback)


def _step_budget(steps_per_epoch: int, target_steps: List[int], clamp: List[int]) -> int:
    """`generic.derived.max_epochs`: the fewest epochs that reach the low end of the
    optimiser-step budget, clamped to a sane range either way."""
    low, _high = target_steps
    lo_clamp, hi_clamp = clamp
    epochs = math.ceil(low / max(steps_per_epoch, 1))
    return max(lo_clamp, min(hi_clamp, epochs))


def _landed_spec(task: str) -> Optional[Dict[str, Any]]:
    from adaptrna_agentic.codegen.discovery import landed_spec

    return landed_spec(task)


def _config_path(task: str) -> str:
    """A generated task's config lives beside its code, not under engine/configs."""
    from adaptrna_agentic.codegen.discovery import CUSTOM_PACKAGE, TASKS_DIRNAME

    custom = REPO_ROOT / CUSTOM_PACKAGE / TASKS_DIRNAME / task / "config.yaml"
    if custom.exists():
        return f"{CUSTOM_PACKAGE}/{TASKS_DIRNAME}/{task}/config.yaml"

    return f"engine/configs/tasks/{task}.yaml"


def _backbone_config(registry=None):
    if registry is None:
        from adaptrna_agentic.toolhub.registry import Registry

        registry = Registry()

    return registry.manifest.backbone


def _resolve_weights(weights: str) -> Path:
    from adaptrna_agentic.toolhub.manifest import resolve_path

    path = resolve_path(weights)
    if not path.exists():
        raise ToolHubError(
            f"The ToolHub's backbone checkpoint '{weights}' does not exist (resolved to "
            f"'{path}'). Point the hub at it with "
            f"`toolhub config --weights /path/to/giga-v1.pt` before training."
        )

    return path


def _run_name(task: str, arm: str) -> str:
    return "_".join([task, arm, datetime.now().strftime("%Y%m%d_%H%M%S")])


def _eta(generic: Dict[str, Any], quick: bool) -> str:
    reference = generic.get("wall_clock", {}).get("reference") or "unknown"
    if quick:
        return f"a few minutes (truncated); the full run would be {reference}"
    return reference
