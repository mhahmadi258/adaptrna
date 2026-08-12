"""ConfigRecommender: a data profile in, an executable training plan out.

Deterministic and table-driven (MASTER_PLAN §3.1): every number comes from
`adaptrna_agentic.knowledge`, and every rationale line is generated from the same
entries — so what the model tells the user and what the run actually does cannot drift
apart. The LLM narrates this plan; it never invents a hyperparameter.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

from adaptrna_agentic.knowledge import arm as arm_knowledge
from adaptrna_agentic.knowledge import task_knowledge_or_generic, template_for, universal
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
    profile: Dict[str, Any],
    task: Optional[str] = None,
    arm: str = "lora",
    quick: bool = False,
    seed: int = 42,
    run_name: Optional[str] = None,
    task_options: Optional[Dict[str, Any]] = None,
    registry=None,
) -> Dict[str, Any]:
    """
    Build a training plan from a data profile.

    Args:
        profile: output of `profile_dataset`.
        task: override the profile's matched task.
        arm: "lora" (default, the only arm that yields a servable tool) or "full_ft".
        quick: truncate the run with the engine's own `trainer.max_steps`.
        task_options: task-specific `data.*` choices, e.g. {"ss_type": "acceptor"}.
        registry: ToolHub registry supplying the backbone to train against (defaults to
            the configured hub). Training must use the backbone the hub serves, or the
            resulting adapter could not be served next to the existing tools.
    """
    task = task or profile.get("layout_match")
    if not task:
        raise ToolHubError(
            "This data does not match any shipped task's layout, so there is nothing to "
            "train yet. " + (profile.get("layout_reason") or "")
        )

    knowledge = task_knowledge_or_generic(task)
    arm_spec = arm_knowledge(arm)
    template = template_for(task)
    options = dict(task_options or {})

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

    # --- data ------------------------------------------------------------------
    data_root = _data_root(profile, task)
    overrides["data.root"] = str(data_root)
    if task == "splice_site":
        overrides["data.test_root"] = str(_splice_test_root(data_root))

    allowed = (template or {}).get("data_layout", {}).get("key_options", {})
    for key, value in options.items():
        dotted = key if key.startswith("data.") else f"data.{key}"
        if allowed and dotted in allowed and value not in allowed[dotted]:
            raise ToolHubError(
                f"'{value}' is not a valid {dotted} for {task}. "
                f"Options: {allowed[dotted]}"
            )
        overrides[dotted] = value

    # --- arm settings, straight from the knowledge base -------------------------
    for section in ("optim", "trainer"):
        for key, value in (arm_spec.get(section) or {}).items():
            overrides[f"{section}.{key}"] = value

    if arm == "lora":
        for key, value in arm_spec["lora"].items():
            overrides[f"lora.{key}"] = value

    for key, value in universal()["trainer"].items():
        overrides[f"trainer.{key}"] = value

    for key, value in (knowledge.get("defaults", {}).get("trainer") or {}).items():
        overrides.setdefault(f"trainer.{key}", value)

    rationale.extend(arm_spec["why"])
    rationale.extend(universal()["why"])
    for mode in arm_spec.get("failure_modes", []):
        rationale.append(
            f"Not {mode['setting']}: {mode['symptom']} Remedy: {mode['remedy']}"
        )

    if arm == "full_ft":
        warnings.append(arm_spec["artifact_note"])

    # --- quick run --------------------------------------------------------------
    if quick:
        overrides["trainer.max_steps"] = QUICK_RUN_MAX_STEPS
        overrides["data.num_workers"] = QUICK_RUN_NUM_WORKERS
        warnings.append(
            f"Quick run: capped at {QUICK_RUN_MAX_STEPS} steps. The result is a smoke "
            f"test, NOT comparable to the reference metrics for this task."
        )

    # --- task caveats -----------------------------------------------------------
    warnings.extend(knowledge.get("caveats", []))

    # --- assemble ---------------------------------------------------------------
    run_name = run_name or _run_name(task, arm, options)
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
        "primary_metric": knowledge["primary_metric"],
        "reference": knowledge["reference"],
        "estimated_wall_clock": _eta(knowledge, quick),
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


def _data_root(profile: Dict[str, Any], task: str) -> Path:
    path = Path(profile["path"]).expanduser()
    root = path if path.is_dir() else path.parent
    return root.resolve()


def _splice_test_root(data_root: Path) -> Path:
    """The Spliceator benchmark species live in a sibling `test_data/` directory."""
    sibling = data_root.parent / "test_data"
    return sibling if sibling.exists() else data_root


def _run_name(task: str, arm: str, options: Dict[str, Any]) -> str:
    parts = [task]
    for key in ("ss_type", "dataset", "val_split"):
        value = options.get(key) or options.get(f"data.{key}")
        if value:
            parts.append(str(value))
    parts.append(arm)
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    return "_".join(parts)


def _eta(knowledge: Dict[str, Any], quick: bool) -> str:
    reference = knowledge.get("wall_clock", {}).get("reference") or "unknown"
    if quick:
        return f"a few minutes (truncated); the full run would be {reference}"
    return reference
