"""Context assembly for ToolSmith and Verifier.

Everything here already exists somewhere in the project — the engine's subclass contract,
its worked example, the external-tool contract. This module's only job is to put the
right pieces in front of the right agent.

Phase 13: the platform ships no task definitions, so there is no "closest known task
shape" to show the generator any more (D6) — Stage 4 of that phase replaces it with a
spec-driven recipe section and re-points the worked example at the deterministic
template's own output. Until then this module still targets the pre-Phase-13 flow
(profile + name/description in, not an approved DatasetSpec).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

from adaptrna_agentic.settings import REPO_ROOT

EXAMPLE_DIR = REPO_ROOT / "engine" / "examples" / "ncrna_classification"
CONTRACT_FILE = REPO_ROOT / "agentic" / "adaptrna_agentic" / "toolhub" / "external" / "contract.py"

#: The engine's subclass contract, as the README states it. Kept here (rather than parsed
#: out of the engine) so the generator sees a stable, complete statement of the hooks.
SUBCLASS_CONTRACT = """\
A task is a `BaseDownstreamModule` subclass decorated with `@register_task("name")`.

Required hooks:
  build_head(embed_dim, **head_config) -> nn.Module
        `embed_dim` is the backbone width, supplied by the base class; every other
        argument comes from the config's `head:` block. Raise TypeError on unexpected
        keys so config drift is loud.
  extract_features(representation, tokens)
        Turns the backbone's per-token representation into what the head consumes.
        Return a tensor, or a tuple that is splatted into the head as positional args.
  compute_loss(outputs, batch) -> scalar tensor
  update_metrics(outputs, batch, stage) -> None
  compute_metrics(stage) -> dict[str, scalar]
  build_datamodule(cfg) -> LightningDataModule        (@staticmethod)

Optional hooks:
  build_metrics(stage) -> metric object exposed as self.metrics[stage]
  batch_tokens(batch) -> Tensor                       (default: batch[0])
  postprocess_predictions(outputs, tokens, sequences) (task-native prediction type)
  adapter_extra_payload() / load_adapter_extra(extra) (NON-tensor state)
  on_fit_start_hook()                                 (setup needing the datamodule)

Class attributes:
  TASK_NAME               set by the decorator
  ADAPTER_EXTRA_PREFIXES  extra *tensor* state_dict prefixes to ship in the adapter file
  PRIMARY_METRIC          the metric this task is judged on, e.g. "test/f1_score"
  DEFAULT_PREDICT_BATCH_SIZE  lower it for heads that are quadratic in sequence length

Naming is fixed framework-wide: the backbone is always `self.backbone`, the head is
always `self.head`. Never edit anything under engine/ — a task needs no framework change.
"""

#: The two failure modes that produce plausible-looking wrong numbers rather than errors.
SILENT_FAILURE_RULES = """\
Two questions decide whether this task is correct, and both fail SILENTLY — the adapter
loads without error and the numbers look plausible:

1. Does the task own state that predictions depend on but that is not a head weight?
   - a tensor or buffer  -> its prefix MUST be in ADAPTER_EXTRA_PREFIXES
   - a plain Python value (a tuned threshold, a class mapping)
                         -> implement adapter_extra_payload() and load_adapter_extra()
   Omit either and the state silently reverts to its default after a save/load.

2. Does the head need CLS, EOS or padded positions excluded? `extract_features` is the
   only place that happens: representation[:, 0] takes the CLS token,
   representation[..., 1:-1, :] drops CLS and EOS, and padded positions must be masked
   for anything that pools over the sequence.
"""

HARD_REQUIREMENTS = """\
Hard requirements for the files you produce:

* `task.py` — one @register_task("<task_name>") subclass of BaseDownstreamModule.
* `datamodule.py` — a lightning.pytorch.LightningDataModule whose loaders yield batches
  your compute_loss/update_metrics accept. Read the user's real columns and file names.
* `config.yaml` — `task:` MUST equal the registered name. Include head:, data:, lora:,
  optim:, trainer: blocks. Everything unset is inherited from engine/configs/base.yaml.
* Import the datamodule inside build_datamodule with the absolute path
  `from adaptrna_custom.tasks.<task_name>.datamodule import <Class>` — the package is
  imported under that name both while it is verified and after it lands.
* Use `lightning.pytorch`, never the standalone `pytorch_lightning` package.
* Return JSON-serialisable, task-native values from postprocess_predictions.
"""


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def worked_example() -> str:
    """The engine's own end-to-end example of adding a task in three files."""
    blocks = []
    for filename in ("task.py", "datamodule.py", "config.yaml"):
        content = _read(EXAMPLE_DIR / filename)
        if content:
            blocks.append(f"--- examples/ncrna_classification/{filename}\n{content}")

    return "\n\n".join(blocks)


def task_system_prompt() -> str:
    return (
        "You write new tasks for the AdaptRNA fine-tuning engine. A task is exactly three "
        "files and requires no change to the engine itself.\n\n"
        "Write complete, runnable files — no placeholders, no TODOs, no invented helper "
        "modules. Follow the worked example's structure and the engine's contract "
        "exactly, and read the user's data as it actually is on disk."
    )


def task_user_prompt(
    task_name: str,
    description: str,
    profile: Dict[str, Any],
    feedback: Optional[str] = None,
) -> str:
    sections = [
        f"# Task to build\n\nName: `{task_name}`\nWhat the user wants: {description}",
        f"# The data\n\n```json\n{json.dumps(profile, indent=2, default=str)}\n```",
    ]

    sections += [
        f"# The engine's subclass contract\n\n```\n{SUBCLASS_CONTRACT}\n```",
        f"# Silent-failure rules\n\n{SILENT_FAILURE_RULES}",
        f"# Requirements\n\n{HARD_REQUIREMENTS}",
        f"# A complete worked example\n\n```python\n{worked_example()}\n```",
    ]

    if feedback:
        sections.append(
            "# Your previous attempt failed verification\n\n"
            f"{feedback}\n\nFix these specific problems. Keep everything that worked."
        )

    return "\n\n".join(sections)


def verifier_system_prompt() -> str:
    return (
        "You review generated code for the AdaptRNA engine, in a fresh context and "
        "independently of whoever wrote it.\n\n"
        "An automated harness has already proved whether the code imports, trains one "
        "step, round-trips through an adapter file and serves. Your job is what the "
        "harness cannot check: whether the code does what the user actually asked, and "
        "whether it falls into either silent-failure trap. Judge the code in front of "
        "you — do not assume good intent, and do not re-litigate what the harness "
        "already proved."
    )


def verifier_user_prompt(
    description: str,
    profile: Dict[str, Any],
    files: Dict[str, str],
    harness_summary: str,
) -> str:
    listing = "\n\n".join(
        f"--- {name}\n{content}" for name, content in sorted(files.items())
    )

    return "\n\n".join([
        f"# What the user asked for\n\n{description}",
        f"# The data\n\n```json\n{json.dumps(profile, indent=2, default=str)}\n```",
        f"# Automated verification (already run)\n\n```\n{harness_summary}\n```",
        f"# The generated code\n\n```python\n{listing}\n```",
        "# Your checklist\n\n" + SILENT_FAILURE_RULES + "\n"
        "Also check: does the datamodule read the columns this data actually has? Does "
        "the loss match the target type? Do the metrics suit the task? Is the config's "
        "`task:` the registered name?\n\n"
        "Approve only if you would be comfortable with these numbers in a paper.",
    ])


def external_tool_prompt(package: str, description: str, feedback: Optional[str] = None) -> str:
    """Flow E: a wrapper module following the Phase 3 contract."""
    sections = [
        f"# Wrapper to build\n\nPython package: `{package}`\nWhat it should expose: {description}",
        f"# The contract you must satisfy\n\n```python\n{_read(CONTRACT_FILE)}\n```",
        "# The hand-written reference implementation\n\n```python\n"
        + _read(REPO_ROOT / "agentic" / "adaptrna_agentic" / "toolhub" / "external" / "vienna.py")
        + "\n```",
        "Produce ONE module: a module-level `SPEC: ExternalToolSpec` plus one plain "
        "function per FunctionSpec. Validate inputs BEFORE importing the wrapped package. "
        "Golden cases must be values you are confident about a priori — never invented "
        "numbers.",
    ]
    if feedback:
        sections.append(f"# Your previous attempt failed verification\n\n{feedback}")

    return "\n\n".join(sections)
