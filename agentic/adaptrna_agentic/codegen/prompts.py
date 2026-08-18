"""Context assembly for ToolSmith and Verifier.

Everything here already exists somewhere in the project — the engine's subclass contract,
a worked example, the external-tool contract. This module's only job is to put the right
pieces in front of the right agent.

Phase 13 (D6): the platform ships no task definitions, so there is no "closest known task
shape" to show the generator any more, and no shipped example to read. What replaces both:
the approved `DatasetSpec` itself (`spec_section`) — the contract with the user, already
agreed at gate 1 — the one `target_shapes.yaml` recipe matching its target type
(`recipe_section`), and a worked example that is the deterministic template's own output
against a synthetic fixture spec, carrying no RNA task identity at all
(`worked_example`). This is the fallback path only: `codegen/templates/` covers the
declared case with no model call (plan §7.2).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

from adaptrna_agentic.settings import REPO_ROOT

CONTRACT_FILE = REPO_ROOT / "agentic" / "adaptrna_agentic" / "toolhub" / "external" / "contract.py"

#: The tiny synthetic sequence,label table `worked_example()` renders the template
#: against — shared with the Stage 0 harness controls and the Stage 1 template tests, so
#: nothing extra needs to be written or kept in sync (plan §7.6).
_WORKED_EXAMPLE_CSV = REPO_ROOT / "agentic" / "tests" / "fixtures" / "data" / "binary.csv"

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
* The datamodule reads EXACTLY `spec["sequence_column"]` and `spec["label_column"]` from
  `spec["path"]`, and ignores every other column.
* It implements the approved split (`spec["split"]`) and nothing else — no re-shuffling,
  no second seed, no different fractions.
* `PRIMARY_METRIC` MUST equal `spec["head"]["primary_metric"]` exactly, and it MUST be a
  key `compute_metrics` actually returns.
* Rows whose sequence contains characters outside ACGTUN are handled per
  `spec["on_invalid"]` — "fail" (raise loudly) by default, "drop" only if the spec says so.
"""


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def spec_section(spec: Dict[str, Any]) -> str:
    """The approved spec, described as what it is: an agreement already made."""
    return (
        "# The approved dataset spec\n\n"
        "This is the contract with the user, already agreed at gate 1 (confirm_data_"
        "profile) — read exactly the columns and implement exactly the split it names, "
        "nothing else.\n\n"
        f"```json\n{json.dumps(spec, indent=2, default=str)}\n```"
    )


def recipe_section(spec: Dict[str, Any]) -> str:
    """The one target_shapes.yaml entry matching the approved target type — nothing else,
    and no task identity."""
    from adaptrna_agentic.knowledge import target_shape

    target_type = spec.get("target_type")
    shape = target_shape(target_type)

    return (
        f"# Recipe for a '{target_type}' target\n\n"
        f"head: {shape['head']}\n"
        f"extract_features: {shape['extract_features']}\n"
        f"loss: {shape['loss']}\n"
        f"metrics: {shape['metrics']}\n"
        f"primary_metric: {shape['primary_metric']}\n"
        f"predict_output: {shape['predict_output']}\n"
        f"pad_sensitive: {shape['pad_sensitive']}\n\n"
        f"Silent-failure trap for this shape: {shape['adapter_state']}"
    )


def split_instructions(spec: Dict[str, Any]) -> str:
    """Exact split instructions generated from the approved spec's split policy."""
    split = spec.get("split") or {}

    if split.get("mode") == "column":
        return (
            "# Split policy\n\n"
            f"Column mode: read column '{split.get('column')}' and assign each row to "
            f"train/val/test using exactly this mapping (drop rows matching none of "
            f"it): {json.dumps(split.get('mapping'))}"
        )

    fractions = split.get("fractions") or {}
    stratified = " Stratify by label." if split.get("stratify") else ""
    return (
        "# Split policy\n\n"
        f"Random split with fractions {json.dumps(fractions)}, seed {split.get('seed')}."
        f"{stratified} No re-shuffling, no second seed, no different fractions."
    )


def worked_example() -> str:
    """The template's own rendered output against a synthetic fixture spec — a neutral,
    guaranteed-correct example carrying no RNA task identity, replacing the shipped
    example read (D6). Nothing extra to write or maintain: it is the same reviewed
    template that produces production code, rendered for the target type at hand."""
    from adaptrna_agentic.codegen.templates import render as templates

    fixture_spec = {
        "target_type": "binary",
        "task_name": "worked_example",
        "tool_description": "a worked example for the code generator",
        "sequence_column": "sequence",
        "label_column": "label",
        "path": str(_WORKED_EXAMPLE_CSV),
        "format": {"separator": ",", "compression": None},
        "classes": ["0", "1"],
        "positive_class": "1",
        "head": {"primary_metric": "test/f1_score"},
        "split": {
            "mode": "random", "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42, "stratify": True,
        },
    }
    files = templates.render(fixture_spec)

    return "\n\n".join(f"--- {name}\n{content}" for name, content in sorted(files.items()))


def task_system_prompt() -> str:
    return (
        "You write the data loader and head for a task the AdaptRNA fine-tuning engine "
        "does not yet have. A task is exactly three files (task.py, datamodule.py, "
        "config.yaml) and requires no change to the engine itself. You are only called "
        "for specs the deterministic template cannot express — the worked example below "
        "is that same template, rendered for a shape close to this one.\n\n"
        "Write complete, runnable files — no placeholders, no TODOs, no invented helper "
        "modules. Follow the worked example's structure and the engine's contract "
        "exactly, and read the user's data exactly as the approved spec describes it."
    )


def task_user_prompt(spec: Dict[str, Any], feedback: Optional[str] = None) -> str:
    sections = [
        f"# Task to build\n\nName: `{spec.get('task_name')}`\n"
        f"What the user wants: {spec.get('tool_description')}",
        spec_section(spec),
        recipe_section(spec),
        split_instructions(spec),
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
    spec: Dict[str, Any],
    files: Dict[str, str],
    harness_summary: str,
    rendered: bool = False,
) -> str:
    listing = "\n\n".join(
        f"--- {name}\n{content}" for name, content in sorted(files.items())
        if name != "spec.json"
    )

    if rendered:
        framing = (
            "This code was rendered deterministically from the approved spec below, by a "
            "reviewed template — there is no author whose judgment you are auditing. Ask "
            "only the narrower question: does this code do what this spec says, for this "
            "data? A rejection here means the template does not fit this spec, not that "
            "someone made a mistake."
        )
    else:
        framing = (
            "Judge the code in front of you — do not assume good intent, and do not "
            "re-litigate what the harness already proved."
        )

    return "\n\n".join([
        f"# What the user asked for\n\n{description}",
        f"# The data\n\n```json\n{json.dumps(spec, indent=2, default=str)}\n```",
        f"# Automated verification (already run)\n\n```\n{harness_summary}\n```",
        f"# The generated code\n\n```python\n{listing}\n```",
        f"# How to review this\n\n{framing}",
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
        "Produce ONE module: a module-level `SPEC: ExternalToolSpec` plus one plain "
        "function per FunctionSpec. Validate inputs BEFORE importing the wrapped package, "
        "so a bad argument raises at the call boundary instead of failing inside the "
        "package (or, if it isn't installed, surfacing as an unrelated ImportError). "
        "Golden cases must be values you are confident about a priori — never invented "
        "numbers.",
    ]
    if feedback:
        sections.append(f"# Your previous attempt failed verification\n\n{feedback}")

    return "\n\n".join(sections)
