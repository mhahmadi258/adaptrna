"""Deterministic rendering of task.py / datamodule.py / config.yaml from an approved
DatasetSpec (Phase 13 D13/D14).

`covers(spec)` is a whitelist predicate: known fields, in-range values, nothing else. It
is deliberately conservative and does not need to be exhaustively correct — a harness
failure or a review rejection on a spec it accepts is the real backstop (it routes to the
LLM fallback), not a bug in this function. See plan §7.2.
"""

from pathlib import Path
from typing import Any, Dict
import re

import jinja2

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_VERSION = int((TEMPLATE_DIR / "TEMPLATE_VERSION").read_text().strip())

SUPPORTED_TARGET_TYPES = ("binary", "multiclass", "regression")
SUPPORTED_SPLIT_MODES = ("random", "column")
MAX_CLASSES = 20

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=jinja2.StrictUndefined,
)
#: Python-literal embedding (None/True/False, single-quoted strings) — distinct from
#: `tojson`, whose null/true/false are valid YAML but not valid Python.
_ENV.filters["pyrepr"] = repr

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


class _NotCovered(Exception):
    """Internal signal for `covers()` — never escapes this module."""


def covers(spec: Dict[str, Any]) -> bool:
    try:
        _check_covers(spec)
        return True
    except _NotCovered:
        return False


def render(spec: Dict[str, Any]) -> Dict[str, str]:
    """Render task.py / datamodule.py / config.yaml for an approved spec.

    Callers must check `covers(spec)` first (the pipeline uses it to choose the template
    path over the LLM fallback). This re-checks and raises rather than silently emitting
    code for a spec it does not actually support.
    """
    try:
        _check_covers(spec)
    except _NotCovered as exc:
        raise ValueError(f"template does not cover this spec: {exc}") from exc

    context = _context(spec)
    return {
        "task.py": _ENV.get_template("task.py.j2").render(**context),
        "datamodule.py": _ENV.get_template("datamodule.py.j2").render(**context),
        "config.yaml": _ENV.get_template("config.yaml.j2").render(**context),
    }


# ---------------------------------------------------------------- covers()

def _check_covers(spec: Dict[str, Any]) -> None:
    target_type = spec.get("target_type")
    if target_type not in SUPPORTED_TARGET_TYPES:
        raise _NotCovered("target_type")

    task_name = spec.get("task_name")
    if not isinstance(task_name, str) or not _IDENTIFIER.match(task_name):
        raise _NotCovered("task_name")

    if not spec.get("sequence_column") or not spec.get("label_column"):
        raise _NotCovered("sequence_column/label_column")

    if not spec.get("path"):
        raise _NotCovered("path")

    fmt = spec.get("format") or {}
    if fmt.get("separator") not in (",", "\t"):
        raise _NotCovered("format.separator")

    head = spec.get("head") or {}
    if not head.get("primary_metric"):
        raise _NotCovered("head.primary_metric")

    if target_type in ("binary", "multiclass"):
        classes = spec.get("classes") or []
        if not (1 < len(classes) <= MAX_CLASSES):
            raise _NotCovered("classes")
        if target_type == "binary":
            if len(classes) != 2:
                raise _NotCovered("classes")
            positive_class = spec.get("positive_class")
            if positive_class is None or str(positive_class) not in [str(c) for c in classes]:
                raise _NotCovered("positive_class")

    split = spec.get("split") or {}
    mode = split.get("mode")
    if mode not in SUPPORTED_SPLIT_MODES:
        raise _NotCovered("split.mode")

    if mode == "random":
        fractions = split.get("fractions") or {}
        if set(fractions) != {"train", "val", "test"}:
            raise _NotCovered("split.fractions")
        if any(not isinstance(v, (int, float)) or v < 0 for v in fractions.values()):
            raise _NotCovered("split.fractions")
        if abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise _NotCovered("split.fractions")
        if not isinstance(split.get("seed"), int):
            raise _NotCovered("split.seed")
    else:
        if not split.get("column"):
            raise _NotCovered("split.column")
        mapping = split.get("mapping") or {}
        if not mapping or not all(isinstance(v, list) and v for v in mapping.values()):
            raise _NotCovered("split.mapping")


# ---------------------------------------------------------------- context

def _context(spec: Dict[str, Any]) -> Dict[str, Any]:
    target_type = spec["target_type"]
    split = spec["split"]
    fmt = spec.get("format") or {}

    return {
        "template_version": TEMPLATE_VERSION,
        "class_name": _class_name(spec["task_name"]),
        "task_name": spec["task_name"],
        "tool_description": spec.get("tool_description") or spec["task_name"],
        "target_type": target_type,
        "sequence_column": spec["sequence_column"],
        "label_column": spec["label_column"],
        "separator": fmt.get("separator", ","),
        "compression": fmt.get("compression"),
        "on_invalid": spec.get("on_invalid", "fail"),
        "classes": spec.get("classes"),
        "positive_class": spec.get("positive_class") if target_type == "binary" else None,
        "primary_metric": spec["head"]["primary_metric"],
        "data_path": spec["path"],
        "split_mode": split["mode"],
        "split_fractions": split.get("fractions"),
        "split_seed": split.get("seed"),
        "split_stratify": bool(split.get("stratify")) and target_type != "regression",
        "split_column": split.get("column"),
        "split_mapping": split.get("mapping"),
    }


def _class_name(task_name: str) -> str:
    return "".join(part.capitalize() for part in task_name.split("_")) + "Module"
