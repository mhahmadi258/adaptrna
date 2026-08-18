"""`AdapterRuntime`: one shared backbone serving every active adapter tool.

Lazy by decision (plans/PHASE_2_TOOLHUB_CORE.md §2): registry operations never load the
backbone — the first forward-pass call does, via the engine's `RiNALMoHub`, and every
active adapter is made resident in that single backbone. Deactivation is routing-level
(this class refuses to route to a disabled tool; peft cannot cleanly uninject an
adapter); `rebuild()` is the full cleanup.

Engine imports live inside functions so importing this module stays torch-free.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import threading
import math

from adaptrna_agentic.toolhub.manifest import ToolEntry, resolve_path
from adaptrna_agentic.toolhub.registry import Registry, ToolHubError


class AdapterRuntime:
    def __init__(self, registry: Registry):
        self.registry = registry
        self._hub = None
        self._resident: set = set()
        # Serialises inference on the shared backbone. `RiNALMoHub.predict` calls
        # `activate()`, which flips the active adapter on *every* tuner layer, so two
        # concurrent predictions for different adapters could interleave that mutation
        # and silently answer from the wrong one. The lock lives here rather than at
        # the call sites so every caller — CLI, chat tools, HTTP — is covered.
        self.inference_lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        """Is the backbone resident in this process?"""
        return self._hub is not None

    # ------------------------------------------------------------------ lifecycle

    def warmup(self) -> List[str]:
        """Eagerly load the backbone and make every active adapter resident.

        Returns the problems it hit rather than raising: warmup is a convenience, and one
        tool with a missing artifact must not stop a chat from starting. The per-tool
        check still fires — with the same message — the moment that tool is actually used.
        """
        if self._hub is None:
            self._hub = self._build_hub()

        return self._register_active(strict=False)

    def rebuild(self) -> None:
        """Drop the resident hub entirely — the full-cleanup counterpart of routing-level
        deactivation."""
        self._hub = None
        self._resident = set()

    def _build_hub(self):
        backbone = self.registry.manifest.backbone

        weights = None
        if backbone.weights:
            weights_path = resolve_path(backbone.weights)
            if not weights_path.exists():
                raise ToolHubError(
                    f"Backbone weights '{backbone.weights}' not found (resolved to "
                    f"'{weights_path}'). Point the ToolHub at your checkpoint with "
                    f"`toolhub config --weights /path/to/giga-v1.pt`, or see the engine "
                    f"README's Setup section to download it."
                )
            weights = str(weights_path)

        try:
            from rinalmo_hub.hub import RiNALMoHub
        except ImportError as exc:
            raise ToolHubError(
                "The engine package is not installed in this environment. "
                "Run `pip install -e ./engine` from the repo root."
            ) from exc

        self._load_custom_tasks()

        return RiNALMoHub(
            backbone_weights=weights,
            lm_config=backbone.lm_config,
            device=backbone.resolved_device(),
            dtype=backbone.resolved_dtype(),
        )

    @staticmethod
    def _load_custom_tasks() -> None:
        """Generated tasks must be imported before the hub can serve their adapters.

        The engine resolves an adapter's task through `get_task(name)`, which only knows
        what `@register_task` has registered — so without this a generated task would
        train fine and then fail to serve.
        """
        from adaptrna_agentic.codegen.discovery import load_all

        load_all()

    @staticmethod
    def _require_artifact(entry) -> None:
        """A missing adapter file otherwise surfaces as a raw torch/OS error from deep
        inside the engine's loader, which never names the tool that is broken."""
        path = entry.artifact_path()
        if path is None or not path.exists():
            raise ToolHubError(
                f"Tool '{entry.name}' points at '{path}', which does not exist. "
                f"Restore the file, or drop the tool with `toolhub remove {entry.name}`. "
                f"(`toolhub doctor` lists every such mismatch.)"
            )

    def _register_active(self, strict: bool = True) -> List[str]:
        problems: List[str] = []

        for entry in self.registry.list():
            if entry.type == "adapter" and entry.active and entry.name not in self._resident:
                try:
                    self._require_artifact(entry)
                    self._hub.register(str(entry.artifact_path()), name=entry.name)
                except ToolHubError as exc:
                    if strict:
                        raise
                    problems.append(str(exc))
                    continue
                self._resident.add(entry.name)

        return problems

    def _ensure_tool(self, name: str) -> ToolEntry:
        entry = self.registry.get(name)

        if entry.type != "adapter":
            raise ToolHubError(
                f"Tool '{name}' is of type '{entry.type}'; only adapter tools run on the "
                f"backbone runtime. Use `toolhub call {name} ...` for external tools."
            )
        if not entry.active:
            raise ToolHubError(
                f"Tool '{name}' is disabled. Enable it with `toolhub activate {name}`."
            )

        if self._hub is None:
            self._hub = self._build_hub()
        self._require_artifact(entry)
        if name not in self._resident:
            # Covers tools registered or re-activated after the hub was built.
            self._hub.register(str(entry.artifact_path()), name=name)
            self._resident.add(name)

        return entry

    # ------------------------------------------------------------------ inference

    def predict(
        self,
        name: str,
        sequences: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
    ):
        """Run `name` over `sequences`, returning the task's native output type.

        Serialised: the hub activates this tool's adapter across the whole backbone
        before running, so overlapping calls for different tools would answer from
        whichever adapter was activated last.
        """
        with self.inference_lock:
            entry = self._ensure_tool(name)

            if isinstance(sequences, str):
                sequences = [sequences]

            # Per-tool serving policy; None falls through to the task's own default.
            effective_batch = batch_size or entry.serving.get("batch_size")
            return self._hub.predict(name, list(sequences), batch_size=effective_batch)

    # ------------------------------------------------------------------ smoke test

    def smoke_test(self, name: str) -> Dict[str, Any]:
        """Run the tool on its stored test sequences and validate the output shape/range.

        Returns a report dict: {name, task, ok, checks: [str], outputs: summary}.
        """
        entry = self.registry.get(name)
        sequences = list(entry.test.get("sequences") or [])
        if not sequences:
            return {"name": name, "ok": False, "checks": ["no test sequences configured"]}

        checks: List[str] = []
        ok = True

        try:
            outputs = self.predict(name, sequences)
        except Exception as exc:  # noqa: BLE001 -- the report is the error channel here
            return {"name": name, "task": entry.task, "ok": False,
                    "checks": [f"predict failed: {exc}"]}

        out_list = list(outputs)
        if len(out_list) == len(sequences):
            checks.append(f"one output per sequence ({len(out_list)}): ok")
        else:
            checks.append(f"expected {len(sequences)} outputs, got {len(out_list)}: FAIL")
            ok = False

        validator = _validator_for(entry)
        problems = validator(out_list, sequences)
        if problems:
            checks.extend(f"{p}: FAIL" for p in problems)
            ok = False
        else:
            checks.append(f"{entry.task} output form: ok")

        expected = entry.test.get("expected")
        if expected is not None:
            tolerance = float(entry.test.get("tolerance", 1e-4))
            mismatches = _compare_expected(out_list, expected, tolerance)
            if mismatches:
                checks.extend(f"{m}: FAIL" for m in mismatches)
                ok = False
            else:
                checks.append(f"matches expected values (tol {tolerance}): ok")

        return {
            "name": name,
            "task": entry.task,
            "ok": ok,
            "checks": checks,
            "outputs": _summarize(out_list),
        }


# ---------------------------------------------------------------------- validators
#
# Phase 13: keyed by the tool's own recorded target type (`provenance["spec"]["target_
# type"]`), not a hardcoded task-name set — every generated task gets one, not just one
# shipped example did (plan §10). A tool with no spec falls back to `_validate_generic`;
# absence is reported, not guessed.

def _as_floats(outputs) -> Optional[List[float]]:
    try:
        return [float(value) for value in outputs]
    except (TypeError, ValueError):
        return None


def _validate_generic(outputs, sequences) -> List[str]:
    values = _as_floats(outputs)
    if values is not None and any(not math.isfinite(v) for v in values):
        return ["non-finite values in output"]
    return []


def _validate_binary(outputs, sequences) -> List[str]:
    """One probability per sequence, in [0, 1] (target_shapes.yaml: binary)."""
    values = _as_floats(outputs)
    if values is None:
        return ["expected one probability per sequence"]

    problems = []
    if any(not math.isfinite(v) for v in values):
        problems.append("non-finite probabilities")
    if any(v < 0.0 or v > 1.0 for v in values):
        problems.append("probabilities outside [0, 1]")
    return problems


def _validate_regression(outputs, sequences) -> List[str]:
    """One finite scalar per sequence, in the original target scale (target_shapes.yaml:
    regression)."""
    values = _as_floats(outputs)
    if values is None:
        return ["expected one scalar value per sequence"]
    if any(not math.isfinite(v) for v in values):
        return ["non-finite predictions"]
    return []


def _validate_multiclass(outputs, sequences, classes: Sequence[str]) -> List[str]:
    """A class label from the recorded `classes`, per sequence (target_shapes.yaml:
    multiclass — "one class label plus per-class probabilities per sequence")."""
    allowed = {str(c) for c in classes}

    for output in outputs:
        label = output.get("label") if isinstance(output, dict) else None
        if label is None:
            return [f"expected a class label per sequence, got {output!r}"]
        if allowed and str(label) not in allowed:
            return [f"label {label!r} is not one of the recorded classes {sorted(allowed)}"]

    return []


def _validator_for(entry: ToolEntry):
    spec = (entry.provenance or {}).get("spec")
    if not spec:
        return _validate_generic

    target_type = spec.get("target_type")
    if target_type == "binary":
        return _validate_binary
    if target_type == "regression":
        return _validate_regression
    if target_type == "multiclass":
        classes = spec.get("classes") or []
        return lambda outputs, sequences: _validate_multiclass(outputs, sequences, classes)

    return _validate_generic


def _compare_expected(outputs, expected, tolerance: float) -> List[str]:
    values = _as_floats(outputs)
    if values is None:
        return ["expected-value comparison only supports scalar-per-sequence outputs"]
    if len(values) != len(expected):
        return [f"expected {len(expected)} values, got {len(values)}"]

    return [
        f"output[{i}] = {v:.6f}, expected {e:.6f} (tol {tolerance})"
        for i, (v, e) in enumerate(zip(values, expected))
        if abs(v - float(e)) > tolerance
    ]


def _summarize(outputs) -> Any:
    values = _as_floats(outputs)
    if values is not None:
        return [round(v, 4) for v in values]
    return [str(getattr(o, "shape", type(o).__name__)) for o in outputs]
