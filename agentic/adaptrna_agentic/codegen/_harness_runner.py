"""The verification harness, executed inside the sandbox.

    python -m adaptrna_agentic.codegen._harness_runner <spec.json>

Runs the checks in plans/PHASE_6_TOOLSMITH_VERIFIER.md §5 against a nano backbone and
emits one JSON report. Every check is deterministic; each maps to a way generated code
actually breaks. Nothing here asks a model anything.

Check 6 — adapter round-trip *prediction* equivalence — is the important one. The
project's number-one silent failure is task state that predictions depend on but that
never reaches the adapter file. Instead of asking a reviewer "did you remember
ADAPTER_EXTRA_PREFIXES?", this proves it: randomise everything the adapter is supposed to
carry, predict, save, reload into a fresh module, predict again, and require the outputs
to be identical.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import sys
import traceback

LM_CONFIG = "nano"

#: Small heads keep a full forward pass instant on CPU.
NANO_LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 3}

DEFAULT_SEQUENCES = [
    "GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC",
    "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC",
]


class CheckFailed(Exception):
    """A check that failed for a reason we understand and can report."""


class CheckSkipped(Exception):
    """A check that could not run (usually: the dataset is not available here)."""


def main(argv: Optional[List[str]] = None) -> int:
    from adaptrna_agentic.codegen.sandbox import emit_payload

    argv = argv if argv is not None else sys.argv[1:]
    spec = json.loads(Path(argv[0]).read_text())

    harness = Harness(spec)
    report = harness.run()
    emit_payload(report)

    return 0 if report["ok"] else 1


class Harness:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.task_name: str = spec["task_name"]
        self.task_module: Optional[str] = spec.get("task_module")
        self.config_path: Optional[str] = spec.get("config_path")
        self.sequences: List[str] = spec.get("sequences") or DEFAULT_SEQUENCES
        self.checks: List[Dict[str, Any]] = []

        self.cfg = None
        self.head_config: Dict[str, Any] = {}
        self.task_config: Dict[str, Any] = {}
        self.batch = None

    # ------------------------------------------------------------------ plumbing

    def run(self) -> Dict[str, Any]:
        from adaptrna_agentic.codegen.discovery import ensure_importable

        # Repo root first, then staging roots *above* it: staged code is verified before
        # it lands, and its tree mirrors the final layout, so `adaptrna_custom.tasks.<x>`
        # must resolve to the staged copy rather than to the (identically named, not yet
        # containing it) package at the repo root. Order is load-bearing — inserting the
        # repo root afterwards would shadow the staging tree.
        ensure_importable()

        for entry in reversed(self.spec.get("sys_path") or []):
            if entry in sys.path:
                sys.path.remove(entry)
            sys.path.insert(0, entry)

        for name, method in (
            ("import", self.check_import),
            ("config_head", self.check_config_head),
            ("datamodule", self.check_datamodule),
            ("forward_backward", self.check_forward_backward),
            ("metrics", self.check_metrics),
            ("adapter_roundtrip", self.check_adapter_roundtrip),
            ("serving", self.check_serving),
        ):
            if self.spec.get("only") and name not in self.spec["only"]:
                continue
            self._record(name, method)

        failed = [c for c in self.checks if c["status"] == "fail"]

        return {
            "task": self.task_name,
            "ok": not failed,
            "checks": self.checks,
            "failed": [c["name"] for c in failed],
        }

    def _record(self, name: str, method) -> None:
        try:
            detail = method()
            self.checks.append({"name": name, "status": "pass", "detail": detail or ""})
        except CheckSkipped as exc:
            self.checks.append({"name": name, "status": "skip", "detail": str(exc)})
        except CheckFailed as exc:
            self.checks.append({"name": name, "status": "fail", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- unexpected: report with a traceback
            self.checks.append({
                "name": name, "status": "fail",
                "detail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            })

    # ------------------------------------------------------------------ checks

    def check_import(self) -> str:
        """1 — the module imports and registers the task under the expected name."""
        import importlib

        import rinalmo_hub.tasks  # noqa: F401 -- the shipped tasks
        from rinalmo_hub.registry import available_tasks

        if self.task_module:
            try:
                importlib.import_module(self.task_module)
            except Exception as exc:
                raise CheckFailed(
                    f"importing '{self.task_module}' raised {type(exc).__name__}: {exc}"
                ) from exc

        if self.task_name not in available_tasks():
            raise CheckFailed(
                f"'{self.task_name}' is not registered after import. Available: "
                f"{available_tasks()}. Check the @register_task name matches the config's "
                f"`task:` field."
            )

        return f"registered; available tasks: {available_tasks()}"

    def check_config_head(self) -> str:
        """2 — the config resolves and its `head:` block actually builds a head."""
        from rinalmo_hub.config import resolve_config

        if not self.config_path:
            raise CheckSkipped("no config path given")

        path = Path(self.config_path)
        if not path.is_absolute():
            from adaptrna_agentic.settings import REPO_ROOT

            path = REPO_ROOT / path
        if not path.exists():
            raise CheckFailed(f"config '{path}' does not exist")

        try:
            self.cfg = resolve_config(path)
        except Exception as exc:
            raise CheckFailed(f"the config did not resolve: {exc}") from exc

        if self.cfg.get("task") != self.task_name:
            raise CheckFailed(
                f"the config's task is '{self.cfg.get('task')}' but the module registers "
                f"'{self.task_name}' — they must match"
            )

        self.head_config = dict(self.cfg.get("head") or {})
        self.task_config = dict(self.cfg.get("task_config") or {})

        try:
            self._build_module()
        except TypeError as exc:
            raise CheckFailed(
                f"build_head rejected the config's head kwargs {sorted(self.head_config)}: "
                f"{exc}"
            ) from exc

        return f"config resolves; head built from {sorted(self.head_config) or 'defaults'}"

    def check_datamodule(self) -> str:
        """3 — the datamodule reads the real data and yields a batch."""
        if self.cfg is None:
            raise CheckSkipped("the config did not resolve")

        from rinalmo_hub.registry import get_task

        task_cls = get_task(self.task_name)

        try:
            datamodule = task_cls.build_datamodule(self.cfg)
        except Exception as exc:
            raise CheckFailed(f"build_datamodule raised {type(exc).__name__}: {exc}") from exc

        try:
            datamodule.prepare_data()
            datamodule.setup("fit")
            loader = datamodule.train_dataloader()
            self.batch = next(iter(loader))
        except FileNotFoundError as exc:
            raise CheckSkipped(f"dataset not available here: {exc}") from exc
        except Exception as exc:
            raise CheckFailed(
                f"drawing a batch raised {type(exc).__name__}: {exc}. The datamodule must "
                f"match the data actually on disk."
            ) from exc

        shapes = [tuple(getattr(item, "shape", ())) or type(item).__name__
                  for item in (self.batch if isinstance(self.batch, (list, tuple)) else [self.batch])]

        return f"batch drawn; element shapes/types: {shapes}"

    def check_forward_backward(self) -> str:
        """4 — forward runs, the loss is finite, and gradients reach the right places."""
        import torch

        if self.batch is None:
            raise CheckSkipped("no batch available")

        # apply_lora freezes the backbone, which is how a LoRA run actually trains — and
        # what keeps the masked-LM head out of the trainable set below.
        module = self._build_module(lora=dict(NANO_LORA), apply_lora=True)
        module.train()

        try:
            outputs = module(module.batch_tokens(self.batch))
        except Exception as exc:
            raise CheckFailed(
                f"forward raised {type(exc).__name__}: {exc}. extract_features must return "
                f"what the head consumes."
            ) from exc

        try:
            loss = module.compute_loss(outputs, self.batch)
        except Exception as exc:
            raise CheckFailed(f"compute_loss raised {type(exc).__name__}: {exc}") from exc

        if not torch.isfinite(loss).all():
            raise CheckFailed(f"the loss is not finite ({loss})")

        loss.backward()

        # `RiNALMo.forward` always runs its masked-LM head, whose output never reaches a
        # downstream loss (this is why the engine configures DDP with
        # find_unused_parameters=True). Never counted as a defect in the task.
        trainable = [
            (n, p) for n, p in module.named_parameters()
            if p.requires_grad and "lm_mask_head" not in n
        ]
        without_grad = [n for n, p in trainable if p.grad is None]
        if without_grad:
            raise CheckFailed(
                f"{len(without_grad)} trainable parameter(s) received no gradient, e.g. "
                f"{without_grad[:3]} — the graph is detached somewhere"
            )

        frozen_with_grad = [
            n for n, p in module.named_parameters()
            if not p.requires_grad and p.grad is not None
        ]
        if frozen_with_grad:
            raise CheckFailed(f"frozen parameters received gradients: {frozen_with_grad[:3]}")

        return f"loss {float(loss):.4f}; {len(trainable)} trainable tensors all received grads"

    def check_metrics(self) -> str:
        """5 — metrics update and reduce to finite scalars."""
        import torch

        if self.batch is None:
            raise CheckSkipped("no batch available")

        module = self._build_module()
        module.eval()

        with torch.no_grad():
            outputs = module(module.batch_tokens(self.batch))

        try:
            module.update_metrics(outputs, self.batch, "val")
            metrics = module.compute_metrics("val")
        except Exception as exc:
            raise CheckFailed(f"metrics raised {type(exc).__name__}: {exc}") from exc

        if not isinstance(metrics, dict):
            raise CheckFailed(f"compute_metrics returned {type(metrics).__name__}, not a dict")

        bad = [k for k, v in metrics.items() if not _finite_scalar(v)]
        if bad:
            raise CheckFailed(f"non-finite metric value(s): {bad}")

        return f"metrics: {sorted(metrics)}"

    def check_adapter_roundtrip(self) -> str:
        """6 — everything predictions depend on survives save → load.

        Randomise every tensor the adapter is supposed to carry (the head, the LoRA
        weights, and any task-owned state outside the backbone), predict, save, load into
        a *fresh* module and predict again. If any prediction-affecting state is missing
        from the file, the two predictions differ — which is exactly the silent failure
        this project keeps warning about.
        """
        import tempfile

        import torch

        torch.manual_seed(0)
        source = self._build_module(lora=dict(NANO_LORA), apply_lora=True)
        _randomise_adapter_state(source)
        source.eval()

        before = self._predict(source)

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "roundtrip_adapter.pt"
            try:
                source.save_adapter(path)
            except Exception as exc:
                raise CheckFailed(f"save_adapter raised {type(exc).__name__}: {exc}") from exc

            # Same seed -> an identical fresh backbone, so any difference in the
            # predictions can only come from state the adapter failed to carry.
            torch.manual_seed(0)
            target = self._build_module(lora=dict(NANO_LORA), apply_lora=True)
            try:
                target.load_adapter(path)
            except Exception as exc:
                raise CheckFailed(f"load_adapter raised {type(exc).__name__}: {exc}") from exc

        target.eval()
        after = self._predict(target)

        if not _same(before, after):
            raise CheckFailed(
                "predictions changed after a save/load round trip: the adapter file is "
                "missing state the predictions depend on. Tensor state belongs in "
                "ADAPTER_EXTRA_PREFIXES; plain Python values belong in "
                "adapter_extra_payload()/load_adapter_extra(). "
                f"before={_preview(before)} after={_preview(after)}"
            )

        return f"predictions identical across save/load ({_preview(before)})"

    def check_serving(self) -> str:
        """7 — the adapter serves through RiNALMoHub, the way a registered tool does."""
        import tempfile

        import torch

        from rinalmo_hub.hub import RiNALMoHub

        torch.manual_seed(0)
        module = self._build_module(lora=dict(NANO_LORA), apply_lora=True)
        _randomise_adapter_state(module)

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "serving_adapter.pt"
            module.save_adapter(path)

            torch.manual_seed(0)
            hub = RiNALMoHub(backbone_weights=None, lm_config=LM_CONFIG, device="cpu")
            try:
                name = hub.register(path)
                outputs = hub.predict(name, self.sequences)
            except Exception as exc:
                raise CheckFailed(
                    f"serving through RiNALMoHub raised {type(exc).__name__}: {exc}"
                ) from exc

        count = len(outputs) if hasattr(outputs, "__len__") else 1
        if count != len(self.sequences):
            raise CheckFailed(
                f"predicted {count} output(s) for {len(self.sequences)} sequences"
            )

        return f"served {count} prediction(s) through the hub"

    # ------------------------------------------------------------------ helpers

    def _build_module(self, lora=None, apply_lora: bool = False):
        from rinalmo_hub.registry import get_task

        module = get_task(self.task_name)(
            lm_config=LM_CONFIG,
            head_config=self.head_config,
            task_config=self.task_config,
            lora=lora,
        )
        if lora is not None and apply_lora:
            module.apply_lora(verbose=False)

        return module

    def _predict(self, module):
        import torch

        from rinalmo.data.alphabet import Alphabet

        tokens = torch.tensor(
            Alphabet().batch_tokenize(list(self.sequences)), dtype=torch.long
        )
        with torch.no_grad():
            outputs = module(tokens)

        return module.postprocess_predictions(outputs, tokens, list(self.sequences))


def _randomise_adapter_state(module) -> None:
    """Move every tensor the adapter should carry off its default value.

    Deliberately *not* the backbone's own weights: those are rebuilt identically from the
    seed in the target module, so touching them would make the comparison meaningless.
    LoRA tensors live inside the backbone but belong to the adapter, hence the name check.
    """
    import torch

    generator = torch.Generator().manual_seed(1234)

    def belongs_to_adapter(name: str) -> bool:
        return (not name.startswith("backbone.")) or ("lora_" in name)

    with torch.no_grad():
        for name, param in module.named_parameters():
            if belongs_to_adapter(name):
                param.copy_(torch.rand(param.shape, generator=generator) * 0.1 + 0.01)

        for name, buffer in module.named_buffers():
            if belongs_to_adapter(name) and buffer.dtype.is_floating_point:
                buffer.copy_(torch.rand(buffer.shape, generator=generator) * 0.5 + 0.5)


def _finite_scalar(value) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same(left, right) -> bool:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and bool(torch.equal(left, right))

    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and bool(np.array_equal(left, right))

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))

    return bool(left == right)


def _preview(value) -> str:
    text = str(value)
    return text if len(text) <= 160 else text[:160] + " …"


if __name__ == "__main__":
    raise SystemExit(main())
