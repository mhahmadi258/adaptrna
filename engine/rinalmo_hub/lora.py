"""LoRA injection, freezing and multi-adapter switching.

Generalised from the single-adapter implementation in the source repo. The differences:

* the adapter name is a parameter, so several adapters can be resident in one backbone;
* the backbone is always found at `self.backbone`, never via a per-class attribute name;
* `set_active_adapter` / `disable_adapters` expose peft's switching to callers.

Everything else is carried over deliberately -- see the comments, each marks something that
was learned by getting it wrong.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence
import warnings

import torch

# Modules within a single transformer block that LoRA adapts by default. The flash attention
# path uses a single fused `Wqkv` projection, so one adapter there covers q, k and v; there
# are no separate q/k/v Linears to target. Both targets are bias-free, hence `bias="none"`.
LORA_TARGET_MODULES = ["mh_attn.Wqkv", "mh_attn.out_proj"]

DEFAULT_ADAPTER_NAME = "default"


@dataclass
class LoRASpec:
    """The geometry of a LoRA configuration: everything needed to reproduce the injection."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    layer_stride: int = 3
    layers: Optional[List[int]] = None
    target_modules: List[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> Optional["LoRASpec"]:
        if cfg is None:
            return None

        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"Unknown LoRA config keys: {unknown}. Known keys: {sorted(known)}")

        spec = cls(**cfg)
        if spec.target_modules is None:
            spec.target_modules = list(LORA_TARGET_MODULES)
        if spec.layers is not None:
            spec.layers = [int(i) for i in spec.layers]
        spec.target_modules = list(spec.target_modules)

        return spec

    def to_dict(self) -> dict:
        return asdict(self)

    def layer_indices(self, num_blocks: int) -> List[int]:
        """Block indices this spec adapts."""
        if self.layers is not None:
            indices = sorted(set(self.layers))
            out_of_range = [i for i in indices if i < 0 or i >= num_blocks]
            if out_of_range:
                raise ValueError(
                    f"LoRA layers {out_of_range} are outside the backbone's {num_blocks} blocks"
                )
            return indices

        if self.layer_stride < 1:
            raise ValueError(f"lora.layer_stride must be >= 1, got {self.layer_stride}")

        # Every stride-th block, counting from the input: stride 3 picks 0, 3, ... 30 out of
        # the 33 "giga" blocks.
        return list(range(0, num_blocks, self.layer_stride))

    def target_module_names(self, num_blocks: int) -> List[str]:
        return [
            f"transformer.blocks.{i}.{target}"
            for i in self.layer_indices(num_blocks)
            for target in self.target_modules
        ]


def backbone_num_blocks(backbone: torch.nn.Module) -> int:
    return backbone.config["model"]["transformer"].num_blocks


def inject_lora(
    backbone: torch.nn.Module,
    spec: LoRASpec,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    verbose: bool = True,
) -> List[str]:
    """
    Inject `spec` into `backbone` under `adapter_name`; returns the adapted module names.

    Must be called *after* the pretrained backbone weights are loaded: injection moves the
    original weight of every target to `<target>.base_layer.weight`, so a strict load
    afterwards fails.
    """
    # `inject_adapter_in_model` swaps the target `nn.Linear`s in place. `get_peft_model`
    # would instead wrap the backbone and rename every state dict key to `base_model.model.*`,
    # which breaks checkpoint compatibility with plain RiNALMo weights.
    from peft import LoraConfig, inject_adapter_in_model

    num_blocks = backbone_num_blocks(backbone)
    targets = spec.target_module_names(num_blocks)

    lora_config = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=targets,
        bias="none",  # Both `Wqkv` and `out_proj` are bias-free
    )

    try:
        with warnings.catch_warnings():
            # Injecting a second adapter into the same backbone is the whole point of the
            # hub, so peft's "Already found a `peft_config` attribute" warning is not news.
            warnings.filterwarnings("ignore", message=".*multiple adapters.*")
            inject_adapter_in_model(lora_config, backbone, adapter_name=adapter_name)
    except ValueError as exc:
        raise RuntimeError(_mismatch_message(0, targets, spec)) from exc

    # peft only complains when *nothing* matched. A partial match is the dangerous one: it
    # trains a handful of blocks plus the head and looks exactly like "LoRA doesn't work",
    # so check the count ourselves.
    adapted = adapted_module_names(backbone, adapter_name)
    if len(adapted) != len(targets):
        raise RuntimeError(_mismatch_message(len(adapted), targets, spec))

    if verbose:
        print(
            f"LoRA adapter '{adapter_name}' on blocks {spec.layer_indices(num_blocks)} "
            f"({len(adapted)} modules adapted, r={spec.r}, alpha={spec.alpha})"
        )

    return adapted


def _mismatch_message(adapted: int, targets: List[str], spec: LoRASpec) -> str:
    return (
        f"LoRA injection adapted {adapted} modules but {len(targets)} were requested. "
        f"Check `lora.target_modules` ({spec.target_modules}) against the backbone's module "
        f"names, e.g. 'transformer.blocks.0.mh_attn.Wqkv'."
    )


def _tuner_layers(backbone: torch.nn.Module):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in backbone.named_modules():
        if isinstance(module, BaseTunerLayer):
            yield name, module


def adapted_module_names(backbone: torch.nn.Module, adapter_name: str) -> List[str]:
    """Names of the modules carrying `adapter_name`."""
    return [name for name, module in _tuner_layers(backbone) if adapter_name in module.lora_A]


def resident_adapters(backbone: torch.nn.Module) -> List[str]:
    names = set()
    for _, module in _tuner_layers(backbone):
        names.update(module.lora_A.keys())

    return sorted(names)


def freeze_backbone_except_lora(model: torch.nn.Module) -> None:
    """
    `inject_adapter_in_model` freezes nothing by itself, so do it explicitly.

    Applied to the backbone only; the prediction head stays trainable.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def set_active_adapter(backbone: torch.nn.Module, adapter_name: str) -> None:
    """
    Make `adapter_name` the active adapter on every layer that carries it.

    peft keeps a per-adapter dict inside each `lora.Linear`, and its forward skips any
    active adapter it does not hold, so layers adapted by a *different* task's geometry are
    left as plain base layers. That is what lets adapters with different `layer_stride`
    values coexist in one backbone.
    """
    resident = resident_adapters(backbone)
    if adapter_name not in resident:
        raise KeyError(f"Adapter '{adapter_name}' is not resident. Resident adapters: {resident}")

    for _, module in _tuner_layers(backbone):
        module.set_adapter(adapter_name)


@contextmanager
def disable_adapters(backbone: torch.nn.Module):
    """Temporarily run the bare pretrained backbone, with every adapter bypassed."""
    layers = [module for _, module in _tuner_layers(backbone)]
    try:
        for module in layers:
            module.enable_adapters(False)
        yield backbone
    finally:
        for module in layers:
            module.enable_adapters(True)


# peft stores per-adapter weights in `<target>.<submodule>.<adapter_name>.weight`, so the
# adapter name is baked into every LoRA state-dict key. Training always writes "default";
# the hub injects under the task name so several adapters can be resident, which means the
# keys have to be remapped on the way in.
_LORA_SUBMODULES = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector")


def adapter_names_in_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    names = set()
    for key in state_dict:
        parts = key.split(".")
        for i, part in enumerate(parts[:-1]):
            if part in _LORA_SUBMODULES:
                names.add(parts[i + 1])

    return sorted(names)


def rename_adapter_in_state_dict(
    state_dict: Dict[str, torch.Tensor], new_name: str
) -> Dict[str, torch.Tensor]:
    """Rewrite every `lora_*.<old_name>.*` key to use `new_name`."""
    names = adapter_names_in_state_dict(state_dict)
    if not names:
        return dict(state_dict)
    if len(names) > 1:
        raise ValueError(f"State dict carries several LoRA adapter names ({names}); cannot rename")

    old_name = names[0]
    if old_name == new_name:
        return dict(state_dict)

    renamed = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        for i, part in enumerate(parts[:-1]):
            if part in _LORA_SUBMODULES and parts[i + 1] == old_name:
                parts[i + 1] = new_name
        renamed[".".join(parts)] = value

    return renamed


def lora_adapter_name_of(key: str) -> Optional[str]:
    """The adapter a LoRA state-dict key belongs to, or None if it is not a LoRA key."""
    parts = key.split(".")
    for i, part in enumerate(parts[:-1]):
        if part in _LORA_SUBMODULES:
            return parts[i + 1]

    return None


def is_adapter_key(
    key: str, extra_prefixes: Sequence[str] = (), adapter_name: Optional[str] = None
) -> bool:
    """
    Does this state-dict key belong in *this* adapter's file?

    `adapter_name` matters as soon as a backbone holds more than one adapter -- which is the
    normal case inside `RiNALMoHub`. Without it, one task's adapter state dict would sweep
    up every other resident task's LoRA weights.
    """
    if key.startswith(("head.",) + tuple(extra_prefixes)):
        return True

    name = lora_adapter_name_of(key)
    if name is None:
        return False

    return adapter_name is None or name == adapter_name


def trainable_parameter_summary(model: torch.nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {"trainable": trainable, "total": total}


class LoRAAdapterMixin:
    """
    LoRA plumbing for a downstream `LightningModule` that owns `self.backbone` and `self.head`.

    Mixed into `BaseDownstreamModule`. Kept free of Lightning imports so it stays testable
    on its own.
    """

    # Extra state-dict prefixes to ship inside the adapter file, on top of `lora_*` and
    # `head.*`. Anything a task learns that is not a head weight but that predictions
    # depend on belongs here, e.g. `("scaler.",)` for the MRL target scaler.
    ADAPTER_EXTRA_PREFIXES: Sequence[str] = ()

    lora_spec: Optional[LoRASpec] = None
    lora_adapter_name: str = DEFAULT_ADAPTER_NAME

    @property
    def use_lora(self) -> bool:
        return self.lora_spec is not None

    def apply_lora(self, adapter_name: str = DEFAULT_ADAPTER_NAME, verbose: bool = True) -> None:
        if self.lora_spec is None:
            raise RuntimeError("apply_lora() called but no LoRA spec is configured")

        inject_lora(self.backbone, self.lora_spec, adapter_name=adapter_name, verbose=verbose)
        self.lora_adapter_name = adapter_name

        # Freeze the backbone but not the head: `freeze_backbone_except_lora` is applied to
        # the backbone subtree only.
        freeze_backbone_except_lora(self.backbone)

        if verbose:
            counts = trainable_parameter_summary(self)
            print(
                f"{counts['trainable']:,} trainable / {counts['total']:,} total "
                f"({100 * counts['trainable'] / counts['total']:.3f}%)"
            )

    def is_adapter_key(self, key: str) -> bool:
        return is_adapter_key(
            key,
            self.ADAPTER_EXTRA_PREFIXES,
            adapter_name=self.lora_adapter_name if self.use_lora else None,
        )

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if self.is_adapter_key(k)}

    def expected_adapter_keys(self) -> set:
        return {k for k in self.state_dict() if self.is_adapter_key(k)}

    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor], source: str = "<dict>") -> None:
        """
        Load adapter tensors, checking the key sets in *both* directions.

        `strict=False` below is what makes the missing backbone keys acceptable, but it
        would just as happily accept a stale file and leave the head randomly initialised,
        so neither `expected - got` nor `got - expected` may be non-empty.
        """
        expected = self.expected_adapter_keys()
        missing = sorted(expected - set(state_dict))
        unknown = sorted(set(state_dict) - expected)

        if missing:
            raise KeyError(
                f"Adapter '{source}' is missing {len(missing)} key(s) this model expects, "
                f"e.g. {missing[:3]}. A LoRA geometry mismatch (different `layer_stride`, "
                f"`r` or `target_modules`) is the usual cause."
            )
        if unknown:
            raise KeyError(
                f"Adapter '{source}' has {len(unknown)} key(s) this model does not expect, "
                f"e.g. {unknown[:3]}."
            )

        self.load_state_dict(state_dict, strict=False)

    @property
    def strict_loading(self) -> bool:
        # LoRA checkpoints only carry the adapters, head and extras, so Lightning must not
        # insist on the backbone keys when restoring one.
        return not self.use_lora

    def on_save_checkpoint(self, checkpoint):
        # Without this a "LoRA" checkpoint still stores the full frozen backbone: 7.8 GB
        # per file instead of 18 MB.
        if self.use_lora:
            checkpoint["state_dict"] = {
                k: v for k, v in checkpoint["state_dict"].items() if self.is_adapter_key(k)
            }
