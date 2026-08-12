"""`BaseDownstreamModule` -- everything a downstream task does *not* have to reimplement.

Backbone construction, LoRA injection and freezing, adapter save/load, checkpoint slimming,
optimizer and scheduler, and the generic train/val/test steps live here. A task subclass
supplies only the pieces that are genuinely task-specific; see `rinalmo_hub/tasks/` for the
three shipped ones and the README's "Adding a new task" walkthrough for the contract.

Naming is fixed once and for all: the backbone is `self.backbone`, the head is `self.head`.
The source repo called the backbone `self.rinalmo` in two scripts and `self.lm` in four,
which made the `ft_schedules/*.yaml` regexes non-interchangeable between tasks.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn

from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo

from rinalmo_hub.adapter import build_metadata, load_adapter, save_adapter
from rinalmo_hub.lora import (
    DEFAULT_ADAPTER_NAME,
    LoRAAdapterMixin,
    LoRASpec,
)

STAGES = ("train", "val", "test")

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


class BaseDownstreamModule(LoRAAdapterMixin, pl.LightningModule):
    """
    Base class for every downstream task.

    Subclass contract (see the README walkthrough for a full example):

        build_head(embed_dim, **head_config) -> nn.Module      required
        extract_features(representation, tokens)               required
        compute_loss(outputs, batch)                           required
        update_metrics(outputs, batch, stage)                  required
        compute_metrics(stage) -> dict                         required
        build_datamodule(cfg) -> LightningDataModule           required (staticmethod)

        build_metrics(stage) -> nn.Module | None               optional
        batch_tokens(batch) -> Tensor                          optional (default: batch[0])
        postprocess_predictions(outputs, tokens, sequences)    optional (default: outputs)
        adapter_extra_payload() / load_adapter_extra(extra)    optional, non-tensor state
    """

    #: Set by `@register_task`.
    TASK_NAME: Optional[str] = None

    #: Extra *tensor* state-dict prefixes shipped inside the adapter file, e.g. `("scaler.",)`.
    ADAPTER_EXTRA_PREFIXES: Sequence[str] = ()

    #: Sequences per forward pass in `RiNALMoHub.predict`. Tasks whose head is quadratic in
    #: sequence length, or that cannot tolerate padding, override this.
    DEFAULT_PREDICT_BATCH_SIZE: int = 8

    #: Metric name the README tells users to read for this task, and whether higher is better.
    PRIMARY_METRIC: Optional[str] = None

    def __init__(
        self,
        lm_config: str = "giga",
        head_config: Optional[dict] = None,
        lora: Optional[dict] = None,
        optim: Optional[dict] = None,
        finetune: Optional[dict] = None,
        task_config: Optional[dict] = None,
        attach_backbone: bool = True,
    ):
        """
        Args:
            lm_config: RiNALMo backbone size -- "nano", "micro", "mega" or "giga".
            head_config: kwargs forwarded to `build_head`, and stored in the adapter file.
            lora: LoRA geometry dict, or None for full fine-tuning / head-only training.
            optim: `{name, lr, weight_decay, scheduler: {...}}`.
            finetune: `{schedule, initial_denom_lr}` for gradual unfreezing (full FT only).
            task_config: task-specific knobs that are not head kwargs, from the YAML's
                `task_config:` block (e.g. sec-struct's `tune_threshold_every_n_epoch`).
            attach_backbone: build a backbone of our own. `RiNALMoHub` passes False and
                assigns a shared backbone afterwards, so that N tasks cost one backbone.
        """
        super().__init__()
        self.save_hyperparameters()

        self.lm_config = lm_config
        self.config = model_config(lm_config)
        self.embed_dim = self.config["model"]["transformer"].embed_dim
        self.pad_idx = self.config["model"]["embedding"].padding_idx

        # Registered as a child either way, so `RiNALMoHub` can drop a shared backbone in.
        self.add_module("backbone", RiNALMo(self.config) if attach_backbone else None)

        self.head_config = dict(head_config or {})
        self.head = self.build_head(self.embed_dim, **self.head_config)

        self.lora_spec = LoRASpec.from_dict(lora)
        self.lora_adapter_name = DEFAULT_ADAPTER_NAME

        self.optim_config = dict(optim or {})
        self.finetune_config = dict(finetune or {})
        self.task_config = dict(task_config or {})

        metrics = {stage: self.build_metrics(stage) for stage in STAGES}
        self.metrics = nn.ModuleDict({k: v for k, v in metrics.items() if v is not None})

    # ------------------------------------------------------------------ subclass hooks

    def build_head(self, embed_dim: int, **head_config) -> nn.Module:
        raise NotImplementedError

    def extract_features(self, representation: torch.Tensor, tokens: torch.Tensor):
        """
        Turn the backbone's per-token representation into what the head consumes.

        Return a tensor, or a tuple that is splatted into the head as positional arguments.
        Getting this wrong is silent, so it is an explicit hook rather than a convention:
        splice site wants the CLS token, MRL wants padded positions zeroed plus the pad
        mask, secondary structure wants CLS and EOS dropped.
        """
        raise NotImplementedError

    def compute_loss(self, outputs, batch) -> torch.Tensor:
        raise NotImplementedError

    def update_metrics(self, outputs, batch, stage: str) -> None:
        raise NotImplementedError

    def compute_metrics(self, stage: str) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def build_datamodule(cfg) -> pl.LightningDataModule:
        raise NotImplementedError

    def build_metrics(self, stage: str) -> Optional[nn.Module]:
        """Per-stage metric object exposed as `self.metrics[stage]`. Optional."""
        return None

    def reset_metrics(self, stage: str) -> None:
        if stage in self.metrics:
            self.metrics[stage].reset()

    def batch_tokens(self, batch) -> torch.Tensor:
        """Pull the token tensor out of a batch. Default: the first element."""
        return batch[0]

    def postprocess_predictions(self, outputs, tokens, sequences):
        """Map raw head outputs to the task's native prediction type. Used by the hub."""
        return outputs

    def adapter_extra_payload(self) -> dict:
        """*Non-tensor* task state to ship in the adapter file (e.g. a tuned threshold)."""
        return {}

    def load_adapter_extra(self, extra: dict) -> None:
        """Restore whatever `adapter_extra_payload` produced. Must tolerate an empty dict."""
        return None

    def on_fit_start_hook(self) -> None:
        """Task-specific setup that needs the datamodule, e.g. fitting a target scaler."""
        return None

    # ------------------------------------------------------------------ backbone / LoRA

    def load_pretrained_backbone(self, weights_path: Union[str, Path]) -> None:
        """
        Load pretrained RiNALMo weights.

        Order matters: this must run *before* `apply_lora()`, because injection moves each
        target's weight to `<target>.base_layer.weight` and a strict load would then fail.
        """
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(state_dict)

    # ------------------------------------------------------------------ adapter I/O

    @property
    def training_arm(self) -> str:
        """Which arm produced these weights: "lora" or "full_ft". Recorded in the adapter."""
        return "lora" if self.use_lora else "full_ft"

    def save_adapter(self, path: Union[str, Path], metadata: Optional[dict] = None) -> Path:
        metadata = dict(metadata or build_metadata())
        metadata.setdefault("arm", self.training_arm)

        return save_adapter(
            path,
            task=self.TASK_NAME,
            lm_config=self.lm_config,
            lora=self.lora_spec.to_dict() if self.lora_spec is not None else None,
            head_config=self.head_config,
            state_dict=self.adapter_state_dict(),
            extra=self.adapter_extra_payload(),
            metadata=metadata,
        )

    def load_adapter(self, path: Union[str, Path]) -> dict:
        payload = load_adapter(path, lm_config=self.lm_config)

        if payload["task"] != self.TASK_NAME:
            raise ValueError(
                f"Adapter '{path}' is for task '{payload['task']}', not '{self.TASK_NAME}'"
            )

        expected_lora = self.lora_spec.to_dict() if self.lora_spec is not None else None
        if payload["lora"] != expected_lora:
            raise ValueError(
                f"Adapter '{path}' has LoRA geometry {payload['lora']} but this model was "
                f"built with {expected_lora}. Build the model from the adapter's own config "
                f"(the CLI does this automatically when you pass --adapter)."
            )

        self.load_adapter_state_dict(payload["state_dict"], source=str(path))
        self.load_adapter_extra(payload["extra"])
        print(f"Loaded {len(payload['state_dict'])} adapter tensors from '{path}'")

        return payload

    # ------------------------------------------------------------------ forward / steps

    def forward(self, tokens: torch.Tensor):
        representation = self.backbone(tokens)["representation"]
        features = self.extract_features(representation, tokens)

        if isinstance(features, tuple):
            return self.head(*features)

        return self.head(features)

    def _step(self, batch, stage: str):
        outputs = self(self.batch_tokens(batch))
        loss = self.compute_loss(outputs, batch)

        self.log(f"{stage}/loss", loss, sync_dist=True, add_dataloader_idx=False)
        self.update_metrics(outputs, batch, stage)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _log_epoch_metrics(self, stage: str) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            self.reset_metrics(stage)
            return

        metrics = self.compute_metrics(stage)
        if metrics:
            self.log_dict(metrics, sync_dist=True, add_dataloader_idx=False)

        self.reset_metrics(stage)

    def on_train_epoch_start(self):
        self.reset_metrics("train")

    def on_validation_epoch_start(self):
        self.reset_metrics("val")

    def on_test_epoch_start(self):
        self.reset_metrics("test")

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def on_fit_start(self):
        self.on_fit_start_hook()

    # ------------------------------------------------------------------ optimization

    def configure_optimizers(self):
        cfg = self.optim_config
        name = str(cfg.get("name", "adam")).lower()

        if name not in _OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{name}'. Known: {sorted(_OPTIMIZERS)}")

        # LoRA mode: only the adapters and the head have `requires_grad`, and handing the
        # optimizer the frozen backbone as well is what makes "LoRA" runs quietly cost
        # full-FT memory. Full FT: every parameter, unless a `GradualUnfreezing` schedule
        # has already frozen most of them in its `setup` hook -- from then on the callback
        # owns the param groups and adds each unfrozen block itself.
        params = [p for p in self.parameters() if p.requires_grad]

        kwargs = {"lr": float(cfg.get("lr", 1e-4))}
        weight_decay = cfg.get("weight_decay")
        if weight_decay is not None:
            kwargs["weight_decay"] = float(weight_decay)
        if name == "sgd":
            kwargs["momentum"] = float(cfg.get("momentum", 0.9))

        optimizer = _OPTIMIZERS[name](params, **kwargs)

        scheduler = self._build_scheduler(optimizer, cfg.get("scheduler"))
        if scheduler is None:
            return {"optimizer": optimizer}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def _build_scheduler(optimizer, cfg: Optional[dict]):
        if not cfg:
            return None

        cfg = dict(cfg)
        name = str(cfg.pop("name", "none")).lower()
        interval = cfg.pop("interval", "step")

        if name in ("none", "null", ""):
            return None

        if name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(cfg.pop("start_factor", 1.0)),
                end_factor=float(cfg.pop("end_factor", 0.1)),
                total_iters=int(cfg.pop("total_iters", 5000)),
            )
        elif name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(cfg.pop("T_max", 200_000)),
                eta_min=float(cfg.pop("eta_min", 1e-5)),
            )
        elif name == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=float(cfg.pop("factor", 1.0)),
                total_iters=int(cfg.pop("total_iters", 1)),
            )
        else:
            raise ValueError(f"Unknown scheduler '{name}'. Known: linear, cosine, constant, none")

        if cfg:
            raise ValueError(f"Unused scheduler config keys for '{name}': {sorted(cfg)}")

        return {"scheduler": scheduler, "interval": interval}
