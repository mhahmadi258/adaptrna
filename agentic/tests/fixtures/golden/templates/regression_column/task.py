# rendered by adaptrna template v2 from spec.json
"""regression target from a flat sequence/label table"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


class PooledRegressionHead(nn.Module):
    """Mean-pool the non-padded positions, then an MLP down to one scalar."""

    def __init__(self, embed_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, representation, pad_mask):
        keep = (~pad_mask).unsqueeze(-1).to(representation.dtype)
        pooled = (representation * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.mlp(pooled).squeeze(-1)


class TargetScaler(nn.Module):
    """Standardises the regression target: mean/std, fitted once on the training split.

    Held as buffers on their own submodule so they have a stable state-dict prefix —
    `ADAPTER_EXTRA_PREFIXES` below ships them inside the adapter file. Predictions depend
    on them (`postprocess_predictions` un-scales through this same object), so losing them
    on reload would silently return numbers on the wrong scale.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def fit(self, targets: torch.Tensor) -> None:
        std = targets.std()
        if not torch.isfinite(std) or std < 1e-6:
            std = torch.ones_like(std)
        self.mean.copy_(targets.mean().view(1))
        self.std.copy_(std.view(1))

    def transform(self, targets):
        return (targets - self.mean) / self.std

    def inverse_transform(self, targets):
        return targets * self.std + self.mean


@register_task("regression_column")
class RegressionColumnModule(BaseDownstreamModule):
    """regression target from a flat sequence/label table"""

    TASK_NAME = "regression_column"
    # The fitted target mean/std live in `scaler.mean` / `scaler.std` as buffers.
    ADAPTER_EXTRA_PREFIXES = ("scaler.",)
    PRIMARY_METRIC = "test/mse"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = TargetScaler()

    def build_head(self, embed_dim, hidden_dim: int = 32, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")

        return PooledRegressionHead(embed_dim, hidden_dim)

    def build_metrics(self, stage):
        if stage == "train":
            return None

        return nn.ModuleDict({
            "mse": MeanSquaredError(), "mae": MeanAbsoluteError(), "r2": R2Score(),
        })

    def extract_features(self, representation, tokens):
        # Mask padded positions, then let the head mean-pool over the rest.
        pad_mask = tokens.eq(self.pad_idx)
        representation = representation.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return representation, pad_mask

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return F.mse_loss(outputs, self.scaler.transform(labels).to(outputs.dtype))

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return

        _, labels = batch
        # Un-scale before updating metrics, so R2/MSE/MAE stay in the original target
        # scale rather than the standardised units the loss trains on.
        preds = self.scaler.inverse_transform(outputs.float())
        for metric in self.metrics[stage].values():
            metric.update(preds, labels.float())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {}

        return {f"{stage}/{name}": metric.compute()
                for name, metric in self.metrics[stage].items()}

    def postprocess_predictions(self, outputs, tokens, sequences):
        # Original target scale, not the standardised units the loss trains on.
        return self.scaler.inverse_transform(outputs.float())

    def on_fit_start_hook(self) -> None:
        targets = self._training_targets()
        if targets.numel() == 0:
            raise RuntimeError(
                f"The '{self.TASK_NAME}' training split is empty, so the target scaler "
                f"cannot be fitted and every prediction would come back on the wrong scale."
            )

        self.scaler.fit(targets)

    def _training_targets(self) -> torch.Tensor:
        # Read labels directly rather than through a DataLoader: the dataset's tokens are
        # unpadded (per-sequence length), so the default collate would fail to stack them,
        # and only the labels are needed here anyway.
        dataset = self.trainer.datamodule.train_dataset
        labels = [dataset[i][1] for i in range(len(dataset))]
        return torch.stack(labels).float().view(-1)

    @staticmethod
    def build_datamodule(cfg):
        from adaptrna_custom.tasks.regression_column.datamodule import CsvDataModule

        data = cfg["data"]
        return CsvDataModule(
            data_root=data["root"],
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 32),
            num_workers=data.get("num_workers", 0),
        )
