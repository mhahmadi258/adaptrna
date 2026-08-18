# rendered by adaptrna template v1 from spec.json
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


@register_task("regression_column")
class RegressionColumnModule(BaseDownstreamModule):
    """regression target from a flat sequence/label table"""

    TASK_NAME = "regression_column"
    ADAPTER_EXTRA_PREFIXES = ()
    PRIMARY_METRIC = "test/mse"

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
        return F.mse_loss(outputs, labels.to(outputs.dtype))

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return

        _, labels = batch
        for metric in self.metrics[stage].values():
            metric.update(outputs.float(), labels.float())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {}

        return {f"{stage}/{name}": metric.compute()
                for name, metric in self.metrics[stage].items()}

    def postprocess_predictions(self, outputs, tokens, sequences):
        return outputs.float()

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
