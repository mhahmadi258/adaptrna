# rendered by adaptrna template v1 from spec.json
"""binary target from a flat sequence/label table"""

import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall,
)

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


@register_task("binary_column")
class BinaryColumnModule(BaseDownstreamModule):
    """binary target from a flat sequence/label table"""

    TASK_NAME = "binary_column"
    ADAPTER_EXTRA_PREFIXES = ()
    PRIMARY_METRIC = "test/f1_score"

    def build_head(self, embed_dim, hidden_dim: int = 32, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")

        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def build_metrics(self, stage):
        if stage == "train":
            return None

        return nn.ModuleDict({
            "acc": BinaryAccuracy(), "precision": BinaryPrecision(),
            "recall": BinaryRecall(), "f1_score": BinaryF1Score(),
        })

    def extract_features(self, representation, tokens):
        # CLS only; EOS and padding never reach the head.
        return representation[:, 0]

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return nn.functional.binary_cross_entropy_with_logits(
            outputs, labels.unsqueeze(1).to(outputs.dtype)
        )

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return

        _, labels = batch
        probs = torch.sigmoid(outputs.float())
        for metric in self.metrics[stage].values():
            metric.update(probs, labels.unsqueeze(1).int())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {}

        return {f"{stage}/{name}": metric.compute()
                for name, metric in self.metrics[stage].items()}

    def postprocess_predictions(self, outputs, tokens, sequences):
        return torch.sigmoid(outputs.float()).squeeze(-1)

    @staticmethod
    def build_datamodule(cfg):
        from adaptrna_custom.tasks.binary_column.datamodule import CsvDataModule

        data = cfg["data"]
        return CsvDataModule(
            data_root=data["root"],
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 32),
            num_workers=data.get("num_workers", 0),
        )
