# rendered by adaptrna template v3 from spec.json
"""binary target from a flat sequence/label table"""

import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall,
)

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task

CLASSES = ['0', '1']
POSITIVE_CLASS = '0'

@register_task("binary_column")
class BinaryColumnModule(BaseDownstreamModule):
    """binary target from a flat sequence/label table

    Positive class: '0'. `postprocess_predictions` returns the
    probability of this class.
    """

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

        return MetricCollection({
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
        # Probability of the positive class, '0'.
        return torch.sigmoid(outputs.float()).squeeze(-1)

    def adapter_extra_payload(self) -> dict:
        # The class order (and, for binary, which class is positive) decides what a
        # prediction MEANS. Ship it in the adapter file so a reload can check itself
        # against it rather than silently relabelling.
        return {"classes": CLASSES, "positive_class": POSITIVE_CLASS}

    def load_adapter_extra(self, extra: dict) -> None:
        if not extra:
            return  # tolerate an older adapter that predates this payload

        if extra.get("classes") != CLASSES:
            raise ValueError(
                f"This adapter was saved with classes {extra.get('classes')}, but the "
                f"loaded module defines CLASSES = {CLASSES}. Loading it would silently "
                f"relabel every prediction."
            )
        if extra.get("positive_class") != POSITIVE_CLASS:
            raise ValueError(
                f"This adapter was saved with positive_class = "
                f"{extra.get('positive_class')!r}, but the loaded module defines "
                f"POSITIVE_CLASS = {POSITIVE_CLASS!r}. Loading it would silently invert "
                f"every prediction."
            )

    @staticmethod
    def build_datamodule(cfg):
        from adaptrna_custom.tasks.binary_column.datamodule import CsvDataModule

        data = cfg["data"]
        return CsvDataModule(
            data_root=data["root"],
            val_root=data.get("val_root"),
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 32),
            num_workers=data.get("num_workers", 0),
        )
