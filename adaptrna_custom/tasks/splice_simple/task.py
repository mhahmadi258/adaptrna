"""Binary donor splice-site classification from flat CSV files.

The data is three CSVs (train/val/test) with a `sequence` column of fixed-length
400 nt DNA and a 0/1 `label` column. Head/loss/metrics follow the shipped
`splice_site` task shape: a linear classifier on the CLS token, BCE-with-logits,
and acc / precision / recall / specificity / f1.

Import this module once so `@register_task` fires:

    python -c "import adaptrna_custom.tasks.splice_simple.task" ...
"""

import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from rinalmo.data.alphabet import Alphabet

from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


class SpliceSimpleHead(nn.Module):
    """Linear classifier on the CLS embedding -> one logit per sequence."""

    def __init__(self, c_in: int, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(c_in, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        # x: (B, c_in) -> (B,)
        return self.mlp(x).squeeze(-1)


@register_task("splice_simple")
class SpliceSimpleModule(BaseDownstreamModule):
    TASK_NAME = "splice_simple"

    # Question 1: the task owns a decision threshold that predictions depend on and
    # that is NOT a head weight. It is a plain Python float, so it travels in the
    # adapter via adapter_extra_payload()/load_adapter_extra(). No extra tensors or
    # buffers exist, so ADAPTER_EXTRA_PREFIXES stays empty.
    ADAPTER_EXTRA_PREFIXES = ()

    PRIMARY_METRIC = "test/f1_score"

    def build_head(
        self,
        embed_dim,
        head_embed_dim: int = 128,
        dropout: float = 0.1,
        threshold: float = 0.5,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")

        # Plain Python state used by metrics and by postprocess_predictions.
        self.threshold = float(threshold)

        return SpliceSimpleHead(c_in=embed_dim, embed_dim=head_embed_dim, dropout=dropout)

    def build_metrics(self, stage):
        if stage == "train":
            return None

        threshold = float(getattr(self, "threshold", 0.5))

        return MetricCollection({
            "acc": BinaryAccuracy(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "specificity": BinarySpecificity(threshold=threshold),
            "f1_score": BinaryF1Score(threshold=threshold),
        })

    def batch_tokens(self, batch):
        # Batches are (tokens, labels).
        return batch[0]

    def extract_features(self, representation, tokens):
        # Question 2: the head is a per-sequence classifier, so it consumes the CLS
        # token only -- EOS and padded positions never reach it.
        return representation[:, 0]

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return nn.functional.binary_cross_entropy_with_logits(
            outputs.float(), labels.float()
        )

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics or self.metrics[stage] is None:
            return

        _, labels = batch
        probs = torch.sigmoid(outputs.float())
        self.metrics[stage].update(probs, labels.int())

    def compute_metrics(self, stage):
        if stage not in self.metrics or self.metrics[stage] is None:
            return {}

        return {
            f"{stage}/{name}": value
            for name, value in self.metrics[stage].compute().items()
        }

    def postprocess_predictions(self, outputs, tokens, sequences):
        probs = torch.sigmoid(outputs.float())
        threshold = float(getattr(self, "threshold", 0.5))

        return [
            {"probability": float(p), "label": int(p >= threshold)}
            for p in probs.detach().cpu().tolist()
        ]

    # --- non-tensor state that predictions depend on -------------------------

    def adapter_extra_payload(self):
        return {"threshold": float(getattr(self, "threshold", 0.5))}

    def load_adapter_extra(self, extra):
        if not extra:
            return

        self.threshold = float(extra.get("threshold", getattr(self, "threshold", 0.5)))

        for stage, collection in self.metrics.items():
            if collection is None:
                continue
            for metric in collection.values():
                if hasattr(metric, "threshold"):
                    metric.threshold = self.threshold

    @staticmethod
    def build_datamodule(cfg):
        from adaptrna_custom.tasks.splice_simple.datamodule import SpliceSimpleDataModule

        data = cfg["data"]

        return SpliceSimpleDataModule(
            data_root=data["root"],
            train_file=data.get("train_file", "splice_simple_train.csv"),
            val_file=data.get("val_file", "splice_simple_val.csv"),
            test_file=data.get("test_file", "splice_simple_test.csv"),
            sequence_column=data.get("sequence_column", "sequence"),
            label_column=data.get("label_column", "label"),
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 32),
            num_workers=data.get("num_workers", 0),
            pin_memory=data.get("pin_memory", False),
        )
