# rendered by adaptrna template v3 from spec.json
"""multiclass target from a flat sequence/label table"""

import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task

CLASSES = ['0', '1', '2']
NUM_CLASSES = len(CLASSES)

@register_task("multiclass_column")
class MulticlassColumnModule(BaseDownstreamModule):
    """multiclass target from a flat sequence/label table"""

    TASK_NAME = "multiclass_column"
    ADAPTER_EXTRA_PREFIXES = ()
    PRIMARY_METRIC = "test/macro_f1"

    def build_head(self, embed_dim, hidden_dim: int = 32, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")

        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, NUM_CLASSES)
        )

    def build_metrics(self, stage):
        if stage == "train":
            return None

        return MetricCollection({
            "acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
            "macro_f1": MulticlassF1Score(num_classes=NUM_CLASSES, average="macro"),
        })

    def extract_features(self, representation, tokens):
        # CLS only; EOS and padding never reach the head.
        return representation[:, 0]

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return nn.functional.cross_entropy(outputs, labels)

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return

        _, labels = batch
        for metric in self.metrics[stage].values():
            metric.update(outputs.float(), labels)

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {}

        return {f"{stage}/{name}": metric.compute()
                for name, metric in self.metrics[stage].items()}

    def postprocess_predictions(self, outputs, tokens, sequences):
        probs = torch.softmax(outputs.float(), dim=-1)
        predictions = []
        for row in probs.tolist():
            index = max(range(len(row)), key=row.__getitem__)
            predictions.append({
                "label": CLASSES[index],
                "probabilities": dict(zip(CLASSES, row)),
            })
        return predictions

    def adapter_extra_payload(self) -> dict:
        # The class order (and, for binary, which class is positive) decides what a
        # prediction MEANS. Ship it in the adapter file so a reload can check itself
        # against it rather than silently relabelling.
        return {"classes": CLASSES}

    def load_adapter_extra(self, extra: dict) -> None:
        if not extra:
            return  # tolerate an older adapter that predates this payload

        if extra.get("classes") != CLASSES:
            raise ValueError(
                f"This adapter was saved with classes {extra.get('classes')}, but the "
                f"loaded module defines CLASSES = {CLASSES}. Loading it would silently "
                f"relabel every prediction."
            )

    @staticmethod
    def build_datamodule(cfg):
        from adaptrna_custom.tasks.multiclass_column.datamodule import CsvDataModule

        data = cfg["data"]
        return CsvDataModule(
            data_root=data["root"],
            val_root=data.get("val_root"),
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 32),
            num_workers=data.get("num_workers", 0),
        )
