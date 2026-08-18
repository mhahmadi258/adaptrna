# rendered by adaptrna template v1 from spec.json
"""multiclass target from a flat sequence/label table"""

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task

CLASSES = ['0', '1', '2']
NUM_CLASSES = len(CLASSES)

@register_task("multiclass_random")
class MulticlassRandomModule(BaseDownstreamModule):
    """multiclass target from a flat sequence/label table"""

    TASK_NAME = "multiclass_random"
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

        return nn.ModuleDict({
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
        return outputs.float().argmax(dim=-1)

    @staticmethod
    def build_datamodule(cfg):
        from adaptrna_custom.tasks.multiclass_random.datamodule import CsvDataModule

        data = cfg["data"]
        return CsvDataModule(
            data_root=data["root"],
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 32),
            num_workers=data.get("num_workers", 0),
        )
