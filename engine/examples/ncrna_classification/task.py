"""Step 1 of "adding a new task": the task module.

ncRNA family classification, ported from the source repo's `train_ncrna_classification.py`
without touching a single core file. This is the acceptance test for the abstraction --
if adding a task ever needs an edit to `module.py`, `lora.py`, `adapter.py`, `hub.py` or the
CLI, the abstraction is wrong.

To use it, import this module once so `@register_task` fires:

    python -c "import examples.ncrna_classification.task" ...

or add `from . import ncrna_classification` to `rinalmo_hub/tasks/__init__.py` after moving
the file there.
"""

import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from rinalmo.data.alphabet import Alphabet
from rinalmo.model.downstream import ncRNAClassificationHead

from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


@register_task("ncrna_classification")
class ncRNAClassificationModule(BaseDownstreamModule):
    TASK_NAME = "ncrna_classification"

    # Question 1 from the walkthrough: does the task own state that predictions depend on
    # but that is not a head weight? No -- the class ids are fixed by the dataset.
    ADAPTER_EXTRA_PREFIXES = ()

    PRIMARY_METRIC = "test/f1"

    def build_head(self, embed_dim, head_embed_dim: int = 256, n_classes: int = 88, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")

        self.n_classes = n_classes

        return ncRNAClassificationHead(c_in=embed_dim, embed_dim=head_embed_dim, n_classes=n_classes)

    def build_metrics(self, stage):
        if stage == "train":
            return None

        return MetricCollection({
            "acc": MulticlassAccuracy(num_classes=self.n_classes, average="micro"),
            "f1": MulticlassF1Score(num_classes=self.n_classes, average="weighted"),
        })

    def batch_tokens(self, batch):
        # Batches are (family, tokens, class_id).
        return batch[1]

    def extract_features(self, representation, tokens):
        # Question 2 from the walkthrough: CLS token, like splice site.
        return representation[:, 0]

    def compute_loss(self, outputs, batch):
        *_, labels = batch
        return nn.functional.cross_entropy(outputs, labels)

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return

        *_, labels = batch
        self.metrics[stage].update(outputs.float(), labels)

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {}

        return {f"{stage}/{name}": value for name, value in self.metrics[stage].compute().items()}

    def postprocess_predictions(self, outputs, tokens, sequences):
        return torch.softmax(outputs.float(), dim=-1).argmax(dim=-1)

    @staticmethod
    def build_datamodule(cfg):
        from examples.ncrna_classification.datamodule import ncRNADataModule

        data = cfg["data"]

        return ncRNADataModule(
            data_root=data["root"],
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 1),
            num_workers=data.get("num_workers", 0),
            pin_memory=data.get("pin_memory", False),
            boundary_noise=data.get("boundary_noise", ""),
        )
