"""Splice site prediction (donor / acceptor), Spliceator data.

Binary classification from the CLS token. Metrics are accumulated as a confusion matrix and
reduced exactly the way the source repo did, so numbers stay comparable with the published
F1 of 95.82 for the donor LoRA arm.
"""

import torch
import torch.nn as nn
from torchmetrics.classification import BinaryConfusionMatrix

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.splice_site_prediction.datamodule import SpliceSiteDataModule
from rinalmo.model.downstream import SpliceSitePredictionHead
import rinalmo.utils.splice_site_metrics as ss_metrics

from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


@register_task("splice_site")
class SpliceSiteModule(BaseDownstreamModule):
    TASK_NAME = "splice_site"
    ADAPTER_EXTRA_PREFIXES = ()
    PRIMARY_METRIC = "test/f1_score"

    def build_head(self, embed_dim, head_embed_dim: int = 128, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys for splice_site: {sorted(kwargs)}")

        return SpliceSitePredictionHead(c_in=embed_dim, embed_dim=head_embed_dim)

    def build_metrics(self, stage):
        # Train batches are not scored: the source repo logged loss only during training,
        # and a confusion matrix per training step buys nothing for a comparison run.
        if stage == "train":
            return None

        return BinaryConfusionMatrix()

    def extract_features(self, representation, tokens):
        # CLS token.
        return representation[:, 0]

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return nn.functional.binary_cross_entropy_with_logits(outputs, labels.unsqueeze(1).to(outputs.dtype))

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return

        _, labels = batch
        self.metrics[stage].update(torch.sigmoid(outputs.float()), labels.unsqueeze(1).int())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {}

        matrix = self.metrics[stage].compute()
        tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        total = tp + tn + fp + fn

        if total == 0:
            return {}

        # `splice_site_metrics` returns percentages rounded to two decimals, matching the
        # numbers reported in the source project.
        acc = ss_metrics.accuracy(tp, tn, total)
        precision = ss_metrics.precision(tp, fp) if (tp > 0 or fp > 0) else torch.tensor(0.0)
        recall = ss_metrics.recall(tp, fn) if (tp > 0 or fn > 0) else torch.tensor(0.0)
        specificity = ss_metrics.specificity(tn, fp) if (tn > 0 or fp > 0) else torch.tensor(0.0)
        f1 = ss_metrics.f1_score(precision, recall) if (precision > 0 or recall > 0) else torch.tensor(0.0)

        return {
            f"{stage}/acc": acc.float(),
            f"{stage}/precision": precision.float(),
            f"{stage}/recall": recall.float(),
            f"{stage}/specificity": specificity.float(),
            f"{stage}/f1_score": f1.float(),
        }

    def postprocess_predictions(self, outputs, tokens, sequences):
        """Probability that each sequence contains a splice site of the trained type."""
        return torch.sigmoid(outputs.float()).squeeze(-1)

    @staticmethod
    def build_datamodule(cfg):
        data = cfg["data"]

        return SpliceSiteDataModule(
            ss_type=data.get("ss_type", "donor"),
            species=data.get("species", "Danio"),
            dataset_id=data.get("dataset_id", "db_1"),
            data_root=data.get("root"),
            test_data_root=data.get("test_root"),
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 1),
            num_workers=data.get("num_workers", 0),
            pin_memory=data.get("pin_memory", False),
            skip_data_preparation=not data.get("prepare", False),
        )
