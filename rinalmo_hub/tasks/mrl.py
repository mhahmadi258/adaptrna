"""Mean ribosome load (MRL) regression on the synthetic human 5'UTR library.

Two things here are worth knowing before changing anything:

1. `ADAPTER_EXTRA_PREFIXES = ("scaler.",)`. The target scaler registers `_mean` and `_std`
   as buffers, and predictions are un-scaled through them. Drop them from the adapter file
   and it still loads without error, then "un-scales" with mean 0 / std 1 and returns
   plausible-looking numbers on the wrong scale.
2. The scaler is fitted in `on_fit_start`, from the training targets. The source fitted it
   inside `training_step` during epoch 0 and returned `None`, which made epoch 0 a
   non-training epoch (shifting every fine-tuning schedule index by one) and made the task
   impossible to run under DDP -- Lightning raises "Skipping the training_step by returning
   None in distributed training is not supported".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from rinalmo.data.alphabet import Alphabet
from rinalmo.model.downstream import RibosomeLoadingPredictionHead
from rinalmo.utils.scaler import StandardScaler

from rinalmo_hub.data.mrl import MRLDataModule
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


@register_task("mrl")
class MRLModule(BaseDownstreamModule):
    TASK_NAME = "mrl"

    # The fitted target mean/std live in `scaler._mean` / `scaler._std` as buffers.
    ADAPTER_EXTRA_PREFIXES = ("scaler.",)

    PRIMARY_METRIC = "test/r2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = StandardScaler()

    def build_head(self, embed_dim, head_embed_dim: int = 32, num_blocks: int = 6,
                   dropout: float = 0.2, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys for mrl: {sorted(kwargs)}")

        return RibosomeLoadingPredictionHead(
            c_in=embed_dim, embed_dim=head_embed_dim, num_blocks=num_blocks, dropout=dropout
        )

    def build_metrics(self, stage):
        if stage == "train":
            return None

        return MetricCollection({"r2": R2Score(), "mse": MeanSquaredError(), "mae": MeanAbsoluteError()})

    def extract_features(self, representation, tokens):
        # Zero the padding token representations, then hand the head both the features and
        # the mask so it can mask again before pooling.
        pad_mask = tokens.eq(self.pad_idx)
        representation = representation.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return representation, pad_mask

    def compute_loss(self, outputs, batch):
        _, targets = batch
        return F.mse_loss(outputs, self.scaler.transform(targets).to(outputs.dtype))

    def _unscaled(self, outputs):
        return self.scaler.inverse_transform(outputs.float()).clamp(min=0.0)

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return

        _, targets = batch
        preds = self._unscaled(outputs)
        self.metrics[stage].update(preds, targets.float())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {}

        return {f"{stage}/{name}": value for name, value in self.metrics[stage].compute().items()}

    def postprocess_predictions(self, outputs, tokens, sequences):
        """Mean ribosome load, on the original scale."""
        return self._unscaled(outputs)

    # ------------------------------------------------------------------ target scaler

    def on_fit_start_hook(self):
        targets = self._training_targets()
        if targets.numel() == 0:
            raise RuntimeError(
                "The MRL training split is empty, so the target scaler cannot be fitted and "
                "every prediction would come back as NaN. `train_eval_split` reserves the "
                "top 100 sequences per length for Random7600, so a dataset with fewer than "
                "that per length leaves nothing to train on."
            )

        # Refit in place: replacing the module would move the buffers off the training device.
        self.scaler._seen_samples = []
        self.scaler.partial_fit(targets)

        print(
            f"Fitted MRL target scaler on {len(targets):,} training targets: "
            f"mean={self.scaler._mean.item():.4f} std={self.scaler._std.item():.4f}"
        )

    def _training_targets(self) -> torch.Tensor:
        """
        Ribosome-load targets of the training split, without a forward pass.

        Reads them straight off the underlying dataframe when possible; the dataloader walk
        is the fallback for any datamodule that does not expose one.
        """
        train_dataset = self.trainer.datamodule.train_dataset

        base = getattr(train_dataset, "dataset", None)
        indices = getattr(train_dataset, "indices", None)
        if base is not None and indices is not None and hasattr(base, "df") and "rl" in base.df.columns:
            values = base.df.iloc[list(indices)]["rl"].to_numpy()
            return torch.tensor(values, dtype=torch.float32)

        loader = DataLoader(train_dataset, batch_size=1024, num_workers=0)
        return torch.cat([targets.float().view(-1) for _, targets in loader])

    @staticmethod
    def build_datamodule(cfg):
        data = cfg["data"]

        return MRLDataModule(
            data_root=data["root"],
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 1),
            num_workers=data.get("num_workers", 0),
            pin_memory=data.get("pin_memory", False),
            skip_data_preparation=not data.get("prepare", False),
            val_split=data.get("val_split", "random7600"),
            holdout_fraction=data.get("holdout_fraction", 0.05),
            holdout_seed=data.get("holdout_seed", 0),
        )
