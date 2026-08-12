"""MRL datamodule with a selectable validation split.

The vendored `RibosomeLoadingDataModule` uses `Random7600` as the validation set *and*
reports it as a result. Nothing currently selects on it (the final epoch is tested, there is
no `ModelCheckpoint(monitor=...)`), so the published numbers are clean -- but adding early
stopping or best-checkpoint selection on `val/r2` would silently contaminate a headline
metric. `val_split: holdout` carves a validation set out of the training sequences instead,
leaving `Random7600` untouched as a pure test set.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from rinalmo.data.downstream.ribosome_loading.datamodule import (
    VARYING_LEN_25_TO_100_CSV,
    RibosomeLoadingDataModule,
)
from rinalmo.data.downstream.ribosome_loading.dataset import RibosomeLoadingDataset

VAL_SPLITS = ("random7600", "holdout")


class MRLDataModule(RibosomeLoadingDataModule):
    def __init__(self, *args, val_split: str = "random7600", holdout_fraction: float = 0.05,
                 holdout_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)

        if val_split not in VAL_SPLITS:
            raise ValueError(f"data.val_split must be one of {VAL_SPLITS}, got '{val_split}'")

        self.val_split = val_split
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed

    def setup(self, stage: Optional[str] = None):
        dataset = RibosomeLoadingDataset(self.data_root / VARYING_LEN_25_TO_100_CSV, alphabet=self.alphabet)
        train_dataset, random7600, human7600 = dataset.train_eval_split(num_eval_samples_per_len=100)

        self.random7600_dataset = random7600
        self.human7600_dataset = human7600

        if self.val_split == "random7600":
            self.train_dataset = train_dataset
            self.val_dataset = random7600
        else:
            generator = torch.Generator().manual_seed(self.holdout_seed)
            order = torch.randperm(len(train_dataset), generator=generator).tolist()
            n_val = max(1, int(round(self.holdout_fraction * len(train_dataset))))

            indices = list(train_dataset.indices)
            self.val_dataset = Subset(dataset, [indices[i] for i in order[:n_val]])
            self.train_dataset = Subset(dataset, [indices[i] for i in order[n_val:]])

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
