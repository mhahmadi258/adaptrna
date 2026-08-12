"""Step 3 of "adding a new task": a datamodule.

A standard `LightningDataModule` yielding `(family, tokens, class_id)` batches. Sequences are
variable length, so a collate function pads them to the batch maximum -- copy this pattern
whenever your dataset does not pad in `__getitem__`.
"""

from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rinalmo.data.alphabet import Alphabet


class ncRNADataset(Dataset):
    def __init__(self, csv_path: Union[Path, str], alphabet: Alphabet):
        self.data = pd.read_csv(csv_path)
        self.alphabet = alphabet

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = torch.tensor(self.alphabet.encode(row["sequence"]), dtype=torch.int64)

        return row["name"], tokens, int(row["class_id"])


class PadCollate:
    def __init__(self, pad_tkn_idx: int):
        self.pad_tkn_idx = pad_tkn_idx

    def __call__(self, batch):
        max_len = max(len(tokens) for _, tokens, _ in batch)
        padded = torch.full((len(batch), max_len), self.pad_tkn_idx, dtype=torch.int64)

        for i, (_, tokens, _) in enumerate(batch):
            padded[i, : len(tokens)] = tokens

        families = [family for family, _, _ in batch]
        class_ids = torch.tensor([class_id for _, _, class_id in batch], dtype=torch.int64)

        return families, padded, class_ids


class ncRNADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        boundary_noise: str = "",
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.boundary_noise = boundary_noise

    def setup(self, stage: Optional[str] = None):
        for split in ("train", "val", "test"):
            path = self.data_root / f"{split}{self.boundary_noise}.csv"
            setattr(self, f"{split}_dataset", ncRNADataset(path, self.alphabet))

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=PadCollate(self.alphabet.pad_idx),
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)
