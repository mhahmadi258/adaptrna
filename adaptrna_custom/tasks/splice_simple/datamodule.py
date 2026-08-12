"""Datamodule for `splice_simple`.

Reads three flat CSVs (train / val / test) with the user's real columns:

    sequence,label
    ACGT...(400 nt),1

DNA is uppercased and T -> U before tokenisation so the RNA alphabet sees
canonical symbols; anything else becomes the alphabet's unknown token via
`Alphabet.encode`. All sequences are 400 nt, but the collate function still pads
to the batch maximum so mixed-length CSVs work unchanged.
"""

from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rinalmo.data.alphabet import Alphabet

_DNA_TO_RNA = str.maketrans({"T": "U", "t": "U"})


class SpliceSimpleDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[Path, str],
        alphabet: Alphabet,
        sequence_column: str = "sequence",
        label_column: str = "label",
    ):
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"splice_simple: missing CSV file {csv_path}")

        self.data = pd.read_csv(csv_path)

        for column in (sequence_column, label_column):
            if column not in self.data.columns:
                raise KeyError(
                    f"splice_simple: column '{column}' not in {csv_path} "
                    f"(found {list(self.data.columns)})"
                )

        self.data = self.data.dropna(subset=[sequence_column, label_column]).reset_index(drop=True)

        self.alphabet = alphabet
        self.sequence_column = sequence_column
        self.label_column = label_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = str(row[self.sequence_column]).strip().upper().translate(_DNA_TO_RNA)

        tokens = torch.tensor(self.alphabet.encode(seq), dtype=torch.int64)
        label = torch.tensor(int(float(row[self.label_column])), dtype=torch.int64)

        return tokens, label


class PadCollate:
    def __init__(self, pad_tkn_idx: int):
        self.pad_tkn_idx = pad_tkn_idx

    def __call__(self, batch):
        max_len = max(len(tokens) for tokens, _ in batch)
        padded = torch.full((len(batch), max_len), self.pad_tkn_idx, dtype=torch.int64)

        for i, (tokens, _) in enumerate(batch):
            padded[i, : len(tokens)] = tokens

        labels = torch.stack([label for _, label in batch])

        return padded, labels


class SpliceSimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        train_file: str = "splice_simple_train.csv",
        val_file: str = "splice_simple_val.csv",
        test_file: str = "splice_simple_test.csv",
        sequence_column: str = "sequence",
        label_column: str = "label",
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.files = {"train": train_file, "val": val_file, "test": test_file}
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        for split, filename in self.files.items():
            if getattr(self, f"{split}_dataset") is not None:
                continue

            dataset = SpliceSimpleDataset(
                self.data_root / filename,
                alphabet=self.alphabet,
                sequence_column=self.sequence_column,
                label_column=self.label_column,
            )
            setattr(self, f"{split}_dataset", dataset)

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

    def predict_dataloader(self):
        return self._loader(self.test_dataset)
