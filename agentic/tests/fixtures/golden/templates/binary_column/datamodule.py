# rendered by adaptrna template v2 from spec.json
"""CSV/TSV datamodule for 'binary_column', rendered from the approved dataset spec.

Reads exactly the sequence and label columns approved at gate 1, from whatever path
`data.root` in the config names -- so pointing this task at a new file of the same shape
(same columns, same split policy) works without regenerating any code. Implements exactly
the approved split policy: no re-shuffling, no second seed. Sequences are padded per batch
(not to the dataset-wide maximum), so one long outlier does not inflate every batch.
"""

import re

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SEQUENCE_COLUMN = "sequence"
LABEL_COLUMN = "label"
SEPARATOR = ','
COMPRESSION = None
ON_INVALID = 'fail'
CLASSES = ['0', '1']
POSITIVE_CLASS = '0'
KEEP_COLUMNS = [SEQUENCE_COLUMN, LABEL_COLUMN, "source"]

_VALID_SEQUENCE = re.compile(r"^[ACGTUNacgtun]+$")


def _read_frame(path):
    frame = pd.read_csv(path, sep=SEPARATOR, compression=COMPRESSION)
    frame = frame[KEEP_COLUMNS].dropna()
    frame[SEQUENCE_COLUMN] = frame[SEQUENCE_COLUMN].astype(str)

    invalid = ~frame[SEQUENCE_COLUMN].str.match(_VALID_SEQUENCE)
    if invalid.any():
        if ON_INVALID == "drop":
            frame = frame[~invalid]
        else:
            bad = frame.loc[invalid, SEQUENCE_COLUMN].head(3).tolist()
            raise ValueError(
                f"{int(invalid.sum())} sequence(s) in '{path}' contain characters outside "
                f"ACGTUN, e.g. {bad}. The approved spec's on_invalid policy is 'fail'; "
                f"re-run profile_dataset and choose 'drop' at gate 1 to discard them instead."
            )

    return frame.reset_index(drop=True)


def _split(frame):
    column = "source"
    mapping = {'train': ['human', 'mouse'], 'val': ['fly'], 'test': ['zebrafish']}

    def rows_for(split_name):
        values = {str(v) for v in mapping.get(split_name, [])}
        return frame[frame[column].astype(str).isin(values)]

    return rows_for("train"), rows_for("val"), rows_for("test")


def _validate_split(train_frame, val_frame, test_frame):
    column = "source"
    mapping = {'train': ['human', 'mouse'], 'val': ['fly'], 'test': ['zebrafish']}
    counts = {"train": len(train_frame), "val": len(val_frame), "test": len(test_frame)}

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(
            f"the {', '.join(empty)} split is empty after applying the column split "
            f"mapping {mapping} on column '{column}' -- check that the mapping's values "
            f"actually appear in the data (a typo produces this silently otherwise)."
        )


class SequenceDataset(Dataset):
    """Tokens are left unpadded here; the datamodule's collate_fn pads per batch."""

    def __init__(self, frame, alphabet):
        self.frame = frame.reset_index(drop=True)
        self.alphabet = alphabet

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        tokens = torch.tensor(
            self.alphabet.encode(row[SEQUENCE_COLUMN]), dtype=torch.long
        )
        value = str(row[LABEL_COLUMN])
        if value not in CLASSES:
            raise ValueError(f"label value {value!r} is not one of the approved classes {CLASSES}")
        label = torch.tensor(1.0 if value == POSITIVE_CLASS else 0.0, dtype=torch.float32)
        return tokens, label


class CsvDataModule(pl.LightningDataModule):
    def __init__(self, data_root, alphabet, batch_size=32, num_workers=0, pin_memory=False):
        super().__init__()
        self.data_root = data_root
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        frame = _read_frame(self.data_root)
        train_frame, val_frame, test_frame = _split(frame)
        _validate_split(train_frame, val_frame, test_frame)

        self.train_dataset = SequenceDataset(train_frame, self.alphabet)
        self.val_dataset = SequenceDataset(val_frame, self.alphabet)
        self.test_dataset = SequenceDataset(test_frame, self.alphabet)

    def _collate(self, batch):
        """Pad every sequence in this batch to the batch's own max length, not the
        dataset's — a single outlier elsewhere in the file should not inflate every
        batch's memory footprint."""
        tokens, labels = zip(*batch)
        max_len = max(t.shape[0] for t in tokens)
        padded = torch.full((len(tokens), max_len), self.alphabet.pad_idx, dtype=torch.long)
        for i, seq_tokens in enumerate(tokens):
            padded[i, :seq_tokens.shape[0]] = seq_tokens
        return padded, torch.stack(labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self._collate)
