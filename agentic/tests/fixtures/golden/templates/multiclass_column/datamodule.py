# rendered by adaptrna template v1 from spec.json
"""CSV/TSV datamodule for 'multiclass_column', rendered from the approved dataset spec.

Reads exactly the sequence and label columns approved at gate 1, from whatever path
`data.root` in the config names -- so pointing this task at a new file of the same shape
(same columns, same split policy) works without regenerating any code. Implements exactly
the approved split policy: no re-shuffling, no second seed.
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
CLASSES = ['0', '1', '2']
CLASS_INDEX = {value: index for index, value in enumerate(CLASSES)}
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


class SequenceDataset(Dataset):
    def __init__(self, frame, alphabet, pad_to_len):
        self.frame = frame.reset_index(drop=True)
        self.alphabet = alphabet
        self.pad_to_len = pad_to_len

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        tokens = torch.tensor(
            self.alphabet.encode(row[SEQUENCE_COLUMN], pad_to_len=self.pad_to_len),
            dtype=torch.long,
        )
        label = torch.tensor(CLASS_INDEX[str(row[LABEL_COLUMN])], dtype=torch.long)
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

        pad_to_len = max(len(s) for s in frame[SEQUENCE_COLUMN]) + 2
        self.train_dataset = SequenceDataset(train_frame, self.alphabet, pad_to_len)
        self.val_dataset = SequenceDataset(val_frame, self.alphabet, pad_to_len)
        self.test_dataset = SequenceDataset(test_frame, self.alphabet, pad_to_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)
