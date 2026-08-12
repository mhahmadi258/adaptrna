"""Deliberately broken generated tasks, one per failure mode the harness must catch.

Each is a *complete, plausible* task — the kind of thing a model would actually write —
with exactly one defect. If the harness passes any of these, it is not doing its job.
"""

GOOD_DATAMODULE = '''
"""A minimal CSV datamodule: sequence,label."""

from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CsvSequenceDataset(Dataset):
    def __init__(self, csv_path, alphabet, pad_to_len=-1):
        self.frame = pd.read_csv(csv_path)
        self.alphabet = alphabet
        self.pad_to_len = pad_to_len or -1

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        tokens = torch.tensor(
            self.alphabet.encode(str(row["sequence"]), pad_to_len=self.pad_to_len),
            dtype=torch.long,
        )
        return tokens, torch.tensor(float(row["label"]), dtype=torch.float32)


class CsvDataModule(pl.LightningDataModule):
    def __init__(self, data_root, alphabet, batch_size=2, num_workers=0, pin_memory=False):
        super().__init__()
        self.data_root = Path(data_root)
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        pad = max(len(s) for s in pd.read_csv(self.data_root / "train.csv")["sequence"]) + 2
        self.train_dataset = CsvSequenceDataset(self.data_root / "train.csv", self.alphabet, pad)
        self.val_dataset = CsvSequenceDataset(self.data_root / "val.csv", self.alphabet, pad)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
'''

_TASK_TEMPLATE = '''
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


@register_task("{task_name}")
class GeneratedModule(BaseDownstreamModule):
    TASK_NAME = "{task_name}"
    ADAPTER_EXTRA_PREFIXES = {extra_prefixes}
    PRIMARY_METRIC = "test/acc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
{extra_state}

    def build_head(self, embed_dim, {head_signature}):
{head_body}

    def build_metrics(self, stage):
        return None if stage == "train" else BinaryAccuracy()

    def extract_features(self, representation, tokens):
{extract_body}

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return nn.functional.binary_cross_entropy_with_logits(
            outputs, labels.unsqueeze(1).to(outputs.dtype)
        )

    def update_metrics(self, outputs, batch, stage):
        if stage in self.metrics:
            _, labels = batch
            self.metrics[stage].update(torch.sigmoid(outputs.float()), labels.unsqueeze(1).int())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {{}}
        return {{f"{{stage}}/acc": self.metrics[stage].compute()}}

    def postprocess_predictions(self, outputs, tokens, sequences):
        return torch.sigmoid(outputs.float()).squeeze(-1){postprocess_extra}

    @staticmethod
    def build_datamodule(cfg):
        from adaptrna_custom.tasks.{task_name}.datamodule import CsvDataModule

        data = cfg["data"]
        return CsvDataModule(
            data_root=data["root"],
            alphabet=Alphabet(),
            batch_size=data.get("batch_size", 2),
            num_workers=data.get("num_workers", 0),
        )
'''

_GOOD_SIGNATURE = "hidden_dim: int = 32, **kwargs"

_GOOD_HEAD = """        if kwargs:
            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")

        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )"""

_GOOD_EXTRACT = "        return representation[:, 0]"

CONFIG_TEMPLATE = """task: {task_name}
lm_config: nano

head:
  hidden_dim: 32

data:
  root: {data_root}
  batch_size: 2
  num_workers: 0

lora:
  r: 4
  alpha: 8
  dropout: 0.0
  layer_stride: 3

optim:
  name: adamw
  lr: 3.0e-4

trainer:
  max_epochs: 1
  precision: bf16-mixed
  gradient_clip_val: 1.0
"""


def good_task(task_name: str) -> str:
    """A correct task — the control."""
    return _TASK_TEMPLATE.format(
        task_name=task_name, extra_prefixes="()", extra_state="        pass",
        head_signature=_GOOD_SIGNATURE, head_body=_GOOD_HEAD,
        extract_body=_GOOD_EXTRACT, postprocess_extra="",
    )


def task_with_unsaved_state(task_name: str) -> str:
    """Owns a calibration buffer that predictions use — but never declares it in
    ADAPTER_EXTRA_PREFIXES, so it silently reverts to its default after a round trip.

    This is the project's number-one silent failure, in the exact shape a model writes it.
    """
    return _TASK_TEMPLATE.format(
        task_name=task_name,
        extra_prefixes="()",                                   # <- the defect
        extra_state='        self.register_buffer("calibration", torch.tensor([1.0]))',
        head_signature=_GOOD_SIGNATURE, head_body=_GOOD_HEAD,
        extract_body=_GOOD_EXTRACT,
        postprocess_extra=" * self.calibration",               # predictions depend on it
    )


def task_with_saved_state(task_name: str) -> str:
    """The same task, declared correctly — the positive control."""
    return _TASK_TEMPLATE.format(
        task_name=task_name,
        extra_prefixes='("calibration",)',
        extra_state='        self.register_buffer("calibration", torch.tensor([1.0]))',
        head_signature=_GOOD_SIGNATURE, head_body=_GOOD_HEAD,
        extract_body=_GOOD_EXTRACT,
        postprocess_extra=" * self.calibration",
    )


def task_with_head_ignoring_config(task_name: str) -> str:
    """build_head does not accept the config's `hidden_dim`."""
    return _TASK_TEMPLATE.format(
        task_name=task_name, extra_prefixes="()", extra_state="        pass",
        head_signature="**kwargs",                              # <- does not accept hidden_dim
        head_body='        if kwargs:\n            raise TypeError(f"Unexpected head config keys: {sorted(kwargs)}")\n\n        return nn.Linear(embed_dim, 1)',
        extract_body=_GOOD_EXTRACT, postprocess_extra="",
    )


def task_with_bad_extract_features(task_name: str) -> str:
    """Forgets to pool: hands the head a per-token tensor, so the loss shape is wrong."""
    return _TASK_TEMPLATE.format(
        task_name=task_name, extra_prefixes="()", extra_state="        pass",
        head_signature=_GOOD_SIGNATURE, head_body=_GOOD_HEAD,
        extract_body="        return representation",           # <- no CLS slice
        postprocess_extra="",
    )


BAD_DATAMODULE = GOOD_DATAMODULE.replace('row["sequence"]', 'row["seq"]')
