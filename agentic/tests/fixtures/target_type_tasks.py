"""Hand-written, known-good tasks — one per supported target type.

These replace the shipped engine tasks as `test_harness.py`'s PASS controls (Phase 13, D1):
the agentic layer must not name a shipped task, so the harness needs a control it can own.
Each function below returns the complete source of a `BaseDownstreamModule` subclass, in the
same shape a human or the codegen pipeline would write for that target type. They carry no
RNA-domain task identity — just the three head shapes this build supports.

Shared with the codegen regression tests (Phase 13 §13): the fixture CSVs under
`tests/fixtures/data/` are the canonical tiny datasets used wherever a real file is needed.
"""

from pathlib import Path

FIXTURE_DATA_DIR = Path(__file__).parent / "data"

BINARY_CSV = FIXTURE_DATA_DIR / "binary.csv"
MULTICLASS_CSV = FIXTURE_DATA_DIR / "multiclass.csv"
REGRESSION_CSV = FIXTURE_DATA_DIR / "regression.csv"

CONFIG_TEMPLATE = """task: {task_name}
lm_config: nano

head:
{head_config}

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


def _config(task_name: str, data_root, head_config: str) -> str:
    return CONFIG_TEMPLATE.format(task_name=task_name, data_root=data_root, head_config=head_config)


# ---------------------------------------------------------------- binary

BINARY_DATAMODULE = '''
"""A minimal CSV datamodule: sequence,label (0/1)."""

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

_BINARY_TASK = '''
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


@register_task("{task_name}")
class GeneratedModule(BaseDownstreamModule):
    TASK_NAME = "{task_name}"
    ADAPTER_EXTRA_PREFIXES = ()
    PRIMARY_METRIC = "test/f1_score"

    def build_head(self, embed_dim, hidden_dim: int = 32, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {{sorted(kwargs)}}")

        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def build_metrics(self, stage):
        if stage == "train":
            return None
        return nn.ModuleDict({{"acc": BinaryAccuracy(), "f1_score": BinaryF1Score()}})

    def extract_features(self, representation, tokens):
        # CLS only; EOS and padding never reach the head.
        return representation[:, 0]

    def compute_loss(self, outputs, batch):
        _, labels = batch
        return nn.functional.binary_cross_entropy_with_logits(
            outputs, labels.unsqueeze(1).to(outputs.dtype)
        )

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return
        _, labels = batch
        probs = torch.sigmoid(outputs.float())
        for metric in self.metrics[stage].values():
            metric.update(probs, labels.unsqueeze(1).int())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {{}}
        return {{f"{{stage}}/{{name}}": metric.compute()
                for name, metric in self.metrics[stage].items()}}

    def postprocess_predictions(self, outputs, tokens, sequences):
        return torch.sigmoid(outputs.float()).squeeze(-1)

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


def binary_task(task_name: str) -> str:
    return _BINARY_TASK.format(task_name=task_name)


def binary_config(task_name: str, data_root) -> str:
    return _config(task_name, data_root, "  hidden_dim: 32")


# ---------------------------------------------------------------- multiclass

MULTICLASS_DATAMODULE = '''
"""A minimal CSV datamodule: sequence,label (integer class index)."""

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
        return tokens, torch.tensor(int(row["label"]), dtype=torch.long)


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

_MULTICLASS_TASK = '''
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task

NUM_CLASSES = {num_classes}


@register_task("{task_name}")
class GeneratedModule(BaseDownstreamModule):
    TASK_NAME = "{task_name}"
    ADAPTER_EXTRA_PREFIXES = ()
    PRIMARY_METRIC = "test/macro_f1"

    def build_head(self, embed_dim, hidden_dim: int = 32, num_classes: int = NUM_CLASSES, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {{sorted(kwargs)}}")

        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes)
        )

    def build_metrics(self, stage):
        if stage == "train":
            return None
        return nn.ModuleDict({{
            "acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
            "macro_f1": MulticlassF1Score(num_classes=NUM_CLASSES, average="macro"),
        }})

    def extract_features(self, representation, tokens):
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
            return {{}}
        return {{f"{{stage}}/{{name}}": metric.compute()
                for name, metric in self.metrics[stage].items()}}

    def postprocess_predictions(self, outputs, tokens, sequences):
        return outputs.float().argmax(dim=-1)

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


def multiclass_task(task_name: str, num_classes: int = 3) -> str:
    return _MULTICLASS_TASK.format(task_name=task_name, num_classes=num_classes)


def multiclass_config(task_name: str, data_root, num_classes: int = 3) -> str:
    return _config(task_name, data_root, f"  hidden_dim: 32\n  num_classes: {num_classes}")


# ---------------------------------------------------------------- regression

REGRESSION_DATAMODULE = '''
"""A minimal CSV datamodule: sequence,label (continuous target)."""

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

_REGRESSION_TASK = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from rinalmo.data.alphabet import Alphabet
from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task


class PooledRegressionHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, representation, pad_mask):
        keep = (~pad_mask).unsqueeze(-1).to(representation.dtype)
        pooled = (representation * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.mlp(pooled).squeeze(-1)


@register_task("{task_name}")
class GeneratedModule(BaseDownstreamModule):
    TASK_NAME = "{task_name}"
    ADAPTER_EXTRA_PREFIXES = ()
    PRIMARY_METRIC = "test/mse"

    def build_head(self, embed_dim, hidden_dim: int = 32, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys: {{sorted(kwargs)}}")

        return PooledRegressionHead(embed_dim, hidden_dim)

    def build_metrics(self, stage):
        if stage == "train":
            return None
        return nn.ModuleDict({{"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}})

    def extract_features(self, representation, tokens):
        # Mask padded positions, then let the head mean-pool over the rest.
        pad_mask = tokens.eq(self.pad_idx)
        representation = representation.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return representation, pad_mask

    def compute_loss(self, outputs, batch):
        _, targets = batch
        return F.mse_loss(outputs, targets.to(outputs.dtype))

    def update_metrics(self, outputs, batch, stage):
        if stage not in self.metrics:
            return
        _, targets = batch
        for metric in self.metrics[stage].values():
            metric.update(outputs.float(), targets.float())

    def compute_metrics(self, stage):
        if stage not in self.metrics:
            return {{}}
        return {{f"{{stage}}/{{name}}": metric.compute()
                for name, metric in self.metrics[stage].items()}}

    def postprocess_predictions(self, outputs, tokens, sequences):
        return outputs.float()

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


def regression_task(task_name: str) -> str:
    return _REGRESSION_TASK.format(task_name=task_name)


def regression_config(task_name: str, data_root) -> str:
    return _config(task_name, data_root, "  hidden_dim: 32")


# ---------------------------------------------------------------- registry

TARGET_TYPES = {
    "binary": {
        "csv": BINARY_CSV,
        "task": binary_task,
        "datamodule": BINARY_DATAMODULE,
        "config": binary_config,
    },
    "multiclass": {
        "csv": MULTICLASS_CSV,
        "task": multiclass_task,
        "datamodule": MULTICLASS_DATAMODULE,
        "config": multiclass_config,
    },
    "regression": {
        "csv": REGRESSION_CSV,
        "task": regression_task,
        "datamodule": REGRESSION_DATAMODULE,
        "config": regression_config,
    },
}


def split_train_val(csv_path: Path, val_fraction: float = 0.25):
    """Split a fixture CSV's rows into (train_text, val_text), header preserved in both."""
    lines = csv_path.read_text().strip().splitlines()
    header, rows = lines[0], lines[1:]
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    train_text = "\n".join([header, *train_rows]) + "\n"
    val_text = "\n".join([header, *val_rows]) + "\n"
    return train_text, val_text
