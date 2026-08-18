"""Shared fixtures: real nano adapter files built through the engine's *public* API only,
mirroring the engine's own test philosophy — random nano backbone, no weights, no data.

Phase 13 (D1): the agentic layer must not name a shipped task, even in its own test
fixtures — these two module classes exist only so the fixtures below have a real, valid
adapter file to exercise registry/runtime/CLI/API mechanics against. They are defined
here, not imported from `rinalmo_hub.tasks`, and never registered under a name that
collides with anything shipped.

Engine/torch imports live inside fixtures so the Phase 0 tests stay import-light.
"""

import pytest

NANO_LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 3}

_CLASS_CACHE = {}


def _demo_binary_module_class():
    """A minimal binary classifier, matching the shape `codegen/templates/task.py.j2`
    renders for a binary target — defined here rather than borrowed from a shipped task.
    Cached at module scope: `register_task` refuses a second registration under the same
    name, and both `nano_splice_adapter` and `full_ft_export` need this class.
    """
    if _CLASS_CACHE.get("binary") is not None:
        return _CLASS_CACHE["binary"]

    import torch
    import torch.nn as nn
    from torchmetrics.classification import BinaryAccuracy

    from rinalmo_hub.module import BaseDownstreamModule
    from rinalmo_hub.registry import register_task

    @register_task("demo_binary")
    class _DemoBinaryModule(BaseDownstreamModule):
        TASK_NAME = "demo_binary"
        PRIMARY_METRIC = "test/acc"

        def build_head(self, embed_dim, head_embed_dim: int = 16, **kwargs):
            return nn.Sequential(
                nn.Linear(embed_dim, head_embed_dim), nn.GELU(), nn.Linear(head_embed_dim, 1)
            )

        def build_metrics(self, stage):
            return None if stage == "train" else BinaryAccuracy()

        def extract_features(self, representation, tokens):
            return representation[:, 0]

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
                return {}
            return {f"{stage}/acc": self.metrics[stage].compute()}

        def postprocess_predictions(self, outputs, tokens, sequences):
            return torch.sigmoid(outputs.float()).squeeze(-1)

    _CLASS_CACHE["binary"] = _DemoBinaryModule
    return _DemoBinaryModule


def _demo_regression_module_class():
    """A minimal pooled regressor whose head is pad-sensitive — matching the shape
    `codegen/templates/task.py.j2` renders for a regression target, so fixtures that need
    a pad-sensitive tool have one without naming a shipped task for it. Cached at module
    scope for the same reason as `_demo_binary_module_class`."""
    if _CLASS_CACHE.get("regression") is not None:
        return _CLASS_CACHE["regression"]

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

    from rinalmo_hub.module import BaseDownstreamModule
    from rinalmo_hub.registry import register_task

    class _PooledHead(nn.Module):
        def __init__(self, embed_dim, hidden_dim=8):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )

        def forward(self, representation, pad_mask):
            keep = (~pad_mask).unsqueeze(-1).to(representation.dtype)
            pooled = (representation * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
            return self.mlp(pooled).squeeze(-1)

    @register_task("demo_regression")
    class _DemoRegressionModule(BaseDownstreamModule):
        TASK_NAME = "demo_regression"
        PRIMARY_METRIC = "test/mse"

        def build_head(self, embed_dim, head_embed_dim: int = 8, **kwargs):
            return _PooledHead(embed_dim, head_embed_dim)

        def build_metrics(self, stage):
            if stage == "train":
                return None
            return nn.ModuleDict({"mse": MeanSquaredError(), "mae": MeanAbsoluteError()})

        def extract_features(self, representation, tokens):
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
                return {}
            return {f"{stage}/{name}": metric.compute() for name, metric in self.metrics[stage].items()}

        def postprocess_predictions(self, outputs, tokens, sequences):
            return outputs.float()

    _CLASS_CACHE["regression"] = _DemoRegressionModule
    return _DemoRegressionModule


def _build_nano_module(task_cls, lora):
    import torch

    torch.manual_seed(0)
    module = task_cls(lm_config="nano", head_config={}, lora=lora)
    if lora is not None:
        module.apply_lora(verbose=False)
    module.eval()

    # Move every trainable tensor off its initialised value (lora_B starts at zero),
    # so saved adapters carry distinguishable state.
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name or name.startswith("head."):
                param.copy_(torch.rand(param.shape, generator=generator) * 0.1 + 0.01)

    return module


@pytest.fixture(scope="session")
def nano_splice_adapter(tmp_path_factory):
    """A real, valid nano LoRA adapter — real forward pass, real save/load — for tests
    that just need *some* registered tool to exercise. (Named for its original shipped-
    task role in this test suite's history; the module it wraps names no shipped task.)
    """
    path = tmp_path_factory.mktemp("adapters") / "demo_binary_adapter.pt"
    _build_nano_module(_demo_binary_module_class(), dict(NANO_LORA)).save_adapter(path)
    return path


@pytest.fixture(scope="session")
def nano_regression_adapter(tmp_path_factory):
    """The pad-sensitive counterpart to `nano_splice_adapter`, for tests exercising two
    tools on one shared backbone or pad-sensitive serving specifically."""
    path = tmp_path_factory.mktemp("adapters") / "demo_regression_adapter.pt"
    _build_nano_module(_demo_regression_module_class(), dict(NANO_LORA)).save_adapter(path)
    return path


@pytest.fixture(scope="session")
def full_ft_export(tmp_path_factory):
    """A head-only export from a full-FT module — what the registry must refuse."""
    path = tmp_path_factory.mktemp("adapters") / "demo_binary_full_ft.pt"
    _build_nano_module(_demo_binary_module_class(), None).save_adapter(path)
    return path


@pytest.fixture
def nano_registry(tmp_path):
    """A fresh registry on a tmp dir, configured for a random nano backbone."""
    from adaptrna_agentic.toolhub.registry import Registry

    registry = Registry(data_dir=tmp_path / "toolhub_data")
    registry.configure_backbone(lm_config="nano", weights="null", device="cpu", dtype="auto")
    return registry
