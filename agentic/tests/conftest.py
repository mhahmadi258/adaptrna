"""Shared fixtures: real nano adapter files built through the engine's *public* API only,
mirroring the engine's own test philosophy — random nano backbone, no weights, no data.

Engine/torch imports live inside fixtures so the Phase 0 tests stay import-light.
"""

import pytest

NANO_LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 3}

HEAD_CONFIGS = {
    "splice_site": {"head_embed_dim": 16},
    "mrl": {"head_embed_dim": 8, "num_blocks": 1},
}


def _build_nano_module(task: str, lora):
    import torch

    import rinalmo_hub.tasks  # noqa: F401 -- registers the shipped tasks
    from rinalmo_hub.registry import get_task

    torch.manual_seed(0)
    module = get_task(task)(lm_config="nano", head_config=HEAD_CONFIGS[task], lora=lora)
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
    path = tmp_path_factory.mktemp("adapters") / "splice_site_adapter.pt"
    _build_nano_module("splice_site", dict(NANO_LORA)).save_adapter(path)
    return path


@pytest.fixture(scope="session")
def nano_mrl_adapter(tmp_path_factory):
    path = tmp_path_factory.mktemp("adapters") / "mrl_adapter.pt"
    _build_nano_module("mrl", dict(NANO_LORA)).save_adapter(path)
    return path


@pytest.fixture(scope="session")
def full_ft_export(tmp_path_factory):
    """A head-only export from a full-FT module — what the registry must refuse."""
    path = tmp_path_factory.mktemp("adapters") / "splice_site_full_ft.pt"
    _build_nano_module("splice_site", None).save_adapter(path)
    return path


@pytest.fixture
def nano_registry(tmp_path):
    """A fresh registry on a tmp dir, configured for a random nano backbone."""
    from adaptrna_agentic.toolhub.registry import Registry

    registry = Registry(data_dir=tmp_path / "toolhub_data")
    registry.configure_backbone(lm_config="nano", weights="null", device="cpu", dtype="auto")
    return registry
