"""Shared builders for the CPU test suite.

Everything here runs on a randomly-initialised `nano` backbone (6 blocks, width 320) on the
CPU: no pretrained weights, no datasets, no GPU.
"""

import torch

import rinalmo_hub.tasks  # noqa: F401  -- registers every task
from rinalmo_hub.registry import get_task

LM_CONFIG = "nano"
NUM_BLOCKS = 6  # `nano` transformer depth, see rinalmo/config.py

TASKS = ["mrl", "sec_struct", "splice_site"]

# Head configs small enough that a full forward pass on CPU stays instant.
HEAD_CONFIGS = {
    "splice_site": {"head_embed_dim": 16},
    "mrl": {"head_embed_dim": 8, "num_blocks": 1},
    "sec_struct": {"num_resnet_blocks": 1, "conv_dim": 8},
}

LORA_CONFIGS = {
    "stride1": {"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 1},
    "stride3": {"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 3},
}

SEQUENCES = [
    "GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC",
    "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC",
    "GGGAAACCCUUUGGGAAACCCUUUGGGAAACC",
]


def build_module(task, lora=None, seed=0, attach_backbone=True, task_config=None):
    """A task module on the `nano` backbone, with LoRA already injected when asked for."""
    torch.manual_seed(seed)

    module = get_task(task)(
        lm_config=LM_CONFIG,
        head_config=HEAD_CONFIGS[task],
        lora=lora,
        task_config=task_config or {},
        attach_backbone=attach_backbone,
    )
    if lora is not None and attach_backbone:
        module.apply_lora(verbose=False)

    module.eval()

    return module


def randomise_trainable(module, seed=1234):
    """
    Move every trainable tensor off its initialised value.

    `lora_B` starts at zero and a freshly built head is a different random draw each time,
    so without this a "round trip" test can pass on tensors that were never distinguishable.
    """
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_" in name or name.startswith("head."):
                param.copy_(torch.rand(param.shape, generator=generator) * 0.1 + 0.01)

    return module
