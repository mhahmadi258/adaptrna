"""Test 2 -- LoRA injection adapts exactly the intended modules, and freezes everything else.

A silent zero-match would train only the head and look exactly like "LoRA doesn't work", and
a missing freeze would quietly train the whole backbone at LoRA learning rates. Both are
checked here rather than discovered eight GPU-hours later.
"""

import pytest
import torch

from rinalmo_hub.lora import (
    LoRASpec,
    adapted_module_names,
    disable_adapters,
    inject_lora,
    resident_adapters,
    set_active_adapter,
)
from tests.helpers import LORA_CONFIGS, NUM_BLOCKS, TASKS, build_module


@pytest.mark.parametrize(
    "stride,expected_blocks",
    [(1, [0, 1, 2, 3, 4, 5]), (2, [0, 2, 4]), (3, [0, 3])],
)
def test_layer_indices_follow_the_stride(stride, expected_blocks):
    spec = LoRASpec(layer_stride=stride)
    assert spec.layer_indices(NUM_BLOCKS) == expected_blocks


def test_giga_stride_3_picks_eleven_of_thirty_three_blocks():
    spec = LoRASpec(layer_stride=3)
    assert spec.layer_indices(33) == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    assert len(spec.target_module_names(33)) == 22  # two targets per adapted block


def test_explicit_layers_override_the_stride():
    spec = LoRASpec(layer_stride=3, layers=[5, 1, 1])
    assert spec.layer_indices(NUM_BLOCKS) == [1, 5]


def test_out_of_range_layers_raise():
    with pytest.raises(ValueError):
        LoRASpec(layers=[NUM_BLOCKS]).layer_indices(NUM_BLOCKS)


@pytest.mark.parametrize("name", ["stride1", "stride3"])
def test_adapted_module_count_matches_request(name):
    lora = LORA_CONFIGS[name]
    module = build_module("splice_site", lora=lora)

    spec = LoRASpec.from_dict(lora)
    expected = spec.target_module_names(NUM_BLOCKS)
    adapted = adapted_module_names(module.backbone, "default")

    assert sorted(adapted) == sorted(expected)
    assert len(adapted) == 2 * len(spec.layer_indices(NUM_BLOCKS))


@pytest.mark.parametrize("name", ["stride1", "stride3"])
def test_only_lora_and_head_are_trainable(name):
    module = build_module("splice_site", lora=LORA_CONFIGS[name])

    for param_name, param in module.backbone.named_parameters():
        expected = "lora_" in param_name
        assert param.requires_grad is expected, f"{param_name} requires_grad={param.requires_grad}"

    assert all(p.requires_grad for p in module.head.parameters())


def test_zero_match_target_modules_raises():
    module = build_module("splice_site")
    module.lora_spec = LoRASpec(layer_stride=1, target_modules=["mh_attn.does_not_exist"])

    with pytest.raises(RuntimeError, match="adapted 0 modules"):
        module.apply_lora(verbose=False)


def test_partial_match_target_modules_raises():
    """The dangerous case: peft is happy, but half the requested targets were never found."""
    module = build_module("splice_site")
    module.lora_spec = LoRASpec(
        layer_stride=1, target_modules=["mh_attn.Wqkv", "mh_attn.does_not_exist"]
    )

    with pytest.raises(RuntimeError, match=r"adapted 6 modules but 12 were requested"):
        module.apply_lora(verbose=False)


def test_injection_moves_the_original_weight_to_base_layer():
    """Why the pretrained backbone has to be loaded *before* injection."""
    module = build_module("splice_site")
    before = module.backbone.transformer.blocks[0].mh_attn.Wqkv.weight.detach().clone()

    module.lora_spec = LoRASpec(**LORA_CONFIGS["stride3"])
    module.apply_lora(verbose=False)

    adapted = module.backbone.transformer.blocks[0].mh_attn.Wqkv
    assert torch.equal(adapted.base_layer.weight, before)
    assert "transformer.blocks.0.mh_attn.Wqkv.base_layer.weight" in module.backbone.state_dict()


@pytest.mark.parametrize("task", TASKS)
def test_lora_b_starts_at_zero_so_injection_is_output_preserving(task, tokens):
    """A freshly injected adapter must not change the backbone's output."""
    plain = build_module(task, seed=7)
    adapted = build_module(task, lora=LORA_CONFIGS["stride1"], seed=7)

    batch = tokens[:1] if task == "sec_struct" else tokens
    with torch.no_grad():
        before = plain.backbone(batch)["representation"]
        after = adapted.backbone(batch)["representation"]

    assert torch.equal(before, after)


def test_disable_adapters_restores_the_base_model(tokens):
    module = build_module("splice_site", lora=LORA_CONFIGS["stride1"])

    with torch.no_grad():
        for name, param in module.backbone.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.full_like(param, 0.05))

        adapted = module.backbone(tokens)["representation"]
        with disable_adapters(module.backbone):
            base = module.backbone(tokens)["representation"]
        adapted_again = module.backbone(tokens)["representation"]

    assert not torch.allclose(adapted, base)
    assert torch.equal(adapted, adapted_again)


def test_set_active_adapter_rejects_unknown_names():
    module = build_module("splice_site", lora=LORA_CONFIGS["stride3"])

    assert resident_adapters(module.backbone) == ["default"]
    with pytest.raises(KeyError):
        set_active_adapter(module.backbone, "nope")


def test_inject_lora_is_callable_without_a_module():
    """`lora.py` is usable on a bare backbone, not only through the mixin."""
    module = build_module("splice_site")
    adapted = inject_lora(module.backbone, LoRASpec(layer_stride=2, r=2), adapter_name="x", verbose=False)

    assert len(adapted) == 6  # blocks 0, 2, 4 x two targets
    assert resident_adapters(module.backbone) == ["x"]
