"""Tests 7-9 -- the ones that need a GPU, the pretrained backbone, or a dataset.

None of these run at build time. Each is written, each is skipped unless you point it at
what it needs:

    # test 7: gradient flow in bf16 on a GPU
    pytest tests/test_user_run.py -m gpu

    # test 8: the regression test -- the strongest guarantee the port preserved behaviour
    RINALMO_WEIGHTS=weights/giga-v1.pt \
    RINALMO_DONOR_ADAPTER=adapters/donor_adapter.pt \
    RINALMO_SPLICE_TEST_DATA=dataset/test_data \
        pytest tests/test_user_run.py -m "weights and data"

Environment variables:
    RINALMO_WEIGHTS             path to giga-v1.pt
    RINALMO_DONOR_ADAPTER       6.08 MB splice-site donor LoRA adapter
    RINALMO_SPLICE_TEST_DATA    directory holding Danio/Fly/Thaliana/Worm
"""

import os
from pathlib import Path

import pytest
import torch

from tests.helpers import LORA_CONFIGS, build_module

WEIGHTS = os.environ.get("RINALMO_WEIGHTS")
DONOR_ADAPTER = os.environ.get("RINALMO_DONOR_ADAPTER")
SPLICE_TEST_DATA = os.environ.get("RINALMO_SPLICE_TEST_DATA")

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
needs_weights = pytest.mark.skipif(
    not (WEIGHTS and Path(WEIGHTS).exists()), reason="set RINALMO_WEIGHTS to giga-v1.pt"
)
needs_donor_adapter = pytest.mark.skipif(
    not (DONOR_ADAPTER and Path(DONOR_ADAPTER).exists()),
    reason="set RINALMO_DONOR_ADAPTER to the trained donor adapter",
)
needs_splice_data = pytest.mark.skipif(
    not (SPLICE_TEST_DATA and Path(SPLICE_TEST_DATA).exists()),
    reason="set RINALMO_SPLICE_TEST_DATA to the splice-site benchmark directory",
)


# ---------------------------------------------------------------- test 7 (GPU)


@pytest.mark.gpu
@needs_gpu
@pytest.mark.parametrize("task", ["splice_site", "mrl"])
def test_gradient_flow_in_bf16(task):
    """
    Every `lora_B` and every head parameter gets a non-zero gradient, no frozen parameter
    gets one, and FlashAttention does not trip its dtype assert.

    `lora_B` starts at zero, so this is also the check that gradients reach the adapters at
    all -- which they only do because gradient checkpointing runs with `use_reentrant=False`.
    """
    module = build_module(task, lora=LORA_CONFIGS["stride1"]).cuda().train()

    tokens = torch.randint(5, 10, (2, 32), device="cuda")
    targets = torch.rand(2, device="cuda") * 4.0

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = module(tokens)
        loss = module.compute_loss(outputs, (tokens, targets))

    loss.backward()

    lora_b = {n: p for n, p in module.named_parameters() if "lora_B" in n}
    head = {n: p for n, p in module.head.named_parameters()}
    frozen = {n: p for n, p in module.backbone.named_parameters() if not p.requires_grad}

    assert lora_b and head
    for name, param in {**lora_b, **head}.items():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.any(param.grad != 0), f"{name} gradient is all zeros"

    for name, param in frozen.items():
        assert param.grad is None, f"frozen parameter {name} received a gradient"


@pytest.mark.gpu
@needs_gpu
def test_flash_attention_path_is_used_on_gpu():
    """The reference CPU path must not silently take over a GPU run."""
    from rinalmo.model.attention import FLASH_ATTN_AVAILABLE

    assert FLASH_ATTN_AVAILABLE, "flash_attn is not importable; GPU runs would fall back to the slow path"

    module = build_module("splice_site").cuda().eval()
    tokens = torch.randint(5, 10, (2, 32), device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.isfinite(module(tokens)).all()


@pytest.mark.gpu
@needs_gpu
def test_reference_attention_agrees_with_the_flash_kernel():
    """
    The CPU tests must exercise the same computation the GPU runs.

    `rinalmo/model/attention.py` gained a plain-PyTorch path so the suite can run a forward
    pass without CUDA. If its rotary convention or its `key_padding_mask` polarity drifted
    from flash-attn's, every CPU test would still pass while testing the wrong thing.
    """
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo

    torch.manual_seed(0)
    model = RiNALMo(model_config("nano")).cuda().eval()
    tokens = torch.randint(5, 20, (2, 40), device="cuda")

    with torch.no_grad():
        reference = model(tokens)["representation"].float()      # fp32 -> reference path
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flash = model(tokens)["representation"].float()      # bf16 -> flash kernel

    # bf16 carries ~3 decimal digits, so agreement is judged relative to the signal.
    relative = (reference - flash).abs().mean() / reference.abs().mean()
    assert relative < 0.01, f"reference and flash paths diverge (relative error {relative:.4f})"


# ---------------------------------------------------------------- test 8 (regression)

EXPECTED_DONOR_F1 = 95.81500244140625
EXPECTED_DONOR_LOSS = 0.15762274647900157


@pytest.mark.weights
@pytest.mark.data
@needs_gpu
@needs_weights
@needs_donor_adapter
@needs_splice_data
def test_donor_adapter_reproduces_the_published_metrics(tmp_path):
    """
    Test 8, and the strongest guarantee the port preserved behaviour: the existing 6.08 MB
    donor adapter plus `giga-v1.pt` must reproduce the numbers the source project measured.

    The forward pass is bitwise deterministic (only FlashAttention's *backward* is not), so
    these are exact equalities, not tolerances.
    """
    from rinalmo_hub.cli.evaluate import build_parser, main

    metrics_path = tmp_path / "metrics.json"
    args = [
        "--adapter", DONOR_ADAPTER,
        "--pretrained_weights", WEIGHTS,
        "--config", "configs/tasks/splice_site.yaml",
        "--set", f"data.test_root={SPLICE_TEST_DATA}",
        "--set", "data.root=null",
        "--set", "trainer.devices=1",
        "--output_dir", str(tmp_path),
        "--metrics_out", str(metrics_path),
    ]
    build_parser().parse_args(args)  # fails fast on a bad flag
    main(args)

    import json
    metrics = json.loads(metrics_path.read_text())

    assert metrics["test/f1_score"] == pytest.approx(EXPECTED_DONOR_F1, abs=1e-6)
    assert metrics["test/loss"] == pytest.approx(EXPECTED_DONOR_LOSS, abs=1e-6)


@pytest.mark.weights
@needs_weights
def test_pretrained_weights_load_strictly_into_the_backbone():
    """
    The vendored backbone must still accept `giga-v1.pt` with `strict=True`.

    Adding a buffered module anywhere under `RiNALMo` -- a rotary cache, say -- would add
    state-dict keys the released checkpoint does not have, and this is what catches it.
    """
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo

    model = RiNALMo(model_config("giga"))
    state_dict = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

    missing, unexpected = model.load_state_dict(state_dict, strict=True), None
    assert missing.missing_keys == [] and missing.unexpected_keys == []
    assert unexpected is None


@pytest.mark.weights
@needs_weights
def test_lora_injection_on_giga_adapts_eleven_blocks():
    """The parameter counts quoted in the README, on the real backbone."""
    from rinalmo_hub.lora import LoRASpec, adapted_module_names
    from rinalmo_hub.registry import get_task

    module = get_task("splice_site")(
        lm_config="giga", head_config={"head_embed_dim": 128},
        lora={"r": 16, "alpha": 32, "dropout": 0.05, "layer_stride": 3},
    )
    module.load_pretrained_backbone(WEIGHTS)
    module.apply_lora(verbose=False)

    assert LoRASpec(layer_stride=3).layer_indices(33) == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    assert len(adapted_module_names(module.backbone, "default")) == 22

    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    assert trainable == 1_515_777
    assert 100 * trainable / total == pytest.approx(0.232, abs=5e-4)
