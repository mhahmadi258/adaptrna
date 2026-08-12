"""Tests 5 and 6 -- multi-adapter residency in one backbone.

Test 5: three adapters resident in one hub, each task's output bit-identical to that task
loaded alone. Test 6 verifies the §6 assumption the design rests on -- that adapters with
*different* LoRA geometries can coexist, because peft keeps a per-adapter dict inside each
`lora.Linear` and its forward skips any active adapter a given layer does not hold.
"""

import pytest
import torch

from rinalmo_hub.hub import RiNALMoHub
from rinalmo_hub.lora import adapted_module_names, resident_adapters
from tests.helpers import LM_CONFIG, LORA_CONFIGS, TASKS, build_module, randomise_trainable


@pytest.fixture(scope="module")
def backbone_weights(tmp_path_factory):
    """
    A fixed random `nano` backbone shared by every hub in this file.

    Without it each `RiNALMoHub` would initialise its own random backbone and "same adapter,
    two hubs" comparisons would be meaningless.
    """
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo

    torch.manual_seed(1234)
    path = tmp_path_factory.mktemp("backbone") / "nano.pt"
    torch.save(RiNALMo(model_config(LM_CONFIG)).state_dict(), path)

    return path


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    """One trained-looking adapter per task, all on the same `nano` backbone."""
    directory = tmp_path_factory.mktemp("adapters")
    paths = {}

    for seed, task in enumerate(TASKS):
        module = randomise_trainable(
            build_module(task, lora=LORA_CONFIGS["stride3"], seed=seed), seed=100 + seed
        )
        if task == "mrl":
            with torch.no_grad():
                module.scaler._mean.fill_(4.2)
                module.scaler._std.fill_(1.7)
        if task == "sec_struct":
            module.threshold = 0.13

        paths[task] = module.save_adapter(directory / f"{task}.pt")

    return paths


def _hub(adapter_paths, backbone_weights):
    hub = RiNALMoHub(backbone_weights=backbone_weights, lm_config=LM_CONFIG, device="cpu")
    for path in adapter_paths:
        hub.register(path)

    return hub


def test_register_reads_the_task_from_the_file(adapters, backbone_weights):
    hub = _hub(adapters.values(), backbone_weights)

    assert hub.available() == TASKS
    assert resident_adapters(hub.backbone) == TASKS


def test_registering_twice_raises(adapters, backbone_weights):
    hub = _hub([adapters["mrl"]], backbone_weights)

    with pytest.raises(ValueError, match="already registered"):
        hub.register(adapters["mrl"])


def test_one_backbone_serves_every_task(adapters, backbone_weights):
    hub = _hub(adapters.values(), backbone_weights)

    backbones = {id(hub.module(task).backbone) for task in TASKS}
    assert backbones == {id(hub.backbone)}


def test_non_tensor_state_survives_registration(adapters, backbone_weights):
    hub = _hub(adapters.values(), backbone_weights)

    assert hub.module("sec_struct").threshold == pytest.approx(0.13)
    assert hub.module("mrl").scaler._mean.item() == pytest.approx(4.2)


@pytest.mark.parametrize("task", TASKS)
def test_multi_adapter_output_is_bit_identical_to_solo(task, adapters, sequences, backbone_weights):
    """Test 5. Three adapters resident must not perturb each other by a single bit."""
    solo = _hub([adapters[task]], backbone_weights)
    shared = _hub(adapters.values(), backbone_weights)

    expected = solo.predict(task, sequences)
    got = shared.predict(task, sequences)

    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, got)
    else:
        assert len(expected) == len(got)
        for a, b in zip(expected, got):
            assert (a == b).all()


def test_switching_back_and_forth_is_stateless(adapters, sequences, backbone_weights):
    """Predicting for another task in between must not change an answer."""
    hub = _hub(adapters.values(), backbone_weights)

    first = hub.predict("mrl", sequences)
    hub.predict("splice_site", sequences)
    hub.predict("sec_struct", sequences[:1])
    again = hub.predict("mrl", sequences)

    assert torch.equal(first, again)


def test_no_adapter_context_gives_the_base_model(adapters, sequences, backbone_weights):
    hub = _hub(adapters.values(), backbone_weights)

    adapted = hub.predict("splice_site", sequences)
    with hub.no_adapter():
        base = hub.predict("splice_site", sequences)
    restored = hub.predict("splice_site", sequences)

    assert not torch.allclose(adapted, base)
    assert torch.equal(adapted, restored)


def test_predictions_are_task_native(adapters, sequences, backbone_weights):
    hub = _hub(adapters.values(), backbone_weights)

    splice = hub.predict("splice_site", sequences)
    assert splice.shape == (len(sequences),)
    assert ((splice >= 0) & (splice <= 1)).all()

    mrl = hub.predict("mrl", sequences)
    assert mrl.shape == (len(sequences),)
    assert (mrl >= 0).all()  # inverse-scaled and clamped

    structures = hub.predict("sec_struct", sequences)
    assert len(structures) == len(sequences)
    for structure, sequence in zip(structures, sequences):
        assert structure.shape == (len(sequence), len(sequence))


def test_full_ft_export_is_refused(adapters, tmp_path):
    """A full-FT file carries only its head; serving it here would pair it with the
    pretrained backbone and silently produce nonsense."""
    module = build_module("splice_site", lora=None)
    path = module.save_adapter(tmp_path / "full_ft.pt")

    hub = RiNALMoHub(lm_config=LM_CONFIG, device="cpu")
    with pytest.raises(ValueError, match="full fine-tuning export"):
        hub.register(path)


# ---------------------------------------------------------------- test 6: heterogeneity


@pytest.fixture(scope="module")
def heterogeneous_adapters(tmp_path_factory):
    """Two splice-site adapters that adapt different numbers of blocks."""
    directory = tmp_path_factory.mktemp("heterogeneous")
    paths = {}

    for seed, name in enumerate(("stride1", "stride3")):
        module = randomise_trainable(
            build_module("splice_site", lora=LORA_CONFIGS[name], seed=seed), seed=500 + seed
        )
        paths[name] = module.save_adapter(directory / f"{name}.pt")

    return paths


def test_heterogeneous_adapters_coexist(heterogeneous_adapters, sequences, backbone_weights):
    """Test 6. Different `layer_stride` values, one backbone, correct switching."""
    hub = RiNALMoHub(backbone_weights=backbone_weights, lm_config=LM_CONFIG, device="cpu")
    hub.register(heterogeneous_adapters["stride1"], name="dense")
    hub.register(heterogeneous_adapters["stride3"], name="sparse")

    assert hub.available() == ["dense", "sparse"]

    # `dense` reaches every block; `sparse` only blocks 0 and 3 of the six.
    assert len(adapted_module_names(hub.backbone, "dense")) == 12
    assert len(adapted_module_names(hub.backbone, "sparse")) == 4

    dense_only = RiNALMoHub(backbone_weights=backbone_weights, lm_config=LM_CONFIG, device="cpu")
    dense_only.register(heterogeneous_adapters["stride1"], name="dense")

    sparse_only = RiNALMoHub(backbone_weights=backbone_weights, lm_config=LM_CONFIG, device="cpu")
    sparse_only.register(heterogeneous_adapters["stride3"], name="sparse")

    assert torch.equal(hub.predict("dense", sequences), dense_only.predict("dense", sequences))
    assert torch.equal(hub.predict("sparse", sequences), sparse_only.predict("sparse", sequences))

    # The two adapters must not be interchangeable -- otherwise the test above is vacuous.
    assert not torch.allclose(hub.predict("dense", sequences), hub.predict("sparse", sequences))


def test_activating_a_sparse_adapter_leaves_unadapted_blocks_bare(heterogeneous_adapters, backbone_weights):
    """peft's forward skips an active adapter a layer does not hold, rather than erroring."""
    hub = RiNALMoHub(backbone_weights=backbone_weights, lm_config=LM_CONFIG, device="cpu")
    hub.register(heterogeneous_adapters["stride1"], name="dense")
    hub.register(heterogeneous_adapters["stride3"], name="sparse")

    hub.activate("sparse")

    block_1 = hub.backbone.transformer.blocks[1].mh_attn.Wqkv
    assert "sparse" not in block_1.lora_A
    assert "dense" in block_1.lora_A

    tokens = hub.tokenize(["GGCAUUACGG"])
    with torch.no_grad():
        assert torch.isfinite(hub.backbone(tokens)["representation"]).all()
