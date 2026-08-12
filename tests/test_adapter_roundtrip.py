"""Tests 3 and 4 -- adapter round trips per task, and the guards that stop a bad load.

The two failure modes these cover both produce *plausible* numbers rather than errors:

* dropping `scaler.` from MRL's adapter loads fine and then un-scales with mean 0 / std 1;
* `strict=False` will accept a stale file and leave the head randomly initialised.
"""

import pytest
import torch

from rinalmo_hub.adapter import describe_adapter, load_adapter, save_adapter
from tests.helpers import HEAD_CONFIGS, LORA_CONFIGS, TASKS, build_module, randomise_trainable


def _trained(task, lora=LORA_CONFIGS["stride3"], seed=0):
    """A module whose trainable tensors have all been moved off their initial values."""
    module = randomise_trainable(build_module(task, lora=lora, seed=seed))

    if task == "mrl":
        # Sentinel scaler statistics: if these do not survive the round trip, every
        # prediction comes back on the wrong scale.
        with torch.no_grad():
            module.scaler._mean.fill_(4.2)
            module.scaler._std.fill_(1.7)
    if task == "sec_struct":
        # Non-tensor state: a plain Python float that `state_dict` knows nothing about.
        module.threshold = 0.13

    return module


@pytest.mark.parametrize("task", TASKS)
def test_adapter_round_trip(task, tmp_path):
    trained = _trained(task)
    path = trained.save_adapter(tmp_path / f"{task}.pt")

    # A *different* random draw, so equality afterwards cannot be an accident.
    fresh = build_module(task, lora=LORA_CONFIGS["stride3"], seed=99)
    head_before = {k: v.clone() for k, v in fresh.head.state_dict().items()}

    fresh.load_adapter(path)

    saved = trained.adapter_state_dict()
    loaded = fresh.adapter_state_dict()

    assert set(saved) == set(loaded)
    for key in saved:
        assert torch.equal(saved[key], loaded[key]), key

    # ... and the head demonstrably changed, i.e. the load was not a no-op.
    assert any(not torch.equal(head_before[k], v) for k, v in fresh.head.state_dict().items())


def test_mrl_adapter_carries_the_scaler(tmp_path):
    """ADAPTER_EXTRA_PREFIXES = ("scaler.",) -- omitting it is silent and wrong."""
    trained = _trained("mrl")
    path = trained.save_adapter(tmp_path / "mrl.pt")

    payload = load_adapter(path)
    assert "scaler._mean" in payload["state_dict"]
    assert "scaler._std" in payload["state_dict"]

    fresh = build_module("mrl", lora=LORA_CONFIGS["stride3"], seed=99)
    assert fresh.scaler._mean.item() == 0.0 and fresh.scaler._std.item() == 1.0

    fresh.load_adapter(path)
    assert fresh.scaler._mean.item() == pytest.approx(4.2)
    assert fresh.scaler._std.item() == pytest.approx(1.7)


def test_sec_struct_adapter_carries_the_threshold(tmp_path):
    """Non-tensor state travels in `extra`, not smuggled into the state dict."""
    trained = _trained("sec_struct")
    path = trained.save_adapter(tmp_path / "ss.pt")

    payload = load_adapter(path)
    assert payload["extra"]["threshold"] == pytest.approx(0.13)
    assert not any("threshold" in k for k in payload["state_dict"])

    fresh = build_module("sec_struct", lora=LORA_CONFIGS["stride3"], seed=99)
    assert fresh.threshold == 0.5

    fresh.load_adapter(path)
    assert fresh.threshold == pytest.approx(0.13)


def test_load_adapter_extra_tolerates_a_missing_threshold():
    """The source's unguarded `state_dict["threshold"]` KeyError'd on any dict lacking it."""
    module = build_module("sec_struct")
    module.load_adapter_extra({})

    assert module.threshold == 0.5


@pytest.mark.parametrize("task", TASKS)
def test_adapter_holds_only_adapter_tensors(task, tmp_path):
    path = _trained(task).save_adapter(tmp_path / f"{task}.pt")
    payload = load_adapter(path)

    for key in payload["state_dict"]:
        assert "lora_" in key or key.startswith(("head.", "scaler.")), key
    assert not any(k.startswith("backbone.transformer.blocks.0.mh_attn.Wqkv.base_layer")
                   for k in payload["state_dict"])


@pytest.mark.parametrize("task", TASKS)
def test_adapter_records_what_the_hub_needs(task, tmp_path):
    path = _trained(task).save_adapter(tmp_path / f"{task}.pt")
    payload = load_adapter(path)

    assert payload["format_version"] == 2
    assert payload["task"] == task
    assert payload["lm_config"] == "nano"
    assert payload["head_config"] == HEAD_CONFIGS[task]
    assert payload["lora"]["layer_stride"] == 3
    assert payload["metadata"]["arm"] == "lora"
    assert "created" in payload["metadata"]


# ---------------------------------------------------------------- test 4: the guards


def test_mismatched_layer_stride_raises(tmp_path):
    path = _trained("splice_site", lora=LORA_CONFIGS["stride3"]).save_adapter(tmp_path / "a.pt")
    other = build_module("splice_site", lora=LORA_CONFIGS["stride1"])

    with pytest.raises(ValueError, match="LoRA geometry"):
        other.load_adapter(path)


def test_full_state_dict_raises_on_unknown_keys(tmp_path):
    trained = _trained("splice_site")
    fresh = build_module("splice_site", lora=LORA_CONFIGS["stride3"], seed=99)

    with pytest.raises(KeyError, match="does not expect"):
        fresh.load_adapter_state_dict(trained.state_dict(), source="full-state-dict")


def test_missing_keys_raise(tmp_path):
    trained = _trained("splice_site")
    truncated = dict(trained.adapter_state_dict())
    truncated.pop(next(k for k in truncated if k.startswith("head.")))

    fresh = build_module("splice_site", lora=LORA_CONFIGS["stride3"], seed=99)
    with pytest.raises(KeyError, match="missing"):
        fresh.load_adapter_state_dict(truncated, source="truncated")


def test_giga_adapter_onto_a_mega_backbone_raises(tmp_path):
    """The `lm_config` guard, checked before any shape error can confuse the message."""
    path = tmp_path / "giga.pt"
    save_adapter(
        path,
        task="splice_site",
        lm_config="giga",
        lora=LORA_CONFIGS["stride3"],
        head_config={"head_embed_dim": 128},
        state_dict={},
    )

    with pytest.raises(ValueError, match="was trained on the 'giga' backbone"):
        load_adapter(path, lm_config="mega")


def test_wrong_task_raises(tmp_path):
    path = _trained("mrl").save_adapter(tmp_path / "mrl.pt")
    other = build_module("splice_site", lora=LORA_CONFIGS["stride3"])

    with pytest.raises(ValueError, match="is for task 'mrl'"):
        other.load_adapter(path)


def test_unsupported_format_version_raises(tmp_path):
    path = tmp_path / "v1.pt"
    torch.save({"format_version": 1, "state_dict": {}}, path)

    with pytest.raises(ValueError, match="format version 1"):
        load_adapter(path)


def test_describe_adapter_reports_the_essentials(tmp_path):
    path = _trained("mrl").save_adapter(tmp_path / "mrl.pt")
    text = describe_adapter(path)

    assert "task           : mrl" in text
    assert "lm_config      : nano" in text
    assert "scaler._mean" in text
