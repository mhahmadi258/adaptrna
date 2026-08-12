"""Acceptance test for the design (§7).

A fourth task -- ncRNA classification -- is added by three files under
`examples/ncrna_classification/` and nothing else. This test asserts that: no core file was
edited, the task appears in the registry, it builds on `nano`, LoRA injects, a forward pass
runs, and its adapter round-trips like any shipped task.

If this test ever needs a change to `module.py`, `lora.py`, `adapter.py`, `hub.py` or the
CLI to keep passing, the abstraction is wrong.
"""

import hashlib
from pathlib import Path

import pytest
import torch

from examples.ncrna_classification.task import ncRNAClassificationModule  # noqa: F401
from rinalmo_hub.config import resolve_config
from rinalmo_hub.hub import RiNALMoHub
from rinalmo_hub.registry import available_tasks, get_task
from tests.helpers import randomise_trainable

TASK = "ncrna_classification"
HEAD_CONFIG = {"head_embed_dim": 8, "n_classes": 5}
LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 3}

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_FILES = [
    "rinalmo_hub/module.py",
    "rinalmo_hub/lora.py",
    "rinalmo_hub/adapter.py",
    "rinalmo_hub/hub.py",
    "rinalmo_hub/registry.py",
    "rinalmo_hub/cli/common.py",
    "rinalmo_hub/cli/train.py",
    "rinalmo_hub/cli/evaluate.py",
    "rinalmo_hub/cli/predict.py",
]


def _build(seed=0, lora=LORA):
    torch.manual_seed(seed)
    module = get_task(TASK)(lm_config="nano", head_config=HEAD_CONFIG, lora=lora)

    if lora is not None:
        module.apply_lora(verbose=False)

    return module.eval()


def test_importing_the_example_registers_a_fourth_task():
    assert TASK in available_tasks()


def test_no_core_file_mentions_the_new_task():
    """The strongest form of "no edits to core": core does not know this task exists."""
    for relative in CORE_FILES:
        text = (REPO_ROOT / relative).read_text()
        assert "ncrna" not in text.lower(), f"{relative} references the example task"


def test_the_example_is_exactly_three_files():
    directory = REPO_ROOT / "examples" / "ncrna_classification"
    files = sorted(
        p.name for p in directory.iterdir()
        if p.is_file() and p.name != "__init__.py"
    )

    assert files == ["config.yaml", "datamodule.py", "task.py"]


def test_it_builds_and_runs_a_forward_pass(tokens, sequences):
    module = _build()

    assert module.use_lora
    with torch.no_grad():
        outputs = module(tokens)

    assert outputs.shape == (len(sequences), HEAD_CONFIG["n_classes"])
    assert torch.isfinite(outputs).all()


def test_the_generic_training_step_works_without_a_trainer():
    """`compute_loss` / `update_metrics` / `compute_metrics` wire up through the base class."""
    module = _build()

    tokens = torch.randint(5, 20, (4, 16))
    batch = (["fam"] * 4, tokens, torch.tensor([0, 1, 2, 3]))

    with torch.no_grad():
        outputs = module(module.batch_tokens(batch))
        loss = module.compute_loss(outputs, batch)
        module.update_metrics(outputs, batch, "test")

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(module.compute_metrics("test")) == {"test/acc", "test/f1"}


def test_adapter_round_trips_like_any_other_task(tmp_path, sequences):
    trained = randomise_trainable(_build(seed=0))
    path = trained.save_adapter(tmp_path / "ncrna.pt")

    fresh = _build(seed=99)
    fresh.load_adapter(path)

    saved, loaded = trained.adapter_state_dict(), fresh.adapter_state_dict()
    assert set(saved) == set(loaded)
    assert all(torch.equal(saved[k], loaded[k]) for k in saved)


def test_it_serves_from_the_hub(tmp_path, sequences):
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo

    torch.manual_seed(4321)
    weights = tmp_path / "nano.pt"
    torch.save(RiNALMo(model_config("nano")).state_dict(), weights)

    path = randomise_trainable(_build()).save_adapter(tmp_path / "ncrna.pt")

    hub = RiNALMoHub(backbone_weights=weights, lm_config="nano", device="cpu")
    assert hub.register(path) == TASK

    predictions = hub.predict(TASK, sequences)
    assert predictions.shape == (len(sequences),)
    assert ((predictions >= 0) & (predictions < HEAD_CONFIG["n_classes"])).all()


def test_the_example_config_resolves():
    cfg = resolve_config(REPO_ROOT / "examples" / "ncrna_classification" / "config.yaml")

    assert cfg.task == TASK
    assert cfg.head.n_classes == 88
    assert cfg.lora.layer_stride == 3
    assert cfg.trainer.precision == "bf16-mixed"   # inherited from configs/base.yaml
