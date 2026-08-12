"""Test 1 -- the registry sees exactly the three shipped tasks, and each one constructs."""

import pathlib

import pytest
import torch

from rinalmo_hub.registry import available_tasks, get_task
from tests.helpers import HEAD_CONFIGS, LORA_CONFIGS, TASKS, build_module




def test_available_tasks():
    """
    Test 1. Run in a clean interpreter: the registry is process-global, and this suite also
    imports the `examples/ncrna_classification` task to prove a fourth one can be added.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import rinalmo_hub.tasks; from rinalmo_hub.registry import available_tasks; "
         "print(','.join(available_tasks()))"],
        capture_output=True, text=True, check=True,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
    )

    assert result.stdout.strip().splitlines()[-1] == ",".join(TASKS)


def test_shipped_tasks_are_registered():
    assert set(TASKS) <= set(available_tasks())


def test_task_names_match_registry():
    for task in TASKS:
        assert get_task(task).TASK_NAME == task


def test_every_datamodule_comes_from_the_unified_lightning_package():
    """
    Mixing `pytorch_lightning` and `lightning.pytorch` fails only at `trainer.fit`, with a
    bare `ValueError: Expected a parent` from `is_overridden` -- no mention of the package,
    the datamodule or the task. Importing the classes is enough to catch it, and costs
    nothing: no data is touched.
    """
    import lightning.pytorch as pl

    from rinalmo.data.downstream.ribosome_loading.datamodule import RibosomeLoadingDataModule
    from rinalmo.data.downstream.secondary_structure.datamodule import SecondaryStructureDataModule
    from rinalmo.data.downstream.splice_site_prediction.datamodule import SpliceSiteDataModule
    from rinalmo_hub.data.mrl import MRLDataModule

    for cls in (SpliceSiteDataModule, RibosomeLoadingDataModule,
                SecondaryStructureDataModule, MRLDataModule):
        assert issubclass(cls, pl.LightningDataModule), (
            f"{cls.__name__} subclasses the wrong LightningDataModule; `Trainer` will reject it"
        )


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        get_task("no_such_task")


@pytest.mark.parametrize("task", TASKS)
def test_module_constructs(task):
    module = build_module(task)

    assert module.backbone is not None
    assert module.head is not None
    assert module.use_lora is False
    assert module.embed_dim == 320  # "nano" width


@pytest.mark.parametrize("task", TASKS)
def test_module_constructs_with_lora(task):
    module = build_module(task, lora=LORA_CONFIGS["stride3"])

    assert module.use_lora is True
    trainable = {n for n, p in module.named_parameters() if p.requires_grad}
    assert any("lora_" in n for n in trainable)


@pytest.mark.parametrize("task", TASKS)
def test_forward_runs_on_cpu(task, tokens, sequences):
    module = build_module(task, lora=LORA_CONFIGS["stride3"])

    with torch.no_grad():
        if task == "sec_struct":
            # One variable-length structure at a time; the head is quadratic in length.
            outputs = module(tokens[:1])
            assert outputs.shape == (1, tokens.shape[1] - 2, tokens.shape[1] - 2)
        else:
            outputs = module(tokens)
            assert outputs.shape[0] == len(sequences)

    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("task", TASKS)
def test_head_config_is_honoured(task):
    module = build_module(task)
    assert module.head_config == HEAD_CONFIGS[task]


def test_unknown_head_config_key_raises():
    from rinalmo_hub.registry import get_task

    with pytest.raises(TypeError):
        get_task("splice_site")(lm_config="nano", head_config={"nonsense": 1})
