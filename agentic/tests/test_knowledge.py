"""The knowledge base must keep carrying the load-bearing numbers: this suite fails
loudly if an edit drops one, because the recommender has no other source for them."""

import pytest

from adaptrna_agentic.knowledge import (
    arm,
    load_knowledge,
    task_knowledge,
    template_for,
    templates,
    universal,
)


def test_lora_validated_settings_present():
    lora = arm("lora")

    assert lora["optim"]["lr"] == pytest.approx(3.0e-4)
    assert lora["trainer"]["gradient_clip_val"] == 1.0
    assert lora["lora"]["layer_stride"] == 3
    assert lora["lora"]["r"] == 16
    assert lora["lora"]["alpha"] == 32


def test_lora_collapse_failure_mode_documented():
    modes = arm("lora")["failure_modes"]
    collapse = next(m for m in modes if "1e-3" in m["setting"])

    assert "constant-output" in collapse["symptom"]
    assert "3e-4" in collapse["remedy"]


def test_full_ft_settings_and_failure_mode():
    full_ft = arm("full_ft")

    assert full_ft["optim"]["lr"] == pytest.approx(1.0e-5)
    destroyed = next(m for m in full_ft["failure_modes"] if "1e-4" in m["setting"])
    assert "R^2" in destroyed["symptom"]


def test_precision_and_nondeterminism_note():
    assert universal()["trainer"]["precision"] == "bf16-mixed"

    note = universal()["nondeterminism"]["note"]
    assert "95.21" in note and "95.82" in note


@pytest.mark.parametrize("task", ["splice_site", "mrl", "sec_struct"])
def test_every_task_has_metric_and_tolerance(task):
    knowledge = task_knowledge(task)

    assert knowledge["primary_metric"].startswith("test/")
    assert knowledge["reference"]["tolerance"] > 0
    assert knowledge["higher_is_better"] is True


def test_splice_site_reference_band_and_tolerance():
    reference = task_knowledge("splice_site")["reference"]

    low, high = reference["band"]
    assert low <= 95.82 <= high          # published donor LoRA F1
    assert low <= 97.48 <= high          # this repo's own donor run
    assert reference["tolerance"] == 1.0  # FlashAttention non-determinism


def test_mrl_random7600_caveat_present():
    caveats = " ".join(task_knowledge("mrl")["caveats"])

    assert "random7600" in caveats
    assert "holdout" in caveats
    assert "early stopping" in caveats


@pytest.mark.parametrize("task", ["splice_site", "mrl", "sec_struct"])
def test_every_template_names_head_loss_metrics(task):
    template = template_for(task)

    assert template is not None
    shape = template["shape"]
    for key in ("head", "loss", "metrics", "extract_features", "predict_output"):
        assert shape.get(key), f"{task} template missing {key}"
    assert template["data_layout"]["description"]


def test_templates_cover_the_shipped_tasks():
    assert {t["task"] for t in templates()} == {"splice_site", "mrl", "sec_struct"}


def test_no_match_guidance_is_honest_about_new_datamodules():
    guidance = load_knowledge()["no_match_guidance"]

    assert "three files" in guidance
    assert "does not yet automate" in guidance
