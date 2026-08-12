"""Prompt assembly: a generator that never sees the contract writes code that fails
check 1, so what goes into the prompt is worth asserting."""

from adaptrna_agentic.codegen import prompts


PROFILE = {"path": "/data/train.csv", "target_type": "binary", "length_median": 400,
           "sequence_column": "sequence", "target_column": "label"}


def test_task_prompt_carries_everything_the_generator_needs():
    text = prompts.task_user_prompt("my_task", "detect X in sequences", PROFILE)

    # the contract
    assert "build_head(embed_dim, **head_config)" in text
    assert "extract_features" in text
    assert "ADAPTER_EXTRA_PREFIXES" in text
    # the silent-failure rules, both halves
    assert "adapter_extra_payload()" in text
    assert "CLS, EOS or padded positions" in text
    # the data
    assert "/data/train.csv" in text
    assert "detect X in sequences" in text
    # the worked example
    assert "ncrna_classification" in text
    assert "@register_task" in text
    # the import rule that makes staged code and landed code identical
    assert "adaptrna_custom.tasks.<task_name>.datamodule" in text


def test_binary_profile_selects_the_classification_template():
    template = prompts.template_for_profile(PROFILE)

    assert template["task"] == "splice_site"
    assert "CLS" in template["shape"]["extract_features"]


def test_continuous_profile_selects_the_regression_template():
    template = prompts.template_for_profile(
        {"target_type": "continuous", "length_median": 50}
    )

    assert template["task"] == "mrl"


def test_selected_template_is_included_in_the_prompt():
    text = prompts.task_user_prompt("t", "d", PROFILE)

    assert "Closest known task shape" in text
    assert "SpliceSitePredictionHead" in text


def test_feedback_is_appended_for_a_retry():
    text = prompts.task_user_prompt("t", "d", PROFILE, feedback="round trip failed")

    assert "previous attempt failed verification" in text
    assert "round trip failed" in text
    assert "Keep everything that worked" in text


def test_verifier_prompt_shows_code_harness_and_checklist():
    text = prompts.verifier_user_prompt(
        "detect X", PROFILE, {"task.py": "print('hi')"}, "task 'x': PASS"
    )

    assert "print('hi')" in text
    assert "task 'x': PASS" in text
    assert "Your checklist" in text
    assert "comfortable with these numbers in a paper" in text


def test_verifier_is_told_not_to_redo_the_harness():
    system = prompts.verifier_system_prompt()

    assert "fresh context" in system
    assert "do not re-litigate what the harness" in system


def test_external_tool_prompt_carries_the_contract_and_reference():
    text = prompts.external_tool_prompt("ViennaRNA", "fold RNA")

    assert "ExternalToolSpec" in text
    assert "SPEC" in text
    assert "Validate inputs BEFORE importing" in text
    assert "never invented numbers" in text
