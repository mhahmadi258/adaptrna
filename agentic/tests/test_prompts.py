"""Prompt assembly: a generator that never sees the contract writes code that fails
check 1, so what goes into the prompt is worth asserting.

Phase 13 (D6): the fallback prompt carries the approved spec and exactly one
target_shapes.yaml recipe — never a shipped task's name or a "closest known task shape"
picked from a knowledge base of examples that no longer exists.
"""

from adaptrna_agentic.codegen import prompts

SPEC = {
    "target_type": "binary",
    "task_name": "my_task",
    "tool_description": "detect X in sequences",
    "sequence_column": "sequence",
    "label_column": "label",
    "path": "/data/train.csv",
    "format": {"separator": ",", "compression": None},
    "classes": ["0", "1"],
    "positive_class": "1",
    "head": {"primary_metric": "test/f1_score"},
    "split": {
        "mode": "random", "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "seed": 42, "stratify": True,
    },
}

COLUMN_SPEC = {
    **SPEC,
    "task_name": "grouped_task",
    "split": {"mode": "column", "column": "source", "mapping": {"train": ["human"], "test": ["fly"]}},
}

#: Forbidden per D1/D11 — none of these strings may appear in a fallback prompt, whose
#: only knowledge of "what to build" comes from the approved spec and the target-type
#: recipe, never a shipped task's identity.
_FORBIDDEN = ("splice_site", "mrl", "sec_struct", "ncrna_classification", "Spliceator", "bpRNA")


def test_task_prompt_carries_everything_the_generator_needs():
    text = prompts.task_user_prompt(SPEC)

    # the contract
    assert "build_head(embed_dim, **head_config)" in text
    assert "extract_features" in text
    assert "ADAPTER_EXTRA_PREFIXES" in text
    # the silent-failure rules, both halves
    assert "adapter_extra_payload()" in text
    assert "CLS, EOS or padded positions" in text
    # the approved spec
    assert "/data/train.csv" in text
    assert "detect X in sequences" in text
    assert '"target_type": "binary"' in text
    # exactly one recipe, for the approved target type
    assert text.count("# Recipe for a '") == 1
    assert "Recipe for a 'binary' target" in text
    # the split policy, spelled out
    assert "Random split with fractions" in text
    assert "seed 42" in text
    # the worked example — the template's own rendered output, no RNA task identity
    assert "@register_task(\"worked_example\")" in text
    # the import rule that makes staged code and landed code identical
    assert "adaptrna_custom.tasks.<task_name>.datamodule" in text


def test_task_prompt_names_no_shipped_task():
    text = prompts.task_user_prompt(SPEC)

    for forbidden in _FORBIDDEN:
        assert forbidden not in text, f"'{forbidden}' leaked into the fallback prompt"


def test_split_instructions_cover_column_mode():
    text = prompts.task_user_prompt(COLUMN_SPEC)

    assert "Column mode" in text
    assert "'source'" in text
    assert '"train": ["human"]' in text


def test_hard_requirements_name_the_spec_driven_checks():
    text = prompts.task_user_prompt(SPEC)

    assert "spec[\"sequence_column\"]" in text
    assert "spec[\"split\"]" in text
    assert 'PRIMARY_METRIC' in text and 'spec["head"]["primary_metric"]' in text
    assert 'spec["on_invalid"]' in text


def test_recipe_section_matches_the_target_shape():
    text = prompts.recipe_section(SPEC)

    assert "binary_cross_entropy_with_logits" in text
    assert "positive_class" in text  # the silent-failure trap for this shape


def test_worked_example_covers_all_three_target_types():
    """The template's own output, rendered for a shape close to whatever spec is at
    hand — proven here to actually work for all three, since the fallback prompt could
    ask for any of them."""
    from adaptrna_agentic.codegen.templates import render as templates

    base = {
        "task_name": "worked_example", "tool_description": "x",
        "sequence_column": "sequence", "label_column": "label",
        "path": "/tmp/x.csv", "format": {"separator": ",", "compression": None},
        "split": {"mode": "random", "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
                  "seed": 42, "stratify": True},
    }
    cases = [
        {**base, "target_type": "binary", "classes": ["0", "1"], "positive_class": "1",
         "head": {"primary_metric": "test/f1_score"}},
        {**base, "target_type": "multiclass", "classes": ["0", "1", "2"],
         "head": {"primary_metric": "test/macro_f1"}},
        {**base, "target_type": "regression", "head": {"primary_metric": "test/mse"}},
    ]
    for spec in cases:
        assert templates.covers(spec), spec["target_type"]
        assert templates.render(spec)  # does not raise


def test_feedback_is_appended_for_a_retry():
    text = prompts.task_user_prompt(SPEC, feedback="round trip failed")

    assert "previous attempt failed verification" in text
    assert "round trip failed" in text
    assert "Keep everything that worked" in text


def test_verifier_prompt_shows_code_harness_and_checklist():
    text = prompts.verifier_user_prompt(
        "detect X", SPEC, {"task.py": "print('hi')"}, "task 'x': PASS"
    )

    assert "print('hi')" in text
    assert "task 'x': PASS" in text
    assert "Your checklist" in text
    assert "comfortable with these numbers in a paper" in text


def test_verifier_prompt_excludes_spec_json_from_the_code_listing():
    text = prompts.verifier_user_prompt(
        "detect X", SPEC, {"task.py": "print('hi')", "spec.json": "{}"}, "task 'x': PASS"
    )

    assert "--- spec.json" not in text


def test_verifier_prompt_asks_the_narrower_question_when_rendered():
    generated = prompts.verifier_user_prompt(
        "detect X", SPEC, {"task.py": "x"}, "PASS", rendered=False
    )
    rendered = prompts.verifier_user_prompt(
        "detect X", SPEC, {"task.py": "x"}, "PASS", rendered=True
    )

    assert "rendered deterministically" in rendered
    assert "does this code do what this spec says" in rendered
    assert "rendered deterministically" not in generated


def test_verifier_is_told_not_to_redo_the_harness():
    system = prompts.verifier_system_prompt()

    assert "fresh context" in system
    assert "do not re-litigate what the harness" in system


def test_no_prompt_function_names_a_shipped_task():
    text = "\n".join([
        prompts.task_system_prompt(),
        prompts.task_user_prompt(SPEC),
        prompts.verifier_system_prompt(),
        prompts.verifier_user_prompt("d", SPEC, {"task.py": "x"}, "PASS"),
    ])

    for forbidden in _FORBIDDEN:
        assert forbidden not in text


def test_external_tool_prompt_carries_the_contract_and_reference():
    text = prompts.external_tool_prompt("ViennaRNA", "fold RNA")

    assert "ExternalToolSpec" in text
    assert "SPEC" in text
    assert "Validate inputs BEFORE importing" in text
    assert "never invented numbers" in text
