"""Golden-file tests for the deterministic template renderer (Phase 13 §7.3/§13).

Each of the three target types x two split modes renders byte-for-byte expected output.
Cheap, fast, no model call — and the thing that makes a template change reviewable as a
diff (regenerate the golden files and read the diff, rather than eyeballing rendered code
from scratch).
"""

from pathlib import Path

import pytest

from adaptrna_agentic.codegen.harness import summarize, verify_task
from adaptrna_agentic.codegen.staging import stage_task
from adaptrna_agentic.codegen.templates import render as templates
from fixtures.template_specs import TEMPLATE_SPECS

GOLDEN_ROOT = Path(__file__).parent / "fixtures" / "golden" / "templates"
FIXTURE_DATA = Path(__file__).parent / "fixtures" / "data"

SEQS = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]


@pytest.mark.parametrize("case", sorted(TEMPLATE_SPECS))
def test_render_matches_golden_files(case):
    spec = TEMPLATE_SPECS[case]
    assert templates.covers(spec), f"template should cover {case}"

    rendered = templates.render(spec)
    golden_dir = GOLDEN_ROOT / case

    for filename, content in rendered.items():
        expected = (golden_dir / filename).read_text()
        assert content == expected, (
            f"{case}/{filename} does not match its golden file — if this change is "
            f"intentional, regenerate the golden files and review the diff"
        )


def test_render_refuses_a_spec_it_does_not_cover():
    spec = dict(TEMPLATE_SPECS["binary_random"])
    spec["target_type"] = "per_position"  # not one of binary/multiclass/regression

    assert not templates.covers(spec)
    with pytest.raises(ValueError):
        templates.render(spec)


@pytest.mark.parametrize("target_type,csv_name,classes,metric", [
    ("binary", "binary.csv", ["0", "1"], "test/f1_score"),
    ("multiclass", "multiclass.csv", ["0", "1", "2"], "test/macro_f1"),
    ("regression", "regression.csv", None, "test/mse"),
])
def test_rendered_task_passes_the_full_harness_with_no_model_call(
    tmp_path, target_type, csv_name, classes, metric
):
    """The declared case never reaches a model (Phase 13 §7.2/§18): render, then verify
    against real data. Neither `render()` nor `verify_task()` imports an LLM client at
    all, so there is nothing here that could call one."""
    task_name = f"rendered_{target_type}"
    spec = {
        "target_type": target_type,
        "task_name": task_name,
        "tool_description": f"{target_type} target rendered from the template",
        "sequence_column": "sequence",
        "label_column": "label",
        "path": str((FIXTURE_DATA / csv_name).resolve()),
        "format": {"separator": ",", "compression": None},
        "classes": classes,
        "head": {"primary_metric": metric},
        "split": {
            "mode": "random",
            "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
            "stratify": True,
        },
    }
    assert templates.covers(spec)

    files = templates.render(spec)
    staged = stage_task(task_name, files, data_dir=tmp_path / "toolhub_data")

    report = verify_task(
        task_name,
        task_module=staged.module_path,
        config_path=str(staged.config_path),
        sys_path=[str(staged.root)],
        sequences=SEQS,
        timeout=600,
    )

    assert report["ok"], summarize(report)
