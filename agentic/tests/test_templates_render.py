"""Golden-file tests for the deterministic template renderer (Phase 13 §7.3/§13).

Each of the three target types x two split modes renders byte-for-byte expected output.
Cheap, fast, no model call — and the thing that makes a template change reviewable as a
diff: after changing anything under `codegen/templates/`, run
`python agentic/scripts/update_template_golden.py` and read the diff before trusting it.
"""

from pathlib import Path

import pytest

from adaptrna_agentic.codegen.harness import summarize, verify_task
from adaptrna_agentic.codegen.staging import stage_task
from adaptrna_agentic.codegen.templates import render as templates
from fixtures import broken_task_sources as broken_sources
from fixtures.template_specs import TEMPLATE_SPECS

GOLDEN_ROOT = Path(__file__).parent / "fixtures" / "golden" / "templates"
FIXTURE_DATA = Path(__file__).parent / "fixtures" / "data"

SEQS = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]


def _rendered_spec(target_type, csv_name, **overrides):
    spec = {
        "target_type": target_type,
        "task_name": f"rendered_{target_type}",
        "tool_description": f"{target_type} target rendered from the template",
        "sequence_column": "sequence",
        "label_column": "label",
        "path": str((FIXTURE_DATA / csv_name).resolve()),
        "format": {"separator": ",", "compression": None},
        "split": {
            "mode": "random",
            "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
            "stratify": True,
        },
    }
    spec.update(overrides)
    return spec


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


@pytest.mark.parametrize("target_type,csv_name,extra", [
    ("binary", "binary.csv", {"classes": ["0", "1"], "positive_class": "1",
                              "head": {"primary_metric": "test/f1_score"}}),
    ("multiclass", "multiclass.csv", {"classes": ["0", "1", "2"],
                                      "head": {"primary_metric": "test/macro_f1"}}),
    ("regression", "regression.csv", {"head": {"primary_metric": "test/mse"}}),
])
def test_rendered_task_passes_the_full_harness_with_no_model_call(
    tmp_path, target_type, csv_name, extra
):
    """The declared case never reaches a model (Phase 13 §7.2/§18): render, then verify
    against real data. Neither `render()` nor `verify_task()` imports an LLM client at
    all, so there is nothing here that could call one."""
    spec = _rendered_spec(target_type, csv_name, **extra)
    assert templates.covers(spec)

    files = templates.render(spec)
    staged = stage_task(spec["task_name"], files, data_dir=tmp_path / "toolhub_data")

    report = verify_task(
        spec["task_name"],
        task_module=staged.module_path,
        config_path=str(staged.config_path),
        sys_path=[str(staged.root)],
        sequences=SEQS,
        timeout=600,
    )

    assert report["ok"], summarize(report)


def test_multiclass_prediction_returns_the_class_label_not_an_index(tmp_path):
    """Fix 1: `postprocess_predictions` must answer with the label the human approved at
    gate 1 ('intron'), not its numeric position in `classes` (1) — the served tool's
    answer has to make sense on its own."""
    spec = _rendered_spec(
        "multiclass", "multiclass.csv",
        classes=["exon", "intron", "utr"],
        head={"primary_metric": "test/macro_f1"},
    )
    # Relabel the fixture's 0/1/2 column into these class names so the rendered task
    # actually sees them.
    import pandas as pd

    raw = pd.read_csv(FIXTURE_DATA / "multiclass.csv")
    raw["label"] = raw["label"].map({0: "exon", 1: "intron", 2: "utr"})
    relabelled = tmp_path / "multiclass_named.csv"
    raw.to_csv(relabelled, index=False)
    spec["path"] = str(relabelled)

    assert templates.covers(spec)
    files = templates.render(spec)
    staged = stage_task(spec["task_name"], files, data_dir=tmp_path / "toolhub_data")

    report = verify_task(
        spec["task_name"],
        task_module=staged.module_path,
        config_path=str(staged.config_path),
        sys_path=[str(staged.root)],
        sequences=SEQS,
        timeout=600,
    )
    assert report["ok"], summarize(report)

    # check_serving proves predictions flow through RiNALMoHub; run a small script in a
    # *fresh* subprocess (not in-process — `adaptrna_custom` may already be cached in
    # sys.modules from other tests, which would shadow the staged copy) to inspect the
    # actual shape of postprocess_predictions' output.
    import json
    import subprocess
    import sys

    script = f"""
import json, sys
sys.path.insert(0, {str(staged.root)!r})
import rinalmo_hub.tasks  # noqa: F401
import importlib, torch
from rinalmo.data.alphabet import Alphabet

module = importlib.import_module({staged.module_path!r})
task_cls = next(
    v for v in vars(module).values()
    if isinstance(v, type) and getattr(v, "TASK_NAME", None) == {spec["task_name"]!r}
)
instance = task_cls(lm_config="nano", head_config={{"hidden_dim": 32}})
instance.eval()
tokens = torch.tensor(Alphabet().batch_tokenize({SEQS!r}), dtype=torch.long)
with torch.no_grad():
    outputs = instance(tokens)
predictions = instance.postprocess_predictions(outputs, tokens, {SEQS!r})
print(json.dumps(predictions))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    predictions = json.loads(result.stdout.strip().splitlines()[-1])

    assert len(predictions) == len(SEQS)
    for prediction in predictions:
        assert prediction["label"] in ("exon", "intron", "utr")
        assert set(prediction["probabilities"]) == {"exon", "intron", "utr"}


def test_regression_missing_scaler_prefix_is_caught_by_check_six(tmp_path):
    """Fix 2's catch test: the scaler buffers exist, but if `ADAPTER_EXTRA_PREFIXES` ever
    stopped declaring them, check 6 (adapter_roundtrip) must fail — not pass on an empty
    set of adapter-owned state."""
    spec = _rendered_spec("regression", "regression.csv", head={"primary_metric": "test/mse"})
    files = templates.render(spec)

    broken_task = broken_sources.drop_adapter_extra_prefix(files["task.py"], '("scaler.",)')
    assert broken_task != files["task.py"]  # sanity: the substitution actually matched

    staged = stage_task(
        spec["task_name"], {**files, "task.py": broken_task},
        data_dir=tmp_path / "toolhub_data",
    )

    report = verify_task(
        spec["task_name"],
        task_module=staged.module_path,
        config_path=str(staged.config_path),
        sys_path=[str(staged.root)],
        sequences=SEQS,
        timeout=600,
    )

    assert not report["ok"]
    failed = {c["name"] for c in report["checks"] if c["status"] == "fail"}
    assert "adapter_roundtrip" in failed


def test_validate_split_raises_on_an_unexpectedly_empty_random_split():
    """Fix 4, random mode: `_validate_split` (rendered into the datamodule) must raise
    when a split with a non-zero requested fraction comes out empty.

    Exercised directly against the rendered function rather than through a real dataset:
    sklearn's own `train_test_split` already refuses to silently produce an empty split
    from too little data (it raises its own, differently-worded error first), which makes
    that path hard to reach honestly through real data — but a future edit to the split
    strategy could easily lose that guarantee, which is exactly what this validator is a
    backstop against.
    """
    spec = _rendered_spec(
        "binary", "binary.csv",
        classes=["0", "1"], positive_class="1",
        head={"primary_metric": "test/f1_score"},
    )
    datamodule_source = templates.render(spec)["datamodule.py"]

    namespace: dict = {}
    exec(compile(datamodule_source, "<rendered datamodule>", "exec"), namespace)

    import pandas as pd

    non_empty = pd.DataFrame({"sequence": ["ACGU"], "label": ["0"]})
    empty = pd.DataFrame({"sequence": [], "label": []})

    with pytest.raises(ValueError, match="empty"):
        namespace["_validate_split"](non_empty, empty, non_empty)


def test_empty_column_split_raises(tmp_path):
    """Fix 4, column mode: a mapping value that matches nothing in the data must raise,
    not silently produce a 0-row split."""
    spec = _rendered_spec(
        "binary", "binary.csv",
        classes=["0", "1"], positive_class="1",
        head={"primary_metric": "test/f1_score"},
    )
    spec["split"] = {
        "mode": "column", "column": "grp",
        "mapping": {"train": ["human"], "val": ["human"], "test": ["typo_value"]},
    }
    csv_with_group = tmp_path / "grouped.csv"
    raw_lines = (FIXTURE_DATA / "binary.csv").read_text().strip().splitlines()
    header, rows = raw_lines[0], raw_lines[1:]
    csv_with_group.write_text(
        "\n".join([f"{header},grp"] + [f"{row},human" for row in rows]) + "\n"
    )
    spec["path"] = str(csv_with_group)

    files = templates.render(spec)
    staged = stage_task(spec["task_name"], files, data_dir=tmp_path / "toolhub_data")

    report = verify_task(
        spec["task_name"],
        task_module=staged.module_path,
        config_path=str(staged.config_path),
        sys_path=[str(staged.root)],
        sequences=SEQS,
        timeout=600,
    )

    assert not report["ok"]
    detail = next(c["detail"] for c in report["checks"] if c["name"] == "datamodule")
    assert "empty" in detail
    assert "test" in detail


def test_binary_positive_class_is_independent_of_class_order(tmp_path):
    """Fix 5: `classes: ["1", "0"]` must not silently flip which class is positive —
    polarity comes only from `positive_class`."""
    reversed_order = _rendered_spec(
        "binary", "binary.csv",
        classes=["1", "0"], positive_class="1",
        head={"primary_metric": "test/f1_score"},
    )
    natural_order = _rendered_spec(
        "binary", "binary.csv",
        classes=["0", "1"], positive_class="1",
        head={"primary_metric": "test/f1_score"},
    )
    reversed_order["task_name"] = "polarity_reversed"
    natural_order["task_name"] = "polarity_natural"

    for spec in (reversed_order, natural_order):
        assert templates.covers(spec)
        datamodule_source = templates.render(spec)["datamodule.py"]
        assert 'POSITIVE_CLASS = \'1\'' in datamodule_source
        # The label-encoding line must key off POSITIVE_CLASS, never off CLASSES[...] or
        # a positional index into `classes`.
        assert "value == POSITIVE_CLASS" in datamodule_source
