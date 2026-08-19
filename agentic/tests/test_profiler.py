"""`profile_dataset` against synthetic fixtures — deterministic, no engine, no real
datasets, no layout matching (Phase 13). One CSV/TSV in; a proposed `DatasetSpec` out."""

import gzip
import random

import pytest

from adaptrna_agentic.profiling.profiler import PROFILE_SOURCE, profile_dataset
from adaptrna_agentic.toolhub.errors import ToolHubError

random.seed(0)


def _sequence(length: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(length))


def _binary_rows(n=40, length=32):
    return ["sequence,label"] + [f"{_sequence(length)},{i % 2}" for i in range(n)]


# ---------------------------------------------------------------------- accepted input

def test_profile_is_stamped_and_versioned(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\n".join(_binary_rows()) + "\n")

    spec = profile_dataset(path)

    assert spec["source"] == PROFILE_SOURCE
    assert spec["spec_version"] == 2
    assert spec["path"] == str(path.resolve())


def test_sequence_and_label_detected_by_name(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\n".join(_binary_rows()) + "\n")

    spec = profile_dataset(path)

    assert spec["sequence_column"] == "sequence"
    assert spec["label_column"] == "label"
    assert spec["target_type"] == "binary"
    assert spec["classes"] == ["0", "1"]
    assert spec["class_counts"] == {"0": 20, "1": 20}
    assert spec["positive_class"] in spec["classes"]


def test_sequence_and_label_detected_by_content(tmp_path):
    path = tmp_path / "unnamed.csv"
    rows = ["col_a,col_b"] + [f"{_sequence(80)},{i % 2}" for i in range(30)]
    path.write_text("\n".join(rows))

    spec = profile_dataset(path)

    assert spec["sequence_column"] == "col_a"
    assert spec["label_column"] == "col_b"
    assert spec["target_type"] == "binary"


def test_tsv_is_accepted(tmp_path):
    path = tmp_path / "data.tsv"
    rows = ["sequence\tlabel"] + [f"{_sequence(32)}\t{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert spec["format"]["separator"] == "\t"
    assert spec["target_type"] == "binary"


def test_gzipped_csv_is_accepted(tmp_path):
    path = tmp_path / "data.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(_binary_rows()) + "\n")

    spec = profile_dataset(path)

    assert spec["format"]["compression"] == "gzip"
    assert spec["target_type"] == "binary"


def test_headerless_csv_is_detected_and_warned(tmp_path):
    path = tmp_path / "data.csv"
    rows = [f"{_sequence(32)},{i % 2}" for i in range(40)]  # no header row at all
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert spec["format"]["header"] is False
    assert spec["sequence_column"] == "0"
    assert spec["label_column"] == "1"
    assert spec["target_type"] == "binary"
    assert any("No header row detected" in w for w in spec["warnings"])


def test_a_normal_header_is_not_flagged_as_missing(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\n".join(_binary_rows()) + "\n")

    spec = profile_dataset(path)

    assert spec["format"]["header"] is True
    assert not any("No header row detected" in w for w in spec["warnings"])


def test_semicolon_delimiter_is_sniffed(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence;label"] + [f"{_sequence(32)};{i % 2}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert spec["format"]["separator"] == ";"
    assert spec["target_type"] == "binary"


def test_sniffing_never_reads_more_than_a_few_lines(tmp_path):
    """Delimiter sniffing must stay cheap on a huge file. The first 10 lines are
    well-formed and semicolon-delimited; everything after is one massive undelimited
    line -- if the sniffer ever pulled it into its sample, `csv.Sniffer` would find no
    consistent delimiter and this would fall back to ',' instead of ';', so a wrong
    result here is a real correctness signal, not just a timing one."""
    from adaptrna_agentic.profiling.profiler import _sniff_separator

    path = tmp_path / "huge.csv"
    lines = [f"{_sequence(32)};{i % 2}" for i in range(10)]
    lines.append("X" * 2_000_000)
    path.write_text("\n".join(lines) + "\n")

    assert _sniff_separator(path, compression=None) == ";"


# ---------------------------------------------------------------------- multiclass etc.

def test_multiclass_detected(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label"] + [f"{_sequence(32)},{i % 3}" for i in range(30)]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert spec["target_type"] == "multiclass"
    assert spec["classes"] == ["0", "1", "2"]
    assert spec["positive_class"] is None


def test_regression_detected(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,affinity"] + [f"{_sequence(60)},{i * 0.37}" for i in range(30)]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert spec["target_type"] == "regression"
    assert spec["classes"] is None
    assert spec["target_summary"]


def test_rna_alphabet_detected(tmp_path):
    path = tmp_path / "rna.csv"
    rows = ["sequence,label"] + [f"{'ACGU' * 20},{i % 2}" for i in range(10)]
    path.write_text("\n".join(rows))

    assert profile_dataset(path)["alphabet"] == "rna"


def test_task_name_and_description_are_proposed(tmp_path):
    path = tmp_path / "My Donor Sites.csv"
    path.write_text("\n".join(_binary_rows()) + "\n")

    spec = profile_dataset(path)

    assert spec["task_name"].isidentifier()
    assert spec["task_name"] == spec["task_name"].lower()
    assert spec["tool_description"]


def test_head_recipe_matches_target_type(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\n".join(_binary_rows()) + "\n")

    spec = profile_dataset(path)

    assert spec["head"]["primary_metric"] == "test/f1_score"
    assert spec["head"]["loss"] == "binary_cross_entropy_with_logits"


# ---------------------------------------------------------------------- split proposal

def test_default_split_is_random_80_10_10(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\n".join(_binary_rows(n=100)) + "\n")

    spec = profile_dataset(path)

    assert spec["split"]["mode"] == "random"
    assert spec["split"]["fractions"] == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert spec["split"]["seed"] == 42
    assert spec["split"]["stratify"] is True
    counts = spec["split"]["row_counts"]
    assert counts["train"] + counts["val"] + counts["test"] == 100


def test_a_split_name_column_is_proposed_as_column_mode(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label,fold"]
    for i in range(40):
        fold = "train" if i < 30 else "test"
        rows.append(f"{_sequence(32)},{i % 2},{fold}")
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert spec["split"]["mode"] == "column"
    assert spec["split"]["column"] == "fold"
    assert spec["split"]["row_counts"]["train"] == 30
    assert spec["split"]["row_counts"]["test"] == 10
    assert "fold" in spec["split_candidates"]


def test_validation_path_is_proposed_as_file_mode(tmp_path):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train_path.write_text("\n".join(_binary_rows(n=40)) + "\n")
    val_path.write_text("\n".join(_binary_rows(n=10)) + "\n")

    spec = profile_dataset(train_path, validation_path=val_path)

    assert spec["split"]["mode"] == "file"
    assert spec["split"]["validation_path"] == str(val_path.resolve())
    assert spec["split"]["test_fraction"] == 0.0
    assert spec["split"]["row_counts"] == {"train": 40, "val": 10, "test": 0}
    assert spec["split"]["dropped_rows"] == 0


def test_file_mode_cross_file_leakage_is_warned(tmp_path):
    shared = _sequence(32)
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train_path.write_text("\n".join(["sequence,label", f"{shared},0"] + _binary_rows(n=39)[1:]) + "\n")
    val_path.write_text("\n".join(["sequence,label", f"{shared},1"] + _binary_rows(n=9)[1:]) + "\n")

    spec = profile_dataset(train_path, validation_path=val_path)

    assert any("train and val" in w for w in spec["warnings"])


def test_validation_path_missing_file_is_refused(tmp_path):
    train_path = tmp_path / "train.csv"
    train_path.write_text("\n".join(_binary_rows()) + "\n")

    with pytest.raises(ToolHubError, match="does not exist"):
        profile_dataset(train_path, validation_path=tmp_path / "nowhere.csv")


def test_validation_path_missing_columns_is_refused(tmp_path):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train_path.write_text("\n".join(_binary_rows()) + "\n")
    val_path.write_text("not_sequence,not_label\nACGU,0\n")

    with pytest.raises(ToolHubError, match="must have the same columns"):
        profile_dataset(train_path, validation_path=val_path)


# ---------------------------------------------------------------------- quality checks

def test_duplicate_sequences_are_warned(tmp_path):
    path = tmp_path / "data.csv"
    seq = _sequence(32)
    rows = ["sequence,label", f"{seq},0", f"{seq},1"] + _binary_rows(n=38)[1:]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert any("appear more than once" in w for w in spec["warnings"])


def test_class_imbalance_is_warned(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label"]
    rows += [f"{_sequence(32)},0" for _ in range(95)]
    rows += [f"{_sequence(32)},1" for _ in range(5)]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert any("minority class" in w for w in spec["warnings"])


def test_non_nucleotide_characters_are_warned(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label", f"{_sequence(30)}XX,0"] + _binary_rows(n=39)[1:]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert any("ACGTUN" in w for w in spec["warnings"])
    assert spec["on_invalid"] == "fail"


def test_missing_sequence_values_are_warned(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label", ",0"] + _binary_rows(n=39)[1:]
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert any("missing sequence" in w for w in spec["warnings"])


def test_length_spread_is_warned(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label"] + [f"{_sequence(20)},{i % 2}" for i in range(39)]
    rows.append(f"{_sequence(200)},0")  # one wild outlier
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert any("median" in w and "nt" in w for w in spec["warnings"])


def test_tiny_dataset_is_warned(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("\n".join(_binary_rows(n=10)) + "\n")

    spec = profile_dataset(path)

    assert any("usable rows" in w for w in spec["warnings"])


def test_leakage_across_column_split_is_warned(tmp_path):
    path = tmp_path / "data.csv"
    shared = _sequence(32)
    rows = ["sequence,label,fold", f"{shared},0,train", f"{shared},1,test"]
    for i in range(38):
        fold = "train" if i < 28 else "test"
        rows.append(f"{_sequence(32)},{i % 2},{fold}")
    path.write_text("\n".join(rows) + "\n")

    spec = profile_dataset(path)

    assert spec["split"]["mode"] == "column"
    assert any("train and test" in w for w in spec["warnings"])


# ---------------------------------------------------------------------- refusals (D7, D8)

def test_directory_is_refused(tmp_path):
    with pytest.raises(ToolHubError, match="single table"):
        profile_dataset(tmp_path)


def test_fasta_is_refused(tmp_path):
    path = tmp_path / "windows.fasta"
    path.write_text(">seq1\nACGUACGU\n")

    with pytest.raises(ToolHubError, match="single table"):
        profile_dataset(path)


def test_unknown_suffix_is_refused(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_text("not actually parquet")

    with pytest.raises(ToolHubError, match="single table"):
        profile_dataset(path)


def test_per_position_label_is_refused(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,structure"]
    for _ in range(30):
        seq = _sequence(40)
        rows.append(f"{seq},{'.' * 40}")
    path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ToolHubError, match="[Pp]er-position"):
        profile_dataset(path)


def test_too_many_free_text_classes_is_refused(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["sequence,label"] + [f"{_sequence(30)},class_{i}" for i in range(30)]
    path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ToolHubError, match="binary classification, multiclass"):
        profile_dataset(path)


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError, match="No such dataset path"):
        profile_dataset("/definitely/not/here.csv")
