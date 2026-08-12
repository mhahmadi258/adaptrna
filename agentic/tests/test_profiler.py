"""DataProfiler against synthetic fixtures — deterministic, no engine, no real datasets."""

import gzip
import random

import pytest

from adaptrna_agentic.profiling.profiler import profile_dataset

random.seed(0)


def _sequence(length: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(length))


@pytest.fixture
def mrl_layout(tmp_path):
    """A directory holding the gzipped CSV the engine's MRL datamodule reads."""
    root = tmp_path / "mrl_data"
    root.mkdir()
    rows = ["utr,rl,set,total_reads"]
    for i in range(50):
        rows.append(f"{_sequence(random.randint(25, 100))},{4.0 + i * 0.01},random,{100 + i}")

    with gzip.open(root / "GSM4084997_varying_length_25to100.csv.gz", "wt") as handle:
        handle.write("\n".join(rows))

    return root


@pytest.fixture
def spliceator_layout(tmp_path):
    """The Spliceator fold layout: <root>/GS_1/db_N/{Train,Val}_{type}_400.csv."""
    root = tmp_path / "train_data"
    fold = root / "GS_1" / "db_1"
    fold.mkdir(parents=True)

    for split in ("Train", "Val"):
        for ss_type in ("donor", "acceptor"):
            rows = [
                f"group{i};{_sequence(400)};{i % 2}"
                for i in range(40 if split == "Train" else 10)
            ]
            (fold / f"{split}_{ss_type}_400.csv").write_text("\n".join(rows) + "\n")

    return root


def test_mrl_layout_is_matched(mrl_layout):
    profile = profile_dataset(mrl_layout)

    assert profile["layout_match"] == "mrl"
    assert "GSM4084997" in profile["layout_reason"]
    assert profile["gzipped_csvs"] == ["GSM4084997_varying_length_25to100.csv.gz"]


def test_mrl_csv_profiled_directly(mrl_layout):
    profile = profile_dataset(mrl_layout / "GSM4084997_varying_length_25to100.csv.gz")

    assert profile["kind"] == "table"
    assert profile["sequence_column"] == "utr"
    assert profile["target_column"] == "rl"
    assert profile["target_type"] == "continuous"
    assert 25 <= profile["length_median"] <= 100
    assert profile["alphabet"] == "dna"
    assert profile["layout_match"] == "mrl"       # sibling file satisfies the layout


def test_spliceator_layout_is_matched_with_folds(spliceator_layout):
    profile = profile_dataset(spliceator_layout)

    assert profile["layout_match"] == "splice_site"
    assert profile["folds"] == ["db_1"]
    assert set(profile["ss_types"]) == {"donor", "acceptor"}
    assert profile["target_type"] == "binary"
    assert profile["length_median"] == 400
    assert profile["splits"]["Train_donor_400.csv"] == 40


def test_unknown_schema_names_the_nearest_template(tmp_path):
    path = tmp_path / "mystery.csv"
    rows = ["sequence,affinity"] + [f"{_sequence(60)},{i * 0.5}" for i in range(30)]
    path.write_text("\n".join(rows))

    profile = profile_dataset(path)

    assert profile["layout_match"] is None
    assert "No shipped task reads this layout" in profile["layout_reason"]
    assert "mrl" in profile["layout_reason"]           # nearest by shape: continuous target
    assert "three files" in profile["guidance"]        # honest about what is missing


def test_sequence_and_target_detection_by_content(tmp_path):
    path = tmp_path / "unnamed.csv"
    rows = ["col_a,col_b"] + [f"{_sequence(80)},{i % 2}" for i in range(30)]
    path.write_text("\n".join(rows))

    profile = profile_dataset(path)

    assert profile["sequence_column"] == "col_a"      # sniffed, not named
    assert profile["target_column"] == "col_b"
    assert profile["target_type"] == "binary"
    assert profile["target_summary"]["class_counts"]


def test_rna_alphabet_detected(tmp_path):
    path = tmp_path / "rna.csv"
    rows = ["sequence,y"] + [f"{'ACGU' * 20},{i}" for i in range(10)]
    path.write_text("\n".join(rows))

    assert profile_dataset(path)["alphabet"] == "rna"


def test_missing_values_reported(tmp_path):
    path = tmp_path / "gappy.csv"
    rows = ["sequence,rl", f"{_sequence(50)},", f"{_sequence(50)},3.2"]
    path.write_text("\n".join(rows))

    profile = profile_dataset(path)

    assert profile["missing_values"].get("rl") == 1


def test_fasta_profiled(tmp_path):
    path = tmp_path / "windows.fasta"
    records = [f">seq{i}\n{_sequence(400)}" for i in range(5)]
    path.write_text("\n".join(records) + "\n")

    profile = profile_dataset(path)

    assert profile["kind"] == "fasta"
    assert profile["sampled_rows"] == 5
    assert profile["length_median"] == 400


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError, match="No such dataset path"):
        profile_dataset("/definitely/not/here.csv")
