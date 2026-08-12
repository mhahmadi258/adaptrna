"""DataProfiler: describe a dataset, and say which shipped task can train on it.

Deterministic and LLM-free (MASTER_PLAN §3.1). At the boundary of what the engine can
read, the profiler's job is to be *clear* rather than to improvise: it reports the shape
it sees, names the nearest task template, and states plainly when a new datamodule would
be needed — silently reshaping a user's file into some upstream layout is how you get
confidently wrong numbers later (plan §2).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import gzip

from adaptrna_agentic.knowledge import load_knowledge, templates

#: Column names that usually hold the sequence / the target.
_SEQUENCE_HINTS = ("seq", "sequence", "utr", "rna", "dna")
_TARGET_HINTS = ("rl", "label", "target", "y", "value", "score", "class")

_NUCLEOTIDES = set("ACGTUN")

#: A column is treated as a sequence column when at least this fraction of its sampled
#: values are nucleotide-only strings of length >= 10.
_SEQUENCE_PURITY = 0.9

_SAMPLE_ROWS = 2000


def profile_dataset(path: Union[str, Path]) -> Dict[str, Any]:
    """Profile a file or directory. Returns a plain dict (agent-tool ready)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No such dataset path: '{path}'")

    profile: Dict[str, Any] = {"path": str(path)}

    if path.is_dir():
        profile.update(_profile_directory(path))
    else:
        profile.update(_profile_file(path))

    match, reason = _match_layout(path, profile)
    profile["layout_match"] = match
    profile["layout_reason"] = reason
    if match is None:
        profile["guidance"] = load_knowledge()["no_match_guidance"].strip()

    return profile


# ---------------------------------------------------------------------- directories

def _profile_directory(path: Path) -> Dict[str, Any]:
    entries = sorted(p.name for p in path.iterdir())
    info: Dict[str, Any] = {
        "kind": "directory",
        "entries": entries[:40],
        "entry_count": len(entries),
    }

    # Spliceator layout: <root>/GS_1/db_N/{Train,Val}_{type}_400.csv
    folds_root = path / "GS_1"
    if folds_root.is_dir():
        folds = sorted(p.name for p in folds_root.iterdir() if p.is_dir())
        info["folds"] = folds
        sample = folds_root / (folds[0] if folds else "")
        if sample.is_dir():
            csvs = sorted(p.name for p in sample.glob("*.csv"))
            info["fold_files"] = csvs
            info["ss_types"] = sorted(
                {name.split("_")[1] for name in csvs if "_" in name}
            )
            first = sample / csvs[0] if csvs else None
            if first is not None:
                info["splits"] = _spliceator_split_sizes(sample, csvs)
                info.update(_sequence_stats_from_spliceator(first))

    # MRL layout: a gzipped CSV with the known name
    mrl_files = sorted(p.name for p in path.glob("*.csv.gz"))
    if mrl_files:
        info["gzipped_csvs"] = mrl_files

    return info


def _spliceator_split_sizes(fold_dir: Path, csvs: List[str]) -> Dict[str, int]:
    sizes = {}
    for name in csvs:
        with open(fold_dir / name) as handle:
            sizes[name] = sum(1 for _ in handle)
    return sizes


def _sequence_stats_from_spliceator(csv_path: Path) -> Dict[str, Any]:
    """Spliceator folds are ';'-separated with no header: group;sequence;label."""
    import pandas as pd

    frame = pd.read_csv(csv_path, sep=";", header=None, nrows=_SAMPLE_ROWS)
    if frame.shape[1] < 3:
        return {}

    sequences = frame[1].astype(str)
    labels = frame[2]

    return {
        "sequence_column": "column 1 (';'-separated, headerless)",
        "target_column": "column 2",
        "target_type": _target_type(labels),
        "target_summary": _target_summary(labels),
        **_length_stats(sequences),
        "alphabet": _alphabet(sequences),
    }


# ---------------------------------------------------------------------- files

def _profile_file(path: Path) -> Dict[str, Any]:
    suffixes = "".join(path.suffixes).lower()

    if any(s in suffixes for s in (".csv", ".tsv", ".txt")):
        return _profile_table(path)
    if any(s in suffixes for s in (".fa", ".fasta")):
        return _profile_fasta(path)

    return {"kind": "unknown", "size_bytes": path.stat().st_size}


def _profile_table(path: Path) -> Dict[str, Any]:
    import pandas as pd

    separator = "\t" if ".tsv" in "".join(path.suffixes).lower() else ","
    frame = pd.read_csv(path, sep=separator, nrows=_SAMPLE_ROWS)

    info: Dict[str, Any] = {
        "kind": "table",
        "size_bytes": path.stat().st_size,
        "compressed": path.suffix == ".gz",
        "columns": [str(c) for c in frame.columns][:40],
        "sampled_rows": int(len(frame)),
        "missing_values": {
            str(c): int(frame[c].isna().sum())
            for c in frame.columns
            if frame[c].isna().any()
        },
    }

    sequence_column = _detect_sequence_column(frame)
    if sequence_column is not None:
        sequences = frame[sequence_column].astype(str)
        info["sequence_column"] = str(sequence_column)
        info.update(_length_stats(sequences))
        info["alphabet"] = _alphabet(sequences)

    target_column = _detect_target_column(frame, sequence_column)
    if target_column is not None:
        info["target_column"] = str(target_column)
        info["target_type"] = _target_type(frame[target_column])
        info["target_summary"] = _target_summary(frame[target_column])

    return info


def _profile_fasta(path: Path) -> Dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    sequences, current = [], []

    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            if line:
                current.append(line)
            if len(sequences) >= _SAMPLE_ROWS:
                break
    if current:
        sequences.append("".join(current))

    import pandas as pd

    series = pd.Series(sequences, dtype=str)
    return {
        "kind": "fasta",
        "size_bytes": path.stat().st_size,
        "sampled_rows": int(len(series)),
        "sequence_column": "FASTA records",
        **_length_stats(series),
        "alphabet": _alphabet(series),
    }


# ---------------------------------------------------------------------- helpers

def _looks_like_sequences(series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False

    hits = sum(
        1 for value in sample
        if len(value) >= 10 and set(value.upper()) <= _NUCLEOTIDES
    )
    return hits / len(sample) >= _SEQUENCE_PURITY


def _detect_sequence_column(frame) -> Optional[str]:
    # Name hints first, then fall back to content sniffing.
    for column in frame.columns:
        if str(column).lower() in _SEQUENCE_HINTS and _looks_like_sequences(frame[column]):
            return column

    # `dtype == object` is not portable: pandas 3 gives string columns a `str` dtype.
    # Skipping numeric/bool kinds and sniffing the rest works on both 2.x and 3.x.
    for column in frame.columns:
        if frame[column].dtype.kind not in "ifbc" and _looks_like_sequences(frame[column]):
            return column

    return None


def _detect_target_column(frame, sequence_column) -> Optional[str]:
    candidates = [c for c in frame.columns if c != sequence_column]

    for column in candidates:
        if str(column).lower() in _TARGET_HINTS:
            return column

    numeric = [
        c for c in candidates
        if frame[c].dtype.kind in "if" and not str(c).lower().startswith("unnamed")
    ]
    return numeric[0] if numeric else None


def _target_type(series) -> str:
    values = series.dropna()
    if values.empty:
        return "unknown"

    unique = values.unique()
    if values.dtype.kind in "if":
        if len(unique) == 2:
            return "binary"
        if len(unique) <= 20 and all(float(v).is_integer() for v in unique):
            return "multiclass"
        return "continuous"

    return "binary" if len(unique) == 2 else "categorical"


def _target_summary(series) -> Dict[str, Any]:
    values = series.dropna()
    if values.empty:
        return {}

    if values.dtype.kind in "if" and len(values.unique()) > 20:
        return {
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "mean": round(float(values.mean()), 4),
        }

    counts = values.value_counts().head(10)
    return {"class_counts": {str(k): int(v) for k, v in counts.items()}}


def _length_stats(series) -> Dict[str, Any]:
    lengths = series.dropna().astype(str).str.len()
    if lengths.empty:
        return {}

    return {
        "length_min": int(lengths.min()),
        "length_median": int(lengths.median()),
        "length_max": int(lengths.max()),
    }


def _alphabet(series) -> str:
    letters = set()
    for value in series.dropna().astype(str).head(200):
        letters |= set(value.upper())

    if not letters:
        return "unknown"
    if letters <= _NUCLEOTIDES:
        return "dna" if "T" in letters and "U" not in letters else "rna"
    return "other"


# ---------------------------------------------------------------------- layout match

def _match_layout(path: Path, profile: Dict[str, Any]):
    """Which shipped task can read this on disk? Returns (task_name | None, reason)."""
    for template in templates():
        task = template["task"]
        required = template["data_layout"].get("required_paths", [])
        root = path if path.is_dir() else path.parent

        if all((root / name).exists() for name in required):
            return task, (
                f"Found {required} under '{root}' — the layout the '{task}' task reads "
                f"({template['label']})."
            )

    nearest = _nearest_template(profile)
    if nearest:
        return None, (
            f"No shipped task reads this layout. The closest shape is '{nearest['task']}' "
            f"({nearest['label']}), which expects: "
            f"{nearest['data_layout']['description']}"
        )

    return None, "No shipped task reads this layout, and no template matches its shape."


def _nearest_template(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target_type = profile.get("target_type")
    median = profile.get("length_median")
    if target_type is None:
        return None

    best, best_score = None, -1.0
    for template in templates():
        match = template["match"]
        score = 0.0

        if match.get("target_type") == target_type:
            score += 2.0
        elif target_type in ("binary", "multiclass") and match.get("target_type") == "binary":
            score += 1.0

        if median is not None and "typical_length" in match:
            typical = match["typical_length"]
            score += max(0.0, 1.0 - abs(median - typical) / max(typical, 1))

        if score > best_score:
            best, best_score = template, score

    return best if best_score > 0 else None
