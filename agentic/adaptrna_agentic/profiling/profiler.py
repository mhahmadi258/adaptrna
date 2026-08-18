"""profile_dataset: read one delimited table and propose a `DatasetSpec`.

Deterministic and LLM-free (MASTER_PLAN §3.1). This build starts empty (Phase 13): it
ships no task definitions and accepts exactly one thing — a `.csv`/`.tsv` (optionally
gzipped) table with a sequence column and a label column. Everything else is refused,
loudly and by name, rather than silently reshaped into a layout the caller did not ask for.

See plans/PHASE_13_COLD_START_SINGLE_CSV.md §3-4 for the `DatasetSpec` this module builds
and the gate it is put through.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re

from adaptrna_agentic.toolhub.errors import ToolHubError

#: Stamped by `profile_dataset`; `confirm_data_profile` refuses anything else, the same
#: mechanical guard `recommend_training_config`'s `PLAN_SOURCE` gives `start_training`.
PROFILE_SOURCE = "profile_dataset"

#: Stamped by `confirm_data_profile` on approval; `create_task_tool` (Phase 13 §7.5)
#: refuses a spec that does not carry this — a model cannot hand-assemble a spec and
#: skip the human gate.
SPEC_SOURCE = "confirm_data_profile"

SPEC_VERSION = 1

#: Column names that usually hold the sequence / the label.
_SEQUENCE_HINTS = ("seq", "sequence", "utr", "rna", "dna")
_LABEL_HINTS = ("rl", "label", "target", "y", "value", "score", "class")

_NUCLEOTIDES = set("ACGTUN")

#: A column is treated as a sequence column when at least this fraction of its sampled
#: values are nucleotide-only strings of length >= 10.
_SEQUENCE_PURITY = 0.9

_MAX_CLASSES = 20
_MIN_USABLE_ROWS = 200
_IMBALANCE_THRESHOLD = 0.10
_LENGTH_SPREAD_RATIO = 4.0

#: A column is a split candidate when it has few enough distinct values to plausibly name
#: a group (species, fold, pre-existing split, ...).
_MAX_SPLIT_CANDIDATE_VALUES = 10

_SPLIT_NAME_HINTS = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test", "testing"},
}

_TASK_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: One entry per supported target type — the recipe `DatasetSpec.head` is filled from.
#: Duplicated (deliberately, for now) with the same three recipes baked into
#: `codegen/templates/task.py.j2`; Phase 13 §6's `target_shapes.yaml` (Stage 3) becomes
#: the single source of truth for both, but the profiler must not block on it existing.
_TARGET_RECIPES = {
    "binary": {
        "kind": "cls_classifier",
        "loss": "binary_cross_entropy_with_logits",
        "metrics": ["acc", "precision", "recall", "f1_score"],
        "primary_metric": "test/f1_score",
        "pad_sensitive": False,
    },
    "multiclass": {
        "kind": "cls_classifier",
        "loss": "cross_entropy",
        "metrics": ["acc", "macro_f1"],
        "primary_metric": "test/macro_f1",
        "pad_sensitive": False,
    },
    "regression": {
        "kind": "pooled_regressor",
        "loss": "mse",
        "metrics": ["r2", "mse", "mae"],
        "primary_metric": "test/mse",
        "pad_sensitive": True,
    },
}

_UNSUPPORTED_INPUT = (
    "This build trains from a single table (.csv/.tsv, optionally gzipped) containing "
    "one sequence column and one label column. '{path}' is {what}."
)


# =============================================================================== step 1

def profile_dataset(path: Union[str, Path]) -> Dict[str, Any]:
    """Profile one CSV/TSV table and propose a `DatasetSpec` for gate 1.

    Refuses a directory, a FASTA file, or anything whose label column is not binary,
    multiclass (<=20 classes) or a continuous regression target — with a statement of
    what is supported and what was found instead. Nothing is generated or trained from
    this call; it is a read-only proposal for `confirm_data_profile` to put in front of
    the user.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No such dataset path: '{path}'")

    if path.is_dir():
        raise ToolHubError(_UNSUPPORTED_INPUT.format(path=path, what="a directory"))

    separator, compression = _format_of(path)
    if separator is None:
        raise ToolHubError(
            _UNSUPPORTED_INPUT.format(path=path, what=f"a '{path.suffix}' file")
        )

    import pandas as pd

    frame = pd.read_csv(path, sep=separator, compression=compression)
    rows = int(len(frame))

    sequence_column = _detect_sequence_column(frame)
    if sequence_column is None:
        raise ToolHubError(
            f"No column in '{path.name}' looks like a sequence (nucleotide strings of "
            f"reasonable length). This build trains from a single table with one "
            f"sequence column and one label column."
        )

    label_column = _detect_label_column(frame, sequence_column)
    if label_column is None:
        raise ToolHubError(
            f"No usable label column found in '{path.name}' alongside sequence column "
            f"'{sequence_column}'. This build trains from a single table with one "
            f"sequence column and one label column."
        )

    sequences = frame[sequence_column]
    labels = frame[label_column]

    target_type, classes_or_message = _classify_target(labels, sequences)
    if target_type is None:
        raise ToolHubError(classes_or_message)
    classes = classes_or_message

    class_counts, positive_class, target_summary = _target_stats(
        target_type, classes, labels
    )

    ignored_columns = [
        str(c) for c in frame.columns if c not in (sequence_column, label_column)
    ]
    split_candidates = _split_candidates(frame, sequence_column, label_column)
    split = _propose_split(frame, label_column, target_type, split_candidates)

    spec: Dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "source": PROFILE_SOURCE,
        "path": str(path.resolve()),
        "format": {
            "separator": separator, "compression": compression,
            "rows": rows, "header": True,
        },
        "sequence_column": str(sequence_column),
        "label_column": str(label_column),
        "ignored_columns": ignored_columns,
        "on_invalid": "fail",
        "target_type": target_type,
        "classes": classes,
        "positive_class": positive_class,
        "class_counts": class_counts,
        "target_summary": target_summary,
        "alphabet": _alphabet(sequences),
        "length": _length_stats(sequences),
        "split": split,
        "split_candidates": split_candidates,
        "task_name": _default_task_name(path),
        "tool_description": _default_description(target_type, path),
        "head": dict(_TARGET_RECIPES[target_type]),
    }
    spec["warnings"] = _quality_warnings(frame, sequence_column, sequences, class_counts, split)

    return spec


# =============================================================================== gate 1

def confirm_profile(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Re-validate a (possibly human-edited) spec against the real file, and re-stamp it.

    This is the load-bearing half of gate 1 (plan §4): a user who edits `target_type`,
    the split policy, or `task_name` gets a spec whose `classes`, `class_counts`,
    `row_counts` and `head` all follow — never a spec that says one thing and trains
    another. Refuses anything not stamped by `profile_dataset`.
    """
    if spec.get("source") != PROFILE_SOURCE:
        raise ToolHubError(
            "This spec did not come from profile_dataset. The profiler's interpretation "
            "must be approved as given, possibly edited, never assembled by hand — call "
            "profile_dataset and pass its result through unchanged (with your edits on "
            "top, if any)."
        )

    spec = dict(spec)
    path = Path(spec.get("path", ""))
    if not path.is_file():
        raise ToolHubError(f"'{path}' no longer exists.")

    fmt = spec.get("format") or {}
    separator = fmt.get("separator", ",")
    compression = fmt.get("compression")

    import pandas as pd

    frame = pd.read_csv(path, sep=separator, compression=compression)

    sequence_column = spec.get("sequence_column")
    label_column = spec.get("label_column")
    if sequence_column not in frame.columns:
        raise ToolHubError(f"Column '{sequence_column}' is not in '{path.name}'.")
    if label_column not in frame.columns:
        raise ToolHubError(f"Column '{label_column}' is not in '{path.name}'.")
    if sequence_column == label_column:
        raise ToolHubError("sequence_column and label_column must be different columns.")

    target_type = spec.get("target_type")
    if target_type not in _TARGET_RECIPES:
        raise ToolHubError(
            f"'{target_type}' is not a supported target type. This build trains binary, "
            f"multiclass or regression targets from one sequence+label table."
        )

    sequences = frame[sequence_column]
    labels = frame[label_column]

    classes = None
    if target_type in ("binary", "multiclass"):
        observed = sorted(str(v) for v in labels.dropna().unique())
        classes = [str(c) for c in (spec.get("classes") or observed)]
        if target_type == "binary" and len(classes) != 2:
            raise ToolHubError(
                f"target_type is 'binary' but classes is {classes} — binary needs "
                f"exactly two classes. Switch target_type to 'multiclass', or fix classes."
            )
        if target_type == "multiclass" and not (1 < len(classes) <= _MAX_CLASSES):
            raise ToolHubError(
                f"target_type is 'multiclass' but classes has {len(classes)} entries — "
                f"this build supports 2-{_MAX_CLASSES} classes."
            )

    class_counts, positive_class, target_summary = _target_stats(
        target_type, classes, labels, requested_positive=spec.get("positive_class")
    )

    task_name = spec.get("task_name")
    if not task_name or not _TASK_NAME_RE.match(task_name):
        raise ToolHubError(
            f"'{task_name}' is not a valid task name — lowercase letters, digits and "
            f"underscores, starting with a letter."
        )
    _reject_existing_task_name(task_name)

    split = _validate_and_recompute_split(spec.get("split") or {}, frame, label_column)

    spec.update({
        "source": SPEC_SOURCE,
        "sequence_column": str(sequence_column),
        "label_column": str(label_column),
        "target_type": target_type,
        "classes": classes,
        "positive_class": positive_class,
        "class_counts": class_counts,
        "target_summary": target_summary,
        "alphabet": _alphabet(sequences),
        "length": _length_stats(sequences),
        "split": split,
        "task_name": task_name,
        "tool_description": spec.get("tool_description") or _default_description(target_type, path),
        "head": dict(_TARGET_RECIPES[target_type]),
    })
    spec["warnings"] = _quality_warnings(frame, sequence_column, sequences, class_counts, split)

    return spec


def _reject_existing_task_name(task_name: str) -> None:
    from adaptrna_agentic.codegen.discovery import custom_task_names

    if task_name in custom_task_names():
        raise ToolHubError(
            f"A task named '{task_name}' already exists in adaptrna_custom/tasks/. "
            f"Landed code is yours: edit it directly, choose another name here."
        )


# =============================================================================== format

def _format_of(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """`(separator, compression)`, or `(None, None)` if this is not a table this build
    reads (D8: only .csv/.tsv, optionally .gz)."""
    suffixes = [s.lower() for s in path.suffixes]

    compression = None
    if suffixes and suffixes[-1] == ".gz":
        compression = "gzip"
        suffixes = suffixes[:-1]

    if not suffixes or suffixes[-1] not in (".csv", ".tsv"):
        return None, None

    return ("\t" if suffixes[-1] == ".tsv" else ","), compression


# =============================================================================== columns

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
    for column in frame.columns:
        if str(column).lower() in _SEQUENCE_HINTS and _looks_like_sequences(frame[column]):
            return column

    # `dtype == object` is not portable across pandas versions; skipping numeric/bool
    # kinds and sniffing the rest works everywhere.
    for column in frame.columns:
        if frame[column].dtype.kind not in "ifbc" and _looks_like_sequences(frame[column]):
            return column

    return None


def _detect_label_column(frame, sequence_column) -> Optional[str]:
    candidates = [c for c in frame.columns if c != sequence_column]

    for column in candidates:
        if str(column).lower() in _LABEL_HINTS:
            return column

    numeric = [
        c for c in candidates
        if frame[c].dtype.kind in "if" and not str(c).lower().startswith("unnamed")
    ]
    if numeric:
        return numeric[0]

    # A categorical label with no name hint: the first low-cardinality non-sequence-like
    # text column.
    for column in candidates:
        if frame[column].dtype.kind not in "ifbc":
            nunique = frame[column].nunique(dropna=True)
            if 0 < nunique <= _MAX_CLASSES and not _looks_like_sequences(frame[column]):
                return column

    return None


# =============================================================================== target

def _classify_target(labels, sequences) -> Tuple[Optional[str], Any]:
    """`(target_type, classes)` on success; `(None, refusal_message)` otherwise (D7)."""
    values = labels.dropna()
    if values.empty:
        return None, f"Column '{labels.name}' has no usable (non-missing) values."

    unique = values.unique()

    if values.dtype.kind in "if":
        if len(unique) == 2:
            return "binary", sorted(str(v) for v in unique)
        if len(unique) <= _MAX_CLASSES and all(float(v).is_integer() for v in unique):
            return "multiclass", sorted((str(int(v)) for v in unique), key=int)
        return "regression", None

    str_values = values.astype(str)
    median_label_len = str_values.str.len().median()
    seq_lengths = sequences.dropna().astype(str).str.len()
    median_seq_len = seq_lengths.median() if not seq_lengths.empty else 0

    per_position = (
        median_label_len > 20 and median_seq_len
        and abs(median_label_len - median_seq_len) / median_seq_len < 0.2
    )
    if per_position:
        return None, (
            f"The label column '{labels.name}' holds per-position strings the same "
            f"length as the sequences. This build trains three head shapes from one "
            f"sequence+label table: binary classification, multiclass classification "
            f"(≤{_MAX_CLASSES} classes), and regression on a continuous target. "
            f"Per-position and structure targets are not supported here."
        )

    if len(unique) == 2:
        return "binary", sorted(str(v) for v in unique)
    if len(unique) <= _MAX_CLASSES:
        return "multiclass", sorted(str(v) for v in unique)

    return None, (
        f"The label column '{labels.name}' holds {len(unique)} distinct free-text "
        f"values — more than the {_MAX_CLASSES} this build supports as multiclass "
        f"classes, and not numeric enough to be a regression target. This build trains "
        f"binary classification, multiclass classification (≤{_MAX_CLASSES} "
        f"classes), and regression on a continuous target from one sequence+label table."
    )


def _target_stats(target_type, classes, labels, requested_positive=None):
    """`(class_counts, positive_class, target_summary)` for the given target type.

    An explicitly-set `requested_positive` that does not name one of `classes` is a
    rejection, not a silent substitution — same treatment as bad split fractions. Only a
    genuinely absent value (the proposal path, or a spec edited away from binary and back)
    falls back to the default heuristic.
    """
    if target_type in ("binary", "multiclass"):
        counts = labels.dropna().astype(str).value_counts()
        class_counts = {c: int(counts.get(c, 0)) for c in classes}
        positive_class = None
        if target_type == "binary":
            if requested_positive is None:
                positive_class = _default_positive_class(class_counts)
            else:
                positive_class = str(requested_positive)
                if positive_class not in class_counts:
                    raise ToolHubError(
                        f"positive_class '{positive_class}' is not one of {classes} — it "
                        f"must name one of the approved classes."
                    )
        return class_counts, positive_class, None

    values = labels.dropna()
    target_summary = {
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
    }
    return None, None, target_summary


def _default_positive_class(class_counts: Dict[str, int]) -> str:
    """The minority class — 'the thing being detected' is usually the rarer one."""
    return min(class_counts, key=lambda c: (class_counts[c], c))


# =============================================================================== split

def _split_candidates(frame, sequence_column, label_column) -> Dict[str, Dict[str, int]]:
    candidates: Dict[str, Dict[str, int]] = {}
    for column in frame.columns:
        if column in (sequence_column, label_column):
            continue
        nunique = frame[column].nunique(dropna=True)
        if 0 < nunique <= _MAX_SPLIT_CANDIDATE_VALUES:
            counts = frame[column].value_counts()
            candidates[str(column)] = {str(k): int(v) for k, v in counts.items()}

    return candidates


def _split_name_mapping(values) -> Optional[Dict[str, List[str]]]:
    mapping: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for value in values:
        lowered = str(value).strip().lower()
        for split_name, hints in _SPLIT_NAME_HINTS.items():
            if lowered in hints:
                mapping[split_name].append(value)
                break

    if mapping["train"] and (mapping["val"] or mapping["test"]):
        return {name: vals for name, vals in mapping.items() if vals}

    return None


def _propose_split(frame, label_column, target_type, split_candidates):
    for column, value_counts in split_candidates.items():
        mapping = _split_name_mapping(value_counts.keys())
        if mapping:
            return _column_split(frame, column, mapping)

    return _random_split(frame, target_type)


def _random_split(frame, target_type, fractions=None, seed=42, stratify=None):
    fractions = fractions or {"train": 0.8, "val": 0.1, "test": 0.1}
    if stratify is None:
        stratify = target_type in ("binary", "multiclass")

    n = len(frame)
    row_counts = {
        "train": round(n * fractions["train"]),
        "val": round(n * fractions["val"]),
    }
    row_counts["test"] = max(n - row_counts["train"] - row_counts["val"], 0)

    return {
        "mode": "random", "fractions": fractions, "seed": seed, "stratify": stratify,
        "column": None, "mapping": None,
        "row_counts": row_counts, "dropped_rows": 0,
    }


def _column_split(frame, column, mapping):
    row_counts = {}
    accounted = 0
    for split_name in ("train", "val", "test"):
        values = {str(v) for v in mapping.get(split_name, [])}
        count = int(frame[column].astype(str).isin(values).sum())
        row_counts[split_name] = count
        accounted += count

    return {
        "mode": "column", "column": str(column), "mapping": mapping,
        "fractions": None, "seed": None, "stratify": None,
        "row_counts": row_counts, "dropped_rows": max(len(frame) - accounted, 0),
    }


def _validate_and_recompute_split(split, frame, label_column):
    mode = split.get("mode")

    if mode == "random":
        fractions = split.get("fractions") or {}
        if set(fractions) != {"train", "val", "test"}:
            raise ToolHubError(
                f"split.fractions must set train, val and test — got {sorted(fractions)}."
            )
        if any(not isinstance(v, (int, float)) or v < 0 for v in fractions.values()):
            raise ToolHubError(f"split.fractions must be non-negative numbers: {fractions}")
        total = sum(fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ToolHubError(f"split.fractions {fractions} sum to {total}, not 1.0.")

        return _random_split(
            frame, target_type=None, fractions=fractions,
            seed=split.get("seed", 42),
            stratify=bool(split.get("stratify", True)),
        )

    if mode == "column":
        column = split.get("column")
        if column not in frame.columns:
            raise ToolHubError(f"split.column '{column}' is not in the file.")
        mapping = split.get("mapping") or {}
        if not any(mapping.get(name) for name in ("train", "val", "test")):
            raise ToolHubError(
                "split.mapping must name at least one value for at least one split."
            )
        for name, values in mapping.items():
            if name not in ("train", "val", "test"):
                raise ToolHubError(f"split.mapping has an unknown split name '{name}'.")
            if not isinstance(values, list) or not values:
                raise ToolHubError(f"split.mapping['{name}'] must be a non-empty list.")

        return _column_split(frame, column, mapping)

    raise ToolHubError(f"split.mode must be 'random' or 'column', got {mode!r}.")


# =============================================================================== warnings

def _quality_warnings(frame, sequence_column, sequences, class_counts, split) -> List[str]:
    warnings: List[str] = []

    for warning in (
        _duplicate_warning(sequences),
        _imbalance_warning(class_counts),
        _invalid_sequence_warning(sequences),
        _missing_values_warning(frame, sequence_column),
        _length_spread_warning(_length_stats(sequences)),
        _tiny_data_warning(split["row_counts"]),
    ):
        if warning:
            warnings.append(warning)

    if split["mode"] == "column":
        leakage = _leakage_warning(frame, sequence_column, split["column"], split["mapping"])
        if leakage:
            warnings.append(leakage)
        if split["dropped_rows"]:
            warnings.append(
                f"{split['dropped_rows']:,} row(s) do not match any value named in "
                f"split.mapping and are dropped."
            )
    elif split["fractions"] and split["fractions"].get("test", 0) == 0:
        warnings.append(
            "test fraction is 0 — there is no held-out number for this run; the primary "
            "metric falls back to val/*."
        )

    return warnings


def _duplicate_warning(sequences) -> Optional[str]:
    counts = sequences.value_counts()
    duplicated = counts[counts > 1]
    if duplicated.empty:
        return None

    dup_rows = int(duplicated.sum())
    pct = 100.0 * dup_rows / len(sequences)
    return (
        f"{dup_rows:,} sequences ({pct:.1f}%) appear more than once in the file; "
        f"duplicates are split independently, so the same sequence can land in both "
        f"train and test."
    )


def _leakage_warning(frame, sequence_column, column, mapping) -> Optional[str]:
    import itertools

    groups = {}
    for split_name, values in mapping.items():
        wanted = {str(v) for v in values}
        subset = frame[frame[column].astype(str).isin(wanted)]
        groups[split_name] = set(subset[sequence_column])

    overlaps = []
    for a, b in itertools.combinations(sorted(groups), 2):
        shared = groups[a] & groups[b]
        if shared:
            overlaps.append(f"{len(shared):,} sequences appear in both the {a} and {b} groups")

    if not overlaps:
        return None
    return "; ".join(overlaps) + "."


def _imbalance_warning(class_counts) -> Optional[str]:
    if not class_counts:
        return None

    total = sum(class_counts.values())
    if total == 0:
        return None

    minority_class = min(class_counts, key=class_counts.get)
    minority_frac = class_counts[minority_class] / total
    if minority_frac >= _IMBALANCE_THRESHOLD:
        return None

    return (
        f"class '{minority_class}' is only {minority_frac * 100:.1f}% of labeled rows "
        f"({class_counts[minority_class]:,} of {total:,}) — a minority class below 10%."
    )


def _invalid_sequence_warning(sequences) -> Optional[str]:
    str_sequences = sequences.dropna().astype(str)
    invalid = ~str_sequences.str.upper().str.match(r"^[ACGTUN]+$")
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        return None

    bad_chars: set = set()
    for value in str_sequences[invalid].head(50):
        bad_chars |= set(value.upper()) - _NUCLEOTIDES

    return (
        f"{n_invalid:,} row(s) contain sequence characters outside ACGTUN "
        f"(e.g. {sorted(bad_chars)[:5]}). By default these rows make the datamodule fail "
        f"loudly (on_invalid='fail'); set on_invalid='drop' at this gate to discard them "
        f"instead."
    )


def _missing_values_warning(frame, sequence_column) -> Optional[str]:
    n_missing_seq = int(frame[sequence_column].isna().sum())
    if not n_missing_seq:
        return None

    return f"{n_missing_seq:,} row(s) have a missing sequence and are dropped."


def _length_spread_warning(length: Dict[str, int]) -> Optional[str]:
    median = length.get("median") or 0
    max_len = length.get("max") or 0
    if median <= 0:
        return None

    ratio = max_len / median
    if ratio <= _LENGTH_SPREAD_RATIO:
        return None

    return (
        f"sequence lengths range up to {max_len} nt against a median of {median} nt "
        f"({ratio:.1f}x) — batch size is derived from the median and memory is driven "
        f"by the max."
    )


def _tiny_data_warning(row_counts: Dict[str, int]) -> Optional[str]:
    usable = sum(row_counts.values())
    if usable >= _MIN_USABLE_ROWS:
        return None

    return f"only {usable:,} usable rows after splitting — a run this small will not mean much."


# =============================================================================== helpers

def _length_stats(series) -> Dict[str, int]:
    lengths = series.dropna().astype(str).str.len()
    if lengths.empty:
        return {"min": 0, "median": 0, "max": 0}

    return {
        "min": int(lengths.min()), "median": int(lengths.median()), "max": int(lengths.max()),
    }


def _alphabet(series) -> str:
    letters: set = set()
    for value in series.dropna().astype(str).head(200):
        letters |= set(value.upper())

    if not letters:
        return "unknown"
    if letters <= _NUCLEOTIDES:
        return "dna" if "T" in letters and "U" not in letters else "rna"
    return "other"


def _default_task_name(path: Path) -> str:
    stem = path.name
    for suffix in (".gz", ".csv", ".tsv"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]

    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") or "dataset"
    if slug[0].isdigit():
        slug = f"t_{slug}"

    return slug


def _default_description(target_type: str, path: Path) -> str:
    return f"{target_type} target trained from {path.name}"
