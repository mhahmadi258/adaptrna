#!/usr/bin/env python
"""Regenerate `dod_data/` — the demo dataset for the cold-start codegen flow.

Derives flat `sequence,label` CSVs from a source fold directory, in a schema no shipped
engine task can read. That is the point — it is what makes the "build me a task for this
data" flow necessary to demonstrate.

The data itself is git-ignored (like every other dataset in this repo); this script is
tracked so it stays reproducible.

    python agentic/scripts/make_demo_data.py --source DIR [--out DIR]
"""

from pathlib import Path
import argparse

COLUMNS = ["group", "sequence", "label"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Fold directory containing Train_/Val_donor_400.csv")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[2] / "dod_data")
    args = parser.parse_args()

    import pandas as pd

    train = pd.read_csv(args.source / "Train_donor_400.csv", sep=";", header=None, names=COLUMNS)
    val = pd.read_csv(args.source / "Val_donor_400.csv", sep=";", header=None, names=COLUMNS)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame, n, seed in (
        ("train", train, 4000, 42),
        ("val", val, 800, 42),
        ("test", val, 800, 7),
    ):
        subset = frame.sample(n=n, random_state=seed)[["sequence", "label"]]
        subset.to_csv(args.out / f"demo_simple_{name}.csv", index=False)
        print(f"wrote {args.out / f'demo_simple_{name}.csv'} ({len(subset)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
