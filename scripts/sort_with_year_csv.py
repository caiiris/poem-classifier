"""
Sort PoetryFoundationData_with_year.csv:
  1. Poems with no newlines (single-line) at the top, then poems with newlines.
  2. Within each group, sort by Year (ascending; rows with missing Year go last in that group).

Reads and overwrites: data/PoetryFoundationData_with_year.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "PoetryFoundationData_with_year.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    poems = df["Poem"].fillna("")
    has_newline = poems.str.contains("\n", regex=False)
    # False (no newline) first, then True (has newline). Within each, sort by Year.
    df = df.assign(_has_newline=has_newline)
    df = df.sort_values(
        ["_has_newline", "Year"],
        ascending=[True, True],
        na_position="last",
    ).drop(columns=["_has_newline"])
    df.to_csv(CSV_PATH, index=False)
    no_nl = (~has_newline).sum()
    with_nl = has_newline.sum()
    print(f"Sorted {len(df)} rows: {no_nl} without newlines (top), {with_nl} with newlines (bottom); by year within each.")


if __name__ == "__main__":
    main()
