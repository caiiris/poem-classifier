"""
Build a new CSV of poems that were previously excluded (single-line / no \\n)
by converting 2+ consecutive spaces into line breaks, and (for those still
single-line) inserting line breaks where punctuation is directly followed
by a letter (missing space = likely lost line break).

Reads: data/PoetryFoundationData_with_year.csv
Writes: data/PoetryFoundationData_spaces_as_newlines.csv

- Only includes rows where the original Poem had 0 or 1 line (when split on \\n).
- Poem column: 2+ spaces → newline; poems that are still single-line after that
  get newline after .,;: when the next char is a letter with no space (e.g. ",And" → ",\\nAnd").
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "PoetryFoundationData_with_year.csv"
OUTPUT_CSV = DATA_DIR / "PoetryFoundationData_spaces_as_newlines.csv"

# Same logic as in plot_all_features_log.py etc.
def get_lines(text: str) -> list[str]:
    return [l.strip() for l in text.split("\n") if l.strip()]


def is_multiline(text: str) -> bool:
    return len(get_lines(text)) > 1


def spaces_to_newlines(text: str) -> str:
    """Replace any sequence of 2 or more spaces with a single newline."""
    if not isinstance(text, str) or not text.strip():
        return text
    return re.sub(r" {2,}", "\n", text)


# Period, comma, colon, semicolon only; if immediately followed by a letter (no space), treat as line break
PUNCT_NO_SPACE_RE = re.compile(r"([.,;:])([a-zA-Z])")


def punctuation_no_space_to_newline(text: str) -> str:
    """Insert newline after . , ; or : when the next character is a letter (no space)."""
    if not isinstance(text, str) or not text.strip():
        return text
    return PUNCT_NO_SPACE_RE.sub(r"\1\n\2", text)


def main():
    df = pd.read_csv(INPUT_CSV)
    # Poems we have been ignoring: not multiline (0 or 1 line when split on \n)
    ignored = ~df["Poem"].fillna("").apply(is_multiline)
    subset = df[ignored].copy()
    subset["Poem"] = subset["Poem"].fillna("").apply(spaces_to_newlines)
    # Leave poems that are already multiline alone; for the rest, apply punctuation rule
    still_single = ~subset["Poem"].apply(is_multiline)
    subset.loc[still_single, "Poem"] = subset.loc[still_single, "Poem"].apply(
        punctuation_no_space_to_newline
    )
    subset.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(subset)} poems (previously single-line) to {OUTPUT_CSV}")
    now_multiline = subset["Poem"].apply(is_multiline).sum()
    print(f"Of those, {now_multiline} have 2+ lines after spaces→newlines and/or punctuation→newline.")


if __name__ == "__main__":
    main()
