"""
Test special character density on all multi-line poems.
Features (occurrences per 100 words):
  1. slash_density: / characters
  2. ampersand_density: & characters
  3. paren_density: ( and ) characters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "data/PoetryFoundationData_with_year.csv"
PLOT_DIR = "data/plots"

ERA_BINS = [
    ("Pre-1800", -9999, 1800),
    ("1800-1900", 1800, 1900),
    ("1900-1950", 1900, 1950),
    ("Post-1950", 1950, 9999),
]


def get_lines(poem_text: str) -> list[str]:
    return [l.strip() for l in poem_text.split("\n") if l.strip()]


def is_multiline(poem_text: str) -> bool:
    return len(get_lines(poem_text)) > 1


def assign_era(year: float) -> str | None:
    for label, lo, hi in ERA_BINS:
        if lo <= year < hi:
            return label
    return None


def char_density(poem_text: str, chars: set[str]) -> float:
    words = poem_text.split()
    if not words:
        return 0.0
    count = sum(1 for ch in poem_text if ch in chars)
    return count / len(words) * 100


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["era"] = df["Year"].apply(assign_era)
    df = df[df["Poem"].apply(is_multiline)].copy()
    print(f"Processing {len(df)} multi-line poems")

    rows = []
    for _, r in df.iterrows():
        text = r["Poem"]
        rows.append({
            "era": r["era"],
            "slash_density": char_density(text, {"/"}),
            "ampersand_density": char_density(text, {"&"}),
            "paren_density": char_density(text, {"(", ")"}),
        })
    results = pd.DataFrame(rows)

    features = ["slash_density", "ampersand_density", "paren_density"]

    print("\n" + "=" * 100)
    print("MEAN / VARIANCE PER ERA (occurrences per 100 words)")
    print("=" * 100)
    for feat in features:
        print(f"\n  {feat}:")
        for era_label, _, _ in ERA_BINS:
            vals = results.loc[results["era"] == era_label, feat].dropna()
            if len(vals) == 0:
                continue
            nonzero = (vals > 0).sum()
            print(f"    {era_label:<12s}  mean={vals.mean():.3f}  var={vals.var():.4f}  "
                  f"nonzero={nonzero}/{len(vals)} ({nonzero/len(vals):.0%})  n={len(vals)}")

    # --- Plots ---
    titles = {
        "slash_density": "Slash (/) per 100 Words",
        "ampersand_density": "Ampersand (&) per 100 Words",
        "paren_density": "Parentheses per 100 Words",
    }

    for feat_col in features:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        fig.suptitle(titles[feat_col], fontsize=14, fontweight="bold")

        for ax, (era_label, _, _) in zip(axes, ERA_BINS):
            vals = results.loc[
                results[feat_col].notna() & (results["era"] == era_label), feat_col
            ].values
            if len(vals) == 0:
                ax.set_title(era_label)
                continue

            clip = np.percentile(vals[vals > 0], 99) if (vals > 0).sum() > 10 else vals.max() + 1
            plot_vals = vals[vals <= clip]

            ax.hist(plot_vals, bins=25, density=True, alpha=0.6, edgecolor="black", label="data")

            m = vals.mean()
            nonzero_pct = (vals > 0).sum() / len(vals) * 100
            ax.set_title(f"{era_label}\nmean={m:.3f}, {nonzero_pct:.0f}% nonzero", fontsize=10)
            ax.legend(fontsize=8)

        plt.tight_layout()
        out = f"{PLOT_DIR}/{feat_col}_distribution.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved: {out}")
        plt.close()


if __name__ == "__main__":
    main()
