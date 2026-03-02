"""
Test 3 stylistic features on a small sample of 10 multi-line poems per era.
Features:
  1. cap_rate: fraction of lines starting with an uppercase letter
  2. punct_rate: fraction of lines ending with punctuation
  3. colon_density: colons + semicolons per 100 words
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "data/PoetryFoundationData_with_year.csv"
PLOT_DIR = "data/plots"
SAMPLE_PER_ERA = 40
SEED = 42

ERA_BINS = [
    ("Pre-1800", -9999, 1800),
    ("1800-1900", 1800, 1900),
    ("1900-1950", 1900, 1950),
    ("Post-1950", 1950, 9999),
]

PUNCT_CHARS = set(".,;:!?—–-")


def get_lines(poem_text: str) -> list[str]:
    return [l.strip() for l in poem_text.split("\n") if l.strip()]


def is_multiline(poem_text: str) -> bool:
    return len(get_lines(poem_text)) > 1


def cap_rate(lines: list[str]) -> float:
    if not lines:
        return np.nan
    caps = sum(1 for l in lines if l and l[0].isupper())
    return caps / len(lines)


def punct_rate(lines: list[str]) -> float:
    if not lines:
        return np.nan
    ended = sum(1 for l in lines if l and l[-1] in PUNCT_CHARS)
    return ended / len(lines)


def colon_density(poem_text: str) -> float:
    words = poem_text.split()
    if not words:
        return 0.0
    count = sum(1 for ch in poem_text if ch in (";", ":"))
    return count / len(words) * 100


def assign_era(year: float) -> str | None:
    for label, lo, hi in ERA_BINS:
        if lo <= year < hi:
            return label
    return None


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["era"] = df["Year"].apply(assign_era)
    df = df[df["Poem"].apply(is_multiline)].copy()

    sample = df.copy()

    rows = []
    for _, r in sample.iterrows():
        lines = get_lines(r["Poem"])
        rows.append({
            "Title": r["Title"].strip()[:50],
            "Poet": r["Poet"],
            "Year": int(r["Year"]),
            "era": r["era"],
            "cap_rate": cap_rate(lines),
            "punct_rate": punct_rate(lines),
            "colon_density": colon_density(r["Poem"]),
        })
    results = pd.DataFrame(rows)

    print("=" * 100)
    print("MEAN / VARIANCE PER ERA")
    print("=" * 100)
    for feat in ["cap_rate", "punct_rate", "colon_density"]:
        print(f"\n  {feat}:")
        for era_label, _, _ in ERA_BINS:
            vals = results.loc[results["era"] == era_label, feat].dropna()
            if len(vals) == 0:
                continue
            print(f"    {era_label:<12s}  mean={vals.mean():.3f}  var={vals.var():.4f}  n={len(vals)}")

    # --- Plots ---
    features_config = [
        ("cap_rate", "Line-Initial Capitalization Rate", "beta"),
        ("punct_rate", "End-of-Line Punctuation Rate", "beta"),
        ("colon_density", "Colon/Semicolon per 100 Words", "gamma"),
    ]

    for feat_col, feat_title, dist_type in features_config:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        fig.suptitle(feat_title, fontsize=14, fontweight="bold")

        for ax, (era_label, _, _) in zip(axes, ERA_BINS):
            vals = results.loc[results[feat_col].notna() & (results["era"] == era_label), feat_col].values
            if len(vals) == 0:
                ax.set_title(era_label)
                continue

            ax.hist(vals, bins=25, density=True, alpha=0.6, edgecolor="black", label="data")

            m, v = vals.mean(), vals.var()
            x_min, x_max = max(0, vals.min() - 0.1), vals.max() + 0.1

            if dist_type == "beta" and v > 0 and 0 < m < 1:
                try:
                    a_fit, b_fit, loc_fit, scale_fit = stats.beta.fit(vals, floc=0, fscale=1)
                    x = np.linspace(0.01, 0.99, 200)
                    ax.plot(x, stats.beta.pdf(x, a_fit, b_fit), "r-", lw=2, label=f"Beta({a_fit:.1f},{b_fit:.1f})")
                except Exception:
                    pass
            elif dist_type == "gamma" and v > 0:
                try:
                    a_fit, loc_fit, scale_fit = stats.gamma.fit(vals, floc=0)
                    x = np.linspace(x_min, x_max, 200)
                    ax.plot(x, stats.gamma.pdf(x, a_fit, loc=0, scale=scale_fit), "r-", lw=2,
                            label=f"Gamma(a={a_fit:.1f})")
                except Exception:
                    pass

            ax.set_title(f"{era_label}\nmean={m:.2f}, var={v:.3f}", fontsize=10)
            ax.legend(fontsize=8)

        plt.tight_layout()
        out = f"{PLOT_DIR}/{feat_col}_distribution.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved: {out}")
        plt.close()


if __name__ == "__main__":
    main()
