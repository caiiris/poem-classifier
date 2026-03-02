"""
Test 3 abstraction metrics on 40 multi-line poems per era.
Metrics (all use Brysbaert concreteness ratings, 1=abstract, 5=concrete):
  1. mean_concreteness: average score of matched words
  2. abstract_density:  words with score < 2.0 per 100 words
  3. concrete_abstract_ratio: count(score>=4.0) / (count(score<=2.0) + 1)
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "data/PoetryFoundationData_with_year.csv"
CONCRETENESS_PATH = "data/concreteness_ratings.csv"
PLOT_DIR = "data/plots"
SAMPLE_PER_ERA = 40
SEED = 42

ERA_BINS = [
    ("Pre-1800", -9999, 1800),
    ("1800-1900", 1800, 1900),
    ("1900-1950", 1900, 1950),
    ("Post-1950", 1950, 9999),
]

STRIP_RE = re.compile(r"[^a-z\s]")


def load_concreteness(path: str) -> dict[str, float]:
    lex = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, score = line.rsplit(",", 1)
            lex[word.lower()] = float(score)
    return lex


def tokenize(poem_text: str) -> list[str]:
    text = poem_text.lower()
    text = STRIP_RE.sub(" ", text)
    return [w for w in text.split() if w.isalpha() and len(w) > 1]


def mean_concreteness(tokens: list[str], lex: dict[str, float]) -> float:
    scores = [lex[t] for t in tokens if t in lex]
    return np.mean(scores) if scores else np.nan


def abstract_density(tokens: list[str], lex: dict[str, float]) -> float:
    if not tokens:
        return np.nan
    abstract_count = sum(1 for t in tokens if t in lex and lex[t] < 2.0)
    return abstract_count / len(tokens) * 100


def concrete_abstract_ratio(tokens: list[str], lex: dict[str, float]) -> float:
    concrete = sum(1 for t in tokens if t in lex and lex[t] >= 4.0)
    abstract = sum(1 for t in tokens if t in lex and lex[t] <= 2.0)
    if concrete == 0 and abstract == 0:
        return np.nan
    return concrete / (abstract + 1)


def coverage(tokens: list[str], lex: dict[str, float]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in lex) / len(tokens)


def get_lines(poem_text: str) -> list[str]:
    return [l.strip() for l in poem_text.split("\n") if l.strip()]


def is_multiline(poem_text: str) -> bool:
    return len(get_lines(poem_text)) > 1


def assign_era(year: float) -> str | None:
    for label, lo, hi in ERA_BINS:
        if lo <= year < hi:
            return label
    return None


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    lex = load_concreteness(CONCRETENESS_PATH)
    print(f"Loaded {len(lex):,} concreteness ratings")

    df = pd.read_csv(CSV_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["era"] = df["Year"].apply(assign_era)
    df = df[df["Poem"].apply(is_multiline)].copy()

    sample = df.copy()
    print(f"Processing {len(sample)} multi-line poems (full dataset)")

    rows = []
    for _, r in sample.iterrows():
        tokens = tokenize(r["Poem"])
        rows.append({
            "Year": int(r["Year"]),
            "era": r["era"],
            "concrete_abstract_ratio": concrete_abstract_ratio(tokens, lex),
        })
    results = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("CONCRETE-TO-ABSTRACT RATIO — MEAN / VARIANCE PER ERA")
    print("=" * 100)
    for era_label, _, _ in ERA_BINS:
        vals = results.loc[results["era"] == era_label, "concrete_abstract_ratio"].dropna()
        if len(vals) == 0:
            continue
        print(f"  {era_label:<12s}  mean={vals.mean():.3f}  var={vals.var():.4f}  n={len(vals)}")

    # --- Plot ---
    feat_col = "concrete_abstract_ratio"
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    fig.suptitle("Concrete-to-Abstract Ratio", fontsize=14, fontweight="bold")

    for ax, (era_label, _, _) in zip(axes, ERA_BINS):
        vals = results.loc[
            results[feat_col].notna() & (results["era"] == era_label), feat_col
        ].values
        if len(vals) == 0:
            ax.set_title(era_label)
            continue

        ax.hist(vals, bins=25, density=True, alpha=0.6, edgecolor="black", label="data")

        m, v = vals.mean(), vals.var()
        if v > 0:
            try:
                a_fit, loc_fit, scale_fit = stats.gamma.fit(vals, floc=0)
                x = np.linspace(0, vals.max() + 0.5, 200)
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
