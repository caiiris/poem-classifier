"""
Test 3 new features on all multi-line poems:
  1. line_length_var: variance of word count per line
  2. adj_noun_ratio: adjectives / (nouns + 1)
  3. adv_verb_ratio: adverbs / (verbs + 1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import nltk

CSV_PATH = "data/PoetryFoundationData_with_year.csv"
PLOT_DIR = "data/plots"

ERA_BINS = [
    ("Pre-1800", -9999, 1800),
    ("1800-1900", 1800, 1900),
    ("1900-1950", 1900, 1950),
    ("Post-1950", 1950, 9999),
]

ADJ_TAGS = {"JJ", "JJR", "JJS"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
ADV_TAGS = {"RB", "RBR", "RBS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}


def get_lines(poem_text: str) -> list[str]:
    return [l.strip() for l in poem_text.split("\n") if l.strip()]


def is_multiline(poem_text: str) -> bool:
    return len(get_lines(poem_text)) > 1


def assign_era(year: float) -> str | None:
    for label, lo, hi in ERA_BINS:
        if lo <= year < hi:
            return label
    return None


def line_length_variance(lines: list[str]) -> float:
    if len(lines) < 2:
        return np.nan
    lengths = [len(l.split()) for l in lines]
    return np.var(lengths, ddof=0)


def pos_ratios(poem_text: str) -> tuple[float, float]:
    tokens = nltk.word_tokenize(poem_text)
    tagged = nltk.pos_tag(tokens)

    adj = sum(1 for _, tag in tagged if tag in ADJ_TAGS)
    noun = sum(1 for _, tag in tagged if tag in NOUN_TAGS)
    adv = sum(1 for _, tag in tagged if tag in ADV_TAGS)
    verb = sum(1 for _, tag in tagged if tag in VERB_TAGS)

    adj_noun = adj / (noun + 1)
    adv_verb = adv / (verb + 1)
    return adj_noun, adv_verb


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
    total = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"  {i}/{total} ...")

        lines = get_lines(r["Poem"])
        adj_noun, adv_verb = pos_ratios(r["Poem"])
        rows.append({
            "Year": int(r["Year"]),
            "era": r["era"],
            "line_length_var": line_length_variance(lines),
            "adj_noun_ratio": adj_noun,
            "adv_verb_ratio": adv_verb,
        })

    results = pd.DataFrame(rows)
    print(f"  {total}/{total} done\n")

    print("=" * 100)
    print("MEAN / VARIANCE PER ERA")
    print("=" * 100)
    for feat in ["line_length_var", "adj_noun_ratio", "adv_verb_ratio"]:
        print(f"\n  {feat}:")
        for era_label, _, _ in ERA_BINS:
            vals = results.loc[results["era"] == era_label, feat].dropna()
            if len(vals) == 0:
                continue
            print(f"    {era_label:<12s}  mean={vals.mean():.3f}  var={vals.var():.4f}  n={len(vals)}")

    # --- Clipped stats for line_length_var (outliers from broken formatting) ---
    clip_95 = results["line_length_var"].quantile(0.95)
    clipped = results[results["line_length_var"] <= clip_95].copy()
    print(f"\n  line_length_var (clipped at p95={clip_95:.0f}):")
    for era_label, _, _ in ERA_BINS:
        vals = clipped.loc[clipped["era"] == era_label, "line_length_var"].dropna()
        if len(vals) == 0:
            continue
        print(f"    {era_label:<12s}  mean={vals.mean():.3f}  var={vals.var():.4f}  n={len(vals)}")

    # --- Plots ---
    features_config = [
        ("line_length_var", "Line Length Variance (words per line)", "gamma"),
        ("adj_noun_ratio", "Adjective-to-Noun Ratio", "gamma"),
        ("adv_verb_ratio", "Adverb-to-Verb Ratio", "gamma"),
    ]

    for feat_col, feat_title, dist_type in features_config:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        fig.suptitle(feat_title, fontsize=14, fontweight="bold")

        for ax, (era_label, _, _) in zip(axes, ERA_BINS):
            era_data = results.loc[
                results[feat_col].notna() & (results["era"] == era_label), feat_col
            ]
            if feat_col == "line_length_var":
                era_data = era_data[era_data <= clip_95]
            vals = era_data.values
            if len(vals) == 0:
                ax.set_title(era_label)
                continue

            ax.hist(vals, bins=25, density=True, alpha=0.6, edgecolor="black", label="data")

            m, v = vals.mean(), vals.var()
            if v > 0:
                try:
                    a_fit, loc_fit, scale_fit = stats.gamma.fit(vals, floc=0)
                    x = np.linspace(0, np.percentile(vals, 99), 200)
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
