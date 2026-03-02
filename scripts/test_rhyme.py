"""
Test rhyme rate on all multi-line poems.
Rhyme rate: fraction of consecutive line-end pairs that share a rhyme
(where rhyme = phoneme sequence from last stressed vowel to end of word).

Uses the CMU Pronouncing Dictionary via nltk.
"""

import re
import pandas as pd
import numpy as np
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

STRIP_PUNCT = re.compile(r"[^a-z]")


def build_rhyme_dict() -> dict[str, list[str]]:
    """
    Returns a dict mapping lowercase word -> list of rhyme tails.
    Multiple pronunciations are kept (a word can have >1 entry in cmudict).
    Rhyme tail = phonemes from last stressed vowel to end.
    """
    cmu = nltk.corpus.cmudict.dict()
    rhyme_dict = {}
    for word, pronunciations in cmu.items():
        tails = set()
        for phones in pronunciations:
            # Find last stressed vowel (contains '1' or '2')
            for i in range(len(phones) - 1, -1, -1):
                if phones[i][-1] in ("1", "2"):
                    tails.add(tuple(phones[i:]))
                    break
        if tails:
            rhyme_dict[word] = list(tails)
    return rhyme_dict


def get_rhyme_tails(word: str, rhyme_dict: dict) -> list[tuple]:
    clean = STRIP_PUNCT.sub("", word.lower())
    return rhyme_dict.get(clean, [])


def words_rhyme(w1: str, w2: str, rhyme_dict: dict) -> bool:
    if not w1 or not w2:
        return False
    clean1 = STRIP_PUNCT.sub("", w1.lower())
    clean2 = STRIP_PUNCT.sub("", w2.lower())
    if clean1 == clean2:
        return False  # identical words don't count as a rhyme
    tails1 = set(get_rhyme_tails(w1, rhyme_dict))
    tails2 = set(get_rhyme_tails(w2, rhyme_dict))
    return bool(tails1 & tails2)


def get_lines(poem_text: str) -> list[str]:
    return [l.strip() for l in poem_text.split("\n") if l.strip()]


def is_multiline(poem_text: str) -> bool:
    return len(get_lines(poem_text)) > 1


def assign_era(year: float) -> str | None:
    for label, lo, hi in ERA_BINS:
        if lo <= year < hi:
            return label
    return None


def last_word(line: str) -> str:
    words = line.split()
    return words[-1] if words else ""


def rhyme_rate(lines: list[str], rhyme_dict: dict) -> float:
    """
    Check all pairs of lines separated by 0 or 1 line (AABB and ABAB patterns).
    Returns fraction of eligible pairs that rhyme.
    """
    if len(lines) < 2:
        return np.nan

    ends = [last_word(l) for l in lines]
    # Only check pairs where both words are in the dictionary
    pairs_checked = 0
    pairs_rhymed = 0

    for gap in (1, 2):  # adjacent (AABB) and alternating (ABAB)
        for i in range(len(ends) - gap):
            w1, w2 = ends[i], ends[i + gap]
            t1 = get_rhyme_tails(w1, rhyme_dict)
            t2 = get_rhyme_tails(w2, rhyme_dict)
            if not t1 or not t2:
                continue  # skip if either word missing from dict
            pairs_checked += 1
            if words_rhyme(w1, w2, rhyme_dict):
                pairs_rhymed += 1

    if pairs_checked == 0:
        return np.nan
    return pairs_rhymed / pairs_checked


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading CMU Pronouncing Dictionary...")
    rhyme_dict = build_rhyme_dict()
    print(f"  {len(rhyme_dict):,} words loaded")

    df = pd.read_csv(CSV_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["era"] = df["Year"].apply(assign_era)
    df = df[df["Poem"].apply(is_multiline)].copy()
    print(f"Processing {len(df)} multi-line poems\n")

    rows = []
    for _, r in df.iterrows():
        lines = get_lines(r["Poem"])
        rows.append({
            "era": r["era"],
            "rhyme_rate": rhyme_rate(lines, rhyme_dict),
        })
    results = pd.DataFrame(rows)

    print("=" * 100)
    print("RHYME RATE PER ERA")
    print("=" * 100)
    for era_label, _, _ in ERA_BINS:
        vals = results.loc[results["era"] == era_label, "rhyme_rate"].dropna()
        if len(vals) == 0:
            continue
        print(f"  {era_label:<12s}  mean={vals.mean():.3f}  var={vals.var():.4f}  n={len(vals)}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    fig.suptitle("Rhyme Rate (fraction of line-end pairs that rhyme)", fontsize=14, fontweight="bold")

    for ax, (era_label, _, _) in zip(axes, ERA_BINS):
        vals = results.loc[
            results["rhyme_rate"].notna() & (results["era"] == era_label), "rhyme_rate"
        ].values
        if len(vals) == 0:
            ax.set_title(era_label)
            continue

        ax.hist(vals, bins=25, density=True, alpha=0.6, edgecolor="black", label="data")

        m, v = vals.mean(), vals.var()
        if v > 0 and 0 < m < 1:
            try:
                a_fit, b_fit, loc_fit, scale_fit = stats.beta.fit(vals, floc=0, fscale=1)
                x = np.linspace(0.01, 0.99, 200)
                ax.plot(x, stats.beta.pdf(x, a_fit, b_fit), "r-", lw=2,
                        label=f"Beta({a_fit:.1f},{b_fit:.1f})")
            except Exception:
                pass

        ax.set_title(f"{era_label}\nmean={m:.3f}, var={v:.4f}", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = f"{PLOT_DIR}/rhyme_rate_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
