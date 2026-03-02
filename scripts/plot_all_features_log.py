"""
Compute all 6 features on all multi-line poems and plot log1p-transformed
distributions per era.

Features:
  1. cap_rate          - fraction of lines starting uppercase
  2. punct_rate        - fraction of lines ending with punctuation
  3. colon_density     - colons/semicolons per 100 words
  4. concrete_abstract_ratio - concrete words / (abstract words + 1)
  5. adv_verb_ratio    - adverbs / (verbs + 1)
  6. rhyme_rate        - fraction of line-end pairs that rhyme
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import nltk

CSV_PATH = "data/PoetryFoundationData_with_year.csv"
CONCRETENESS_PATH = "data/concreteness_ratings.csv"
PLOT_DIR = "data/plots"

ERA_BINS = [
    ("Pre-1800", -9999, 1800),
    ("1800-1900", 1800, 1900),
    ("1900-1950", 1900, 1950),
    ("Post-1950", 1950, 9999),
]

PUNCT_CHARS = set(".,;:!?—–-")
STRIP_RE = re.compile(r"[^a-z\s]")
STRIP_PUNCT = re.compile(r"[^a-z]")
ADV_TAGS = {"RB", "RBR", "RBS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}


# ── helpers ──────────────────────────────────────────────────────────────────

def assign_era(year: float) -> str | None:
    for label, lo, hi in ERA_BINS:
        if lo <= year < hi:
            return label
    return None


def get_lines(text: str) -> list[str]:
    return [l.strip() for l in text.split("\n") if l.strip()]


def is_multiline(text: str) -> bool:
    return len(get_lines(text)) > 1


# ── feature 1 & 2 ─────────────────────────────────────────────────────────

def cap_rate(lines: list[str]) -> float:
    if not lines:
        return np.nan
    return sum(1 for l in lines if l and l[0].isupper()) / len(lines)


def punct_rate(lines: list[str]) -> float:
    if not lines:
        return np.nan
    return sum(1 for l in lines if l and l[-1] in PUNCT_CHARS) / len(lines)


# ── feature 3 ─────────────────────────────────────────────────────────────

def colon_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for ch in text if ch in (";", ":")) / len(words) * 100


# ── feature 4 ─────────────────────────────────────────────────────────────

def load_concreteness(path: str) -> dict[str, float]:
    lex = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                word, score = line.rsplit(",", 1)
                lex[word.lower()] = float(score)
    return lex


def tokenize(text: str) -> list[str]:
    t = STRIP_RE.sub(" ", text.lower())
    return [w for w in t.split() if w.isalpha() and len(w) > 1]


def concrete_abstract_ratio(tokens: list[str], lex: dict) -> float:
    concrete = sum(1 for t in tokens if t in lex and lex[t] >= 4.0)
    abstract = sum(1 for t in tokens if t in lex and lex[t] <= 2.0)
    if concrete == 0 and abstract == 0:
        return np.nan
    return concrete / (abstract + 1)


# ── feature 5 ─────────────────────────────────────────────────────────────

def adv_verb_ratio(text: str) -> float:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    adv = sum(1 for _, tag in tagged if tag in ADV_TAGS)
    verb = sum(1 for _, tag in tagged if tag in VERB_TAGS)
    return adv / (verb + 1)


# ── feature 6 ─────────────────────────────────────────────────────────────

def build_rhyme_dict() -> dict:
    cmu = nltk.corpus.cmudict.dict()
    rd = {}
    for word, pronunciations in cmu.items():
        tails = set()
        for phones in pronunciations:
            for i in range(len(phones) - 1, -1, -1):
                if phones[i][-1] in ("1", "2"):
                    tails.add(tuple(phones[i:]))
                    break
        if tails:
            rd[word] = list(tails)
    return rd


def get_rhyme_tails(word: str, rd: dict) -> list:
    return rd.get(STRIP_PUNCT.sub("", word.lower()), [])


def rhyme_rate(lines: list[str], rd: dict) -> float:
    if len(lines) < 2:
        return np.nan
    ends = [l.split()[-1] if l.split() else "" for l in lines]
    checked = rhymed = 0
    for gap in (1, 2):
        for i in range(len(ends) - gap):
            w1, w2 = ends[i], ends[i + gap]
            t1, t2 = get_rhyme_tails(w1, rd), get_rhyme_tails(w2, rd)
            if not t1 or not t2:
                continue
            checked += 1
            clean1 = STRIP_PUNCT.sub("", w1.lower())
            clean2 = STRIP_PUNCT.sub("", w2.lower())
            if clean1 != clean2 and set(map(tuple, t1)) & set(map(tuple, t2)):
                rhymed += 1
    return rhymed / checked if checked else np.nan


# ── main ──────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading resources...")
    lex = load_concreteness(CONCRETENESS_PATH)
    rd = build_rhyme_dict()
    print(f"  Concreteness: {len(lex):,} words  |  CMU dict: {len(rd):,} words")

    df = pd.read_csv(CSV_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["era"] = df["Year"].apply(assign_era)
    df = df[df["Poem"].apply(is_multiline)].copy()
    total = len(df)
    print(f"Processing {total} multi-line poems...\n")

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"  {i}/{total}")
        text = r["Poem"]
        lines = get_lines(text)
        tokens = tokenize(text)
        rows.append({
            "era": r["era"],
            "cap_rate": cap_rate(lines),
            "punct_rate": punct_rate(lines),
            "colon_density": colon_density(text),
            "concrete_abstract_ratio": concrete_abstract_ratio(tokens, lex),
            "adv_verb_ratio": adv_verb_ratio(text),
            "rhyme_rate": rhyme_rate(lines, rd),
        })
    print(f"  {total}/{total} done\n")
    results = pd.DataFrame(rows)

    # ── print raw stats ───────────────────────────────────────────────────
    feats = ["cap_rate", "punct_rate", "colon_density",
             "concrete_abstract_ratio", "adv_verb_ratio", "rhyme_rate"]
    print("=" * 100)
    print("RAW MEAN / VARIANCE PER ERA")
    print("=" * 100)
    for feat in feats:
        print(f"\n  {feat}:")
        for era_label, _, _ in ERA_BINS:
            vals = results.loc[results["era"] == era_label, feat].dropna()
            if len(vals):
                print(f"    {era_label:<12s}  mean={vals.mean():.4f}  var={vals.var():.5f}  n={len(vals)}")

    # ── plot log1p-transformed distributions ─────────────────────────────
    feat_titles = {
        "cap_rate": "Line-Initial Capitalisation Rate  [log1p]",
        "punct_rate": "End-of-Line Punctuation Rate  [log1p]",
        "colon_density": "Colon/Semicolon per 100 Words  [log1p]",
        "concrete_abstract_ratio": "Concrete-to-Abstract Ratio  [log1p]",
        "adv_verb_ratio": "Adverb-to-Verb Ratio  [log1p]",
        "rhyme_rate": "Rhyme Rate  [log1p]",
    }

    for feat in feats:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        fig.suptitle(feat_titles[feat], fontsize=13, fontweight="bold")

        for ax, (era_label, _, _) in zip(axes, ERA_BINS):
            raw = results.loc[
                results[feat].notna() & (results["era"] == era_label), feat
            ].values
            if len(raw) == 0:
                ax.set_title(era_label)
                continue

            vals = np.log1p(raw)
            m, v = vals.mean(), vals.var()

            ax.hist(vals, bins=25, density=True, alpha=0.6, edgecolor="black", label="data")

            # fit Normal to the log-transformed values
            if v > 0:
                try:
                    mu, sigma = stats.norm.fit(vals)
                    x = np.linspace(vals.min() - 0.2, vals.max() + 0.2, 200)
                    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2,
                            label=f"N({mu:.2f},{sigma:.2f})")
                except Exception:
                    pass

            ax.set_title(f"{era_label}\nmean={m:.3f}, var={v:.4f}", fontsize=10)
            ax.set_xlabel("log1p(value)")
            ax.legend(fontsize=8)

        plt.tight_layout()
        out = f"{PLOT_DIR}/{feat}_log_distribution.png"
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close()


if __name__ == "__main__":
    main()
