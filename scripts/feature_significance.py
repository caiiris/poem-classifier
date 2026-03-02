"""
Statistical significance testing for all 6 stylistic features across 4 eras.

Tests used (all non-parametric, appropriate for zero-inflated / heavy-tailed data):
  - Kruskal-Wallis H-test: overall test across all 4 eras
  - Pairwise Mann-Whitney U tests: which specific era pairs differ
  - Bonferroni correction: p-values multiplied by 6 (number of pairs)
  - Rank-biserial r: effect size  (0.1=small, 0.3=medium, 0.5=large)
"""

import re
import sys
import numpy as np
import pandas as pd
from scipy import stats
import nltk

CSV_PATH = "data/PoetryFoundationData_with_year.csv"
CONCRETENESS_PATH = "data/concreteness_ratings.csv"

ERA_BINS = [
    ("Pre-1800",  -9999, 1800),
    ("1800-1900",  1800, 1900),
    ("1900-1950",  1900, 1950),
    ("Post-1950",  1950, 9999),
]
ERA_LABELS = [e[0] for e in ERA_BINS]

PUNCT_CHARS  = set(".,;:!?—–-")
STRIP_RE     = re.compile(r"[^a-z\s]")
STRIP_PUNCT  = re.compile(r"[^a-z]")
ADV_TAGS     = {"RB", "RBR", "RBS"}
VERB_TAGS    = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
N_PAIRS      = 6   # C(4,2)


# ── helpers ──────────────────────────────────────────────────────────────────

def assign_era(year):
    for label, lo, hi in ERA_BINS:
        if lo <= year < hi:
            return label
    return None


def get_lines(text):
    return [l.strip() for l in text.split("\n") if l.strip()]


def is_multiline(text):
    return len(get_lines(text)) > 1


# ── features ─────────────────────────────────────────────────────────────────

def cap_rate(lines):
    if not lines:
        return np.nan
    return sum(1 for l in lines if l and l[0].isupper()) / len(lines)


def punct_rate(lines):
    if not lines:
        return np.nan
    return sum(1 for l in lines if l and l[-1] in PUNCT_CHARS) / len(lines)


def colon_density(text):
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for ch in text if ch in (";", ":")) / len(words) * 100


def load_concreteness(path):
    lex = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                word, score = line.rsplit(",", 1)
                lex[word.lower()] = float(score)
    return lex


def tokenize(text):
    t = STRIP_RE.sub(" ", text.lower())
    return [w for w in t.split() if w.isalpha() and len(w) > 1]


def concrete_abstract_ratio(tokens, lex):
    concrete = sum(1 for t in tokens if t in lex and lex[t] >= 4.0)
    abstract = sum(1 for t in tokens if t in lex and lex[t] <= 2.0)
    if concrete == 0 and abstract == 0:
        return np.nan
    return concrete / (abstract + 1)


def adv_verb_ratio(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    adv  = sum(1 for _, tag in tagged if tag in ADV_TAGS)
    verb = sum(1 for _, tag in tagged if tag in VERB_TAGS)
    return adv / (verb + 1)


def build_rhyme_dict():
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


def get_rhyme_tails(word, rd):
    return rd.get(STRIP_PUNCT.sub("", word.lower()), [])


def rhyme_rate(lines, rd):
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
            c1 = STRIP_PUNCT.sub("", w1.lower())
            c2 = STRIP_PUNCT.sub("", w2.lower())
            if c1 != c2 and set(map(tuple, t1)) & set(map(tuple, t2)):
                rhymed += 1
    return rhymed / checked if checked else np.nan


# ── statistics ────────────────────────────────────────────────────────────────

def effect_size_label(r):
    r = abs(r)
    if r >= 0.5:
        return "large"
    if r >= 0.3:
        return "medium"
    if r >= 0.1:
        return "small"
    return "negligible"


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def run_tests(era_data: dict[str, np.ndarray], feat_name: str):
    """Run Kruskal-Wallis + all pairwise Mann-Whitney U with Bonferroni."""

    groups = [era_data[label].dropna().values for label in ERA_LABELS]

    # Kruskal-Wallis
    h_stat, kw_p = stats.kruskal(*[g for g in groups if len(g) > 0])

    print(f"\nFeature: {feat_name}")
    print(f"  Kruskal-Wallis: H={h_stat:.2f}, p={kw_p:.2e}  {sig_stars(kw_p)}")
    print(f"  Pairwise Mann-Whitney U (Bonferroni-corrected, n_pairs={N_PAIRS}):")
    print(f"    {'Pair':<30s}  {'U':>10s}  {'p_corr':>10s}  {'sig':>4s}  {'r':>6s}  effect")

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for i, j in pairs:
        g1, g2 = groups[i], groups[j]
        if len(g1) == 0 or len(g2) == 0:
            continue
        u_stat, raw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        p_corr = min(raw_p * N_PAIRS, 1.0)
        r = 1 - (2 * u_stat) / (len(g1) * len(g2))
        pair_label = f"{ERA_LABELS[i]} vs {ERA_LABELS[j]}"
        print(f"    {pair_label:<30s}  {u_stat:>10.0f}  {p_corr:>10.4f}  {sig_stars(p_corr):>4s}  "
              f"{r:>+.3f}  {effect_size_label(r)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading resources...")
    lex = load_concreteness(CONCRETENESS_PATH)
    rd  = build_rhyme_dict()
    print(f"  Concreteness: {len(lex):,} words  |  CMU dict: {len(rd):,} words")

    df = pd.read_csv(CSV_PATH)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["era"] = df["Year"].apply(assign_era)
    df = df[df["Poem"].apply(is_multiline)].copy()
    total = len(df)
    print(f"Computing features on {total} multi-line poems...\n")

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"  {i}/{total}", flush=True)
        text  = r["Poem"]
        lines = get_lines(text)
        toks  = tokenize(text)
        rows.append({
            "era":                    r["era"],
            "cap_rate":               cap_rate(lines),
            "punct_rate":             punct_rate(lines),
            "colon_density":          colon_density(text),
            "concrete_abstract_ratio": concrete_abstract_ratio(toks, lex),
            "adv_verb_ratio":         adv_verb_ratio(text),
            "rhyme_rate":             rhyme_rate(lines, rd),
        })
    print(f"  {total}/{total} done\n")

    results = pd.DataFrame(rows)

    feats = ["cap_rate", "punct_rate", "colon_density",
             "concrete_abstract_ratio", "adv_verb_ratio", "rhyme_rate"]

    print("=" * 90)
    print("SIGNIFICANCE TESTS (non-parametric)")
    print("=" * 90)

    for feat in feats:
        era_data = {label: results.loc[results["era"] == label, feat]
                    for label in ERA_LABELS}
        run_tests(era_data, feat)

    # ── summary table ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("SUMMARY: largest effect sizes per feature")
    print("=" * 90)
    print(f"  {'Feature':<28s}  {'KW p-value':>12s}  {'Max |r|':>8s}  "
          f"{'Best pair':<35s}  effect")

    for feat in feats:
        era_data = {label: results.loc[results["era"] == label, feat]
                    for label in ERA_LABELS}
        groups = [era_data[label].dropna().values for label in ERA_LABELS]
        _, kw_p = stats.kruskal(*[g for g in groups if len(g) > 0])

        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        best_r, best_pair = 0.0, ""
        for i, j in pairs:
            g1, g2 = groups[i], groups[j]
            if len(g1) == 0 or len(g2) == 0:
                continue
            u_stat, _ = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            r = abs(1 - (2 * u_stat) / (len(g1) * len(g2)))
            if r > best_r:
                best_r = r
                best_pair = f"{ERA_LABELS[i]} vs {ERA_LABELS[j]}"

        print(f"  {feat:<28s}  {kw_p:>12.2e}  {best_r:>8.3f}  "
              f"{best_pair:<35s}  {effect_size_label(best_r)}")


if __name__ == "__main__":
    main()
