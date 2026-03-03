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
from scipy import stats, optimize
import nltk

CSV_PATH = "data/PoetryFoundationData_with_year.csv"
CONCRETENESS_PATH = "data/concreteness_ratings.csv"
PLOT_DIR = "data/plots/preliminary"
PLOT_DIR_DIST_TRY = "data/plots/dist try 1"
PLOT_DIR_DIST_TRY_2 = "data/plots/dist try 2"

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


# ── ZOIB and zero-inflated Gamma (for dist try 2) ─────────────────────────

def _zoib_nll(params: np.ndarray, x: np.ndarray) -> float:
    """Negative log-likelihood for Zero-One-Inflated Beta. params = (l0, l1, log_a, log_b)."""
    l0, l1, log_a, log_b = params
    e0, e1 = np.exp(l0), np.exp(l1)
    p0 = e0 / (1 + e0 + e1)
    p1 = e1 / (1 + e0 + e1)
    p_mid = 1 - p0 - p1
    a, b = np.exp(log_a), np.exp(log_b)
    n0 = np.sum(x == 0)
    n1 = np.sum(x == 1)
    mid = (x > 0) & (x < 1)
    n_mid = np.sum(mid)
    nll = 0.0
    if n0 > 0:
        nll -= n0 * np.log(np.clip(p0, 1e-12, 1))
    if n1 > 0:
        nll -= n1 * np.log(np.clip(p1, 1e-12, 1))
    if n_mid > 0 and p_mid > 1e-12:
        x_mid = x[mid]
        nll -= n_mid * np.log(np.clip(p_mid, 1e-12, 1))
        nll -= np.sum(stats.beta.logpdf(np.clip(x_mid, 1e-8, 1 - 1e-8), a, b))
    if np.isnan(nll) or nll <= 0:
        return 1e10
    return nll


def fit_zoib(x: np.ndarray) -> tuple[float, float, float, float] | None:
    """Fit ZOIB. Returns (p0, p1, a, b) or None on failure."""
    x = np.asarray(x, dtype=float)
    n0, n1 = np.sum(x == 0), np.sum(x == 1)
    mid = (x > 0) & (x < 1)
    if np.sum(mid) < 2:
        return None
    # Initial: p0, p1 from empirical; a, b from Beta fit on (0,1) subset
    p0_init = n0 / len(x)
    p1_init = n1 / len(x)
    x_mid = x[mid]
    try:
        a_init, b_init, _, _ = stats.beta.fit(np.clip(x_mid, 1e-6, 1 - 1e-6), floc=0, fscale=1)
    except Exception:
        a_init, b_init = 2.0, 2.0
    l0_init = np.log(p0_init + 1e-6) - np.log(1 - p0_init - p1_init + 1e-6)
    l1_init = np.log(p1_init + 1e-6) - np.log(1 - p0_init - p1_init + 1e-6)
    try:
        res = optimize.minimize(
            _zoib_nll,
            x0=[l0_init, l1_init, np.log(a_init), np.log(b_init)],
            args=(x,),
            method="L-BFGS-B",
            bounds=[(-20, 20), (-20, 20), (-5, 10), (-5, 10)],
        )
        if not res.success:
            return None
        l0, l1, log_a, log_b = res.x
        e0, e1 = np.exp(l0), np.exp(l1)
        p0 = e0 / (1 + e0 + e1)
        p1 = e1 / (1 + e0 + e1)
        return (float(p0), float(p1), float(np.exp(log_a)), float(np.exp(log_b)))
    except Exception:
        return None


def fit_zero_inflated_gamma(x: np.ndarray) -> tuple[float, float, float] | None:
    """Fit zero-inflated Gamma: P(x=0)=p, and given x>0, x ~ Gamma(shape, scale). Returns (p, shape, scale) or None."""
    x = np.asarray(x, dtype=float)
    n0 = np.sum(x <= 0)
    pos = x > 0
    if np.sum(pos) < 2:
        return None
    p0 = n0 / len(x)
    try:
        shape, loc, scale = stats.gamma.fit(x[pos], floc=0)
        return (float(p0), float(shape), float(scale))
    except Exception:
        return None


def main():
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR_DIST_TRY, exist_ok=True)
    os.makedirs(PLOT_DIR_DIST_TRY_2, exist_ok=True)

    print("Loading resources...")
    lex = load_concreteness(CONCRETENESS_PATH)
    rd = build_rhyme_dict()
    print(f"  Concreteness: {len(lex):,} words  |  CMU dict: {len(rd):,} words")

    df = pd.read_csv(CSV_PATH)
    n_total = len(df)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    n_with_year = len(df)
    df["era"] = df["Year"].apply(assign_era)
    df = df[df["Poem"].apply(is_multiline)].copy()
    total = len(df)
    print(f"Processing {total} multi-line poems (dropped {n_total - n_with_year} without Year, {n_with_year - total} single-line; features need 2+ lines).\n")

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
    RATE_FEATURES = {"cap_rate", "punct_rate", "rhyme_rate"}  # [0,1] → fit Beta
    print("=" * 100)
    print("RAW MEAN / VARIANCE PER ERA")
    print("=" * 100)
    for feat in feats:
        print(f"\n  {feat}:")
        for era_label, _, _ in ERA_BINS:
            vals = results.loc[results["era"] == era_label, feat].dropna()
            if len(vals):
                print(f"    {era_label:<12s}  mean={vals.mean():.4f}  var={vals.var():.5f}  n={len(vals)}")

    # ── plot distributions (raw and log1p), fitted and unfitted ───────────
    feat_titles_log = {
        "cap_rate": "Line-Initial Capitalisation Rate  [log1p]",
        "punct_rate": "End-of-Line Punctuation Rate  [log1p]",
        "colon_density": "Colon/Semicolon per 100 Words  [log1p]",
        "concrete_abstract_ratio": "Concrete-to-Abstract Ratio  [log1p]",
        "adv_verb_ratio": "Adverb-to-Verb Ratio  [log1p]",
        "rhyme_rate": "Rhyme Rate  [log1p]",
    }
    feat_titles_raw = {
        "cap_rate": "Line-Initial Capitalisation Rate  [raw]",
        "punct_rate": "End-of-Line Punctuation Rate  [raw]",
        "colon_density": "Colon/Semicolon per 100 Words  [raw]",
        "concrete_abstract_ratio": "Concrete-to-Abstract Ratio  [raw]",
        "adv_verb_ratio": "Adverb-to-Verb Ratio  [raw]",
        "rhyme_rate": "Rhyme Rate  [raw]",
    }

    for out_dir, use_modern in [(PLOT_DIR_DIST_TRY, False), (PLOT_DIR_DIST_TRY_2, True)]:
        for feat in feats:
            for use_log in (True, False):
                titles = feat_titles_log if use_log else feat_titles_raw
                for with_fit in (True, False):
                    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
                    suffix = "fitted" if with_fit else "unfitted"
                    scale = "log" if use_log else "raw"
                    fig.suptitle(titles[feat] + f"  ({suffix})", fontsize=13, fontweight="bold")

                    for ax, (era_label, _, _) in zip(axes, ERA_BINS):
                        raw_vals = results.loc[
                            results[feat].notna() & (results["era"] == era_label), feat
                        ].values
                        if len(raw_vals) == 0:
                            ax.set_title(era_label)
                            continue

                        vals = np.log1p(raw_vals) if use_log else raw_vals.astype(float)
                        m, v = vals.mean(), vals.var()

                        ax.hist(vals, bins=25, density=True, alpha=0.6, edgecolor="black", label="data")

                        if with_fit and v > 0:
                            try:
                                if not use_log and feat in RATE_FEATURES:
                                    if use_modern:
                                        zoib = fit_zoib(raw_vals)
                                        if zoib is not None:
                                            p0, p1, a, b = zoib
                                            x_plot = np.linspace(max(1e-6, vals.min() - 0.02), min(1 - 1e-6, vals.max() + 0.02), 200)
                                            p_mid = 1 - p0 - p1
                                            ax.plot(x_plot, p_mid * stats.beta.pdf(x_plot, a, b), "r-", lw=2,
                                                    label=f"ZOIB p0={p0:.2f} p1={p1:.2f} Beta({a:.2f},{b:.2f})")
                                    else:
                                        r = np.clip(raw_vals, 1e-6, 1 - 1e-6)
                                        a, b, _loc, _scale = stats.beta.fit(r, floc=0, fscale=1)
                                        x_plot = np.linspace(max(1e-6, vals.min() - 0.02), min(1 - 1e-6, vals.max() + 0.02), 200)
                                        ax.plot(x_plot, stats.beta.pdf(x_plot, a, b), "r-", lw=2,
                                                label=f"Beta({a:.2f},{b:.2f})")
                                elif not use_log:
                                    if use_modern:
                                        zig = fit_zero_inflated_gamma(raw_vals)
                                        if zig is not None:
                                            p0, shape, scale_param = zig
                                            x_plot = np.linspace(max(1e-6, vals.min()), vals.max() + 0.1 * (vals.max() - vals.min() + 0.1), 200)
                                            ax.plot(x_plot, (1 - p0) * stats.gamma.pdf(x_plot, shape, loc=0, scale=scale_param), "r-", lw=2,
                                                    label=f"ZI-Gamma p0={p0:.2f} k={shape:.2f} scale={scale_param:.2f}")
                                    else:
                                        r = np.maximum(raw_vals.astype(float), 1e-6)
                                        shape, loc, scale_param = stats.gamma.fit(r, floc=0)
                                        x_plot = np.linspace(max(1e-6, vals.min()), vals.max() + 0.1 * (vals.max() - vals.min() + 0.1), 200)
                                        ax.plot(x_plot, stats.gamma.pdf(x_plot, shape, loc=0, scale=scale_param), "r-", lw=2,
                                                label=f"Gamma(k={shape:.2f}, scale={scale_param:.2f})")
                                else:
                                    mu, sigma = stats.norm.fit(vals)
                                    x_plot = np.linspace(vals.min() - 0.2, vals.max() + 0.2, 200)
                                    ax.plot(x_plot, stats.norm.pdf(x_plot, mu, sigma), "r-", lw=2,
                                            label=f"N({mu:.2f},{sigma:.2f})")
                            except Exception:
                                pass

                        ax.set_title(f"{era_label}\nmean={m:.3f}, var={v:.4f}", fontsize=10)
                        ax.set_xlabel("log1p(value)" if use_log else "value")
                        ax.legend(fontsize=8)

                    plt.tight_layout()
                    out = f"{out_dir}/{feat}_{scale}_{suffix}.png"
                    plt.savefig(out, dpi=150)
                    print(f"Saved: {out}")
                    plt.close()


if __name__ == "__main__":
    main()
