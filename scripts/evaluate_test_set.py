"""
Final test-set evaluation for the 3-class poetry era classifier.

Trains all models on data/splits/3class_train.csv,
evaluates on the held-out data/splits/3class_test.csv,
and writes plots + tables to results/.

Run:
    python scripts/evaluate_test_set.py
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_curve, auc,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
TRAIN_CSV    = "data/splits/3class_train.csv"
TEST_CSV     = "data/splits/3class_test.csv"
ERA_ORDER    = ["Pre-1800", "1800-1900", "Post-1900"]
ERA_LABELS   = ["Pre-1800", "1800–1900", "Post-1900"]   # display labels
RANDOM_STATE = 42

FEATURE_COLS = [
    "cap_rate", "punct_rate", "colon_density",
    "concrete_abstract_ratio", "adv_verb_ratio", "rhyme_rate",
    "archaic_density", "imageability",
    "uses_rhyme", "uses_colons", "caps_all", "high_punct", "uses_archaism",
]
RAW_FEATS = [
    "cap_rate", "punct_rate", "colon_density",
    "concrete_abstract_ratio", "adv_verb_ratio", "rhyme_rate",
    "archaic_density",
]
FEATURE_NAMES = {
    "cap_rate":                "Capitalisation rate",
    "punct_rate":              "Punctuation rate",
    "colon_density":           "Colon/semicolon density",
    "concrete_abstract_ratio": "Concrete-to-abstract ratio",
    "adv_verb_ratio":          "Adverb-to-verb ratio",
    "rhyme_rate":              "Rhyme rate",
    "archaic_density":         "Archaic word density",
    "imageability":            "Imageability",
    "uses_rhyme":              "Uses rhyme (binary)",
    "uses_colons":             "Uses colons (binary)",
    "caps_all":                "All-caps lines (binary)",
    "high_punct":              "High punctuation (binary)",
    "uses_archaism":           "Uses archaism (binary)",
}

PALETTE = {
    "Pre-1800":  "#4e79a7",
    "1800-1900": "#f28e2b",
    "Post-1900": "#59a14f",
}
MODEL_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]


# ── Bayesian Hurdle NB ────────────────────────────────────────────────────────
class BayesianHurdleNB:
    HURDLE_FEATURES = {
        "rhyme_rate":       {"gate_is_upper": False, "threshold": 0,    "dist": "beta"},
        "colon_density":    {"gate_is_upper": False, "threshold": 0,    "dist": "gamma"},
        "cap_rate":         {"gate_is_upper": True,  "threshold": 0.95, "dist": "beta"},
        "punct_rate":       {"gate_is_upper": True,  "threshold": 0.90, "dist": "beta"},
        "archaic_density":  {"gate_is_upper": False, "threshold": 0,    "dist": "gamma"},
    }
    NON_HURDLE = ["concrete_abstract_ratio", "adv_verb_ratio"]

    def __init__(self, uniform_prior=True):
        self.uniform_prior = uniform_prior
        self.classes_ = None
        self.log_priors_ = {}
        self.hurdle_params_ = {}
        self.non_hurdle_params_ = {}

    def fit(self, df, era_order, y_col="era"):
        self.classes_ = era_order
        if self.uniform_prior:
            for c in self.classes_:
                self.log_priors_[c] = np.log(1.0 / len(self.classes_))
        else:
            cc = df[y_col].value_counts(); total = len(df)
            for c in self.classes_:
                self.log_priors_[c] = np.log(cc.get(c, 1) / total)

        for feat, cfg in self.HURDLE_FEATURES.items():
            self.hurdle_params_[feat] = {}
            for c in self.classes_:
                vals = df.loc[df[y_col] == c, feat].dropna().values.astype(float)
                if len(vals) == 0:
                    self.hurdle_params_[feat][c] = None; continue
                if cfg["gate_is_upper"]:
                    p_gate = np.mean(vals >= cfg["threshold"])
                    cond   = vals[vals < cfg["threshold"]]
                else:
                    p_gate = np.mean(vals > cfg["threshold"])
                    cond   = vals[vals > cfg["threshold"]]
                p_gate = np.clip(p_gate, 1e-6, 1 - 1e-6)
                dist_p = None
                if len(cond) >= 2:
                    try:
                        if cfg["dist"] == "beta":
                            a, b, _, _ = stats.beta.fit(np.clip(cond, 1e-6, 1-1e-6), floc=0, fscale=1)
                            dist_p = ("beta", a, b)
                        else:
                            sh, _, sc = stats.gamma.fit(np.maximum(cond, 1e-6), floc=0)
                            dist_p = ("gamma", sh, sc)
                    except Exception:
                        pass
                self.hurdle_params_[feat][c] = {
                    "p_gate": p_gate, "gate_is_upper": cfg["gate_is_upper"],
                    "threshold": cfg["threshold"], "dist_params": dist_p,
                }

        for feat in self.NON_HURDLE:
            self.non_hurdle_params_[feat] = {}
            for c in self.classes_:
                vals = df.loc[df[y_col] == c, feat].dropna().values.astype(float)
                if len(vals) < 2:
                    self.non_hurdle_params_[feat][c] = None; continue
                try:
                    sh, _, sc = stats.gamma.fit(np.maximum(vals, 1e-6), floc=0)
                    self.non_hurdle_params_[feat][c] = ("gamma", sh, sc)
                except Exception:
                    self.non_hurdle_params_[feat][c] = None
        return self

    def _ll_hurdle(self, x, p):
        if p is None: return 0.0
        in_gate = (x >= p["threshold"]) if p["gate_is_upper"] else (x > p["threshold"])
        if in_gate: return np.log(p["p_gate"])
        ll = np.log(1 - p["p_gate"])
        if p["dist_params"]:
            dt = p["dist_params"][0]
            if dt == "beta":
                ll += stats.beta.logpdf(np.clip(x, 1e-6, 1-1e-6), p["dist_params"][1], p["dist_params"][2])
            else:
                ll += stats.gamma.logpdf(max(x, 1e-6), p["dist_params"][1], loc=0, scale=p["dist_params"][2])
        return ll

    def _ll_nh(self, x, p):
        if p is None or np.isnan(x): return 0.0
        _, sh, sc = p
        return stats.gamma.logpdf(max(x, 1e-6), sh, loc=0, scale=sc)

    def predict(self, df):
        preds = []
        for _, row in df.iterrows():
            lps = {}
            for c in self.classes_:
                lp = self.log_priors_[c]
                for feat in self.HURDLE_FEATURES:
                    if not np.isnan(row[feat]):
                        lp += self._ll_hurdle(row[feat], self.hurdle_params_[feat][c])
                for feat in self.NON_HURDLE:
                    lp += self._ll_nh(row[feat], self.non_hurdle_params_[feat][c])
                lps[c] = lp
            preds.append(max(lps, key=lps.get))
        return np.array(preds)

    def predict_proba(self, df):
        rows = []
        for _, row in df.iterrows():
            lps = {}
            for c in self.classes_:
                lp = self.log_priors_[c]
                for feat in self.HURDLE_FEATURES:
                    if not np.isnan(row[feat]):
                        lp += self._ll_hurdle(row[feat], self.hurdle_params_[feat][c])
                for feat in self.NON_HURDLE:
                    lp += self._ll_nh(row[feat], self.non_hurdle_params_[feat][c])
                lps[c] = lp
            arr = np.array([lps[c] for c in self.classes_])
            arr -= arr.max(); arr = np.exp(arr); arr /= arr.sum()
            rows.append(arr)
        return np.array(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────
def savefig(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion(y_true, y_pred, title, path):
    cm  = confusion_matrix(y_true, y_pred, labels=ERA_ORDER)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, data, fmt, subtitle in [
        (axes[0], cm,  "d",    "Counts"),
        (axes[1], cmn, ".2f",  "Recall-normalised"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(ERA_ORDER))); ax.set_xticklabels(ERA_LABELS, rotation=20, ha="right")
        ax.set_yticks(range(len(ERA_ORDER))); ax.set_yticklabels(ERA_LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(subtitle, fontsize=11)
        for i in range(len(ERA_ORDER)):
            for j in range(len(ERA_ORDER)):
                val = data[i, j]
                txt = f"{val:{fmt}}"
                colour = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=12, fontweight="bold", color=colour)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    savefig(path)


def plot_per_class_metrics(results, path):
    """Grouped bar chart: precision, recall, F1 per era for each model."""
    models = list(results.keys())
    metrics = ["precision", "recall", "f1-score"]
    metric_labels = ["Precision", "Recall", "F1"]
    n_models = len(models)
    n_eras   = len(ERA_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Per-class metrics on test set", fontsize=13, fontweight="bold")

    x = np.arange(n_eras)
    width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        for i, (mname, report) in enumerate(results.items()):
            vals = [report[era][metric] for era in ERA_ORDER]
            bars = ax.bar(x + offsets[i], vals, width,
                          label=mname, color=MODEL_COLORS[i], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=45)
        ax.set_title(mlabel, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(ERA_LABELS, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Score")
    handles = [mpatches.Patch(color=MODEL_COLORS[i], label=m) for i, m in enumerate(models)]
    fig.legend(handles=handles, loc="lower center", ncol=n_models,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    savefig(path)


def plot_model_summary(summary_df, path):
    """Grouped bar: accuracy + macro F1 for each model."""
    models = summary_df["Model"].tolist()
    acc    = summary_df["Accuracy"].tolist()
    mf1    = summary_df["Macro F1"].tolist()

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#4e79a7", alpha=0.9, edgecolor="white")
    b2 = ax.bar(x + w/2, mf1, w, label="Macro F1",  color="#f28e2b", alpha=0.9, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — test set (3-class)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(path)


def plot_feature_importance(names, importances, title, path, color="#4e79a7"):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh([names[i] for i in order], importances[order],
                   color=color, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, importances[order]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    savefig(path)


def plot_roc_curves(model_probas, y_test, path):
    """One-vs-rest ROC for each era, each model."""
    y_bin = label_binarize(y_test, classes=ERA_ORDER)
    n_eras = len(ERA_ORDER)

    fig, axes = plt.subplots(1, n_eras, figsize=(15, 5), sharey=True)
    fig.suptitle("ROC curves (one-vs-rest) — test set", fontsize=13, fontweight="bold")

    for col, (era, era_label) in enumerate(zip(ERA_ORDER, ERA_LABELS)):
        ax = axes[col]
        for i, (mname, proba) in enumerate(model_probas.items()):
            fpr, tpr, _ = roc_curve(y_bin[:, col], proba[:, col])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=MODEL_COLORS[i], lw=2,
                    label=f"{mname}  (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(era_label, fontsize=11)
        ax.set_xlabel("False Positive Rate")
        if col == 0:
            ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(path)


def plot_summary_table(summary_df, per_class_results, path):
    """Render a clean summary table as a figure."""
    rows = []
    for _, r in summary_df.iterrows():
        mname = r["Model"]
        report = per_class_results[mname]
        for era, elabel in zip(ERA_ORDER, ERA_LABELS):
            rows.append({
                "Model": mname,
                "Era": elabel,
                "Precision": f"{report[era]['precision']:.3f}",
                "Recall":    f"{report[era]['recall']:.3f}",
                "F1":        f"{report[era]['f1-score']:.3f}",
                "Support":   int(report[era]['support']),
            })

    df = pd.DataFrame(rows)
    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(12, 0.45 * n_rows + 1.5))
    ax.axis("off")

    col_labels = ["Model", "Era", "Precision", "Recall", "F1", "Support"]
    table = ax.table(
        cellText=df.values, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    header_color = "#2c3e50"
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color="white", fontweight="bold")

    era_palette = {"Pre-1800": "#dce9f5", "1800–1900": "#fde8cc", "Post-1900": "#d5edd6"}
    for i, (_, row_data) in enumerate(df.iterrows(), start=1):
        bg = era_palette.get(row_data["Era"], "white")
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg if j > 0 else "white")

    ax.set_title("Per-class classification report — test set", fontsize=12,
                 fontweight="bold", pad=12)
    savefig(path, tight=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n{'='*70}")
    print("  3-class poetry era classifier — final test-set evaluation")
    print(f"{'='*70}")

    # Load imageability lexicon
    glasgow = pd.read_csv("data/glasgow_norms.csv").set_index("word")
    img_lex = glasgow["imageability"].to_dict()

    def _img_score(text):
        ws = re.findall(r"[a-z']+", str(text).lower())
        vs = [img_lex[w] for w in ws if w in img_lex]
        return float(np.mean(vs)) if len(vs) >= 5 else float("nan")

    poem_df = pd.read_csv("data/PoetryFoundationData_with_year.csv")[["Title","Poet","Poem"]]
    poem_df["imageability"] = poem_df["Poem"].apply(_img_score)
    img_lookup = poem_df.set_index(["Title","Poet"])["imageability"].to_dict()

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    if "era_3class" in train_df.columns:
        train_df["era"] = train_df["era_3class"]
        test_df["era"]  = test_df["era_3class"]

    # Add imageability; fill missing with era mean from train
    for df in [train_df, test_df]:
        df["imageability"] = df.apply(
            lambda r: img_lookup.get((r["Title"], r["Poet"]), float("nan")), axis=1)
    for era in ERA_ORDER:
        m = train_df.loc[train_df["era"] == era, "imageability"].mean()
        for df in [train_df, test_df]:
            df.loc[(df["era"] == era) & df["imageability"].isna(), "imageability"] = m

    y_train = train_df["era"].values
    y_test  = test_df["era"].values

    print(f"\nTrain: {len(train_df):,}  |  Test: {len(test_df):,}")
    print("\nTest set class distribution:")
    for era in ERA_ORDER:
        n = (test_df["era"] == era).sum()
        print(f"  {era:<14s}  {n:>4d}  ({n/len(test_df):.1%})")

    X_train = train_df[FEATURE_COLS].fillna(0).values
    X_test  = test_df[FEATURE_COLS].fillna(0).values

    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_te_sc   = scaler.transform(X_test)

    le = LabelEncoder(); le.fit(ERA_ORDER)
    y_tr_enc = le.transform(y_train)
    y_te_enc = le.transform(y_test)
    class_counts  = np.bincount(y_tr_enc)
    sample_weights = np.array([len(y_tr_enc) / (len(ERA_ORDER) * class_counts[y]) for y in y_tr_enc])

    # ── Train all models ──────────────────────────────────────────────────────
    print("\nTraining models…")

    bnb = BayesianHurdleNB(uniform_prior=True)
    bnb.fit(train_df[RAW_FEATS + ["era"]], ERA_ORDER)

    lr = LogisticRegression(solver="lbfgs", class_weight="balanced",
                            max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_tr_sc, y_train)

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="mlogloss", random_state=RANDOM_STATE,
                        n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_tr_enc, sample_weight=sample_weights)

    # ── Predict ───────────────────────────────────────────────────────────────
    models = {
        "BH-NB":              (bnb.predict(test_df[RAW_FEATS]),
                               bnb.predict_proba(test_df[RAW_FEATS])),
        "Logistic Regression":(lr.predict(X_te_sc),
                               lr.predict_proba(X_te_sc)),
        "Random Forest":      (rf.predict(X_test),
                               rf.predict_proba(X_test)),
        "XGBoost":            (le.inverse_transform(xgb.predict(X_test)),
                               xgb.predict_proba(X_test)),
    }

    # Align proba columns to ERA_ORDER
    def align_proba(model_obj, proba, model_type):
        if model_type == "xgb":
            # xgb classes are encoded integers; columns already in le order
            idx = [list(ERA_ORDER).index(c) for c in ERA_ORDER]
            return proba[:, [le.transform([c])[0] for c in ERA_ORDER]]
        classes = list(model_obj.classes_)
        return proba[:, [classes.index(c) for c in ERA_ORDER]]

    model_probas = {
        "BH-NB":               models["BH-NB"][1],
        "Logistic Regression": align_proba(lr,  models["Logistic Regression"][1], "lr"),
        "Random Forest":       align_proba(rf,  models["Random Forest"][1],       "rf"),
        "XGBoost":             models["XGBoost"][1][:, [le.transform([c])[0] for c in ERA_ORDER]],
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    summary_rows = []
    per_class_results = {}

    for mname, (y_pred, _) in models.items():
        acc  = accuracy_score(y_test, y_pred)
        mf1  = f1_score(y_test, y_pred, labels=ERA_ORDER, average="macro", zero_division=0)
        report = classification_report(y_test, y_pred, labels=ERA_ORDER,
                                       output_dict=True, zero_division=0)
        per_class_results[mname] = report
        summary_rows.append({"Model": mname, "Accuracy": acc, "Macro F1": mf1})

        print(f"\n{'─'*50}")
        print(f"  {mname}")
        print(f"  Accuracy: {acc:.4f}  |  Macro F1: {mf1:.4f}  ({int(acc*len(y_test))}/{len(y_test)})")
        print(classification_report(y_test, y_pred, labels=ERA_ORDER,
                                    target_names=ERA_LABELS, digits=3, zero_division=0))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{RESULTS_DIR}/summary.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR}/summary.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")

    for mname, (y_pred, _) in models.items():
        slug = mname.lower().replace(" ", "_").replace("-", "")
        plot_confusion(y_test, y_pred, f"{mname} — confusion matrix (test set)",
                       f"{RESULTS_DIR}/confusion_{slug}.png")

    plot_model_summary(summary_df, f"{RESULTS_DIR}/model_comparison.png")
    plot_per_class_metrics(per_class_results, f"{RESULTS_DIR}/per_class_metrics.png")
    plot_roc_curves(model_probas, y_test, f"{RESULTS_DIR}/roc_curves.png")
    plot_summary_table(summary_df, per_class_results, f"{RESULTS_DIR}/report_table.png")

    # RF feature importances
    rf_imp = rf.feature_importances_
    plot_feature_importance(
        [FEATURE_NAMES[c] for c in FEATURE_COLS], rf_imp,
        "Random Forest — feature importances (test set)",
        f"{RESULTS_DIR}/feature_importance_rf.png", color="#4e79a7",
    )

    # XGBoost feature importances
    xgb_imp = xgb.feature_importances_
    plot_feature_importance(
        [FEATURE_NAMES[c] for c in FEATURE_COLS], xgb_imp,
        "XGBoost — feature importances (gain, test set)",
        f"{RESULTS_DIR}/feature_importance_xgb.png", color="#59a14f",
    )

    print(f"\n{'='*70}")
    print(f"  Done. All outputs saved to {RESULTS_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
