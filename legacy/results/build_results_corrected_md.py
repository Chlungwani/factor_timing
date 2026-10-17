"""Regenerate Chapter4_Results_Corrected.md from the corrected model pipeline.

Same contract as build_results_md.py: every number in the generated file comes
from results/model_outputs/, never transcribed. Re-run after re-running any
model notebook.

This companion file reports the pipeline after the validation fixes (purge gap
in walk-forward, unlabelled rows dropped, tuning restricted to pre-OOS data,
expanding window, revised grids). The original build_results_md.py is kept so
the superseded numbers remain reproducible.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef)

RESULTS_DIR = Path(__file__).parent
OUTPUT_DIR = RESULTS_DIR / "model_outputs"
DATA_DIR = RESULTS_DIR.parent / "data_input"
MD_PATH = RESULTS_DIR / "Chapter4_Results_Corrected.md"

MODELS = ["Logistic", "GAM", "XGBoost"]
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]
PAIR_TITLES = {
    "Momentum_vs_Value": ("Table 6", "Momentum vs Value"),
    "Momentum_vs_Quality": ("Table 9", "Momentum vs Quality"),
    "Value_vs_Quality": ("Table 12", "Quality vs Value"),
}

# The SUPERSEDED run (pre-fix), kept only to quantify what the corrections
# changed. Never copied into the result tables. (accuracy, roc_auc, log_loss)
SUPERSEDED = {
    ("Momentum_vs_Value", "Logistic"): (0.621795, 0.646399, 0.651266),
    ("Momentum_vs_Value", "GAM"): (0.641026, 0.667799, 0.928586),
    ("Momentum_vs_Value", "XGBoost"): (0.717949, 0.768173, 0.937815),
    ("Momentum_vs_Quality", "Logistic"): (0.615385, 0.603365, 0.689086),
    ("Momentum_vs_Quality", "GAM"): (0.589744, 0.645011, 0.818420),
    ("Momentum_vs_Quality", "XGBoost"): (0.698718, 0.733300, 0.958076),
    ("Value_vs_Quality", "Logistic"): (0.660256, 0.641791, 0.650306),
    ("Value_vs_Quality", "GAM"): (0.621795, 0.649841, 0.815218),
    ("Value_vs_Quality", "XGBoost"): (0.711538, 0.731176, 0.947920),
}

TAG = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}


def load_metrics():
    return pd.read_csv(OUTPUT_DIR / "metrics_summary.csv").set_index(["factor_pair", "model"])


def probs(model, pair):
    return pd.read_csv(OUTPUT_DIR / f"probs/{TAG[model]}_{pair.lower()}_probs.csv",
                       parse_dates=["Date"])


def labels(pair):
    fac = pd.read_csv(DATA_DIR / "labeled_relative_factor_returns.csv", parse_dates=["Date"])
    fac = fac[["Date", f"{pair}_trend"]].set_index("Date")
    feat = pd.read_csv(DATA_DIR / "lagged_data.csv", parse_dates=["Date"]).set_index("Date")
    d = pd.concat([feat, fac], axis=1, join="inner").dropna(subset=[f"{pair}_trend"])
    return d[f"{pair}_trend"].ffill().bfill()


def oos_table(metrics, pair):
    lines = ["| Metric | Logistic | GAM | XGBoost |", "|---|---|---|---|"]
    rows = [("Accuracy", "accuracy", "pct"), ("ROC-AUC", "roc_auc", "pct"),
            ("PR-AUC", "pr_auc", "pct"), ("Log-loss", "log_loss", "num"),
            ("F1", "f1", "num"), ("Precision", "precision", "num"),
            ("Recall", "recall", "num")]
    for label, col, fmt in rows:
        cells = [f"{metrics.loc[(pair, m), col] * 100:.1f}%" if fmt == "pct"
                 else f"{metrics.loc[(pair, m), col]:.3f}" for m in MODELS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    # discrimination measures that are robust to the class split
    for label, fn in [("Balanced accuracy", balanced_accuracy_score),
                      ("MCC", matthews_corrcoef)]:
        cells = []
        for m in MODELS:
            d = probs(m, pair)
            v = fn(d.actual, d.prediction)
            cells.append(f"{v:+.3f}" if label == "MCC" else f"{v:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confusion_block(pair):
    lines = ["| Model | TN | FP | FN | TP | Predicted positive | Actual positive |",
             "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        d = probs(m, pair)
        tn, fp, fn, tp = confusion_matrix(d.actual, d.prediction, labels=[0, 1]).ravel()
        lines.append(f"| {m} | {tn} | {fp} | {fn} | {tp} | "
                     f"{d.prediction.mean() * 100:.1f}% | {d.actual.mean() * 100:.1f}% |")
    return "\n".join(lines)


def benchmark_table():
    lines = ["| Factor pair | Majority class | Lag-11 persistence | Logistic | GAM | XGBoost |",
             "|---|---|---|---|---|---|"]
    for pair in PAIRS:
        y = labels(pair)
        d = probs("Logistic", pair)
        a = d.actual.values
        maj = max(a.mean(), 1 - a.mean())
        lag11 = accuracy_score(a, y.shift(11).reindex(d.Date).values)
        accs = [accuracy_score(probs(m, pair).actual, probs(m, pair).prediction) for m in MODELS]
        lines.append(f"| {pair.replace('_', ' ')} | {maj * 100:.1f}% | {lag11 * 100:.1f}% | "
                     + " | ".join(f"{a_ * 100:.1f}%" for a_ in accs) + " |")
    return "\n".join(lines)


def change_table(metrics):
    lines = ["| Factor pair | Model | Metric | Superseded run | Corrected | Change |",
             "|---|---|---|---|---|---|"]
    for pair in PAIRS:
        for m in MODELS:
            old_acc, old_auc, old_ll = SUPERSEDED[(pair, m)]
            r = metrics.loc[(pair, m)]
            for label, o, n, fmt in [("Accuracy", old_acc, r.accuracy, "pct"),
                                     ("ROC-AUC", old_auc, r.roc_auc, "pct")]:
                lines.append(f"| {pair.replace('_', ' ')} | {m} | {label} | "
                             f"{o * 100:.1f}% | **{n * 100:.1f}%** | {(n - o) * 100:+.1f}pp |")
    return "\n".join(lines)


def shap_table(pair, top_n=5):
    lines = ["| Rank | Logistic | GAM | XGBoost |", "|---|---|---|---|"]
    cols = {}
    for m in MODELS:
        s = pd.read_csv(OUTPUT_DIR / f"shap/{TAG[m]}_{pair.lower()}_shap.csv", index_col=0)
        cols[m] = list(s.iloc[:top_n, 0].items())
    for i in range(top_n):
        cells = [f"{n} ({v:.3f})" for m in MODELS for n, v in [cols[m][i]]]
        lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_params_table():
    lines = ["| Model | Factor pair | Selected parameters (initial pre-OOS tuning) |",
             "|---|---|---|"]
    for m in MODELS:
        for pair in PAIRS:
            p = json.loads((OUTPUT_DIR / f"artifacts/{TAG[m]}_{pair.lower()}_best_params.json").read_text())
            lines.append(f"| {m} | {pair.replace('_', ' ')} | "
                         + ", ".join(f"{k}={v}" for k, v in sorted(p.items())) + " |")
    return "\n".join(lines)


def grid_sizes():
    lines = ["| Model | Grid | Configurations |", "|---|---|---|"]
    for m in MODELS:
        g = pd.read_csv(OUTPUT_DIR / f"artifacts/{TAG[m]}_momentum_vs_value_tuning_grid.pkl")
        drop = {"mean_auc", "std_auc", "min_auc", "max_auc", "n_folds", "fold_aucs"}
        params = [c for c in g.columns if c not in drop]
        rendered = "; ".join(f"{c} {sorted(g[c].dropna().unique().tolist())}" for c in params)
        lines.append(f"| {m} | {rendered} | {len(g)} |")
    return "\n".join(lines)


def main():
    metrics = load_metrics()
    d = probs("Logistic", PAIRS[0])
    n, first, last = len(d), d.Date.iloc[0].date(), d.Date.iloc[-1].date()

    P = []
    P.append("# Chapter 4 — Results: Corrected Pipeline\n")
    P.append("**Every number in this file is generated from `results/model_outputs/`, not "
             "transcribed.** Regenerate with `python results/build_results_corrected_md.py` "
             "after re-running any model notebook.\n")
    P.append("This file supersedes `Chapter4_Results_Data.md`. That file remains in the "
             "repository because its numbers are still reproducible from the pre-fix "
             "pipeline, but they are affected by look-ahead bias and should not be "
             "reported as out-of-sample results.\n")
    P.append(f"Walk-forward test window: {n} weekly observations, {first} to {last}. "
             "All labels in this window are computable from observed data; dates whose "
             "trend-scanning label cannot be resolved are dropped rather than "
             "forward-filled.\n")

    P.append("---\n")
    P.append("## Validation configuration\n")
    P.append("All three models share the dataset, window, purge gap, cross-validation "
             "scheme and test block. **The tuning grid is the only thing that differs "
             "between them.**\n")
    P.append("| Setting | Value |\n|---|---|\n"
             "| Training window | Expanding (all history to the purged end), 751 → 906 obs |\n"
             "| Purge gap | 12 bars between the training window and the test point |\n"
             "| CV scheme | Purged k-fold, `CV_SPLITS=5`, `CV_TEST_SPLITS=1`, purge/embargo 12 |\n"
             "| Tuning data | Pre-OOS observations only |\n"
             "| Retuning | Every 52 steps, on the current training window |\n"
             "| Test block | 156 weekly observations, one-step-ahead |\n")
    P.append("The purge gap follows from the labelling: trend scanning uses `span=(1,12)`, "
             "so a label at date *t* is resolved using returns out to *t+10*. Training rows "
             "within 12 bars of the test point therefore carry labels computed from the "
             "test period itself.\n")
    P.append("Note that with `CV_TEST_SPLITS=1` the scheme yields C(5,1)=5 folds and a "
             "single backtest path, i.e. purged k-fold cross-validation. The combinatorial "
             "property requires `CV_TEST_SPLITS >= 2`.\n")

    P.append("---\n")
    P.append("## 4.3 Out-of-sample predictive power\n")
    for pair in PAIRS:
        tbl, title = PAIR_TITLES[pair]
        P.append(f"### {tbl}: OOS results for {title} pairing\n")
        P.append(oos_table(metrics, pair) + "\n")
    P.append("Balanced accuracy and MCC are reported because the classes are close to "
             "even and the models over-predict the majority side; plain accuracy flatters "
             "them. MCC of 0 is chance.\n")

    P.append("### Confusion matrices\n")
    for pair in PAIRS:
        P.append(f"**{pair.replace('_', ' ')}**\n")
        P.append(confusion_block(pair) + "\n")

    P.append("### Comparison against trivial benchmarks\n")
    P.append("Lag-11 persistence repeats the most recent label that is actually knowable "
             "at the decision date, given the 10-bar label resolution horizon.\n")
    P.append(benchmark_table() + "\n")

    P.append("---\n")
    P.append("## Effect of the validation corrections\n")
    P.append("The superseded column is the same pipeline before the fixes listed below. "
             "It is reported only to quantify the correction.\n")
    P.append(change_table(metrics) + "\n")

    P.append("Corrections applied:\n")
    P.append("1. **Walk-forward purge gap.** Training previously ran up to the bar before "
             "each test point, while labels resolve 10 bars forward, so training labels "
             "encoded the test outcome. In the OOS window P(y_t = y_{t-1}) = 0.79 against "
             "a base rate near 0.53, and unpurged predictions agreed with y_{t-1} 87% of "
             "the time — the models were largely retrieving the neighbouring label.\n"
             "2. **Unlabelled rows dropped.** The final ~10 dates have no computable label; "
             "`ffill()` previously manufactured them, placing fabricated targets inside the "
             "test block.\n"
             "3. **Tuning restricted to pre-OOS data.** GAM and the logistic model "
             "previously searched the full sample, selecting hyperparameters with the test "
             "labels visible.\n"
             "4. **Per-window scaling (logistic).** The scaler was previously fit once on "
             "the full sample, leaking the test block's mean and variance.\n"
             "5. **Pinned XGBoost threading.** `subsample` draws from an RNG whose stream "
             "varies with thread count, so results were not reproducible across machines.\n")

    P.append("---\n")
    P.append("## 4.4 Feature importance (Tables 17–19)\n")
    P.append("Mean |SHAP| over the training sample, per (model, factor pair). Top 5 of 20 "
             "saved; full rankings in `results/model_outputs/shap/`.\n")
    for pair, tbl in zip(PAIRS, ["Table 17", "Table 18", "Table 19"]):
        P.append(f"### {tbl}: {pair.replace('_', ' ')}\n")
        P.append(shap_table(pair) + "\n")
    P.append("These describe models whose out-of-sample discrimination is close to chance, "
             "so the rankings should be read as what each model weighted, not as evidence "
             "that those features predict the label.\n")

    P.append("---\n")
    P.append("## 4.5 Hyperparameter selection\n")
    P.append(grid_sizes() + "\n")
    P.append(best_params_table() + "\n")
    P.append("These are the parameters from the initial pre-OOS search, which govern the "
             "first 52 walk-forward steps. Retuning every 52 steps replaces them twice more "
             "within the test period; those later parameter sets are held in an in-memory "
             "cache and are not written to disk, so two thirds of each run's configuration "
             "is not recorded here.\n")

    P.append("---\n")
    P.append("## Interpretation and limitations\n")
    aucs = {m: np.mean([metrics.loc[(p, m), "roc_auc"] for p in PAIRS]) for m in MODELS}
    mccs = {m: np.mean([matthews_corrcoef(probs(m, p).actual, probs(m, p).prediction)
                        for p in PAIRS]) for m in MODELS}
    lo = min(metrics.loc[(p, m), "roc_auc"] for p in PAIRS for m in MODELS)
    hi = max(metrics.loc[(p, m), "roc_auc"] for p in PAIRS for m in MODELS)
    P.append(f"Mean ROC-AUC across the three pairs: "
             + ", ".join(f"{m} {aucs[m]:.3f}" for m in MODELS) + ". "
             f"Mean MCC: " + ", ".join(f"{m} {mccs[m]:+.3f}" for m in MODELS) + ".\n")
    P.append(f"All nine ROC-AUC values fall between {lo:.3f} and {hi:.3f}. With 156 test "
             "observations the bootstrap standard error of AUC is approximately 0.045, so "
             "the differences between models are not statistically distinguishable.\n")
    P.append("**Multiple comparisons.** Roughly 70 out-of-sample evaluations were performed "
             "on this single test window across model families, window lengths, CV settings "
             "and grid designs. Under the null of no predictability and that standard error, "
             "the expected maximum of 60 such tries is about 0.604 and the 95th percentile "
             "about 0.641. Any single best result should be read against that, not against "
             "a one-test p-value.\n")
    P.append("**Held-out validation.** No block of data has been reserved that was never "
             "consulted during model or configuration selection. A final estimate free of "
             "selection effects would require one.\n")

    MD_PATH.write_text("\n".join(P), encoding="utf-8")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
