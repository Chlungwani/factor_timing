"""Block-bootstrap confidence intervals for the final out-of-sample ROC-AUCs.

Reads the walk-forward predictions in ``results/model_outputs/probs`` and writes
CSV tables to ``results/model_outputs/bootstrap``.  No model is refitted.

Trend-scanning labels are persistent (consecutive test weeks usually share a
label), so weeks are resampled in contiguous blocks (circular moving-block
bootstrap) rather than individually.  The default block length of 12 weeks
matches the label look-ahead horizon.  All models and pairs share the same
156 test dates, so each resample uses the same weeks for every model and pair:
model differences are paired, and cross-pair means are resampled jointly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RESULTS_DIR = Path(__file__).resolve().parent
PROBS_DIR = RESULTS_DIR / "model_outputs" / "probs"
OUT_DIR = RESULTS_DIR / "model_outputs" / "bootstrap"

MODELS = ["Logistic", "GAM", "XGBoost"]
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]
TAGS = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}
COMPARISONS = [("XGBoost", "Logistic"), ("XGBoost", "GAM"), ("GAM", "Logistic")]

N_BOOT = 10_000
BLOCK_LENGTH = 12
SENSITIVITY_BLOCKS = [1, 6, 12, 24]
SEED = 0


def load_predictions():
    """Return actuals[pair] and probabilities[(model, pair)], aligned on date."""
    actuals, probs, dates = {}, {}, None
    for pair in PAIRS:
        for model in MODELS:
            path = PROBS_DIR / f"{TAGS[model]}_{pair.lower()}_probs.csv"
            frame = pd.read_csv(path, parse_dates=["Date"])
            if dates is None:
                dates = frame["Date"]
            if not frame["Date"].equals(dates):
                raise ValueError(f"Dates in {path.name} differ from the other files")
            if pair in actuals and not np.array_equal(actuals[pair], frame["actual"]):
                raise ValueError(f"Actuals in {path.name} differ across models")
            actuals[pair] = frame["actual"].to_numpy()
            probs[(model, pair)] = frame["probability"].to_numpy()
    return actuals, probs, len(dates)


def block_indices(n, block_length, n_boot, rng):
    """Circular moving-block bootstrap: n_boot rows of n resampled indices."""
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets) % n
    return idx.reshape(n_boot, -1)[:, :n]


def bootstrap_aucs(actuals, probs, n, block_length, n_boot, seed):
    """AUC for every (model, pair) on every resample; resamples shared by all."""
    rng = np.random.default_rng(seed)
    idx = block_indices(n, block_length, n_boot, rng)
    aucs = {key: np.full(n_boot, np.nan) for key in probs}
    for b in range(n_boot):
        rows = idx[b]
        for pair in PAIRS:
            y = actuals[pair][rows]
            if y.min() == y.max():  # single-class resample: AUC undefined
                continue
            for model in MODELS:
                aucs[(model, pair)][b] = roc_auc_score(y, probs[(model, pair)][rows])
    return aucs


def summarise(point, draws):
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi, len(draws)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actuals, probs, n = load_predictions()
    point = {
        (model, pair): roc_auc_score(actuals[pair], probs[(model, pair)])
        for model in MODELS
        for pair in PAIRS
    }

    aucs = bootstrap_aucs(actuals, probs, n, BLOCK_LENGTH, N_BOOT, SEED)

    # 1. Individual AUCs, with intervals widened for the nine model-pair results
    #    (Bonferroni: level 1 - 0.05/9, percentiles 0.28 and 99.72)
    tail = 100 * 0.05 / (len(MODELS) * len(PAIRS)) / 2
    rows = []
    for pair in PAIRS:
        for model in MODELS:
            draws = aucs[(model, pair)]
            est, lo, hi, valid = summarise(point[(model, pair)], draws)
            lo_b, hi_b = np.percentile(draws[~np.isnan(draws)], [tail, 100 - tail])
            rows.append({
                "factor_pair": pair, "model": model, "roc_auc": est,
                "ci_low": lo, "ci_high": hi,
                "ci_low_bonferroni": lo_b, "ci_high_bonferroni": hi_b,
                "p_auc_gt_0.5": np.nanmean(draws > 0.5),
                "valid_resamples": valid,
            })
    auc_table = pd.DataFrame(rows)

    # 2. Paired model differences within each pair
    rows = []
    for pair in PAIRS:
        for a, b in COMPARISONS:
            diff = aucs[(a, pair)] - aucs[(b, pair)]
            est, lo, hi, valid = summarise(point[(a, pair)] - point[(b, pair)], diff)
            rows.append({
                "factor_pair": pair, "comparison": f"{a} - {b}", "auc_diff": est,
                "ci_low": lo, "ci_high": hi,
                "p_diff_gt_0": np.nanmean(diff > 0),
                "valid_resamples": valid,
            })
    diff_table = pd.DataFrame(rows)

    # 3. Mean AUC across the three pairs, and paired differences in that mean
    mean_draws = {m: np.mean([aucs[(m, p)] for p in PAIRS], axis=0) for m in MODELS}
    mean_point = {m: np.mean([point[(m, p)] for p in PAIRS]) for m in MODELS}
    rows = []
    for model in MODELS:
        est, lo, hi, valid = summarise(mean_point[model], mean_draws[model])
        rows.append({
            "quantity": f"{model} mean AUC", "estimate": est, "ci_low": lo,
            "ci_high": hi, "p_gt_ref": np.nanmean(mean_draws[model] > 0.5),
            "reference": 0.5, "valid_resamples": valid,
        })
    for a, b in COMPARISONS:
        diff = mean_draws[a] - mean_draws[b]
        est, lo, hi, valid = summarise(mean_point[a] - mean_point[b], diff)
        rows.append({
            "quantity": f"{a} - {b} mean AUC", "estimate": est, "ci_low": lo,
            "ci_high": hi, "p_gt_ref": np.nanmean(diff > 0),
            "reference": 0.0, "valid_resamples": valid,
        })
    mean_table = pd.DataFrame(rows)

    # 4. Sensitivity of the individual-AUC intervals to block length
    rows = []
    for block in SENSITIVITY_BLOCKS:
        draws_by_key = (
            aucs if block == BLOCK_LENGTH
            else bootstrap_aucs(actuals, probs, n, block, N_BOOT, SEED)
        )
        for pair in PAIRS:
            for model in MODELS:
                draws = draws_by_key[(model, pair)]
                _, lo, hi, _ = summarise(point[(model, pair)], draws)
                rows.append({
                    "block_length": block, "factor_pair": pair, "model": model,
                    "ci_low": lo, "ci_high": hi,
                    "p_auc_gt_0.5": np.nanmean(draws > 0.5),
                })
    sensitivity_table = pd.DataFrame(rows)

    auc_table.to_csv(OUT_DIR / "auc_ci.csv", index=False)
    diff_table.to_csv(OUT_DIR / "auc_paired_differences.csv", index=False)
    mean_table.to_csv(OUT_DIR / "auc_mean_across_pairs.csv", index=False)
    sensitivity_table.to_csv(OUT_DIR / "auc_ci_block_sensitivity.csv", index=False)

    settings = (f"n={n} test weeks, {N_BOOT} resamples, block length "
                f"{BLOCK_LENGTH}, seed {SEED}")
    pd.set_option("display.width", 140)
    print(f"Settings: {settings}\n")
    print(auc_table.round(3).to_string(index=False), "\n")
    print(diff_table.round(3).to_string(index=False), "\n")
    print(mean_table.round(3).to_string(index=False), "\n")
    print(sensitivity_table.pivot_table(
        index=["factor_pair", "model"], columns="block_length",
        values=["ci_low", "ci_high"]).round(3).to_string())
    print(f"\nWrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
