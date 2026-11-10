"""Supervisor item A4: accuracy intervals and the majority-class baseline, on the AUC bootstrap's resamples.

Reuses results/bootstrap_auc_ci.py (circular moving-block bootstrap, block 12, 10,000 resamples, seed 0),
so each accuracy interval comes from the same resampled weeks as the AUC interval beside it in Table 4.
Also checks that the AUC intervals reproduce model_outputs/bootstrap/auc_ci.csv.

Usage: python a4_accuracy_ci.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LEGACY = Path(__file__).resolve().parents[3] / "legacy"
sys.path.insert(0, str(LEGACY / "results"))
import bootstrap_auc_ci as b  # noqa: E402

OUT = Path(__file__).resolve().parent / "a4_accuracy_ci.csv"


def main():
    actuals, probs, n = b.load_predictions()
    preds = {}
    for pair in b.PAIRS:
        for model in b.MODELS:
            frame = pd.read_csv(b.PROBS_DIR / f"{b.TAGS[model]}_{pair.lower()}_probs.csv")
            preds[(model, pair)] = frame["prediction"].to_numpy()
    rng = np.random.default_rng(b.SEED)
    idx = b.block_indices(n, b.BLOCK_LENGTH, b.N_BOOT, rng)
    aucs = b.bootstrap_aucs(actuals, probs, n, b.BLOCK_LENGTH, b.N_BOOT, b.SEED)

    saved = pd.read_csv(b.OUT_DIR / "auc_ci.csv")
    rows = []
    for pair in b.PAIRS:
        y = actuals[pair]
        majority = max(y.mean(), 1 - y.mean())
        for model in b.MODELS:
            hit = (preds[(model, pair)] == y).astype(float)
            acc = hit[idx].mean(axis=1)
            auc_lo, auc_hi = np.nanpercentile(aucs[(model, pair)], [2.5, 97.5])
            ref = saved[(saved.factor_pair == pair) & (saved.model == model)].iloc[0]
            assert abs(auc_lo - ref.ci_low) < 1e-12 and abs(auc_hi - ref.ci_high) < 1e-12, (model, pair)
            lo, hi = np.percentile(acc, [2.5, 97.5])
            rows.append({"factor_pair": pair, "model": model,
                         "accuracy": hit.mean(), "acc_ci_low": lo, "acc_ci_high": hi,
                         "majority_class_accuracy": majority,
                         "roc_auc": roc_auc_score(y, probs[(model, pair)]),
                         "auc_ci_low": auc_lo, "auc_ci_high": auc_hi})
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
