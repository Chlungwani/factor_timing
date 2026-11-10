"""Every result that Chapters 4-5, the abstract and Appendix C quote, computed from one model-outputs folder.

Run on the outputs before and after the purged rerun; compare_facts() lists what changed, so each sentence that
quotes a changed value can be rewritten from the new file.

Usage: python results_facts.py <model_outputs_dir> <out.json>
       python results_facts.py --compare <old.json> <new.json>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve

MODELS = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]
RELVOL = re.compile(r"^(relvol|volratio)_")


def lead_ranges(curves, model, step=0.01):
    """False-positive-rate intervals over which `model` has the strictly highest true-positive rate."""
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    tpr = {m: np.interp(grid, f, t) for m, (f, t) in curves.items()}
    best = np.array([tpr[model][i] > max(tpr[m][i] for m in curves if m != model) + 1e-12 for i in range(len(grid))])
    out, start = [], None
    for i, b in enumerate(best):
        if b and start is None:
            start = grid[i]
        if (not b or i == len(grid) - 1) and start is not None:
            out.append([float(start), float(grid[i if b else i - 1])])
            start = None
    return out


def facts(outputs):
    outputs = Path(outputs)
    F = {}
    for pair in PAIRS:
        curves = {}
        for model, tag in MODELS.items():
            d = pd.read_csv(outputs / "probs" / f"{tag}_{pair.lower()}_probs.csv", parse_dates=["Date"])
            y, p = d.actual.values, d.probability.values
            pred = (p >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            k = f"{pair}|{model}"
            F[k + "|accuracy"] = accuracy_score(y, pred)
            F[k + "|roc_auc"] = roc_auc_score(y, p)
            F[k + "|log_loss"] = log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))
            F[k + "|confusion"] = [int(tn), int(fp), int(fn), int(tp)]
            F[k + "|pred_pos_share"] = pred.mean()
            F[k + "|confidently_wrong"] = int(((p > 0.9) & (y == 0)).sum() + ((p < 0.1) & (y == 1)).sum())
            for j in range(3):
                s = slice(52 * j, 52 * (j + 1))
                F[k + f"|period{j + 1}|prob_range"] = [float(p[s].min()), float(p[s].max())]
                F[k + f"|period{j + 1}|pred_pos_share"] = float(pred[s].mean())
                F[k + f"|period{j + 1}|accuracy"] = float((pred[s] == y[s]).mean())
                F[k + f"|period{j + 1}|roc_auc"] = (roc_auc_score(y[s], p[s]) if len(set(y[s])) == 2 else None)
            fpr, tpr, _ = roc_curve(y, p)
            curves[model] = (fpr, tpr)
            shap = pd.read_csv(outputs / "shap" / f"{tag}_{pair.lower()}_shap.csv", index_col=0).iloc[:, 0]
            top = shap.sort_values(ascending=False).head(10)
            F[k + "|shap_top10"] = list(top.index)
            F[k + "|shap_top10_sum"] = float(top.sum())
            F[k + "|shap_relvol_ranks"] = [i + 1 for i, f in enumerate(top.index) if RELVOL.match(f)]
            sched = json.loads((outputs / "artifacts" / f"{tag}_{pair.lower()}_walkforward_params.json").read_text())
            F[k + "|schedule"] = [{kk: v for kk, v in s.items() if kk not in ("fit_intercept", "max_iter")}
                                  for s in sched]
            # save_artifact writes DataFrames as CSV whatever the extension
            grid = pd.read_csv(outputs / "artifacts" / f"{tag}_{pair.lower()}_tuning_grid.pkl")
            grid = grid[grid["mean_auc"].notna()]
            top_row = grid.sort_values("mean_auc", ascending=False, kind="stable").iloc[0]
            F[k + "|cpcv_mean_auc"] = float(top_row["mean_auc"])
            F[k + "|cpcv_path_aucs"] = [float(a) for a in re.findall(r"\d\.\d+", top_row["path_aucs"])]
            F[k + "|cpcv_mean_path_auc"] = float(top_row["mean_path_auc"])
        y = d.actual.values
        F[f"{pair}|class_counts"] = [int((y == 0).sum()), int((y == 1).sum())]
        F[f"{pair}|majority"] = max(y.mean(), 1 - y.mean())
        for model in MODELS:
            F[f"{pair}|{model}|roc_lead_fpr"] = lead_ranges(curves, model)
    boot = outputs / "bootstrap"
    if (boot / "auc_ci.csv").exists():
        ci = pd.read_csv(boot / "auc_ci.csv")
        for _, r in ci.iterrows():
            F[f"{r.factor_pair}|{r.model}|auc_ci"] = [r.ci_low, r.ci_high]
        dd = pd.read_csv(boot / "auc_paired_differences.csv")
        for _, r in dd.iterrows():
            F[f"{r.factor_pair}|{r.comparison}|auc_diff_ci"] = [r.auc_diff, r.ci_low, r.ci_high]
    return F


def compare(old, new, tol=5e-4):
    old, new = json.load(open(old)), json.load(open(new))
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        same = (a == b) if not isinstance(a, float) else (b is not None and abs(a - b) < tol)
        if isinstance(a, list) and isinstance(b, list) and a and isinstance(a[0], float):
            same = len(a) == len(b) and all(abs(x - z) < tol for x, z in zip(a, b))
        if not same:
            print(f"{k}\n   old: {a}\n   new: {b}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        json.dump(facts(sys.argv[1]), open(sys.argv[2], "w"), indent=1, default=float)
        print("wrote", sys.argv[2])
