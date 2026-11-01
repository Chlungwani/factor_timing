"""Compare the thesis's custom elastic-net logistic regression with scikit-learn."""
import json, re, sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.metrics import roc_auc_score

ROOT = "C:/Users/ccwis/OneDrive/Documents/1. MEng - Data Science/Project/factor_timing/legacy"
src = open("logistic_classifier_roc.ipynb.py", encoding="utf-8").read()
start = src.index("class LogisticRegression:")
end = src.index("CV_SPLITS = 5")
ns = {"np": np}
exec(src[start:end], ns)
Custom = ns["LogisticRegression"]

rows = []
for pair in ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]:
    lab = pd.read_csv(f"{ROOT}/data_input/labeled_relative_factor_returns.csv", parse_dates=["Date"]).set_index("Date")[[f"{pair}_trend"]]
    X = pd.read_csv(f"{ROOT}/data_input/lagged_data.csv", parse_dates=["Date"]).set_index("Date")
    d = pd.concat([X, lab], axis=1, join="inner").dropna(subset=[f"{pair}_trend"]).ffill().bfill()
    Xv, y = d.iloc[:, :-1].values[:-156], d.iloc[:, -1].values[:-156].astype(int)
    Xs = StandardScaler().fit_transform(Xv)
    p = json.load(open(f"{ROOT}/results/model_outputs/artifacts/logistic_{pair.lower()}_best_params.json"))
    for alpha, lam in [(p["alpha"], p["lambda_reg"]), (0.0, 1.0), (0.5, 10.0)]:
        prm = dict(p, alpha=alpha, lambda_reg=lam)
        c = Custom(**prm); c.fit(Xs, y)
        pc = c.predict_proba(Xs)
        m = len(y)
        # custom: mean logloss + (lam/m)(a|b|1 + (1-a)/2 |b|2^2)  ==  sklearn with C = 1/lam
        sk = SkLR(penalty="elasticnet", solver="saga", l1_ratio=alpha, C=1.0 / lam, tol=1e-10, max_iter=200000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); sk.fit(Xs, y)
        ps = sk.predict_proba(Xs)[:, 1]
        bc, bs = c.beta[1:], sk.coef_[0]
        rows.append(dict(pair=pair, alpha=alpha, lam=lam, selected=(alpha == p["alpha"] and lam == p["lambda_reg"]),
                         max_abs_prob_diff=np.max(np.abs(pc - ps)), coef_corr=np.corrcoef(bc, bs)[0, 1] if bs.std() > 0 and bc.std() > 0 else np.nan,
                         max_abs_coef_diff=np.max(np.abs(bc - bs)), auc_custom=roc_auc_score(y, pc), auc_sklearn=roc_auc_score(y, ps)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
out.to_csv("logistic_check.csv", index=False)
