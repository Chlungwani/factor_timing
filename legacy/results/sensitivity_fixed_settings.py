"""Sensitivity of the walk-forward results to the hyperparameters: every setting of each model's tuning grid,
held fixed for all 156 predictions.

The study selects hyperparameters by CPCV and retunes them annually (Section 3.8). This script asks how the
out-of-sample results would differ under each alternative setting of the same grid (Table 3), so that the
CPCV-selected results can be placed within the range the grid allows. Each run uses the model notebook's own
data preparation and walk_forward_validation, with retune_every=None, so the only difference from the main
results is that the setting is not reselected. A setting whose fit fails in any week is recorded as failed, as
in the CPCV search (Section 3.8.1).

Outputs (results/model_outputs/sensitivity/)
  <model>_<pair>_fixed.csv        one row per setting: the setting, accuracy, ROC-AUC, log-loss, or the failure
  <model>_<pair>_fixed_probs.csv  the 156 predicted probabilities of each setting (one column per setting)

Usage: python sensitivity_fixed_settings.py [workers] [model ...] [pair ...]   (default: every model and pair)
       python sensitivity_fixed_settings.py time      (times one setting per model)
"""
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import io
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent
LEGACY = RESULTS.parent
OUTPUTS = RESULTS / "model_outputs"
OUT = OUTPUTS / "sensitivity"
NOTEBOOKS = {"logistic": "logistic_classifier_roc.ipynb", "gam": "gam.ipynb", "xgboost": "xgboost_cpcv_wf.ipynb"}
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]
PARAMS = {"logistic": ["alpha", "lambda_reg"], "gam": ["lam", "n_splines"],
          "xgboost": ["learning_rate", "max_depth", "min_child_weight", "reg_lambda", "subsample"]}
FIXED = {"gam": {"max_iter": 1000, "fit_intercept": True}}       # constant in the GAM grid

_NS = {}


def notebook(model):
    """The notebook's definitions up to run_pipeline, executed once per worker process."""
    if model not in _NS:
        cells = ["".join(c["source"]) for c in json.load(open(LEGACY / "model_training" / NOTEBOOKS[model],
                                                               encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        stop = next(i for i, c in enumerate(cells) if "def run_pipeline" in c)
        ns = {"__name__": f"nb_{model}"}
        cwd = os.getcwd()
        try:
            os.chdir(LEGACY / "model_training")
            with contextlib.redirect_stdout(io.StringIO()):
                for src in cells[:stop]:
                    exec(compile(src.replace("os.chdir(", "(lambda *a: None)("), NOTEBOOKS[model], "exec"), ns)
        finally:
            os.chdir(cwd)
        _NS[model] = ns
    return _NS[model]


def load_data(pair):
    """The notebooks' load_data (labels joined to the lagged features, last unlabelled weeks dropped, the weeks
    before a series starts flagged)."""
    return notebook("logistic")["load_data"](pair)


def grid(model, pair):
    """The settings of the model's tuning grid, as saved by the notebook (the grid is the same for every pair)."""
    g = pd.read_csv(OUTPUTS / "artifacts" / f"{model}_{pair.lower()}_tuning_grid.pkl")      # saved as CSV
    settings = g[PARAMS[model]].drop_duplicates().sort_values(PARAMS[model]).to_dict("records")
    return [{k: (int(v) if k in ("max_depth", "n_splines", "min_child_weight") else float(v)) for k, v in s.items()}
            for s in settings]


def run(model, pair, setting):
    ns = notebook(model)
    data = load_data(pair)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = np.nan_to_num(X, nan=np.nan, posinf=1e10, neginf=-1e10)     # NaN: filled per training window
    params = {**setting, **FIXED.get(model, {})}
    n = ns["FINAL_TEST_POINTS"]
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model == "gam":
                actuals, preds, probs = ns["walk_forward_validation"](X, y, data.columns[:-1].tolist(), n_test=n,
                                                                      base_params=params, retune_every=None)
            else:
                actuals, preds, probs = ns["walk_forward_validation"](X, y, n_test=n, base_params=params,
                                                                      retune_every=None)
    except Exception as exc:                                      # e.g. a GAM fit that diverges in some week
        return model, pair, setting, None, f"{type(exc).__name__}: {str(exc)[:120]}", time.time() - t0
    return model, pair, setting, (np.asarray(actuals), np.asarray(probs, dtype=float)), None, time.time() - t0


def summarise(model, pair, results):
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    rows, probs = [], {}
    for k, (setting, out, error) in enumerate(results):
        row = {"setting": k, **setting}
        if out is None:
            row.update(failed=error)
        else:
            actual, p = out
            row.update(accuracy=accuracy_score(actual, (p >= 0.5).astype(int)), roc_auc=roc_auc_score(actual, p),
                       log_loss=log_loss(actual, np.clip(p, 1e-12, 1 - 1e-12)), failed="")
            probs[k] = p
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{model}_{pair.lower()}_fixed.csv", index=False)
    dates = load_data(pair).index[-len(next(iter(probs.values()))):] if probs else None
    if probs:
        pd.DataFrame(probs, index=dates).rename_axis("Date").to_csv(OUT / f"{model}_{pair.lower()}_fixed_probs.csv")


def main(workers, models, pairs=PAIRS):
    jobs = [(m, p, s) for m in models for p in pairs for s in grid(m, p)]
    order = {"gam": 0, "xgboost": 1, "logistic": 2}                       # longest first
    jobs.sort(key=lambda j: order[j[0]])
    done = {(m, p): {} for m in models for p in pairs}
    expected = {(m, p): grid(m, p) for m in models for p in pairs}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            model, pair, setting, out, error, secs = fut.result()
            done[(model, pair)][json.dumps(setting, sort_keys=True)] = (setting, out, error)
            if i % 10 == 0 or error:
                print(f"{time.strftime('%H:%M:%S')}  {i}/{len(jobs)}  last {model} {pair} {secs:.0f}s"
                      + (f"  FAILED {setting}: {error}" if error else ""), flush=True)
            if len(done[(model, pair)]) == len(expected[(model, pair)]):
                ordered = [done[(model, pair)][json.dumps(s, sort_keys=True)] for s in expected[(model, pair)]]
                summarise(model, pair, ordered)
                print(f"{time.strftime('%H:%M:%S')}  wrote {model} {pair}", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        for model in NOTEBOOKS:
            setting = grid(model, PAIRS[0])[0]
            *_, error, secs = run(model, PAIRS[0], setting)
            print(model, setting, f"{secs:.0f}s", error or "")
    else:
        workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
        args = sys.argv[2:]
        main(workers, [a for a in args if a in NOTEBOOKS] or list(NOTEBOOKS), [a for a in args if a in PAIRS] or PAIRS)
