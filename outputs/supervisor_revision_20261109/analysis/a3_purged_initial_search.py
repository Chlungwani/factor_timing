"""Supervisor item A3: rerun only the initial CPCV search with a 12-week purge at the evaluation boundary.

The notebooks tune the initial search on X[:-156] (758 rows), whose last ten labels draw on the first
evaluation weeks. The retuning searches already use the purged window. Here the initial search is run on
X[:-156 - 12] (746 rows) with each notebook's own tune_hyperparameters, and the selected setting is
compared with results/model_outputs/artifacts/<model>_<pair>_best_params.json. Nothing in model_outputs
is written. If every selection is unchanged, the walk-forward predictions are unchanged too.

Usage: python a3_purged_initial_search.py
"""
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

LEGACY = Path(__file__).resolve().parents[3] / "legacy"
sys.path.insert(0, str(LEGACY / "results"))
from rerun_pipeline import MODEL_NOTEBOOKS, PAIRS, code_cells  # noqa: E402

OUT = Path(__file__).resolve().parent / "a3_purged_initial_search"
ARTIFACTS = LEGACY / "results" / "model_outputs" / "artifacts"
N_TEST, PURGE = 156, 12


def task(model, pair):
    log = OUT / f"{model.lower()}_{pair.lower()}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        cells = code_cells(LEGACY / "model_training" / MODEL_NOTEBOOKS[model])
        stop = next(i for i, c in enumerate(cells) if "def run_pipeline" in c)
        ns = {"__name__": f"nb_{model.lower()}"}
        for src in cells[:stop + 1]:
            exec(compile(src, MODEL_NOTEBOOKS[model], "exec"), ns)
        data = ns["load_data"](pair)
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        nan = 0.0 if model == "GAM" else np.nan          # as each notebook's run_pipeline
        X = np.nan_to_num(X, nan=nan, posinf=1e10, neginf=-1e10)
        n = len(X) - N_TEST - PURGE
        Xt, yt = X[:n], y[:n]
        if model != "XGBoost":
            from sklearn.preprocessing import StandardScaler
            Xt = StandardScaler().fit_transform(Xt)
        args = (Xt, yt, data.columns[:-1].tolist()) if model == "GAM" else (Xt, yt)
        best, grid = ns["tune_hyperparameters"](*args, return_grid=True)
        first_date, last_date = data.index[0].date(), data.index[n - 1].date()
    slug = f"{model.lower()}_{pair.lower()}"
    grid.to_pickle(OUT / f"{slug}_tuning_grid.pkl")
    old = json.load(open(ARTIFACTS / f"{slug}_best_params.json"))
    row = grid.iloc[0]
    result = {"model": model, "pair": pair, "n_obs": int(n), "first": str(first_date), "last": str(last_date),
              "best_params": best, "old_best_params": old, "unchanged": best == old,
              "mean_auc": float(row["mean_auc"]), "mean_path_auc": float(row["mean_path_auc"]),
              "path_aucs": [float(a) for a in row["path_aucs"]], "minutes": (time.time() - t0) / 60}
    json.dump(result, open(OUT / f"{slug}_result.json", "w"), indent=2, default=str)
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    order = [(m, p) for m in ("GAM", "XGBoost", "Logistic") for p in PAIRS]
    results = []
    with ProcessPoolExecutor(max_workers=len(order)) as pool:
        futures = [pool.submit(task, m, p) for m, p in order]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"{time.strftime('%H:%M:%S')}  {r['model']:8s} {r['pair']:20s} n={r['n_obs']} "
                  f"unchanged={r['unchanged']}  {r['best_params']}  ({r['minutes']:.1f} min)", flush=True)
    json.dump(results, open(OUT / "summary.json", "w"), indent=2, default=str)
    print("changed:", [(r["model"], r["pair"]) for r in results if not r["unchanged"]])
