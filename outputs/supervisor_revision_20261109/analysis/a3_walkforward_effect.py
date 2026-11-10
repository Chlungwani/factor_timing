"""Supervisor item A3, part 2: effect of the boundary purge on the out-of-sample results.

For every model and pair whose purged initial search (a3_purged_initial_search.py) selected a different setting,
predictions 1-52 are refitted with that setting using the notebook's own walk_forward_validation on the data
truncated by 104 weeks (which reproduces the first 52 walk-forward steps exactly; checked against the saved
predictions with the original setting). Predictions 53-156 used later, already purged selections and are kept.
Writes a3_walkforward_effect.csv. Nothing in model_outputs is written.

Usage: python a3_walkforward_effect.py
"""
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

LEGACY = Path(__file__).resolve().parents[3] / "legacy"
sys.path.insert(0, str(LEGACY / "results"))
from rerun_pipeline import MODEL_NOTEBOOKS, code_cells  # noqa: E402

HERE = Path(__file__).resolve().parent
SEARCH = HERE / "a3_purged_initial_search"
PROBS = LEGACY / "results" / "model_outputs" / "probs"
N_TEST, FIRST = 156, 52


def first_block(model, pair, params):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cells = code_cells(LEGACY / "model_training" / MODEL_NOTEBOOKS[model])
        stop = next(i for i, c in enumerate(cells) if "def run_pipeline" in c)
        ns = {"__name__": f"nb_{model.lower()}"}
        for src in cells[:stop + 1]:
            exec(compile(src, MODEL_NOTEBOOKS[model], "exec"), ns)
        data = ns["load_data"](pair)
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        X = np.nan_to_num(X, nan=0.0 if model == "GAM" else np.nan, posinf=1e10, neginf=-1e10)
        cut = N_TEST - FIRST
        args = (X[:-cut], y[:-cut])
        if model == "GAM":
            a, _, p = ns["walk_forward_validation"](*args, data.columns[:-1].tolist(), FIRST, params,
                                                    lookback=None, retune_every=None)
        else:
            a, _, p = ns["walk_forward_validation"](*args, n_test=FIRST, base_params=params, retune_every=None)
    return a, p


def task(model, pair):
    res = json.load(open(SEARCH / f"{model.lower()}_{pair.lower()}_result.json"))
    saved = pd.read_csv(PROBS / f"{model.lower()}_{pair.lower()}_probs.csv")
    a_old, p_old = first_block(model, pair, res["old_best_params"])
    assert np.array_equal(a_old, saved.actual.values[:FIRST])
    assert np.allclose(p_old, saved.probability.values[:FIRST], atol=1e-9), np.abs(p_old - saved.probability.values[:FIRST]).max()
    _, p_new = first_block(model, pair, res["best_params"])
    y = saved.actual.values
    p_all = np.concatenate([p_new, saved.probability.values[FIRST:]])
    row = {"model": model, "pair": pair, "old_params": res["old_best_params"], "new_params": res["best_params"]}
    for tag, p in (("old", saved.probability.values), ("new", p_all)):
        row[f"{tag}_accuracy"] = accuracy_score(y, (p >= 0.5).astype(int))
        row[f"{tag}_roc_auc"] = roc_auc_score(y, p)
        row[f"{tag}_log_loss"] = log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))
        row[f"{tag}_auc_first52"] = roc_auc_score(y[:FIRST], p[:FIRST])
    pd.DataFrame({"Date": saved.Date, "actual": y, "probability_purged_initial": p_all}).to_csv(
        HERE / f"a3_{model.lower()}_{pair.lower()}_probs.csv", index=False)
    return row


if __name__ == "__main__":
    todo = []
    for f in sorted(SEARCH.glob("*_result.json")):
        r = json.load(open(f))
        if not r["unchanged"] and not (HERE / f"a3_{r['model'].lower()}_{r['pair'].lower()}_probs.csv").exists():
            todo.append((r["model"], r["pair"]))
    print("refitting predictions 1-52 for", todo, flush=True)
    rows = []
    out = HERE / "a3_walkforward_effect.csv"
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
    with ProcessPoolExecutor(max_workers=max(1, len(todo))) as pool:
        for row in pool.map(task, *zip(*todo)) if todo else []:
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows)[["model", "pair", "old_roc_auc", "new_roc_auc", "old_accuracy", "new_accuracy",
                              "old_log_loss", "new_log_loss"]].round(3).to_string(index=False))
