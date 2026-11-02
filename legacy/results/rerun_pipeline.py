"""Rebuild the model inputs from factor_returns.csv and rerun the three models (annual retuning).

Stages
  data    the trend-scanning labels (label_creation/relative_factor_returns_ts.ipynb), the factor-relative
          features (features/factor_momentum.ipynb) and the lagged model input
          (model_training/data_preprocessing.ipynb), run in <data_dir> (default legacy/data_input).
  models  run_pipeline(pair) of each model notebook (logistic_classifier_roc, gam, xgboost_cpcv_wf) for
          every factor pair, nine runs in parallel. Outputs go to legacy/results/model_outputs exactly as the
          notebooks write them; the rows of metrics_summary.csv are written one at a time at the end.

The notebooks' code is executed unchanged, except that their os.chdir to the original machine's data
folder is pointed at <data_dir>. Plotting cells are not run.

Usage: python rerun_pipeline.py data [data_dir]
       python rerun_pipeline.py models <log_dir>
"""
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")            # nine runs share the cores; one thread each
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1]
MODEL_NOTEBOOKS = {"Logistic": "logistic_classifier_roc.ipynb", "GAM": "gam.ipynb", "XGBoost": "xgboost_cpcv_wf.ipynb"}
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]


def code_cells(notebook):
    return ["".join(c["source"]) for c in json.load(open(notebook, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def run_cells(notebook, cells, data_dir, name="__main__"):
    """Execute the given code cells in one namespace, with the notebook's os.chdir pointed at data_dir."""
    ns = {"__name__": name}
    moved = 0
    for src in cells:
        src, n = re.subn(r"os\.chdir\((['\"]).*?\1\)", lambda _: f"os.chdir(r'{data_dir}')", src)
        moved += n
        exec(compile(src, str(notebook), "exec"), ns)
    assert moved == 1, f"{notebook}: expected one os.chdir, found {moved}"
    return ns


def data_stage(data_dir):
    """Labels, factor-relative features and lagged model input, as the three notebooks build them."""
    cwd = os.getcwd()
    try:
        # labels: everything up to the cell that saves them (later cells print and plot)
        cells = code_cells(LEGACY / "label_creation" / "relative_factor_returns_ts.ipynb")
        last = next(i for i, c in enumerate(cells) if "labeled_relative_factor_returns.csv" in c and "to_csv" in c)
        run_cells("relative_factor_returns_ts.ipynb", cells[:last + 1], data_dir)
        cells = code_cells(LEGACY / "features" / "factor_momentum.ipynb")
        last = next(i for i, c in enumerate(cells) if "factor_relative_features.csv" in c and "to_csv" in c)
        run_cells("factor_momentum.ipynb", cells[:last + 1], data_dir)
        cells = code_cells(LEGACY / "model_training" / "data_preprocessing.ipynb")
        run_cells("data_preprocessing.ipynb", cells, data_dir)
    finally:
        os.chdir(cwd)


def model_task(model, pair, log_dir):
    """run_pipeline(pair) of one model notebook; returns its metrics row instead of writing it."""
    log = Path(log_dir) / f"{model.lower()}_{pair.lower()}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        cells = code_cells(LEGACY / "model_training" / MODEL_NOTEBOOKS[model])
        stop = next(i for i, c in enumerate(cells) if "def run_pipeline" in c)
        ns = {"__name__": f"nb_{model.lower()}"}
        for src in cells[:stop + 1]:
            exec(compile(src, MODEL_NOTEBOOKS[model], "exec"), ns)
        rows = []
        ns["append_metrics_row"] = rows.append
        ns["run_pipeline"](pair)
    return model, pair, rows, time.time() - t0


def models_stage(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    order = [(m, p) for m in ("XGBoost", "GAM", "Logistic") for p in PAIRS]       # longest first
    rows = []
    with ProcessPoolExecutor(max_workers=len(order)) as pool:
        futures = [pool.submit(model_task, m, p, log_dir) for m, p in order]
        for fut in as_completed(futures):
            model, pair, got, secs = fut.result()
            rows += got
            print(f"{time.strftime('%H:%M:%S')}  done {model:8s} {pair:20s} in {secs / 60:5.1f} min", flush=True)
    # metrics_summary.csv: the notebook's own update (replace the row of the same model and pair), one at a time
    cells = code_cells(LEGACY / "model_training" / MODEL_NOTEBOOKS["Logistic"])
    ns = {"__name__": "nb_metrics"}
    for src in cells:
        if "def append_metrics_row" in src or "OUTPUT_DIR = " in src or src.lstrip().startswith(("# Libraries", "import")):
            exec(compile(src, "metrics", "exec"), ns)
    for row in rows:
        ns["append_metrics_row"](row)
    print("metrics_summary.csv updated:", len(rows), "rows")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "data":
        data_stage(Path(sys.argv[2]) if len(sys.argv) > 2 else LEGACY / "data_input")
    elif stage == "models":
        models_stage(sys.argv[2])
    else:
        sys.exit(__doc__)
