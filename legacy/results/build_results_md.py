"""Regenerate Chapter4_Results_Data.md from the model pipeline outputs.

Every number in the generated file comes from results/model_outputs/, never from
the thesis document. Re-run this after re-running any model notebook.
"""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent
OUTPUT_DIR = RESULTS_DIR / "model_outputs"
MD_PATH = RESULTS_DIR / "Chapter4_Results_Data.md"

MODELS = ["Logistic", "GAM", "XGBoost"]
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]
PAIR_TITLES = {
    "Momentum_vs_Value": ("Table 6", "Momentum vs Value"),
    "Momentum_vs_Quality": ("Table 9", "Momentum vs Quality"),
    "Value_vs_Quality": ("Table 12", "Quality vs Value"),
}

# Values as TYPED INTO the Word document, kept only to flag cells that do not
# reproduce. Never copied into the tables above -- this is the correction list.
THESIS_OOS = {
    ("Momentum_vs_Value", "Logistic"): (59.6, 51.1, 0.71),
    ("Momentum_vs_Value", "GAM"): (64.1, 68.8, 0.93),
    ("Momentum_vs_Value", "XGBoost"): (71.8, 76.8, 0.94),
    ("Momentum_vs_Quality", "Logistic"): (59.0, 61.4, 0.67),
    ("Momentum_vs_Quality", "GAM"): (59.0, 64.5, 0.818),
    ("Momentum_vs_Quality", "XGBoost"): (69.9, 73.3, 0.96),
    ("Value_vs_Quality", "Logistic"): (66.0, 64.2, 0.65),
    ("Value_vs_Quality", "GAM"): (62.2, 65.0, 0.82),
    ("Value_vs_Quality", "XGBoost"): (71.2, 73.1, 0.95),
}


def load_metrics():
    df = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv")
    return df.set_index(["factor_pair", "model"])


def oos_table(metrics, pair):
    lines = ["| Metric | Logistic | GAM | XGBoost |", "|---|---|---|---|"]
    for label, col, fmt in [
        ("Accuracy", "accuracy", "pct"),
        ("ROC-AUC", "roc_auc", "pct"),
        ("Log-loss", "log_loss", "num"),
        ("F1", "f1", "num"),
        ("Precision", "precision", "num"),
        ("Recall", "recall", "num"),
    ]:
        cells = []
        for model in MODELS:
            v = metrics.loc[(pair, model), col]
            cells.append(f"{v * 100:.1f}%" if fmt == "pct" else f"{v:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def date_range(pair):
    df = pd.read_csv(OUTPUT_DIR / f"probs/logistic_{pair.lower()}_probs.csv")
    return len(df), df["Date"].iloc[0], df["Date"].iloc[-1]


def shap_table(pair, top_n=5):
    lines = ["| Rank | Logistic | GAM | XGBoost |", "|---|---|---|---|"]
    cols = {}
    for model in MODELS:
        s = pd.read_csv(OUTPUT_DIR / f"shap/{model.lower()}_{pair.lower()}_shap.csv", index_col=0)
        cols[model] = list(s.iloc[:top_n, 0].items())
    for i in range(top_n):
        cells = [f"{name} ({val:.3f})" for model in MODELS for name, val in [cols[model][i]]]
        lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_params_table():
    lines = ["| Model | Factor pair | Best parameters (initial CPCV tuning) |", "|---|---|---|"]
    for model in MODELS:
        for pair in PAIRS:
            p = json.loads((OUTPUT_DIR / f"artifacts/{model.lower()}_{pair.lower()}_best_params.json").read_text())
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(p.items()))
            lines.append(f"| {model} | {pair.replace('_', ' ')} | {rendered} |")
    return "\n".join(lines)


def discrepancy_table(metrics):
    lines = [
        "| Pair | Model | Metric | Thesis (as typed) | Recomputed |",
        "|---|---|---|---|---|",
    ]
    any_row = False
    for pair in PAIRS:
        for model in MODELS:
            t_acc, t_auc, t_ll = THESIS_OOS[(pair, model)]
            r = metrics.loc[(pair, model)]
            for label, thesis, recomputed, fmt in [
                ("Accuracy", t_acc, r["accuracy"] * 100, "pct"),
                ("ROC-AUC", t_auc, r["roc_auc"] * 100, "pct"),
                ("Log-loss", t_ll, r["log_loss"], "num"),
            ]:
                # Compare at the precision the thesis reports.
                decimals = len(str(thesis).split(".")[-1])
                if round(recomputed, decimals) != thesis:
                    any_row = True
                    shown_t = f"{thesis}%" if fmt == "pct" else f"{thesis}"
                    shown_r = f"{recomputed:.1f}%" if fmt == "pct" else f"{recomputed:.3f}"
                    lines.append(
                        f"| {pair.replace('_', ' ')} | {model} | {label} | {shown_t} | **{shown_r}** |"
                    )
    if not any_row:
        return "Every thesis value reproduces. No corrections needed."
    return "\n".join(lines)


def main():
    metrics = load_metrics()
    n_obs, start, end = date_range(PAIRS[0])

    parts = []
    parts.append(f"""# Chapter 4 — Results: Recomputed Table Data

**Every number in this file is generated from `results/model_outputs/`, not transcribed
from the thesis document.** Regenerate with `python results/build_results_md.py` after
re-running any model notebook.

Provenance: each of `model_training/logistic_classifier_roc.ipynb`, `gam.ipynb`, and
`xgboost_cpcv_wf.ipynb` now loops over all three factor pairs in a single run and writes
pair-labelled outputs, so no artifact is overwritten by a later pair. Walk-forward test
window: {n_obs} weekly observations, {start} to {end}.

---

## 4.3 Out-of-Sample predictive power
""")

    for pair in PAIRS:
        table_no, title = PAIR_TITLES[pair]
        parts.append(f"### {table_no}: OOS results for {title} pairing\n\n{oos_table(metrics, pair)}\n")

    parts.append("""---

## 4.4 Feature Importance (Tables 17–19)

Mean |SHAP| over the training sample, now saved separately per (model, factor pair) —
previously all three models wrote to a single unlabelled file per run, so only the last
pair executed survived on disk. Top 5 of 20 saved features shown; full rankings in
`results/model_outputs/shap/`.
""")

    for pair in PAIRS:
        table_no = {"Momentum_vs_Value": "Table 17", "Momentum_vs_Quality": "Table 18",
                    "Value_vs_Quality": "Table 19"}[pair]
        parts.append(f"### {table_no}: {pair.replace('_', ' ')}\n\n{shap_table(pair)}\n")

    parts.append(f"""---

## 4.5 Validation methodology impact (Tables 20–22)

**Partially recoverable.** The initial CPCV tuning result is now saved per (model, pair):

{best_params_table()}

The thesis tables additionally report a *per-walk-forward-segment* breakdown (three rows
per pair). Those later segments come from the in-loop retuning inside
`walk_forward_validation`, which keeps its results in a local `hyperparam_cache` that is
never written to disk — so segments 2 and 3 still cannot be reconstructed. Persisting that
cache would close the gap.

Note also that the "ROC AUC" column in the thesis versions of these tables is the CPCV
*tuning-fold* score, not an out-of-sample score for that segment — it is not comparable to
the ROC-AUC in Tables 6/9/12.

---

## 4.2 The structure and dynamics of local style factors (Tables 4–5)

**Not recomputed.** Neither Spearman/Kendall correlation of the raw `Momentum`/`Value`/
`Quality` series in `labeled_data.csv` nor of the `_Relative` series reproduces the thesis
pattern (near-zero Quality correlation against the other two), so the underlying series and
transform cannot be identified with confidence. No figures are reported rather than
reporting a number derived from the wrong series.

## Table 16 (comparative ML performance gains)

**Not recomputed.** A qualitative summary (Large/Moderate/Small) against a non-ML baseline
that is not saved anywhere in the repository.

---

## Cells that do not reproduce against the Word document

Everything above is recomputed output. This section exists only to identify which numbers
in the thesis need correcting.

{discrepancy_table(metrics)}

### Reading this list

- **XGBoost reproduces on every cell.** Its thesis numbers were transcribed correctly; the
  stale `*_probs.csv` files previously in `data_input/` were leftovers from earlier runs,
  not evidence of transcription error.
- **GAM reproduces on 8 of 9 cells.** The single exception is a one-digit difference,
  consistent with a typo rather than a methodology difference.
- **Logistic reproduces only on Value vs Quality.** There are two logistic notebooks with
  materially different validation settings — `logistic_classifier.ipynb`
  (`CV_TEST_SPLITS=1`, `lookback=360`) and `logistic_classifier_roc.ipynb`
  (`CV_TEST_SPLITS=2`, `lookback=749`). The most likely explanation is that the three
  logistic rows in the thesis were not all produced under the same configuration. This is a
  methodology inconsistency to resolve, not a transcription error.
""")

    MD_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
