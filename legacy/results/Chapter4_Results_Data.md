# Chapter 4 — Results: Recomputed Table Data

**Every number in this file is generated from `results/model_outputs/`, not transcribed
from the thesis document.** Regenerate with `python results/build_results_md.py` after
re-running any model notebook.

Provenance: each of `model_training/logistic_classifier_roc.ipynb`, `gam.ipynb`, and
`xgboost_cpcv_wf.ipynb` now loops over all three factor pairs in a single run and writes
pair-labelled outputs, so no artifact is overwritten by a later pair. Walk-forward test
window: 156 weekly observations, 2022-04-11 to 2025-03-31.

---

## 4.3 Out-of-Sample predictive power

### Table 6: OOS results for Momentum vs Value pairing

| Metric | Logistic | GAM | XGBoost |
|---|---|---|---|
| Accuracy | 62.2% | 64.1% | 71.8% |
| ROC-AUC | 64.6% | 66.8% | 76.8% |
| Log-loss | 0.651 | 0.929 | 0.938 |
| F1 | 0.718 | 0.714 | 0.758 |
| Precision | 0.641 | 0.673 | 0.767 |
| Recall | 0.815 | 0.761 | 0.750 |

### Table 9: OOS results for Momentum vs Quality pairing

| Metric | Logistic | GAM | XGBoost |
|---|---|---|---|
| Accuracy | 61.5% | 59.0% | 69.9% |
| ROC-AUC | 60.3% | 64.5% | 73.3% |
| Log-loss | 0.689 | 0.818 | 0.958 |
| F1 | 0.630 | 0.568 | 0.728 |
| Precision | 0.680 | 0.689 | 0.733 |
| Recall | 0.586 | 0.483 | 0.724 |

### Table 12: OOS results for Quality vs Value pairing

| Metric | Logistic | GAM | XGBoost |
|---|---|---|---|
| Accuracy | 66.0% | 62.2% | 71.2% |
| ROC-AUC | 64.2% | 65.0% | 73.1% |
| Log-loss | 0.650 | 0.815 | 0.948 |
| F1 | 0.539 | 0.528 | 0.672 |
| Precision | 0.646 | 0.569 | 0.657 |
| Recall | 0.463 | 0.493 | 0.687 |

---

## 4.4 Feature Importance (Tables 17–19)

Mean |SHAP| over the training sample, now saved separately per (model, factor pair) —
previously all three models wrote to a single unlabelled file per run, so only the last
pair executed survived on disk. Top 5 of 20 saved features shown; full rankings in
`results/model_outputs/shap/`.

### Table 17: Momentum vs Value

| Rank | Logistic | GAM | XGBoost |
|---|---|---|---|
| 1 | USDZARV1Y.Curncy_QS (0.121) | relvol_value_vs_quality_12 (2.650) | relvol_momentum_vs_value_12 (0.196) |
| 2 | USDZARV1M.Curncy_QS (0.115) | relvol_momentum_vs_quality_12 (2.530) | mom1m_momentum_vs_value (0.177) |
| 3 | US_RB_Curvature (0.109) | GlobalPropertyVol12 (1.395) | SA_RB_Curvature (0.171) |
| 4 | ZAR18M.Curncy_QS (0.106) | MXWD_Fundamental_PC1 (1.297) | volratio_momentum_value_12 (0.171) |
| 5 | USDZAR_EntDyn (0.099) | USDZAR_EntDyn (1.291) | mom1m_momentum_vs_quality (0.161) |

### Table 18: Momentum vs Quality

| Rank | Logistic | GAM | XGBoost |
|---|---|---|---|
| 1 | zrel_value_vs_quality_12 (0.110) | relvol_momentum_vs_quality_12 (2.047) | zrel_value_vs_quality_12 (0.656) |
| 2 | TOP40.Index.Price.to.Ebitda_Diff (0.102) | relvol_value_vs_quality_12 (2.003) | TOP40.Index.Price.to.Ebitda_Diff (0.617) |
| 3 | GLD_StructBreak (0.096) | volratio_value_quality_12 (0.662) | MXWD.Index.PE.Ratio_QS (0.554) |
| 4 | zrel_momentum_vs_quality_12 (0.095) | TOP40.Index.Price.to.Ebitda_Diff (0.560) | VolTermStructPC1 (0.404) |
| 5 | GLD_RollKurt (0.083) | volratio_momentum_quality_12 (0.503) | GLD_StructBreak (0.384) |

### Table 19: Value vs Quality

| Rank | Logistic | GAM | XGBoost |
|---|---|---|---|
| 1 | zrel_value_vs_quality_12 (0.136) | relvol_momentum_vs_quality_12 (1.794) | zrel_value_vs_quality_12 (0.550) |
| 2 | zrel_momentum_vs_quality_12 (0.126) | relvol_value_vs_quality_12 (1.770) | MXWD.Index.PE.Ratio_QS (0.468) |
| 3 | TOP40.Index.Price.to.Ebitda_Diff (0.087) | GlobalPropertyVol12 (0.833) | GenComIndexPC1_QS (0.367) |
| 4 | RandPPP_Factor_Inst_TS_ST (0.079) | RandPPP_Factor_Inst_TS_ST (0.659) | zrel_momentum_vs_quality_12 (0.347) |
| 5 | USDZAR_EntDyn (0.065) | SA_NB_Slope (0.553) | GLD_StructBreak (0.335) |

---

## 4.5 Validation methodology impact (Tables 20–22)

**Partially recoverable.** The initial CPCV tuning result is now saved per (model, pair):

| Model | Factor pair | Best parameters (initial CPCV tuning) |
|---|---|---|
| Logistic | Momentum vs Value | alpha=0.5, base_score=0.6, lambda_reg=0.01, learning_rate=0.01, num_iterations=500, regularization=elasticnet |
| Logistic | Momentum vs Quality | alpha=1, base_score=0.6, lambda_reg=0.1, learning_rate=0.01, num_iterations=500, regularization=elasticnet |
| Logistic | Value vs Quality | alpha=1, base_score=0.5, lambda_reg=0.1, learning_rate=0.01, num_iterations=500, regularization=elasticnet |
| GAM | Momentum vs Value | fit_intercept=True, lam=1, max_iter=200, n_splines=10 |
| GAM | Momentum vs Quality | fit_intercept=True, lam=100, max_iter=200, n_splines=5 |
| GAM | Value vs Quality | fit_intercept=True, lam=1, max_iter=200, n_splines=5 |
| XGBoost | Momentum vs Value | base_score=0.6, colsample_bytree=0.8, learning_rate=0.1, max_depth=3, subsample=0.6 |
| XGBoost | Momentum vs Quality | base_score=0.5, colsample_bytree=1.0, learning_rate=0.5, max_depth=10, subsample=0.6 |
| XGBoost | Value vs Quality | base_score=0.5, colsample_bytree=1.0, learning_rate=0.5, max_depth=10, subsample=0.8 |

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

| Pair | Model | Metric | Thesis (as typed) | Recomputed |
|---|---|---|---|---|
| Momentum vs Value | Logistic | Accuracy | 59.6% | **62.2%** |
| Momentum vs Value | Logistic | ROC-AUC | 51.1% | **64.6%** |
| Momentum vs Value | Logistic | Log-loss | 0.71 | **0.651** |
| Momentum vs Value | GAM | ROC-AUC | 68.8% | **66.8%** |
| Momentum vs Quality | Logistic | Accuracy | 59.0% | **61.5%** |
| Momentum vs Quality | Logistic | ROC-AUC | 61.4% | **60.3%** |
| Momentum vs Quality | Logistic | Log-loss | 0.67 | **0.689** |

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
