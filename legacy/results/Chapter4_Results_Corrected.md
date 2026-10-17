# Chapter 4 — Results: Corrected Pipeline

**Every number in this file is generated from `results/model_outputs/`, not transcribed.** Regenerate with `python results/build_results_corrected_md.py` after re-running any model notebook.

This file supersedes `Chapter4_Results_Data.md`. That file remains in the repository because its numbers are still reproducible from the pre-fix pipeline, but they are affected by look-ahead bias and should not be reported as out-of-sample results.

Walk-forward test window: 156 weekly observations, 2022-01-31 to 2025-01-20. All labels in this window are computable from observed data; dates whose trend-scanning label cannot be resolved are dropped rather than forward-filled.

---

## Validation configuration

All three models share the dataset, window, purge gap, cross-validation scheme and test block. **The tuning grid is the only thing that differs between them.**

| Setting | Value |
|---|---|
| Training window | Expanding (all history to the purged end), 751 → 906 obs |
| Purge gap | 12 bars between the training window and the test point |
| CV scheme | Purged k-fold, `CV_SPLITS=5`, `CV_TEST_SPLITS=1`, purge/embargo 12 |
| Tuning data | Pre-OOS observations only |
| Retuning | Every 52 steps, on the current training window |
| Test block | 156 weekly observations, one-step-ahead |

The purge gap follows from the labelling: trend scanning uses `span=(1,12)`, so a label at date *t* is resolved using returns out to *t+10*. Training rows within 12 bars of the test point therefore carry labels computed from the test period itself.

Note that with `CV_TEST_SPLITS=1` the scheme yields C(5,1)=5 folds and a single backtest path, i.e. purged k-fold cross-validation. The combinatorial property requires `CV_TEST_SPLITS >= 2`.

---

## 4.3 Out-of-sample predictive power

### Table 6: OOS results for Momentum vs Value pairing

| Metric | Logistic | GAM | XGBoost |
|---|---|---|---|
| Accuracy | 44.9% | 56.4% | 60.9% |
| ROC-AUC | 48.3% | 54.9% | 56.6% |
| PR-AUC | 59.0% | 57.1% | 60.1% |
| Log-loss | 0.737 | 1.545 | 0.689 |
| F1 | 0.594 | 0.646 | 0.677 |
| Precision | 0.496 | 0.579 | 0.615 |
| Recall | 0.741 | 0.729 | 0.753 |
| Balanced accuracy | 0.420 | 0.548 | 0.595 |
| MCC | -0.205 | +0.103 | +0.200 |

### Table 9: OOS results for Momentum vs Quality pairing

| Metric | Logistic | GAM | XGBoost |
|---|---|---|---|
| Accuracy | 59.0% | 48.1% | 55.8% |
| ROC-AUC | 59.8% | 55.8% | 58.1% |
| PR-AUC | 62.3% | 55.2% | 60.4% |
| Log-loss | 0.677 | 0.926 | 0.709 |
| F1 | 0.683 | 0.362 | 0.571 |
| Precision | 0.580 | 0.523 | 0.590 |
| Recall | 0.831 | 0.277 | 0.554 |
| Balanced accuracy | 0.573 | 0.495 | 0.558 |
| MCC | +0.172 | -0.012 | +0.116 |

### Table 12: OOS results for Quality vs Value pairing

| Metric | Logistic | GAM | XGBoost |
|---|---|---|---|
| Accuracy | 47.4% | 53.8% | 45.5% |
| ROC-AUC | 53.1% | 50.5% | 56.2% |
| PR-AUC | 49.6% | 45.7% | 60.1% |
| Log-loss | 0.698 | 1.323 | 0.694 |
| F1 | 0.624 | 0.455 | 0.612 |
| Precision | 0.472 | 0.517 | 0.462 |
| Recall | 0.919 | 0.405 | 0.905 |
| Balanced accuracy | 0.496 | 0.532 | 0.477 |
| MCC | -0.015 | +0.066 | -0.089 |

Balanced accuracy and MCC are reported because the classes are close to even and the models over-predict the majority side; plain accuracy flatters them. MCC of 0 is chance.

### Confusion matrices

**Momentum vs Value**

| Model | TN | FP | FN | TP | Predicted positive | Actual positive |
|---|---|---|---|---|---|---|
| Logistic | 7 | 64 | 22 | 63 | 81.4% | 54.5% |
| GAM | 26 | 45 | 23 | 62 | 68.6% | 54.5% |
| XGBoost | 31 | 40 | 21 | 64 | 66.7% | 54.5% |

**Momentum vs Quality**

| Model | TN | FP | FN | TP | Predicted positive | Actual positive |
|---|---|---|---|---|---|---|
| Logistic | 23 | 50 | 14 | 69 | 76.3% | 53.2% |
| GAM | 52 | 21 | 60 | 23 | 28.2% | 53.2% |
| XGBoost | 41 | 32 | 37 | 46 | 50.0% | 53.2% |

**Value vs Quality**

| Model | TN | FP | FN | TP | Predicted positive | Actual positive |
|---|---|---|---|---|---|---|
| Logistic | 6 | 76 | 6 | 68 | 92.3% | 47.4% |
| GAM | 54 | 28 | 44 | 30 | 37.2% | 47.4% |
| XGBoost | 4 | 78 | 7 | 67 | 92.9% | 47.4% |

### Comparison against trivial benchmarks

Lag-11 persistence repeats the most recent label that is actually knowable at the decision date, given the 10-bar label resolution horizon.

| Factor pair | Majority class | Lag-11 persistence | Logistic | GAM | XGBoost |
|---|---|---|---|---|---|
| Momentum vs Value | 54.5% | 47.4% | 44.9% | 56.4% | 60.9% |
| Momentum vs Quality | 53.2% | 45.5% | 59.0% | 48.1% | 55.8% |
| Value vs Quality | 52.6% | 50.0% | 47.4% | 53.8% | 45.5% |

---

## Effect of the validation corrections

The superseded column is the same pipeline before the fixes listed below. It is reported only to quantify the correction.

| Factor pair | Model | Metric | Superseded run | Corrected | Change |
|---|---|---|---|---|---|
| Momentum vs Value | Logistic | Accuracy | 62.2% | **44.9%** | -17.3pp |
| Momentum vs Value | Logistic | ROC-AUC | 64.6% | **48.3%** | -16.4pp |
| Momentum vs Value | GAM | Accuracy | 64.1% | **56.4%** | -7.7pp |
| Momentum vs Value | GAM | ROC-AUC | 66.8% | **54.9%** | -11.8pp |
| Momentum vs Value | XGBoost | Accuracy | 71.8% | **60.9%** | -10.9pp |
| Momentum vs Value | XGBoost | ROC-AUC | 76.8% | **56.6%** | -20.2pp |
| Momentum vs Quality | Logistic | Accuracy | 61.5% | **59.0%** | -2.6pp |
| Momentum vs Quality | Logistic | ROC-AUC | 60.3% | **59.8%** | -0.6pp |
| Momentum vs Quality | GAM | Accuracy | 59.0% | **48.1%** | -10.9pp |
| Momentum vs Quality | GAM | ROC-AUC | 64.5% | **55.8%** | -8.7pp |
| Momentum vs Quality | XGBoost | Accuracy | 69.9% | **55.8%** | -14.1pp |
| Momentum vs Quality | XGBoost | ROC-AUC | 73.3% | **58.1%** | -15.2pp |
| Value vs Quality | Logistic | Accuracy | 66.0% | **47.4%** | -18.6pp |
| Value vs Quality | Logistic | ROC-AUC | 64.2% | **53.1%** | -11.1pp |
| Value vs Quality | GAM | Accuracy | 62.2% | **53.8%** | -8.3pp |
| Value vs Quality | GAM | ROC-AUC | 65.0% | **50.5%** | -14.5pp |
| Value vs Quality | XGBoost | Accuracy | 71.2% | **45.5%** | -25.6pp |
| Value vs Quality | XGBoost | ROC-AUC | 73.1% | **56.2%** | -16.9pp |

Corrections applied:

1. **Walk-forward purge gap.** Training previously ran up to the bar before each test point, while labels resolve 10 bars forward, so training labels encoded the test outcome. In the OOS window P(y_t = y_{t-1}) = 0.79 against a base rate near 0.53, and unpurged predictions agreed with y_{t-1} 87% of the time — the models were largely retrieving the neighbouring label.
2. **Unlabelled rows dropped.** The final ~10 dates have no computable label; `ffill()` previously manufactured them, placing fabricated targets inside the test block.
3. **Tuning restricted to pre-OOS data.** GAM and the logistic model previously searched the full sample, selecting hyperparameters with the test labels visible.
4. **Per-window scaling (logistic).** The scaler was previously fit once on the full sample, leaking the test block's mean and variance.
5. **Pinned XGBoost threading.** `subsample` draws from an RNG whose stream varies with thread count, so results were not reproducible across machines.

---

## 4.4 Feature importance (Tables 17–19)

Mean |SHAP| over the training sample, per (model, factor pair). Top 5 of 20 saved; full rankings in `results/model_outputs/shap/`.

### Table 17: Momentum vs Value

| Rank | Logistic | GAM | XGBoost |
|---|---|---|---|
| 1 | USDZARV1Y.Curncy_QS (0.034) | relvol_momentum_vs_quality_12 (1.706) | volratio_momentum_value_12 (0.095) |
| 2 | USDZARV1M.Curncy_QS (0.032) | relvol_value_vs_quality_12 (1.655) | relvol_momentum_vs_value_12 (0.078) |
| 3 | VolTermStructPC1 (0.012) | RandPPP_Factor_Inst_TS_ST (0.868) | mom1m_momentum_vs_quality (0.061) |
| 4 | EURUSDV6M.Curncy_QS (0.004) | ZAR18M.Curncy_QS (0.852) | US_RB_Curvature (0.047) |
| 5 | JALSH_Fundamental_PC1 (0.003) | SA_RB_Curvature (0.819) | SA_RB_Curvature (0.046) |

### Table 18: Momentum vs Quality

| Rank | Logistic | GAM | XGBoost |
|---|---|---|---|
| 1 | zrel_value_vs_quality_12 (0.033) | relvol_momentum_vs_quality_12 (1.890) | VolTermStructPC1 (0.205) |
| 2 | zrel_momentum_vs_quality_12 (0.029) | relvol_value_vs_quality_12 (1.853) | TOP40.Index.Price.to.Ebitda_Diff (0.153) |
| 3 | TOP40.Index.Price.to.Ebitda_Diff (0.024) | TOP40.Index.Price.to.Ebitda_Diff (0.600) | zrel_value_vs_quality_12 (0.140) |
| 4 | GLD_StructBreak (0.021) | volratio_value_quality_12 (0.538) | GLD_StructBreak (0.096) |
| 5 | mom3m_momentum_vs_quality (0.020) | MichgnConcIndx (0.454) | volratio_momentum_quality_12 (0.082) |

### Table 19: Value vs Quality

| Rank | Logistic | GAM | XGBoost |
|---|---|---|---|
| 1 | zrel_value_vs_quality_12 (0.053) | GlobalPropertyVol12 (1.508) | zrel_value_vs_quality_12 (0.020) |
| 2 | zrel_momentum_vs_quality_12 (0.008) | relvol_value_vs_quality_12 (1.322) | USDZARV1M.Curncy_QS (0.006) |
| 3 | USDZAR_EntDyn (0.004) | RandPPP_Factor_Inst_TS_ST (1.125) | zrel_momentum_vs_quality_12 (0.005) |
| 4 | RandPPP_Factor_Inst_TS_ST (0.004) | relvol_momentum_vs_quality_12 (1.058) | volratio_value_quality_12 (0.004) |
| 5 | SA_RB_Curvature (0.004) | USDZAR_EntDyn (0.912) | MichgnConcIndx (0.003) |

These describe models whose out-of-sample discrimination is close to chance, so the rankings should be read as what each model weighted, not as evidence that those features predict the label.

---

## 4.5 Hyperparameter selection

| Model | Grid | Configurations |
|---|---|---|
| Logistic | alpha [0.0, 0.5, 1.0]; base_score [0.5]; lambda_reg [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]; learning_rate [0.05]; num_iterations [5000]; regularization ['elasticnet'] | 18 |
| GAM | fit_intercept [True]; lam [1, 10, 100, 1000, 10000]; max_iter [1000]; n_splines [4, 5, 7, 10] | 20 |
| XGBoost | learning_rate [0.01, 0.03, 0.1]; max_depth [2, 3, 4]; min_child_weight [5, 10, 20]; reg_lambda [1, 10, 100, 1000]; subsample [0.6, 0.8] | 216 |

| Model | Factor pair | Selected parameters (initial pre-OOS tuning) |
|---|---|---|
| Logistic | Momentum vs Value | alpha=0.5, base_score=0.5, lambda_reg=100.0, learning_rate=0.05, num_iterations=5000, regularization=elasticnet |
| Logistic | Momentum vs Quality | alpha=0.0, base_score=0.5, lambda_reg=1000.0, learning_rate=0.05, num_iterations=5000, regularization=elasticnet |
| Logistic | Value vs Quality | alpha=0.5, base_score=0.5, lambda_reg=100.0, learning_rate=0.05, num_iterations=5000, regularization=elasticnet |
| GAM | Momentum vs Value | fit_intercept=True, lam=1000, max_iter=1000, n_splines=5 |
| GAM | Momentum vs Quality | fit_intercept=True, lam=100, max_iter=1000, n_splines=4 |
| GAM | Value vs Quality | fit_intercept=True, lam=1, max_iter=1000, n_splines=10 |
| XGBoost | Momentum vs Value | learning_rate=0.01, max_depth=3, min_child_weight=10, reg_lambda=1, subsample=0.6 |
| XGBoost | Momentum vs Quality | learning_rate=0.03, max_depth=3, min_child_weight=20, reg_lambda=1, subsample=0.6 |
| XGBoost | Value vs Quality | learning_rate=0.03, max_depth=2, min_child_weight=20, reg_lambda=1000, subsample=0.6 |

These are the parameters from the initial pre-OOS search, which govern the first 52 walk-forward steps. Retuning every 52 steps replaces them twice more within the test period; those later parameter sets are held in an in-memory cache and are not written to disk, so two thirds of each run's configuration is not recorded here.

---

## Interpretation and limitations

Mean ROC-AUC across the three pairs: Logistic 0.537, GAM 0.537, XGBoost 0.570. Mean MCC: Logistic -0.016, GAM +0.052, XGBoost +0.076.

All nine ROC-AUC values fall between 0.483 and 0.598. With 156 test observations the bootstrap standard error of AUC is approximately 0.045, so the differences between models are not statistically distinguishable.

**Multiple comparisons.** Roughly 70 out-of-sample evaluations were performed on this single test window across model families, window lengths, CV settings and grid designs. Under the null of no predictability and that standard error, the expected maximum of 60 such tries is about 0.604 and the 95th percentile about 0.641. Any single best result should be read against that, not against a one-test p-value.

**Held-out validation.** No block of data has been reserved that was never consulted during model or configuration selection. A final estimate free of selection effects would require one.
