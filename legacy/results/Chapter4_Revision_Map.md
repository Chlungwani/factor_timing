# Chapter 4 correction map for the March 2026 thesis

This file is an editing guide. The original thesis document, `27717801_ResearchAssignment_vf - EG Mar 2026.docx`, has not been edited.

## Central change in interpretation

The existing chapter treats XGBoost as the dominant model and increasing complexity as a consistent source of out-of-sample improvement. The corrected results do not support either claim. Across the nine model–factor-pair combinations, ROC-AUC ranges from 0.483 to 0.598. The model ranking changes by factor pair, mean MCC is close to zero for every model class, and several high accuracy/F1 values are produced by strongly one-sided class predictions.

The revised thesis should state that the evidence for factor-timing predictability is weak and pair-dependent, and that no general complexity premium is established.

## Surgical change list

| Current location | Action | Reason / replacement |
|---|---|---|
| Chapter 4 overview | Replace | Remove claims that the framework “guarantees” comparable results or that observed differences necessarily reflect architecture. Use the corrected validation description and cautious scope in the replacement draft. |
| Section 4.2, Figures 5–7 and Tables 4–5 | Hold and regenerate | The supervisor requested active-space analysis. The repository has weekly factor returns but no aligned weekly benchmark return series, so the current raw-factor analysis cannot be converted reproducibly. Do not guess or use the older monthly workbook. |
| Section 4.3 introduction | Replace | Expand the metric set to include PR-AUC, balanced accuracy, MCC, confusion-matrix interpretation, and simple benchmarks. |
| Table 6 | Replace all values and prose | Corrected momentum–value results: XGBoost is best on point estimates but only modestly above simple benchmarks; logistic is below chance on balanced metrics. |
| Existing Tables 7–8 | Replace and reclassify | Use Table 7 for the corrected momentum–value confusion-matrix diagnostics, and convert the corrected ROC and probability plots to Figures 8–9 rather than labelling plots as tables. |
| Table 9 | Replace all values and prose | Logistic, not XGBoost, has the strongest momentum–quality metrics. This directly reverses the monotonic-complexity narrative. |
| Existing Tables 10–11 | Replace and reclassify | Use Table 10 for the corrected momentum–quality confusion-matrix diagnostics, and convert the corrected ROC and probability plots to Figures 10–11. |
| Table 12 | Replace all values and prose | No model performs convincingly for value–quality. XGBoost ranks moderately but predicts almost every observation as positive. |
| Existing Tables 13–15 | Replace, reclassify, and repair numbering | Use Table 13 for the corrected value–quality confusion-matrix diagnostics and use the corrected ROC and probability plots as Figures 12–13. The existing caption “Table 13: Table 14” is internally inconsistent. |
| Table 16 | Replace | The old Large/Moderate/Small summary is not supported. Use the quantitative majority-class and feasible lag-11 benchmark table. |
| Section 4.3 conclusion | Rewrite completely | Remove “XGBoost dominant,” “ranking consistently improves,” and “complexity consistently worsens calibration.” None survives the corrected metrics. |
| Tables 17–19 | Replace rankings and prose | Use corrected SHAP rankings. Present them as model-use descriptions, not proof of predictive economic drivers. |
| Local SHAP panels | Remove unless regenerated | Only aggregate mean absolute SHAP values are saved reproducibly. Observation-level local decompositions are not available from the corrected output files. |
| Section 4.5 introduction | Replace | Correct window is expanding, not a 773-week rolling window. Correct test dates are 2022-01-31 to 2025-01-20. |
| “CPCV” terminology | Correct throughout | With five groups and one held-out group, the implementation yields one back-test path and is effectively purged k-fold. It should not be described as a multi-path combinatorial design. |
| Tables 20–22 | Replace | The saved artefacts contain only the initial parameter selection. The later two retuning parameter sets are not persisted, so the current segment tables are not reproducible. |
| Chapter 4 conclusion | Replace | Use the cautious conclusion in the replacement draft. |

## Corrected headline results

| Factor pair | Best accuracy | Best ROC-AUC | Defensible interpretation |
|---|---:|---:|---|
| Momentum vs value | XGBoost, 60.9% | XGBoost, 56.6% | Modest point-estimate advantage; not strong discrimination. |
| Momentum vs quality | Logistic, 59.0% | Logistic, 59.8% | Best evidence in the corrected study, but still weak and class-asymmetric. |
| Value vs quality | GAM, 53.8% | XGBoost, 56.2% | No reliable classifier; XGBoost ranking does not translate to thresholded accuracy. |

Mean ROC-AUC is 0.537 for logistic regression, 0.537 for GAM, and 0.570 for XGBoost. Mean MCC is -0.016, 0.052, and 0.076, respectively.

## Cross-chapter consistency changes that are mandatory

Although the request centres on Chapter 4, leaving the following text unchanged would make the thesis internally contradictory.

| Location | Current problem | Required correction |
|---|---|---|
| Abstract, paragraph 1 | Calls the implementation CPCV. | Describe it as five-fold purged cross-validation unless `CV_TEST_SPLITS` is increased and the models are rerun. |
| Abstract, paragraph 2 | States XGBoost superiority, factor-pair gains from nonlinearity, and logistic calibration superiority. | Replace with the corrected conclusion: weak, pair-dependent predictability; no uniform complexity gain; XGBoost has only the highest mean AUC. |
| Chapter 1 research objectives / outline | Implies robust gains and validation. | Retain the research question but avoid language that presupposes successful prediction. |
| Chapter 3 validation method | Says CPCV produces multiple paths, uses a 773-week rolling window, and shows old grids. | State five purged folds with one held-out group, an expanding window, 12-bar walk-forward purge, pre-OOS tuning, retuning every 52 steps, and the corrected grids. |
| Chapter 3 performance metrics | Discusses only accuracy, ROC-AUC and log-loss despite the supervisor’s confusion-matrix comment. | Add PR-AUC, balanced accuracy, MCC and confusion matrices; explain one-sided prediction risk. |
| Chapter 3 feature importance | Promises both global and local explanations. | Limit the claim to saved global mean absolute SHAP unless local values are regenerated and persisted. |
| Chapter 5 summary of findings | Says nonlinear models consistently outperform, performance improves with flexibility, momentum–value is strongest, XGBoost is best calibrated only weakly, and validation proves leakage control. | Rewrite. Momentum–quality has the highest corrected AUC; model rankings vary; corrected temporal controls materially lower estimates; repeated use of the same test block remains a limitation. |
| Chapter 5 implications | Claims empirical support for ML factor allocation. | Reframe as insufficient evidence for deployment; require untouched validation and an after-cost portfolio test. |
| Chapter 5 limitations | Omits repeated test-set reuse and unavailable retuning history. | Add both limitations, plus the active-space data gap. |
| Concluding remarks | Claims meaningful improvements from flexibility. | Replace with cautious, non-confirmatory conclusion. |

## Reproducibility gaps to close before final submission

1. Obtain or export the weekly benchmark series used to define active factor returns, aligned to `factor_returns.csv`.
2. Regenerate Section 4.2 in active space and document the benchmark, frequency, date alignment, and missing-value treatment.
3. Persist every retuning event: prediction date, training start/end, purge end, selected parameters, cross-validation score, and random seed.
4. If the thesis is to retain the term CPCV, rerun with at least two held-out groups and document the resulting paths; otherwise use “purged k-fold”.
5. Reserve a final untouched block or clearly disclose that the 156-week period has been reused across approximately 70 evaluations and is therefore a development set rather than a pristine final test.
6. Recompute any local SHAP visualisations from saved observation-level values, or remove them.
7. Update the abstract and Chapter 5 at the same time as Chapter 4 so the reported contribution remains consistent.

## Files prepared for the revision

- `Chapter4_Corrected_Replacement_Draft.docx`: standalone, visually checked Word draft for review and copy-editing.
- `Chapter4_Corrected_Replacement_Draft.md`: replacement prose and tables for the corrected Chapter 4 results.
- `chapter4_corrected_figures/`: corrected ROC, walk-forward probability, and global SHAP figures.
- `build_chapter4_corrected_figures.py`: reproducible figure builder reading only the corrected model outputs.
- `format_chapter4_draft_docx.py`: deterministic formatting pass for the standalone Word draft.
- `Chapter4_Results_Corrected.md`: machine-generated source table for the corrected metrics and parameter selections.

