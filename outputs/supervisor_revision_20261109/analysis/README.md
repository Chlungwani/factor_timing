# Corrected factor timing analysis

This folder contains the completed analysis used in the separate corrected thesis. The original thesis and legacy analysis files were not overwritten by this pipeline.

The feature screen, quantile binning and MI clustering use data through 1 November 2021. Screening removes 23 candidates with over 30% missing observations and ten constant inputs; 326 candidates give 29 representatives at Ward profile cut height two. Adding 18 relative-factor predictors gives 47 inputs. Forward filling uses only past observations; pre-inception rows are excluded. Four-week lagging leaves 784 complete rows and 774 resolved labels. The initial purged training window has 606 rows; the evaluation uses the original 156 dates from 31 January 2022 to 20 January 2025.

Each model is tuned initially and after forecasts 52 and 104. Every CPCV fold fits its own scaler for logistic regression and GAM; all searches and forecast fits use a 12-week purge. The grids contain 18 logistic, 20 GAM and 216 XGBoost settings. Ineligible, nonconvergent configurations are recorded in the grid JSON files. The GAM reference attributions are exact centred additive-term contributions; logistic attributions are exact and XGBoost uses Tree SHAP.

## Contents

- `inputs`: archived input CSVs used for this revision.
- `model_input.csv`, `lagged_features.csv`: the final labelled and lagged matrices.
- `representatives.csv`, `clusters.json`, `mi_distance_matrix.csv`: feature-selection outputs.
- `selection_sensitivity.csv`: pre-evaluation cut-height and bin sensitivity.
- `probs`: all 1,404 weekly model–pair probabilities and labels.
- `artifacts`: every tuning grid, selected schedule and individual metrics.
- `shap`: complete reference-model importance rankings.
- `metrics_with_intervals.csv`, `paired_inference.csv`, `bootstrap_block_sensitivity.csv`: verified performance and inference.
- `figures`: thesis figures regenerated from these results.
- `logs`: completed execution records.
- `provenance.json`: software versions, input hashes, code hashes and verification statements.
- `reproduce`: standalone analysis scripts that use the archived inputs in this folder.

## Reproduction

From this folder, use Python with the versions recorded in `provenance.json`:

```powershell
python reproduce/corrected_pipeline.py prepare
python reproduce/corrected_pipeline.py models
python reproduce/selection_diagnostics.py
python reproduce/finalise_analysis.py
```

These commands replace the analysis outputs in this folder. The completed run took approximately 77 minutes for its longest model job on this machine. The scripts run independent model–pair jobs in parallel and limit numerical libraries to one thread per job. The target labels are the archived trend-scanning labels, with maximum windows of 11 weekly observations including the start, equivalent to ten future weeks. Unresolved labels are excluded.

The inference uses 10,000 paired circular moving-block resamples with seed zero and 12-week blocks, with six- and 24-week sensitivity checks. ROC-AUC and accuracy intervals are percentile intervals. Paired tests are approximate null-centred bootstrap tests; Holm adjustments cover three pair comparisons within each metric. Bonferroni ROC-AUC intervals cover nine model–pair estimates.

Upstream PCA, quantile-scaling, rolling-statistic and data-vintage recipes remain provider-defined inputs; these scripts do not establish their historical point-in-time provenance.
