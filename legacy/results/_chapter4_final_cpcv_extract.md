# Chapter 4: Results

## 4.1 Overview of the evaluation framework

This chapter presents the final out-of-sample results for the prediction of relative leadership among the momentum, value and quality factors in the South African equity market. Three model classes are evaluated: regularised logistic regression, a generalised additive model (GAM), and extreme gradient boosting (XGBoost). These models provide a progression from a linear specification, through additive nonlinear effects, to a flexible tree-based model capable of representing nonlinearities and interactions.

Hyperparameters are selected using combinatorial purged cross-validation (CPCV) with five ordered groups and two held-out groups. This produces ten training-validation combinations and four back-test paths. A 12-week purge and embargo is imposed to address the overlapping information horizon introduced by the trend-scanning labels. Final performance is evaluated using an expanding walk-forward procedure. Each model is initially trained on 751 eligible observations, with the training sample expanding to 906 observations by the end of the evaluation period. Hyperparameters are reconsidered every 52 predictions.

The final evaluation block contains 156 weekly observations from 31 January 2022 to 20 January 2025. All models are evaluated on the same dates and use a fixed probability threshold of 0.5. The positive class indicates leadership by the first-named factor. For example, a positive momentum-versus-value observation indicates that the trend-scanning label favours momentum over value.

Performance is assessed using accuracy, ROC-AUC, precision-recall AUC (PR-AUC), log-loss, precision, recall, F1, balanced accuracy and the Matthews correlation coefficient (MCC). Accuracy is interpreted relative to the majority-class and lag-11 persistence benchmarks. Balanced accuracy and MCC are emphasised because several specifications produce strongly asymmetric class predictions. ROC-AUC measures ranking ability independently of the 0.5 threshold, while log-loss evaluates the quality of the predicted probabilities.

## 4.2 Evaluation sample and benchmark context

Table 4.1 reports the realised class distribution and the two simple reference rules. The three targets are reasonably balanced, although momentum leadership is slightly more common in the two momentum pairs. The majority-class benchmark therefore ranges from 52.6% to 54.5%.

| **Factor pair** | **Negative observations** | **Positive observations** | **Positive share** | **Majority-class accuracy** | **Lag-11 persistence accuracy** |
|:---|:--:|:--:|:--:|:--:|:--:|
| Momentum vs value | 71 | 85 | 54.5% | 54.5% | 45.5% |
| Momentum vs quality | 73 | 83 | 53.2% | 53.2% | 44.8% |
| Value vs quality | 82 | 74 | 47.4% | 52.6% | 49.0% |

**Table 4.1. Target distribution and reference benchmarks**

The lag-11 rule repeats the most recent trend label that is fully observable at the prediction date, given the forward information used to construct the trend-scanning target. It is evaluated over the 145 observations for which an eleven-week lag is available, whereas the majority benchmark and model results use all 156 observations. It is therefore presented as a descriptive reference rather than an exactly sample-matched test of model superiority.

## 4.3 Out-of-sample predictive performance

Table 4.2 reports the complete model comparison. No single model dominates every factor pair and performance dimension. The largest ROC-AUC in the study is 0.589, indicating that the final predictive relationships are modest rather than strongly discriminative.

**Table 4.2. Final out-of-sample model results**

*Panel A: Ranking and probability metrics*

| **Factor pair** | **Model** | **Accuracy** | **ROC-AUC** | **PR-AUC** | **Log-loss** |
|:---|:--:|:--:|:--:|:--:|:--:|
| Momentum vs value | Logistic | 48.1% | 55.1% | 61.5% | 0.688 |
| Momentum vs value | GAM | 59.0% | 58.2% | 58.7% | 1.609 |
| Momentum vs value | XGBoost | **61.5%** | **58.9%** | 61.1% | 0.759 |
| Momentum vs quality | Logistic | 51.3% | 55.0% | 57.9% | **0.687** |
| Momentum vs quality | GAM | 48.1% | 55.7% | 55.8% | 0.932 |
| Momentum vs quality | XGBoost | **55.1%** | **57.8%** | **61.6%** | 0.719 |
| Value vs quality | Logistic | **56.4%** | **58.5%** | 57.6% | **0.684** |
| Value vs quality | GAM | 51.9% | 52.1% | 48.0% | 1.097 |
| Value vs quality | XGBoost | 48.7% | 56.5% | **59.8%** | 0.691 |

*Panel B: Threshold-dependent classification metrics*

| **Factor pair** | **Model** | **F1** | **Precision** | **Recall** | **Balanced accuracy** | **MCC** |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| Momentum vs value | Logistic | 0.646 | 0.514 | 0.871 | 0.442 | -0.216 |
| Momentum vs value | GAM | 0.652 | 0.606 | 0.706 | 0.578 | 0.162 |
| Momentum vs value | XGBoost | **0.663** | **0.634** | 0.694 | **0.608** | **0.218** |
| Momentum vs quality | Logistic | **0.624** | 0.529 | **0.759** | 0.496 | -0.009 |
| Momentum vs quality | GAM | 0.362 | 0.523 | 0.277 | 0.495 | -0.012 |
| Momentum vs quality | XGBoost | 0.551 | **0.589** | 0.518 | **0.554** | **0.107** |
| Value vs quality | Logistic | 0.564 | **0.537** | 0.595 | **0.566** | **0.131** |
| Value vs quality | GAM | 0.427 | 0.491 | 0.378 | 0.512 | 0.026 |
| Value vs quality | XGBoost | **0.636** | 0.479 | **0.946** | 0.510 | 0.039 |

<figure>
<img src="media/image1.png" style="width:5.83333in;height:2.03585in" alt="Final out-of-sample ROC curves" />
<figcaption><p><strong>Figure 4.1. Final out-of-sample ROC curves for the three factor pairs.</strong></p></figcaption>
</figure>

### 4.3.1 Momentum versus value

Momentum versus value provides the strongest thresholded classification result in the study. XGBoost achieves accuracy of 61.5%, which is 7.1 percentage points above the 54.5% majority-class benchmark. It also records the highest ROC-AUC (0.589), balanced accuracy (0.608) and MCC (0.218) for this pair. The GAM ranks second, with accuracy of 59.0%, ROC-AUC of 0.582, balanced accuracy of 0.578 and MCC of 0.162.

The point estimates therefore indicate that nonlinear structure contributes to this factor pair. The size of the effect remains moderate: the best AUC is below 0.60 and the best MCC is 0.218. The results support a relative advantage for XGBoost within this pair, but do not indicate strong separation between momentum- and value-leadership regimes.

Logistic regression performs poorly at the fixed threshold despite a ROC-AUC of 0.551 and PR-AUC of 0.615. Its accuracy of 48.1% is below the majority benchmark, balanced accuracy is 0.442 and MCC is negative. This divergence between ranking and thresholded measures is explained by its strong tendency to predict momentum leadership.

<figure>
<img src="media/image2.png" style="width:5.83333in;height:4.17576in" alt="Final walk-forward probabilities for momentum versus value" />
<figcaption><p><strong>Figure 4.2. Final walk-forward probabilities and realised classes for momentum versus value.</strong></p></figcaption>
</figure>

### 4.3.2 Momentum versus quality

Predictive performance is weaker for momentum versus quality. XGBoost again produces the strongest overall classification results, with accuracy of 55.1%, ROC-AUC of 0.578, PR-AUC of 0.616, balanced accuracy of 0.554 and MCC of 0.107. Its accuracy exceeds the majority-class benchmark by 1.9 percentage points. The magnitude of this gain is small, and the balanced metrics remain close to their no-skill values.

Logistic regression records the lowest log-loss, at 0.687, and high positive-class recall of 0.759. However, its accuracy of 51.3% is below the majority benchmark, balanced accuracy is 0.496 and MCC is slightly negative. The comparatively high F1 score of 0.624 is therefore driven partly by frequent positive predictions rather than by balanced discrimination between the two regimes.

The GAM displays the opposite error pattern. Its ROC-AUC of 0.557 suggests limited ranking information, but it predicts the positive class infrequently and achieves recall of only 0.277. Consequently, its accuracy, balanced accuracy and MCC do not improve on chance-like classification.

<figure>
<img src="media/image3.png" style="width:5.83333in;height:4.17576in" alt="Final walk-forward probabilities for momentum versus quality" />
<figcaption><p><strong>Figure 4.3. Final walk-forward probabilities and realised classes for momentum versus quality.</strong></p></figcaption>
</figure>

### 4.3.3 Value versus quality

Value versus quality reverses the ordering observed in the momentum pairs. Logistic regression is the strongest model, with accuracy of 56.4%, ROC-AUC of 0.585, log-loss of 0.684, balanced accuracy of 0.566 and MCC of 0.131. Its accuracy exceeds the 52.6% majority benchmark by 3.8 percentage points. It also generates the most balanced predictions for this pair.

XGBoost produces a ROC-AUC of 0.565 and the highest PR-AUC and F1, but these measures do not translate into useful thresholded classification. It predicts value leadership for 93.6% of the evaluation observations, despite a realised positive rate of 47.4%. This produces recall of 0.946 but specificity of only 0.073. Its accuracy is therefore 48.7%, below both the majority benchmark and logistic regression. The result demonstrates why recall, PR-AUC or F1 should not be interpreted without the confusion matrix and balanced metrics.

The GAM is close to chance across most measures. Its accuracy is 51.9%, ROC-AUC is 0.521, balanced accuracy is 0.512 and MCC is 0.026. The evidence for predictable value-versus-quality leadership is therefore concentrated in the linear specification rather than in the more flexible models.

<figure>
<img src="media/image4.png" style="width:5.83333in;height:4.17576in" alt="Final walk-forward probabilities for value versus quality" />
<figcaption><p><strong>Figure 4.4. Final walk-forward probabilities and realised classes for value versus quality.</strong></p></figcaption>
</figure>

### 4.3.4 Cross-pair model comparison

Table 4.3 averages each metric across the three factor pairs. XGBoost records the highest mean accuracy, ROC-AUC, balanced accuracy and MCC. The average advantage is nevertheless modest, and it is not observed uniformly across the individual factor pairs. Logistic regression performs best for value versus quality and produces the lowest average log-loss, while the GAM does not lead any pair on the principal performance measures.

| **Model** | **Mean accuracy** | **Mean ROC-AUC** | **Mean balanced accuracy** | **Mean MCC** | **Mean log-loss** |
|:---|:--:|:--:|:--:|:--:|:--:|
| Logistic | 51.9% | 56.2% | 50.1% | -0.031 | **0.687** |
| GAM | 53.0% | 55.3% | 52.8% | 0.059 | 1.213 |
| XGBoost | **55.1%** | **57.8%** | **55.7%** | **0.122** | 0.723 |

**Table 4.3. Mean performance across the three factor pairs**

These results provide qualified evidence in favour of XGBoost: it has the strongest average discrimination and class-balanced performance, and it leads both momentum pairs. They do not establish a universal complexity advantage. Logistic regression leads value versus quality and produces substantially better probability quality than the GAM and, on average, XGBoost. The model hierarchy is therefore dependent on the target pair and the aspect of performance being evaluated.

## 4.4 Classification behaviour and probability quality

Table 4.4 reports the confusion-matrix counts and predicted-positive shares. These diagnostics explain several apparent inconsistencies between F1, ROC-AUC and accuracy.

| **Factor pair** | **Model** | **TN** | **FP** | **FN** | **TP** | **Predicted positive** | **Actual positive** |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Momentum vs value | Logistic | 1 | 70 | 11 | 74 | 92.3% | 54.5% |
| Momentum vs value | GAM | 32 | 39 | 25 | 60 | 63.5% | 54.5% |
| Momentum vs value | XGBoost | 37 | 34 | 26 | 59 | 59.6% | 54.5% |
| Momentum vs quality | Logistic | 17 | 56 | 20 | 63 | 76.3% | 53.2% |
| Momentum vs quality | GAM | 52 | 21 | 60 | 23 | 28.2% | 53.2% |
| Momentum vs quality | XGBoost | 43 | 30 | 40 | 43 | 46.8% | 53.2% |
| Value vs quality | Logistic | 44 | 38 | 30 | 44 | 52.6% | 47.4% |
| Value vs quality | GAM | 53 | 29 | 46 | 28 | 36.5% | 47.4% |
| Value vs quality | XGBoost | 6 | 76 | 4 | 70 | 93.6% | 47.4% |

**Table 4.4. Confusion-matrix diagnostics at the 0.5 threshold**

XGBoost provides the most balanced class treatment for the two momentum pairs. For momentum versus value, it predicts the positive class in 59.6% of observations compared with an actual rate of 54.5%. For momentum versus quality, its positive prediction share is 46.8% compared with an actual rate of 53.2%. Logistic regression is most balanced for value versus quality, where its positive prediction rate is 52.6% compared with the actual rate of 47.4%.

The extreme positive-class tendencies of logistic regression for momentum versus value and XGBoost for value versus quality materially inflate their recall and F1 scores. These cases show that favourable positive-class metrics can coexist with poor specificity and weak balanced classification.

The probability results introduce a further trade-off. An uninformative forecast of 0.5 for every observation has a log-loss of approximately 0.693. Logistic regression records log-loss between 0.684 and 0.688 for all three pairs, representing only a small improvement but avoiding severe overconfidence. The GAM produces log-loss between 0.932 and 1.609, indicating that some incorrect forecasts receive excessive probability. XGBoost has stronger average ranking and thresholded performance, but its log-loss exceeds 0.693 for both momentum pairs. Thus, better discrimination does not necessarily imply better probability quality.

## 4.5 Global feature importance

Global feature importance is measured using the mean absolute SHAP value from the saved model explanations. The values indicate which inputs exerted the greatest influence on each fitted model. They do not establish causal relationships or demonstrate that the variables contain stable out-of-sample information. This distinction is important because all final ROC-AUC values remain below 0.60. Rankings are compared more cautiously than raw SHAP magnitudes because the scale and distribution of attributions differ across model classes.

### 4.5.1 Momentum versus value

| **Rank** | **Logistic** | **GAM** | **XGBoost** |
|:---|:--:|:--:|:--:|
| 1 | USDZARV1Y.Curncy_QS (0.034) | relvol_value_vs_quality_12 (2.720) | SA_RB_Curvature (0.211) |
| 2 | USDZARV1M.Curncy_QS (0.032) | relvol_momentum_vs_quality_12 (2.550) | relvol_momentum_vs_value_12 (0.201) |
| 3 | VolTermStructPC1 (0.012) | USDZAR_EntDyn (1.269) | MXWD.Index.PE.Ratio_QS (0.188) |
| 4 | EURUSDV6M.Curncy_QS (0.004) | GlobalPropertyVol12 (1.235) | mom1m_momentum_vs_quality (0.184) |
| 5 | JALSH_Fundamental_PC1 (0.003) | MXWD_Fundamental_PC1 (1.083) | mom1m_momentum_vs_value (0.162) |

**Table 4.5. Top five global SHAP features for momentum versus value**

Logistic regression is dominated by currency-volatility and volatility-term-structure variables. The GAM places the greatest weight on relative-volatility measures, followed by foreign-exchange entropy, global property volatility and a global fundamental component. XGBoost draws on a broader combination of local real-yield-curve curvature, pair-specific relative volatility, global valuation and short-horizon factor momentum. The rankings suggest that the nonlinear models represent momentum-versus-value leadership using both factor-specific instability and wider market-state information.

<figure>
<img src="media/image5.png" style="width:5.83333in;height:2.72026in" alt="Final global feature importance for momentum versus value" />
<figcaption><p><strong>Figure 4.5. Final global SHAP importance for momentum versus value.</strong></p></figcaption>
</figure>

### 4.5.2 Momentum versus quality

| **Rank** | **Logistic** | **GAM** | **XGBoost** |
|:---|:--:|:--:|:--:|
| 1 | zrel_value_vs_quality_12 (0.033) | relvol_momentum_vs_quality_12 (1.891) | VolTermStructPC1 (0.203) |
| 2 | zrel_momentum_vs_quality_12 (0.029) | relvol_value_vs_quality_12 (1.853) | TOP40.Index.Price.to.Ebitda_Diff (0.183) |
| 3 | TOP40.Index.Price.to.Ebitda_Diff (0.024) | TOP40.Index.Price.to.Ebitda_Diff (0.600) | zrel_value_vs_quality_12 (0.145) |
| 4 | GLD_StructBreak (0.021) | volratio_value_quality_12 (0.538) | GLD_StructBreak (0.083) |
| 5 | mom3m_momentum_vs_quality (0.020) | MichgnConcIndx (0.456) | volratio_momentum_quality_12 (0.073) |

**Table 4.6. Top five global SHAP features for momentum versus quality**

Momentum-versus-quality predictions use a combination of standardised relative performance, relative volatility, valuation, commodity-regime and sentiment information. Relative factor variables appear prominently in all three models, while the local-versus-global EBITDA valuation differential ranks in the top three for each specification. Despite this partial overlap, the ordering differs materially across models. The evidence therefore points to multiple representations of the factor pair rather than a single stable driver.

<figure>
<img src="media/image6.png" style="width:5.83333in;height:2.71855in" alt="Final global feature importance for momentum versus quality" />
<figcaption><p><strong>Figure 4.6. Final global SHAP importance for momentum versus quality.</strong></p></figcaption>
</figure>

### 4.5.3 Value versus quality

| **Rank** | **Logistic** | **GAM** | **XGBoost** |
|:---|:--:|:--:|:--:|
| 1 | zrel_value_vs_quality_12 (0.244) | relvol_value_vs_quality_12 (1.999) | zrel_value_vs_quality_12 (0.054) |
| 2 | TOP40.Index.Price.to.Ebitda_Diff (0.153) | relvol_momentum_vs_quality_12 (1.910) | USDZARV1M.Curncy_QS (0.019) |
| 3 | RandPPP_Factor_Inst_TS_ST (0.126) | GlobalPropertyVol12 (0.751) | zrel_momentum_vs_quality_12 (0.018) |
| 4 | SA_NB_Slope (0.118) | RandPPP_Factor_Inst_TS_ST (0.650) | MichgnConcIndx (0.012) |
| 5 | VolTermStructPC1 (0.081) | SA_NB_Slope (0.608) | GenComIndexPC1_QS (0.012) |

**Table 4.7. Top five global SHAP features for value versus quality**

The standardised value-versus-quality relative-performance measure ranks first for logistic regression and XGBoost, while the GAM is dominated by relative-volatility variables. Valuation, purchasing-power-parity, yield-curve, currency-volatility, sentiment and commodity variables also appear. Logistic regression is the best-performing model for this pair, but its AUC of 0.585 remains modest. The feature rankings should therefore be interpreted as a description of the fitted decision function rather than evidence of stable economic causality.

<figure>
<img src="media/image7.png" style="width:5.83333in;height:2.71912in" alt="Final global feature importance for value versus quality" />
<figcaption><p><strong>Figure 4.7. Final global SHAP importance for value versus quality.</strong></p></figcaption>
</figure>

### 4.5.4 Cross-pair feature synthesis

Three broad patterns emerge from the SHAP rankings. First, pair-specific relative-return and relative-volatility measures recur frequently, indicating that the models rely on the recent state and dispersion of the factor relationships. Second, market-state variables - including currency volatility, yield-curve shape, volatility-term-structure and commodity-regime indicators - contribute across several specifications. Third, the importance rankings vary by model class and factor pair. The results do not reveal a single universal economic variable that drives factor leadership across all nine fitted models.

Because the predictive performance is modest, feature attribution provides exploratory evidence about model behaviour rather than confirmation of robust forecasting mechanisms. Stability analysis across retuning windows would be required before claiming that the rankings persist through time.

## 4.6 Validation, reproducibility and limits of inference

The final CPCV design strengthens the internal consistency of the analysis by using two held-out groups, ten training-validation combinations and four back-test paths for hyperparameter selection. The separate expanding walk-forward evaluation maintains chronological ordering and applies the same 12-week temporal gap to every model. The final comparison is therefore based on common target definitions, predictors, dates and validation settings.

Several limitations remain. First, the evaluation block contains only 156 weekly observations, while the trend-scanning labels display temporal persistence. Small differences in point estimates should therefore not be interpreted as statistically established without dependence-aware confidence intervals or paired tests. Second, the same evaluation period has been consulted during model development, meaning that it should not be described as a completely untouched final holdout. Third, the current saved artifacts contain the initial parameter selection, but do not preserve every parameter choice made during the two subsequent 52-week retuning events. Those retuning dates, training windows, CPCV scores and selected parameters should be exported before final submission. Fourth, the analysis evaluates statistical classification and does not establish portfolio profitability after turnover, transaction costs and implementation constraints.

Finally, the descriptive analysis of benchmark-relative factor returns is not included in this report because an aligned weekly benchmark series has not been verified in the repository. Raw factor-return figures should not be presented as active-return results. This omission does not change the pairwise classification results because the final target is defined directly from the relative performance of the two factors, but it limits conclusions about the standalone active behaviour of each factor.

## 4.7 Chapter conclusion

The final results provide modest and factor-pair-dependent evidence of predictability. XGBoost produces the strongest results for momentum versus value and momentum versus quality, while logistic regression performs best for value versus quality. XGBoost also records the highest mean accuracy, ROC-AUC, balanced accuracy and MCC across the three pairs. Its average advantage does not amount to uniform dominance: logistic regression has the strongest probability quality and leads the value-versus-quality pair, while several models produce strongly asymmetric classifications at the 0.5 threshold.

The findings therefore qualify the proposition that increasing model flexibility necessarily improves out-of-sample prediction. Greater flexibility is beneficial for the two momentum pairs, particularly momentum versus value, but the linear model is more effective for value versus quality. Even the best AUC is below 0.60, and the highest MCC is 0.218. The evidence supports limited predictive structure rather than a strongly separable or deployment-ready factor-timing signal.

The SHAP analysis indicates that the fitted models use a combination of relative-factor dynamics, volatility, currency conditions, valuation, yield-curve information, commodity regimes and sentiment. These patterns differ across model classes and factor pairs and should remain exploratory given the modest predictive performance.

| **Research question** | **Final answer** |
|:---|:--:|
| To what extent can the models predict relative factor leadership out of sample? | Predictability is modest and varies by pair. The strongest results are XGBoost for momentum versus value and logistic regression for value versus quality. |
| Does increasing model flexibility produce consistent improvements? | No. XGBoost leads on average and for both momentum pairs, but logistic regression leads value versus quality and provides better probability quality. |
| Which variables exert the greatest influence on the fitted models? | Relative-return and relative-volatility features recur, together with currency, valuation, yield-curve, volatility and sentiment indicators. The rankings are model- and pair-specific. |

**Table 4.8. Synthesis of the research questions**

Overall, the study does not identify a universally superior model or a general complexity premium. It provides a more qualified conclusion: flexible machine-learning models can extract limited predictive information from some South African relative factor relationships, but performance remains close to chance in several specifications and has not yet been translated into demonstrated economic value.
