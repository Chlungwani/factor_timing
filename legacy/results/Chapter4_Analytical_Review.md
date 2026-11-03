# Analysis of the Chapter 4 results

This review addresses the current Chapter 4 amendment draft and the model outputs saved on 2 November 2026. Its main finding is that **XGBoost has the strongest reported point estimates, but the nature and strength of its advantage differ across the three factor pairs**. Value versus quality provides the clearest evidence of discrimination; momentum versus value exposes a gap between ranking and probability quality; momentum versus quality exposes a gap between aggregate accuracy and recognising both leadership states.

The chapter already contains useful analysis, particularly in its conclusion. The main editorial change is to bring that reasoning into the individual results sections. Each section should establish a finding, explain the evidence behind it, consider an alternative explanation, and state what it implies for the research question. More descriptions of curve shapes or lists of features will add less than these connections.

All nine sets of saved predictions were checked against the reported accuracy, ROC-AUC and log-loss. The additional diagnostics below were calculated from the same predictions, without fitting or selecting new models. The existing Word documents have not been changed.

## The argument the chapter should develop

The research question concerns whether greater functional flexibility improves prediction. The answer has three parts:

1. XGBoost leads all three pairs on the three measures in Table 4. This establishes the ordering of the observed results.
2. The GAM does not provide an intermediate improvement between logistic regression and XGBoost. The results therefore provide no consistent progression from linearity to smooth nonlinearity to interactions.
3. The paired confidence intervals do not establish XGBoost's superiority over logistic regression. The evidence supports an advantage for the fitted tree ensemble in this evaluation, with the strongest evidence concentrated in value versus quality. It does not establish a general benefit from increasing model complexity.

Distinguish four questions throughout: Does a model rank outcomes better than chance? Does it outperform another model? Does it assign useful probabilities? Does it produce an implementable investment strategy? A positive answer to one does not settle the others.

## What the results mean by factor pair

| Factor pair | Evidence for XGBoost | Main qualification | Analytical conclusion |
|---|---|---|---|
| Momentum versus value | Accuracy 62.8%; AUC 0.625; balanced accuracy 62.1% | AUC interval 0.438–0.797 includes chance; log-loss 0.715 exceeds the constant 0.5 forecast's 0.693 | The strongest observed classification for this pair still supplies probabilities whose overall log-loss is worse than an uninformative forecast |
| Momentum versus quality | Accuracy 61.5%; AUC 0.589; log-loss 0.678 | Predicts momentum in 89.1% of weeks; identifies 96.5% of momentum-led weeks but only 19.7% of quality-led weeks | Aggregate accuracy conceals an asymmetric classifier; the ranking also changes substantially between periods |
| Value versus quality | Accuracy 60.3%; AUC 0.664; AUC interval 0.569–0.757, or 0.529–0.797 after the nine-comparison adjustment | Paired AUC difference against logistic regression includes zero; performance deteriorates after the first period | The clearest evidence of above-chance ranking, with a more consistent ranking across periods, but an uncertain incremental gain over the linear benchmark |

The observed majority-class shares are 54.5%, 54.5% and 50.6%, respectively. XGBoost's accuracies exceed these descriptive reference levels by 8.3, 7.1 and 9.6 percentage points. These shares use the evaluation outcomes and should be labelled as retrospective reference levels. A deployable majority-class rule would have to select its class using information available before each forecast.

### Section 4.2.1 Momentum versus value

The key result is the disagreement between measures. XGBoost improves classification and ranking relative to the other fitted models, but its probability estimates still lose to a constant 0.5 forecast under log-loss. This gives the discussion a substantive finding beyond identifying the highest number in a table.

The annual results also qualify the pooled AUC. XGBoost's AUCs are 0.601, 0.571 and 0.555 in the three successive 52-week periods. Its pooled AUC of 0.625 is higher than all three. Pooled AUC compares weeks across periods as well as within periods, so part of the pooled separation may reflect differences between periods. These figures suggest modest within-period discrimination and a declining point estimate, rather than uniformly strong weekly timing.

The confident errors provide a plausible explanation for the poor probability scores of the other models. Logistic regression is confidently wrong in 17 weeks, accounting for 41.1% of its total log-loss; the GAM is confidently wrong in 22 weeks, accounting for 62.0%. Thus a relatively small subset of observations contributes a disproportionate share of the probability loss. The weak penalties selected by these models are consistent with this pattern, but the results do not isolate regularisation as its cause.

**Suggested replacement or concluding paragraph**

> For momentum versus value, XGBoost's advantage is strongest in classification rather than probability quality. Its accuracy exceeds the observed majority-class share by 8.3 percentage points, and its balanced accuracy of 62.1% indicates that the result is not solely due to favouring momentum. However, its log-loss of 0.715 remains worse than the 0.693 obtained by assigning each outcome a probability of 0.5. It therefore separates the classes more successfully than the other fitted models without producing a corresponding improvement over an uninformative probability forecast. Its within-period AUC also declines from 0.601 to 0.571 and 0.555, and its pooled AUC confidence interval includes 0.5. The evidence supports a modest observed classification advantage whose statistical reliability and probability quality remain uncertain.

### Section 4.2.2 Momentum versus quality

The main analytical issue is asymmetric detection. XGBoost correctly identifies 82 of 85 momentum-led weeks but only 14 of 71 quality-led weeks. Balanced accuracy is consequently 58.1%, compared with 57.0% for logistic regression. Its advantage over logistic regression narrows from 3.8 percentage points in overall accuracy to 1.1 points in balanced accuracy. MCC still favours XGBoost, at 0.259 versus 0.141, so the appropriate conclusion is that its observed advantage treats the two classes unevenly.

Separate a constant **class decision** from a constant **probability**. In the first 52-week period, every XGBoost probability exceeds 0.5, yet its AUC is 0.746. The model ranks outcomes within that period despite making the same thresholded decision every week. In the second period it again always predicts momentum, but its AUC falls to 0.333. Its third-period AUC recovers to 0.748. These shifts show that classification behaviour and ranking stability are separate issues.

The present statement that most ranking information comes from a single year needs qualification: all three models rank well in the third period, but XGBoost also ranks well in the first. The common second-period deterioration is suggestive of changing predictive relationships, although the subperiod results do not identify a particular economic regime or its cause.

**Suggested replacement or concluding paragraph**

> Momentum versus quality gives a less convincing case for a broadly useful timing model than the aggregate accuracy suggests. XGBoost identifies almost all momentum-led weeks but only 19.7% of quality-led weeks, so its predictions provide limited ability to recognise transitions towards quality. Its balanced accuracy exceeds logistic regression's by only 1.1 percentage points. The probability ranking is also unstable: XGBoost's AUC changes from 0.746 to 0.333 and then 0.748 across the three evaluation periods. Predicting momentum throughout the first two periods therefore represents the same class decision under very different ranking performance. The results suggest that XGBoost's overall advantage depends on the interaction between the decision threshold, the prevailing leadership state and period-specific predictive relationships.

### Section 4.2.3 Value versus quality

This is the strongest positive finding because several pieces of evidence agree. XGBoost's AUC exceeds 0.5 under the reported block bootstrap, including the adjustment for nine model–pair results. It recognises both classes more evenly, with sensitivity of 58.4% and specificity of 62.0%. Its subperiod AUCs are 0.777, 0.633 and 0.629, whereas logistic regression falls from 0.788 to 0.443 and 0.467.

The distinction between evidence against chance and evidence against logistic regression matters here. XGBoost's paired AUC advantage over logistic regression is 0.066, with a 95% interval of −0.037 to 0.177. The results support above-chance discrimination for XGBoost, but do not establish that it is superior to logistic regression on this pair.

There is also deterioration that the present discussion underplays. XGBoost's accuracy falls from 67.3% to 57.7% and 55.8%; its log-loss rises from 0.579 to 0.671 and 0.759. Ranking remains above chance in point estimates, but later probabilities are less useful under log-loss. These small, dependent subperiod samples are descriptive checks, not three independent confirmations.

**Suggested replacement or concluding paragraph**

> Value versus quality provides the clearest evidence of predictable relative leadership. XGBoost's AUC interval remains above 0.5 after adjustment for the nine reported model–pair comparisons, and its within-period AUC exceeds 0.6 in all three periods. This persistence contrasts with logistic regression, whose strong first-period ranking does not continue in the following periods. Nevertheless, XGBoost's declining accuracy and rising log-loss indicate that the strength and probability quality of the relationship weaken over time. Its paired AUC advantage over logistic regression also has an interval that includes zero. The finding therefore supports above-chance ranking by the tree ensemble more clearly than it establishes an incremental benefit over the regularised linear model.

## Section 4.3 Probability quality and classification errors

Use this section to connect the measures. Accuracy evaluates decisions at 0.5; AUC evaluates ranking across thresholds; log-loss evaluates the probabilities assigned to the realised outcomes. The momentum–value result demonstrates that good relative ranking can coexist with disappointing probability quality even when the same model leads the reported measures.

Two wording corrections are needed:

- The sentence beginning “Only XGBoost has a log-loss below the uninformative 0.693” is inaccurate. Logistic regression also does so for value versus quality, with log-loss 0.686. XGBoost is the only model to do so for **two** pairs.
- Similar predicted-positive and actual-positive shares do not establish balanced class treatment. Use sensitivity and specificity to support that claim. A classifier can match the overall class share while misidentifying many individual observations.

Do not equate lower log-loss with better calibration alone. Log-loss reflects both calibration and discrimination. Without a reliability analysis, the defensible wording is “better probability quality under log-loss”. The [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html) explains this distinction.

**Suggested synthesis paragraph**

> The combined diagnostics show that successful ranking, balanced class detection and useful probabilities do not necessarily occur together. XGBoost's momentum–value forecasts improve ranking and classification relative to the alternatives, but lose to an uninformative forecast under log-loss. Its momentum–quality forecasts have lower aggregate log-loss but recognise relatively few quality-led weeks. Value versus quality gives the most consistent combination of discrimination and class detection, although its probability quality deteriorates in the final period. These differences matter because a model used to rank opportunities may be assessed differently from one used to trigger allocation changes or set probability-based portfolio weights. The present results assess predictive performance and do not establish the investment value of those decisions.

## Section 4.4 Feature importance

The current paragraphs largely list features and then cite related literature. Strengthen them by explaining how each pattern relates to the model's measured performance, while keeping proposed mechanisms explicitly tentative.

- **Momentum versus value:** relative volatility leads all three models, but only XGBoost has an appreciable pooled ranking advantage. Agreement on a feature category therefore does not establish that the category is sufficient for prediction. The result suggests that how a model represents and combines the information may matter.
- **Momentum versus quality:** logistic regression and XGBoost share the leading valuation and yield-curve variables, while their AUCs differ by only 0.019. This is consistent with substantial overlap in the information they use. It does not show that valuation successfully times the pair, particularly given the unstable subperiod results.
- **Value versus quality:** recent relative performance and relative volatility are prominent in the stronger models. This supports a tentative interpretation involving factor dynamics. Mean absolute SHAP values do not reveal whether stronger recent value performance predicts continuation or reversal, or whether higher relative volatility favours value or quality.

There is a specific implementation limit to state. The notebook SHAP functions fit a reference model on the pre-evaluation training sample using the initially selected parameters and calculate importance on that training sample. They do not aggregate the explanations of the successive walk-forward models. Consequently, their rankings cannot explain why forecast performance changed between evaluation periods. In particular, the small momentum–quality SHAP magnitudes describe the reference fit; they do not independently demonstrate what drove its changing walk-forward forecasts.

The [SHAP documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) defines the values in relation to model output. [Kumar et al. (2020)](https://proceedings.mlr.press/v119/kumar20e.html) discuss limits to treating these attributions as feature importance explanations.

**Suggested synthesis paragraph**

> The feature rankings suggest that the fitted models represent different aspects of relative factor dynamics. Volatility measures are prominent for momentum versus value, valuation and yield-curve information lead the linear and tree models for momentum versus quality, and recent relative performance is prominent for value versus quality. These patterns help describe the information used by the reference fits, but their relevance must be assessed alongside predictive performance. An important feature in a weak classifier is not thereby a reliable timing signal, and mean absolute attributions establish neither the direction of a relationship nor its economic cause. Because the explanations concern training-sample reference fits, they also do not establish the stability of these relationships throughout the walk-forward evaluation.

## Section 4.5 Answering the research question

The comparison is consistent with interactions being useful, but it does not identify their contribution. XGBoost and the GAM differ in their representations, penalties and fitting procedures. The GAM's weaker results cannot isolate interactions as the reason XGBoost performs better. Gu, Kelly and Xiu identify interaction-related gains in a different return-prediction setting; that supplies a possible interpretation for discussion, rather than evidence that the same mechanism has been established here. See their [published paper](https://dachxiu.chicagobooth.edu/download/ML.pdf).

The reported mean-AUC advantage of XGBoost over the GAM is 0.110, with a 95% interval of 0.006–0.221. The advantage over logistic regression is 0.074, with an interval of −0.045–0.196. Describe the former as an unadjusted paired interval excluding zero; the nine-result adjustment applied to individual AUCs does not automatically apply to these model comparisons.

Three claims should be narrowed:

- Replace “smooth nonlinearity alone adds nothing” with “the fitted additive nonlinear specifications provide no consistent improvement in this evaluation”. This states what was tested without generalising to all GAMs or nonlinear relationships.
- Replace “the result is therefore not merely the best of many tests” with “the result remains above chance after adjustment for the nine reported model–pair comparisons”. This adjustment does not address any earlier development decisions informed by the same evaluation block.
- Remove 0.7 as a universal standard of acceptable discrimination. The study's targets and evaluation do not establish 0.7 as an investment-relevant cutoff. Discuss the observed effect sizes, uncertainty and stability instead.

The bootstrap resamples the saved prediction–outcome sequences without refitting models or repeating feature selection and tuning. Its intervals quantify uncertainty under that resampling design; they do not capture every source of uncertainty in developing the forecasting pipeline. The block-length sensitivity is useful supporting evidence: value–quality XGBoost remains above chance for the reported lengths of 1, 6, 12 and 24 weeks.

**Suggested replacement conclusion**

> The results provide evidence of limited and factor-pair-dependent predictability. XGBoost has the highest accuracy and ROC-AUC and the lowest log-loss for each pair, but this ordering does not imply equally strong evidence across the three forecasting problems. Value versus quality provides the clearest result: its AUC remains above chance after adjustment for the nine reported model–pair comparisons, and its ranking persists across the three evaluation periods. Momentum versus value combines an observed classification advantage with weaker probability quality, while momentum versus quality combines asymmetric class detection with substantial variation in ranking performance.
>
> The results provide no consistent progression in performance as functional flexibility increases. The fitted GAMs fail to improve consistently on logistic regression, and although XGBoost performs better in point estimates, its mean-AUC advantage over the regularised linear model remains uncertain under the paired bootstrap. Interactions and differences in regularisation offer possible explanations for the tree ensemble's advantage, but the present comparison does not isolate their effects. The study therefore supports selective predictive structure in relative factor leadership, with its strongest evidence in value versus quality, while leaving the general superiority and investment value of more flexible models unresolved.

## How to integrate the changes

Keep the existing tables and figures. Replace the repeated descriptions of which ROC curve is highest with the analytical paragraphs above. Retain only the numerical details needed to substantiate each argument. Add the sensitivity and specificity of the momentum–quality classifier and the three-period deterioration in value–quality probability quality; both sharpen the interpretation without requiring new model runs.

Keep short implications beside the relevant findings in Sections 4.2–4.4, then use Section 4.5 to answer the research question across pairs. Broader investment implications and future research can remain in Chapter 5, consistent with the study's stated scope.

## Evidence used

- Current chapter: [Chapter 4 amendment draft](Chapter4_Results_Amendments_Highlighted.docx), Sections 4.2–4.5.
- Reported metrics: [metrics summary](model_outputs/metrics_summary.csv); additional diagnostics recomputed from the nine files in [saved probabilities](model_outputs/probs/).
- Uncertainty: [individual AUC intervals](model_outputs/bootstrap/auc_ci.csv), [paired AUC differences](model_outputs/bootstrap/auc_paired_differences.csv), [mean AUC comparisons](model_outputs/bootstrap/auc_mean_across_pairs.csv), and [block-length sensitivity](model_outputs/bootstrap/auc_ci_block_sensitivity.csv).
- Explanation scope: the `compute_shap_importance` functions and their calls in [XGBoost](../model_training/xgboost_cpcv_wf.ipynb), [logistic regression](../model_training/logistic_classifier_roc.ipynb) and [GAM](../model_training/gam.ipynb).

The older Chapter 4 revision map and alignment notes contain superseded results. Their numerical conclusions should not be transferred into this version of the chapter.
