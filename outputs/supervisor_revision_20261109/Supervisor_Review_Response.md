# Response to supervisor pre-submission review

The separate thesis copy implements the supervisor’s 5 November 2026 review against the confirmed source `legacy/27717801_ResearchAssignment_vf - EG Mar 2026.docx`. The source thesis is preserved. As subsequently requested, the revision includes a full corrected analysis, rather than only wording changes around the previous results.

## Assessment and resulting evidence

The concerns about back-filling, evaluation-period feature selection, the initial tuning boundary and unsupported inference were substantive. They required analytical changes before the old headline findings could be retained. This revision removes back-filling, freezes feature selection before evaluation, purges the first tuning boundary and fits preprocessing within each CPCV training fold. All nine model–pair runs completed and their metrics were independently recomputed from the saved predictions.

The joint corrections change the predictor set from 50 to 47 and the usable labelled history to 774 weeks beginning 29 March 2010. The 156 evaluation dates and target labels are unchanged. Changes in performance reflect this joint redesign; they are not an ablation identifying the contribution of each correction separately.

XGBoost has the largest point-estimate ROC-AUC for all three pairs, but no paired AUC advantage over logistic regression is established. Only its value-versus-quality unadjusted 95% interval excludes 0.5; no model–pair AUC interval excludes 0.5 after adjustment across the nine estimates. The abstract and conclusions reflect this limited evidence. XGBoost no longer leads every metric: logistic regression has higher accuracy for momentum versus value and value versus quality, and slightly lower log-loss for momentum versus value.

## Response by major comment

| Comment | Evaluation and action |
| --- | --- |
| A1 Back-filling | The original audit found five affected predictors. The new pipeline uses forward filling only and drops pre-inception rows before fitting. It creates zero back-filled cells. The first usable modelling date becomes 29 March 2010. |
| A2 Feature selection overlap | Screening, quantile bins, MI distances, Ward profile clustering and representative choice are fitted using observations no later than 1 November 2021. The selection uses 548 complete pre-evaluation weeks, not the 2022–2025 block. The retained 29 representatives plus 18 designed features are used for every model and pair. |
| A3 Initial tuning boundary | Initial tuning now uses the same purged 606-observation window as the first forecast fit. Scalers are fitted within each CPCV training fold, as well as within each expanding forecast window. The saved original notebook fitted its initial scaler on the pre-evaluation sample, rather than the whole evaluation sample; fold-specific scaling and the boundary purge still required correction. |
| A4 Inference | Table 4 now has 95% block-bootstrap intervals for accuracy and AUC plus majority-class accuracy. Appendix C reports paired AUC and log-loss comparisons, approximate centred-bootstrap tests, Holm adjustments and six-/24-week block sensitivity. The abstract and Chapter 5 distinguish unadjusted evidence, multiplicity and model differences. |
| A5 CPCV versus evaluation | Appendix C reports newly computed combination and path diagnostics, including a no-feature baseline. Chapter 4 discusses period dependence. The mean of ten combination AUCs selects parameters; four pooled paths are diagnostics. |
| A6 Missing appendices | Appendices A–C are rebuilt with complete, contiguous headings and tables. The new Appendix C.5 contains inference. Claims based on the old all-fixed-setting walk-forward sensitivity results are removed; those results do not apply to the new feature set and training history. |
| A7 Counts and horizons | Counts are reconciled from the files: 939 factor-return weeks, 784 complete lagged predictor rows, 774 labelled observations, 606 initial purged training rows, 761 final training rows and 156 evaluation weeks. The code’s maximum is 11 observations including the starting date, hence ten future weeks and ten unresolved final labels. The 12-week purge is conservative. Figures 3–5 are regenerated to match the actual target series. Figures 1–2 retain their separate descriptive monthly sample, explicitly identified in the text. |
| A8 Feature definitions | Appendix A lists all 47 exact modelling names and their definitions. The relative-factor recipes are specified from the code, including 4-/12-week momentum and 12-week risk measures. Duplicate and unrelated candidate-feature tables are removed. BCMPEBLS.Index uses the author’s requested description “US business-cycle index”, with a Word comment asking for source confirmation. Upstream provider recipes remain a documented limitation. |
| A9 Interpretation | The results discuss class recognition, majority benchmarks, confident errors, AUC versus probability quality, regularisation, spline dimension and convergence. The GAM’s outcome is not presented as an inherent failure of additive nonlinear models. Attribution discussion uses the new rankings and selected settings; the old top-ten relative-volatility claim is removed. Identical momentum class totals are verified; the sequences differ in 34 weeks. |
| A10 Local literature | The South African section distinguishes the 2022 style-momentum paper from the 2024 ML comparison and discusses Fraser and Page plus Muller and Ward. Missing references are added and duplicate entries merged. Publisher metadata is used to correct authors, outlets, volumes, pages, DOIs and issue years; journal versions are separated from earlier working papers. |

## Presentation and editorial corrections

The full title precedes “by” and the author’s name. The placeholder acknowledgements page is removed; no personal acknowledgement is invented. The declaration wording is corrected. Chapter and section numbering are explicit and consistent; appendices are labelled A, B and C. Captions are placed beside their tables or figures, all 17 figures and 19 numbered tables appear in their respective lists, and contents and page fields are refreshed in Word.

Equations are numbered consecutively from 1 to 20. The trend regression uses CR and within-window index s. The boosting objective applies the sigmoid to the accumulated tree score, and leaf weights v_j are distinct from observation weights w_t. Legacy equation-number fields that reset cached numbers are replaced. Typographical corrections span Word runs, and UK spelling is used in the thesis prose while published reference titles retain their source spelling.

Citation sources are deduplicated and citation fields use title suppression. The Word switch `\t` suppresses a title; the earlier diagnosis that it enabled a title was corrected. Published issue years are used with the corresponding volume/page records, including Piovezan et al. 2024, Feng et al. 2024 and Leippold et al. 2022. The bibliography is restricted to cited sources.

## Author checks before submission

1. Confirm the Bloomberg description for BCMPEBLS.Index. The current description follows the author’s instruction and the thesis contains a verification comment.
2. Confirm upstream PCA estimation windows, quantile-scaling histories, rolling-statistic recipes and macroeconomic release vintages where source metadata is available. The thesis explicitly limits its claim about historical availability; the model rerun cannot independently establish provenance absent from the supplied inputs.
3. Review the new, more cautious conclusions and the decision to omit the unfinished acknowledgements. Personal acknowledgements can be supplied by the author if desired.

The statistical results and reproducible analysis are stored in the adjacent `analysis` folder. See `analysis/README.md` and `analysis/provenance.json` for inputs, execution and checks.

## Final bibliography checks

The final pass removes journal volume and page details from the Arnott et al. working-paper entry, consistent with its [SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3040949). The Haddad et al. reference now uses the [journal DOI](https://academic.oup.com/rfs/article/33/5/1980/5753962), and the Piotroski reference uses the supplement and page range on the [author's publication record](https://www.gsb.stanford.edu/faculty-research/faculty/joseph-d-piotroski). The Lintner and Murphy page ranges follow the [journal issue contents](https://www.jstor.org/stable/i333068) and [publisher record](https://journals.ametsoc.org/abstract/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml). Campbell et al. is identified as the original edition, with the [publisher's 1997 copyright year](https://press.princeton.edu/books/hardcover/9780691043012/the-econometrics-of-financial-markets).
