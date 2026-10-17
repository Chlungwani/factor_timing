# Chapters 1–3 alignment, Chapter 4 structure, and enhancement backlog

Companion to `Chapter4_Revision_Map.md`. Scope: the **final corrected results only** (the nine
model–pair runs in `results/model_outputs/metrics_summary.csv`, test block 2022-01-31 to
2025-01-20, 156 weekly observations). The superseded pre-correction numbers are not referenced
except where the thesis text must be changed to stop claiming them.

Everything quantitative below was recomputed from `results/model_outputs/probs/*.csv` and
`data_input/factor_returns.csv`.

---

## Part 0 — Two findings that change the editing job

### 0.1 The pairwise target is already benchmark-neutral

`Chapter4_Revision_Map.md` holds Section 4.2 back because no weekly benchmark series exists.
That is correct as to the data: `data_input/benchmark_returns.csv` and
`data_input/factor_benchmark_returns.xlsx` are **monthly** (243 and 257 rows, 2004–2025), while
`data_input/factor_returns.csv` is **weekly** (944 rows, 2007-03-05 to 2025-03-31).

But the modelling target does not need the benchmark. Verified numerically:

```
Momentum_vs_Value  ==  factor_returns.Momentum - factor_returns.Value
corr = 1.0000,  max|error| = 0.000000
```

Because every label is a **difference of two factor returns**, any common benchmark cancels
exactly:

$$(r_A - r_B) \;=\; (r_A - r_{\text{bm}}) - (r_B - r_{\text{bm}})$$

So the labels, the relative-factor features, and all of Sections 4.3–4.5 are **identical in raw
and active space**. This is an algebraic identity, not an approximation.

**What this means for the write-up.** The supervisor's active-space request is fully satisfied for
the model results by stating this identity in one short paragraph. Only the *descriptive level*
analysis in 4.2 — cumulative factor performance, rolling factor returns, factor-vs-factor
dependence — is genuinely benchmark-dependent, and that can be presented monthly from the
existing workbook (2004–2025, 257 months) as a descriptive section, clearly labelled as monthly
and distinct from the weekly modelling sample. There is no need to hold 4.2 back entirely.

### 0.2 Chapter 2 already predicts the corrected results

The literature review does not need rewriting. It needs **re-pointing**. It already contains the
sceptical strand that the corrected results confirm:

| Paragraph | Existing claim | Relation to corrected results |
|---|---|---|
| ¶234 | "marginal benefits of increasingly complex models may exhibit diminishing returns… even well-specified linear models can sometimes match or exceed the net performance of more complex architectures on a true out-of-sample basis" | **Directly confirmed.** Logistic wins momentum–quality. |
| ¶237 | In emerging markets "the magnitude of outperformance is smaller and more sensitive to market structural characteristics" | **Directly confirmed.** AUC range 0.483–0.598. |
| ¶242 | Flexible-model gains "often reflect sensitivity to noise rather than persistent structure" | **Confirmed by the leakage audit.** Best AUC fell 0.768 → 0.598. |
| ¶244 | Transaction costs "materially erode gross returns"; few papers model them | **Now testable.** See Part 3, Enhancement B. |
| ¶245 | Data-snooping risk is "particularly salient in factor timing, where signals are inherently weak" | **Confirmed.** Nothing survives Holm correction. |
| ¶246 | Evidence "challenges the notion of universally superior ML architectures" | **Directly confirmed.** Model ranking changes by pair. |

The thesis is therefore not a failed study with a broken literature review. It is a study whose
corrected results land on the **sceptical** side of a trade-off its own Chapter 2 set up on both
sides. That is a defensible and coherent contribution. The required edits are to the *framing
sentences* that lean optimistic, not to the substance of the review.

---

## Part 1 — Chapter 1 (Introduction)

| Para | Current text | Required change |
|---|---|---|
| Abstract ¶1 | "optimised using purged combinatorial cross-validation (CPCV)" | → "five-fold **purged** cross-validation". With `CV_TEST_SPLITS = 1` the design yields five folds and **one** back-test path, so it is not combinatorial. Either correct the term or rerun with ≥2 held-out groups. |
| Abstract ¶1 | "evaluated under a **rolling** walk-forward validation framework" | → "**expanding**-window walk-forward" (751 → 906 observations). |
| Abstract ¶2 | "results demonstrate the superiority of XGBoost" | Replace. XGBoost has the highest *mean* AUC (0.570) but loses momentum–quality to logistic. State: weak, pair-dependent predictability; no uniform complexity premium. |
| Abstract ¶2 | "Nonlinear models deliver the strongest gains in factor pairs characterised by instability and dispersion" | Not supported — this was built on the leaked ranking. Delete. |
| Abstract ¶2 | "XGBoost provides the strongest discrimination… logistic regression produces more stable probability calibration" | Reverse-check: logistic has the *lowest* log-loss for momentum–quality (0.677) and best Brier skill (+0.028); XGBoost has the lowest log-loss for the other two pairs. There is no clean hierarchy. |
| ¶149 | "Building on prior evidence that shows that tree-based models outperform simpler approaches within the South African markets, we use XGBoost" | Keep as *motivation* (it is a citation, not a finding), but add a forward pointer that this study does not reproduce that ordering. |
| ¶150 | "This layered design allows for a structured evaluation of whether increasing model complexity yields meaningful economic value" | **Keep verbatim.** This is the correct framing and the answer is now "largely no". |
| ¶161 (Problem statement) | "we aim to **show meaningful predictive capacity** of ML" | Change to a neutral question: "we assess **whether** ML models can extract robust predictive signals". As written it presupposes the result. This is the single most important one-line fix in Chapter 1. |
| ¶165 (Objective 3) | "evaluate the robustness… using purged combinatorial cross-validation" | Same CPCV correction. |
| ¶174 | Lists only "average accuracy, ROC-AUC, and log-loss" | Add PR-AUC, balanced accuracy, MCC, confusion matrices. Mandatory — the supervisor asked for confusion matrices and the one-sided-prediction problem cannot be seen without them. |
| ¶175 | "feature importance is analysed using Shapley value decompositions" | Keep, but scope to **global mean |SHAP|** only (see Chapter 3 below). |
| ¶183 | "This framework operationalising purging, **annual** hyperparameter re-tuning, and forward-only evaluation, to produce **robust performance** across the different model classes" | Two fixes: retuning is every **52 predictions** (3 events over the test block), and "to produce robust performance" overclaims — the framework produces *credible estimates*, not good ones. |
| ¶192 (outline) | "examines robustness across validation windows" | The saved artefacts only contain the *initial* pre-OOS parameter selection; two of three retuning events were never persisted. Soften to "documents the validation configuration". |

**Chapter 1 verdict:** mechanically small. Roughly a dozen sentence-level edits, concentrated in
the abstract and ¶161. The chapter's architecture and the three objectives survive intact —
objective 2 ("whether greater model flexibility translates into improved predictive accuracy")
now simply has a negative answer, which is still an answer.

---

## Part 2 — Chapter 2 (Literature Review)

**Verdict: the lightest-touch chapter. Do not restructure it.**

| Para | Change |
|---|---|
| ¶249 | "providing substantial improvement in the predictive accuracy of factor signals over linear methods" — attribute this to the cited studies rather than stating it as settled fact ("studies report…"). |
| ¶250 | Already excellent. Strengthen the last sentence so the methodological choices are framed as *tests* of the sceptical critiques, not as safeguards that guarantee success. |
| ¶252 | The cited SA study (XGBoost lowest RMSE on monthly data) is now in tension with this study's findings. Add one sentence noting the differences in design — monthly vs weekly, regression vs classification, and whether comparable purging was applied — so the contrast reads as a contribution rather than a contradiction. |
| ¶253–254 | Reframe the gap statement. Currently: existing work is "limited in scope and methodological design". Keep that, and add that this study's contribution is partly **negative-result** evidence under strict temporal controls — which ¶245 already argues is what the field needs. |
| New closing ¶ | Add a short hypothesis statement derived from ¶234 and ¶246: *if* the diminishing-returns and market-contingency strands are correct, then in a small, noisy emerging-market sample we should expect weak, unstable, pair-dependent predictability with no monotonic complexity premium. This converts the corrected results from a disappointment into a **pre-registered-style expectation that was met.** |

That last addition is the highest-value single edit available in Chapter 2. It costs one
paragraph and reframes the whole thesis.

---

## Part 3 — Chapter 3 (Methodology)

This is where the real work is. The chapter currently documents a **pipeline that no longer
exists**.

### 3.1 Factual corrections (not optional — these are now false statements)

| Para | Current | Corrected |
|---|---|---|
| ¶387–393 | "Combinatorial cross-purged validation"; "CPCV generates multiple unique combinations"; "the number of back-test paths is …" | With N=5 groups and k=1 held out: 5 folds, **1** back-test path. Retitle the subsection **"Purged k-fold cross-validation"** and remove the combinatorial path arithmetic and Figure 3, or rerun with k≥2. |
| ¶400 | "models are trained on a rolling look-back window of **773 weeks**" | **Expanding** window, **751 → 906** observations. |
| ¶400 | "hyperparameters are re-optimised every 52 weeks… repeated three times" | Correct (3 retuning events), but add that **only the first selection is persisted to disk**. |
| ¶393 | "purge gap should be at least the size of the look-ahead window" | State the actual implementation: **12-bar purge** on walk-forward and **12-bar purge/embargo** in CV, against a trend-scanning horizon reaching *t*+10. |
| ¶403–411 | Metrics = accuracy, ROC-AUC, log-loss | Add PR-AUC, balanced accuracy, MCC, confusion matrices. Add the explanation of why F1/recall alone mislead under one-sided prediction. |
| ¶419, ¶421 | "Feature importance is evaluated using both **global and local** Shapley decompositions" + a full paragraph on regime-conditional local analysis | Only global mean |SHAP| (20 features per model-pair) is saved. Delete the local-decomposition promise or regenerate and persist observation-level values. |
| ¶360 | Custom logistic implementation | Add that **scaling is re-estimated inside each training window** (this was a leakage fix and is worth stating explicitly as a methodological safeguard). |
| ¶308–324 | Trend-scanning labels | Add that **unresolved end-of-sample labels are dropped, not forward-filled** — also a leakage fix, also worth claiming credit for. |

### 3.2 New subsection to add: 3.8.3 "Sources of look-ahead bias and their treatment"

Do not bury the leakage audit in Chapter 4. Put a short subsection in Chapter 3 that tabulates
each leakage channel and its fix, then report the *consequence* in Chapter 4:

| Channel | Pre-correction behaviour | Treatment |
|---|---|---|
| Label horizon overlap | Training ran to *t*−1 while labels at *t*−1 needed returns to *t*+9 | 12-bar purge before every test point |
| Unresolved final labels | Forward-filled | Dropped |
| Hyperparameter search scope | Could inspect the full sample | Restricted to data available at the prediction date |
| Logistic preprocessing | Scaler fitted on the full dataset | Re-fitted per training window |
| CV fold adjacency | No embargo | 12-bar purge/embargo around each held-out group |

Then in Chapter 4: best ROC-AUC **0.768 → 0.598**. That single number pair is one of the
strongest empirical contributions in the thesis — it quantifies, on a real emerging-market
dataset, how much apparent factor-timing skill is manufactured by ordinary validation sloppiness.
Frame it that way.

### 3.3 New subsection to add: 3.9.4 "Benchmark-neutrality of the pairwise target"

Three or four sentences stating the identity from Part 0.1 above, with the verification. This
closes the supervisor's active-space comment for Sections 4.3–4.5 at essentially zero cost.

### 3.4 New subsection to add: 3.9.5 "Statistical inference and multiple testing"

Required to support Enhancement A. Specify: block bootstrap (block length 12 to respect label
overlap) for AUC confidence intervals; Pesaran–Timmermann for directional accuracy; Holm
correction across the nine model–pair combinations; and the deflated-Sharpe selection benchmark
for the economic test.

---

## Part 4 — Proposed Chapter 4 structure

### 4.1 Problem with the current structure

The existing chapter is organised **pair-first**: three near-identical subsections (4.3.1–4.3.3),
each with its own metrics table, confusion table, ROC figure and probability figure. That is
~6 tables and ~6 figures to make three points, and it forces the reader to hold nine numbers in
mind to see the one finding that matters — that the model ranking is unstable across pairs.

The cross-pair comparison, which is the actual result, arrives last (4.3.4) and has to re-state
everything.

### 4.2 Recommended structure: finding-first, with pair detail demoted

```
4.1  Evaluation framework and validation configuration        (~1.5 pp)
     - common test block, purge, retuning, benchmark-neutrality identity
     - Table: corrected validation configuration
     - the one-paragraph CPCV/purged-k-fold correction

4.2  The factor environment                                   (~2 pp)
     - monthly active-space descriptives, 2004-2025, clearly labelled
     - weekly raw-space pairwise spreads (benchmark-neutral by construction)
     - Fig: cumulative + rolling pairwise spreads; Table: rank dependence

4.3  Out-of-sample predictive performance                     (~4 pp)
     4.3.1  Headline comparison
            >> Table 6: ONE 9-row master table (3 pairs x 3 models),
               columns = Acc | Bal.Acc | AUC [95% CI] | PR-AUC | MCC | LogLoss | Brier skill
               plus majority-class and lag-11 benchmark columns
            >> Fig 8: 3-panel ROC grid (one panel per pair, 3 curves each)
     4.3.2  Statistical significance of the ranking            <- NEW, Enhancement A
            >> Table 7: AUC bootstrap CIs, PT test, Holm-adjusted p-values
            >> finding: no combination survives correction
     4.3.3  Why thresholded metrics mislead here               <- reframed
            >> Table 8: ONE 9-row confusion-matrix table (TN/FP/FN/TP, pred+%, actual+%)
            >> Table 9: probability dispersion + MCC at 0.5 vs MCC at best threshold
            >> finding: the 0.5 threshold fails, not the models
     4.3.4  Cross-pair synthesis
            >> Table 10: benchmark comparison; no monotonic complexity premium

4.4  Economic significance                                    (~3 pp)  <- NEW, Enhancement B
     >> Table 11: L/S rotation strategy, gross and net of 15/30/50 bp
     >> Table 12: turnover and signal-flip counts
     >> Fig: cumulative net strategy equity curves vs always-long
     >> Table 13: deflated-Sharpe selection benchmark
     >> finding: statistically weak signals are economically material below ~40 bp

4.5  Feature importance                                       (~2.5 pp)
     >> Table 14: ONE 9-column top-5 table (pairs x models), or a 3-panel figure grid
     >> explicitly scoped as fitted-model behaviour, not economic causality
     >> 4.5.4 limits of the evidence

4.6  Validation audit: the cost of look-ahead bias            (~1.5 pp)
     >> Table 15: leakage channel -> treatment -> effect
     >> headline: best AUC 0.768 -> 0.598
     >> this is a contribution, presented as one

4.7  Chapter conclusion                                       (~1 pp)
```

### 4.3 Why this is more efficient

| | Current | Proposed |
|---|---:|---:|
| Tables in 4.3 | 9 (Tables 6–16) | 5 |
| Figures in 4.3 | 6 | 1 (3-panel grid) |
| Pages to first statement of the main finding | ~9 | ~2 |
| Pair-specific prose | 3 × ~500 words | 3 × ~150 words inside 4.3.1 |

Consolidating the per-pair tables into master tables is the key move. A 9-row table with pairs as
row groups shows the ranking instability **visually, in one glance** — which is the finding. Three
separate 3-column tables actively hide it.

Keep the per-pair *probability* figures only if space allows; they are the least informative
exhibits in the current draft because, as Part 3 Enhancement C shows, several probability series
are nearly flat.

---

## Part 5 — Enhancements, ranked

All numbers below are computed from the final corrected outputs and are ready to drop in.

### Enhancement A — Statistical inference on the model ranking  ★ highest value, lowest cost

The chapter currently compares point estimates across models that are all near chance. Without
confidence intervals, a reader cannot tell whether "XGBoost 0.566 vs logistic 0.483" is a finding
or noise. It is noise.

**Block-bootstrapped 95% CIs for ROC-AUC** (5,000 resamples, block length 12 to respect the
label-overlap horizon):

| Pair | Model | AUC | 95% CI | P(AUC ≤ 0.5) |
|---|---|---:|---|---:|
| Mom v Value | Logistic | 0.483 | [0.325, 0.649] | 0.577 |
| Mom v Value | GAM | 0.549 | [0.430, 0.665] | 0.232 |
| Mom v Value | XGBoost | 0.566 | [0.406, 0.711] | 0.211 |
| Mom v Quality | Logistic | 0.598 | [0.494, 0.691] | **0.031** |
| Mom v Quality | GAM | 0.558 | [0.426, 0.693] | 0.184 |
| Mom v Quality | XGBoost | 0.581 | [0.480, 0.683] | 0.059 |
| Value v Quality | Logistic | 0.531 | [0.426, 0.643] | 0.293 |
| Value v Quality | GAM | 0.505 | [0.371, 0.651] | 0.469 |
| Value v Quality | XGBoost | 0.562 | [0.449, 0.665] | 0.127 |

**All nine confidence intervals contain 0.50.** Only momentum–quality logistic has a one-sided
bootstrap p-value below 0.05, and it is the *simplest* model.

**Pesaran–Timmermann test** of directional accuracy against the independence benchmark:

| Pair | Model | Accuracy | P\* | PT stat | p |
|---|---|---:|---:|---:|---:|
| Mom v Value | XGBoost | 0.609 | 0.515 | 2.51 | **0.006** |
| Mom v Quality | Logistic | 0.590 | 0.517 | 2.15 | **0.016** |
| Mom v Quality | XGBoost | 0.558 | 0.500 | 1.45 | 0.074 |
| Mom v Value | GAM | 0.564 | 0.517 | 1.29 | 0.099 |
| Value v Quality | GAM | 0.538 | 0.507 | 0.83 | 0.204 |
| Mom v Quality | GAM | 0.481 | 0.486 | −0.15 | 0.558 |
| Value v Quality | Logistic | 0.474 | 0.478 | −0.19 | 0.574 |
| Value v Quality | XGBoost | 0.455 | 0.478 | −1.12 | 0.869 |
| Mom v Value | Logistic | 0.449 | 0.528 | −2.57 | 0.995 |

Two of nine are significant at 5% uncorrected. With nine hypotheses the **Holm** threshold for the
smallest p-value is 0.05/9 = 0.0056, and the smallest observed p is 0.006. **Nothing survives
multiple-testing correction.**

This is the single most defensible addition to the thesis. It converts "our numbers are weak"
into "we tested whether the numbers are distinguishable from chance, with an appropriate
correction, and they are not" — which is a methodological strength, not a weakness.

### Enhancement B — Economic significance  ★ highest value overall

Chapter 3 ¶411 explicitly defers this ("falls outside the scope of this study") and Chapter 2 ¶244
identifies it as a major gap in the literature. The data to close it is already in the repo:
weekly `factor_returns.csv` plus 156 weeks of saved signals. **This is the biggest available
upgrade to the thesis and costs one afternoon.**

Long/short pairwise rotation: go long the first-named factor and short the second when the model
predicts the positive class, reverse otherwise. Position at *t*, return realised *t*→*t*+1.

**Net annualised Sharpe at three cost levels:**

| Pair | Model | 0 bp | 15 bp | 30 bp | Ann. 2-way turnover |
|---|---|---:|---:|---:|---:|
| Mom v Quality | XGBoost | 1.26 | 1.06 | **0.86** | 29.3× |
| Mom v Quality | Logistic | 1.23 | 1.04 | **0.84** | 28.0× |
| Mom v Value | GAM | 0.81 | 0.68 | 0.54 | 8.7× |
| Mom v Value | XGBoost | 0.69 | 0.67 | **0.66** | **0.7×** |
| Value v Quality | GAM | 0.72 | 0.57 | 0.42 | 22.7× |
| Value v Quality | Logistic | 0.50 | 0.42 | 0.35 | 12.0× |
| Mom v Quality | GAM | 0.68 | 0.55 | 0.41 | 20.0× |
| Value v Quality | XGBoost | 0.28 | 0.21 | 0.14 | 10.7× |
| Mom v Value | Logistic | −0.16 | −0.26 | −0.37 | 6.7× |

**The central result, and the most interesting thing in the corrected study:** momentum–quality
delivers ~18.7% net annualised return at 30 bp with Sharpe 0.86, from a model whose ROC-AUC is
0.581 and whose AUC confidence interval contains 0.5. A small directional edge applied to a
volatile spread (22% annualised) with a stable signal is economically material even when it is
statistically indistinguishable from chance at this sample size.

That is a genuinely publishable observation and it reconciles the whole chapter: it explains why
the weak statistical results are still worth reporting, and it is exactly the accuracy-vs-economic-
value distinction Chapter 2 ¶240–246 sets up.

**Report the cost sensitivity honestly** — momentum–quality flips 44 times in 156 weeks:

| Cost (bp) | 0 | 20 | 30 | 40 | **50** | 60 | 75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Sharpe | 1.23 | 0.97 | 0.84 | 0.71 | **0.58** | 0.45 | 0.26 |
| XGBoost Sharpe | 1.26 | 0.99 | 0.86 | 0.72 | **0.59** | 0.45 | 0.25 |

**Deflated-Sharpe selection benchmark** (Bailey & López de Prado): with N = 9 strategies tried,
the expected maximum Sharpe under *no* skill is **0.58**. The observed maximum net of 30 bp is
0.86, which exceeds it. But at 50 bp the observed maximum is 0.59 ≈ 0.58 and the result
**dissolves exactly at the selection benchmark.**

State this break-even explicitly. The honest conclusion — "economically material below roughly
40 bp round-trip, indistinguishable from selection bias at 50 bp and above" — is far stronger
than either an unqualified Sharpe or silence, and it is precisely the kind of claim an examiner
rewards.

Caveat to state: the same 156-week block was used throughout model development, so this is a
development-set Sharpe. Also note that ~29× annual turnover on a long/short SA factor spread
makes the 30 bp assumption optimistic; justify the cost level chosen against SA equity market
evidence.

### Enhancement C — Diagnose the one-sided-prediction problem properly  ★ fixes a real misreading

The current draft describes logistic and XGBoost on value–quality as "near-always-positive
classifiers" and reads this as aggressive prediction. **That diagnosis is wrong, and the correct
one is more interesting.** The probabilities are nearly *constant*:

| Pair | Model | Prob. range | Prob. SD | MCC @ 0.5 | Best threshold | MCC @ best | Brier skill |
|---|---|---|---:|---:|---:|---:|---:|
| Value v Quality | Logistic | 0.44–0.58 | **0.021** | −0.015 | 0.54 | 0.124 | −0.013 |
| Value v Quality | XGBoost | 0.49–0.58 | **0.023** | −0.089 | 0.57 | **0.256** | −0.005 |
| Mom v Quality | Logistic | 0.34–0.71 | 0.049 | 0.172 | 0.47 | 0.233 | **+0.028** |
| Mom v Value | XGBoost | 0.13–0.60 | 0.131 | 0.200 | 0.38 | 0.243 | **+0.005** |
| Mom v Quality | XGBoost | 0.14–0.89 | 0.165 | 0.116 | 0.64 | 0.178 | −0.024 |
| Mom v Quality | GAM | 0.00–0.83 | 0.220 | −0.012 | 0.30 | 0.100 | −0.261 |
| Mom v Value | GAM | 0.03–1.00 | 0.306 | 0.103 | 0.20 | 0.182 | −0.314 |
| Value v Quality | GAM | 0.00–1.00 | 0.322 | 0.066 | 0.20 | 0.098 | −0.432 |
| Mom v Value | Logistic | 0.17–0.95 | 0.140 | −0.205 | 0.60 | 0.164 | −0.083 |

Three findings the current draft misses entirely:

1. **The heavily regularised models barely move off the base rate.** Value–quality XGBoost spans
   0.49–0.58 (SD 0.023) — consistent with its `reg_lambda = 1000`. It is not predicting "value
   always wins"; it is predicting "roughly 53%, always", and the 0.5 threshold then assigns
   almost everything to the positive class. **The threshold is the failure, not the model.**
2. **Threshold tuning rescues MCC in every single case.** Value–quality XGBoost goes from −0.089
   to **+0.256**. This should be presented as a threshold-sensitivity exhibit, with the honest
   caveat that the threshold must be selected on training data — which is exactly the future-work
   recommendation to make.
3. **The GAM is badly miscalibrated, and that is why its log-loss is poor.** Brier skill scores of
   −0.26 to −0.43 with probabilities saturating at 0.00 and 1.00 explain the 1.32–1.55 log-losses
   mechanically. Only two of nine models beat the base rate on Brier skill, and both are the
   *least* flexible specifications available for their pair.

### Enhancement D — Reframe the leakage audit as a contribution  ★ free

Already drafted as §4.5.3, but presented apologetically. Promote it to its own section (4.6 in the
structure above) and lead with the number: **best ROC-AUC 0.768 → 0.598 from removing five
leakage channels.** Few emerging-market factor-timing papers quantify this on their own pipeline.
Chapter 2 ¶242 and ¶245 already argue the field needs exactly this. Claim it.

### Enhancement E — Cheap robustness checks

- **Model agreement.** Pairwise agreement rates and McNemar tests between the three models on each
  pair. If models disagree substantially while all sitting near chance, that supports the
  "different representations of noise" reading over "different signals".
- **Simple ensemble.** Chapter 2 ¶233 and ¶243 both recommend combination methods, and the thesis
  never tests one. Averaging the three probability series per pair is ~10 lines of code and
  directly answers a question the lit review raises. Chapter 5 already lists it as future work —
  it is cheap enough to move into the results.
- **Subperiod stability.** Split the 156 weeks into three 52-week blocks and report accuracy and
  AUC per block. With three retuning events this aligns naturally with the parameter regimes and
  substitutes for the unreproducible Tables 20–22.

### Enhancement F — Do not attempt before submission

- Regenerating weekly active-space descriptives (benchmark is monthly; see Part 0.1 — unnecessary
  for 4.3–4.5 in any case).
- Rerunning with k ≥ 2 held-out groups to justify "CPCV". Correct the terminology instead.
- Regenerating local SHAP. Delete the claim instead.
- Reserving a fresh untouched test block. Not possible retrospectively; disclose the reuse of the
  156-week block as a limitation instead.

---

## Part 6 — Chapter 5 consequences

`Chapter4_Revision_Map.md` already lists these. Two additions arising from the enhancements above:

- **Summary of findings** gains a genuinely positive result to report, provided it is correctly
  qualified: statistically weak signals produced economically material net returns on
  momentum–quality below roughly 40 bp round-trip costs, dissolving at the selection benchmark
  above that. This belongs in the abstract too.
- **Limitations** must add: 156-week test block reused across development (≈70 evaluations);
  retuning history not persisted; threshold selected ex post in the sensitivity analysis; cost
  assumptions not calibrated to SA market-impact evidence.

---

## Part 7 — Suggested order of work

| # | Task | Effort | Value |
|---|---|---|---|
| 1 | Chapter 3 factual corrections (§3.1) — the chapter currently describes a pipeline that does not exist | 0.5 day | Blocking |
| 2 | Enhancement A — bootstrap CIs, PT, Holm (numbers above are final) | 0.5 day | Very high |
| 3 | Enhancement B — economic value (numbers above are final; needs equity-curve figure) | 1 day | Very high |
| 4 | Restructure Chapter 4 into master tables (Part 4) | 1.5 days | High |
| 5 | Enhancement C — threshold and calibration diagnostics | 0.5 day | High |
| 6 | Chapter 1 abstract + ¶161 edits | 0.25 day | Blocking |
| 7 | Chapter 2 re-pointing + new hypothesis paragraph (§Part 2) | 0.5 day | High / cheap |
| 8 | Enhancement D — promote the leakage audit | 0.25 day | High / free |
| 9 | Chapter 5 rewrite | 1 day | Blocking |
| 10 | Enhancement E — ensemble, agreement, subperiods | 0.5 day | Moderate |
| 11 | Section 4.2 monthly active-space descriptives | 0.5 day | Moderate |
