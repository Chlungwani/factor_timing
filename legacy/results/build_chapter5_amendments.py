"""Build the Chapter 5 (Conclusion) amendment document.

Chapter 5 was written for the earlier results, in which predictive performance rose with model flexibility, momentum
versus value was the most predictable pair, value versus quality the least, and XGBoost's probabilities were less
reliable than those of logistic regression. Chapter 4 now reports the opposite on each point: XGBoost leads every
pair on every measure but the GAM does not improve on logistic regression, value versus quality is the most
predictable pair, and XGBoost has the best probabilities. The amendment keeps the author's four subheadings and
updates what they contain:
  - a short chapter opening (aim, design, outline), as in the other chapters;
  - Summary of key findings: four findings that answer the objectives of Section 1.3 from the Chapter 4 results,
    Appendix B.4 and Appendix C.5, each opening with the finding, then the evidence, then its qualification;
  - Implications: rewritten from the new findings (pair-specific timing, complexity matched to evidence, evaluation
    beyond accuracy and ROC-AUC, reporting the sensitivity to validation choices);
  - Limitations: the author's paragraphs kept where they still hold; the feature-selection and sample paragraphs
    updated (Appendix B.4; Section 3.9);
  - Recommendations: portfolio implementation, selection and calibration of probabilities, feature selection
    within the training window, model combination, and a wider scope.
Every number is read from results/model_outputs, and the script stops if a sentence's premise does not hold.
Citations are Word citation fields with the thesis's Source Manager tags; the works cited are listed after the
chapter, with corrected details where the thesis reference list prints them incompletely.

Starts from the current thesis (a snapshot, since Word keeps the original open), keeps only Chapter 5 and applies
the amendments in place, highlighted, with an amendment log at the end.

Usage: python build_chapter5_amendments.py [out_dir] [thesis_snapshot.docx]
"""
import copy
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree

from build_chapter1_amendments import B_NS, MISSING, cite, norm
from build_chapter2_amendments import new_paragraph
from build_chapter3_references import CORRECTED
from build_chapter4_amendments import compose, find_para, fresh_copy, heading, metrics, style, text
from build_minimal_chapter3 import Editor, accept_revisions, add_log, remove, set_heading_start, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUTPUTS = RESULTS / "model_outputs"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = OUT_DIR / "Chapter5_Conclusion_Revised_Highlighted.docx"

MV, MQ, VQ = "Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"
LN2 = 0.6931


def check(m):
    """The premises of the new text, from the model outputs (Table 4, Table 6, Table C.6 and the Appendix C.5 runs)."""
    v = lambda pair, model, col: m.loc[(pair, model), col]
    for pair in (MV, MQ, VQ):
        assert all(v(pair, "XGBoost", "acc") > v(pair, k, "acc") for k in ("Logistic", "GAM"))
        assert all(v(pair, "XGBoost", "auc") > v(pair, k, "auc") for k in ("Logistic", "GAM"))
        assert all(v(pair, "XGBoost", "ll") < v(pair, k, "ll") for k in ("Logistic", "GAM"))
        assert v(pair, "GAM", "ll") == m.xs(pair)["ll"].max() and v(pair, "GAM", "ll") > LN2
    assert [p for p in (MV, MQ, VQ) if v(p, "GAM", "auc") < v(p, "Logistic", "auc")] == [MQ, VQ]
    xgb = m.xs("XGBoost", level="model")
    assert (f"{xgb.auc.min():.3f}", f"{xgb.auc.max():.3f}") == ("0.589", "0.664")
    assert (f"{100 * xgb.acc.min():.1f}", f"{100 * xgb.acc.max():.1f}") == ("60.3", "62.8")
    assert 0.35 < 1 - xgb.acc.max() and 1 - xgb.acc.min() < 0.42                 # "close to two in five" wrong
    below = sorted((p, k) for (p, k), x in m["ll"].items() if x < LN2)
    assert below == [(MQ, "XGBoost"), (VQ, "Logistic"), (VQ, "XGBoost")], below
    assert f"{v(MQ, 'XGBoost', 'auc') - v(MQ, 'Logistic', 'auc'):.3f}" == "0.019"
    assert (v(VQ, "XGBoost", "TP"), v(VQ, "XGBoost", "TP") + v(VQ, "XGBoost", "FN")) == (45, 77)
    assert (v(VQ, "XGBoost", "TN"), v(VQ, "XGBoost", "TN") + v(VQ, "XGBoost", "FP")) == (49, 79)
    assert (v(MQ, "XGBoost", "TP"), v(MQ, "XGBoost", "TN"), f"{100 * v(MQ, 'XGBoost', 'pred_pos'):.1f}") == (82, 14, "89.1")
    assert f"{v(MV, 'XGBoost', 'auc'):.3f}" == "0.625" and f"{v(MV, 'XGBoost', 'll'):.3f}" == "0.715"
    assert f"{v(VQ, 'XGBoost', 'll'):.3f}" == "0.670" and f"{100 * v(MQ, 'XGBoost', 'acc'):.1f}" == "61.5"
    # the GAM performs worst on most measures (most of the 9 pair-measure comparisons)
    worst = sum(v(p, "GAM", c) == (m.xs(p)[c].max() if c == "ll" else m.xs(p)[c].min())
                for p in (MV, MQ, VQ) for c in ("acc", "auc", "ll"))
    assert worst >= 5, worst
    # XGBoost's selected trees have depth two to four at every tuning point (Table C.6)
    depths = {s["max_depth"] for p in (MV, MQ, VQ)
              for s in json.loads((OUTPUTS / "artifacts" / f"xgboost_{p.lower()}_walkforward_params.json").read_text())}
    assert depths == {2, 3, 4}, depths
    # logistic regression, momentum vs value: the weakest penalty at every tuning point (Table C.6)
    lams = {s["lambda_reg"] for s in json.loads(
        (OUTPUTS / "artifacts" / "logistic_momentum_vs_value_walkforward_params.json").read_text())}
    assert lams == {0.01}, lams
    # Appendix C.5: the selected logistic MvV log-loss is the highest of its fixed settings
    fixed = pd.read_csv(OUTPUTS / "sensitivity" / "logistic_momentum_vs_value_fixed.csv")
    assert f"{fixed['log_loss'].max():.3f}" == f"{v(MV, 'Logistic', 'll'):.3f}" == "1.140"


def references(document, editor, master, sources):
    """A list of the works cited in Chapter 5: as the thesis reference list prints them, or, where it prints them
    with incomplete details, corrected (highlighted, from the Chapter 1 and 3 reference lists)."""
    body = document.element.body
    tags = []
    for it in body.iter(qn("w:instrText")):
        if "CITATION" in (it.text or ""):
            tags += [it.text.split("CITATION")[1].split()[0]] + re.findall(r"\\m (\S+)", it.text)
    tags = list(dict.fromkeys(tags))
    corrected = {tag: parts for tag, (parts, _) in CORRECTED.items()}
    corrected["Arn19"] = MISSING["Arn19"]                        # printed without volume and issue
    entries, fixed = [], []
    for tag in tags:
        s = sources[tag]
        last, year = next(s.iter(B_NS + "Last")).text, s.findtext(B_NS + "Year")
        hit = [p for p in master if norm(text(p)).startswith(norm(last) + " ") and f"({year}" in text(p)]
        assert len(hit) == 1, (tag, len(hit))
        if tag not in corrected:
            entries.append(copy.deepcopy(hit[0]))
            continue
        p = fresh_copy(hit[0])
        run = next(r for r in p.iter(qn("w:r")) if r.find(qn("w:t")) is not None)
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        for part in corrected[tag]:
            p.append(editor.new_run(run, part if isinstance(part, str) else part[0], italic=not isinstance(part, str)))
        entries.append(p)
        fixed.append(tag)
    for p in entries:                                  # field characters of the bibliography do not travel
        for r in [r for r in p.iter(qn("w:r")) if r.find(qn("w:fldChar")) is not None or r.find(qn("w:instrText")) is not None]:
            r.getparent().remove(r)
    entries.sort(key=lambda p: norm(text(p)))
    title = document.add_paragraph()
    title.add_run().add_break(WD_BREAK.PAGE)
    run = title.add_run("References cited in Chapter 5")
    run.bold, run.font.size = True, Pt(13)
    intro = document.add_paragraph(
        "Every work cited in the revised Chapter 5, in APA style. All are in the thesis reference list. Unhighlighted "
        "entries are as that list prints them; highlighted entries are printed there with incomplete details and are "
        "given as Word will print them once the Source Manager entry is completed (see the amendment log).")
    intro.runs[0].font.size = Pt(10)
    sect = body.find(qn("w:sectPr"))
    for el in [title._p, intro._p] + entries:
        sect.addprevious(el)
    return tags, fixed


def build():
    m = metrics()
    check(m)

    document = Document(SOURCE)
    body = document.element.body
    accept_revisions(body)
    strip_comments(body)

    # the thesis reference list (Word's bibliography) and the Source Manager, for the reference list
    paras = list(body.iter(qn("w:p")))
    ref_head = [p for p in paras if text(p).strip() == "References"][-1]
    master = []
    for p in paras[paras.index(ref_head) + 1:]:
        if "Heading" in style(p):                       # the appendices follow the reference list
            break
        if text(p).strip():
            master.append(copy.deepcopy(p))
    with zipfile.ZipFile(SOURCE) as z:
        bib = next(etree.fromstring(z.read(n)) for n in z.namelist()
                   if n.startswith("customXml/item") and "Props" not in n and b"Sources" in z.read(n)[:400])
    sources = {s.findtext(B_NS + "Tag"): s for s in bib.findall(B_NS + "Source")}

    # keep only Chapter 5: from its "Chapter" label to its last paragraph (the last recommendation)
    children = list(body)
    start = children.index(heading(body, "Conclusion"))
    first = start - 1 if style(children[start - 1]) == "Chapter" else start
    last = children.index(find_para(body, startswith="Finally, future research could explore cross-market"))
    for i, el in enumerate(children):
        if el.tag != qn("w:sectPr") and not first <= i <= last:
            body.remove(el)
    set_heading_start(document, 5)

    e = Editor(True)
    log = []
    note = lambda section, change, reason: log.append((section, change, reason))
    para = lambda start: find_para(body, startswith=start)

    # ------------------------------------------------------------------ headings --
    # The author's four subheadings (unnumbered Heading 3: numbering is switched off on each) are kept as they are,
    # so the text refers to them by content, never by a section number.
    titles = ("Summary of key findings", "Implications for factor timing and quantitative asset management",
              "Limitations of the study", "Recommendations for future research")
    heads = [heading(body, t) for t in titles]
    assert [list(body).index(h) for h in heads] == sorted(list(body).index(h) for h in heads)

    # ------------------------------------------------------------------ chapter opening --
    summary_lead = para("Several important findings emerge from the empirical analysis")
    opening = new_paragraph(e, summary_lead, [
        "This study examined whether machine learning models of increasing functional flexibility can predict which "
        "factor leads in the momentum versus value, momentum versus quality and value versus quality pairs of the "
        "South African equity market. This chapter summarises the findings, sets out their implications for factor timing and quantitative asset "
        "management, and discusses the limitations of the study and directions for future research."])
    title = heading(body, "Conclusion")
    assert title.getnext() is heads[0], "the chapter already has an opening"
    title.addnext(opening)
    note("Opening", "New chapter opening: the research question and an outline of the four subsections. The four subheadings are unchanged (wording, order and style).", "Consistency with "
         "Chapters 1, 2 and 4, which open by saying what the chapter does.")

    # ------------------------------------------------------------------ summary of key findings --
    compose(e, summary_lead, ["Four findings emerge from the empirical analysis."])
    note("Summary", "'Several important findings emerge from the empirical analysis' becomes 'Four findings emerge from "
         "the empirical analysis'.", "Concise; the findings are now four.")

    p = para("First, the results demonstrate that machine learning models are capable")
    compose(e, p, [
        "First, relative factor leadership in the South African equity market is partly predictable out of sample, "
        "but the predictable component is modest. The strongest model, XGBoost, reaches ROC-AUCs between 0.589 and "
        "0.664 and accuracies between 60.3% and 62.8% across the three pairs (Table 4), so that close to two in five "
        "of its weekly classifications are incorrect. The results therefore fall between the two positions on factor "
        "timing set out in Section 1.1: relative leadership can be predicted, but only to a limited degree."])
    note("Summary (first)", "Rewritten. Old: 'Across all factor pairs, the nonlinear models consistently outperform the "
         "linear benchmark … supports … that factor premia are influenced by complex, nonlinear interactions'. New: "
         "leadership is partly predictable but the predictable component is modest (XGBoost ROC-AUC 0.589–0.664, "
         "accuracy 60.3–62.8%), between the two positions on factor timing in Section 1.1.",
         "Table 4: the GAM, a nonlinear model, does not outperform logistic regression on ROC-AUC for two pairs.")

    p = para("Second, predictive performance improves systematically as model flexibility increases")
    compose(e, p, [
        "Second, greater model flexibility does not improve predictive performance monotonically. XGBoost has the "
        "highest accuracy and ROC-AUC and the lowest log-loss for every pair, but the GAM ranks below logistic "
        "regression on ROC-AUC for two of the three pairs and has the highest log-loss for all three (Table 4). The "
        "gain is therefore associated with boosted trees, which can represent interactions between predictors, rather "
        "than with smooth nonlinear effects of individual predictors. A similar pattern is reported for US stock "
        "returns, where an additive spline model fails to outperform a linear one robustly and the gains of more "
        "flexible models are attributed to interactions between predictors ", cite(("Shi20", "Gu, Kelly, & Xiu, 2020")),
        ". XGBoost's lead in ranking holds across its hyperparameter settings for value versus quality and momentum "
        "versus value, but not for momentum versus quality, where its ROC-AUC exceeds that of logistic regression by "
        "only 0.019 (Appendix C.5). The GAM's weak ranking for value versus quality, by contrast, partly reflects "
        "the setting selected at its first tuning point."])
    note("Summary (second)", "Rewritten. Old: 'predictive performance improves systematically as model flexibility "
         "increases. Logistic regression provides the weakest results … GAM improves predictive performance … XGBoost "
         "delivers the strongest results'. New: flexibility does not help monotonically; the gain comes with boosted "
         "trees, not smooth additive effects (not isolated by the study), as Gu, Kelly and Xiu (2020) report for US "
         "stock returns; the lead holds across XGBoost's settings for two pairs, not for momentum versus quality "
         "(0.019 ahead of logistic regression); the GAM's weak value versus quality ranking partly reflects its first "
         "selection.", "Table 4; Appendix C.5. Gu, Kelly and Xiu (2020) checked in the paper: their additive "
         "spline model 'fails to robustly outperform the linear specification' and the gains come from interactions.")

    p = para("Third, the strength of predictability varies across factor pairs")
    compose(e, p, [
        "Third, predictability differs markedly across the factor pairs. Value versus quality provides the strongest "
        "evidence: XGBoost ranks the weeks best (ROC-AUC 0.664), recognises both leadership states (45 of the 77 "
        "value-led and 49 of the 79 quality-led weeks) and has a log-loss (0.670) below 0.693, the value of a "
        "constant 50% forecast. For momentum versus value, XGBoost is the only model with a ROC-AUC "
        "well above 0.5 (0.625), but its log-loss (0.715) exceeds 0.693. Momentum versus quality provides the "
        "weakest evidence of timing ability: XGBoost's accuracy of 61.5% results from predicting momentum in 89.1% "
        "of weeks, so that it identifies 82 of the 85 momentum-led weeks but only 14 of the 71 quality-led weeks "
        "(Section 4.3)."])
    note("Summary (third)", "Rewritten. Old: momentum versus value 'exhibits the strongest predictive structure across all "
         "models … value versus quality demonstrates the weakest predictive signal'. New: value versus quality is the "
         "strongest (both states recognised, log-loss below 0.693), momentum versus value is ranked only by XGBoost, "
         "and momentum versus quality is the weakest (XGBoost predicts momentum in 89.1% of weeks).",
         "Tables 4 and 6: the order of the pairs is reversed.")

    p = para("Fourth, feature importance analysis reveals distinct economic drivers")
    compose(e, p, [
        "Fourth, the models draw on different information for each pair, and relative volatility recurs across pairs "
        "and models. Relative-volatility measures rank first in all three reference fits for momentum versus value, "
        "and the value–quality volatility ratio ranks among the four leading features of every model for value "
        "versus quality, where the three-month relative return of value against quality leads XGBoost and ranks "
        "second for logistic regression. For momentum versus quality, logistic regression and XGBoost share their two "
        "leading features, the change in the TOP40 price-to-EBITDA ratio and South African real-yield-curve "
        "curvature (Section 4.4). These features correspond to established timing signals, but as the rankings "
        "describe reference fits, they show which information the models use, not why XGBoost predicts better."])
    note("Summary (fourth)", "Rewritten from Section 4.4. Old: momentum versus value 'primarily influenced by … relative "
         "volatility and correlation dynamics', value versus quality 'largely anchored in valuation differentials', "
         "'dispersion-related variables emerge consistently'. New: relative volatility leads every model for momentum "
         "versus value and ranks in the top four of every model for value versus quality; valuation and real-yield "
         "curvature lead two models for momentum versus quality; the three-month value–quality relative return leads "
         "XGBoost for value versus quality. The caveat on reference fits is given in one clause (it is set out in "
         "Sections 3.10 and 4.4).", "Section 4.4: the old drivers no longer match the reference fits.")

    p = para("Fifth, the results highlight the trade-off between predictive accuracy and probability calibration")
    remove(p)
    note("Summary (fifth)", "Deleted: 'Fifth, the results highlight the trade-off between predictive accuracy and "
         "probability calibration. While XGBoost produces the strongest classification and ranking performance, it "
         "also generates less well-calibrated probability forecasts relative to logistic regression …'. No finding on "
         "probability quality replaces it.", "Contradicted by Table 4 (XGBoost has the lowest log-loss for every "
         "pair); author, 4 November: a separate finding on ranking versus probability quality does not add much to "
         "the discussion. The log-loss results remain in the third finding and the third implication.")

    p = para("Finally, the stability of predictive performance across walk-forward windows")
    remove(p)
    note("Summary (finally)", "Deleted: 'the stability of predictive performance across walk-forward windows suggests "
         "that the modelling framework successfully mitigates overfitting … provides a robust validation architecture "
         "capable of generating realistic estimates'.", "Chapter 4 does not report stable performance across the "
         "three walk-forward periods (for momentum versus quality XGBoost predicted momentum in every week until "
         "January 2024, Section 4.2.2), and Appendix C.5 shows that the GAM's results depend on the selection. "
         "What the validation shows is now in the second finding and the last implication.")

    # ------------------------------------------------------------------ implications --
    p = para("First, the results reinforce the notion that factor returns exhibit nonlinear")
    compose(e, p, [
        "First, factor timing models should be assessed and applied pair by pair. The results support the use of "
        "model forecasts as one input to the tilt between value and quality, give weaker support for momentum "
        "versus value and the least support for momentum versus quality. A single rotation model applied uniformly "
        "across styles would therefore combine an informative signal with weak ones. Because the predictable "
        "component is modest, the forecasts are better suited to moderating the tilt between two factors than to "
        "switching the whole allocation between them."])
    p = para("Second, the results highlight the importance of aligning model complexity")
    compose(e, p, [
        "Second, model complexity should be matched to the evidence rather than presumed to add value. The results "
        "do not support a general preference for nonlinear models: the additive nonlinear model performed worst "
        "on most measures, and XGBoost's advantage was obtained with "
        "shallow trees of depth two to four (Table C.6), a constrained form of flexibility that can represent "
        "interactions among only a few predictors. A regularised linear model therefore remains a necessary "
        "benchmark, and a more flexible model is justified when it improves on that benchmark out of sample across "
        "several measures and hyperparameter settings, as XGBoost does for value versus quality. Its leading "
        "features are also economically interpretable, so the added flexibility did not cost the interpretability "
        "that practitioners require (Section 2.2.3)."])
    p = para("Third, the methodological framework employed in this study demonstrates")
    compose(e, p, [
        "Third, forecasts should be evaluated on more than aggregate accuracy and ROC-AUC before they inform "
        "allocations. The momentum versus quality results show that accuracy can be raised by favouring the more "
        "frequent class, and the momentum versus value results show that a model can rank the weeks reasonably well while "
        "its probabilities are less informative than a constant 50% forecast. Allocation rules that scale exposures "
        "by the predicted probability (Section 3.9) require probabilities that improve on such a forecast, a "
        "condition met by only three of the nine fitted models (Table 4)."])
    p = para("Finally, the study provides empirical support for the potential role of machine learning")
    compose(e, p, [
        "Finally, results from financial machine learning should be reported together with their sensitivity to the "
        "validation choices. Purged cross-validation and walk-forward evaluation limit information leakage (Section "
        "3.8), but they do not remove the dependence of the results on the hyperparameters selected. Evaluating "
        "every setting in each model's grid (Appendix C.5) separates the conclusions that hold across settings from "
        "those that rest on a single selection, in line with the call to report all the specifications tried in "
        "machine learning backtests ", cite(("Arn19", "Arnott, Harvey, & Markowitz, 2019")), "."])
    note("Implications", "All four implications rewritten; the opening sentence is kept. Old: (1) factor returns are nonlinear, so "
         "'traditional linear econometric models may … be insufficient'; (2) align model complexity with the economic "
         "structure of the pair; (3) purged cross-validation, walk-forward evaluation and feature reduction 'are "
         "essential'; (4) ML 'may assist portfolio managers in adjusting exposures'. New: (1) assess and apply timing "
         "models pair by pair, as one input to a moderated tilt; (2) match complexity to the evidence: the GAM is "
         "worst on most measures and XGBoost's lead comes with trees of depth 2–4 (Table C.6), so a regularised "
         "linear benchmark stays necessary, and its leading features are interpretable (Section 2.2.3); (3) evaluate "
         "beyond accuracy and ROC-AUC (class-specific errors, log-loss), since probability-weighted allocation "
         "(Section 3.9) needs probabilities better than a 50% forecast, which only three of nine models give; "
         "(4) report the sensitivity of results to validation choices (Appendix C.5; Arnott, Harvey, & Markowitz, "
         "2019).", "The old points (1) and (2) are contradicted by Table 4 (the additive nonlinear model does not "
         "improve on the linear one); point (3) restated the methodology; point (4) is now part of (1).")

    # ------------------------------------------------------------------ limitations --
    p = para("First, the analysis focuses on predictive accuracy rather than full portfolio implementation")
    e.replace(p, "predictive accuracy", "predictive performance")
    note("Limitations (first)", "'predictive accuracy' becomes 'predictive performance'.", "Accuracy is one of the three "
         "measures (Section 3.9); the study evaluates all three.")

    p = para("Furthermore, the study concentrates on three style factors")
    compose(e, p, [
        "Furthermore, the study concentrates on three style factors: momentum, value and quality. Other factors, "
        "such as size, growth and low volatility, which were set aside for the reasons given in Section 3.2, may "
        "exhibit different time-varying behaviour."])
    note("Limitations (second)", "Second paragraph on the factors: 'other factors such as size, low volatility, or "
         "profitability' becomes 'size, growth and low volatility, which were set aside for the reasons given in "
         "Section 3.2'; its last sentence ('Future research could extend the analysis by testing alternative factor "
         "construction methodologies …') moves to the last recommendation.",
         "Section 3.2 names size, growth and low volatility as the factors set aside; profitability is part of the "
         "quality factor (ROE, ROIC). Recommendations belong under the last subheading.")

    p = para("A third limitation relates to the feature selection procedure")
    compose(e, p, [
        "A third limitation relates to the feature selection procedure. The mutual-information clustering retains the "
        "most central feature of each cluster without reference to the target (Section 3.6), so an informative "
        "feature can be excluded. The clustering was also estimated on a sample that includes the evaluation block; "
        "on the pre-evaluation weeks alone, 11 of the 32 representatives change, nine of them to another member of "
        "the same cluster (Appendix B.4). The selected features are therefore one reasonable subset of the "
        "candidates, and the effect of the selection on out-of-sample performance is not tested."])
    note("Limitations (third)", "Feature-selection limitation rewritten: the unsupervised criterion in one sentence (Section 3.6 "
         "describes the method, so its description is not repeated), plus the estimation on a sample that includes "
         "the evaluation block and its effect (11 of 32 representatives change, 9 within the same cluster) from "
         "Appendix B.4.", "Appendix B.4 and B.7; no repetition of Section 3.6.")

    p = para("Forth, the empirical analysis is limited to the South African equity market")
    e.replace(p, "Forth,", "Fourth,")
    note("Limitations (fourth)", "'Forth' becomes 'Fourth'.", "Spelling.")

    p = para("Finally, the modelling framework relies on weekly data and a finite historical sample")
    compose(e, p, [
        "Finally, the evaluation rests on a short sample. The 156 weekly predictions span three years, the labels "
        "persist over several weeks and the hyperparameters are selected three times, so the effective number of "
        "independent observations is smaller than the number of weeks (Section 3.9). The differences between the "
        "models are therefore point estimates, and the results may reflect the market conditions of 2022 to 2025 "
        "rather than a stable relationship."])
    note("Limitations (finally)", "Sample limitation made specific: 156 weekly predictions over three years, persistent labels, "
         "three tuning points; the differences between models are point estimates and may reflect 2022–2025 "
         "conditions. Deleted: 'Although robust validation procedures were applied, the limited sample size inherent "
         "in financial datasets may constrain the stability of highly complex models.'", "Section 3.9 ('small "
         "differences in these point estimates are interpreted with caution'). The sentence on complex models no "
         "longer fits: the most complex model is the most stable across settings (Appendix C.5).")
    note("Limitations", "Unchanged: the opening sentence and the paragraph on the proprietary factor definitions ('The second "
         "limitation relates to …'). Not added as a limitation: the SHAP rankings describe reference fits on the "
         "initial tuning sample, which Sections 3.10 and 4.4 state and the fourth finding recalls.", "Still accurate; "
         "no repetition.")

    # ------------------------------------------------------------------ recommendations --
    p = para("First, future work should extend the analysis to portfolio implementation")
    extend_text = (" The value versus quality forecasts of XGBoost, which recognise both leadership states, are the "
                   "natural starting point, and probability-weighted tilts can be compared with binary switching "
                   "between the two factors.")
    e.append(p, extend_text)
    note("Recommendations (first)", "Sentence added: start from the value versus quality forecasts of XGBoost and compare "
         "probability-weighted tilts with binary switching.", "Links the recommendation to the findings.")

    old_second = para("Second, future research could also explore hybrid feature selection frameworks")
    selection = new_paragraph(e, old_second, [
        "Second, hyperparameter selection could take probability quality into account. ROC-AUC, the selection "
        "criterion, depends only on the order of the predicted probabilities ", cite(("Faw06", "Fawcett, 2006")),
        ", so the selection retained overconfident settings: for momentum versus value it chose the weakest logistic "
        "penalty at every tuning point, whose log-loss of 1.140 is the highest in its grid (Appendix C.5). Selecting "
        "on log-loss, or on ranking and probability measures together, would penalise such settings, and the "
        "probabilities of the selected models could be recalibrated, for example with Platt scaling or isotonic "
        "regression ", cite(("Nic05", "Niculescu-Mizil & Caruana, 2005")),
        ". Averaging the forecasts of several neighbouring settings, rather than relying on a single selected setting, "
        "would also reduce the dependence of the results on the selection."])
    old_second.addprevious(selection)
    note("Recommendations (second)", "New recommendation: select hyperparameters on log-loss or on ranking and probability "
         "measures together, recalibrate the probabilities (Platt scaling or isotonic regression; Niculescu-Mizil & "
         "Caruana, 2005), and average neighbouring settings. Its first sentence gives the reason: "
         "selection on ROC-AUC, which depends only on the order of the probabilities "
         "(Fawcett, 2006), kept the weakest logistic penalty for momentum versus value (log-loss 1.140, the highest in "
         "its grid).", "Appendix C.5: selection on ROC-AUC allowed overconfident settings, and the GAM's results "
         "depend on its selection (second finding).")
    compose(e, old_second, [
        "Third, the feature selection could be estimated within each training window and combined with supervised "
        "criteria. Re-estimating the clustering on the data available at each tuning point would remove "
        "evaluation-period information from the selection, and combining information-based clustering with a "
        "supervised measure of each feature's relevance to the target may retain predictors that the unsupervised "
        "procedure excludes."])
    note("Recommendations (third)", "Feature-selection recommendation (was second) rewritten: estimate the clustering within each "
         "training window, and combine it with a supervised relevance measure. The restatement of the method ('While "
         "this study employs information-theoretic clustering …') is deleted.", "Third limitation (Appendix B.4); "
         "no repetition of Section 3.6.")

    p = para("Thirdly, the application of ensemble frameworks")
    compose(e, p, [
        "Fourth, combining the forecasts of the three model classes may improve robustness. For momentum versus "
        "quality the ROC curves of logistic regression and XGBoost intersect (Section 4.2.2), so neither model "
        "dominates across thresholds, and a combination of models whose errors differ can outperform its members ",
        cite(("Die00", "Dietterich, 2000")), ". Stacked generalisation (Section 2.1.3) provides one framework for "
        "such a combination."])
    note("Recommendations (fourth)", "Ensemble recommendation (was 'Thirdly') tied to the result that motivates it: the crossing "
         "ROC curves for momentum versus quality (Section 4.2.2); combinations of models with different errors can "
         "outperform their members (Dietterich, 2000); stacking refers back to Section 2.1.3.", "Links the "
         "recommendation to the findings; 'Thirdly' to 'Fourth' for consistent ordinals.")

    p = para("Finally, future research could explore cross-market generalisation")
    compose(e, p, [
        "Finally, the analysis could be extended in scope. A longer evaluation period would allow the differences "
        "between the models to be tested formally; other factors and alternative factor definitions would show "
        "whether the timing signals are robust to the construction of the factors; and "
        "applying the same framework to other emerging markets would show how market structure influences the "
        "effectiveness of machine learning in factor timing."])
    note("Recommendations (finally)", "Last recommendation widened: a longer evaluation period with formal tests of the "
         "differences between models (fifth limitation); other factors and factor definitions (moved here from the "
         "limitations); other emerging markets (the author's original point).", "Each limitation now has a matching "
         "recommendation. The formal tests are named as an extension only, since the study does not compare the "
         "models statistically; delete the clause if it should not appear.")

    # ------------------------------------------------------------------ other chapters and references --
    note("Other chapters", "Chapter 1 outline (Research assignment outline) describes Chapter 5 as 'Conclusions and "
         "Recommendations' and says it proposes 'portfolio implementation considerations and regime-sensitive model "
         "extensions'. The chapter is titled 'Conclusion' and does not propose regime-sensitive extensions (neither "
         "did the old version). Either update that sentence in Chapter 1 or ask for a regime-sensitive "
         "recommendation here.", "Consistency between Chapters 1 and 5.")
    tags, fixed = references(document, e, master, sources)
    assert set(tags) == {"Shi20", "Faw06", "Arn19", "Nic05", "Die00"}, tags
    note("References", "Chapter 5 now cites five works, all with sources in the thesis Source Manager and entries in "
         "its reference list: Gu, Kelly and Xiu (2020, Shi20), Fawcett (2006, Faw06), Arnott, Harvey and Markowitz "
         "(2019, Arn19), Niculescu-Mizil and Caruana (2005, Nic05) and Dietterich (2000, Die00). The reference list "
         "after the chapter gives corrected entries (highlighted) for " + ", ".join(sorted(fixed)) + ". Source "
         "Manager changes: Shi20, journal name 'The Review of Financial Studies' with Volume 33 and Issue 5 in their "
         "own fields; Faw06, journal name without the trailing comma, Volume 27, Issue 8; Arn19, Volume 1, Issue 1.",
         "Reference list (the same corrections as in the Chapter 1 and 3 reference lists).")

    add_log(document, log)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
