"""Build the Chapter 1 (Introduction) amendment document: separate the chapter introduction from the background.

The chapter opening had grown to eight paragraphs that mixed three jobs: background (factor premia, timing
evidence, machine learning, South Africa), the study's aim and an overview of the method. Most of it was
repeated in Section 1.1 (Background) and Section 1.4 (Research methodology), and the research gap was stated
three times. The amendment (author, 2 November 2026):
  - the chapter opening becomes a short orientation: the topic, the aim and an outline of Sections 1.1-1.6;
  - Section 1.1 takes the background material, merged, in one funnel (factor premia, their variation over
    time and the case for timing, the mixed evidence, machine learning, emerging markets and South Africa),
    ending with the gap;
  - Section 1.4 takes the method overview (the classification formulation, the three model classes);
  - repeats are removed and small errors fixed.
The author's sentences are kept wherever the content only moves. Citations are Word citation fields with the
author's Source Manager tags; the few that have no source entry stay typed and are listed in the log.

Starts from the current thesis (a snapshot, since Word keeps the original open), keeps only Chapter 1 and
applies the amendments in place, highlighted, with an amendment log at the end.

Usage: python build_chapter1_amendments.py [out_dir] [thesis_snapshot.docx]
"""
import copy
import re
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree

from build_chapter4_amendments import compose, find_para, fresh_copy, heading, style, text
from build_minimal_chapter3 import Editor, accept_revisions, add_log, set_heading_start, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = OUT_DIR / "Chapter1_Introduction_Revised_Highlighted.docx"


def cite(*pairs):
    """A citation field: Source Manager tags and the APA text Word shows for them."""
    return [tag for tag, _ in pairs], "(" + "; ".join(shown for _, shown in pairs) + ")"


FF92, FF93, FF15 = ("Fam92", "Fama & French, 1992"), ("Fam93", "Fama & French, 1993"), ("Fam15", "Fama & French, 2015")

# Works cited in Chapter 1 that the thesis reference list does not contain, in APA 7 (italic parts as tuples).
# Details checked against the publishers' records or the author's literature-review reference file
# (2 November 2026). The first group is in the Source Manager, with incomplete details; the second has no source.
MISSING = {
    "Bar15": ["Barroso, P., & Santa-Clara, P. (2015). Momentum has its moments. ",
              ("Journal of Financial Economics, 116",), "(1), 111–120."],
    "Had20": ["Haddad, V., Kozak, S., & Santosh, S. (2020). Factor timing. ", ("Review of Financial Studies, 33",),
              "(5), 1980–2018."],
    "Asn17": ["Asness, C. S., Chandra, S., Ilmanen, A., & Israel, R. (2017). Contrarian factor timing is deceptively "
              "difficult. ", ("Journal of Portfolio Management, 43",), "(5), 72–87."],
    "Arn19": ["Arnott, R., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the era of machine "
              "learning. ", ("Journal of Financial Data Science, 1",), "(1), 64–74."],
    "Gri22": ["Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep "
              "learning on typical tabular data? ", ("Advances in Neural Information Processing Systems, 35",),
              ", 507–520."],
    "Han23": ["Hanauer, M. X., & Kalsbach, T. (2023). Machine learning and the cross-section of emerging market stock "
              "returns. ", ("Emerging Markets Review, 55",), ", Article 101022."],
    "Mut07": ["Mutooni, R., & Muller, C. (2007). Equity style timing. ", ("Investment Analysts Journal, 36",),
              "(65), 15–24. https://doi.org/10.1080/10293523.2007.11082485"],
}
TYPED = {
    "Fraser & Page, 2000": ["Fraser, E., & Page, M. (2000). Value and momentum strategies: Evidence from the "
                            "Johannesburg Stock Exchange. ", ("Investment Analysts Journal, 29",),
                            "(51), 25–35. https://doi.org/10.1080/10293523.2000.11082404"],
    "Muller & Ward, 2013": ["Muller, C., & Ward, M. (2013). Style-based effects on the Johannesburg Stock Exchange: "
                            "A graphical time-series approach. ", ("Investment Analysts Journal, 42",),
                            "(77), 1–16. https://doi.org/10.1080/10293523.2013.11082552"],
    "Page et al. (2022)": ["Page, D., McClelland, D., & Auret, C. (2022). Style rotation on the JSE. ",
                           ("Finance Research Letters, 46",), "(Part B). [Article number to be confirmed on "
                                                             "ScienceDirect.]"],
}
B_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/bibliography}"


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def reference_list(document, editor, master, sources):
    """A page listing every work cited in Chapter 1: as the thesis reference list prints it, or, for the works
    missing from that list, in full (highlighted)."""
    body = document.element.body
    tags = []
    for it in body.iter(qn("w:instrText")):
        if "CITATION" in (it.text or ""):
            tags += [it.text.split("CITATION")[1].split()[0]] + re.findall(r"\\m (\S+)", it.text)
    tags = list(dict.fromkeys(tags))
    chapter = text(body)
    assert all(k in chapter for k in TYPED), [k for k in TYPED if k not in chapter]
    entries, missing = [], []
    for tag in tags:
        s = sources[tag]
        last = next(s.iter(B_NS + "Last")).text
        year, title = s.findtext(B_NS + "Year"), " ".join(norm(s.findtext(B_NS + "Title")).split()[:4])
        hit = [p for p in master if norm(text(p)).startswith(norm(last)) and f"({year}" in text(p) and title in norm(text(p))]
        if hit:
            entries.append(copy.deepcopy(hit[0]))
        else:
            assert tag in MISSING, f"{tag} ({last}, {year}) is cited but neither in the reference list nor in MISSING"
            missing.append(MISSING[tag])
    missing += list(TYPED.values())
    template = entries[0]
    for parts in missing:
        p = fresh_copy(template)
        run = next(r for r in p.iter(qn("w:r")) if r.find(qn("w:t")) is not None)
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        for part in parts:
            p.append(editor.new_run(run, part if isinstance(part, str) else part[0], italic=not isinstance(part, str)))
        entries.append(p)
    for p in entries:                                  # field characters of the bibliography do not travel
        for r in [r for r in p.iter(qn("w:r")) if r.find(qn("w:fldChar")) is not None or r.find(qn("w:instrText")) is not None]:
            r.getparent().remove(r)
    entries.sort(key=lambda p: norm(text(p)))
    title = document.add_paragraph()
    title.add_run().add_break(WD_BREAK.PAGE)
    run = title.add_run("References cited in Chapter 1")
    run.bold, run.font.size = True, Pt(13)
    intro = document.add_paragraph(
        "Every work cited in the revised Chapter 1, in APA style. Unhighlighted entries are as the thesis reference "
        "list prints them. Highlighted entries are cited in Chapter 1 but missing from the thesis reference list, and "
        "are given here in full; the amendment log lists the Source Manager changes that bring them into it.")
    intro.runs[0].font.size = Pt(10)
    sect = document.element.body.find(qn("w:sectPr"))
    for el in [title._p, intro._p] + entries:
        sect.addprevious(el)
    return len(tags) + len(TYPED), len(missing)


def build():
    document = Document(SOURCE)
    body = document.element.body
    accept_revisions(body)
    strip_comments(body)

    # the thesis reference list (Word's bibliography) and the Source Manager, for the reference page
    paras = list(body.iter(qn("w:p")))
    ref_head = [p for p in paras if text(p).strip() == "References" and style(p).startswith("Heading")][-1]
    master = []
    for p in paras[paras.index(ref_head) + 1:]:
        if style(p).startswith("Heading"):
            break
        if text(p).strip():
            master.append(copy.deepcopy(p))
    with zipfile.ZipFile(SOURCE) as z:
        bib = next(etree.fromstring(z.read(n)) for n in z.namelist()
                   if n.startswith("customXml/item") and "Props" not in n and b"Sources" in z.read(n)[:400])
    sources = {s.findtext(B_NS + "Tag"): s for s in bib.findall(B_NS + "Source")}

    # keep only Chapter 1: from its "Chapter" label to the element before the next chapter label
    children = list(body)
    start = children.index(heading(body, "Introduction"))
    end = next(i for i, el in enumerate(children) if i > start and el.tag == qn("w:p") and style(el) == "Heading1")
    first = start - 1 if style(children[start - 1]) == "Chapter" else start
    last = end - 2 if style(children[end - 1]) == "Chapter" else end - 1
    for i, el in enumerate(children):
        if el.tag != qn("w:sectPr") and not first <= i <= last:
            body.remove(el)
    set_heading_start(document, 1)

    e = Editor(True)
    log = []
    note = lambda section, change, reason: log.append((section, change, reason))
    para = lambda start: find_para(body, startswith=start)

    opening = [para(s) for s in (
        "Equity research identifies systematic differences", "The evidence supporting factor timing is mixed",
        "Classical econometric approaches have been a source", "These issues are especially relevant in emerging",
        "The study investigates the relative performance", "Three model classes are compared",
        "The three model classes constitute a systematic progression", "The combination of weak signals")]

    # Chapter opening: what the study does and why, in brief (the detail is in Section 1.4), then the outline
    aim, approach, outline = opening[0], fresh_copy(opening[0]), fresh_copy(opening[0])
    lop = (("Lop18", "López de Prado, 2018"), ("Lop20", "López de Prado, 2020"))
    compose(e, aim, [
        "Equity factors such as momentum, value and quality earn well-documented long-term premia ",
        cite(FF93, ("Jeg93", "Jegadeesh & Titman, 1993"), ("Qua19", "Asness, Frazzini, & Pedersen, 2019")),
        ", but their returns vary materially over time, with extended periods of strong performance interrupted by "
        "deep drawdowns and, in the case of momentum, occasional crashes ",
        cite(("Dan16", "Daniel & Moskowitz, 2016"), ("Ilm21", "Ilmanen, Israel, Lee, Moskowitz, & Thapar, 2021")),
        ". If changes in relative factor performance are partly predictable, investors could tilt their allocation "
        "towards the factor expected to outperform ",
        cite(("Asn00", "Asness, Friedman, Krail, & Liew, 2000"), ("Gup19", "Gupta & Kelly, 2019"),
             ("Arn23", "Arnott, Kalesnik, & Linnainmaa, 2023")),
        ". This study examines whether machine learning models can predict which of the momentum, value and "
        "quality factors will lead in the South African equity market, and whether more flexible models predict "
        "better than a regularised linear model, a question raised by evidence that machine learning improves "
        "return prediction ", cite(("Shi20", "Gu, Kelly, & Xiu, 2020")),
        " but that weak financial signals leave flexible models prone to fitting noise ",
        cite(("Isr20", "Israel, Kelly, & Moskowitz, 2020")), "."])
    # the gap in brief, and how the study proposes to close it (a summary of Sections 1.1 and 1.4)
    compose(e, approach, [
        "Recent South African studies have begun to apply machine learning to style rotation ",
        cite(("Mak23", "Makamo, 2023"), ("Pag241", "Page, McClelland, & Auret, 2024")),
        ", building on earlier local evidence on equity style timing ", cite(("Mut07", "Mutooni & Muller, 2007")),
        ", but limited evidence isolates the incremental contribution of nonlinearity and interaction effects "
        "under a common validation framework: linear and nonlinear models have not been compared on identical "
        "targets, predictors and evaluation periods. This study proposes such a comparison. Factor timing is framed "
        "as a classification problem for three factor pairs, momentum versus value, momentum versus quality and "
        "value versus quality, with trend-scanning labels ", cite(*lop),
        " identifying which factor of each pair leads over the following weeks. The labels are predicted from "
        "lagged market, macroeconomic, valuation, sentiment and factor-specific variables, with redundant "
        "candidates removed by mutual-information clustering ",
        cite(lop[0], ("Sul16", "Sulaiman & Labadin, 2016")),
        ", using three model classes of increasing flexibility: logistic regression as the linear benchmark ",
        cite(("Hos13", "Hosmer, Lemeshow, & Sturdivant, 2013")),
        ", a generalised additive model (GAM) that allows smooth nonlinear effects ",
        cite(("Has86", "Hastie & Tibshirani, 1986")),
        ", and XGBoost, which also captures interactions between predictors ",
        cite(("Che16", "Chen & Guestrin, 2016")),
        ". To guard against overfitting, hyperparameters are selected by combinatorial purged cross-validation ",
        cite(*lop), ", and the models are evaluated on three years of weekly predictions from an expanding "
        "walk-forward procedure using accuracy, ROC-AUC and log-loss. Shapley values ",
        cite(("Sha532", "Shapley, 1953"), ("Lun17", "Lundberg & Lee, 2017")),
        " then show which predictors each model relies on."])
    compose(e, outline, [
        "Section 1.1 provides the background to the study, Section 1.2 states the research problem, Section 1.3 "
        "defines the scope and objectives, Section 1.4 sets out the research methodology in more detail, Section "
        "1.5 describes the contribution to data science and Section 1.6 outlines the structure of the assignment."])
    aim.addnext(approach)
    approach.addnext(outline)
    for p in opening[1:]:
        p.getparent().remove(p)
    note("1 (opening)", "Rewritten as three paragraphs: the problem and the aim; a summary of the research gap and of "
         "how the study proposes to close it (the classification formulation with trend-scanning labels, the "
         "predictors and their reduction, the three model classes, CPCV tuning, expanding walk-forward evaluation "
         "and the measures, the Shapley analysis), with the detail left to Sections 1.1 and 1.4; and an outline of "
         "Sections 1.1–1.6. Each claim is cited from the thesis's sources, the methods with the sources Section 1.4 "
         "cites; Mutooni and Muller (2007) is now a citation field, so it reaches the reference list. The background material of the old opening is moved to Section 1.1 and the "
         "classification formulation and model descriptions to Section 1.4. Deleted as repeats: 'The three model "
         "classes constitute a systematic progression … across the three factor pairs' (Sections 1.2 and 1.3, "
         "objective 3) and 'The study therefore separates hyperparameter selection from final out-of-sample "
         "evaluation and preserves the chronological ordering of the data' (Section 1.4).", "Author: the opening "
         "explains what the study does, and the background gives the context. Limit repetition.")

    # 1.1 Background: one funnel from factor premia to the gap
    b1, b2, b6 = (para(s) for s in ("The empirical study of equity factors developed", "Empirically, such factors do",
                                    "The South African setting provides"))
    compose(e, b1, [
        "The empirical study of equity factors developed from the recognition that market beta alone does not "
        "explain the cross-section of expected returns. The Capital Asset Pricing Model (CAPM) provided the "
        "starting point ", cite(("Lin65", "Lintner, 1965"), ("Sha64", "Sharpe, 1964")),
        ", and multi-factor models subsequently identified systematic differences in expected returns associated "
        "with company characteristics known as factors ", cite(FF92, FF93), ". Momentum ",
        cite(("Jeg93", "Jegadeesh & Titman, 1993"), ("Car97", "Carhart, 1997")), ", value ", cite(FF93),
        ", quality ", cite(("Qua19", "Asness, Frazzini, & Pedersen, 2019"), ("Pio00", "Piotroski, 2000")),
        ", profitability ", cite(FF15), " and low volatility ",
        cite(("Bli14", "Blitz, Falkenstein, & van Vliet, 2014"), ("Bli07", "Blitz, 2007")),
        " are widely documented sources of long-term return, with the momentum and value premia documented across "
        "markets and asset classes ", cite(("Asn13", "Asness, Moskowitz, & Pedersen, 2013")),
        ". The economic origin of these premia remains debated, straddling rational risk-based ", cite(FF93, FF15),
        " and behavioural bias-based explanations ",
        cite(("Lak94", "Lakonishok, Shleifer, & Vishny, 1994"), ("Dan98", "Daniel, Hirshleifer, & Subrahmanyam, 1998")),
        ", suggesting that their drivers are multifaceted and not attributable to a single, unified theory."])
    compose(e, b2, [
        "These premia do not offer consistent rewards. Factor returns exhibit significant cyclicality and regime "
        "dependence, with prolonged periods of outperformance, deep drawdowns and occasional crashes that challenge "
        "static, buy-and-hold factor investing ",
        cite(("Ilm12", "Ilmanen & Kizer, 2012"), ("Bar15", "Barroso & Santa-Clara, 2015"),
             ("Dan16", "Daniel & Moskowitz, 2016"), ("Bes21", "Bessler, 2021")),
        ". This time variation creates the empirical motivation for factor timing: if changes in relative factor "
        "performance are partly predictable, investors may be able to allocate capital dynamically towards the "
        "factors expected to outperform ",
        cite(("Arn16", "Arnott, Beck, Kalesnik, & West, 2016"), ("Arn23", "Arnott, Kalesnik, & Linnainmaa, 2023")),
        "."])
    b3, b4, b5 = fresh_copy(b2), fresh_copy(b2), fresh_copy(b2)
    compose(e, b3, [
        "The evidence supporting factor timing is mixed. Several studies report that dynamic rules based on "
        "valuation, momentum or common factor signals improve risk-adjusted returns ",
        cite(("Asn00", "Asness, Friedman, Krail, & Liew, 2000"), ("Gup19", "Gupta & Kelly, 2019"),
             ("Had20", "Haddad, Kozak, & Santosh, 2020")),
        ", while others find that most factor timing relationships are weak, unstable or difficult to exploit "
        "consistently out of sample, and that valuation-based timing adds little beyond a static factor allocation ",
        cite(("Asn17", "Asness, Chandra, Ilmanen, & Israel, 2017"),
             ("Ilm21", "Ilmanen, Israel, Lee, Moskowitz, & Thapar, 2021")),
        ". The relevant empirical question is therefore whether a specified forecasting method produces reliable "
        "out-of-sample improvement, rather than whether factor timing is theoretically possible."])
    compose(e, b4, [
        "Classical econometric approaches have been a source of valuable insights in generating factor timing "
        "signals, but their reliance on linearity assumptions limits their ability to capture nonlinear interactions "
        "and complex market dynamics that may underpin short-run factor rotations ",
        cite(("Cam97", "Campbell, Lo, & MacKinlay, 1997"), ("Con01", "Cont, 2001")),
        ". Machine learning (‘ML’ hereafter) methods can account for these complexities and offer complementary "
        "perspectives to conventional approaches ",
        cite(("Mor16", "Moritz & Zimmermann, 2016"), ("Shi20", "Gu, Kelly, & Xiu, 2020"),
             ("Gig22", "Giglio, Kelly, & Xiu, 2022"), ("Wan23", "Wang, 2023")),
        ". However, this greater flexibility also increases the capacity to fit noise. This concern is particularly "
        "important in finance, where predictive signals are weak, samples are limited and relationships may change "
        "over time, and where backtests can overstate performance when model selection repeatedly uses the same "
        "observations or when training information overlaps with the period used for evaluation ",
        cite(("Har16", "Harvey, Liu, & Zhu, 2016"), ("Arn19", "Arnott, Harvey, & Markowitz, 2019"),
             ("Isr20", "Israel, Kelly, & Moskowitz, 2020")),
        "."])
    compose(e, b5, [
        "These issues are especially relevant in emerging equity markets. Greater informational inefficiency may "
        "leave more exploitable structure, but shorter data histories, higher idiosyncratic risk and more frequent "
        "structural breaks increase the risk that flexible models extract spurious patterns ",
        cite(("Leu21", "Leung, Lohre, Mischlich, Shea, & Stroh, 2021"),
             ("Nus23", "Cakici, Fieberg, Metko, & Zaremba, 2023")),
        ". Although ML methods have shown predictive value in emerging-market return studies, their performance is "
        "not necessarily transferable across markets ", cite(("Han23", "Hanauer & Kalsbach, 2023")), "."])
    compose(e, b6, [
        "The South African setting provides a relevant but comparatively under-researched context. Value and "
        "momentum effects have long been documented on the Johannesburg Stock Exchange, and subsequent work has "
        "shown that local style returns vary through time (Fraser & Page, 2000; Muller & Ward, 2013). Mutooni and "
        "Muller (2007) provide early evidence on equity style timing and Page et al. (2022) examine style rotation, "
        "while recent studies apply machine learning models to the same broad problem ",
        cite(("Mak23", "Makamo, 2023"), ("Pag241", "Page, McClelland, & Auret, 2024")),
        ". This literature establishes the relevance of local factor timing, but it does not isolate the "
        "incremental contribution of nonlinearity and interaction effects through a controlled comparison of a "
        "linear model, an additive nonlinear model and an interaction-capable tree model using identical targets, "
        "predictors and evaluation periods."])
    b2.addnext(b3)
    b3.addnext(b4)
    b4.addnext(b5)
    note("1.1", "Rewritten as one sequence from factor premia to the gap, in six paragraphs, merging the background "
         "material of the old chapter opening with the old Section 1.1: (1) factor premia and their origins; (2) "
         "their variation over time and the motivation for timing; (3) the mixed evidence on timing; (4) machine "
         "learning, its flexibility and the risk of fitting noise, with the backtest-overfitting point moved here "
         "from the old opening; (5) emerging markets; (6) South Africa, ending with the gap, now stated once (it "
         "was also at the end of the old fourth opening paragraph and in Section 1.2). The author's sentences are "
         "kept where the content only moved.", "Author: what goes into the background; limit repetition.")
    note("1.1", "Citations: Blitz, Falkenstein and van Vliet (2014), a paper on the low-volatility effect, is no longer "
         "cited for the motivation of factor timing (now Arnott, Beck, Kalesnik & West, 2016, from the old "
         "Section 1.1); it stays cited for low volatility. Asness, Frazzini and Pedersen (2019), on the quality "
         "factor, is no longer cited for the weakness of timing relationships; it stays cited for quality. "
         "'Daniel & Moskowitz, Momentum crashes, 2016' and 'Fama & French, A five-factor asset pricing model, 2015' "
         "printed a title because the Source Manager holds duplicates (see the notes below); 'Mul-ler' corrected.",
         "Citations that support the statement; Prof Gilbert's comments 'Fix reference' on both citations of the "
         "old first paragraph.")

    # 1.2 Problem statement: the gap is stated at the end of 1.1
    compose(e, para("This study addresses whether machine learning models can identify"), [
        "This study addresses whether machine learning models can identify predictive signals in the relative "
        "performance of South African equity factors. In response to the gap identified in Section 1.1, it compares "
        "logistic regression, generalised additive models and XGBoost on the same factor pairs, predictor set and "
        "time-series validation framework, to determine whether greater model flexibility improves the "
        "out-of-sample prediction of relative momentum, value and quality performance."])
    note("1.2", "'Existing local research provides limited evidence comparing models with different levels of "
         "flexibility …' replaced by a reference to the gap at the end of Section 1.1; double full stop removed.",
         "Repetition.")

    # 1.3 Scope and objectives: the data sentence repeats Section 1.4; scope and objectives as two paragraphs
    scope = para("The forecasting problem is defined in terms of the relative")
    objectives = fresh_copy(scope)
    compose(e, scope, [
        "The forecasting problem is defined in terms of the relative performance of three factor pairs: momentum "
        "versus value, momentum versus quality, and value versus quality, a formulation consistent with recent "
        "applications of ML to directional financial prediction ",
        cite(("Cot24", "Cotturo, Liu, & Proner, 2024"), ("Pio22", "Piovezan & de Andrea Jr., 2022")),
        ". The study concentrates on statistical predictive performance rather than portfolio construction. It "
        "therefore does not claim that classification accuracy automatically translates into an implementable or "
        "profitable factor-rotation strategy. Transaction costs, portfolio turnover, market impact and "
        "probability-based allocation rules fall outside the primary empirical scope and are identified as areas "
        "for future research."])
    compose(e, objectives, [
        "The primary objective of this study is to determine whether machine learning models with different levels "
        "of functional flexibility can predict the relative performance of momentum, value and quality factors in "
        "the South African equity market under a common time-series consistent evaluation framework. Within this "
        "scope, the study has the following specific objectives:"])
    scope.addnext(objectives)
    note("1.3", "Deleted 'The empirical design draws on a broad data universe comprising historical market "
         "information, macroeconomic indicators, and factor-specific time-series inputs, reflecting evidence that "
         "both price-based and macro-state variables can influence factor premia (Gu, Kelly, & Xiu, 2020; Feng, "
         "Giglio, & Xiu, 2020)': Section 1.4 says the same with the same citations. Scope and objectives split into "
         "two paragraphs (they ran together without a space); 'the study the following specific objectives' "
         "corrected to 'the study has the following specific objectives'.", "Repetition; language.")

    # 1.4 Research methodology: takes the formulation and the model classes from the old opening
    compose(e, para("The empirical design consists of three primary methodological"), [
        "The study formulates factor timing as a directional classification problem. For each pair of factors, the "
        "target identifies which factor subsequently exhibits the stronger trend, so the models predict relative "
        "factor leadership and are not required to estimate the magnitude of future returns. The empirical design "
        "has three main components: a target-labelling technique suited to financial time series, a comparison of "
        "model classes of increasing flexibility, and a validation framework that preserves the temporal ordering "
        "of the data."])
    note("1.4", "First paragraph rewritten: the classification formulation from the old chapter opening ('This study "
         "formulates factor timing as a directional classification problem …'), followed by the three components "
         "as one sentence (they were sentence fragments: 'Firstly, a sophisticated target labelling technique …'). "
         "'We focus on predicting relative outperformance between factor pairs rather than return magnitudes' "
         "deleted (now the second sentence).", "Author: method overview belongs in Section 1.4; language.")
    compose(e, para("To assess the incremental value of model flexibility"), [
        "To assess the incremental value of model flexibility, three model classes of increasing flexibility are "
        "compared: a regularised logistic regression, which provides the linear benchmark ",
        cite(("Zou05", "Zou & Hastie, 2005"), ("Hos13", "Hosmer, Lemeshow, & Sturdivant, 2013")),
        "; a semi-parametric GAM, which permits smooth nonlinear effects while "
        "retaining an additive structure ",
        cite(("Has86", "Hastie & Tibshirani, 1986"), ("GJa21", "James, Witten, Hastie, & Tibshirani, 2021")),
        "; and XGBoost, a tree-based method that can also represent higher-order interactions ",
        cite(("JHF01", "Friedman, 2001"), ("Che16", "Chen & Guestrin, 2016"),
             ("Gri22", "Grinsztajn, Oyallon, & Varoquaux, 2022")),
        "."])
    note("1.4", "Model classes: the description of each class from the old chapter opening ('provides a regularised "
         "linear benchmark', 'permits smooth nonlinear effects while retaining an additive structure', 'higher-order "
         "interactions') merged into this paragraph, which listed the same three classes. From the old paragraph's "
         "citations, Grinsztajn et al. (2022) is kept (it is cited nowhere else); the typed '(Cox 1953)' and the "
         "citations of Hastie and Tibshirani (1986), Hosmer et al. (2013) and Wood (2017) for XGBoost are not.",
         "Author: method overview belongs in Section 1.4; citations that support the statement.")
    p = para("Model development and evaluation follow a temporally ordered")
    e.replace(p, "This technique addresses the risk of data leakage by generating multiple, distinct train/test splits. "
                 "Crucially, for each split, it purges any training observations whose information horizon overlaps "
                 "with the test set, ensuring that the model is never tested on data influenced by information used "
                 "during training.",
              "It evaluates each candidate setting on several training–validation combinations and, in each "
              "combination, purges the training observations whose information horizon overlaps with the validation "
              "data, so that no setting is evaluated on data whose information was used in training.")
    e.replace(p, "evaluated using a walk-forward procedure which approximates",
              "evaluated using an expanding walk-forward procedure, which approximates")
    note("1.4", "CPCV described in the terms of Section 3.8.1 (training–validation combinations, validation data; "
         "was 'train/test splits' and 'test set'); the walk-forward is expanding, as in Section 3.8.2.",
         "Consistency with Chapter 3.")
    p = para("Model performance is assessed using a set of complementary")
    e.replace(p, "including average accuracy, the area under", "including accuracy, the area under")
    e.replace(p, "Together these measures capture directional accuracy (average accuracy), ranking ability (ROC-AUC), "
                 "and probability calibration (log-loss)",
              "Together these measures capture classification accuracy, ranking ability (ROC-AUC) and the quality of "
              "the predicted probabilities (log-loss)")
    note("1.4", "'average accuracy' corrected to 'accuracy'; log-loss described as measuring the quality of the "
         "predicted probabilities, not calibration (Section 3.9: log-loss is not a pure calibration measure).",
         "Consistency with Chapter 3.")
    p = para("Through this integrated framework")
    p.getparent().remove(p)
    note("1.4", "Deleted 'Through this integrated framework, the study evaluates the predictive performance, "
         "stability, and interpretability … over traditional linear approaches in forecasting relative factor "
         "performance.' It restates the aim (Sections 1.2 and 1.3), and stability is not assessed.", "Repetition.")

    # 1.5 and 1.6: small corrections
    e.replace(para("First, comparative modelling contribution."), "First, comparative modelling contribution. The study "
              "provides a comparison", "First, the study makes a comparative modelling contribution: it provides a "
              "comparison")
    note("1.5", "'First, comparative modelling contribution.' (a sentence fragment) joined to the next sentence.",
         "Language.")
    e.replace(para("Chapter 1 - Introduction:"), "contribution to data science", "contribution to data science.")
    p = para("Chapter 3 – Methodology:")
    e.replace(p, "We presents", "Chapter 3 presents")
    e.replace(p, "rolling walk-forward", "expanding walk-forward")
    p = para("Chapter 4 – Research Results:")
    old = text(p)[text(p).index("This chapter presents the out-of-sample empirical findings."):]
    e.replace(p, old, "This chapter presents the out-of-sample results. It compares the three models on accuracy, "
                      "ROC-AUC and log-loss for each factor pair, with their ROC curves and predicted probabilities, "
                      "examines how the classification errors are distributed between the two classes and the quality "
                      "of the predicted probabilities, reports the features with the largest global SHAP importance, "
                      "and assesses whether the more flexible models add predictive value beyond the regularised "
                      "linear benchmark.")
    note("1.6", "Chapter 1: full stop added. Chapter 3: 'We presents' corrected; 'rolling walk-forward' corrected to "
         "'expanding'. Chapter 4: described as it now stands (as in Section 4.1); it no longer contains 'the "
         "dynamics and dependence structure of the selected style factors' (now Section 3.3) or 'robustness across "
         "validation windows'. Chapter 5 is left as it is until it is rewritten.", "Accuracy of the outline.")

    # Reference page, and notes for the author: the Source Manager and typed citations
    cited, missing = reference_list(document, e, master, sources)
    note("References", f"Reference page added after the chapter: the {cited} works cited in Chapter 1. {missing} of "
         "them are missing from the thesis reference list and are highlighted there, in full.",
         "Author: some references are not in the master reference list.")
    note("References", "Why seven of them are missing although they have a source in the Source Manager: the thesis "
         "reference list is Word's bibliography, which lists only works cited with a citation field when it was last "
         "updated. Barroso and Santa-Clara (2015), Haddad, Kozak and Santosh (2020), Asness et al. (2017), Arnott, "
         "Harvey and Markowitz (2019) and Grinsztajn et al. (2022) are already cited with fields; Hanauer and "
         "Kalsbach (2023) and Mutooni and Muller (2007) become fields in this revision. After copying the chapter "
         "into the thesis, update the bibliography (References > Update Citations and Bibliography).",
         "Reference list.")
    note("References", "Complete these Source Manager entries first, or Word will print them without volume and issue: "
         "Bar15 volume 116, issue 1; Had20 volume 33, issue 5; Asn17 volume 43, issue 5; Arn19 volume 1, issue 1; "
         "Han23 volume 55 (now in the Pages field) and article number 101022; Mut07 volume 36, issue 65, DOI "
         "10.1080/10293523.2007.11082485; Gri22 volume 35. The full entries are on the reference page.",
         "Reference list; details checked against the publishers' records.")
    note("References", "Errors in existing Source Manager entries, visible in the reference list (not changed here): "
         "Asness, Frazzini and Pedersen (2019) prints 'Pedersen, L. 2.' (L. H.); Fama and French (2015) prints the "
         "journal as 'Journal of Financial' (Journal of Financial Economics); Daniel and Moskowitz (2016) prints "
         "'Moskowitz, D. J.' (T. J.); Hosmer et al. (2013) 'Logisitc'; Asness et al. (2000) has page '1' (50–60); "
         "López de Prado (2020) has edition '9 ed.' and London (Cambridge University Press, no edition); Chen and "
         "Guestrin (2016) lacks the proceedings and pages (Proceedings of the 22nd ACM SIGKDD International "
         "Conference on Knowledge Discovery and Data Mining, 785–794).", "Reference list.")
    note("Citations", "Typed, because the thesis has no source for them: Fraser and Page (2000), Muller and Ward "
         "(2013) and Page et al. (2022). Add them in the Source Manager with the details on the reference page, "
         "then replace the typed citations with citation fields so that they reach the reference list. The article "
         "number of Page et al. (2022) could not be confirmed (Finance Research Letters 46, Part B); check it on "
         "ScienceDirect. 'Mutooni and Muller (2007)' in the South African paragraph of Section 1.1 is a typed "
         "narrative citation; the source is now cited with a field in the chapter opening.", "Reference list.")
    note("Citations", "Duplicate entries in the Source Manager: Placeholder1 (Fama & French, 2015, also Fam15), "
         "Placeholder3 (Hastie et al., 2009, also Has09, which lacks Tibshirani as an author), Placeholder5 "
         "(Carhart, 1997, also Car97) and Placeholder6 (Gu, Kelly & Xiu, 2020, also Shi20, with the surname entered "
         "as 'Shihao'). Word adds a title to a citation when two sources share authors and year, which is why "
         "'Fama & French, A five-factor asset pricing model, 2015' appeared. Point any citation that uses a "
         "Placeholder entry to the main entry, then delete the Placeholder entries.", "Reference list.")

    add_log(document, log)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
