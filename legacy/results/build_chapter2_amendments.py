"""Build the Chapter 2 (Literature Review) amendment document.

The author restructured the chapter on 2-3 November 2026: Section 2.1 ('Foundational framework on factor timing'),
which restated much of the revised Section 1.1, is removed, so Machine learning in finance becomes Section 2.1, and
the challenges of factor timing open Section 2.2 without a heading of their own (the former Section 2.3.1). This
script puts the rewritten challenges text there: an opening that names five challenges, then one paragraph on each
(identifying genuine premia, their uncertain drivers, the length of factor cycles, the fragility of reported timing
gains, and the complexity of the relationship between predictors and factor returns), built from the material of
the former Sections 2.1 and 2.3.1, with references back to Sections 1.1 and 2.1. It adds a chapter opening that
outlines Sections 2.1-2.3, answers Prof Gilbert's comments on Sections 2.2.1, 2.2.3 and 2.3 (C47, C49, C50, C53),
removes repetition, and corrects grammar and citations in the rest of the chapter. Citations are Word citation
fields; narrative citations are the authors' names followed by a field that shows the year only (switches \\n \\t),
so that they still reach the reference list.

Starts from the current thesis (a snapshot), keeps only Chapter 2 and applies the amendments in place,
highlighted, with an amendment log at the end.

Usage: python build_chapter2_amendments.py [out_dir] [thesis_snapshot.docx]
"""
import re
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from build_chapter1_amendments import B_NS, norm
from build_chapter4_amendments import compose, find_para, fresh_copy, heading, style, text
from build_minimal_chapter3 import Editor, accept_revisions, add_log, remove, set_heading_start, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = OUT_DIR / "Chapter2_Literature_Review_Minimal_Amendments_Highlighted.docx"


def cite(*pairs):
    """A parenthetical citation field: Source Manager tags and the APA text Word shows for them."""
    return [tag for tag, _ in pairs], "(" + "; ".join(shown for _, shown in pairs) + ")"


def narrative(authors, tag, year):
    """'Cochrane (2011)': the names typed, the year a citation field (author and title suppressed)."""
    return [f"{authors} ", ([f"{tag} \\n  \\t"], f"({year})")]


def extend(e, p, parts):
    """Append highlighted text and citation fields (as in compose) to the end of a paragraph."""
    scratch = fresh_copy(p)
    compose(e, scratch, parts)
    for child in list(scratch):
        if child.tag != qn("w:pPr"):
            p.append(child)


def new_paragraph(e, like, parts):
    """A highlighted paragraph formatted like `like` (not yet placed in the document)."""
    p = fresh_copy(like)
    compose(e, p, parts)
    return p


def retag(e, p, match, instr, shown):
    """Change the one citation field in `p` whose instruction contains `match`: `instr` is its new instruction and
    `shown` the highlighted text it shows until Word updates its fields."""
    fields, current = [], None
    for r in p.iter(qn("w:r")):
        fc = r.find(qn("w:fldChar"))
        kind = fc.get(qn("w:fldCharType")) if fc is not None else None
        if kind == "begin":
            current = {"instr": [], "result": [], "in_result": False}
        elif kind == "separate":
            current["in_result"] = True
        elif kind == "end":
            fields.append(current)
            current = None
        elif current is not None:
            if r.find(qn("w:instrText")) is not None:
                current["instr"].append(r)
            elif r.find(qn("w:t")) is not None and current["in_result"]:
                current["result"].append(r)
    instructions = ["".join(r.find(qn("w:instrText")).text or "" for r in f["instr"]) for f in fields]
    hits = [f for f, ins in zip(fields, instructions) if "CITATION" in ins and match in ins.split()]
    assert len(hits) == 1, f"expected one citation field with {match!r}, found {len(hits)}"
    field = hits[0]
    field["instr"][0].find(qn("w:instrText")).text = instr
    for r in field["instr"][1:] + field["result"][1:]:
        r.getparent().remove(r)
    t = field["result"][0].find(qn("w:t"))
    t.text = shown
    t.set(qn("xml:space"), "preserve")
    e.mark(field["result"][0])


def cut(e, p, old):
    """Delete `old` (one occurrence) from a paragraph; deletions are recorded in the log, not highlighted."""
    e.replace(p, old, "")


def build():
    document = Document(SOURCE)
    body = document.element.body
    accept_revisions(body)
    strip_comments(body)

    # the thesis reference list (Word's bibliography) and the Source Manager, to check the cited works
    paras = list(body.iter(qn("w:p")))
    ref_head = [p for p in paras if text(p).strip() == "References" and style(p).startswith("Heading")][-1]
    master = []
    for p in paras[paras.index(ref_head) + 1:]:
        if style(p).startswith("Heading"):
            break
        if text(p).strip():
            master.append(text(p))
    with zipfile.ZipFile(SOURCE) as z:
        bib = next(etree.fromstring(z.read(n)) for n in z.namelist()
                   if n.startswith("customXml/item") and "Props" not in n and b"Sources" in z.read(n)[:400])
    sources = {s.findtext(B_NS + "Tag"): s for s in bib.findall(B_NS + "Source")}

    # keep only Chapter 2: from its "Chapter" label to the element before the next chapter label
    children = list(body)
    start = children.index(heading(body, "Literature Review"))
    end = next(i for i, el in enumerate(children) if i > start and el.tag == qn("w:p") and style(el) == "Heading1")
    first = start - 1 if style(children[start - 1]) == "Chapter" else start
    last = end - 2 if style(children[end - 1]) == "Chapter" else end - 1
    for i, el in enumerate(children):
        if el.tag != qn("w:sectPr") and not first <= i <= last:
            body.remove(el)
    set_heading_start(document, 2)

    e = Editor(True)
    log = []
    note = lambda section, change, reason: log.append((section, change, reason))

    # Author, 3 November 2026 (02:00): the heading 'Challenges with factor timing' is removed, so the challenges of
    # factor timing open Section 2.2 directly (before 2.2.1 Model complexity trade-offs). The rewritten challenges
    # text replaces the three paragraphs the thesis still has there (the former 2.3.1).
    head = heading(body, "Application of machine learning to factor timing in the global literature")
    old = [find_para(body, startswith="Literature establishes that momentum, value and quality are economically"),
           find_para(body, startswith="Predictive relationships shift over time."),
           find_para(body, startswith="Timing results depend on which predictors are tested")]
    following = [el for el in list(body)[list(body).index(head) + 1:] if el.tag == qn("w:p") and text(el).strip()]
    assert following[:3] == old and text(following[3]).strip() == "Model complexity trade-offs", \
        "the opening of Section 2.2 has changed"

    # ------------------------------------------------------------------ chapter opening --
    # What the review is for, and an outline of Sections 2.1-2.3 in the new order
    chapter_opening = new_paragraph(e, old[0], [
        "This chapter reviews the literature that motivates the study's design. Its purpose is to establish what "
        "machine learning can contribute to factor timing, why the problem is difficult and where these methods tend "
        "to fail, and what has not yet been tested in the South African market. Section 2.1 reviews the machine "
        "learning model classes relevant to the study, from regularised linear models to tree ensembles and deep "
        "learning. Section 2.2 reviews the international evidence on machine learning for factor timing: it first "
        "sets out the challenges of factor timing that apply whichever method is used, and then examines the "
        "trade-off between model complexity and robustness, differences across markets and the way predictive and "
        "economic value are evaluated. Section 2.3 reviews the South African evidence and sets out the gap that the "
        "study addresses."])
    title = heading(body, "Literature Review")
    assert style(title.getnext()) == "Heading2", "the chapter already has an opening"
    title.addnext(chapter_opening)
    note("2 (opening)", "New chapter opening: what the review is for (what machine learning can contribute to factor "
         "timing, why the problem is difficult and where these methods fail, what has not been tested in South "
         "Africa) and an outline of Sections 2.1–2.3 in the author's new order. The chapter title is followed "
         "directly by Section 2.1.", "Consistency with Chapter 1 and Section 4.1: a chapter opens by saying what it "
         "does; author's restructuring of 2–3 November.")

    # ------------------------------------------------------------------ 2.1 machine learning in finance --
    p = find_para(body, startswith="ML is a branch of artificial intelligence")
    e.replace(p, "approaches that rely on", "approaches rely on")
    retag(e, p, "Hastie2", " CITATION Bis06 \\l 1033 \\m Has09 \\t \\m Sut18 ",
          "(Bishop, 2006; Hastie, Tibshirani, & Friedman, 2009; Sutton & Barto, 2018)")
    p = find_para(body, startswith="Supervised learning refers to a family")
    retag(e, p, "Has09", " CITATION Has09 \\t \\l 1033 \\m Vap99 ",
          " (Hastie, Tibshirani, & Friedman, 2009; Vapnik, 1999)")
    p = find_para(body, startswith="Within supervised learning, literature distinguishes")
    e.replace(p, "Within supervised learning, literature", "Within supervised learning, the literature")
    note("2.1", "Grammar: 'While traditional econometric approaches that rely on pre-specified functional forms, ML "
         "methods emphasise …' becomes 'While traditional econometric approaches rely on …'; 'literature "
         "distinguishes' becomes 'the literature distinguishes'.", "Grammar.")
    note("2.1", "Citations of The Elements of Statistical Learning now all use source Has09 with the title "
         "suppressed (\\t): the supervised-learning paragraph printed the title ('Hastie & Friedman, The Elements "
         "of Statisitcal Learning, 2009') and the paradigms paragraph used the duplicate source Hastie2. They print "
         "'Hastie, Tibshirani, & Friedman, 2009' once Has09 is corrected (see References).", "Consistent citation "
         "of one source; remove the printed title.")

    p = heading(body, "Regularised & Semi-Parametric models")
    e.replace(p, "Regularised & Semi-Parametric models", "Regularised and semi-parametric models")
    p = heading(body, "Tree-Based Models")
    e.replace(p, "Tree-Based Models", "Tree-based models")
    note("2.1.1, 2.1.2", "Headings in sentence case, as the author changed 'Machine learning in finance': "
         "'Regularised and semi-parametric models', 'Tree-based models'.", "Consistent headings.")

    p = find_para(body, startswith="A central concern in empirical return prediction")
    retag(e, p, "Placeholder3", " CITATION Has09 \\t \\l 1033 ", "(Hastie, Tibshirani, & Friedman, 2009)")
    p = find_para(body, startswith="Early contributions such as ridge regression")
    e.replace(p, "Early contributions such as ridge regression", "Ridge regression")
    e.replace(p, "penalty stabilise", "penalty that stabilises")
    note("2.1.1", "'Early contributions such as ridge regression introduced an ℓ₂ penalty stabilise linear models' "
         "becomes 'Ridge regression introduced an ℓ₂ penalty that stabilises linear models'. The first paragraph "
         "cites Has09 instead of the duplicate source Placeholder3.", "Grammar; one source per work.")

    closing = find_para(body, startswith="The diversity of modelling approaches discussed above")
    remove(closing)
    note("2.1.3", "Deleted the last paragraph of Section 2.1 ('The diversity of modelling approaches discussed above "
         "reflects the broader econometric challenge …'). Its points are made in the opening of Section 2.1 (the "
         "dynamic, potentially nonlinear relationship between signals and factor returns) and in the fifth "
         "challenge that opens Section 2.2 (nonlinear, interactive and state-dependent drivers that led research to "
         "machine learning). Section 2.1 now ends with stacked generalisation, and Section 2.2 opens with the "
         "challenges of factor timing.", "No repetition: after the restructuring the paragraph sits directly "
         "before the fifth challenge's statement of the same point.")

    # ------------------------------------------------------------------ 2.2 opening: the challenges --
    paragraphs = list(old)
    for _ in range(3):
        extra = fresh_copy(old[0])
        paragraphs[-1].addnext(extra)
        paragraphs.append(extra)

    ilm21 = ("Ilm21", "Ilmanen, Israel, Lee, Moskowitz, & Thapar, 2021")
    opening, genuine, drivers, cycles, fragile, complexity = paragraphs
    # Author: "rephrase this sentence to set the scene"
    compose(e, opening, [
        "Although the time variation in factor premia provides a clear rationale for factor timing (Section 1.1), "
        "exploiting it in practice has proved difficult, whichever forecasting method is used. The literature "
        "identifies five recurring challenges: distinguishing genuine premia from statistical artefacts, establishing "
        "the economic drivers of those premia, accommodating the length of factor cycles, obtaining timing gains that "
        "persist out of sample, and capturing the complex relationship between predictors and factor returns."])
    compose(e, genuine, [
        "The first challenge is identifying which premia are genuine. Asset pricing research has uncovered a rapidly "
        "expanding set of cross-sectional return predictors, which ", *narrative("Cochrane", "Coc11", 2011),
        " described as a “zoo” of factors, and subsequent work attributes much of this proliferation to "
        "insufficiently stringent statistical standards: many proposed factors lack economic justification and fail "
        "to replicate out of sample ", cite(("Har16", "Harvey, Liu, & Zhu, 2016"), ("Fen20", "Feng, Giglio, & Xiu, 2020")),
        ". Even genuine premia shrink once they are published, as investors trade on them (McLean & Pontiff, 2016). "
        "Timing a factor whose premium is spurious cannot add value, and a premium that has decayed leaves less to "
        "time. A small core set of factors nevertheless remains widely used in research and practice ",
        cite(("Asn13", "Asness, Moskowitz, & Pedersen, 2013"), ("Fam15", "Fama & French, 2015")),
        ", and the three factors in this study, value, momentum and quality, belong to it."])
    compose(e, drivers, [
        "The second challenge is that the economic drivers of the premia are uncertain. Section 1.1 noted that "
        "risk-based and behavioural explanations both have empirical support, and the two point to different "
        "predictors. If the premia compensate for risk, they should vary with business conditions, as expected "
        "returns on stocks and bonds do ", cite(("Fam89", "Fama & French, 1989")),
        "; on the JSE, size and momentum premia have been found to vary with the business cycle (Kapche Fotso, "
        "2019). If they reflect investor biases, they should be largest when mispricing is hardest to arbitrage away, "
        "since real-world constraints limit arbitrage ", cite(("Shl97", "Shleifer & Vishny, 1997")),
        ". Without a settled theory, a timing model cannot draw its predictors from a single explanation, and factor "
        "returns appear to be influenced by a multifaceted set of variables that extends beyond economic "
        "fundamentals."])
    compose(e, cycles, [
        "The third challenge is the length of factor cycles. An equal allocation across factors is a demanding "
        "benchmark: ", *narrative("DeMiguel, Garlappi and Uppal", "DeM09", 2009),
        " show that equal weighting is difficult to beat out of sample when allocating across assets. Yet value "
        "crashes and growth rallies recur, and even a diversified factor portfolio can endure multi-year "
        "underperformance ",
        cite(("Asn00", "Asness, Friedman, Krail, & Liew, 2000"), ("Arn16", "Arnott, Beck, Kalesnik, & West, 2016")),
        ". Because factor cycles are asynchronous, dynamic reallocation could improve on static diversification ",
        cite(("Ilm12", "Ilmanen & Kizer, 2012")),
        ", but cycles that last several years also mean that a sample of a few decades contains few independent "
        "episodes from which a timing model can learn, a limitation that ", *narrative("Ilmanen et al.", "Ilm21", 2021),
        " address by examining a century of data."])
    compose(e, fragile, [
        "The fourth challenge is that reported timing gains are fragile. Section 1.1 noted that the evidence on "
        "factor timing is mixed, and several features of the problem explain why. The predictable component of factor "
        "returns is small relative to their noise ", cite(("Isr20", "Israel, Kelly, & Moskowitz, 2020")),
        ", and return predictors that perform well over a full sample often fail out of sample and in later "
        "subperiods, as Welch and Goyal (2008) show for the equity premium. Because many predictors and "
        "specifications are tested on the same data, the best reported result is biased upwards unless the search "
        "itself is taken into account (White, 2000). Results are also sensitive to the predictor, evaluation period, "
        "benchmark and treatment of implementation costs ",
        cite(("Asn17", "Asness, Chandra, Ilmanen, & Israel, 2017"), ilm21), ": ",
        *narrative("Arnott et al.", "Arn16", 2016), " find relative valuations useful for timing factors, whereas ",
        *narrative("Asness et al.", "Asn17", 2017), " find that value timing adds little beyond a static allocation, "
        "and Cederburg et al. (2020) find that volatility-managed portfolios do not systematically outperform their "
        "unmanaged counterparts out of sample."])
    # The machine learning methods are reviewed before this section (Section 2.1), so the last paragraph refers
    # back to them instead of introducing them
    compose(e, complexity, [
        "The fifth challenge is the complexity of the relationship between predictors and factor returns. ",
        *narrative("Cotturo, Liu and Proner", "Cot24", 2024),
        " note that the drivers of factor premia may operate over different horizons, that the relationship between "
        "predictors and factor performance may be nonlinear and interactive ",
        cite(("Leu21", "Leung, Lohre, Mischlich, Shea, & Stroh, 2021"), ("Bry23", "Bryzgalova, Pelger, & Zhu, 2023")),
        ", and that these relationships can change across market states ", cite(ilm21),
        ". The set of candidate predictors is also large relative to the available sample (Rapach & Zhou, 2013; "
        "Feng et al., 2018). These difficulties have led recent research to the machine learning methods reviewed in "
        "Section 2.1 ", cite(("Shi20", "Gu, Kelly, & Xiu, 2020")),
        ". Their ability to approximate nonlinear relationships does not, however, imply that the most flexible "
        "model will forecast best, which motivates the comparison of models with different levels of functional "
        "flexibility."])

    note("2.2 (opening)", "The author removed the heading 'Challenges with factor timing' (3 November), so the "
         "challenges text opens Section 2.2. The three paragraphs there (the former Section 2.3.1 'Challenges with "
         "time-series data') are replaced by the rewritten challenges text of the former Section 2.1, which the "
         "author moved here: an opening that names five challenges, then one paragraph on each (identifying genuine "
         "premia; their uncertain economic drivers; the length of factor cycles; the fragility of reported timing "
         "gains; the complexity of the relationship between predictors and factor returns).",
         "Author: remove Section 2.1 and insert it as the opening of Section 2.2.")
    note("2.2 (opening)", "Opening sentence rephrased to set the scene: 'Although the time variation in factor "
         "premia provides a clear rationale for factor timing (Section 1.1), exploiting it in practice has proved "
         "difficult, whichever forecasting method is used.' The list of challenges names what each difficulty is "
         "(genuine premia against statistical artefacts; gains that persist out of sample). 'whichever forecasting "
         "method is used' places the challenges within the review of machine learning applications.",
         "Author: rephrase this sentence to set the scene.")
    note("2.2 (opening)", "Last paragraph: the machine learning methods are now reviewed before this section, so it "
         "refers back to them ('the machine learning methods reviewed in Section 2.1') instead of introducing them, "
         "and 'which do not require a pre-specified functional form …' is deleted (stated in Section 2.1). The "
         "paragraph now leads into Section 2.2.1 (model complexity trade-offs).", "Author's restructuring; no "
         "repetition.")
    note("2.2 (opening)", "Content: kept from the former Section 2.1, which the author deleted from the thesis: the "
         "factor zoo (Cochrane, 2011; Harvey, Liu, & Zhu, 2016; Feng, Giglio, & Xiu, 2020) and the core set of "
         "factors; limits to arbitrage (Shleifer & Vishny, 1997) and the multifaceted drivers of factor returns; "
         "naïve diversification (DeMiguel, Garlappi, & Uppal, 2009), value crashes and asynchronous cycles; the "
         "sensitivity of timing gains; Cotturo et al.'s difficulties in forecasting factor premia. Kept from the "
         "three paragraphs replaced here: the decay of premia after publication (McLean & Pontiff, 2016); the JSE "
         "premia's variation with the business cycle (Kapche Fotso, 2019); the low signal-to-noise ratio of factor "
         "returns (now cited to Israel, Kelly, & Moskowitz, 2020); the failure of full-sample predictors in later "
         "subperiods (Welch & Goyal, 2008); data snooping (White, 2000); valuation timing as an example of predictor "
         "sensitivity (Arnott et al., 2016, against Asness et al., 2017). Added: that risk-based premia should vary "
         "with business conditions (Fama & French, 1989) and behavioural premia with the limits to arbitrage; that "
         "long cycles leave few independent episodes to learn from (Ilmanen et al., 2021); what Cederburg et al. "
         "(2020) find.", "Author: focus on the challenges, no repetition of Chapter 1.")
    note("2.2 (opening)", "Not kept from the replaced paragraphs: the factors and their variation over time and the "
         "motivation for timing (Section 1.1); the momentum-crash example (cited in Section 1.1); the duplicated "
         "citation group '(Harvey, Liu, & Zhu, 2016) (White, 2000; Harvey et al., 2016)'; and 'JSE style-rotation "
         "alphas largely disappear once momentum is added to the benchmark (Page, McClelland, & Auret, 2024)', "
         "which could not be confirmed: Page et al. (2024) report that their tree models beat a style-momentum "
         "benchmark. Restore it with the correct source if the paper (or Page et al., 2022) supports it.",
         "No repetition; accuracy of citations.")
    note("2.2 (opening)", "Narrative citations written as 'Cochrane (2011)', not '(Cochrane, 2011) described …': the "
         "names are typed and the year is a citation field that suppresses the author, so the work still reaches the "
         "reference list. The factor zoo is attributed to Cochrane (2011), the presidential address Prof Gilbert "
         "refers to (source Coc11). Fama and French (2015) is cited from source Fam15, not the duplicate "
         "Placeholder1 that printed the title. (Comments C42–C44 were on the deleted Section 2.1; these changes "
         "answer them for the text that moved.)", "Prof Gilbert's comments C42 (narrative citations), C43 (remove "
         "title) and C44 (reference the factor zoo properly).")

    # ------------------------------------------------------------------ 2.2.1 model complexity trade-offs --
    # Prof Gilbert (C47): set the scene for model choice with the challenges of financial data, and say how deep
    # learning overfits there; consider whether performance is measured by prediction or by portfolio returns.
    trade_off = find_para(body, startswith="The selection of an ML model for factor timing")
    trade_off.addprevious(new_paragraph(e, trade_off, [
        "The challenges set out above bear directly on the choice of model, because short samples, relationships "
        "that change over time and a low signal-to-noise ratio are the conditions in which a flexible model most "
        "readily fits noise rather than persistent structure ", cite(("Isr20", "Israel, Kelly, & Moskowitz, 2020")),
        ". Deep learning is the most exposed: ", *narrative("Gu, Kelly and Xiu", "Shi20", 2020),
        " find that the performance of neural networks in predicting US stock returns peaks at three hidden layers "
        "and declines as more layers are added, which they attribute to the small amount of data and low "
        "signal-to-noise ratio of return prediction compared with fields such as computer vision, where deep "
        "learning has excelled. On medium-sized tabular data, tree-based models still outperform deep learning ", cite(("Gri22", "Grinsztajn, Oyallon, & Varoquaux, 2022")), "."]))
    e.replace(trade_off, "sample variation, when converted", "sample variation, and when converted")
    e.replace(trade_off, "market microstructures, such as", "market characteristics such as")
    tree_trade_off = find_para(body, startswith="Tree-based models such as random forests and XGBoost provide")
    e.replace(tree_trade_off, "evaluation design(", "evaluation design (")
    note("2.2.1", "New first paragraph: the challenges that open Section 2.2 (short samples, changing relationships, "
         "a low signal-to-noise ratio) are the conditions in which flexible models fit noise (Israel, Kelly, & "
         "Moskowitz, 2020); Gu, Kelly and Xiu (2020) find that neural-network performance peaks at three hidden "
         "layers and declines with depth, which they attribute to small data and a low signal-to-noise ratio; tree "
         "models still outperform deep learning on tabular data (Grinsztajn, Oyallon, & Varoquaux, 2022). Each "
         "claim checked against the source (Gu et al., 2020, introduction: 'Shallow learning outperforms "
         "deeper learning').","Prof Gilbert's comment C47 (set the scene for model choice with the "
         "challenges of financial markets and how deep learning overfits in this context).")
    note("2.2.1", "Second paragraph: comma splice corrected ('… sample variation, and when converted into trading "
         "signals, …'); 'market microstructures, such as data quality, liquidity, and idiosyncratic risk' becomes "
         "'market characteristics such as …', since data quality and idiosyncratic risk are not microstructure. "
         "Third paragraph: space added before the citation ('evaluation design (Panda …').",
         "Grammar; accuracy.")

    combination = find_para(body, startswith="More broadly, combination methods represent")
    compose(e, combination, [
        "Combination methods (Section 2.1.3) offer a pragmatic response to the complexity–robustness trade-off, "
        "because they reduce reliance on any single, potentially unstable, complex model ",
        cite(("Kra17", "Krauss, Do, & Huck, 2017"), ("Nus23", "Cakici, Fieberg, Metko, & Zaremba, 2023")),
        ". The variance reduction they provide is most valuable when signals are weak and unstable, as in factor "
        "return forecasting ", cite(("Die00", "Dietterich, 2000"), ("Shi20", "Gu, Kelly, & Xiu, 2020")), "."])
    note("2.2.1", "Combination-methods paragraph shortened from three sentences to two: it refers back to Section "
         "2.1.3, where model combination is introduced, instead of describing it again, and 'improving out-of-sample "
         "robustness' is no longer stated twice. Citations unchanged.", "No repetition.")

    conclusion = find_para(body, startswith="Consequently, the marginal benefits of increasingly complex models")
    conclusion.addnext(new_paragraph(e, conclusion, [
        "Comparisons of model complexity also depend on how performance is measured. When models are compared by the "
        "returns of portfolios built from their forecasts, the result reflects not only the forecasts but also how "
        "they are converted into positions, through the choice of threshold, position size, rebalancing frequency "
        "and cost assumptions, and these choices add variation unrelated to the models' predictive ability. Such "
        "research design choices "
        "can matter more than sampling error: Chen, Hanauer and Kalsbach (2024) find that the variation in machine "
        "learning portfolio returns across alternative design choices exceeds their standard error."]))
    note("2.2.1", "New last paragraph: comparisons by portfolio returns also reflect how forecasts are converted into "
         "positions (threshold, position size, rebalancing, costs), which adds variation unrelated to predictive "
         "ability; Chen, Hanauer and Kalsbach (2024) find that the non-standard error from design choices in machine "
         "learning portfolio returns exceeds the standard error (by 59%). Section 2.3 states the study's response "
         "(models compared on predictive measures).", "Prof Gilbert's comment C47 (is performance measured by "
         "predictive accuracy or portfolio performance; signal implementation adds noise to model comparisons).")

    # ------------------------------------------------------------------ 2.2.2 market-specific patterns --
    p = find_para(body, startswith="Empirical evidence shows that the performance of ML models in factor timing")
    retag(e, p, "MaT23", " CITATION Kra17 \\l 7177 \\m Shi20 ", "(Krauss, Do, & Huck, 2017; Gu, Kelly, & Xiu, 2020)")
    note("2.2.2", "'Within this literature, tree-based ensemble methods are often competitive' is now cited to "
         "Krauss, Do and Huck (2017), who find that random forests outperform gradient-boosted trees and deep "
         "networks on the S&P 500, and Gu, Kelly and Xiu (2020), who find trees and neural networks the best "
         "performing methods. It cited Ma, Liao and Jiang (2023), a deep-learning study of China (not a developed "
         "market), and Ke et al. (2017) and Chen and Guestrin (2016), which introduce LightGBM and XGBoost (cited "
         "for that in Section 2.1.2) but do not compare them in finance.", "Accuracy of citations.")

    # ------------------------------------------------------------------ 2.2.3 predictive accuracy and value --
    note("2.2.3", "No change to the sentence Prof Gilbert queried ('the literature also raises several substantive "
         "critiques …'): it now carries three references (Harvey, Liu, & Zhu, 2016; Leung et al., 2021; Israel, "
         "Kelly, & Moskowitz, 2020), added by the author.", "Prof Gilbert's comment C49 (References?).")
    p = find_para(body, startswith="A central criticism concerns overfitting and model instability")
    cut(e, p, "Empirical evidence suggests that highly parameterised models frequently exhibit strong in-sample "
              "performance but suffer from degradation out-of-sample, especially when applied across different "
              "market regimes or geographies. ")
    e.replace(p, "This instability is further exacerbated in emerging markets",
              "Overfitting is more likely in emerging markets")
    note("2.2.3", "Overfitting paragraph: deleted 'Empirical evidence suggests that highly parameterised models "
         "frequently exhibit strong in-sample performance but suffer from degradation out-of-sample, especially when "
         "applied across different market regimes or geographies.' (uncited; stated in the fourth challenge and in "
         "Section 2.2.1). 'This instability is further exacerbated in emerging markets' becomes 'Overfitting is more "
         "likely in emerging markets'.", "No repetition.")
    p = find_para(body, startswith="A second line of criticism relates to economic interpretability")
    extend(e, p, [
        " Interpretability also affects adoption in practice: investment managers must be able to explain a "
        "strategy to their clients, and a model whose forecasts cannot be explained is difficult to defend when it "
        "underperforms. Explanation methods such as Shapley additive explanations ",
        cite(("Lun17", "Lundberg & Lee, 2017")),
        " partly address this by attributing each forecast to the predictors that produced it."])
    note("2.2.3", "Interpretability paragraph: two sentences added. Investment managers must be able to explain a "
         "strategy to clients, and an unexplained model is hard to defend when it underperforms; explanation methods "
         "such as Shapley additive explanations (Lundberg & Lee, 2017) partly address this. The second sentence "
         "introduces the explanation method the study uses, without referring forward.", "Prof Gilbert's comment C50 "
         "(practitioners avoid black-box models; 'trust the black box' does not resonate with clients when it is not "
         "working).")
    p = find_para(body, startswith="A related critique highlights data-snooping")
    e.replace(p, "A related critique highlights data-snooping and research design biases. The flexibility of ML "
                 "models allows for extensive feature exploration, hyperparameter tuning, and model selection, "
                 "increasing the risk of inadvertently fitting to sample-specific artefacts",
              "A related critique concerns data snooping. The flexibility of ML models adds feature selection, "
              "hyperparameter tuning and model selection to the specification search described above, increasing "
              "the risk of fitting sample-specific artefacts")
    cut(e, p, " This concern is particularly salient in factor timing, where signals are inherently weak and "
              "regime-dependent.")
    note("2.2.3", "Data-snooping paragraph: refers back to the specification search of the fourth challenge (White, "
         "2000) and states what machine learning adds to it (feature selection, hyperparameter tuning, model "
         "selection). Deleted 'This concern is particularly salient in factor timing, where signals are inherently "
         "weak and regime-dependent.' (stated in the opening of Section 2.2).", "No repetition.")
    p = find_para(body, startswith="Finally, the literature questions the generalizability")
    e.replace(p, "As a result, , the", "As a result, the")
    note("2.2.3", "Last paragraph: double comma removed ('As a result, , the evidence').", "Typing error.")

    # ------------------------------------------------------------------ 2.3 South Africa --
    p = heading(body, "Application of machine learning factor timing strategies in South African equity markets")
    e.replace(p, "machine learning factor timing strategies in", "machine learning to factor timing in")
    note("2.3", "Heading: 'Application of machine learning to factor timing in South African equity markets', "
         "parallel to the heading of Section 2.2.", "Grammar; consistent headings.")
    p = find_para(body, startswith="International evidence indicates that")
    extend(e, p, ["."])
    p = find_para(body, startswith="This global evidence highlights a critical trade-off")
    compose(e, p, [
        "The trade-off between the potential performance of complex models and the robustness of simpler ones is "
        "likely to be more pronounced in an emerging market such as South Africa, given the dependence of machine "
        "learning gains on market structure, sample depth and liquidity noted in Section 2.2.2. The criticisms "
        "reviewed in Section 2.2.3 do not undermine the relevance of machine learning for factor timing, but they "
        "require predictive gains to be assessed with validation that preserves temporal ordering, prevents "
        "information leakage and guards against overfitting."])
    note("2.3", "First paragraph: full stop added. Second paragraph shortened from five sentences to two: it refers "
         "back to Sections 2.2.2 and 2.2.3 instead of restating them, the sentence fragment 'A trade-off that is "
         "likely to be even more pronounced …' is joined to its sentence, and 'These considerations motivate the "
         "methodological choices adopted in this study, which prioritise …' is deleted, because the last paragraph "
         "states those choices. 'class imbalance, and transaction cost sensitivity' is not kept: the study does not "
         "evaluate transaction costs, and its classes are close to balanced.", "No repetition; describe only what "
         "the study does.")
    p = find_para(body, startswith="South African factor-timing evidence predates recent ML studies")
    compose(e, p, [
        "South African factor-timing evidence predates recent ML studies. ",
        *narrative("Mutooni and Muller", "Mut07", 2007), " examine equity style timing, and Page, McClelland and "
        "Auret (2022) document short-term style rotation on the JSE. ", *narrative("Makamo", "Mak23", 2023),
        " demonstrates the feasibility of LSTM networks in predicting factor and sector returns in the local equity "
        "market, reporting low in-sample prediction errors. ",
        *narrative("Page, McClelland and Auret", "Pag241", 2024),
        " compare random forests, XGBoost and LightGBM for JSE style rotation and report that the tree-based models "
        "improve on a style-momentum benchmark after costs. The local literature therefore establishes that JSE "
        "style returns vary through time and that ML models can be applied to style rotation. "
        "Evidence remains limited on how a transparent linear benchmark, an additive nonlinear model and a "
        "boosted-tree model compare when they use the same factor pairs, predictor set, labels and time-series "
        "evaluation framework. This controlled comparison defines the research gap addressed by the present study."])
    note("2.3", "South African studies: narrative citations ('Mutooni and Muller (2007) examine …', 'Makamo (2023) "
         "demonstrates …', 'Page, McClelland and Auret (2024) compare …') instead of parenthetical ones as sentence "
         "subjects. '(Page, McClelland, & Auret, 2024) 2022.document short-term style rotation' becomes 'Page, "
         "McClelland and Auret (2022) document short-term style rotation' (the 2022 paper, typed because it has no "
         "source). 'trans-parent' (a line-break hyphen) becomes 'transparent'.", "Prof Gilbert's comment C42 "
         "(narrative citations); typing errors.")
    p = find_para(body, startswith="Relative to the studies reviewed above, we present several methodological")
    compose(e, p, [
        "The study also improves on the design of the studies reviewed above in five ways. First, rather than "
        "forecasting return levels over a fixed horizon, it labels relative factor performance using trend scanning, "
        "which selects the forward horizon with the most statistically significant trend ",
        cite(("Lop20", "Lopez de Prado, 2020")),
        ". Second, hyperparameters are selected with combinatorial purged cross-validation, which removes training "
        "observations whose label horizons overlap the validation data ", cite(("Lop18", "Lopez de Prado, 2018")),
        ". Third, out-of-sample performance is assessed with an expanding walk-forward procedure that also purges "
        "the 12 weeks before each prediction, so that no training label draws on information from the test period. "
        "Fourth, the three model classes are estimated on identical data, predictors and test dates, so that "
        "differences in performance can be attributed to the models rather than to the research design. The models "
        "are also compared on predictive measures rather than portfolio returns, so that the comparison does not "
        "depend on how forecasts are converted into positions. Fifth, differences between models are assessed with "
        "block-bootstrap confidence intervals rather than inferred from point estimates. Together, these features "
        "address the weaknesses identified in this chapter: in-sample evaluation, information leakage from "
        "overlapping labels, single-model designs, portfolio-based comparisons that depend on implementation choices "
        "and the absence of formal model comparisons."])
    note("2.3", "Last paragraph (the author's answer to C53) restructured so that the improvements stand out: "
         "'First' to 'Fifth' (trend-scanning labels; combinatorial purged cross-validation; purged expanding "
         "walk-forward; identical data for the three model classes, compared on predictive measures rather than "
         "portfolio returns; block-bootstrap intervals). 'we present' becomes an impersonal construction. Added "
         "from C47: the models are compared on predictive measures, so the comparison does not depend on how "
         "forecasts become positions. 'tested statistically with a block bootstrap' becomes 'assessed with "
         "block-bootstrap confidence intervals', as Chapter 3 states (the bootstrap gives intervals, not "
         "null-calibrated tests). 'look-ahead bias' becomes 'information leakage', the term used above.",
         "Prof Gilbert's comments C53 (highlight the methodological improvements, trend scanning, CPCV, more "
         "clearly) and C47; consistency with Chapter 3.")

    # ------------------------------------------------------------------ other chapters and references --
    note("Other chapters", "Cross-references to update in the thesis after the restructuring: Section 3.6 says "
         "'Section 2.1 noted that the proliferation of proposed predictors, the factor-zoo (Cochrane, 2011) … and "
         "Section 2.2 that many correlated predictors make estimates unstable'; these are now Section 2.2 (its "
         "opening) and Section 2.1.1. The Chapter 1 outline (Research assignment outline) says Chapter 2 'reviews the "
         "foundational theory of factor premia and the evolution of factor timing strategies, followed by a "
         "structured examination of machine learning applications', which no longer matches the order.",
         "Author's restructuring of 2–3 November.")
    # cited works in the chapter that are missing from the thesis reference list
    tags = []
    for it in body.iter(qn("w:instrText")):
        if "CITATION" in (it.text or ""):
            words = it.text.split("CITATION")[1].split()
            tags += [words[0]] + [b for a, b in zip(words, words[1:]) if a == "\\m"]
    off = []
    for tag in dict.fromkeys(tags):
        s = sources[tag]
        last_name, year = next(s.iter(B_NS + "Last")).text, s.findtext(B_NS + "Year")
        if not any(norm(t).startswith(norm(last_name)) and f"({year}" in t for t in master):
            off.append(f"{last_name} ({year}, {tag})")
    note("References", "Cited with a source but missing from the thesis reference list (they appear when the "
         "bibliography is updated, References > Update Citations and Bibliography): " + ("; ".join(off) or "none")
         + ". Typed because the thesis has no source for them: Cederburg, O'Doherty, Wang and Yan (2020), Rapach and "
         "Zhou (2013), 'Feng et al. (2018)', McLean and Pontiff (2016), Kapche Fotso (2019), Welch and Goyal (2008), "
         "White (2000), Page, McClelland and Auret (2022) and Chen, Hanauer and Kalsbach (2024, 'Design choices, "
         "machine learning, and the cross-section of stock returns', SSRN working paper 5031755; check whether it "
         "has since been published). Add them in the Source Manager and replace the typed citations with fields; the "
         "details of Kapche Fotso (2019) are also marked as outstanding in the literature-review reference file. "
         "'Feng et al. (2018)' may be the 2018 working-paper version of Feng, He, Polson and Xu (2023), which is in "
         "the Source Manager (Fen23); if so, cite Fen23.", "Reference list.")
    note("References", "Source Manager corrections that change how Chapter 2 prints: Has09 (The Elements of "
         "Statistical Learning) holds Tibshirani as Hastie's middle name, so it prints 'Hastie & Friedman'; enter "
         "Hastie, T., Tibshirani, R. and Friedman, J. as three authors, correct 'Statisitcal' and delete the "
         "duplicates Hastie2 and Placeholder3, which no field uses after these amendments. Tib96: 'Tibshirina' "
         "should be 'Tibshirani'. Lop18 and Lop20 give López de Prado's first name as 'M' and 'Marcus', so Word "
         "prints 'Lopez de Prado M.' to tell them apart; enter 'Marcos' in both. Unused duplicates that can be "
         "deleted: Goo94 (Goodfellow et al., dated 1994; Goo16 is cited), Placeholder6 ('Shihao, Kelly, Xiu'; Shi20 "
         "is cited) and Alo22 (Nog22 is cited).", "Reference list.")

    add_log(document, log)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
