"""Build the Chapter 2 (Literature Review) amendment document.

The author restructured the chapter on 2-3 November 2026: Section 2.1 ('Foundational framework on factor timing'),
which restated much of the revised Section 1.1, is removed, so Machine learning in finance becomes Section 2.1, and
the challenges of factor timing open Section 2.2 without a heading of their own (the former Section 2.3.1). This
script puts the rewritten challenges text there: an opening that names five challenges, then one paragraph on each
(identifying genuine premia, their uncertain drivers, the length of factor cycles, the fragility of reported timing
gains, and the complexity of the relationship between predictors and factor returns), built from the material of
the former Sections 2.1 and 2.3.1, with references back to Sections 1.1 and 2.1. It adds a chapter opening that
outlines Sections 2.1-2.3, answers Prof Gilbert's comments on Sections 2.2.1, 2.2.3 and 2.3, removes repetition,
and corrects grammar and citations in the rest of the chapter. Citations are Word citation fields in brackets at
the end of the claim they support, as in the rest of the thesis; works without a source are added to the
document's Source Manager (NEW_SOURCES).

Starts from the current thesis (a snapshot), keeps only Chapter 2 and applies the amendments in place,
highlighted, with an amendment log at the end. The author copies the amendments into the thesis herself, so the
script builds from her latest version and applies only what is still to do (by 3 November, 15:49, the challenges
text and most of Sections 2.2 and 2.3 were in).

Usage: python build_chapter2_amendments.py [out_dir] [thesis_snapshot.docx]
"""
import re
import sys
import uuid
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from build_chapter1_amendments import B_NS, norm
from build_chapter4_amendments import compose, find_para, find_para_any, fresh_copy, heading, style, text
from build_minimal_chapter3 import Editor, accept_revisions, add_log, remove, set_heading_start, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = OUT_DIR / "Chapter2_Literature_Review_Minimal_Amendments_Highlighted.docx"


# Works cited in Sections 2.2 and 2.3 that had no source in the thesis's Source Manager. Details from the literature-review
# reference file ('Literature Review References Timing South African Equity Factors.docx'), except Rapach and Zhou
# (2013) and Chen, Hanauer and Kalsbach (2024), checked separately. Authors are (last name, first, middle).
NEW_SOURCES = [
    dict(Tag="McL16", SourceType="JournalArticle", authors=[("McLean", "R.", "D."), ("Pontiff", "J.", None)],
         Title="Does academic research destroy stock return predictability?", JournalName="The Journal of Finance",
         Year="2016", Volume="71", Issue="1", Pages="5-32"),
    dict(Tag="Wel08", SourceType="JournalArticle", authors=[("Welch", "I.", None), ("Goyal", "A.", None)],
         Title="A comprehensive look at the empirical performance of equity premium prediction",
         JournalName="The Review of Financial Studies", Year="2008", Volume="21", Issue="4", Pages="1455-1508"),
    dict(Tag="Whi00", SourceType="JournalArticle", authors=[("White", "H.", None)],
         Title="A reality check for data snooping", JournalName="Econometrica", Year="2000", Volume="68", Issue="5",
         Pages="1097-1126"),
    dict(Tag="Ced20", SourceType="JournalArticle",
         authors=[("Cederburg", "S.", None), ("O'Doherty", "M.", "S."), ("Wang", "F.", None), ("Yan", "X.", "S.")],
         Title="On the performance of volatility-managed portfolios", JournalName="Journal of Financial Economics",
         Year="2020", Volume="138", Issue="1", Pages="95-117"),
    dict(Tag="Rap13", SourceType="BookSection", authors=[("Rapach", "D.", "E."), ("Zhou", "G.", None)],
         editors=[("Elliott", "G.", None), ("Timmermann", "A.", None)], Title="Forecasting stock returns",
         BookTitle="Handbook of Economic Forecasting", Year="2013", Volume="2A", Pages="328-383",
         Publisher="Elsevier", City="Amsterdam"),
    dict(Tag="Kap19", SourceType="Report", authors=[("Kapche Fotso", "H.", "M.")],
         Title="Cyclicality of size, value and momentum on the Johannesburg Stock Exchange", Year="2019",
         ThesisType="Master's thesis", Publisher="University of the Western Cape", City="Cape Town"),
    dict(Tag="Pag22", SourceType="JournalArticle",
         authors=[("Page", "D.", None), ("McClelland", "D.", None), ("Auret", "C.", None)],
         Title="Style rotation on the JSE", JournalName="Finance Research Letters", Year="2022", Volume="46",
         Issue="Part B", Pages="102504"),
    dict(Tag="Che24", SourceType="Report", authors=[("Chen", "M.", None), ("Hanauer", "M.", "X."), ("Kalsbach", "T.", None)],
         Title="Design choices, machine learning, and the cross-section of stock returns", Year="2024",
         ThesisType="Working paper 5031755", Publisher="SSRN"),
]


def source_element(entry):
    """A Source Manager entry (b:Source) in the form Word writes."""
    b = lambda tag, value=None: etree.Element(B_NS + tag) if value is None else _text(B_NS + tag, value)
    s = b("Source")
    for key in ("Tag", "SourceType"):
        s.append(b(key, entry[key]))
    s.append(b("Guid", "{" + str(uuid.uuid5(uuid.NAMESPACE_URL, "factor-timing/" + entry["Tag"])).upper() + "}"))
    roles = b("Author")
    for role, people in (("Author", entry["authors"]), ("Editor", entry.get("editors", []))):
        if not people:
            continue
        names = b("NameList")
        for last, first, middle in people:
            person = b("Person")
            person.append(b("Last", last))
            person.append(b("First", first))
            if middle:
                person.append(b("Middle", middle))
            names.append(person)
        holder = b(role)
        holder.append(names)
        roles.append(holder)
    s.append(roles)
    for key in ("Title", "BookTitle", "JournalName", "Year", "Volume", "Issue", "Pages", "ThesisType", "Publisher",
                "City"):
        if key in entry:
            s.append(b(key, entry[key]))
    return s


def _text(tag, value):
    el = etree.Element(tag)
    el.text = value
    return el


def add_sources(document, entries):
    """Add Source Manager entries to the document's bibliography part (customXml); returns the new elements."""
    part = next(p for p in document.part.package.iter_parts()
                if str(p.partname).startswith("/customXml/item") and b"<b:Sources" in p.blob[:400])
    root = etree.fromstring(part.blob)
    existing = {s.findtext(B_NS + "Tag") for s in root.findall(B_NS + "Source")}
    added = []
    for entry in entries:
        assert entry["Tag"] not in existing, entry["Tag"]
        added.append(source_element(entry))
        root.append(added[-1])
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    return added


def cite(*pairs):
    """A parenthetical citation field: Source Manager tags and the APA text Word shows for them."""
    return [tag for tag, _ in pairs], "(" + "; ".join(shown for _, shown in pairs) + ")"


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
    added = add_sources(document, [x for x in NEW_SOURCES if x["Tag"] not in sources])   # not yet in the thesis
    for s in added:
        sources[s.findtext(B_NS + "Tag")] = s

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

    # Author, 3 November 2026 (15:49): the rewritten challenges text is in the thesis as the unheaded opening of
    # Section 2.2 (with her own opening sentences and small cuts in the first challenge), its sources McL16-Kap19 are
    # in her Source Manager, and so is most of the rest of Sections 2.2 and 2.3. Only what is still to do is applied;
    # each step checks the thesis text first.
    head = heading(body, "Application of machine learning to factor timing in the global literature")
    challenges = [find_para(body, startswith=f"The {n} challenge") for n in ("first", "second", "third", "fourth",
                                                                           "fifth")]
    following = [el for el in list(body)[list(body).index(head) + 1:] if el.tag == qn("w:p") and text(el).strip()]
    opening = following[0]
    assert following[1:6] == challenges and text(following[6]).strip() == "Model complexity trade-offs", \
        "the opening of Section 2.2 has changed"

    # ------------------------------------------------------------------ chapter opening --
    # What the review is for, and an outline of Sections 2.1-2.3 in the new order
    chapter_opening = new_paragraph(e, challenges[0], [
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
    # Author, 3 November 2026 (14:52): most of the Section 2.1 amendments are in the thesis (the grammar of the first
    # paragraph, 'the literature distinguishes', sentence-case headings, Has09 for The Elements of Statistical
    # Learning in the supervised-learning and regularisation paragraphs, Has09 corrected to three authors, her own
    # wording of the ridge sentence). Only what is still to do is applied; each step checks the thesis text.
    applied = []
    p = find_para(body, startswith="ML is a branch of artificial intelligence")
    if "approaches that rely on" in text(p):
        e.replace(p, "approaches that rely on", "approaches rely on")
    else:
        applied.append("the grammar of the first paragraph ('While traditional econometric approaches rely on …')")
    retag(e, p, "Hastie2", " CITATION Bis06 \\l 1033 \\m Has09 \\t \\m Sut18 ",
          "(Bishop, 2006; Hastie, Tibshirani, & Friedman, 2009; Sutton & Barto, 2018)")
    note("2.1", "The paradigms paragraph cites The Elements of Statistical Learning from source Has09, like the "
         "rest of the chapter, instead of the duplicate source Hastie2.", "One source per work.")
    p = find_para(body, startswith="Supervised learning refers to a family")
    if "Has09 \\t" not in "".join(it.text or "" for it in p.iter(qn("w:instrText"))):
        retag(e, p, "Has09", " CITATION Has09 \\t \\l 1033 \\m Vap99 ",
              " (Hastie, Tibshirani, & Friedman, 2009; Vapnik, 1999)")
    else:
        applied.append("Has09 with the title suppressed in the supervised-learning paragraph")
    hits = [q for q in body.iter(qn("w:p")) if text(q).startswith("Within supervised learning, literature")]
    if hits:
        e.replace(hits[0], "Within supervised learning, literature", "Within supervised learning, the literature")
    else:
        applied.append("'the literature distinguishes'")
    for old_heading, new_heading in (("Regularised & Semi-Parametric models", "Regularised and semi-parametric models"),
                                     ("Tree-Based Models", "Tree-based models")):
        if any(text(q).strip() == old_heading for q in body.iter(qn("w:p"))):
            e.replace(heading(body, old_heading), old_heading, new_heading)
        else:
            applied.append(f"the heading '{new_heading}'")
    p = find_para(body, startswith="A central concern in empirical return prediction")
    if "Placeholder3" in "".join(it.text or "" for it in p.iter(qn("w:instrText"))):
        retag(e, p, "Placeholder3", " CITATION Has09 \\t \\l 1033 ", "(Hastie, Tibshirani, & Friedman, 2009)")
    else:
        applied.append("Has09 instead of Placeholder3 in the regularisation paragraph")
    p = find_para_any(body, "Early contributions such as ridge regression", "Ridge regression introduced")
    if text(p).startswith("Early contributions"):
        e.replace(p, "Early contributions such as ridge regression", "Ridge regression")
    e.replace(p, "penalty that stabilise linear" if "penalty that stabilise linear" in text(p) else "penalty stabilise",
              "penalty that stabilises linear" if "penalty that stabilise linear" in text(p) else "penalty that stabilises")
    note("2.1.1", "Ridge sentence: 'an L2 penalty that stabilise linear models' becomes 'an L2 penalty that "
         "stabilises linear models' (subject–verb agreement).", "Grammar.")
    if applied:
        note("2.1", "Already in the thesis (author, 3 November), so not repeated here: " + "; ".join(applied) + ".",
             "–")

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
    applied = []
    theirs = "We outline some of the common challenges encountered in when attempting to generate factor timing signals"
    if theirs in text(opening):
        e.replace(opening, theirs, "The challenges commonly encountered when generating factor timing signals are "
                                   "outlined below")
        note("2.2 (opening)", "Second sentence: 'We outline some of the common challenges encountered in when "
             "attempting to generate factor timing signals' becomes 'The challenges commonly encountered when "
             "generating factor timing signals are outlined below': 'in when' corrected, and impersonal, as in the "
             "rest of the thesis.", "Grammar; academic register.")
    note("2.2 (opening)", "Already in the thesis (author, 3 November): the five challenges, with every citation a "
         "bracketed Word field at the end of the claim it supports, as in the rest of the thesis, and their new "
         "sources (McL16, Wel08, Whi00, Ced20, Rap13, Kap19). This text answers Prof Gilbert's comments on the "
         "deleted Section 2.1 (C42–C44): no author as the subject of a sentence, Fama and French (2015) from source "
         "Fam15 rather than the duplicate Placeholder1 that printed the title, and the factor zoo cited to Cochrane "
         "(2011). 'Feng et al., 2018' is cited as Feng, He, Polson and Xu (2023, Fen23), the published version of "
         "Feng, Polson and Xu's 2018 working paper 'Deep learning in characteristics-sorted factor models'; replace "
         "it if another paper was meant.", "–")

    # ------------------------------------------------------------------ 2.2.1 model complexity trade-offs --
    merged = find_para(body, startswith="The challenges set out above bear directly on the choice of model")
    if "market microstructures, such as" in text(merged):
        e.replace(merged, "market microstructures, such as", "market characteristics such as")
        note("2.2.1", "First paragraph: 'market microstructures, such as data quality, liquidity, and idiosyncratic "
             "risk' becomes 'market characteristics such as …', since data quality and idiosyncratic risk are not "
             "microstructure.", "Accuracy.")
    assert "Gu, Kelly, & Xiu" not in text(merged)
    note("2.2.1", "Not restored: the two sentences on deep learning that followed the first sentence in the previous "
         "amendments (in predicting US stock returns, neural-network performance peaks at three hidden layers and "
         "declines with depth, which Gu, Kelly and Xiu (2020) attribute to small data and a low signal-to-noise "
         "ratio; tree-based models still outperform deep learning on tabular data, Grinsztajn, Oyallon and "
         "Varoquaux, 2022). The author's merged paragraph leaves them out. Prof Gilbert's comment on this subsection "
         "asked how deep-learning models tend to overfit in this context, so add them back if that point should "
         "stay.", "Author's revision respected.")

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
    assert not any(text(q).startswith("Comparisons of model complexity also depend") for q in body.iter(qn("w:p")))
    conclusion.addnext(new_paragraph(e, conclusion, [
        "Comparisons of model complexity also depend on how performance is measured. When models are compared by the "
        "returns of portfolios built from their forecasts, the result reflects not only the forecasts but also how "
        "they are converted into positions, through the choice of threshold, position size, rebalancing frequency "
        "and cost assumptions, and these choices add variation unrelated to the models' predictive ability. Such "
        "research design choices can matter more than sampling error: the variation in machine learning portfolio "
        "returns across alternative design choices exceeds their standard error ",
        cite(("Che24", "Chen, Hanauer, & Kalsbach, 2024")), "."]))
    note("2.2.1", "New last paragraph: comparisons by portfolio returns also reflect how forecasts are converted into "
         "positions (threshold, position size, rebalancing, costs), which adds variation unrelated to predictive "
         "ability; the non-standard error from researchers' design choices in machine learning portfolio returns "
         "exceeds the standard error, by 59% (Chen, Hanauer, & Kalsbach, 2024; new source Che24). Section 2.3 states "
         "the study's response (models compared on predictive measures).", "Prof Gilbert's comment on this "
         "subsection: is performance measured by predictive accuracy or portfolio performance; signal "
         "implementation adds noise to model comparisons.")

    # ------------------------------------------------------------------ 2.2.2 market-specific patterns --
    p = find_para(body, startswith="Empirical evidence shows that the performance of ML models in factor timing")
    if "MaT23" in "".join(it.text or "" for it in p.iter(qn("w:instrText"))):
        retag(e, p, "MaT23", " CITATION Kra17 \\l 7177 \\m Shi20 ", "(Krauss, Do, & Huck, 2017; Gu, Kelly, & Xiu, 2020)")
        note("2.2.2", "'Within this literature, tree-based ensemble methods are often competitive' is now cited to "
             "Krauss, Do and Huck (2017) and Gu, Kelly and Xiu (2020) instead of Ma, Liao and Jiang (2023), Ke et al. "
             "(2017) and Chen and Guestrin (2016).", "Accuracy of citations.")
    else:
        applied.append("2.2.2: the tree-ensemble sentence cited to Krauss et al. (2017) and Gu et al. (2020)")

    # ------------------------------------------------------------------ 2.2.3 predictive accuracy and value --
    p = find_para(body, startswith="A central criticism concerns overfitting and model instability")
    if "Empirical evidence suggests that highly parameterised" in text(p):
        cut(e, p, "Empirical evidence suggests that highly parameterised models frequently exhibit strong in-sample "
                  "performance but suffer from degradation out-of-sample, especially when applied across different "
                  "market regimes or geographies. ")
        e.replace(p, "This instability is further exacerbated in emerging markets",
                  "Overfitting is more likely in emerging markets")
        note("2.2.3", "Overfitting paragraph: deleted the uncited sentence on in-sample strength and out-of-sample "
             "degradation (stated in the fourth challenge); 'This instability is further exacerbated' becomes "
             "'Overfitting is more likely'.", "No repetition.")
    else:
        applied.append("2.2.3: the overfitting paragraph shortened")
    p = find_para(body, startswith="A second line of criticism relates to economic interpretability")
    if "Interpretability also affects adoption" not in text(p):
        extend(e, p, [
            " Interpretability also affects adoption in practice: investment managers must be able to explain a "
            "strategy to their clients, and a model whose forecasts cannot be explained is difficult to defend when "
            "it underperforms. Explanation methods such as Shapley additive explanations ",
            cite(("Lun17", "Lundberg & Lee, 2017")),
            " partly address this by attributing each forecast to the predictors that produced it."])
        note("2.2.3", "Interpretability paragraph: two sentences added on client adoption and Shapley additive "
             "explanations (Lundberg & Lee, 2017).", "Prof Gilbert's comment C49 (practitioners avoid black-box "
             "models).")
    else:
        applied.append("2.2.3: the sentences on client adoption and Shapley explanations (Prof Gilbert's C49)")
    p = find_para_any(body, "A related critique highlights data-snooping", "A related critique concerns data snooping")
    if text(p).startswith("A related critique highlights"):
        e.replace(p, "A related critique highlights data-snooping and research design biases. The flexibility of ML "
                     "models allows for extensive feature exploration, hyperparameter tuning, and model selection, "
                     "increasing the risk of inadvertently fitting to sample-specific artefacts",
                  "A related critique concerns data snooping. The flexibility of ML models adds feature selection, "
                  "hyperparameter tuning and model selection to the specification search described above, "
                  "increasing the risk of fitting sample-specific artefacts")
    else:
        applied.append("2.2.3: the data-snooping paragraph refers back to the specification search")
    cut(e, p, " This concern is particularly salient in factor timing, where signals are inherently weak and "
              "regime-dependent.")
    note("2.2.3", "Data-snooping paragraph: deleted its last sentence, 'This concern is particularly salient in factor "
         "timing, where signals are inherently weak and regime-dependent.' (stated in the opening of Section 2.2).",
         "No repetition.")
    note("2.2.3", "No change to the sentence Prof Gilbert queried ('the literature also raises several substantive "
         "critiques …'): it carries three references (Harvey, Liu, & Zhu, 2016; Leung et al., 2021; Israel, Kelly, & "
         "Moskowitz, 2020), added by the author.", "Prof Gilbert's comment C48 (References?).")
    p = find_para(body, startswith="Finally, the literature questions the generalizability")
    if "As a result, , the" in text(p):
        e.replace(p, "As a result, , the", "As a result, the")
    else:
        applied.append("2.2.3: the double comma in the last paragraph")

    # ------------------------------------------------------------------ 2.3 South Africa --
    p = heading(body, "Application of machine learning factor timing strategies in South African equity markets")
    e.replace(p, "machine learning factor timing strategies in", "machine learning to factor timing in")
    note("2.3", "Heading: 'Application of machine learning to factor timing in South African equity markets', "
         "parallel to the heading of Section 2.2.", "Grammar; consistent headings.")
    p = find_para(body, startswith="International evidence indicates that")
    if not text(p).rstrip().endswith("."):
        extend(e, p, ["."])
        note("2.3", "First paragraph: full stop added after the citation.", "Typing error.")
    p = find_para_any(body, "This global evidence highlights a critical trade-off",
                      "The trade-off between the potential performance of complex models")
    if text(p).startswith("This global evidence"):
        compose(e, p, [
            "The trade-off between the potential performance of complex models and the robustness of simpler ones is "
            "likely to be more pronounced in an emerging market such as South Africa, given the dependence of machine "
            "learning gains on market structure, sample depth and liquidity noted in Section 2.2.2. The criticisms "
            "reviewed in Section 2.2.3 do not undermine the relevance of machine learning for factor timing, but they "
            "require predictive gains to be assessed with validation that preserves temporal ordering, prevents "
            "information leakage and guards against overfitting."])
        note("2.3", "Second paragraph shortened from five sentences to two (refers back to Sections 2.2.2 and "
             "2.2.3).", "No repetition.")
    else:
        applied.append("2.3: the second paragraph shortened")
    p = find_para(body, startswith="South African factor-timing evidence predates recent ML studies")
    compose(e, p, [
        "South African factor-timing evidence predates recent ML studies. Early work examines equity style timing ",
        cite(("Mut07", "Mutooni & Muller, 2007")), ", and later work documents short-term style rotation on the JSE ",
        cite(("Pag22", "Page, McClelland, & Auret, 2022")), ". More recently, LSTM networks have been applied to "
        "predict factor and sector returns in the local equity market, with low in-sample prediction errors ",
        cite(("Mak23", "Makamo, 2023")), ", and random forests, XGBoost and LightGBM have been compared for JSE style "
        "rotation, with the tree-based models improving on a style-momentum benchmark after costs ",
        cite(("Pag241", "Page, McClelland, & Auret, 2024")), ". The local literature therefore establishes that JSE "
        "style returns vary through time and that ML models can be applied to style rotation. Evidence remains "
        "limited on how a transparent linear benchmark, an additive nonlinear model and a boosted-tree model compare "
        "when they use the same factor pairs, predictor set, labels and time-series evaluation framework. This "
        "controlled comparison defines the research gap addressed by the present study."])
    note("2.3", "South African studies: citations as bracketed fields at the end of each claim, as in the rest of "
         "the thesis, instead of '(Mutooni & Muller, 2007) examine …' with the citation as the subject. 'short-term "
         "style rotation on the JSE' is cited to Page, McClelland and Auret (2022), 'Style rotation on the JSE' "
         "(new source Pag22), not to their 2024 machine-learning paper, which the next sentence cites.",
         "Referencing as in the rest of the thesis; accuracy of citations.")
    p = find_para_any(body, "Relative to the studies reviewed above, we present several methodological",
                      "The study also improves on the design of the studies reviewed above")
    if text(p).startswith("Relative to"):
        compose(e, p, [
            "The study also improves on the design of the studies reviewed above in four ways. First, rather than "
            "forecasting return levels over a fixed horizon, it labels relative factor performance using trend "
            "scanning, which selects the forward horizon with the most statistically significant trend ",
            cite(("Lop20", "Lopez de Prado, 2020")),
            ". Second, hyperparameters are selected with combinatorial purged cross-validation, which removes "
            "training observations whose label horizons overlap the validation data ",
            cite(("Lop18", "Lopez de Prado, 2018")),
            ". Third, out-of-sample performance is assessed with an expanding walk-forward procedure that also purges "
            "the 12 weeks before each prediction, so that no training label draws on information from the test "
            "period. Fourth, the three model classes are estimated on identical data, predictors and test dates, so "
            "that differences in performance can be attributed to the models rather than to the research design. The "
            "models are also compared on predictive measures rather than portfolio returns, so that the comparison "
            "does not depend on how forecasts are converted into positions. Together, these features address the "
            "weaknesses identified in this chapter: in-sample evaluation, information leakage from overlapping "
            "labels, single-model designs and portfolio-based comparisons that depend on implementation choices."])
    else:
        assert "bootstrap" not in text(p).lower()
        e.replace(p, "in five ways", "in four ways")
        e.replace(p, "information leakage from overlapping labels and single-model designs, portfolio-based "
                     "comparisons that depend on implementation choices and the absence of formal model comparisons",
                  "information leakage from overlapping labels, single-model designs and portfolio-based comparisons "
                  "that depend on implementation choices")
    note("2.3", "Last paragraph: the author removed the fifth improvement (block-bootstrap intervals for the "
         "differences between models), so 'in five ways' becomes 'in four ways' and 'the absence of formal model "
         "comparisons' is deleted from the weaknesses the four features address; the list now reads 'in-sample "
         "evaluation, information leakage from overlapping labels, single-model designs and portfolio-based "
         "comparisons that depend on implementation choices'.", "Author: the bootstrap comparison of the models is "
         "an extension the study does not explore.")
    if applied:
        note("2.2, 2.3", "Already in the thesis (author, 3 November), so not repeated here: " + "; ".join(applied)
             + ".", "–")

    # ------------------------------------------------------------------ other chapters and references --
    note("Other chapters", "Cross-references to update in the thesis after the restructuring: Section 3.6 says "
         "'Section 2.1 noted that the proliferation of proposed predictors, the factor-zoo (Cochrane, 2011) … and "
         "Section 2.2 that many correlated predictors make estimates unstable'; these are now Section 2.2 (its "
         "opening) and Section 2.1.1. The Chapter 1 outline (Research assignment outline) says Chapter 2 'reviews the "
         "foundational theory of factor premia and the evolution of factor timing strategies, followed by a "
         "structured examination of machine learning applications', which no longer matches the order.",
         "Author's restructuring of 2–3 November.")
    note("Other chapters", "The block bootstrap is still part of Chapters 3 and 4: the Chapter 3 amendments (Section "
         "3.9) assess uncertainty in ROC-AUC and the paired differences between models with a circular moving-block "
         "bootstrap, and the Chapter 4 outline says Section 4.5 'uses block-bootstrap intervals to assess whether "
         "their differences in mean ROC-AUC are distinguishable from zero'. If the comparison of models by bootstrap "
         "is outside the scope of the study, these need to change too.", "Consistency with the scope of the study.")
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
         + ".", "Reference list.")
    new_tags = [s.findtext(B_NS + "Tag") for s in added]
    described = {"Che24": "Chen, Hanauer and Kalsbach (2024, Che24; 'Design choices, machine learning, and the "
                          "cross-section of stock returns', SSRN working paper 5031755; check whether it has since "
                          "been published)",
                 "Pag22": "Page, McClelland and Auret (2022, Pag22; 'Style rotation on the JSE', Finance Research "
                          "Letters 46(B), Article 102504, as in the literature-review reference file)"}
    assert set(new_tags) <= set(described), new_tags
    note("References", "Added to the Source Manager of this document because the thesis does not have them yet: "
         + "; ".join(described[t] for t in new_tags) + ". Copy them into the thesis before pasting the paragraphs "
         "that cite them: with this document open, References > Manage Sources, select them in the Current List and "
         "Copy them to the Master List; then, in the thesis, open Manage Sources and copy them from the Master List "
         "to its Current List. (Tested: a citation copied into a document that lacks its source shows 'Invalid "
         "source specified'.)", "Citations as fields, as in the rest of the thesis.")
    has09 = [p.findtext(B_NS + "Last") for p in sources["Has09"].iter(B_NS + "Person")]
    lop = {sources[t].findtext(f"{B_NS}Author/{B_NS}Author/{B_NS}NameList/{B_NS}Person/{B_NS}First")
           for t in ("Lop18", "Lop20")}
    fixes = [] if has09 == ["Hastie", "Tibshirani", "Friedman"] else [
        "Has09 (The Elements of Statistical Learning): enter Hastie, Tibshirani and Friedman as three authors"]
    fixes.append("delete the duplicates Hastie2 and Placeholder3 of Has09, which no field uses after these amendments")
    if next(sources["Tib96"].iter(B_NS + "Last")).text == "Tibshirina":
        fixes.append("Tib96: 'Tibshirina' should be 'Tibshirani'")
    if len(lop) > 1:
        fixes.append("Lop18 and Lop20 give López de Prado's first name as " + " and ".join(f"'{x}'" for x in sorted(lop))
                     + ", so Word prints 'Lopez de Prado M.' to tell them apart; enter 'Marcos' in both")
    if "Statisitcal" in (sources["Has09"].findtext(B_NS + "Title") or ""):
        fixes.append("Has09 title: 'Statisitcal' should be 'Statistical'")
    fixes.append("unused duplicates that can be deleted: Goo94 (Goodfellow et al., dated 1994; Goo16 is cited), "
                 "Placeholder6 ('Shihao, Kelly, Xiu'; Shi20 is cited) and Alo22 (Nog22 is cited)")
    note("References", "Source Manager corrections still to make: " + "; ".join(fixes) + ".", "Reference list.")

    add_log(document, log)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
