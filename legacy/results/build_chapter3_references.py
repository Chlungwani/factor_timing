"""Add the reference list for Chapter 3 to the Chapter 3 amendment document.

The list gives every work cited in Chapter 3 of the thesis, with citation fields or typed, in the APA style of the
thesis reference list (Word's bibliography, generated from the Source Manager). Unhighlighted entries are as that
list prints them. Highlighted entries are missing from it, or printed with wrong or incomplete details; they are
given as Word will print them once the Source Manager entry is corrected. Two tables follow for the author: the
Source Manager changes, and the citations in the thesis's Chapter 3 that are typed rather than inserted as fields.
The four missing works are added to the document's Source Manager, to be copied into the thesis.

The section replaces 'Reference additions and corrections' (or a list made by an earlier run) in place; nothing
else in the document changes apart from the review note and one amendment-log row. Details of the corrected and
missing entries were checked against the publishers' records on 4 November 2026.

Usage: python build_chapter3_references.py [chapter3_docx] [thesis_snapshot] [out_docx]
"""
import copy
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

from build_chapter1_amendments import B_NS
from build_chapter2_amendments import source_element
from build_chapter4_amendments import fresh_copy, style, text
from build_minimal_chapter3 import Editor

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
CHAPTER = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "Chapter3_Methodology_Submission_Amendments_Highlighted_new_v2.docx"
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = Path(sys.argv[3]) if len(sys.argv) > 3 else CHAPTER

TITLE = "References cited in Chapter 3"
OLD_TITLES = (TITLE, "Reference additions and corrections")
NEXT_SECTION = "Material for Appendix B"
LOG_SECTION = "References"
LOG_MARK = "Reference list for Chapter 3"

# Works cited in Chapter 3 that have no source in the thesis's Source Manager. Entry text is a list of plain strings
# and 1-tuples (italic), as Word's APA style prints the source.
NEW_SOURCES = [
    (dict(Tag="Ped11", SourceType="JournalArticle",
          authors=[("Pedregosa", "F.", None), ("Varoquaux", "G.", None), ("Gramfort", "A.", None), ("Michel", "V.", None),
                   ("Thirion", "B.", None), ("Grisel", "O.", None), ("Blondel", "M.", None), ("Prettenhofer", "P.", None),
                   ("Weiss", "R.", None), ("Dubourg", "V.", None), ("Vanderplas", "J.", None), ("Passos", "A.", None),
                   ("Cournapeau", "D.", None), ("Brucher", "M.", None), ("Perrot", "M.", None), ("Duchesnay", "É.", None)],
          Title="Scikit-learn: Machine learning in Python", JournalName="Journal of Machine Learning Research",
          Year="2011", Volume="12", Pages="2825–2830"),
     ["Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., . . . Duchesnay, É. (2011). "
      "Scikit-learn: Machine learning in Python. ", ("Journal of Machine Learning Research, 12",), ", 2825–2830."]),
    (dict(Tag="Goo52", SourceType="JournalArticle", authors=[("Good", "I.", "J.")], Title="Rational decisions",
          JournalName="Journal of the Royal Statistical Society: Series B", Year="1952", Volume="14", Issue="1",
          Pages="107–114"),
     ["Good, I. J. (1952). Rational decisions. ", ("Journal of the Royal Statistical Society: Series B, 14",),
      "(1), 107–114."]),
    (dict(Tag="Gne07", SourceType="JournalArticle", authors=[("Gneiting", "T.", None), ("Raftery", "A.", "E.")],
          Title="Strictly proper scoring rules, prediction, and estimation",
          JournalName="Journal of the American Statistical Association", Year="2007", Volume="102", Issue="477",
          Pages="359–378"),
     ["Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. ",
      ("Journal of the American Statistical Association, 102",), "(477), 359–378."]),
    (dict(Tag="Kum20", SourceType="ConferenceProceedings",
          authors=[("Kumar", "I.", "E."), ("Venkatasubramanian", "S.", None), ("Scheidegger", "C.", None),
                   ("Friedler", "S.", "A.")],
          Title="Problems with Shapley-value-based explanations as feature importance measures",
          ConferenceName="Proceedings of the 37th International Conference on Machine Learning", Year="2020",
          Volume="119", Pages="5491–5500", Publisher="PMLR"),
     ["Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., & Friedler, S. A. (2020). Problems with "
      "Shapley-value-based explanations as feature importance measures. ",
      ("Proceedings of the 37th International Conference on Machine Learning",), " (Vol. 119, pp. 5491–5500). PMLR."]),
]

# Entries the thesis reference list prints with wrong or incomplete details: the corrected entry, and the change to
# make in the Source Manager (References > Manage Sources > Edit).
CORRECTED = {
    "Arn23": (["Arnott, R., Kalesnik, V., & Linnainmaa, J. (2023). Factor momentum. ",
               ("Review of Financial Studies, 36",), "(8), 3034–3070."],
              "Pages 3034–3070 (now 2379–2420)."),
    "Qua19": (["Asness, C., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. ",
               ("Review of Accounting Studies, 24",), "(1), 34-112."],
              "Pedersen: first name L., middle name H. (the middle-name box holds '2019.', printed 'L. 2.')."),
    "Ber12": (["Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor "
               "evaluation. ", ("Information Sciences, 191",), ", 192-213."],
              "Journal name without the full stop; Volume 191; Benítez: middle name M."),
    "Bis06": (["Bishop, C. M. (2006). ", ("Pattern Recognition and Machine Learning",), " (1 ed.). New York: Springer."],
              "Title without the final full stop (it is printed twice)."),
    "Cam97": (["Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). ", ("The Econometrics of Financial Markets",),
               ". Princeton: Princeton University Press."],
              "Title without the final full stop; Edition empty (the book has one edition, not two); City Princeton."),
    "Nus23": (["Cakici, N., Fieberg, C., Metko, D., & Zaremba, A. (2023). Machine learning goes global: "
               "Cross-sectional return predictability in international stock markets. ",
               ("Journal of Economic Dynamics and Control, 155",), ", 104725."],
              "Issue empty and Pages 104725 (the article number, now in Issue; Pages holds 1)."),
    "Che16": (["Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ",
               ("Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",),
               " (pp. 785–794). San Francisco: ACM."],
              "Conference name 'Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and "
              "Data Mining'; Pages 785–794; Publisher ACM (now 'ACM Digital Library')."),
    "Coc11": (["Cochrane, J. (2011). Presidential address: Discount rates. ", ("Journal of Finance, 66",),
               "(4), 1047–1108."], "Volume 66; Issue 4."),
    "Cot24": (["Cotturo, P., Liu, F., & Proner, R. (2024). Multi-Factor Timing with Deep Learning. ",
               ("SSRN Working Paper No. 4698640",), "."],
              "Journal name 'SSRN Working Paper No. 4698640'; Volume, Issue and Pages empty (each holds 1)."),
    "Cov06": (["Cover, T., & Thomas, J. (2006). ", ("Elements of Information Theory",), " (2 ed.). Hoboken: Wiley."],
              "Edition 2; City Hoboken."),
    "Dan16": (["Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. ", ("Journal of Financial Economics, 122",),
               "(2), 221-247."], "Moskowitz: first name T. (now D.)."),
    "Dix17": (["Dixon, M., Klabjan, D., & Bang, J. H. (2017). Classification-based Financial Markets Prediction using "
               "Deep Neural Networks. ", ("Algorithmic Finance, 6",), "(3–4), 67–77."],
              "Journal name Algorithmic Finance; Volume 6; Issue 3–4; Pages 67–77 (now 'Arxiv, 1(1), 1')."),
    "Fam15": (["Fama, E., & French, K. (2015). A five-factor asset pricing model. ",
               ("Journal of Financial Economics, 116",), "(1), 1-22."],
              "Journal name Journal of Financial Economics (now 'Journal of Financial'). Placeholder1 is a second "
              "copy of this source, so the list prints the entry twice: delete it."),
    "Fan22": (["Fang, Y., Chen, Y., & Ren, H. (2023). A Factor Pricing Model Based on Machine Learning Algorithm. ",
               ("International Review of Economics & Finance, 88",), ", 280–297."],
              "The 2022 SSRN paper is published: Year 2023; Journal name International Review of Economics & "
              "Finance; Volume 88; Issue empty; Pages 280–297 (now 'SSRN, 1(1), 1'). In-text: (Fang, Chen, & Ren, "
              "2023)."),
    "Faw06": (["Fawcett, T. (2006). An introduction to ROC analysis. ", ("Pattern Recognition Letters, 27",),
               "(8), 861–874."], "Journal name without the trailing comma; Volume 27; Issue 8."),
    "Fen23": (["Feng, G., He, J., Polson, N. G., & Xu, J. (2024). Deep Learning in Characteristics-Sorted Factor "
               "Models. ", ("Journal of Financial and Quantitative Analysis, 59",), "(7), 3001-3036."],
              "Year 2024 (the issue; online in 2023); Volume 59; Issue 7. In-text: (Feng, He, Polson, & Xu, 2024)."),
    "Shi20": (["Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. ",
               ("The Review of Financial Studies, 33",), "(5), 2223–2273."],
              "Journal name The Review of Financial Studies, Volume 33, Issue 5 (volume and issue are now in the "
              "journal name). Placeholder6 is a second copy with 'Shihao' as Gu's surname, printed as 'Shihao, G.': "
              "delete it."),
    "Guy03": (["Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. ",
               ("Journal of Machine Learning Research, 3",), ", 1157–1182."], "Volume 3."),
    "Han24": (["Han, Y. (2024). Machine Learning Based Portfolio Analysis. ",
               ("Highlights in Business, Economics and Management, 44",), ", 135–147."],
              "Issue empty; Pages 135–147 (now 1)."),
    "Has09": (["Hastie, T., Tibshirani, R., & Friedman, J. (2009). ", ("The Elements of Statistical Learning",),
               " (2 ed.). New York: Springer."],
              "Title 'Statistical' (now 'Statisitcal'). Hastie2 and Placeholder3 are copies of this source, so the "
              "list prints it three times: move any citation of them to Has09 and delete them."),
    "Hos13": (["Hosmer, D., Lemeshow, S., & Sturdivant, R. (2013). ", ("Applied Logistic Regression",),
               " (3 ed.). Hoboken: Wiley."], "Title 'Logistic' (now 'Logisitc')."),
    "Jos24": (["Joseph, A. (2024). From interpretability to inference: an estimation framework for universal "
               "approximators. ", ("arXiv:1903.04209",), "."],
              "Journal name arXiv:1903.04209; Pages empty (now 1). This is the current version of Jos19 (Bank of "
              "England Staff Working Paper No. 784, retitled), so '(Joseph, 2019; Joseph, 2024)' cites one paper "
              "twice: keep one."),
    "Kra04": (["Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. ",
               ("Physical Review E, 69",), "(6), 066138."], "Volume 69; Issue 6."),
    "Kra17": (["Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random "
               "forests: Statistical arbitrage on the S&P 500. ", ("European Journal of Operational Research, 259",),
               "(2), 689-702."], "Volume 259; Issue 2."),
    "Lak94": (["Lakonishok, J., Shleifer, A., & Vishny, R. W. (1994). Contrarian investment, extrapolation, and risk. ",
               ("The Journal of Finance, 49",), "(5), 1541-1578."], "Journal name without the trailing comma."),
    "Lop18": (["López de Prado, M. (2018). ", ("Advances in Financial Machine Learning",),
               " (1 ed.). Hoboken: John Wiley & Sons."],
              "Surname López de Prado; first name Marcos (now M; with 'Marcus' in Lop20, Word adds the initial to "
              "every citation: '(Lopez de Prado M. , 2018)'); City Hoboken."),
    "Lop20": (["López de Prado, M. (2020). ", ("Machine Learning for Asset Managers",),
               ". Cambridge: Cambridge University Press."],
              "Surname López de Prado; first name Marcos (now Marcus); Edition empty (now 9); City Cambridge."),
    "Lun17": (["Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. ",
               ("Advances in Neural Information Processing Systems, 30",), ", 4765–4774."],
              "Volume 30; Pages 4765–4774 (now 4768–4777)."),
    "Lun20": (["Lundberg, S., Erion, G., Chen, H., DeGrave, A., Prutkin, J., Nair, B., . . . Lee, S. (2020). From local "
               "explanations to global understanding with explainable AI for trees. ",
               ("Nature Machine Intelligence, 2",), "(1), 56–67."],
              "Title as published (now the preprint's title); Volume 2; Issue 1."),
    "Mur73": (["Murphy, A. (1973). A new vector partition of the probability score. ",
               ("Journal of Applied Meteorology, 12",), "(4), 595–600."], "Volume 12; Issue 4."),
    "Pen05": (["Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: criteria of "
               "max-dependency, max-relevance, and min-redundancy. ",
               ("IEEE Transactions on Pattern Analysis and Machine Intelligence, 27",), "(8), 1226–1238."],
              "Volume 27; Issue 8."),
    "Pio22": (["Piovezan, R. P. B., de Andrade Júnior, P. P., & Ávila, S. L. (2024). Machine learning method for "
               "return direction forecast of exchange traded funds (ETFs) using classification and regression "
               "models. ", ("Computational Economics, 63",), "(5), 1827–1852."],
              "No such article in the Journal of Asset Management: the paper was published in Computational "
              "Economics (the 2022 arXiv version was withdrawn). Authors Piovezan, R. P. B.; de Andrade Júnior, "
              "P. P.; Ávila, S. L. Year 2024; Journal name Computational Economics; Volume 63; Issue 5; Pages "
              "1827–1852. In-text: (Piovezan, de Andrade Júnior, & Ávila, 2024)."),
    "Sae22": (["Saetia, K., & Yokrattanasak, J. (2022). Stock Movement Prediction Using Machine Learning Based on "
               "Technical Indicators and Google Trend Searches in Thailand. ",
               ("International Journal of Financial Studies, 11",), "(1), 5."],
              "Issue 1 (now 5); Pages 5, the article number (now 1)."),
    "Sha532": (["Shapley, L. (1953). A value for n-person games. In H. W. Kuhn, & A. W. Tucker (Eds.), ",
                ("Contributions to the Theory of Games",), " (Vol. 2, pp. 307-317). Princeton: Princeton University "
                "Press."],
               "Type Book Section; Book title Contributions to the Theory of Games; Editors Kuhn, H. W. and Tucker, "
               "A. W.; Volume 2; City Princeton; Publisher Princeton University Press."),
    "Štr10": (["Štrumbelj, E., & Kononenko, I. (2010). An efficient explanation of individual classifications using "
               "game theory. ", ("Journal of Machine Learning Research, 11",), ", 1-18."],
              "Title 'An efficient explanation of individual classifications using game theory' (now the title of "
              "the authors' later paper); Volume 11."),
    "Sul16": (["Sulaiman, M. A., & Labadin, J. (2016). Improved Feature Selection Based on Mutual Information for "
               "Regression Tasks. ", ("Journal of IT in Asia, 6",), ", 11–24."],
              "The paper is in the Journal of IT in Asia, not the Indonesian Journal of Electrical Engineering and "
              "Computer Science: Journal name Journal of IT in Asia; Volume 6; Issue empty; Pages 11–24; Sulaiman: "
              "first name M., middle name A."),
    "Tib96": (["Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. ",
               ("Journal of the Royal Statistical Society: Series B, 58",), "(1), 267-288."],
              "Surname Tibshirani (now 'Tibshirina', also in the citation); Journal name 'Journal …' (now "
              "'ournal …')."),
    "War63": (["Ward, J. (1963). Hierarchical grouping to optimize an objective function. ",
               ("Journal of the American Statistical Association, 58",), "(301), 236–244."], "Volume 58; Issue 301."),
}

# Citations typed in the thesis's Chapter 3 (not citation fields): exact text, how often, section, tags, remark.
TYPED = [
    ("(Campbell, Lo, & MacKinlay, 1997; López de Prado, 2018)", 1, "3.4", ["Cam97", "Lop18"], ""),
    ("(Kraskov, Stögbauer, & Grassberger, 2004; Cover & Thomas, 2006; Peng, Long, & Ding, 2005; Guyon & Elisseeff, "
     "2003)", 1, "3.6", ["Kra04", "Cov06", "Pen05", "Guy03"],
     "It follows the field citing the same works without Peng et al.: add Pen05 to that field and delete the typed "
     "citation."),
    ("Gu et al. (2020); Daniel & Moskowitz (2016)", 2, "3.6, feature table", ["Shi20", "Dan16"],
     "Two rows of the Evidence column; the row above them already uses fields."),
    ("(Pedregosa et al., 2011)", 1, "3.7.2", ["Ped11"], "New source."),
    ("(Fawcett, 2006)", 1, "3.9", ["Faw06"], ""),
    ("(Good, 1952; Gneiting & Raftery, 2007)", 1, "3.9", ["Goo52", "Gne07"],
     "New sources. A field citing Bishop (2006) and Murphy (1973) follows after the full stop: one bracket with all "
     "four, before the full stop."),
    ("(Gneiting & Raftery, 2007; Dixon, Klabjan, & Bang, 2017; López de Prado, 2018)", 1, "3.9",
     ["Gne07", "Dix17", "Lop18"], ""),
    ("(Štrumbelj & Kononenko, 2010)", 2, "3.10", ["Štr10"], ""),
    ("(Joseph, 2019; Joseph, 2024)", 1, "3.10", ["Jos19", "Jos24"], "One paper in two versions: cite one."),
    ("(Lundberg et al., 2020)", 1, "3.10", ["Lun20"], ""),
    ("(Ku-mar et al., 2020; Molnar, 2022)", 1, "3.10", ["Kum20", "Mol22"],
     "New source Kum20. 'Ku-mar' and, in the same section, 'ob-servations', 'sam-ple' and 'to-gether' are line-end "
     "hyphens from pasting."),
]


def sort_key(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", "".join(c for c in s if not unicodedata.combining(c))).split())


def run(content, italic=False, highlight=False, size=None):
    """A run with its properties in schema order."""
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    if italic:
        rpr.append(OxmlElement("w:i"))
        rpr.append(OxmlElement("w:iCs"))
    if size:
        for tag in ("w:sz", "w:szCs"):
            el = OxmlElement(tag)
            el.set(qn("w:val"), str(size))
            rpr.append(el)
    if highlight:
        el = OxmlElement("w:highlight")
        el.set(qn("w:val"), "yellow")
        rpr.append(el)
    if len(rpr):
        r.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = content
    r.append(t)
    return r


def bold(r, size):
    """Bold, highlighted heading text of the given size (half-points)."""
    for old in r.findall(qn("w:rPr")):
        r.remove(old)
    rpr = OxmlElement("w:rPr")
    for tag, val in (("w:b", None), ("w:bCs", None), ("w:sz", size), ("w:szCs", size), ("w:highlight", "yellow")):
        el = OxmlElement(tag)
        if val is not None:
            el.set(qn("w:val"), str(val))
        rpr.append(el)
    r.insert(0, rpr)


def paragraph_like(like, parts, highlight):
    """A paragraph with the paragraph properties of `like` and the given text parts (1-tuples are italic)."""
    p = fresh_copy(like)
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    for part in parts:
        p.append(run(part if isinstance(part, str) else part[0], italic=not isinstance(part, str),
                     highlight=highlight))
    return p


def chapter_citations(body):
    """Field tags (first-use order) and the text outside fields of every paragraph in Chapter 3."""
    tags, plain, on = [], [], False
    for p in body.iter(qn("w:p")):
        if style(p) == "Heading1":
            on = text(p).strip() == "Methodology"
        if not on:
            continue
        depth, instr, outside = 0, "", ""
        for el in p.iter(qn("w:fldChar"), qn("w:instrText"), qn("w:t")):
            if el.tag == qn("w:fldChar"):
                kind = el.get(qn("w:fldCharType"))
                if kind == "begin":
                    depth += 1
                    instr = "" if depth == 1 else instr
                elif kind == "end":
                    depth -= 1
                    if depth == 0 and instr.strip().startswith("CITATION"):
                        tags += [instr.split()[1]] + re.findall(r"\\m (\S+)", instr)
            elif el.tag == qn("w:instrText"):
                instr += el.text or ""
            elif depth == 0:
                outside += el.text or ""
        plain.append(outside)
    assert plain, "Chapter 3 ('Methodology') not found in the thesis"
    return list(dict.fromkeys(tags)), "\n".join(plain)


def thesis_sources(path):
    with zipfile.ZipFile(path) as z:
        root = next(etree.fromstring(z.read(n)) for n in z.namelist()
                    if n.startswith("customXml/item") and "Props" not in n and b"<b:Sources" in z.read(n)[:600])
    return {s.findtext(B_NS + "Tag"): s for s in root.findall(B_NS + "Source")}


def master_entry(master, source):
    """The thesis reference-list paragraph printed for a Source Manager entry."""
    people = list(source.iter(B_NS + "Last"))
    last, year = sort_key(people[0].text), source.findtext(B_NS + "Year")
    title = " ".join(sort_key(source.findtext(B_NS + "Title")).split()[:4])
    hits = [p for p in master if sort_key(text(p)).startswith(last) and f"({year}" in text(p) and title in sort_key(text(p))]
    return hits[0] if hits else None


def label(source):
    people = [p.text for p in source.find(B_NS + "Author").iter(B_NS + "Last")]
    who = people[0] if len(people) == 1 else " & ".join(people) if len(people) == 2 else people[0] + " et al."
    return f"{who} ({source.findtext(B_NS + 'Year')})"


def set_cell(tc, value):
    """Put plain text into a table cell made from a template row (first paragraph, first run)."""
    ps = tc.findall(qn("w:p"))
    for extra in ps[1:]:
        tc.remove(extra)
    runs = ps[0].findall(qn("w:r"))
    for extra in runs[1:]:
        ps[0].remove(extra)
    runs[0].find(qn("w:t")).text = value
    runs[0].find(qn("w:t")).set(qn("xml:space"), "preserve")


def table_like(log_table, header, rows):
    """A table formatted like the amendment log, with the given header and rows."""
    tbl = copy.deepcopy(log_table)
    trs = tbl.findall(qn("w:tr"))
    template = copy.deepcopy(trs[1])
    for tr in trs[1:]:
        tbl.remove(tr)
    for tc, value in zip(trs[0].findall(qn("w:tc")), header):
        set_cell(tc, value)
    for values in rows:
        tr = copy.deepcopy(template)
        for tc, value in zip(tr.findall(qn("w:tc")), values):
            set_cell(tc, value)
        tbl.append(tr)
    for el in tbl.iter():
        for attr in [a for a in el.attrib if a.startswith("{http://schemas.microsoft.com/office/word/2010/wordml}")]:
            del el.attrib[attr]
    return tbl


def add_new_sources(document):
    """Add NEW_SOURCES to the document's Source Manager part, skipping tags it already has."""
    part = next(p for p in document.part.package.iter_parts()
                if str(p.partname).startswith("/customXml/item") and b"<b:Sources" in p.blob[:600])
    root = etree.fromstring(part.blob)
    existing = {s.findtext(B_NS + "Tag") for s in root.findall(B_NS + "Source")}
    added = []
    for entry, _ in NEW_SOURCES:
        if entry["Tag"] in existing:
            continue
        s = source_element(entry)
        for key in ("ConferenceName",):
            if key in entry:
                el = etree.SubElement(s, B_NS + key)
                el.text = entry[key]
        root.append(s)
        added.append(entry["Tag"])
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    return added


def build():
    thesis = Document(SOURCE)
    sources = thesis_sources(SOURCE)
    tags, plain = chapter_citations(thesis.element.body)
    master = [p for p in thesis.element.body.iter(qn("w:p")) if style(p) == "Bibliography" and text(p).strip()]
    new = {entry["Tag"]: parts for entry, parts in NEW_SOURCES}

    # typed citations: every one is known, and none is left over
    rest = plain
    for typed, count, *_ in TYPED:
        assert rest.count(typed) == count, (typed, rest.count(typed))
        rest = rest.replace(typed, "")
    leftover = re.findall(r"[^\n]{0,40}(?:, |\()(?:1[89]|20)\d\d[a-z]?\b[^\n]{0,10}", rest)
    assert not leftover, f"typed citations not in TYPED: {leftover}"
    cited = list(dict.fromkeys(tags + [t for *_, ts, _ in TYPED for t in ts]))
    assert set(CORRECTED) <= set(cited), f"corrections for works no longer cited: {set(CORRECTED) - set(cited)}"

    # entries: as the thesis prints them, or corrected / missing (highlighted)
    entries, as_printed, corrected, missing = [], 0, 0, 0
    like = master[0]
    for tag in cited:
        if tag in new:
            entries.append(paragraph_like(like, new[tag], True)); missing += 1
            continue
        assert tag in sources, f"{tag} is cited but has no source"
        hit = master_entry(master, sources[tag])
        assert hit is not None, f"{tag} ({label(sources[tag])}) is not in the thesis reference list"
        if tag in CORRECTED:
            entries.append(paragraph_like(like, CORRECTED[tag][0], True)); corrected += 1
        else:
            p = fresh_copy(hit)
            for r in [r for r in p.iter(qn("w:r")) if r.find(qn("w:fldChar")) is not None
                      or r.find(qn("w:instrText")) is not None]:
                r.getparent().remove(r)
            entries.append(p); as_printed += 1
    entries.sort(key=lambda p: sort_key(text(p)))

    # the Chapter 3 amendment document: replace the reference section
    document = Document(CHAPTER)
    body = document.element.body
    children = list(body)
    start = next(i for i, el in enumerate(children) if el.tag == qn("w:p") and text(el).strip() in OLD_TITLES)
    end = next(i for i, el in enumerate(children) if i > start and text(el).strip().startswith(NEXT_SECTION))
    head, intro = children[start], children[start + 1]
    log_title = next(el for el in children if text(el).strip().startswith("Amendment log for the submitted chapter"))
    log_table = next(el for el in children[children.index(log_title):] if el.tag == qn("w:tbl"))
    for el in children[start + 1:end]:
        body.remove(el)

    for r in head.findall(qn("w:r"))[1:]:
        head.remove(r)
    for t in head.findall(".//" + qn("w:t"))[1:]:
        t.getparent().remove(t)
    head.find(".//" + qn("w:t")).text = TITLE
    bold(head.find(qn("w:r")), 26)                     # like the other end-matter headings
    subhead = fresh_copy(head)
    ppr = subhead.find(qn("w:pPr"))
    for el in ppr.findall(qn("w:pageBreakBefore")):
        ppr.remove(el)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "360")
    ppr.append(spacing)
    for el in subhead.findall(".//" + qn("w:lastRenderedPageBreak")):
        el.getparent().remove(el)
    bold(subhead.find(qn("w:r")), 22)

    def heading_para(value):
        p = fresh_copy(subhead)
        p.find(".//" + qn("w:t")).text = value
        return p

    def note_para(value):
        p = fresh_copy(intro)
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        p.append(run(value))
        return p

    n_typed = sum(count for _, count, *_ in TYPED)
    intro_text = (
        f"Every work cited in Chapter 3 of the thesis (version of 4 November 2026): {len(cited)} works, in the style "
        "of the thesis reference list, which Word generates from the Source Manager. Unhighlighted entries are as "
        f"that list prints them ({as_printed}). Highlighted entries are missing from it ({missing}) or printed with "
        f"wrong or incomplete details ({corrected}); they are given as Word will print them once the Source Manager "
        "entry is corrected as set out in the first table below. The second table lists the citations in the "
        f"thesis's Chapter 3 that are typed rather than inserted as citation fields ({n_typed}). Giglio, Kelly and "
        "Xiu (2022) and Defazio, Bach and Lacoste-Julien (2014) are cited only in sentences of this review copy that "
        "the thesis does not use, and are not listed.")
    changes = [(tag, label(sources[tag]), CORRECTED[tag][1]) for tag in cited if tag in CORRECTED]
    changes.sort(key=lambda row: sort_key(row[1]))
    changes = [(tag, "New source", "Add: copy it from this document's Source Manager (Manage Sources: this "
                "document's Current List to the Master List, then to the thesis's Current List).")
               for tag in cited if tag in new] + changes
    typed_rows = [(section, typed + (f" (twice)" if count > 1 else ""),
                   " \\m ".join(ts) + (f". {remark}" if remark else ""))
                  for typed, count, section, ts, remark in TYPED]
    section = [note_para(intro_text)] + entries + [
        heading_para("Source Manager changes for the highlighted entries"),
        note_para("Make the changes under References > Manage Sources > Edit, then References > Update Citations and "
                  "Bibliography. The citations of Fang et al., Feng et al., Piovezan et al., Tibshirani and López de "
                  "Prado change with their sources."),
        table_like(log_table, ("Tag", "Work", "Change"), changes),
        heading_para("Typed citations in the thesis's Chapter 3"),
        note_para("Each should be a citation field with the Source Manager tags shown (several tags in one bracket "
                  "are joined with \\m in the field code)."),
        table_like(log_table, ("Section", "Typed citation", "Tags"), typed_rows),
    ]
    anchor = head
    for el in section:
        anchor.addnext(el)
        anchor = el

    # review note at the top, and one amendment-log row
    note = next(p for p in body.iter(qn("w:p")) if text(p).startswith("Review note:"))
    old = "The reference appendix contains only additions and corrections; the amendment log records deletions and reasons."
    if old in text(note):
        Editor(False).replace(note, old, "The reference list after the chapter gives every work the chapter cites, "
                              "with additions and corrections highlighted; the amendment log records deletions and "
                              "reasons.")
    for tr in log_table.findall(qn("w:tr")):
        cells = tr.findall(qn("w:tc"))
        if text(cells[0]) == LOG_SECTION and text(cells[1]).startswith(LOG_MARK):
            log_table.remove(tr)
    row = copy.deepcopy(log_table.findall(qn("w:tr"))[-1])
    for tc, value in zip(row.findall(qn("w:tc")), (
            LOG_SECTION,
            f"{LOG_MARK}: all {len(cited)} works cited in the thesis's Chapter 3, replacing 'Reference additions and "
            f"corrections'. {missing} works are missing from the thesis reference list and {corrected} are printed "
            "with wrong or incomplete details; the Source Manager changes and the typed citations to convert are "
            "listed after the reference list. The missing works are added to this document's Source Manager.",
            "Author: reference list for Chapter 3 (4 November 2026). Details checked against the publishers' "
            "records.")):
        set_cell(tc, value)
    log_table.append(row)

    added = add_new_sources(document)
    document.save(OUTPUT)
    print(f"{OUTPUT.name}: {len(cited)} works ({as_printed} as printed, {corrected} corrected, {missing} missing); "
          f"{len(changes)} Source Manager rows, {len(typed_rows)} typed-citation rows; sources added: {added}")


if __name__ == "__main__":
    build()
