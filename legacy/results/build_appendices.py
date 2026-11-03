"""Build the appendices of the thesis (A, B and C) as one Word document.

Appendix A (feature definitions) is taken unchanged from the thesis. Appendix B (feature selection,
build_appendix_b.py) and Appendix C (CPCV and walk-forward formulation, build_appendix_c.py) replace the thesis's
'work in progress' placeholders. Table B.8, the economic classification of the 50 modelling features, is taken
from the Chapter 3 amendment document, where it was moved out of Section 3.6. The document starts from a
snapshot of the thesis, so it keeps the thesis's styles, page set-up and Source Manager; an amendment log is
added at the end.

Usage: python build_appendices.py [out_dir] [thesis_snapshot.docx] [chapter3_v2_snapshot.docx]
(Word keeps open files locked; snapshot them with cp first.)
"""
import copy
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from build_minimal_chapter3 import accept_revisions, add_log, strip_comments  # before build_appendix_b
from build_appendix_b import content as appendix_b
from build_appendix_c import content as appendix_c

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parents[1]
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
CHAPTER3 = (Path(sys.argv[3]) if len(sys.argv) > 3
            else RESULTS / "Chapter3_Methodology_Submission_Amendments_Highlighted_new_v2.docx")
OUTPUT = OUT_DIR / "Appendices_Amendments_Highlighted.docx"


def text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def style(el):
    s = el.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return s.get(qn("w:val")) if s is not None else ""


def appendix_heading(children, title):
    """The appendix heading itself, not its entry in the table of contents."""
    hits = [el for el in children if el.tag == qn("w:p") and style(el).startswith("Appendix")
            and text(el).strip().startswith(title)]
    assert len(hits) == 1, title
    return hits[0]


def new_page(heading):
    """Start the appendix on a new page (the thesis's appendix headings follow on without a break)."""
    ppr = heading.find(qn("w:pPr"))
    if ppr.find(qn("w:pageBreakBefore")) is None:
        ppr.find(qn("w:pStyle")).addnext(OxmlElement("w:pageBreakBefore"))


def table_b8(path):
    """Table B.8 (header 'Feature Category | Feature | Economic Rationale | Empirical Findings') from Chapter 3."""
    body = Document(path).element.body
    hits = [t for t in body.iter(qn("w:tbl")) if text(t.find(qn("w:tr"))).startswith("Feature Category")]
    assert len(hits) == 1 and len(hits[0].findall(qn("w:tr"))) == 51, "Table B.8 not found"
    return copy.deepcopy(hits[0])


def build():
    document = Document(SOURCE)
    body = document.element.body
    accept_revisions(body)
    strip_comments(body)

    children = list(body)
    head_a = appendix_heading(children, "Appendix A:")
    head_b = appendix_heading(children, "Appendix B:")
    head_c = appendix_heading(children, "Appendix C:")
    start, end_a = children.index(head_a), children.index(head_b)
    assert start < end_a < children.index(head_c)
    # the section break that closes the appendices becomes the document's final section (page set-up, footers)
    brk = next(el for el in children[start:] if el.tag == qn("w:p") and el.find(qn("w:pPr")) is not None
               and el.find(qn("w:pPr")).find(qn("w:sectPr")) is not None)
    appendix_sect = copy.deepcopy(brk.find(qn("w:pPr")).find(qn("w:sectPr")))
    placeholders = [text(el).strip() for el in children[end_a + 1:] if el is not head_c and el.tag != qn("w:sectPr")
                    and text(el).strip()]
    assert all(t in ("[work in progress]", "B.1. Clusters and Cluster Representatives:") for t in placeholders), \
        f"Appendix B or C has content in the thesis: {placeholders}"

    keep = set(map(id, children[start:end_a + 1]))           # Appendix A and the Appendix B heading
    for child in children:
        if id(child) not in keep and child.tag != qn("w:sectPr"):
            body.remove(child)
    final = body.find(qn("w:sectPr"))
    final.addprevious(appendix_sect)
    body.remove(final)

    new_page(head_b)
    new_page(head_c)
    b8 = table_b8(CHAPTER3)
    appendix_b(document, b8)
    body.find(qn("w:sectPr")).addprevious(head_c)
    runs = appendix_c(document)

    log = []
    note = lambda section, change, reason: log.append((section, change, reason))
    note("All", "The three appendices in one document, in the thesis's styles and page set-up: Appendix A from the "
         "thesis, Appendices B and C in place of the 'work in progress' placeholders. Replace the thesis's "
         "appendix section with it. Text that is new since the separate Appendix B (25 October) and Appendix C "
         "(2 November) files is highlighted; those files are superseded.", "Author: put the appendix together in "
         "one document.")
    note("A", "Unchanged from the thesis.", "–")
    note("B, C", "Appendices B and C start on a new page (paragraph setting 'Page break before' on their headings); "
         "in the thesis each appendix follows on from the previous one.", "Each appendix, like each chapter, starts "
         "on a new page.")
    note("B", "The placeholder (an incomplete Table B.1 'Clusters and Cluster Representatives' and 'work in "
         "progress') is replaced by Sections B.1–B.7. B.1–B.5 are unchanged in substance from the separate "
         "Appendix B file. The clustering input (featureset.csv) was not affected by the correction of the factor "
         "return data, so its results stand.", "Complete Appendix B.")
    note("B.2", "'its category in Table 1' becomes 'Table 2' (the summary by category in Section 3.6 is Table 2 in "
         "the Chapter 3 amendments, and the thesis text already says 'Table 2'); the categories in Table B.3 now come "
         "from Table B.8, so they use the names of Table 2 ('Macroeconomic', 'Sentiment indicators' instead of "
         "'Macro-economic', 'Sentiment Indicators'). '(Equation 7)' deleted: Chapter 3 gives the MI distance in words, and Equation 7 is the logit. 'does "
         "not appear among the 20 most influential features of any model in Chapter 4' becomes 'is not among the ten "
         "most influential features', the number Chapter 4 reports; checked against the SHAP importances of the "
         "nine models after the rerun (PPP.ZA.Index_QS is not in the top 20 of any).", "Cross-references; accuracy.")
    note("B.6", "New section 'Economic classification of the modelling features' with Table B.8, the detailed "
         "feature table moved out of Section 3.6 (Chapter 3 amendments: 'add it to Appendix B as Table B.8, in a "
         "new section after B.5'). It is copied from the Chapter 3 amendments with their highlights, which mark the "
         "changes from the submitted Table 1: the category names made consistent with Table 2, the ZAR18M.Curncy_QS "
         "description (a forward-points series, not implied volatility) and the Daniel and Moskowitz (2016) evidence. "
         "'Summary' becomes B.7.", "Supervisor's suggestion to "
         "move the feature table to an appendix; Section 3.6 refers to Table B.8.")
    note("C", "Regenerated from the model outputs of the rerun on the corrected data; Sections C.1–C.4 are unchanged "
         "from the 2 November file except the opening sentence, which now lists C.5.", "Complete Appendix C.")
    note("C.5", "New section 'Sensitivity to the hyperparameters', Table C.7 and Figure C.1: every setting of each "
         f"search grid held fixed for all 156 predictions ({sum(len(v['grid']) for v in runs.values())} walk-forward "
         "runs, sensitivity_fixed_settings.py). Checks: the two settings that were selected at every tuning point "
         "(logistic regression, value vs quality; GAM, momentum vs value) reproduce the Chapter 4 results exactly; "
         "the selected GAM value vs quality forecasts equal, period by period, the fixed runs of the settings in "
         "use. Every number in the text is asserted by the builder.", "Author, 2 November: add the sensitivity "
         "analysis to the appendix, for all the models.")
    note("Other chapters", "Section C.5 qualifies two statements elsewhere. The GAM's weak value vs quality result "
         "comes from the setting selected at the first tuning point (with most fixed settings it reaches about 0.58, "
         "close to logistic regression), so 'the GAM adds nothing' in Section 4.5 could say that the GAM's tuning, "
         "not only its form, explains its result. XGBoost's lead for value vs quality and momentum vs value holds for "
         "most of its settings, which Section 4.5 or Chapter 5 can cite (Appendix C.5). Section 3.8.1 could add that "
         "Appendix C also reports the sensitivity to the hyperparameters.", "For the author to decide; not changed "
         "here (one chapter at a time).")
    add_log(document, log)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
