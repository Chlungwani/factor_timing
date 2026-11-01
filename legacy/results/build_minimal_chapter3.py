"""Build the Chapter 3 (Methodology) amendment documents.

Starts from a copy of the current thesis, accepts the tracked changes already in
the file, removes comments, keeps only Chapter 3, and applies the amendments in
place so that equations, figures, tables and styles are preserved.

Outputs
  Chapter3_Methodology_Minimal_Amendments_Highlighted.docx  amendments in yellow
                                                            plus an amendment log
  Chapter3_Methodology_Revised.docx                         same text, no highlight
"""

import copy
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from lxml import etree

ROOT = Path(r"C:\Users\ccwis\OneDrive\Documents\1. MEng - Data Science\Project\factor_timing")
SOURCE = ROOT / ".tmp_chapter3_minimal" / "thesis_current.docx"
RESULTS = ROOT / "legacy" / "results"
FIGURES = RESULTS / "chapter3_final_figures"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
OUTPUTS = {
    True: OUT_DIR / "Chapter3_Methodology_Amendments_Highlighted.docx",
    False: OUT_DIR / "Chapter3_Methodology_Amendments_Clean.docx",
}
FIRST, LAST = 302, 460          # paragraph indices of Chapter 3 ("Chapter" label to last body paragraph)
FIGURE_WIDTH = Inches(6.27)
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# ---------------------------------------------------------------- cleaning --

def accept_revisions(body):
    for tag in ("w:del", "w:moveFrom"):
        for el in list(body.iter(qn(tag))):
            parent = el.getparent()
            if parent is not None and parent.tag == qn("w:rPr"):
                parent.remove(el)          # deleted paragraph mark: keep paragraph
            elif parent is not None:
                parent.remove(el)
    for tag in ("w:ins", "w:moveTo"):
        for el in list(body.iter(qn(tag))):
            parent = el.getparent()
            if parent.tag in (qn("w:rPr"), qn("w:trPr")):
                parent.remove(el)
                continue
            index = parent.index(el)
            for child in list(el):
                parent.insert(index, child)
                index += 1
            parent.remove(el)
    for tag in ("w:pPrChange", "w:rPrChange", "w:sectPrChange", "w:tblPrChange",
                "w:trPrChange", "w:tcPrChange", "w:tblGridChange", "w:numberingChange",
                "w:moveFromRangeStart", "w:moveFromRangeEnd", "w:moveToRangeStart",
                "w:moveToRangeEnd"):
        for el in list(body.iter(qn(tag))):
            el.getparent().remove(el)


def strip_comments(body):
    for tag in ("w:commentRangeStart", "w:commentRangeEnd"):
        for el in list(body.iter(qn(tag))):
            el.getparent().remove(el)
    for ref in list(body.iter(qn("w:commentReference"))):
        run = ref.getparent()
        run.getparent().remove(run)


def unfield(paragraph_el):
    """Replace complex fields in a paragraph by their current result text."""
    runs = list(paragraph_el.iter(qn("w:r")))
    state = None
    for run in runs:
        fld = run.find(qn("w:fldChar"))
        if fld is not None:
            kind = fld.get(qn("w:fldCharType"))
            state = {"begin": "instr", "separate": "result", "end": None}[kind]
            run.getparent().remove(run)
            continue
        if state == "instr":
            run.getparent().remove(run)


# ----------------------------------------------------------------- editing --

class Editor:
    def __init__(self, highlight):
        self.highlight = highlight

    def mark(self, run_el):
        rpr = run_el.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            run_el.insert(0, rpr)
        for el in rpr.findall(qn("w:highlight")):
            rpr.remove(el)
        if self.highlight:
            hl = OxmlElement("w:highlight")
            hl.set(qn("w:val"), "yellow")
            rpr.append(hl)

    @staticmethod
    def _text_nodes(p_el):
        return [t for t in p_el.iter(qn("w:t"))]

    def new_run(self, template_run, text, italic=None):
        run = OxmlElement("w:r")
        if template_run is not None and template_run.find(qn("w:rPr")) is not None:
            rpr = copy.deepcopy(template_run.find(qn("w:rPr")))
            for el in rpr.findall(qn("w:highlight")) + rpr.findall(qn("w:vertAlign")):
                rpr.remove(el)
            run.append(rpr)
        if italic is not None:
            rpr = run.find(qn("w:rPr"))
            if rpr is None:
                rpr = OxmlElement("w:rPr"); run.append(rpr)
            for el in rpr.findall(qn("w:i")):
                rpr.remove(el)
            if italic:
                rpr.append(OxmlElement("w:i"))
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        run.append(t)
        self.mark(run)
        return run

    def replace(self, p, old, new):
        """Replace `old` (which may span runs) by highlighted `new`."""
        p_el = p._p if hasattr(p, "_p") else p
        nodes = self._text_nodes(p_el)
        full = "".join(n.text or "" for n in nodes)
        count = full.count(old)
        if count != 1:
            raise ValueError(f"expected one match for {old!r}, found {count} in {full[:120]!r}")
        start = full.index(old)
        end = start + len(old)
        pos = 0
        first = last = None
        for n in nodes:
            length = len(n.text or "")
            n_start, n_end = pos, pos + length
            if first is None and n_end > start:
                first = (n, start - n_start)
            if n_end >= end and last is None and n_start < end:
                last = (n, end - n_start)
            pos = n_end
        first_node, first_cut = first
        last_node, last_cut = last
        first_run = first_node.getparent()
        suffix = (last_node.text or "")[last_cut:]
        # clear covered text
        on = False
        for n in nodes:
            if n is first_node:
                on = True
            if on and n is not first_node:
                n.text = ""
            if n is last_node:
                break
        first_node.text = (first_node.text or "")[:first_cut]
        first_node.set(qn("xml:space"), "preserve")
        anchor = first_run
        if new:
            run = self.new_run(first_run, new)
            anchor.addnext(run)
            anchor = run
        if last_node is first_node:
            if suffix:
                tail = copy.deepcopy(first_run)
                for t in tail.findall(qn("w:t")):
                    t.text = ""
                tail.find(qn("w:t")).text = suffix
                tail.find(qn("w:t")).set(qn("xml:space"), "preserve")
                anchor.addnext(tail)
        else:
            last_node.text = suffix
            last_node.set(qn("xml:space"), "preserve")

    def append(self, p, text):
        p_el = p._p if hasattr(p, "_p") else p
        runs = [r for r in p_el.iter(qn("w:r")) if r.find(qn("w:t")) is not None]
        template = runs[-1] if runs else None
        p_el.append(self.new_run(template, text))

    def set_text(self, p, text):
        """Replace all text runs of a paragraph (no math expected)."""
        p_el = p._p if hasattr(p, "_p") else p
        runs = [r for r in p_el.iter(qn("w:r")) if r.find(qn("w:t")) is not None]
        template = copy.deepcopy(runs[0]) if runs else None
        for r in list(p_el.iter(qn("w:r"))):
            if r.find(qn("w:drawing")) is None and r.getparent() is not None:
                r.getparent().remove(r)
        for tag in (f"{{{M_NS}}}oMathPara", f"{{{M_NS}}}oMath"):
            for m in list(p_el.iter(tag)):
                if m.getparent() is not None:
                    m.getparent().remove(m)
        for h in list(p_el.iter(qn("w:hyperlink"))):
            if not list(h):
                h.getparent().remove(h)
        p_el.append(self.new_run(template, text))

    def insert_after(self, anchor, text, like=None):
        """Insert a new paragraph after `anchor`, formatted like `like`."""
        anchor_el = anchor._p if hasattr(anchor, "_p") else anchor
        like_el = (like._p if hasattr(like, "_p") else like) if like is not None else anchor_el
        p_el = OxmlElement("w:p")
        if like_el.find(qn("w:pPr")) is not None:
            p_el.append(copy.deepcopy(like_el.find(qn("w:pPr"))))
        runs = [r for r in like_el.iter(qn("w:r")) if r.find(qn("w:t")) is not None]
        p_el.append(self.new_run(runs[0] if runs else None, text))
        anchor_el.addnext(p_el)
        return p_el

    def math_text(self, p_el, old, new):
        hits = [t for t in p_el.iter(f"{{{M_NS}}}t") if t.text == old]
        if len(hits) != 1:
            raise ValueError(f"math text {old!r} found {len(hits)} times")
        hits[0].text = new
        mrun = hits[0].getparent()
        rpr = mrun.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            mrun.insert(0 if mrun[0].tag != f"{{{M_NS}}}rPr" else 1, rpr)
        if self.highlight:
            hl = OxmlElement("w:highlight"); hl.set(qn("w:val"), "yellow"); rpr.append(hl)

    def cell(self, cell, text):
        paragraph = cell.paragraphs[0]
        for extra in cell.paragraphs[1:]:
            extra._p.getparent().remove(extra._p)
        self.set_text(paragraph, text)

    def picture(self, p, image_path):
        p_el = p._p
        for r in list(p_el.iter(qn("w:r"))):
            if r.find(qn("w:drawing")) is not None or r.find(qn("w:pict")) is not None:
                r.getparent().remove(r)
        run = p.add_run()
        run.add_picture(str(image_path), width=FIGURE_WIDTH)


def remove(p):
    el = p._p if hasattr(p, "_p") else p
    el.getparent().remove(el)


# --------------------------------------------------------------- equations --

MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
_XSLT = []


def mathml_to_omml(mathml, highlight):
    """Convert a MathML fragment to an m:oMath element styled like the thesis equations."""
    if not _XSLT:
        _XSLT.append(etree.XSLT(etree.parse(str(MML2OMML))))
    source = etree.fromstring(f'<math xmlns="http://www.w3.org/1998/Math/MathML">{mathml}</math>')
    omath = _XSLT[0](source).getroot()
    for mrun in omath.iter(f"{{{M_NS}}}r"):
        for old in mrun.findall(qn("w:rPr")):
            mrun.remove(old)
        rpr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        fonts.set(qn("w:ascii"), "Cambria Math")
        fonts.set(qn("w:hAnsi"), "Cambria Math")
        rpr.append(fonts)
        for tag in ("w:sz", "w:szCs"):
            size = OxmlElement(tag); size.set(qn("w:val"), "24"); rpr.append(size)
        if highlight:
            hl = OxmlElement("w:highlight"); hl.set(qn("w:val"), "yellow"); rpr.append(hl)
        mrpr = mrun.find(f"{{{M_NS}}}rPr")
        if mrpr is not None:
            mrpr.addnext(rpr)
        else:
            mrun.insert(0, rpr)
    return omath


def subscript_text(p_el, token):
    """Render a 'C_c'-style token in a paragraph as italic C with an italic subscript c."""
    base, sub = token.split("_")
    for run in list(p_el.iter(qn("w:r"))):
        t = run.find(qn("w:t"))
        if t is None or token not in (t.text or ""):
            continue
        before, after = t.text.split(token, 1)
        t.text = before
        t.set(qn("xml:space"), "preserve")
        anchor = run
        for text, vert in ((base, None), (sub, "subscript"), (after, "tail")):
            piece = copy.deepcopy(run)
            piece.find(qn("w:t")).text = text
            rpr = piece.find(qn("w:rPr"))
            if vert != "tail":
                rpr.append(OxmlElement("w:i"))
            if vert == "subscript":
                va = OxmlElement("w:vertAlign"); va.set(qn("w:val"), "subscript"); rpr.append(va)
            anchor.addnext(piece)
            anchor = piece
        return


def set_equation_number(editor, tbl, number):
    """Replace the text of the number cell of an equation table."""
    number_cell = tbl.findall(f".//{qn('w:tc')}")[1]
    paragraph = number_cell.find(qn("w:p"))
    editor.set_text(paragraph, f"({number})")


def equation_table(editor, template_tbl, mathml, number):
    tbl = copy.deepcopy(template_tbl)
    para = tbl.find(f".//{{{M_NS}}}oMathPara")
    for child in list(para):
        para.remove(child)
    para.append(mathml_to_omml(mathml, editor.highlight))
    set_equation_number(editor, tbl, number)
    return tbl


def p_hat(arg):
    return f'<mover accent="true"><mi>p</mi><mo>&#x0302;</mo></mover><mo>(</mo>{arg}<mo>)</mo>'


I_TILDE = '<mover accent="true"><mi>I</mi><mo>&#x0303;</mo></mover>'
SUM_B = '<munderover><mo>&#x2211;</mo><mrow><mi>{0}</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover>'
EQ_MI = (
    '<mi>I</mi><mo>(</mo><mi>X</mi><mo>;</mo><mi>Y</mi><mo>)</mo><mo>=</mo>'
    + SUM_B.format("a") + '<mrow>' + SUM_B.format("b") + '<mrow>'      # summands wrapped in mrow so the
    + p_hat('<mi>a</mi><mo>,</mo><mi>b</mi>')                            # hat converts to an accent
    + '<mi>ln</mi><mo>&#x2061;</mo><mfenced><mfrac><mrow>' + p_hat('<mi>a</mi><mo>,</mo><mi>b</mi>')
    + '</mrow><mrow>' + p_hat('<mi>a</mi>') + p_hat('<mi>b</mi>') + '</mrow></mfrac></mfenced>'
    + '</mrow></mrow>'
)
EQ_NMI = (
    I_TILDE + '<mo>(</mo><mi>X</mi><mo>;</mo><mi>Y</mi><mo>)</mo><mo>=</mo>'
    '<mfrac><mrow><mi>I</mi><mo>(</mo><mi>X</mi><mo>;</mo><mi>Y</mi><mo>)</mo></mrow>'
    '<mrow><mi>max</mi><mo>&#x2061;</mo><mfenced open="{" close="}"><mrow><mi>H</mi><mo>(</mo><mi>X</mi><mo>)</mo>'
    '<mo>,</mo><mi>H</mi><mo>(</mo><mi>Y</mi><mo>)</mo></mrow></mfenced></mrow></mfrac><mo>,</mo><mspace width="1em"/>'
    '<mi>H</mi><mo>(</mo><mi>X</mi><mo>)</mo><mo>=</mo><mo>&#x2212;</mo>' + SUM_B.format("a")
    + '<mrow>' + p_hat('<mi>a</mi>') + '<mi>ln</mi><mo>&#x2061;</mo>' + p_hat('<mi>a</mi>') + '</mrow>'
)
EQ_DIST = (
    '<mi>d</mi><mo>(</mo><mi>X</mi><mo>,</mo><mi>Y</mi><mo>)</mo><mo>=</mo><mn>1</mn><mo>&#x2212;</mo>'
    + I_TILDE + '<mo>(</mo><mi>X</mi><mo>;</mo><mi>Y</mi><mo>)</mo>'
)
EQ_REP = (
    '<msubsup><mi>x</mi><mi>c</mi><mo>*</mo></msubsup><mo>=</mo>'
    '<munder><mrow><mi>arg</mi><mo>&#x2009;</mo><mi>min</mi></mrow>'
    '<mrow><mi>j</mi><mo>&#x2208;</mo><msub><mi>C</mi><mi>c</mi></msub></mrow></munder>'
    '<mfrac><mn>1</mn><mrow><mo>|</mo><msub><mi>C</mi><mi>c</mi></msub><mo>|</mo></mrow></mfrac>'
    '<munder><mo>&#x2211;</mo><mrow><mi>k</mi><mo>&#x2208;</mo><msub><mi>C</mi><mi>c</mi></msub></mrow></munder>'
    '<mrow><mi>d</mi><mo>(</mo><msub><mi>X</mi><mi>j</mi></msub><mo>,</mo><msub><mi>X</mi><mi>k</mi></msub>'
    '<mo>)</mo></mrow>'
)
NEW_EQUATIONS = 4                 # inserted as (4)-(7); later equations shift by this amount


def set_heading_start(document, start):
    numbering = document.part.numbering_part.element
    for num in numbering.findall(qn("w:num")):
        if num.get(qn("w:numId")) == "4":
            override = OxmlElement("w:lvlOverride")
            override.set(qn("w:ilvl"), "0")
            start_el = OxmlElement("w:startOverride")
            start_el.set(qn("w:val"), str(start))
            override.append(start_el)
            num.append(override)
            return
    raise ValueError("heading numbering definition not found")


# ------------------------------------------------------------------- build --

def build(highlight):
    document = Document(SOURCE)
    body = document.element.body
    P = {i: p for i, p in enumerate(document.paragraphs) if FIRST <= i <= LAST}
    T = {i: t for i, t in enumerate(document.tables)}

    accept_revisions(body)
    strip_comments(body)

    # keep only Chapter 3
    keep = False
    index = -1
    first_el, last_el = P[FIRST]._p, P[LAST]._p
    for child in list(body):
        if child is first_el:
            keep = True
        if child.tag == qn("w:sectPr"):
            continue
        if not keep:
            body.remove(child)
        if child is last_el:
            keep = "done"
        elif keep == "done":
            body.remove(child)
    # the section properties of the chapter's own section, if any, stay on the body
    set_heading_start(document, 3)

    e = Editor(highlight)
    log = []

    def note(section, change, reason):
        log.append((section, change, reason))

    # 3.1 Research design and empirical framework ------------------------------
    e.replace(P[305], "a  supervised", "a supervised")
    e.replace(P[305], "to ensure statistical robustness and economic relevance",
              "designed to limit look-ahead bias and overfitting")
    note("3.1", "Replaced 'to ensure statistical robustness and economic relevance'.",
         "The design limits bias; it cannot ensure robustness, and economic value is not tested.")
    e.replace(P[306], " and reduces look-ahead bias",
              ". Because the labels are forward-looking by construction, the overlap they create with "
              "later observations is handled by purging during validation (Section 3.8)")
    note("3.1", "Removed the claim that trend scanning reduces look-ahead bias.",
         "Trend-scanning labels use future returns; leakage is controlled by purging, not by the labelling.")
    e.replace(P[307], "more than 450", "359")
    note("3.1", "Candidate feature count 'more than 450' changed to 359.",
         "featureset.csv contains 359 candidate variables; Section 3.6 already says 'over 300'.")
    e.replace(P[313], "Hyperparameters are optimised using CPCV, a procedure specifically designed for financial "
                      "time series to eliminate temporal leakage",
              "Hyperparameters are optimised using combinatorial purged cross-validation (CPCV), a procedure "
              "designed for financial time series to limit temporal leakage")
    e.replace(P[313], "rolling walk-forward", "expanding-window walk-forward")
    e.replace(P[313], "Parameters are re-tuned annually to account for evolving market regimes.",
              "Hyperparameters are re-tuned after every 52 weekly predictions (approximately annually) to "
              "account for evolving market regimes.")
    note("3.1", "Rolling window corrected to expanding window; retuning stated as every 52 predictions.",
         "Final configuration (lookback=None, retune_every=52) in all three model notebooks.")
    e.replace(P[314], "(probability calibration).",
              "(probability quality), supplemented by confusion matrices that show how classification errors "
              "are distributed between the two classes.")
    note("3.1", "Added confusion matrices to the evaluation summary.", "Supervisor comment; Chapter 4 reports them.")
    e.replace(P[315], "feature importance is analysed using Shapley values",
              "global feature importance is analysed using Shapley values")
    e.replace(P[316], "along statistical, economic, and structural dimensions",
              "along statistical and structural dimensions")
    e.replace(P[316], "directly addresses", "is designed to address")
    note("3.1", "Removed 'economic' dimension and softened 'directly addresses'.",
         "No portfolio or after-cost test is performed (Section 3.9 states this is out of scope).")

    # 3.2 Factor definitions ------------------------------------------------------
    e.replace(P[321], "earnings revisions incorporate", "earnings revisions, which incorporate")

    # 3.3 Motivation ------------------------------------------------------------------
    neutral = e.insert_after(
        P[330],
        "Because each target is the difference between two factor returns, it does not depend on the choice "
        "of benchmark: for factors A and B and any benchmark return, (r_A \u2212 r_bm) \u2212 (r_B \u2212 r_bm) "
        "= r_A \u2212 r_B. The trend-scanning labels are therefore identical whether the factors are measured "
        "in raw or in active (benchmark-relative) terms.", like=P[330])
    note("3.3", "New paragraph: the pairwise target is benchmark-neutral.",
         "Supervisor's active-return request; algebraic identity verified on factor_returns.csv.")
    e.append(P[331], " From an investment-management perspective, the direction of relative leadership is a "
                     "sufficient signal: a tilt between two styles can be set from the sign of the expected "
                     "relative trend, without forecasting the magnitude of each period's return.")
    note("3.3", "Added the investment-management rationale for a directional target.", "Supervisor comment.")

    # 3.4 Data sources ----------------------------------------------------------------
    e.replace(P[335], "January 2007 to March 2025", "March 2007 to March 2025")
    e.replace(P[336], "over 300 predictors", "359 predictors")
    e.replace(P[336], "(i.e. 929 weekly observation)", "(919 labelled weekly observations per factor pair)")
    note("3.4.1", "Candidate count (359) and modelling sample (919 per pair) made exact.",
         "featureset.csv; labelled sample after dropping unresolved end labels.")
    for i in range(337, 346):
        remove(P[i])
    note("3.4", "Removed empty paragraphs and an empty sub-heading left by an earlier move.",
         "They would print as a blank numbered heading.")

    # 3.5 Trend scanning --------------------------------------------------------------
    e.replace(P[347], "The aim is to identifying periods when a factor outperforms or underperforms relative to "
                      "other style factors.",
              "The aim is to identify periods in which one factor outperforms or underperforms another.")
    e.replace(P[349], "For each distinct pair of factors and the weekly returns series is defined as:",
              "For each distinct pair of factors, the weekly relative return series is defined as:")
    e.math_text(P[351]._p, "H=12 ", "H=11 ")
    e.replace(P[351], "weeks. T", "weekly observations. T")
    e.append(P[351], " Each candidate window starts at the observation date t and contains h weekly observations, "
                     "so the longest window uses cumulative relative returns up to t + 10. Windows with fewer than "
                     "three observations do not yield a defined slope t-statistic and are disregarded.")
    note("3.5", "Maximum window corrected from H = 12 to H = 11 observations (returns up to t + 10).",
         "trend_scanning.ipynb uses span = (1, 12), i.e. windows of 1 to 11 observations.")
    e.append(P[355], "Every observation receives a label: no minimum t-statistic is imposed.")
    e.replace(P[356], "Because labels incorporate forward-looking information up to 12 weeks,",
              "Because each label uses cumulative relative returns up to ten weeks after the observation date,")
    e.replace(P[356], "To eliminate leakage, a purge gap of 12 weeks is imposed during cross-validation, ensuring "
                      "that no training observation contains information overlapping with the validation window",
              "To limit leakage, a 12-week purge separates training observations from each validation or test "
              "observation (with an additional 12-week embargo in cross-validation; Section 3.8), so that no "
              "training label overlaps the evaluation window")
    e.append(P[356], " Labels that cannot be resolved because the sample ends inside the look-ahead window (the "
                     "final ten weekly observations) are discarded rather than forward-filled.")
    note("3.5", "Purge/embargo description aligned with the code; unresolved end-of-sample labels now stated as dropped.",
         "PURGE_GAP = 12 in CV and walk-forward; load_data() drops unlabelled rows.")
    e.picture(P[358], FIGURES / "figure_trend_scanning_labels.png")
    e.replace(P[359], "Trend-scanning results each factor pairs", "Trend-scanning labels for each factor pair")
    unfield(P[361]._p)
    e.replace(P[361], "Figure 2 The black line plots", "In Figure 1, the black line plots")
    e.replace(P[361], "momentum and value (top), value and quality (middle), and momentum and quality (bottom)",
              "momentum versus value (top), value versus quality (middle), and momentum versus quality (bottom)")
    e.replace(P[361], "Uptrend (green) and downtrend (red)", "Uptrend (blue) and downtrend (orange) labels")
    e.append(P[361], " The final weeks, whose labels cannot be resolved, are unshaded. The cumulative series is "
                     "shown, rather than rolling 12-week relative returns, because it is the series to which trend "
                     "scanning is applied.")
    note("3.5", "Figure 1 regenerated from the labels used by the models; cross-reference corrected (was 'Figure 2').",
         "Old figure showed 944 labels including forward-filled end labels; supervisor asked about rolling returns.")

    # 3.6 Feature selection -----------------------------------------------------------
    remove(P[365])          # emptied when its text was moved to Section 3.4
    e.replace(P[370], "Hierarchical agglomeration clustering applied to the features space",
              "Hierarchical agglomerative clustering applied to the feature space")
    unfield(P[372]._p)
    template_eq = T[1]._tbl
    blocks = [
        ("p", "Mutual information is estimated for every pair of candidate features. Each feature is first "
              "discretised into B = 20 equal-frequency (quantile) bins, and the mutual information between two "
              "features X and Y is estimated from the resulting joint frequency table (Cover & Thomas, 2006):"),
        ("eq", EQ_MI, 4),
        ("p", "where p̂(a, b) is the proportion of weeks in which X falls in bin a and Y in bin b, and "
              "p̂(a) and p̂(b) are the corresponding marginal proportions. Mutual information is zero "
              "when the binned series are independent, but its upper bound depends on the entropy of each "
              "feature. It is therefore normalised by the larger of the two marginal entropies:"),
        ("eq", EQ_NMI, 5),
        ("p", "where H(X) is the entropy of the binned feature X, and H(Y) is defined analogously. The "
              "normalised measure lies between zero and one, and equals one only when the two binned series "
              "determine each other exactly. It is converted into a distance:"),
        ("eq", EQ_DIST, 6),
        ("p", "so that features that share more information lie closer together. Because quantile bins depend "
              "only on the ranks of a series, the distance is unaffected by differences in scale and by monotonic "
              "transformations of a feature. Quantile binning also gives every feature approximately uniform "
              "marginal frequencies, so the upward finite-sample bias of this plug-in estimator is of similar "
              "size for every pair. The distance matrix is computed for 343 candidate features, using the 680 "
              "weekly observations from 26 March 2012 to 31 March 2025 on which all of them are available after "
              "forward filling. One of the 344 features that pass the missing-value screen "
              "(LocalGlobalPEFwdSpreadQS) is omitted, because the implementation treats the final column of the "
              "feature file as a target. The median pairwise distance is 0.85, so most pairs of candidate "
              "features share little information."),
    ]
    anchor = P[366]._p
    for block in blocks:
        if block[0] == "p":
            anchor = e.insert_after(anchor, block[1], like=P[366])
        else:
            tbl = equation_table(e, template_eq, block[1], block[2])
            anchor.addnext(tbl)
            anchor = tbl
    e.replace(P[367], "An MI matrix is computed for all pairwise combinations of candidate predictors. This matrix "
                      "is normalised and transformed into a distance metric such that features exhibiting high "
                      "informational similarity correspond to smaller distances. ", "")
    e.replace(P[367], "Hierarchical agglomerative clustering is then applied using Ward linkage, which merges "
                      "clusters in a manner that minimises within-cluster variance at each step",
              "Hierarchical agglomerative clustering is then applied using Ward linkage to the rows of the "
              "distance matrix. Each feature is thus represented by its profile of MI distances to all other "
              "features, and at each step the two clusters whose merger produces the smallest increase in the "
              "within-cluster variance of these profiles are joined")
    e.replace(P[368], " A maximum cluster distance threshold is imposed to determine the final partition of the "
                      "feature space.",
              "The dendrogram is then cut at a fixed linkage height to determine the final partition of the "
              "feature space. Silhouette scores computed on the MI distances are low for every partition "
              "considered (0.04 at 32 clusters, rising only to 0.10 at 100 clusters), which indicates that the "
              "candidate features do not fall into sharply separated groups. The clustering is therefore used to "
              "remove redundancy rather than to identify distinct families of features, and the cut height is "
              "set by judgement, balancing a compact predictor set against the merging of dissimilar features.")
    note("3.6", "Mutual-information feature selection specified formally: quantile binning, MI estimator (Eq. 4), "
                "entropy normalisation (Eq. 5), distance (Eq. 6), estimation sample and feature count, Ward "
                "linkage on distance profiles, and the evidence behind the cut height.",
         "feature_selection.ipynb (normalized_mi, mi_distance_matrix, cluster_features); clustering reproduced "
         "independently: 32 clusters at height 2 and all 32 saved representatives recovered.")
    note("3.6", "Disclosed that LocalGlobalPEFwdSpreadQS is omitted from the clustering (344 screened, 343 clustered).",
         "feature_selection.ipynb sets X = data.iloc[:, :-1], treating the last feature column as a target.")
    e.replace(P[372], "Figure 3 Represents", "Figure 2 represents")
    e.replace(P[372], "The maximum distance for clusters selection is set at 2",
              "The maximum linkage height for cluster selection is set at 2")
    e.replace(P[372], "resulting in a total of 32 clusters.",
              "resulting in a total of 32 clusters (cut heights of 1.5 and 2.5 would give 65 and 18 clusters). "
              "Cluster sizes range from 2 to 48 features, with a median of 5.5.")
    e.replace(P[372], "Within each cluster, the feature exhibiting the lowest average distance to other members "
                      "is retained.",
              "Within each cluster, the feature with the lowest average distance to the other members is "
              "retained, as defined in Equation (7).")
    e.append(P[372], " The clustering is estimated once on the full history of the candidate features, including "
                     "the evaluation period. Because it does not use the target, this does not leak label "
                     "information, but the set of representative features is not re-selected in real time.")
    rep_tbl = equation_table(e, template_eq, EQ_REP, 7)
    P[372]._p.addnext(rep_tbl)
    e.insert_after(rep_tbl, "In Equation (7), C_c is the set of features in cluster c. For example, one cluster groups "
                            "BCMPEBLS.Index with three macroeconomic series (SACWC_YoY, LEI.TOTL.Index and "
                            "OEZAKLAF_YoY); BCMPEBLS.Index has the lowest mean distance to the other members "
                            "(0.32) and is retained as the cluster representative.", like=P[373])
    subscript_text(rep_tbl.getnext(), "C_c")
    note("3.6", "Cross-reference corrected (was 'Figure 3'); cluster sizes, representative rule (Eq. 7) with a worked "
                "example; disclosed that clustering uses the full feature history.",
         "feature_selection.ipynb clusters all dates in featureset.csv; feature_clusters.csv.")
    e.replace(P[378], "lagged by one month relative", "lagged by four weekly observations (approximately one month) relative")
    e.replace(P[380], "Features with more than 30% missing observations were excluded",
              "Before the mutual-information clustering, candidate features with more than 30% missing "
              "observations were excluded (15 of the 359 candidates)")
    e.replace(P[380], "Remaining missing values are imputed using forward filling.",
              "Remaining missing values are imputed using forward filling; gaps at the start of a series, before "
              "its first available observation, are back-filled.")
    note("3.6.1", "Stated that the 30% missing-value screen precedes clustering and removed 15 features.",
         "Supervisor question. featureset.csv has 15 columns above the 30% threshold (e.g. the ZARI*.Curncy_QS "
         "series), so the reply in the comment thread ('no features dropped') should be revisited.")
    e.replace(P[381], "the final modelling dataset contains 929 weekly observations.",
              "the feature matrix contains 929 weekly observations (18 June 2007 to 31 March 2025). Of the 944 "
              "weekly observations from March 2007, the earliest are lost to the 12-week rolling windows used to "
              "construct the relative-factor features and to the four-week lag. After the ten unresolved "
              "end-of-sample labels are removed, 919 labelled observations (18 June 2007 to 20 January 2025) are "
              "available for each factor pair.")
    e.replace(P[381], "thereby eliminating forward information leakage", "thereby limiting forward information leakage")
    note("3.6.1", "Explained how 929 is obtained and added the 919-observation modelling sample.",
         "Supervisor question 'How did you calculate this value?'.")

    # 3.7 Model training --------------------------------------------------------------
    e.replace(P[388], "whether a factor is expected to out- or under-perform the other two factors",
              "whether the first-named factor in each pair is expected to outperform or underperform the second")
    note("3.7.1", "Target described as pairwise (first-named vs second factor).",
         "Each model is fitted to one factor pair, not one factor against the other two.")
    e.replace(P[390], "exceeds a calibrated threshold ", "exceeds a threshold ")
    e.replace(P[390], ":", ", fixed at 0.5 for all models and factor pairs rather than calibrated:")
    note("3.7.1", "Threshold stated as fixed at 0.5.", "predict(threshold=0.5) in all three pipelines.")
    e.replace(P[393], ". The model imposes the assumes that", ". The model assumes that")
    e.append(P[398], " Predictors are standardised with a scaler fitted only on the training window, and the "
                     "fitted mean and variance are applied to the prediction-date observation. Coefficients are "
                     "estimated by batch gradient descent with a learning rate of 0.05 for up to 5,000 iterations, "
                     "stopping early once the change in coefficients falls below 10\u207b\u2074. To verify the custom "
                     "implementation, it was compared with the elastic-net logistic regression of scikit-learn (saga "
                     "solver) under an equivalent penalty (C = 1/\u03bb) on the pre-evaluation sample of each factor "
                     "pair. For the three selected configurations, predicted probabilities differed by at most 0.019 "
                     "and in-sample ROC-AUC by at most 0.005. The small differences arise because the gradient-descent "
                     "solver applies a sub-gradient to the L1 penalty, which shrinks coefficients towards zero "
                     "without setting them exactly to zero.")
    note("3.7.2", "Added standardisation, optimiser settings and a validation of the custom logistic "
                  "regression against scikit-learn.",
         "Supervisor question on the custom function; check in .tmp_chapter3_minimal/logistic_check.py.")
    e.append(P[404], " The number of basis functions is common to all smooth terms and is selected from "
                     "{4, 5, 7, 10} (Table 3).")
    e.append(P[407], " In the implementation, a single smoothing penalty \u03bb is applied to all smooth terms and "
                     "selected by cross-validation, and predictors are standardised within each training window, as "
                     "for logistic regression.")
    note("3.7.3", "GAM implementation details (common basis size and penalty, standardisation).", "gam.ipynb.")
    e.replace(P[418], "In study, XGBoost model is trained using the xgboost python library.",
              "In this study, the XGBoost model is trained using the xgboost Python library with 500 boosting "
              "rounds and the binary logistic objective. Because tree splits are unaffected by monotonic "
              "transformations, predictors are not standardised for XGBoost.")
    remove(P[419])          # empty paragraph that pushed the section break onto a blank page
    t2 = T[19]
    e.cell(t2.rows[11].cells[5], "Minimum child weight (min_child_weight)")
    e.cell(t2.rows[12].cells[5], "L2 regularisation (\u03bb)")
    note("3.7 / Table 2", "XGBoost key hyperparameters updated (minimum child weight and L2 penalty replace "
                          "boosting rounds and \u03bb, \u03b3).",
         "Revised XGBoost grid; boosting rounds are fixed at 500 and \u03b3 is not tuned.")

    # 3.8 Hyperparameter tuning and model validation ---------------------------------
    e.replace(P[424], "Firstly, Combinatorial Cross-Purged Validation is used exclusively for hyperparameter "
                      "optimization within the training windows",
              "Firstly, combinatorial purged cross-validation (CPCV) is used exclusively for hyperparameter "
              "optimisation within the training windows")
    e.replace(P[424], "to assesses", "to assess")
    e.replace(P[425], "Combinatorial cross-purged validation", "Combinatorial purged cross-validation")
    e.replace(P[427], "is used the calculate exact number of training/test combinations sets required",
              "is used to calculate the exact number of training/test combinations required")
    e.replace(P[427], "In our case, the test and training sets are given by ",
              "In our case, the number of held-out groups and the total number of groups are given by ")
    e.set_text(P[428], "With N = 5 and k = 2, there are C(5, 2) = 10 training\u2013validation combinations, and "
                       "Equation (22) gives \u03c6[5, 2] = 4 back-test paths. In this implementation, hyperparameters "
                       "are selected on the mean ROC-AUC across the ten held-out combinations; the four paths are "
                       "not assembled and evaluated separately.")
    note("3.8.1", "CPCV arithmetic corrected and stated explicitly (10 combinations, 4 paths); stated that the "
                  "paths are not assembled.",
         "CV_SPLITS = 5, CV_TEST_SPLITS = 2; tune_hyperparameters() averages the ten fold AUCs.")
    e.picture(P[429], FIGURES / "figure_cpcv_implementation.png")
    e.replace(P[430], "CPCV back-test path, k=1 validation set, N=5 cross validation splits",
              "CPCV with N = 5 chronological groups and k = 2 held-out groups, giving ten combinations and four "
              "back-test paths")
    note("3.8.1", "Figure 3 caption corrected (was k = 1); figure version with the purge and embargo inserted.",
         "chapter3_final_figures/figure_cpcv_implementation.png.")
    e.replace(P[431], "For our analysis, we used a maximum look-ahead window of 12 weeks when assigning trend "
                      "labels to our target variable, meaning that the purge gap should be at least the size of "
                      "the look-ahead window to mitigate look-ahead bias.",
              "For our analysis, the trend-scanning labels use returns up to ten weeks after the observation "
              "date. A 12-week purge is therefore applied before each held-out group and a 12-week embargo after "
              "it, so that the gap exceeds the label horizon.")
    e.replace(P[431], "Parameters are then optimized on the generated blocks. We select the parameter set with the "
                      "highest predictive score on the unseen validation set as input to the walk-forward validation "
                      "phase.",
              "Each candidate parameter set is fitted on the training groups of all ten combinations and scored "
              "by ROC-AUC on the held-out groups; the set with the highest mean ROC-AUC is carried into the "
              "walk-forward phase. The initial search uses only the 763 observations that precede the evaluation "
              "block, with predictors standardised on that sample alone. No purge is applied at the boundary "
              "between this sample and the evaluation block, so the labels of its final ten observations depend "
              "on returns from the first weeks of the evaluation block; this residual overlap is a limitation of "
              "the initial selection.")
    e.replace(P[431], "parameters are re-tuned annually.",
              "the search is repeated on the then-available purged training window after every 52 walk-forward "
              "predictions. Only the initial parameter selection was saved; the parameter sets chosen at the two "
              "later retuning points were not stored.")
    note("3.8.1", "Purge/embargo, selection criterion, tuning sample and retuning schedule stated as implemented.",
         "tune_hyperparameters() and run_pipeline() in all three notebooks; artifacts hold only initial parameters; "
         "X_train_full = X[:-156] has no boundary purge.")
    e.replace(P[432], "Key hyperparameters", "Hyperparameter search grids")
    t3 = T[21]
    grid = [
        ("18", "L1/L2 mixing ratio (\u03b1)", "[ 0 , 0.5 , 1 ]"),
        (None, "Regularisation strength (\u03bb)", "[ 0.01 , 0.1 , 1 , 10 , 100 , 1000 ]"),
        (None, "Learning rate (\u03b7)", "0.05 (fixed)"),
        (None, "Maximum iterations", "5000 (fixed)"),
        (None, "Convergence tolerance", "10\u207b\u2074 (fixed)"),
        ("20", "Number of splines", "[ 4 , 5 , 7 , 10 ]"),
        (None, "Smoothness penalty (\u03bb)", "[ 1 , 10 , 100 , 1000 , 10000 ]"),
        (None, "Spline order", "3, cubic (fixed)"),
        (None, "Maximum iterations", "1000 (fixed)"),
        ("162", "Learning rate (\u03b7)", "[ 0.01 , 0.03 , 0.1 ]"),
        (None, "Maximum tree depth", "[ 2 , 3 , 4 ]"),
        (None, "Minimum child weight", "[ 5 , 10 , 20 ]"),
        (None, "L2 regularisation (\u03bb)", "[ 1 , 10 , 100 , 1000 ]"),
        (None, "Subsample ratio", "[ 0.6 , 0.8 ]"),
    ]
    for row, (count, name, values) in zip(t3.rows[1:], grid):
        cells = row.cells
        if count is not None:
            e.cell(cells[1], count)
        e.cell(cells[2], name)
        e.cell(cells[3], values)
    e.set_text(P[433], "Note: grids are searched exhaustively. XGBoost uses 500 boosting rounds, the binary "
                       "logistic objective and a fixed random seed in every configuration.")
    P[433]._p.find(qn("w:r")).find(qn("w:rPr")) is not None or None
    for r in P[433]._p.iter(qn("w:r")):
        rpr = r.find(qn("w:rPr"))
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "20")
        rpr.append(sz)
    for row in t3.rows:
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = row is not t3.rows[-1]
    P[432].paragraph_format.keep_with_next = True
    note("3.8.1 / Table 3", "Table 3 replaced with the grids actually searched (18, 20 and 162 configurations).",
         "Old table showed superseded grids (e.g. XGBoost learning rate [5, 7, 10]).")
    e.replace(P[434], "Walk forward Cross-Validation", "Walk-forward validation")
    e.replace(P[435], "rolling walk-forward framework", "expanding-window walk-forward framework")
    e.replace(P[435], "and then evaluate its performance", "and then evaluating its performance")
    e.replace(P[435], "preventing data leakage", "limiting data leakage")
    e.picture(P[436], FIGURES / "figure_walk_forward_expanding.png")
    e.replace(P[437], "Walk-forward validation with rolling look-back window",
              "Expanding-window walk-forward validation with a 12-week purge and retuning every 52 predictions")
    e.set_text(P[438], "At each iteration, models are trained on an expanding window containing every labelled "
                       "observation up to 12 weeks before the prediction date, and then predict the label for the "
                       "following week. The training window grows from 751 observations at the first prediction to "
                       "906 at the last. To balance adaptability with parameter stability, hyperparameters are "
                       "re-optimised every 52 predictions. This frequency allows the model to adjust to structural "
                       "changes in market conditions while avoiding excessive sensitivity to transient fluctuations. "
                       "Hyperparameters are therefore selected three times, before the first prediction and after "
                       "the 52nd and 104th predictions, producing 156 one-week-ahead out-of-sample forecasts from "
                       "31 January 2022 to 20 January 2025.")
    note("3.8.2", "Rolling 773-week window replaced by the expanding window (751 to 906 observations); "
                  "Figure 4 redrawn.",
         "lookback=None in walk_forward_validation(); 919 \u2212 156 \u2212 12 = 751.")

    # 3.9 Performance evaluation ------------------------------------------------------
    e.replace(P[441], "directional accuracy, ranking ability, and probability calibration",
              "directional accuracy, ranking ability and probability quality, supplemented by confusion-matrix "
              "analysis of classification errors")
    confusion = e.insert_after(
        P[443],
        "Accuracy is complemented by the confusion matrix, which cross-tabulates predicted and realised classes "
        "into true negatives (TN), false positives (FP), false negatives (FN) and true positives (TP) at the 0.5 "
        "threshold (Fawcett, 2006). The matrix shows whether errors are concentrated in one class. This matters "
        "here because a model that assigns most observations to one class can attain respectable accuracy while "
        "having little ability to identify the other. The share of observations predicted positive is therefore "
        "compared with the realised positive share, and confusion matrices are reported for every model and "
        "factor pair in Chapter 4.", like=P[443])
    note("3.9", "New paragraph on confusion matrices.", "Supervisor comment 'Why not use the confusion matrix?'.")
    e.replace(P[445], "serves as the primary metric for hyperparameter optimisation within the CPCV framework",
              "serves as the primary metric for hyperparameter optimisation, measured as the mean ROC-AUC across "
              "the ten CPCV combinations")
    e.append(P[447], " A constant forecast of 0.5 yields a log-loss of ln 2 \u2248 0.693, which serves as an "
                     "uninformative reference value.")
    e.append(P[449], " Because the evaluation block contains only 156 weekly observations and the labels are "
                     "persistent, small differences in these point estimates are interpreted with caution.")

    # 3.10 Feature importance ---------------------------------------------------------
    e.replace(P[453], "Section 3.5", "Section 3.6")
    e.replace(P[453], "liquidity proxies", "commodity indicators")
    e.replace(P[453], "in generating out-of-sample forecasts", "in generating their forecasts")
    e.replace(P[455], "over the same training window within each walk-forward iteration, the analysis guarantees",
              "on the same sample for every model, the analysis ensures")
    e.insert_after(
        P[455],
        "Shapley values are computed once for each model and factor pair, for a model refitted with the initially "
        "selected hyperparameters on the 763 observations that precede the evaluation block, without a boundary "
        "purge. The values therefore describe these pre-evaluation refits rather than the 156 walk-forward models. "
        "For logistic "
        "regression the values are exact: each contribution equals the coefficient multiplied by the deviation of "
        "the standardised feature from its sample mean. For XGBoost, exact tree Shapley values are obtained with "
        "TreeSHAP (Lundberg, et al., 2020). For the GAM, which has no specialised explainer, model-agnostic "
        "Shapley estimates are computed on the log-odds output, with the training sample as background data.",
        like=P[455])
    note("3.10", "Stated the sample and explainer used for each model; removed the claim that SHAP is computed "
                 "within each walk-forward iteration.",
         "compute_shap_importance() fits once on the pre-evaluation sample in each notebook.")
    e.set_text(P[457], "This study reports global feature importance. A global measure summarises a feature's "
                       "influence across all observations in the explanation sample, whereas a local decomposition "
                       "explains an individual prediction, such as the forecast for a specific week. For example, a "
                       "global ranking may show that currency volatility is among the most influential inputs on "
                       "average, while a local decomposition would show how far it moved the forecast for one "
                       "particular week. Observation-level decompositions were not retained for the final models, so "
                       "the analysis is limited to global importance, and regime-specific interpretation is left for "
                       "future work.")
    e.replace(P[458], "across observations within each walk-forward training window",
              "across the observations of the pre-evaluation training sample")
    remove(P[459])
    note("3.10", "Local/global distinction explained with an example; local (regime-conditional) analysis "
                 "paragraph deleted.",
         "Supervisor comment; Chapter 4 reports global mean |SHAP| only.")

    # renumber the original equations (4)-(18) after the four new feature-selection equations
    for k, old in [(k, k - 1) for k in range(5, 19)] + [(20, 18)]:
        set_equation_number(e, T[k]._tbl, old + NEW_EQUATIONS)
    note("3.7-3.8", "Original Equations (4)-(18) renumbered (8)-(22); the CPCV text now cites Equation (22).",
         "Four equations added in Section 3.6.")

    if highlight:
        add_log(document, log)
    document.save(OUTPUTS[highlight])
    return log


def add_log(document, log):
    body = document.element.body
    sect = body.find(qn("w:sectPr"))
    heading = document.add_paragraph()
    run = heading.add_run("Amendment log (review aid \u2013 remove before submission)")
    run.bold = True; run.font.size = Pt(13)
    intro = document.add_paragraph(
        "Highlighted text in the chapter marks amendments. Deletions cannot be highlighted, so they are listed "
        "here together with the reason for every change.")
    intro.runs[0].font.size = Pt(10)
    table = document.add_table(rows=1, cols=3)
    table.style = document.styles["Table Grid"] if "Table Grid" in [s.name for s in document.styles] else None
    for cell, text in zip(table.rows[0].cells, ("Section", "Change", "Reason / evidence")):
        cell.paragraphs[0].add_run(text).bold = True
    for section, change, reason in log:
        cells = table.add_row().cells
        for cell, text in zip(cells, (section, change, reason)):
            cell.paragraphs[0].add_run(text)
    for row in table.rows:
        for cell, width in zip(row.cells, (Inches(0.9), Inches(2.9), Inches(2.9))):
            cell.width = width
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.alignment = 0
                for run in paragraph.runs:
                    run.font.size = Pt(9)
    # move the new content before sectPr
    for el in [heading._p, intro._p, table._tbl]:
        sect.addprevious(el)


if __name__ == "__main__":
    for highlight in (True, False):
        log = build(highlight)
        print(OUTPUTS[highlight].name, "changes logged:", len(log))
