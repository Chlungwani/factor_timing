"""Build the Chapter 4 (Results) amendment document for the final rerun of all three models.

All three models select hyperparameters on the mean ROC-AUC of the ten CPCV combinations;
logistic regression uses scikit-learn, and a GAM setting that fails on any combination is
excluded. Starts from the current thesis, accepts the tracked changes already in the file,
removes comments, keeps only Chapter 4 and applies the amendments in place, so that figures,
tables and styles are preserved. Every number is read from results/model_outputs, table cells
are rewritten (and highlighted) only where the value changes, and the script stops if a
sentence's premise (e.g. which model leads a pair) no longer holds.

Output
  Chapter4_Results_Amendments_Highlighted.docx   amendments in yellow plus an amendment log

Usage: python build_chapter4_amendments.py [out_dir] [thesis_snapshot.docx]
"""

import copy
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu
from docx.text.paragraph import Paragraph
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score

from build_minimal_chapter3 import Editor, accept_revisions, add_log, set_heading_start, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUTPUTS = RESULTS / "model_outputs"
FIGURES = RESULTS / "chapter4_final_figures"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
# A copy of the thesis: Word keeps the original open and Python is refused read access to it,
# so take a snapshot first (e.g. with cp) and pass its path.
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = OUT_DIR / "Chapter4_Results_Amendments_Highlighted.docx"

MODELS = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}
PAIRS = {"Momentum_vs_Value": "Momentum vs value", "Momentum_vs_Quality": "Momentum vs quality",
         "Value_vs_Quality": "Value vs quality"}
A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
WP_NS = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"


# --------------------------------------------------------------------- numbers --

def metrics():
    rows = []
    for pair in PAIRS:
        for model, tag in MODELS.items():
            d = pd.read_csv(OUTPUTS / "probs" / f"{tag}_{pair.lower()}_probs.csv")
            tn, fp, fn, tp = confusion_matrix(d["actual"], d["prediction"]).ravel()
            rows.append({"pair": pair, "model": model,
                         "acc": accuracy_score(d["actual"], d["prediction"]),
                         "auc": roc_auc_score(d["actual"], d["probability"]),
                         "ll": log_loss(d["actual"], np.clip(d["probability"], 1e-12, 1 - 1e-12)),
                         "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                         "pred_pos": (tp + fp) / len(d), "act_pos": d["actual"].mean()})
    return pd.DataFrame(rows).set_index(["pair", "model"])


def pct(x):
    return f"{100 * x:.1f}%"


def f3(x):
    return f"{x:.3f}"


# --------------------------------------------------------------------- helpers --

def text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def style(el):
    ppr = el.find(qn("w:pPr"))
    s = ppr.find(qn("w:pStyle")) if ppr is not None else None
    return s.get(qn("w:val")) if s is not None else ""


def find_para(body, startswith=None, contains=None):
    hits = [p for p in body.iter(qn("w:p"))
            if (startswith is None or text(p).startswith(startswith)) and (contains is None or contains in text(p))]
    if len(hits) != 1:
        raise ValueError(f"expected one paragraph for {startswith or contains!r}, found {len(hits)}")
    return hits[0]


def heading(body, title, occurrence=0):
    """The occurrence-th paragraph whose whole text is `title` (section headings repeat across 4.2 and 4.4)."""
    return [p for p in body.iter(qn("w:p")) if text(p).strip() == title][occurrence]


def set_bold(run, bold):
    rpr = run.find(qn("w:rPr"))
    if rpr is None:
        rpr = run.makeelement(qn("w:rPr"), {}); run.insert(0, rpr)
    for tag in ("w:b", "w:bCs"):
        for el in rpr.findall(qn(tag)):
            rpr.remove(el)
    if bold:
        rpr.insert(0, rpr.makeelement(qn("w:b"), {}))


def cell_is_bold(cell):
    for run in cell._tc.iter(qn("w:r")):
        rpr = run.find(qn("w:rPr"))
        b = rpr.find(qn("w:b")) if rpr is not None else None
        if b is not None and b.get(qn("w:val")) not in ("0", "false"):
            return True
    return False


def set_cell(editor, cell, value, bold=None):
    """Replace a cell's text with highlighted `value`, keeping the first run's formatting."""
    p = cell.paragraphs[0]._p
    runs = list(p.iter(qn("w:r")))
    template = runs[0] if runs else None
    keep_bold = cell_is_bold(cell) if bold is None else bold
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    run = editor.new_run(template, value)
    set_bold(run, keep_bold)
    p.append(run)


def rewrite(editor, p, value):
    """Replace a paragraph's whole content, citations included, by one highlighted run."""
    template = next(p.iter(qn("w:r")))
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    run = editor.new_run(template, value)
    rpr = run.find(qn("w:rPr"))
    for colour in ([] if rpr is None else rpr.findall(qn("w:color"))):   # the author's red marks her own edits
        rpr.remove(colour)
    p.append(run)


def drawing_runs(elements, document):
    """(run, image part, pixel width) for every picture inside the given body elements."""
    found = []
    for el in elements:
        for run in el.iter(qn("w:r")):
            blip = next(run.iter(A_NS + "blip"), None)
            if blip is not None:
                part = document.part.related_parts[blip.get(qn("r:embed"))]
                found.append((run, part, Image.open(io.BytesIO(part.blob)).size[0]))
    return found


def between(body, start_el, end_el):
    children = list(body)
    return children[children.index(start_el) + 1:children.index(end_el)]


def swap_picture(run, part, png):
    """Replace a picture's image, keeping its width and fitting its height to the new image."""
    data = png.read_bytes()
    part._blob = data
    w, h = Image.open(io.BytesIO(data)).size
    extent = next(run.iter(WP_NS + "extent"))
    cy = str(round(int(extent.get("cx")) * h / w))
    extent.set("cy", cy)
    for ext in run.iter(A_NS + "ext"):
        if ext.getparent().tag == A_NS + "xfrm":
            ext.set("cy", cy)


def add_caption(editor, template, after, value):
    """Insert a highlighted caption paragraph, formatted like `template`, after element `after`."""
    caption = copy.deepcopy(template)
    runs = list(caption.iter(qn("w:r")))
    for run in runs:
        run.getparent().remove(run)
    caption.append(editor.new_run(runs[0], value))
    after.addnext(caption)
    # If the picture's paragraph ends a section, end it after the caption instead, so that the
    # caption is not pushed onto the next page; keep the picture with its caption.
    ppr = after.find(qn("w:pPr"))
    if ppr is None:
        ppr = after.makeelement(qn("w:pPr"), {}); after.insert(0, ppr)
    sect = ppr.find(qn("w:sectPr"))
    if sect is not None:
        caption.find(qn("w:pPr")).append(sect)
    if ppr.find(qn("w:keepNext")) is None:
        keep = ppr.makeelement(qn("w:keepNext"), {})
        style_el = ppr.find(qn("w:pStyle"))
        ppr.insert(1 if style_el is not None else 0, keep)
    return caption


# ----------------------------------------------------------------------- build --

def build():
    m = metrics()
    L = lambda pair: m.loc[(pair, "Logistic")]
    means = m.groupby(level="model")[["acc", "auc", "ll"]].mean()
    boot = pd.read_csv(OUTPUTS / "bootstrap" / "auc_mean_across_pairs.csv").set_index("quantity")
    shap_mv = pd.read_csv(OUTPUTS / "shap" / "logistic_momentum_vs_value_shap.csv", index_col=0).iloc[:, 0]

    # Premises of the rewritten sentences
    mv, mq, vq = "Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"
    best = lambda pair, col, f=max: f(MODELS, key=lambda k: m.loc[(pair, k), col])
    rank = lambda pair, col: sorted(MODELS, key=lambda k: -m.loc[(pair, k), col])
    assert best(mv, "acc") == best(mv, "auc") == "XGBoost" and best(mv, "ll", min) == "Logistic"
    assert rank(mv, "auc") == ["XGBoost", "Logistic", "GAM"] and rank(mv, "acc") == ["XGBoost", "GAM", "Logistic"]
    assert m["acc"].idxmax() == (mv, "XGBoost") and m.loc[(mv, "XGBoost"), "auc"] < 0.6
    assert best(mq, "acc") == best(mq, "auc") == "Logistic" == best(mq, "ll", min)
    assert rank(mq, "auc")[1] == rank(mq, "acc")[1] == "XGBoost"
    assert best(vq, "acc") == best(vq, "auc") == "Logistic" == best(vq, "ll", min)
    assert m["auc"].idxmax() == (mq, "Logistic")
    assert means["auc"].idxmax() == "Logistic" and means["acc"].idxmax() == "XGBoost"
    assert all(best(p, "ll", min) == "Logistic" for p in PAIRS)
    assert all(m.loc[(p, "Logistic"), "pred_pos"] > m.loc[(p, "Logistic"), "act_pos"] for p in (mv, mq))
    assert m.loc[(mv, "GAM"), "pred_pos"] > m.loc[(mv, "GAM"), "act_pos"]
    assert m.loc[(vq, "XGBoost"), "pred_pos"] == m["pred_pos"].max()
    # unchanged sentences in 4.3 that must still hold: the most balanced model per pair
    balance = lambda pair: min(MODELS, key=lambda k: abs(m.loc[(pair, k), "pred_pos"] - m.loc[(pair, k), "act_pos"]))
    assert balance(mv) == balance(mq) == "XGBoost" and balance(vq) == "Logistic"
    diffs = [q for q in boot.index if " - " in q]
    assert len(diffs) == 3 and all(boot.loc[q, "ci_low"] < 0 < boot.loc[q, "ci_high"] for q in diffs), \
        boot.loc[diffs, ["ci_low", "ci_high"]]
    top = lambda tag, n=5: list(pd.read_csv(OUTPUTS / "shap" / f"{tag}_momentum_vs_value_shap.csv", index_col=0)
                                .iloc[:, 0].sort_values(ascending=False).index[:n])
    assert list(shap_mv.index[:5]) == ["USDZARV1M.Curncy_QS", "USDZARV1Y.Curncy_QS", "mom1m_momentum_vs_quality",
                                       "US_RB_Curvature", "VolTermStructPC1"], list(shap_mv.index[:5])
    assert top("gam", 6) == ["relvol_momentum_vs_quality_12", "relvol_value_vs_quality_12", "RandPPP_Factor_Inst_TS_ST",
                             "ZAR18M.Curncy_QS", "SA_RB_Curvature", "MichgnConcIndx"], top("gam", 6)
    assert top("xgboost") == ["relvol_momentum_vs_value_12", "SA_RB_Curvature", "corr_momentum_value_12",
                              "MXWD.Index.PE.Ratio_QS", "mom1m_momentum_vs_value"], top("xgboost")

    document = Document(SOURCE)
    body = document.element.body
    # accept_revisions drops the deletion mark of a deleted table row but keeps the row; remove such
    # rows here, and any table they leave empty
    for mark in list(body.iter(qn("w:del"))):
        if mark.getparent().tag == qn("w:trPr"):
            row = mark.getparent().getparent()
            row.getparent().remove(row)
    for tbl in list(body.iter(qn("w:tbl"))):
        if tbl.find(qn("w:tr")) is None:
            tbl.getparent().remove(tbl)
    # likewise paragraphs whose paragraph mark was deleted: drop those left empty by the deletion
    deleted_marks = [p for p in body.iter(qn("w:p"))
                     if p.find(qn("w:pPr")) is not None and p.find(qn("w:pPr")).find(qn("w:rPr")) is not None
                     and p.find(qn("w:pPr")).find(qn("w:rPr")).find(qn("w:del")) is not None]
    accept_revisions(body)
    for p in deleted_marks:
        if p.getparent() is not None and not text(p).strip() and not list(p.iter(A_NS + "blip")) \
                and not list(p.iter("{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath")) \
                and p.find(qn("w:pPr")).find(qn("w:sectPr")) is None:
            p.getparent().remove(p)
    strip_comments(body)
    for tag in ("w:customXmlInsRangeStart", "w:customXmlInsRangeEnd",
                "w:customXmlDelRangeStart", "w:customXmlDelRangeEnd"):
        for el in list(body.iter(qn(tag))):
            el.getparent().remove(el)

    # keep only Chapter 4: from its "Chapter" label to the element before the next chapter label
    children = list(body)
    results = children.index(heading(body, "Results"))
    conclusion = next(i for i, el in enumerate(children)
                      if i > results and el.tag == qn("w:p") and style(el) == "Heading1")
    first = results - 1 if style(children[results - 1]) == "Chapter" else results
    last = conclusion - 2 if style(children[conclusion - 1]) == "Chapter" else conclusion - 1
    for i, el in enumerate(children):
        if el.tag != qn("w:sectPr") and not first <= i <= last:
            body.remove(el)
    set_heading_start(document, 4)

    e = Editor(True)
    log = []

    def note(section, change, reason):
        log.append((section, change, reason))

    note("All", "All three models rerun. Logistic regression re-estimated with scikit-learn (Section 3.7.2); "
         "hyperparameters selected on the mean ROC-AUC of the ten CPCV combinations; a GAM setting that fails on "
         "any combination is excluded rather than scored on fewer (Section 3.8.1). XGBoost results and the GAM "
         "results for momentum vs quality and value vs quality are unchanged; logistic regression changes for "
         "all pairs, and the GAM for momentum vs value.",
         "Supervisor request (scikit-learn); corrected CPCV implementation. Numbers from results/model_outputs.")
    reason = "Rerun of all three models (first row)."
    gam_reason = ("The GAM previously selected for momentum vs value had been scored on only 8 of the 10 CPCV "
                  "combinations (its fit failed on the other two); that setting is now excluded (first row).")

    # 4.1: everything here is defined in Chapter 3; say it once (author, 30 October 2026)
    p = find_para(body, startswith="This chapter presents the final out-of-sample results")
    rewrite(e, p, "This chapter presents the out-of-sample results of the three models of Section 3.7 for the 156 "
                  "weekly predictions of the walk-forward evaluation (Section 3.8.2), assessed with the measures of "
                  "Section 3.9.")
    for start in ("Hyperparameters are selected using combinatorial purged cross-validation",
                  "The final evaluation block contains 156 weekly observations",
                  "Performance is assessed using accuracy, ROC-AUC and log-loss"):
        para = find_para(body, startswith=start)
        para.getparent().remove(para)
    note("4.1", "Reduced to one sentence referring to Sections 3.7, 3.8.2 and 3.9. Deleted: the description of the "
         "three model classes; the CPCV design, purge and embargo, the walk-forward procedure and the 751-to-906 "
         "training window, with their citations (López de Prado, 2018, 2020; Bergmeir & Benítez, 2012; Tashman, "
         "2000); the evaluation dates and the definition of the positive class; and the metric definitions (Fawcett, "
         "2006; Hanley & McNeil, 1982). All are given in Chapter 3; the back-test paths are reported in Appendix C.",
         "Author: say it once, in the methodology. Citations used only here drop out of the bibliography.")

    # 4.2 opening
    p = find_para(body, startswith="Table 4.2 presents")
    e.replace(p, "The largest ROC-AUC in the study is 0.589", f"The largest ROC-AUC in the study is {f3(m['auc'].max())}")
    note("4.2", "Largest ROC-AUC updated (now logistic regression, momentum vs quality).", reason)

    # Table 4.2: every value that changed, and bold on the best value per pair
    t42 = next(t for t in document.tables if text(t._tbl).startswith("Factor pairModelAccuracyROC-AUCLog-loss"))
    pair_of_row, current = {}, None
    for row in t42.rows[1:]:
        label = row.cells[0].text.strip()
        current = next((k for k, v in PAIRS.items() if v == label), current)
        pair_of_row[row] = (current, row.cells[1].text.strip())
    changed = set()
    for row, (pair, model) in pair_of_row.items():
        r = m.loc[(pair, model)]
        for col, key, fmt, f in ((2, "acc", pct, max), (3, "auc", pct, max), (4, "ll", f3, min)):
            is_best = best(pair, key, f) == model
            if row.cells[col].text.strip() != fmt(r[key]):
                set_cell(e, row.cells[col], fmt(r[key]), bold=is_best)
                changed.add((model, PAIRS[pair].lower()))
            elif cell_is_bold(row.cells[col]) != is_best:
                for run in row.cells[col]._tc.iter(qn("w:r")):
                    set_bold(run, is_best)
    assert changed == {("Logistic", PAIRS[p].lower()) for p in PAIRS} | {("GAM", "momentum vs value")}, changed
    note("4.2 / Table 4.2", "Updated: logistic regression for all three pairs and the GAM for momentum vs value; the "
         "XGBoost rows and the other GAM rows are unchanged. For momentum vs quality, bold (best value) moves from "
         "XGBoost's accuracy and ROC-AUC to logistic regression's.", reason)

    # 4.2.1 Momentum versus value
    p = find_para(body, startswith="Momentum versus Value provides")
    e.replace(p, "The point estimates therefore indicate that nonlinear structure contributes to this factor pair. ", "")
    note("4.2.1", "Duplicated sentence removed ('The point estimates therefore indicate that nonlinear structure "
         "contributes to this factor pair.').", "The sentence appeared twice in a row.")
    lm, gm = L(mv), m.loc[(mv, "GAM")]
    e.replace(p, "The GAM ranks second, with accuracy of 59.0% and ROC-AUC of 0.582. The point estimates therefore "
                 "indicate that nonlinear structure contributes to this factor pair, consistent",
              f"The GAM has the second-highest accuracy, at {pct(gm.acc)}, but the lowest ROC-AUC, at {f3(gm.auc)}. "
              f"XGBoost's lead suggests that nonlinear structure contributes to this factor pair, consistent")
    note("4.2.1", "GAM results updated: it no longer ranks second on ROC-AUC (now third), so 'the point estimates "
         "therefore indicate' nonlinear structure is attributed to XGBoost's lead only.", gam_reason)
    e.replace(p, "Logistic regression performs poorly at the fixed threshold, with accuracy of 48.1% and ROC-AUC of 0.551. "
                 "Its comparatively low log-loss of 0.688 indicates",
              f"Logistic regression has the second-highest ROC-AUC ({f3(lm.auc)}) but the lowest accuracy at the fixed "
              f"threshold ({pct(lm.acc)}), because it predicts the positive class in {pct(lm.pred_pos)} of weeks. "
              f"Its log-loss of {f3(lm.ll)}, the lowest for this pair, indicates")
    note("4.2.1", "Logistic regression results updated; it now ranks second on ROC-AUC rather than performing "
         "'poorly', and its low accuracy is explained by its positive-class tendency.", reason)
    e.replace(p, "The size of the effect remains moderate because the best AUC is below 0.60. ", "")
    note("4.2.1", "Deleted 'The size of the effect remains moderate because the best AUC is below 0.60' (the modest "
         "size of all ROC-AUCs is stated at the start of Section 4.2).", "Repetition.")

    # 4.2.2 Momentum versus quality
    p = find_para(body, startswith="Predictive performance is weaker for momentum versus quality")
    lq, xq = L(mq), m.loc[(mq, "XGBoost")]
    old = text(p)[:text(p).index("The GAM displays the opposite error pattern.")]
    e.replace(p, old,
              f"For momentum versus quality, logistic regression is the strongest model on all three measures, with "
              f"accuracy of {pct(lq.acc)}, ROC-AUC of {f3(lq.auc)} and log-loss of {f3(lq.ll)}. Its ROC-AUC is the "
              f"highest in the study. The confusion "
              f"matrix shows that it predicts the positive class in {pct(lq.pred_pos)} of weeks against an actual rate "
              f"of {pct(lq.act_pos)}, so part of its accuracy comes from favouring momentum leadership. XGBoost ranks "
              f"second, with accuracy of {pct(xq.acc)} and ROC-AUC of {f3(xq.auc)}. ")
    note("4.2.2", "Rewritten: logistic regression now leads this pair on accuracy, ROC-AUC and log-loss; XGBoost "
         "ranks second. Deleted: 'Predictive performance is weaker for momentum versus quality' and the sentences "
         "describing XGBoost as strongest and logistic regression's accuracy as 51.3%.", reason)

    # 4.2.3 Value versus quality
    p = find_para(body, startswith="Value versus quality reverses the ordering")
    lv = L(vq)
    e.replace(p, "Value versus quality reverses the ordering observed in the momentum pairs. Logistic regression is the "
                 "strongest model, with accuracy of 56.4%, ROC-AUC of 0.585 and log-loss of 0.684.",
              f"Logistic regression is also the strongest model for value versus quality, with accuracy of "
              f"{pct(lv.acc)}, ROC-AUC of {f3(lv.auc)} and log-loss of {f3(lv.ll)}.")
    note("4.2.3", "Opening reworded (the pair no longer 'reverses the ordering' of the momentum pairs, since logistic "
         "regression also leads momentum vs quality); figures updated.", reason)
    e.replace(p, "It therefore ranks below logistic regression on all three reported measures and below the GAM on "
                 "accuracy.", "It also ranks below the GAM on accuracy.")
    note("4.2.3", "'It therefore ranks below logistic regression on all three reported measures' deleted: the opening "
         "sentence states that logistic regression leads on all three.", "Repetition.")

    # 4.3 confusion matrices and probability quality
    t44 = next(t for t in document.tables if text(t._tbl).startswith("Factor pairModelTNFPFNTP"))
    current, changed = None, set()
    for row in t44.rows[1:]:
        label = row.cells[0].text.strip()
        current = next((k for k, v in PAIRS.items() if v == label), current)
        model = row.cells[1].text.strip()
        r = m.loc[(current, model)]
        for col, value in zip(range(2, 7), (r.TN, r.FP, r.FN, r.TP, r.pred_pos)):
            value = str(int(value)) if col < 6 else pct(value)
            if row.cells[col].text.strip() != value:
                set_cell(e, row.cells[col], value)
                changed.add((model, PAIRS[current].lower()))
        assert row.cells[7].text.strip() == pct(r.act_pos)
    assert changed == {("Logistic", PAIRS[p].lower()) for p in (mv, mq)} | {("GAM", "momentum vs value")}, changed
    note("4.3 / Table 4.4", "Confusion-matrix counts and predicted-positive shares updated for logistic regression "
         "(both momentum pairs; value vs quality is unchanged) and the GAM for momentum vs value.", reason)

    p = find_para(body, startswith="XGBoost provides the most balanced class treatment")
    rewrite(e, p, "XGBoost provides the most balanced class treatment for the two momentum pairs, and logistic "
                  "regression for value versus quality (Table 4.4).")
    note("4.3", "The predicted and actual positive shares behind this statement deleted from the text; Table 4.4 "
         "reports them.", "Repetition of Table 4.4.")

    p = find_para(body, startswith="The extreme positive-class tendencies")
    xv = m.loc[(vq, "XGBoost")]
    e.replace(p, "The extreme positive-class tendencies of logistic regression for momentum versus value and XGBoost "
                 "for value versus quality",
              f"The positive-class tendencies of logistic regression in the two momentum pairs, of the GAM for "
              f"momentum versus value and, most extremely, of XGBoost for value versus quality ({pct(xv.pred_pos)} "
              f"predicted positive)")
    note("4.3", "Logistic regression's positive-class tendency is no longer extreme (was 92.3% predicted positive for "
         "momentum vs value); the GAM for momentum vs value added.", reason)

    p = find_para(body, startswith="Table 4.1 reports the realised class distribution")
    e.replace(p, " These counts provide the context required to interpret the confusion matrices reported below.", "")
    note("4.3", "Deleted 'These counts provide the context required to interpret the confusion matrices reported "
         "below.'", "Author: say it once.")

    p = find_para(body, startswith="The probability results introduce")
    lls = [L(pair).ll for pair in PAIRS]
    e.replace(p, "An uninformative forecast of 0.5 for every observation has a log-loss of approximately 0.693. ", "")
    e.replace(p, "representing only a small improvement but avoiding severe overconfidence",
              "only slightly below the uninformative 0.693 (Section 3.9) but without severe overconfidence")
    note("4.3", "Deleted 'An uninformative forecast of 0.5 for every observation has a log-loss of approximately "
         "0.693' (defined in Section 3.9, now referred to).", "Author: say it once.")
    e.replace(p, "between 0.684 and 0.688", f"between {f3(min(lls))} and {f3(max(lls))}")
    glls = [m.loc[(pair, "GAM"), "ll"] for pair in PAIRS]
    e.replace(p, "between 0.932 and 1.609", f"between {f3(min(glls))} and {f3(max(glls))}")
    e.replace(p, "XGBoost has stronger average ranking and thresholded performance, but its log-loss exceeds 0.693 for "
                 "both momentum pairs. Thus, better discrimination does",
              "XGBoost has the highest average accuracy, but its log-loss exceeds 0.693 for both momentum pairs. "
              "Thus, better thresholded classification does")
    note("4.3", "Logistic regression and GAM log-loss ranges updated; XGBoost no longer has the strongest average "
         "ranking (ROC-AUC), only the highest average accuracy.", reason)

    # 4.4 feature importance
    p = find_para(body, startswith="Global feature importance is measured")
    rewrite(e, p, "Figures 4.5–4.7 show the ten features with the largest global SHAP importance for each model "
                  "(Section 3.10).")
    note("4.4", "Opening reduced to a reference to Section 3.10, which now explains how the figures are read (the ten "
         "leading features, ranks rather than magnitudes, colours by the feature categories of Table 2) and the "
         "limits of the interpretation. Deleted: the definition of global importance, the caveats that SHAP values "
         "do not establish causal relationships or stable out-of-sample information (Kumar et al., 2020; Molnar, "
         "2022), the comparison and colour conventions, and 'This distinction is important because all final ROC-AUC "
         "values remain below 0.60', which no longer holds (the highest is " + f3(m['auc'].max()) + "). The old "
         "reference to 'Table 1 of Section 3.6' (now the factor-return correlations) is gone with it; category names "
         "are aligned across Table 2, Table B.8 and the figure legends. Figure numbers updated (4.5–4.7).",
         "Author: say it once; Section 3.10 explains the results section.")

    p = find_para(body, startswith="Logistic regression is dominated by currency-volatility")
    e.replace(p, "Logistic regression is dominated by currency-volatility and volatility-term-structure variables.",
              "Logistic regression is led by the two rand-volatility measures, followed by short-horizon factor "
              "momentum, US yield-curve curvature and the volatility term structure.")
    note("4.4.1", "Logistic regression's leading features described from the new ranking (its selected penalty "
         "changed for this pair).", reason)
    e.replace(p, "The GAM places the greatest weight on relative-volatility measures, followed by foreign-exchange "
                 "entropy, global property volatility and a global fundamental component.",
              "The GAM places the greatest weight on relative-volatility measures, followed by rand PPP valuation, "
              "medium-term USDZAR forward pricing, local yield-curve curvature and US consumer sentiment.")
    note("4.4.1", "GAM's leading features described from the ranking of the newly selected setting.", gam_reason)
    e.replace(p, "XGBoost draws on a broader combination of local real-yield-curve curvature, pair-specific relative "
                 "volatility, global valuation and short-horizon factor momentum",
              "XGBoost draws on a broader combination of pair-specific relative volatility, local real-yield-curve "
              "curvature, the momentum-value correlation, global valuation and short-horizon factor momentum")
    note("4.4.1", "XGBoost's leading features reordered to match Figure 4.5 (was 4.3), with the momentum-value "
         "correlation added. The thesis text followed the earlier SHAP values (100 boosting rounds); the figure now "
         "uses the values for the 500-round model used in the walk-forward.", "xgboost_cpcv_wf.ipynb; Figure 4.5.")

    p = find_para(body, startswith="The standardised value-versus-quality relative-performance measure")
    e.replace(p, " Logistic regression is the best-performing model for this pair, but its AUC of 0.585 remains modest. "
                 "The feature rankings should therefore be interpreted as a description of the fitted decision function "
                 "rather than evidence of stable economic causality.", "")
    note("4.4.3", "Deleted 'Logistic regression is the best-performing model for this pair, but its AUC of 0.585 "
         "remains modest. The feature rankings should therefore be interpreted as a description of the fitted "
         "decision function rather than evidence of stable economic causality.' The first point is made in Section "
         "4.2.3 and the second at the start of Section 4.4.", "Repetition.")

    # 4.5 chapter conclusion
    p = find_para(body, startswith="The final results provide modest")
    e.replace(p, "XGBoost produces the strongest results for momentum versus value and momentum versus quality, while "
                 "logistic regression performs best for value versus quality. XGBoost also records the highest mean "
                 "accuracy and ROC-AUC across the three pairs. Its average advantage does not amount to uniform "
                 "dominance: logistic regression has the strongest probability quality and leads the value-versus-quality "
                 "pair, while several",
              f"XGBoost produces the strongest results for momentum versus value, while logistic regression performs "
              f"best for momentum versus quality and value versus quality. Logistic regression also records the highest "
              f"mean ROC-AUC across the three pairs ({f3(means.auc['Logistic'])}, against {f3(means.auc['XGBoost'])} "
              f"for XGBoost and {f3(means.auc['GAM'])} for the GAM) and the lowest log-loss for every pair, whereas "
              f"XGBoost records the highest mean accuracy ({pct(means.acc['XGBoost'])}, against "
              f"{pct(means.acc['Logistic'])}). Several")
    note("4.5", "Model ranking rewritten: logistic regression now leads two pairs and has the highest mean ROC-AUC; "
         "XGBoost keeps momentum vs value and the highest mean accuracy. 'No model dominates uniformly' is not "
         "repeated (stated at the start of Section 4.2).", reason)

    p = find_para(body, startswith="The findings therefore qualify")
    xl = boot.loc["XGBoost - Logistic mean AUC"]
    e.replace(p, "Greater flexibility is beneficial for the two momentum pairs, particularly momentum versus value, but "
                 "the linear model is more effective for value versus quality. Even the best AUC is below 0.60.",
              f"Greater flexibility is beneficial only for momentum versus value, where XGBoost leads; the regularised "
              f"linear model is more effective for the other two pairs. Block-bootstrap 95% intervals for the differences in mean ROC-AUC between the models all include zero (XGBoost minus "
              f"logistic regression: {xl.ci_low:.3f} to {xl.ci_high:.3f}), so the ranking of the models is not "
              f"statistically established.".replace("-0.", "−0."))
    note("4.5", "Flexibility conclusion revised; bootstrap intervals for model differences (method in Section 3.9) "
         "added to qualify the ranking. 'Even the best AUC is below 0.60' deleted (Section 4.2).",
         "Results updated; results/model_outputs/bootstrap/auc_mean_across_pairs.csv.")

    # Figures
    t42_el, mv_head = t42._tbl, heading(body, "Momentum versus Value")
    rocs = drawing_runs(between(body, t42_el, mv_head), document)
    keep = max(rocs, key=lambda x: x[2])
    for run, part, _ in rocs:
        if run is not keep[0]:
            para = run.getparent()
            para.remove(run)
            if not text(para).strip() and not list(para.iter(A_NS + "blip")):
                para.getparent().remove(para)
    swap_picture(keep[0], keep[1], FIGURES / "figure_final_roc_curves.png")
    # The old images carried their titles inside the picture; the regenerated figures have none, so the
    # two figures without a Word caption get one, formatted like the Figure 4.3 caption.
    template = find_para(body, contains="SHAP importance for momentum versus value")
    add_caption(e, template, keep[0].getparent(),
                "Figure 4.1. Out-of-sample ROC curves for the three models, by factor pair (31 January 2022 to "
                "20 January 2025).")
    extra = (f" The thesis also holds {len(rocs) - 1} copies of an earlier render of this figure next to it; they "
             f"are omitted here and should be deleted." if len(rocs) > 1 else "")
    note("4.2 / Figure 4.1", "ROC figure replaced with the regenerated version (logistic regression and GAM "
         "momentum-vs-value curves and AUCs updated; thesis colours and line widths; no title inside the figure). "
         "Caption added (highlighted): the old image carried its title inside the picture." + extra,
         "build_chapter4_corrected_figures.py")

    vq_head = heading(body, "Value versus quality", 0)
    cm_head = heading(body, "Consolidated confusion-matrix analysis and probability quality")
    probs = drawing_runs(between(body, vq_head, cm_head), document)
    assert len(probs) == 1, len(probs)
    # The consolidated probability figure becomes one figure per pair, placed after that pair's discussion.
    consolidated = probs[0][0].getparent()
    cx = int(next(probs[0][0].iter(WP_NS + "extent")).get("cx"))
    discussions = {mv: "Momentum versus Value provides", mq: "For momentum versus quality, logistic regression",
                   vq: "Logistic regression is also the strongest model for value versus quality"}
    leader = {mv: "momentum", mq: "momentum", vq: "value"}
    for number, pair in (("4.2", mv), ("4.3", mq), ("4.4", vq)):
        picture = copy.deepcopy(consolidated)
        for run in list(picture.iter(qn("w:r"))):
            run.getparent().remove(run)
        find_para(body, startswith=discussions[pair]).addnext(picture)
        Paragraph(picture, document._body).add_run().add_picture(
            str(FIGURES / f"figure_final_probabilities_{pair.lower()}.png"), width=Emu(cx))
        add_caption(e, template, picture,
                    f"Figure {number}. {PAIRS[pair].replace(' vs ', ' versus ')}: walk-forward predicted probability "
                    f"that {leader[pair]} leads, by model, with the realised class.")
    consolidated.getparent().remove(consolidated)
    note("4.2 / Figures 4.2–4.4", "The walk-forward probability figure (all three pairs in one image) is replaced by "
         "one figure per pair, each placed after that pair's discussion in Sections 4.2.1–4.2.3, with captions added "
         "(highlighted). Logistic regression and GAM momentum-vs-value lines updated; thesis colours and line widths; "
         "no titles inside the figures.", "User request; build_chapter4_corrected_figures.py.")

    for old, new, pair in (("4.3", "4.5", mv), ("4.4", "4.6", mq), ("4.5", "4.7", vq)):
        caption = find_para(body, contains=f"SHAP importance for {PAIRS[pair].replace(' vs ', ' versus ').lower()}")
        picture = caption.getprevious()
        while not list(picture.iter(A_NS + "blip")):
            picture = picture.getprevious()
        (run, part, _), = drawing_runs([picture], document)
        swap_picture(run, part, FIGURES / f"figure_final_shap_{pair.lower()}.png")
        e.replace(caption, f"Figure {old}.", f"Figure {new}.")
        for r in caption.iter(qn("w:r")):
            e.mark(r)
    note("4.4 / Figures 4.5–4.7", "SHAP figures renumbered (were 4.3–4.5) after the per-pair probability figures. "
         "Any reference to Figures 4.3–4.5 elsewhere in the thesis must follow.", "Figure numbering.")
    note("4.4 / Figures 4.5–4.7", "SHAP figures replaced (captions highlighted), in the thesis colours with no title "
         "inside the figure. Logistic panels: new values; the ranking changes only for momentum vs value. GAM panel "
         "for momentum vs value: newly selected setting. The XGBoost panels also change: they now use the corrected "
         "XGBoost SHAP values (500 boosting rounds, as in the walk-forward), which were regenerated on 17 October "
         "but not yet transplanted into the thesis.", "build_chapter4_corrected_figures.py; xgboost_cpcv_wf.ipynb.")

    # sections emptied by the author's tracked deletions render as blank pages; drop them
    def is_content(el):
        return (el.tag != qn("w:p") or text(el).strip() or style(el) == "Chapter" or list(el.iter(A_NS + "blip"))
                or list(el.iter("{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath")))
    section, emptied = [], 0
    for el in list(body):
        if el.tag == qn("w:sectPr"):
            break
        section.append(el)
        ppr = el.find(qn("w:pPr")) if el.tag == qn("w:p") else None
        if ppr is not None and ppr.find(qn("w:sectPr")) is not None:
            if not any(is_content(x) for x in section):
                for x in section:
                    body.remove(x)
                emptied += 1
            section = []
    if emptied:
        note("4.2", f"{emptied} empty section(s) removed after Figure 4.1. They held the duplicate ROC images deleted "
             "in the thesis and render as blank pages once that deletion is accepted.", "Layout.")

    add_log(document, log)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
