"""Build the Chapter 4 (Results) amendment document from the rerun on the corrected data.

The weekly Quality returns were stamped five weeks early (fix_quality_offset.py); the labels and features were
rebuilt and all three models rerun with annual retuning (rerun_pipeline.py). Every result in Chapter 4 changes, and
so does its reading: XGBoost now leads every pair on every measure.

Starts from the current thesis, accepts the tracked changes already in the file, removes comments, keeps only
Chapter 4 and applies the amendments in place, so that figures, tables and styles are preserved. The text keeps
the structure the author gave Section 4.2 (per pair an opening paragraph, then the ROC figure and a paragraph on
it, then the probability figure and a paragraph on it). Every number is read from results/model_outputs, and the
script stops if a sentence's premise (which model leads, in which weeks a model was confidently wrong, which
hyperparameters it selected) does not hold.

Output
  Chapter4_Results_Amendments_Highlighted.docx   amendments in yellow plus an amendment log

Usage: python build_chapter4_amendments.py [out_dir] [thesis_snapshot.docx]
"""

import copy
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from lxml import etree
import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu
from docx.text.paragraph import Paragraph
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve

from build_minimal_chapter3 import Editor, accept_revisions, add_log, set_heading_start, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUTPUTS = RESULTS / "model_outputs"
LABELS = ROOT / "legacy" / "data_input" / "labeled_relative_factor_returns.csv"
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
W14_NS = "{http://schemas.microsoft.com/office/word/2010/wordml}"


# --------------------------------------------------------------------- numbers --

def metrics():
    rows = []
    for pair in PAIRS:
        for model, tag in MODELS.items():
            d = pd.read_csv(OUTPUTS / "probs" / f"{tag}_{pair.lower()}_probs.csv")
            tn, fp, fn, tp = confusion_matrix(d["actual"], d["prediction"]).ravel()
            ranked = d.sort_values("probability", ascending=False)["actual"]
            fpr, tpr, _ = roc_curve(d["actual"], d["probability"])
            rows.append({"pair": pair, "model": model,
                         # for the ROC descriptions: hits among the highest- and lowest-ranked weeks, and the
                         # true-positive rate at false-positive rates 0, 0.01, ..., 1
                         "top20_pos": int(ranked.head(20).sum()), "bottom30_neg": int((1 - ranked.tail(30)).sum()),
                         "roc": tpr[np.searchsorted(fpr, np.linspace(0, 1, 101), side="right") - 1],
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


def find_para_any(body, *starts):
    """The one paragraph that starts with any of `starts` (the original text or the author's transplant of it)."""
    hits = [p for p in body.iter(qn("w:p")) if text(p).startswith(starts)]
    if len(hits) != 1:
        raise ValueError(f"expected one paragraph starting with one of {starts!r}, found {len(hits)}")
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


def fresh_copy(p):
    """A copy of a paragraph without Word's paragraph ids, which must be unique (Word assigns new ones)."""
    p = copy.deepcopy(p)
    for attr in (W14_NS + "paraId", W14_NS + "textId"):
        p.attrib.pop(attr, None)
    return p


SDT_IDS = random.Random(2026)


def plain_run(template):
    """A text run formatted like `template`, without the author's red, a highlight or a field character."""
    rpr = template.find(qn("w:rPr"))
    run = OxmlElement("w:r")
    if rpr is not None:
        rpr = copy.deepcopy(rpr)
        for el in rpr.findall(qn("w:color")) + rpr.findall(qn("w:highlight")) + rpr.findall(qn("w:vertAlign")):
            rpr.remove(el)
        run.append(rpr)
    return run


def citation(editor, template, tags, shown):
    """A Word citation field for the sources `tags` (their Source Manager tags), showing `shown` until updated."""
    sdt, pr, content = OxmlElement("w:sdt"), OxmlElement("w:sdtPr"), OxmlElement("w:sdtContent")
    sid = OxmlElement("w:id")
    sid.set(qn("w:val"), str(SDT_IDS.randint(10 ** 8, 2 ** 31 - 1)))
    pr.append(sid)
    pr.append(OxmlElement("w:citation"))
    sdt.append(pr)
    sdt.append(content)
    instr = f" CITATION {tags[0]} \\l 7177" + "".join(f" \\m {t}" for t in tags[1:]) + " "
    for kind in ("begin", "instr", "separate", "result", "end"):
        run = plain_run(template)
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = instr
        elif kind == "result":
            el = OxmlElement("w:t")
            el.text = shown
            editor.mark(run)
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), {"begin": "begin", "separate": "separate", "end": "end"}[kind])
        run.append(el)
        content.append(run)
    return sdt


def compose(editor, p, parts):
    """Replace a paragraph's whole content by highlighted text. `parts` holds strings, runs (e.g. from footnote())
    and, for citations, (tags, shown) pairs, which become Word citation fields."""
    template = next(r for r in p.iter(qn("w:r")) if r.find(qn("w:t")) is not None)
    template = plain_run(template)
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    for part in parts:
        if hasattr(part, "tag"):                     # a ready-made run, e.g. a footnote reference
            p.append(part)
        elif isinstance(part, str):
            run = editor.new_run(template, part)
            rpr = run.find(qn("w:rPr"))
            for colour in ([] if rpr is None else rpr.findall(qn("w:color"))):
                rpr.remove(colour)
            p.append(run)
        else:
            p.append(citation(editor, template, *part))


def footnote(editor, document, text):
    """A highlighted footnote: adds the note to the document's footnotes part, in the thesis's footnote styles,
    and returns the reference run for the body text. Word numbers footnotes by position, so the note takes its
    place among the thesis's existing footnotes when the paragraph is copied in."""
    part = next(pt for pt in document.part.package.iter_parts() if str(pt.partname) == "/word/footnotes.xml")
    root = part.element if hasattr(part, "element") else etree.fromstring(part.blob)
    fid = str(max(int(n.get(qn("w:id"))) for n in root.findall(qn("w:footnote"))) + 1)

    def styled_run(style):
        run = etree.SubElement(etree.Element("tmp"), qn("w:r"))
        rpr = etree.SubElement(run, qn("w:rPr"))
        etree.SubElement(rpr, qn("w:rStyle")).set(qn("w:val"), style)
        return run

    note_el = etree.SubElement(root, qn("w:footnote"))
    note_el.set(qn("w:id"), fid)
    p = etree.SubElement(note_el, qn("w:p"))
    etree.SubElement(etree.SubElement(p, qn("w:pPr")), qn("w:pStyle")).set(qn("w:val"), "FootnoteText")
    mark = styled_run("FootnoteReference")
    etree.SubElement(mark, qn("w:footnoteRef"))
    p.append(mark)
    body_run = etree.SubElement(p, qn("w:r"))
    t = etree.SubElement(body_run, qn("w:t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = " " + text
    editor.mark(body_run)
    if not hasattr(part, "element"):
        part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    ref = styled_run("FootnoteReference")
    etree.SubElement(ref, qn("w:footnoteReference")).set(qn("w:id"), fid)
    editor.mark(ref)
    return ref


def set_ppr(p, ppr):
    """Give paragraph `p` a copy of the paragraph properties `ppr` (its style and spacing)."""
    old = p.find(qn("w:pPr"))
    if old is not None:
        p.remove(old)
    p.insert(0, copy.deepcopy(ppr))


def table_caption(editor, template, table_el, value):
    """Insert a highlighted caption paragraph, formatted like `template`, above a table and kept with it."""
    caption = fresh_copy(template)
    runs = list(caption.iter(qn("w:r")))
    for child in list(caption):
        if child.tag != qn("w:pPr"):
            caption.remove(child)
    caption.append(editor.new_run(runs[0], value))
    ppr = caption.find(qn("w:pPr"))
    if ppr.find(qn("w:keepNext")) is None:
        ppr.insert(1 if ppr.find(qn("w:pStyle")) is not None else 0, OxmlElement("w:keepNext"))
    table_el.addprevious(caption)
    return caption


def keep_together(tbl):
    """Keep a table on one page: every paragraph of every row but the last is kept with the next."""
    for row in tbl.findall(qn("w:tr"))[:-1]:
        for p in row.iter(qn("w:p")):
            ppr = p.find(qn("w:pPr"))
            if ppr is None:
                ppr = OxmlElement("w:pPr")
                p.insert(0, ppr)
            if ppr.find(qn("w:keepNext")) is None:
                ppr.insert(1 if ppr.find(qn("w:pStyle")) is not None else 0, OxmlElement("w:keepNext"))


def unlink_seq(p):
    """Replace caption number fields (SEQ) by their current result. In an extract of one chapter Word would
    otherwise renumber the figures from 1 when the fields update; returns the number of fields unlinked."""
    count = 0
    for simple in list(p.iter(qn("w:fldSimple"))):
        if "SEQ" in simple.get(qn("w:instr"), ""):
            for child in list(simple):
                simple.addprevious(child)
            simple.getparent().remove(simple)
            count += 1
    runs, i = list(p.iter(qn("w:r"))), 0
    kind = lambda r: (r.find(qn("w:fldChar")).get(qn("w:fldCharType")) if r.find(qn("w:fldChar")) is not None else None)
    while i < len(runs):
        if kind(runs[i]) == "begin":
            j = next(k for k in range(i + 1, len(runs)) if kind(runs[k]) == "end")
            field = runs[i:j + 1]
            assert not any(kind(r) == "begin" for r in field[1:]), "nested field in a caption"
            instr = "".join(t.text or "" for r in field for t in r.iter(qn("w:instrText")))
            if "SEQ" in instr:
                sep = next(k for k, r in enumerate(field) if kind(r) == "separate")
                for r in field[:sep + 1] + [field[-1]]:
                    r.getparent().remove(r)
                count += 1
            i = j + 1
        else:
            i += 1
    return count


WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen twenty").split()


def num(n, start=False):
    """A count as written in the text: words below ten, or at the start of a sentence (up to twenty)."""
    n = int(n)
    if start:
        assert n <= 20, n
        return WORDS[n].capitalize()
    return WORDS[n] if n < 10 else str(n)


# ----------------------------------------------------------------------- build --

def build():
    m = metrics()
    means = m.groupby(level="model")[["acc", "auc", "ll"]].mean()
    boot = pd.read_csv(OUTPUTS / "bootstrap" / "auc_mean_across_pairs.csv").set_index("quantity")
    mv, mq, vq = "Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"
    best = lambda pair, col, f=max: f(MODELS, key=lambda k: m.loc[(pair, k), col])
    rank = lambda pair, col: sorted(MODELS, key=lambda k: -m.loc[(pair, k), col])
    roc = lambda pair, model: m.loc[(pair, model), "roc"]      # true-positive rate at false-positive rate i/100
    x = np.linspace(0, 1, 101)

    # The evidence for every statement below, from the saved predictions, the hyperparameters selected at each
    # tuning point (Table C.6) and the initial tuning grids; the build stops if any premise fails
    pm = {(pair, k): pd.read_csv(OUTPUTS / "probs" / f"{MODELS[k]}_{pair.lower()}_probs.csv", parse_dates=["Date"])
          for pair in PAIRS for k in MODELS}
    sel = {(pair, k): json.loads((OUTPUTS / "artifacts" / f"{MODELS[k]}_{pair.lower()}_walkforward_params.json")
                                 .read_text()) for pair in PAIRS for k in MODELS}
    grid = lambda pair, k: pd.read_csv(OUTPUTS / "artifacts" / f"{MODELS[k]}_{pair.lower()}_tuning_grid.pkl")  # CSV
    period = lambda pair, k, j: pm[(pair, k)].iloc[52 * j:52 * (j + 1)]      # predictions between tuning points
    years = lambda d: d.Date.dt.year
    pct_r = lambda v: f"{int(100 * v + 0.5)}%"
    f2 = lambda v: f"{v:.2f}"
    acc_of = lambda d: (d.prediction == d.actual).mean()

    def share_y(pair, year):
        d = pm[(pair, "Logistic")]
        return d[years(d) == year].actual.mean()

    def hindsight(pair):
        """Accuracy of predicting, in every week, the factor that led most often in that calendar year."""
        d = pm[(pair, "Logistic")]
        return (d.groupby(years(d)).actual.transform(lambda s: int(s.mean() >= 0.5)) == d.actual).mean()

    def within(pair, k):
        """ROC-AUC within each of the three 52-week periods between tuning points."""
        return [roc_auc_score(s.actual, s.probability) for s in (period(pair, k, j) for j in range(3))]

    def highest(pair, lo, hi):
        """Share of false-positive rates lo/100..hi/100 at which each model's ROC curve is the highest."""
        return pd.Series([max(MODELS, key=lambda k: roc(pair, k)[i]) for i in range(lo, hi + 1)]).value_counts(
            normalize=True)

    below = lambda pair, k: (roc(pair, k) < x).mean()        # share of the range where the curve is below chance

    def ranked(pair, k, n, top=True):
        return pm[(pair, k)].sort_values("probability", ascending=not top).head(n)

    def confident_errors(pair, k):
        """Weeks in which the model gave the outcome that did not occur a probability above 0.9, and their
        share of its log-loss."""
        d = pm[(pair, k)]
        p, y = d["probability"].clip(1e-12, 1 - 1e-12), d["actual"]
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        wrong = ((p > 0.9) & (y == 0)) | ((p < 0.1) & (y == 1))
        return d.loc[wrong], ll[wrong].sum() / ll.sum()

    first = lambda s: s.iloc[0]

    # ---- premises: 4.2 opening
    for pair in PAIRS:
        assert best(pair, "acc") == best(pair, "auc") == best(pair, "ll", min) == "XGBoost", pair
        assert m.loc[(pair, "XGBoost"), "acc"] > max(m.loc[(pair, k), "acc"] for k in ("Logistic", "GAM"))
        assert best(pair, "ll", max) == "GAM", pair
    assert sum(m.loc[(p, "GAM"), "auc"] < m.loc[(p, "Logistic"), "auc"] for p in PAIRS) == 2
    assert m["auc"].idxmax() == (vq, "XGBoost") and m["auc"].max() < 0.7
    # The accuracies are not a carry-over of persistent labels: leadership runs are short, and the most recent label
    # in the training data, 12 weeks before each forecast (the walk-forward purge), rarely matches the outcome
    labels = pd.read_csv(LABELS, parse_dates=["Date"]).set_index("Date")
    carry, run_length = {}, {}
    for pair in PAIRS:
        y, d = labels[f"{pair}_trend"], pm[(pair, "Logistic")]
        assert (y.loc[d.Date].to_numpy() == d.actual.to_numpy()).all()
        carry[pair] = (y.iloc[y.index.get_indexer(d.Date) - 12].to_numpy() == d.actual.to_numpy()).mean()
        changes = np.flatnonzero(np.diff(np.r_[-1, d.actual.to_numpy(), -1]) != 0)
        run_length[pair] = np.diff(changes).mean()
    assert 0.4 < min(carry.values()) and max(carry.values()) <= 0.5
    assert 4 <= min(run_length.values()) and max(run_length.values()) < 6.5
    grid_max = {"lr": 0.1, "lr_min": 0.01, "reg_max": 1000, "reg_min": 1, "depth_min": 2,
                "logit_min": 0.01, "gam_lam_min": 1, "gam_splines_max": 10}

    # ---- premises: 4.2.1 momentum versus value
    lmv, gmv, xmv = (m.loc[(mv, k)] for k in MODELS)
    assert 0.45 < share_y(mv, 2024) < 0.55 and share_y(mv, 2022) < 0.4 and share_y(mv, 2023) > 0.7
    assert lmv.auc < 0.5 < gmv.auc < 0.55 and min(lmv.ll, gmv.ll) > 1
    assert highest(mv, 26, 95).get("XGBoost", 0) >= 0.9 and highest(mv, 0, 25).get("XGBoost", 0) < 0.5
    assert 0.4 < below(mv, "Logistic") < 0.6
    x_bot = ranked(mv, "XGBoost", 30, False)
    x_bot_2022 = int((years(x_bot) == 2022).sum())
    assert x_bot_2022 >= 20 and xmv.bottom30_neg >= 20
    w_xmv, w_lmv = within(mv, "XGBoost"), within(mv, "Logistic")
    assert min(w_xmv) > 0.55 and max(w_lmv) < 0.5
    xs = sel[(mv, "XGBoost")]
    x0, x1, x2 = (period(mv, "XGBoost", j) for j in range(3))
    assert xs[0]["learning_rate"] == xs[1]["learning_rate"] == grid_max["lr_min"] and xs[0]["reg_lambda"] < 1000
    assert xs[1]["reg_lambda"] == grid_max["reg_max"]
    assert xs[2]["learning_rate"] == grid_max["lr"] and xs[2]["reg_lambda"] == grid_max["reg_min"]
    assert (x0.prediction == 1).sum() == 1 and (x1.prediction == 0).sum() == 1
    x_conf, _ = confident_errors(mv, "XGBoost")
    assert len(x_conf) and (x_conf.Date >= first(x2.Date)).all()
    hind = {pair: hindsight(pair) for pair in PAIRS}
    assert abs(xmv.acc - hind[mv]) < 0.01
    assert all(s["lambda_reg"] == grid_max["logit_min"] for s in sel[(mv, "Logistic")])
    assert all(s["lam"] == grid_max["gam_lam_min"] and s["n_splines"] == grid_max["gam_splines_max"]
               for s in sel[(mv, "GAM")])
    for k in ("Logistic", "GAM"):
        assert pm[(mv, k)].probability.min() < 0.05 and pm[(mv, k)].probability.max() > 0.95
    l_conf, l_share = confident_errors(mv, "Logistic")
    g_conf, g_share = confident_errors(mv, "GAM")
    both = set(l_conf.Date) & set(g_conf.Date)
    may_june = sorted(d for d in both if pd.Timestamp("2023-05-01") <= d <= pd.Timestamp("2023-06-30"))
    assert len(may_june) == 4 and all(pm[(mv, k)].set_index("Date").loc[may_june].probability.min() > 0.99
                                      and (pm[(mv, k)].set_index("Date").loc[may_june].actual == 0).all()
                                      for k in ("Logistic", "GAM"))
    lg = grid(mv, "Logistic")
    l_best = lg.groupby("lambda_reg").mean_auc.max()
    assert l_best.idxmax() == 0.01
    flat_gap = l_best[0.01] - l_best[100.0]
    assert 0 < flat_gap < 0.005

    # ---- premises: 4.2.2 momentum versus quality
    lmq, gmq, xmq = (m.loc[(mq, k)] for k in MODELS)
    for col, f in (("acc", max), ("auc", max), ("ll", min)):
        order = sorted(MODELS, key=lambda k: m.loc[(mq, k), col], reverse=f is max)
        assert order == ["XGBoost", "Logistic", "GAM"], (col, order)
    assert xmq.auc - lmq.auc < 0.03
    assert share_y(mq, 2022) < 0.5 and share_y(mq, 2024) < 0.5 < share_y(mq, 2023)
    assert highest(mq, 3, 54).get("Logistic", 0) >= 0.9 and highest(mq, 55, 99).get("XGBoost", 0) >= 0.9
    xq_bot = ranked(mq, "XGBoost", 30, False)
    assert xmq.bottom30_neg >= 20 and (years(xq_bot) == 2024).sum() > 15 and lmq.top20_pos >= 15
    w_mq = {k: within(mq, k) for k in MODELS}
    assert all(w[2] > 0.69 for w in w_mq.values()) and all(w[1] < 0.4 for w in w_mq.values())
    xs = sel[(mq, "XGBoost")]
    q = [period(mq, "XGBoost", j) for j in range(3)]
    ql = [period(mq, "Logistic", j) for j in range(3)]
    assert all(xs[j]["learning_rate"] == grid_max["lr_min"] and xs[j]["reg_lambda"] == grid_max["reg_max"]
               for j in (0, 1))
    assert xs[2]["learning_rate"] == grid_max["lr"] and xs[2]["reg_lambda"] == grid_max["reg_min"]
    assert q[0].prediction.all() and q[1].prediction.all() and acc_of(q[0]) < 0.5
    q01 = pd.concat(q[:2]).probability
    assert acc_of(q[0]) < acc_of(ql[0]) and acc_of(q[1]) > acc_of(ql[1]) and acc_of(q[2]) > acc_of(ql[2])
    assert abs(xmq.acc - hind[mq]) < 0.015
    assert all(s["lambda_reg"] == 10 for s in sel[(mq, "Logistic")]) and len(confident_errors(mq, "Logistic")[0]) == 1
    gq_conf, _ = confident_errors(mq, "GAM")
    assert all(s["lam"] in (1000, 10000) for s in sel[(mq, "GAM")])
    assert np.corrcoef(pm[(mq, "GAM")].probability, pm[(mq, "Logistic")].probability)[0, 1] > 0.8
    assert (pm[(mq, "GAM")].probability.max() - pm[(mq, "GAM")].probability.min()
            > pm[(mq, "Logistic")].probability.max() - pm[(mq, "Logistic")].probability.min())

    # ---- premises: 4.2.3 value versus quality
    lvq, gvq, xvq = (m.loc[(vq, k)] for k in MODELS)
    assert rank(vq, "auc")[1] == "Logistic" and sorted(MODELS, key=lambda k: m.loc[(vq, k), "ll"])[1] == "Logistic"
    assert 0.45 < gvq.acc < 0.55 and gvq.auc < 0.51
    assert share_y(vq, 2022) > 0.5 > max(share_y(vq, 2023), share_y(vq, 2024))
    assert highest(vq, 8, 78).get("XGBoost", 0) == 1 and below(vq, "XGBoost") == 0
    w_xvq, w_lvq = within(vq, "XGBoost"), within(vq, "Logistic")
    assert min(w_xvq) > 0.6 and w_lvq[0] > 0.75 and max(w_lvq[1:]) < 0.5
    # XGBoost is the only model whose ranking holds in every period (logistic regression ranks the first period
    # as well as it does, so "better within every period" would not hold)
    assert min(within(vq, "GAM")) < 0.5 and abs(w_lvq[0] - w_xvq[0]) < 0.02
    assert 0.4 < below(vq, "GAM") < 0.6
    assert all(s["max_depth"] == grid_max["depth_min"] for s in sel[(vq, "XGBoost")])
    v = [period(vq, "XGBoost", j) for j in range(3)]
    assert v[2].probability.min() < v[0].probability.min() and v[2].probability.max() > v[0].probability.max()
    assert len(confident_errors(vq, "XGBoost")[0]) == 1 and len(confident_errors(vq, "Logistic")[0]) == 0
    margin_vq = xvq.acc - hind[vq]
    margin_mom = max(abs(m.loc[(p, "XGBoost"), "acc"] - hind[p]) for p in (mv, mq))
    assert margin_vq > 0.025 and margin_mom < margin_vq
    gv_conf, gv_share = confident_errors(vq, "GAM")
    may_aug = gv_conf[(gv_conf.Date >= "2023-05-01") & (gv_conf.Date <= "2023-08-31")]
    assert len(may_aug) > len(gv_conf) / 2 and (may_aug.actual == 1).all() and (may_aug.probability < 0.1).all()
    assert sel[(vq, "GAM")][1]["lam"] == 1000 and (may_aug.Date >= first(period(vq, "GAM", 1).Date)).all() \
        and (may_aug.Date <= period(vq, "GAM", 1).Date.iloc[-1]).all()

    # ---- premises: 4.3
    balance = lambda pair: min(MODELS, key=lambda k: abs(m.loc[(pair, k), "pred_pos"] - m.loc[(pair, k), "act_pos"]))
    assert balance(mv) == balance(vq) == "XGBoost" and balance(mq) == "Logistic"
    gaps = {(p, k): m.loc[(p, k), "pred_pos"] - m.loc[(p, k), "act_pos"] for p in PAIRS for k in MODELS}
    ordered = sorted(gaps, key=lambda key: -abs(gaps[key]))
    assert ordered[:2] == [(mq, "XGBoost"), (vq, "Logistic")], ordered[:2]
    assert m.loc[(mv, "XGBoost"), "act_pos"] == m.loc[(mq, "XGBoost"), "act_pos"] > 0.5 > m.loc[(vq, "XGBoost"), "act_pos"]
    lls = {k: [m.loc[(p, k), "ll"] for p in PAIRS] for k in MODELS}
    assert [p for p in PAIRS if m.loc[(p, "XGBoost"), "ll"] < np.log(2)] == [mq, vq]
    assert all(m.loc[(p, k), "ll"] > np.log(2) for p in PAIRS for k in ("Logistic", "GAM") if (p, k) != (vq, "Logistic"))
    assert max(abs(m.loc[(p, "Logistic"), "ll"] - np.log(2)) for p in (mq, vq)) < 0.05
    assert all(s["lambda_reg"] == 10 for p in (mq, vq) for s in sel[(p, "Logistic")])

    # ---- premises: 4.4 and 4.5
    top = lambda tag, pair, n=10: list(pd.read_csv(OUTPUTS / "shap" / f"{tag}_{pair.lower()}_shap.csv", index_col=0)
                                       .iloc[:, 0].sort_values(ascending=False).index[:n])
    rel = lambda name: name.startswith(("relvol_", "volratio_"))
    t = {(pair, k): top(MODELS[k], pair) for pair in PAIRS for k in MODELS}
    assert [t[(mv, k)][0] for k in MODELS] == ["relvol_value_vs_quality_12", "relvol_momentum_vs_quality_12",
                                              "relvol_momentum_vs_value_12"]
    assert t[(mv, "Logistic")][1] == "relvol_momentum_vs_value_12"
    shared = {"ZAR18M.Curncy_QS", "USDZAR_EntDyn", "RandPPP_Factor_Inst_TS_ST"}
    assert shared <= set(t[(mv, "Logistic")][:6]) and shared <= set(t[(mv, "GAM")][:6])
    assert t[(mv, "XGBoost")][1:6] == ["corr_momentum_quality_12", "SA_RB_Curvature", "volratio_value_quality_12",
                                       "GlobalBondVol6", "GenComIndexPC1_QS"], t[(mv, "XGBoost")]
    assert t[(mq, "Logistic")][:2] == t[(mq, "XGBoost")][:2] == ["TOP40.Index.Price.to.Ebitda_Diff", "SA_RB_Curvature"]
    assert all("GenComIndexPC1_QS" in t[(mq, k)][:4] for k in ("Logistic", "XGBoost"))
    assert t[(mq, "GAM")][0] == "corr_value_quality_12" and all(rel(f) for f in t[(mq, "GAM")][1:4])
    assert t[(mq, "GAM")].index("TOP40.Index.Price.to.Ebitda_Diff") == 6
    shap_sum = {pair: pd.read_csv(OUTPUTS / "shap" / f"xgboost_{pair.lower()}_shap.csv", index_col=0).iloc[:, 0]
                .sort_values().tail(10).sum() for pair in PAIRS}
    assert shap_sum[mq] < min(shap_sum[mv], shap_sum[vq]) / 6
    assert t[(vq, "XGBoost")][0] == t[(vq, "Logistic")][1] == "mom3m_value_vs_quality"
    assert all("mom3m_momentum_vs_quality" in t[(vq, k)][:5] for k in ("Logistic", "XGBoost"))
    assert all("volratio_value_quality_12" in t[(vq, k)][:4] for k in MODELS)
    assert {"USDZAR_EntDyn", "GlobalPropertyVol12"} <= set(t[(vq, "Logistic")][:3]) & set(t[(vq, "GAM")][:3])
    assert {"GLD_RollSD", "US_RB_Curvature", "SA_RB_Curvature"} <= set(t[(vq, "XGBoost")])
    counts = pd.Series([f for v in t.values() for f in v]).value_counts()
    assert counts.index[0] == "SA_RB_Curvature" and counts.iloc[0] == 8 and counts.iloc[1] < 6, counts.head()
    xl, xg = boot.loc["XGBoost - Logistic mean AUC"], boot.loc["XGBoost - GAM mean AUC"]
    ci = pd.read_csv(OUTPUTS / "bootstrap" / "auc_ci.csv").set_index(["factor_pair", "model"])
    assert [key for key in ci.index if ci.loc[key, "ci_low"] > 0.5] == [(vq, "XGBoost")]
    # and it is still the only one when the intervals are widened for the nine model-pair results (Bonferroni)
    assert [key for key in ci.index if ci.loc[key, "ci_low_bonferroni"] > 0.5] == [(vq, "XGBoost")]
    cvq = ci.loc[(vq, "XGBoost")]
    assert xg.ci_low > 0 > xl.ci_low
    assert all(boot.loc[f"{k} mean AUC", "ci_low"] < 0.5 for k in ("Logistic", "GAM")) \
        and boot.loc["XGBoost mean AUC", "ci_low"] > 0.5

    # ---- premises: the analytical framing (results/Chapter4_Analytical_Review.md): for each pair, whether XGBoost's
    # lead lies in ranking, in detecting both classes or in its probabilities, and whether it holds in every period
    found = lambda pair, k: (m.loc[(pair, k), "TP"], m.loc[(pair, k), "TP"] + m.loc[(pair, k), "FN"])   # first factor
    found0 = lambda pair, k: (m.loc[(pair, k), "TN"], m.loc[(pair, k), "TN"] + m.loc[(pair, k), "FP"])  # second factor
    share = lambda hit_total: hit_total[0] / hit_total[1]
    both_halves = [(p, k) for p in PAIRS for k in MODELS if min(share(found(p, k)), share(found0(p, k))) > 0.5]
    assert both_halves == [(mv, "XGBoost"), (vq, "XGBoost")], both_halves
    period_ll = lambda pair, k: [log_loss(s.actual, s.probability.clip(1e-12, 1 - 1e-12), labels=[0, 1])
                                 for s in (period(pair, k, j) for j in range(3))]
    period_acc = lambda pair, k: [acc_of(period(pair, k, j)) for j in range(3)]
    paired = pd.read_csv(OUTPUTS / "bootstrap" / "auc_paired_differences.csv").set_index(["factor_pair", "comparison"])
    # momentum vs value: ranking and classification without better probabilities than the uninformative forecast
    ll_xmv = period_ll(mv, "XGBoost")
    assert xmv.ll > np.log(2) > max(ll_xmv[:2]) and ll_xmv[2] > 0.8 and ci.loc[(mv, "XGBoost"), "ci_low"] < 0.5
    assert w_xmv[0] > w_xmv[1] > w_xmv[2] and xmv.auc > max(w_xmv)       # pooled above every within-period value
    assert x2.probability.min() < 0.15 and x2.probability.max() > 0.95
    # momentum vs quality: one class detected; the ranking reverses between periods
    assert share(found(mq, "XGBoost")) > 0.95 and share(found0(mq, "XGBoost")) < 0.2
    assert ci.loc[(mq, "XGBoost"), "ci_low"] < 0.5
    assert w_mq["XGBoost"][0] > 0.7 and max(w_mq["Logistic"][0], w_mq["GAM"][0]) < 0.45
    assert all(w_mq[k][1] < 0.4 for k in MODELS) and all(w_mq[k][2] > 0.69 for k in MODELS)
    # value vs quality: above chance even after adjustment, but not distinguishable from logistic regression, and
    # weakening over time
    dvq = paired.loc[(vq, "XGBoost - Logistic")]
    assert dvq.ci_low < 0 < dvq.ci_high
    acc_xvq, ll_xvq = period_acc(vq, "XGBoost"), period_ll(vq, "XGBoost")
    assert acc_xvq[0] > acc_xvq[1] > acc_xvq[2] and ll_xvq[0] < ll_xvq[1] < ll_xvq[2]
    # 4.3: the results below the uninformative log-loss
    below_ln2 = [(p, k) for p in PAIRS for k in MODELS if m.loc[(p, k), "ll"] < np.log(2)]
    assert below_ln2 == [(mq, "XGBoost"), (vq, "Logistic"), (vq, "XGBoost")], below_ln2
    closest_share = min(MODELS, key=lambda k: abs(m.loc[(mq, k), "pred_pos"] - m.loc[(mq, k), "act_pos"]))
    assert closest_share == "Logistic" and share(found0(mq, "Logistic")) < 0.5

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

    # Figures and tables are numbered consecutively through the thesis (Chapter 3 ends with Figure 8 and
    # Table 3; the author numbered the Chapter 4 figures 9-14): find where Chapter 4 continues from
    children = list(body)
    results = children.index(heading(body, "Results"))
    numbered = {"Figure": 0, "Table": 0}
    for el in children[:results]:
        hit = re.match(r"(Figure|Table) (\d+)\b", text(el).strip()) if el.tag == qn("w:p") else None
        if hit and style(el) != "TableofFigures":
            numbered[hit.group(1)] = max(numbered[hit.group(1)], int(hit.group(2)))
    assert numbered == {"Figure": 8, "Table": 3}, numbered
    t_perf, t_class, t_conf = numbered["Table"] + 1, numbered["Table"] + 2, numbered["Table"] + 3

    # keep only Chapter 4: from its "Chapter" label to the element before the next chapter label
    conclusion = next(i for i, el in enumerate(children)
                      if i > results and el.tag == qn("w:p") and style(el) == "Heading1")
    start = results - 1 if style(children[results - 1]) == "Chapter" else results
    last = conclusion - 2 if style(children[conclusion - 1]) == "Chapter" else conclusion - 1
    for i, el in enumerate(children):
        if el.tag != qn("w:sectPr") and not start <= i <= last:
            body.remove(el)
    set_heading_start(document, 4)
    captions = [p for p in body.iter(qn("w:p")) if style(p) == "Caption"]
    # the 4.2 figure captions the author has typed in as text (pasted from this extract) are no longer fields
    typed = [text(p)[:10] for p in captions if re.match(r"Figure (9|1[0-4]):", text(p))
             and "SEQ" not in "".join(t.text or "" for t in p.iter(qn("w:instrText")))
             and not any("SEQ" in (s.get(qn("w:instr")) or "") for s in p.iter(qn("w:fldSimple")))]
    unlinked = sum(unlink_seq(p) for p in captions)
    assert unlinked + len(typed) >= 6, (unlinked, typed)      # the six figure captions of Section 4.2

    e = Editor(True)
    log = []

    def note(section, change, reason):
        log.append((section, change, reason))

    rerun = "Rerun on the corrected data (first row)."
    note("All", "Every result in the chapter is replaced. The weekly Quality returns in the input data were "
         "stamped five weeks early: each week carried the Quality return of five weeks later. The series was "
         "realigned, the labels and features were rebuilt and all three models were rerun with annual retuning, "
         "unchanged otherwise. The Quality-based labels changed in about 40% of evaluation weeks (momentum vs "
         "quality and value vs quality); 12 of the 50 features changed; the momentum vs value labels did not. "
         "XGBoost now leads every pair on every measure, so the reading of the chapter changes: the text is "
         "rewritten throughout, in the structure the author gave Section 4.2.",
         "Data correction (results/fix_quality_offset.py, rerun_pipeline.py). Because features are lagged four "
         "weeks and the offset was five, the old Quality-based features contained one week of future Quality "
         "returns. Every number is read from results/model_outputs.")

    # 4.1 is in the thesis as the author transplanted it (an outline of Sections 4.2-4.5); one clause added for the
    # analysis of Section 4.2 by period
    e.replace(find_para(body, startswith="This chapter presents the out-of-sample results of the three models"),
              "for each factor pair and shows their ROC curves",
              "for each factor pair, examines whether each result holds across the three periods between tuning "
              "points and shows their ROC curves")
    note("4.1", "Outline of Section 4.2: 'examines whether each result holds across the three periods between tuning "
         "points' added, because Section 4.2 now analyses the results by period.", "Analytical review.")

    heads = {"Overview of the Evaluation Framework": "Overview of the evaluation framework",
             "Out-of-Sample predictive performance": "Out-of-sample predictive performance",
             "Momentum versus Value": "Momentum versus value",
             "Feature Importance": "Feature importance"}
    # the author has already put some headings in sentence case in the thesis
    heads = {old: new for old, new in heads.items() if any(text(p).strip() == old for p in body.iter(qn("w:p")))}
    head_els = {old: heading(body, old) for old in heads}
    h421 = head_els.get("Momentum versus Value") or heading(body, "Momentum versus value", 0)
    t42 = next(t for t in document.tables if text(t._tbl).startswith("Factor pairModelAccuracyROC-AUCLog-loss"))
    # The captions are numbered by their position: in the thesis they are fields, some of whose stored results are
    # stale (12-15 after Figure 10), which Word corrects when the fields are updated
    figs, stale = {}, []
    for el in body.iter(qn("w:p")):
        hit = re.match(r"Figure (\d+): (Out-of-sample ROC curves for|Walk-forward predicted probability that) "
                       r"(\w+) (?:versus|leads) (\w+)", text(el))
        if hit:
            pair = f"{hit.group(3).capitalize()}_vs_{hit.group(4).capitalize()}"
            n = numbered["Figure"] + 1 + len(figs)
            if int(hit.group(1)) != n:
                e.replace(el, f"Figure {hit.group(1)}:", f"Figure {n}:")
                stale.append((int(hit.group(1)), n))
            figs[("roc" if hit.group(2).startswith("Out") else "prob", pair)] = (n, el)
    assert len(figs) == 6 and [k for k, _ in sorted(figs, key=lambda key: figs[key][0])] == ["roc", "prob"] * 3, figs
    if typed:
        note("4.2 / figure captions", f"In the thesis, the captions of Figures {typed[0][7:-1]}–{typed[-1][7:-1]} are now "
             "typed text rather than caption fields, so they will not appear in the List of Figures and will not "
             "renumber. Re-insert them with References > Insert Caption (label Figure).", "Numbering.")
    if stale:
        note("4.2 / figure captions", f"Captions numbered {stale[0][0]}–{stale[-1][0]} in the thesis are shown as "
             f"{stale[0][1]}–{stale[-1][1]}. In the thesis they are caption fields whose stored numbers are out of date "
             f"(no Figure 11); select all and press F9 to update them.", "Numbering.")
    fig = lambda kind, pair: figs[(kind, pair)][0]
    shap_first = numbered["Figure"] + 7

    def discussed(kind, pair):
        """The paragraph that follows a figure's caption and discusses it (past any empty paragraph)."""
        p = figs[(kind, pair)][1].getnext()
        while p.tag == qn("w:p") and not text(p).strip() and not list(p.iter(A_NS + "blip")):
            p = p.getnext()
        assert p.tag == qn("w:p") and text(p).strip() and not style(p).startswith("Heading"), text(p)[:60]
        return p
    assert style(discussed("roc", mv)) == "BodyText"
    body_ppr = copy.deepcopy(discussed("roc", mv).find(qn("w:pPr")))

    def figure_text(kind, pair, parts):
        p = discussed(kind, pair)
        set_ppr(p, body_ppr)
        compose(e, p, parts)
        return p

    # Citations are Word citation fields (Source Manager tags); Has09 is the thesis's entry for Hastie et al. (2009)
    faw, woo, hoe = (["Faw06"], "(Fawcett, 2006)"), (["Woo17"], "(Wood, 2017)"), (["Hoe70"], "(Hoerl & Kennard, 1970)")

    # ---------------------------------------------------------------- 4.2 opening and Table 4
    # Author, 2 November 2026: "shorten the analysis a bit. make it concise and direct", then "update the language
    # and frame of reference from that of descriptive to more analytical" (results/Chapter4_Analytical_Review.md):
    # each paragraph states a finding, the evidence for it and its qualification
    compose(e, find_para(body, startswith="Table 4 reports the accuracy"), [
        f"Table {t_perf} reports the accuracy, ROC-AUC and log-loss of each model for the three factor pairs. "
        f"XGBoost has the highest accuracy and ROC-AUC and the lowest log-loss for every pair, but the ordering does "
        f"not follow model flexibility: the GAM, which adds smooth nonlinear effects to the linear specification, has "
        f"a lower ROC-AUC than logistic regression for two pairs and the highest log-loss for all three. Even the "
        f"strongest result, a ROC-AUC of {f3(m['auc'].max())} for value versus quality, indicates modest "
        f"discrimination ", (["Hos13"], "(Hosmer, Lemeshow, & Sturdivant, 2013)"),
        ", as expected when the predictable part of factor returns is small relative to noise ",
        (["Isr20"], "(Israel, Kelly, & Moskowitz, 2020)"),
        f". The accuracies are not a carry-over of persistent labels: the leading factor changed every "
        f"{num(round(min(run_length.values())))} to {num(round(max(run_length.values())))} weeks on average, so the "
        f"most recent label in the training data, 12 weeks before each forecast (Section 3.8.2), matches the outcome "
        f"in only {pct_r(min(carry.values()))} to {pct_r(max(carry.values()))} of weeks."])
    note("4.2", "Opening rewritten for the new results and in analytical form: XGBoost leads every pair on every "
         "measure, but the ordering does not follow flexibility (the GAM is below logistic regression on ROC-AUC for "
         "two pairs and has the highest log-loss for all three). Evidence added: the accuracies are not a carry-over "
         "of persistent labels: leadership spells last " + ", ".join(f"{run_length[p]:.1f}" for p in PAIRS) + " weeks "
         "on average, and the label 12 weeks before each forecast, the most recent one the models see after the "
         "purge, matches the outcome in only " + ", ".join(pct_r(carry[p]) for p in PAIRS) + " of weeks (momentum vs "
         "value, momentum vs quality, value vs quality; data_input/labeled_relative_factor_returns.csv). The 0.7 "
         "threshold of 'acceptable discrimination' is not used: it is a rule of thumb from applied logistic "
         "regression, not a standard for factor timing (analytical review); Hosmer et al. (2013) is cited for "
         "'modest' only.",
         rerun + " Author: analytical rather than descriptive; analysis and references.")

    conf_caption = find_para(body, startswith="Table 4.4. Confusion-matrix diagnostics")
    perf_title = f"Table {t_perf}: Out-of-sample accuracy, ROC-AUC and log-loss by factor pair and model"
    prev = t42._tbl.getprevious()
    if prev is not None and text(prev).strip() == perf_title:          # the author has added it
        ppr = prev.find(qn("w:pPr"))
        if ppr.find(qn("w:keepNext")) is None:
            ppr.insert(1 if ppr.find(qn("w:pStyle")) is not None else 0, OxmlElement("w:keepNext"))
    else:
        table_caption(e, conf_caption, t42._tbl, perf_title)
        note(f"4.2 / Table {t_perf}", f"Caption added (the table had none): '{perf_title}'.", "Consistency with "
             "Chapter 3.")
    pair_of_row, current = {}, None
    for row in t42.rows[1:]:
        label = row.cells[0].text.strip()
        current = next((k for k, v in PAIRS.items() if v == label), current)
        pair_of_row[row] = (current, row.cells[1].text.strip())
    assert sorted(pair_of_row.values()) == sorted(m.index), pair_of_row.values()
    changed = 0
    for row, (pair, model) in pair_of_row.items():
        r = m.loc[(pair, model)]
        for col, key, fmt, f in ((2, "acc", pct, max), (3, "auc", f3, max), (4, "ll", f3, min)):
            cell, value, bold = row.cells[col], fmt(r[key]), best(pair, key, f) == model
            if cell.text.strip() != value or cell_is_bold(cell) != bold:
                set_cell(e, cell, value, bold=bold)
                changed += 1
    if changed:
        note(f"4.2 / Table {t_perf}", f"{changed} of 27 cells changed: the values of the rerun, with the best value "
             "per pair and measure in bold (XGBoost throughout).", rerun)
    note(f"4.3 / Tables {t_class}–{t_conf}", f"Tables are numbered consecutively after Chapter 3 (Tables 1–3), as the "
         f"figures are, so the class distribution becomes Table {t_class} (was 4.1) and the confusion matrices Table "
         f"{t_conf} (was 4.4). In the thesis, insert the captions with References > Insert Caption so that they are "
         f"numbered fields and appear in the List of Tables (in this extract all caption numbers are typed text, "
         f"because the fields would restart from 1 in a single chapter).", "Consistency with Chapter 3.")

    compose(e, find_para(body, startswith="The subsections that follow examine"), [
        # Author, 2 November 2026: "shorten explanation and polish the language to formal academic writing on the
        # graph descriptions in section 4.2"
        "For each pair, the analysis considers whether the advantage of XGBoost lies in ranking, in the recognition "
        "of both leadership states or in probability quality, and whether it persists across the three 52-week "
        "periods between tuning points (Section 3.8.2). The first figure for each pair plots the ROC curves: the "
        "proportion of weeks led by the first factor that are correctly classified (true-positive rate) against the "
        "proportion led by the second factor that are misclassified (false-positive rate), with the diagonal "
        "representing random classification. The second plots the weekly predicted probability that the first "
        "factor leads, with the realised class and the 0.5 threshold. A forecast is confidently wrong when it "
        "assigns a probability above 0.9 to the outcome that did not occur."])
    note("4.2", "Reading guide after Table 4: it now opens with the questions asked of each pair (does XGBoost's lead "
         "lie in ranking, in recognising both leadership states or in its probabilities, and does it hold in each "
         "period between tuning points?), replacing 'The subsections that follow examine, for each pair, where the "
         "more flexible models gain or lose against the linear benchmark'. The figure guide is shortened: the axes of "
         "the ROC curve in one sentence (Section 3.9 does not define them), the probability figures and 'confidently "
         "wrong'; 'The left of each curve therefore reflects …' deleted. Shortened and written in a formal register.",
         "Author: analytical rather than descriptive; shorten and polish the graph descriptions.")
    section_ends = [el for el in between(body, t42._tbl, h421)
                    if el.tag == qn("w:p") and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None]
    assert len(section_ends) <= 1 and all(not text(s).strip() for s in section_ends)

    # ---------------------------------------------------------------- 4.2.1 momentum versus value
    x01 = pd.concat([x0, x1])
    assert x0.actual.mean() < 0.5 < x1.actual.mean()            # value dominated the first period, momentum the second
    assert (x0.prediction == 1).sum() + (x1.prediction == 0).sum() == 2
    # the first paragraph of Section 4.2.1, which the author has partly reworded in the thesis
    first_para = h421.getnext()
    while not text(first_para).strip():
        first_para = first_para.getnext()
    assert text(first_para).startswith(("Momentum versus value provides the strongest",
                                        "For momentum versus value, XGBoost is the only model",
                                        "For momentum versus value, XGBoost's advantage lies")), text(first_para)[:80]
    cmv = ci.loc[(mv, "XGBoost")]
    compose(e, first_para, [
        f"For momentum versus value, XGBoost's advantage lies in ranking and classification rather than in its "
        f"probabilities. It is the only model with a ROC-AUC well above 0.5 ({f3(xmv.auc)}), and it has the highest "
        f"accuracy ({pct(xmv.acc)}), consistent with evidence that tree-based models improve style rotation on the "
        f"JSE ", (["Pag241"], "(Page, McClelland, & Auret, 2024)"),
        f". Its log-loss ({f3(xmv.ll)}), however, is above 0.693, the value for a forecast that gives each factor a "
        f"50% chance every week.",
        footnote(e, document,
                 "Log-loss averages (−ln p) over the 156 weeks, where p is the probability the model gave to the "
                 "outcome that occurred. A forecast of 0.5 in every week, which shows a neutral choice, therefore "
                 "scores −ln 0.5 = ln 2 ≈ 0.693 whatever happens. A model with a higher log-loss has assigned its "
                 "probabilities worse than this uninformative forecast, typically because it was confident about "
                 "outcomes that did not occur."),
        f" Logistic regression and the GAM are close to random (ROC-AUC {f3(lmv.auc)} and {f3(gmv.auc)}), with "
        f"log-losses far higher still ({f3(lmv.ll)} and {f3(gmv.ll)})."])
    # the graph descriptions: shortened and in a formal register (author, 2 November 2026)
    figure_text("roc", mv, [
        f"In Figure {fig('roc', mv)}, the XGBoost curve is the highest at most false-positive rates above "
        f"approximately 0.25. Its within-period ROC-AUCs ({f2(w_xmv[0])}, {f2(w_xmv[1])} and {f2(w_xmv[2])}), however, "
        f"are all below the pooled {f3(xmv.auc)}, indicating that part of its discrimination reflects differences "
        f"between periods ", faw,
        f". XGBoost therefore provides modest timing within periods, whereas logistic regression remains below 0.5 "
        f"in every period."])
    figure_text("prob", mv, [
        f"Figure {fig('prob', mv)} shows that, until January 2024, XGBoost assigned probabilities between "
        f"{f2(x01.probability.min())} and {f2(x01.probability.max())} under the smallest learning rate and, from "
        f"January 2023, the largest L2 penalty in its grid (Table C.6), predicting the dominant factor of each period "
        f"in all but two weeks. Its accuracy ({pct(xmv.acc)}) is therefore close to the {pct(hind[mv])} of a "
        f"hindsight rule that predicts, in every week, the factor that led most often in that calendar year, and "
        f"derives mainly from identifying the prevailing leader rather than from week-to-week timing. Its log-loss "
        f"exceeds 0.693 because its probabilities widened to between {f2(x2.probability.min())} and "
        f"{f2(x2.probability.max())} in the third period. Logistic regression and the GAM, which selected their "
        f"weakest penalties at every tuning point ", (["Zou05", "Woo17"], "(Zou & Hastie, 2005; Wood, 2017)"),
        f", were confidently wrong in {len(l_conf)} and {len(g_conf)} weeks, accounting for {pct_r(l_share)} and "
        f"{pct_r(g_share)} of their log-loss."])
    note("4.2.1", "Opening: the author's version in the thesis kept (the ROC-AUC interval clause, which she removed, is "
         "not restored), with her wording of the 0.693 footnote.", "Author's transplant of 2 November, 22:46.")
    note(f"4.2.1 / Figure {fig('roc', mv)}", "Graph description shortened and written in a formal register: the "
         "XGBoost curve is the highest above a false-positive rate of about 0.25, but its within-period ROC-AUCs are "
         f"lower than the pooled value and decline ({f2(w_xmv[0])}, {f2(w_xmv[1])}, {f2(w_xmv[2])}), so part of the "
         "pooled discrimination reflects differences between periods (Fawcett, 2006); logistic regression is below "
         "0.5 in every period. 'its curve lies below the diagonal over about half the range' deleted (the period "
         "result makes the same point).", "Author: shorten and polish the graph descriptions.")
    note(f"4.2.1 / Figure {fig('prob', mv)}", "Graph description shortened and written in a formal register. 'a rule "
         "that requires hindsight' is restored: Section 4.2.3 refers to 'the hindsight rule of Section 4.2.1', which "
         "needs it. Deleted: the third-period log-loss (0.832) and 'in the first two it was below 0.693' (the "
         "widened probabilities make the point) and the qualification on the penalty, as in the author's version; "
         "the penalties are now stated alongside the confident errors rather than as their cause, and their values "
         "(λ = 0.01; λ = 1 with 10 splines) are left to Table C.6.",
         "Author: shorten and polish the graph descriptions.")

    # ---------------------------------------------------------------- 4.2.2 momentum versus quality
    cmq, (hit_m, all_m), (hit_q, all_q) = ci.loc[(mq, "XGBoost")], found(mq, "XGBoost"), found0(mq, "XGBoost")
    compose(e, find_para_any(body, "Logistic regression is the strongest model on all three measures",
                             "For momentum versus quality, XGBoost's lead in accuracy"), [
        f"For momentum versus quality, XGBoost's lead in accuracy overstates its usefulness, because it recognised "
        f"only one leadership state. It leads on all three measures, but its ROC-AUC ({f3(xmq.auc)}) exceeds logistic "
        f"regression's ({f3(lmq.auc)}) by only {f3(xmq.auc - lmq.auc)}, and the GAM is last on all three. XGBoost "
        f"identified {hit_m} of the {all_m} momentum-led weeks but only {hit_q} of the {all_q} quality-led weeks, so "
        f"its accuracy reflects predicting momentum, the more frequent leader, rather than recognising shifts towards "
        f"quality."])
    q3 = [w_mq[k][2] for k in MODELS]
    assert min(q3) >= 0.70
    figure_text("roc", mq, [
        f"In Figure {fig('roc', mq)}, logistic regression has the highest curve at false-positive rates below "
        f"approximately 0.55 and XGBoost above that point, so neither model dominates. Ranking is also unstable "
        f"across periods: all three models fall below 0.4 in the second period and reach {f2(min(q3))} to "
        f"{f2(max(q3))} in the third, and only XGBoost ranks the first period well ({f2(w_mq['XGBoost'][0])}, "
        f"against {f2(w_mq['Logistic'][0])} and {f2(w_mq['GAM'][0])}). This common reversal suggests a change in the "
        f"relationship between the predictors and factor leadership ", (["Ang12"], "(Ang & Timmermann, 2012)"), "."])
    figure_text("prob", mq, [
        f"Figure {fig('prob', mq)} identifies the source of this asymmetry. Until January 2024, under the smallest "
        f"learning rate and largest L2 penalty in its grid (Table C.6), XGBoost assigned probabilities between "
        f"{f2(q01.min())} and {f2(q01.max())} and therefore predicted momentum in every week. Although its class "
        f"prediction was constant, its probabilities still ranked the first period well, so its accuracy was "
        f"determined by the threshold and the prevailing leader rather than by its ranking. Logistic regression "
        f"(λ = 10) was confidently wrong once; the GAM, whose smoothing penalties leave the slopes of its "
        f"near-linear terms unshrunk ", woo,
        f", produced more extreme probabilities and was confidently wrong in {num(len(gq_conf))} weeks."])
    note("4.2.2", "Opening: the author's version in the thesis kept (the ROC-AUC interval clause, which she removed, is "
         f"not restored): XGBoost's lead in accuracy overstates its usefulness, because it identified {hit_m} of "
         f"{all_m} momentum-led weeks but only {hit_q} of {all_q} quality-led weeks. The counts are given in the text "
         "because Table 6 comes later (no forward reference).", "Author's transplant of 2 November, 22:46.")
    note(f"4.2.2 / Figure {fig('roc', mq)}", "Graph description shortened and written in a formal register: neither "
         "curve dominates; all three models fall below 0.4 in the second period and reach 0.70 or more in the third, "
         f"and only XGBoost ranks the first period well ({f2(w_mq['XGBoost'][0])}); the common reversal suggests a "
         "change in the relationship (Ang & Timmermann, 2012). 'although the results do not identify the regime or "
         "its cause' deleted ('suggests' carries the qualification).",
         "Author: shorten and polish the graph descriptions.")
    note(f"4.2.2 / Figure {fig('prob', mq)}", "Graph description shortened and written in a formal register: XGBoost's "
         "class prediction was constant until January 2024 although its probabilities still ranked the first "
         "period well, so its accuracy was determined by the threshold and the prevailing leader. Deleted: 'and those "
         "of the second worse than chance' (stated for Figure " + str(fig('roc', mq)) + ") and 'incorrectly in most weeks "
         "of the first period and correctly in 73% of the second'.",
         "Author: shorten and polish the graph descriptions.")

    # ---------------------------------------------------------------- 4.2.3 value versus quality
    assert all(0 < period(vq, "XGBoost", j).prediction.mean() < 1 for j in range(3))       # never a constant forecast
    # each factor predicted in at least a quarter of the weeks of every period, whereas in the momentum pairs one
    # factor was predicted in almost every week of the first two periods
    minority = lambda pair, j: min(period(pair, "XGBoost", j).prediction.mean(),
                                   1 - period(pair, "XGBoost", j).prediction.mean())
    assert all(minority(vq, j) >= 0.25 for j in range(3))
    assert all(minority(p, j) < 0.05 for p in (mv, mq) for j in (0, 1))
    neg = lambda s: s.replace("-", "−")
    hit_v, hit_q2 = found(vq, "XGBoost"), found0(vq, "XGBoost")
    # the most persistent ranking: the highest lowest-period ROC-AUC of the nine model-pair results
    assert max(((p, k) for p in PAIRS for k in MODELS), key=lambda key: min(within(*key))) == (vq, "XGBoost")
    compose(e, find_para_any(body, "Logistic regression is also the strongest model for value versus quality",
                             "Value versus quality gives the clearest evidence"), [
        f"Value versus quality gives the clearest evidence of predictability in the study, because the measures "
        f"agree. XGBoost has the highest accuracy ({pct(xvq.acc)}), the highest ROC-AUC of any model and pair "
        f"({f3(xvq.auc)}) and the lowest log-loss ({f3(xvq.ll)}), and it recognised both leadership states: "
        f"{hit_v[0]} of the {hit_v[1]} value-led weeks and {hit_q2[0]} of the {hit_q2[1]} quality-led weeks. Its "
        f"ROC-AUC interval ({f3(cvq.ci_low)} to {f3(cvq.ci_high)}) is the only one of the nine model–pair results to "
        f"exclude 0.5, and it still does so when widened to {100 * (1 - 0.05 / 9):.1f}% ({f3(cvq.ci_low_bonferroni)} "
        f"to {f3(cvq.ci_high_bonferroni)}), the Bonferroni adjustment for the nine results ",
        (["Har16"], "(Harvey, Liu, & Zhu, 2016)"),
        f". Its advantage over logistic regression, second on ROC-AUC and log-loss, is less certain: the difference "
        f"of {f3(dvq.auc_diff)} has a paired interval of {neg(f3(dvq.ci_low))} to {f3(dvq.ci_high)}. The GAM is "
        f"close to random (ROC-AUC {f3(gvq.auc)})."])
    figure_text("roc", vq, [
        f"In Figure {fig('roc', vq)}, the XGBoost curve is the highest between false-positive rates of approximately "
        f"0.08 and 0.78 and never falls below the diagonal. Its within-period ROC-AUC exceeds 0.6 in every period "
        f"({f2(w_xvq[0])}, {f2(w_xvq[1])} and {f2(w_xvq[2])}), the most persistent ranking in the study, whereas "
        f"logistic regression ranks well only in the first ({f2(w_lvq[0])}, against {f2(w_lvq[1])} and "
        f"{f2(w_lvq[2])}). The GAM curve lies below the diagonal over approximately half of the range."])
    figure_text("prob", vq, [
        f"Figure {fig('prob', vq)} shows that, unlike in the momentum pairs, XGBoost predicted each factor in at least "
        f"a quarter of the weeks of every period and was confidently wrong only once. Its accuracy exceeds the "
        f"hindsight rule of Section 4.2.1 by {100 * margin_vq:.1f} percentage points ({pct(xvq.acc)} against "
        f"{pct(hind[vq])}), compared with at most {100 * margin_mom:.1f} points in the momentum pairs, indicating timing "
        f"within years. The relationship nevertheless weakened, with accuracy declining from {pct(acc_xvq[0])} to "
        f"{pct(acc_xvq[1])} and {pct(acc_xvq[2])} and log-loss rising from {f3(ll_xvq[0])} to {f3(ll_xvq[1])} and "
        f"{f3(ll_xvq[2])} across the three periods. The GAM was confidently wrong in {num(len(gv_conf))} weeks and "
        f"logistic regression in none. The pair therefore provides clearer evidence of above-chance ranking by "
        f"XGBoost than of an advantage over the regularised linear model."])
    note("4.2.3", "Opening: the author's version in the thesis kept (as in the 22:06 draft): the clearest evidence "
         f"because the measures agree; XGBoost recognised both states ({hit_v[0]}/{hit_v[1]} and "
         f"{hit_q2[0]}/{hit_q2[1]}); its interval is the only one to exclude 0.5, also after the Bonferroni adjustment "
         "for nine results (Harvey et al., 2016); its advantage over logistic regression is uncertain (paired "
         f"interval {neg(f3(dvq.ci_low))} to {f3(dvq.ci_high)}; results/model_outputs/bootstrap/"
         "auc_paired_differences.csv).", "Author's transplant of 2 November, 22:46.")
    note(f"4.2.3 / Figure {fig('roc', vq)}", "Graph description shortened and written in a formal register: XGBoost's "
         "within-period ROC-AUC exceeds 0.6 in every period, the most persistent ranking in the study (the highest "
         "lowest-period value of the nine results); logistic regression ranks well only in the first period ('so its "
         "second place rests on a single year' deleted).",
         "Author: shorten and polish the graph descriptions.")
    note(f"4.2.3 / Figure {fig('prob', vq)}", "Graph description shortened and written in a formal register: XGBoost "
         "predicted both factors in every period and beat the hindsight rule, which indicates timing within years; "
         f"the relationship weakened over time (accuracy {', '.join(pct(v) for v in acc_xvq)}; log-loss "
         f"{', '.join(f3(v) for v in ll_xvq)} by period). Deleted: 'eight of them between May and August 2023, when "
         "value led but it gave value leadership a "
         "probability below 0.1' (the count of confident errors suffices). The citation (Friedman, 2001; Chen, "
         "Pelger, & Zhu, 2019) in the author's earlier version belonged to a deleted sentence on XGBoost's shrinkage; "
         "Chen, Pelger and Zhu (2019) is a deep-learning paper in any case (XGBoost is Chen & Guestrin, 2016, tag "
         "Che16).", "Author: shorten and polish the graph descriptions.")
    note("4.2", "Section 4.2 rewritten in analytical form: each paragraph opens with a finding, then gives the evidence "
         "and its qualification (results/Chapter4_Analytical_Review.md). The graph descriptions are shortened and "
         "written in a formal academic register (impersonal constructions, 'approximately', no contractions or "
         "colloquial verbs), with each figure cited as evidence rather than described. Secondary statistics (counts "
         "of the highest- and lowest-ranked weeks, single-week examples) and the discussion of the CPCV tuning "
         "surface are omitted.",
         "Author: analytical rather than descriptive; shorten and polish the graph descriptions.")

    # The figures themselves: the current renders from the rerun outputs
    swapped = []
    for (kind, pair), (n, cap) in sorted(figs.items(), key=lambda kv: kv[1][0]):
        pic = cap.getprevious()
        (run, part, _), = drawing_runs([pic], document)
        png = FIGURES / f"figure_final_{'roc' if kind == 'roc' else 'probabilities'}_{pair.lower()}.png"
        if part.blob != png.read_bytes():
            swap_picture(run, part, png)
            for r in cap.iter(qn("w:r")):
                e.mark(r)
            swapped.append(n)
    if swapped:                                  # the others are already the current renders in the thesis
        note(f"4.2 / Figures {swapped[0]}–{swapped[-1]}", "Pictures replaced by the renders from the rerun outputs "
             "(captions highlighted; caption text unchanged).", rerun + " build_chapter4_corrected_figures.py.")
    note("4.2 / citations", "Every citation in the rewritten paragraphs is a Word citation field (Source Manager "
         "tags Hos13, Isr20, Pag241, Faw06, Zou05, Woo17, Ang12, Har16), so the paragraphs can be copied into the "
         "thesis as they are and the reference list updates with them.", "Citations.")
    for old, new in heads.items():
        e.replace(head_els[old], old, new)
    if heads:
        note("Headings", "Headings in sentence case, as in Chapters 1 and 3: "
             + ", ".join(f"'{new}'" for new in heads.values()) + ".", "Consistency.")

    # ---------------------------------------------------------------- 4.3
    p = find_para(body, startswith="Table 4.1 reports the realised class distribution")
    e.replace(p, " These counts provide the context required to interpret the confusion matrices reported below.", "")
    e.replace(p, "Table 4.1", f"Table {t_class}")
    find_para(body, contains="The three targets are reasonably balanced, although momentum leadership is slightly "
                             "more common in the two momentum pairs.")
    t41 = next(t for t in document.tables if text(t._tbl).startswith("Factor pairNegative observations"))
    assert all(not text(el).strip() for el in between(body, p, t41._tbl))        # the table follows the paragraph
    for row in t41.rows[1:]:
        pair = next(k for k, v in PAIRS.items() if v == row.cells[0].text.strip())
        r = m.loc[(pair, "XGBoost")]
        for col, value in ((1, str(int(r.TN + r.FP))), (2, str(int(r.FN + r.TP))), (3, pct(r.act_pos))):
            if row.cells[col].text.strip() != value:
                set_cell(e, row.cells[col], value)
    table_caption(e, conf_caption, t41._tbl, f"Table {t_class}: Realised class distribution in the 156-week "
                                             f"evaluation block")
    note(f"4.3 / Table {t_class}", f"Momentum vs quality and value vs quality rows updated (their labels changed with "
         f"the corrected Quality series). Renumbered (was 4.1) and given a caption (it had none). The sentence on "
         f"balance still holds.", rerun)
    note("4.3", "Deleted 'These counts provide the context required to interpret the confusion matrices reported "
         "below.'", "Author: say it once.")

    e.replace(find_para(body, startswith="Table 4.4 reports the confusion-matrix counts"), "Table 4.4", f"Table {t_conf}")
    t44 = next(t for t in document.tables if text(t._tbl).startswith("Factor pairModelTNFPFNTP"))
    current = None
    for row in t44.rows[1:]:
        label = row.cells[0].text.strip()
        current = next((k for k, v in PAIRS.items() if v == label), current)
        r = m.loc[(current, row.cells[1].text.strip())]
        for col, value in zip(range(2, 8), (r.TN, r.FP, r.FN, r.TP, r.pred_pos, r.act_pos)):
            value = str(int(value)) if col < 6 else pct(value)
            if row.cells[col].text.strip() != value:
                set_cell(e, row.cells[col], value)
    e.replace(conf_caption, "Table 4.4. ", f"Table {t_conf}: ")
    note(f"4.3 / Table {t_conf}", "Confusion-matrix counts and predicted- and actual-positive shares replaced. "
         "Renumbered (was 4.4) in the text and the caption, which follows the thesis format 'Table N: Title'.", rerun)

    lr_mq, lr_vq = m.loc[(mq, "Logistic")], m.loc[(vq, "Logistic")]
    rewrite(e, find_para(body, startswith="XGBoost provides the most balanced class treatment"),
            f"Matching the realised class share does not mean that both classes are recognised. Logistic regression's "
            f"predicted share for momentum versus quality ({pct(lr_mq.pred_pos)}) is the closest to the realised "
            f"{pct(lr_mq.act_pos)}, yet it identified only {found0(mq, 'Logistic')[0]} of the "
            f"{found0(mq, 'Logistic')[1]} quality-led weeks. Only XGBoost, for momentum versus value and value versus "
            f"quality, identified more than half of the weeks of each class (Table {t_conf}).")
    note("4.3", "'XGBoost provides the most balanced class treatment …' replaced: a predicted-positive share close to "
         "the realised share does not show that both classes are recognised (logistic regression for momentum vs "
         "quality is closest in share but identified fewer than half the quality-led weeks). Now judged by the share "
         "of each class identified (from Table " + str(t_conf) + "): only XGBoost for momentum vs value and value vs "
         "quality identifies more than half of each class.", rerun + " Analytical review, Section 4.3.")
    xr = m.loc[(mq, "XGBoost")]
    assert ordered[:2] == [(mq, "XGBoost"), (vq, "Logistic")]       # the two largest imbalances
    rewrite(e, find_para(body, startswith="The extreme positive-class tendencies"),
            f"The 0.5 threshold can also work against a model's ranking. Logistic regression predicted value leadership "
            f"against quality in only {pct(lr_vq.pred_pos)} of weeks, against {pct(lr_vq.act_pos)} realised, and "
            f"identified only {found(vq, 'Logistic')[0]} of the {found(vq, 'Logistic')[1]} value-led weeks, so that "
            f"its second-highest ROC-AUC for that pair translates into an accuracy of only {pct(lr_vq.acc)}, the "
            f"reverse of XGBoost for momentum versus quality (Section 4.2.2).")
    note("4.3", "Paragraph on the two largest imbalances rewritten around logistic regression for value vs quality, "
         "whose threshold works against its ranking (too few positive predictions), as the reverse of XGBoost for "
         "momentum vs quality, which Section 4.2.2 already explains (not repeated here). 'the result of its constant "
         "forecast' deleted (its class decision, not its probabilities, was constant).",
         rerun + " Analytical review, Section 4.3. Author: limit repetition.")
    lq = [m.loc[(p, "Logistic"), "ll"] for p in (mq, vq)]
    p_prob = find_para(body, startswith="The probability results introduce")
    compose(e, p_prob, [
        f"Three results have a log-loss below the uninformative 0.693 (Section 4.2.1): XGBoost for momentum versus "
        f"quality and value versus quality, and logistic regression for value versus quality ({f3(lr_vq.ll)}). "
        f"Logistic regression stays close to 0.693 for the two quality pairs ({f3(min(lq))} and {f3(max(lq))}), "
        f"where it selected a moderate penalty, but reaches {f3(lmv.ll)} for momentum versus value, where it "
        f"selected the weakest; the GAM's log-loss lies between {f3(min(lls['GAM']))} and {f3(max(lls['GAM']))}. "
        f"Probability quality under log-loss therefore varies with the shrinkage each model selected (Sections "
        f"4.2.1–4.2.3), consistent with shrinkage lowering the variance of the estimates and with it the risk of "
        f"extreme probabilities ", (["GJa21"], "(James, Witten, Hastie, & Tibshirani, 2021)"),
        ", although the comparison does not isolate the penalty as the cause."])
    note("4.3", "Probability-quality paragraph corrected and rewritten. Corrected: 'Only XGBoost has a log-loss below "
         f"the uninformative 0.693' was false: logistic regression for value vs quality is also below ({f3(lr_vq.ll)}); "
         "three results are below. Logistic regression's log-loss varies with the penalty it selected; the GAM's is "
         "the highest. 'depends more on the shrinkage … than on its flexibility' replaced by 'varies with the "
         "shrinkage each model selected', with the qualification that the comparison does not isolate the penalty. "
         "Deleted: 'An uninformative forecast of 0.5 … approximately 0.693' (explained in the footnote in Section "
         "4.2.1) and 'better discrimination does not necessarily imply better probability quality' (replaced by the "
         "next paragraph). James et al. (2021), tag GJa21.", rerun + " Analytical review, Section 4.3.")
    synthesis = fresh_copy(p_prob)
    p_prob.addnext(synthesis)
    compose(e, synthesis, [
        "Ranking, the recognition of both classes and probability quality therefore do not coincide; only for value "
        "versus quality does XGBoost combine all three. Which of these matters depends on how the forecasts would be "
        "used: for ordering allocation decisions, for switching between styles or for probability-weighted exposures "
        "(Section 3.9)."])
    note("4.3", "Closing paragraph added: ranking, the recognition of both classes and probability quality do not "
         "coincide (XGBoost combines all three only for value vs quality), and which matters depends on how the "
         "forecasts would be used, as Section 3.9 sets out (binary switching versus probability-weighted exposures).",
         "Analytical review, Section 4.3: connect the measures.")

    # ---------------------------------------------------------------- 4.4 feature importance
    rewrite(e, find_para(body, startswith="Global feature importance is measured"),
            f"Figures {shap_first}–{shap_first + 2} show the ten features with the largest global SHAP importance "
            f"for each model. The values describe the refits on the initial tuning sample (Section 3.10), so they show "
            f"which information each model uses, not why its walk-forward performance changed between periods. The "
            f"comparison of interest is whether the models that predict better rely on different information.")
    note("4.4", "Opening reduced to a reference to Section 3.10, which explains how the figures are read (the ten "
         "leading features, ranks rather than magnitudes, colours by the feature categories of Table 1) and the "
         "limits of the interpretation, followed by the question the section answers: whether the models that "
         "predict better rely on different information. One consequence of Section 3.10 is stated because the "
         "section depends on it: the refits cannot explain why walk-forward performance changed between periods. "
         "Deleted: the definition of global importance, the caveats (Kumar et al., 2020; Molnar, 2022), the "
         "comparison and colour conventions, 'This distinction is important because all final ROC-AUC values remain "
         "below 0.60' (no longer true) and the reference to 'Table 1 of Section 3.6'.",
         "Author: say it once; analytical rather than descriptive. Analytical review, Section 4.4.")
    compose(e, find_para(body, startswith="Logistic regression is dominated by currency-volatility"), [
        "The three models agree on the leading type of information but not on what follows it. A relative-volatility "
        "measure ranks first in all three: that of momentum and value for XGBoost, of "
        "value and quality for logistic regression and of momentum and quality for the GAM. Logistic regression "
        "and the GAM then share three South African macroeconomic variables in their top six, medium-term USDZAR "
        "forward pricing, USDZAR entropy and rand PPP valuation, whereas XGBoost draws on the momentum–quality "
        "correlation, South African real-yield-curve curvature, the value–quality volatility ratio, global bond "
        "volatility and the commodity index component. The prominence of relative volatility is consistent with "
        "evidence that momentum's risk varies predictably over time and that its crashes follow volatile markets ",
        (["Bar15", "Dan16"], "(Barroso & Santa-Clara, 2015; Daniel & Moskowitz, 2016)"),
        ". Since only XGBoost has a ROC-AUC well above 0.5 for this pair (Section 4.2.1), the shared leading feature "
        "does not by itself deliver prediction; XGBoost's advantage may come from the variables the other two models "
        "do not use and from how it combines them."])
    note("4.4.1", "Paragraph rewritten from the new rankings (Figure " + str(shap_first) + ") in analytical form: the "
         "models agree on the leading type of information (relative volatility) but only XGBoost predicts well, so "
         "the shared feature is not sufficient; its advantage may come from the variables only it uses and how it "
         "combines them (stated as a possibility). The citations Ehsani & Linnainmaa (2022) and Gupta & Kelly (2019) "
         "on factor momentum are dropped here, because short-horizon relative returns no longer rank among the "
         "leading features for this pair (factor momentum is cited in Section 4.4.3). Barroso & Santa-Clara (2015) "
         "and Daniel & Moskowitz (2016) kept, as citation fields.", rerun + " Analytical review, Section 4.4.")
    xs_mq = sel[(mq, "XGBoost")]
    assert all(xs_mq[j]["learning_rate"] == grid_max["lr_min"] and xs_mq[j]["reg_lambda"] == grid_max["reg_max"]
               for j in (0, 1))                         # the SHAP refit uses the setting of the first tuning point
    compose(e, find_para(body, startswith="Momentum-versus-quality predictions use"), [
        f"The two models closest in performance also use the most similar information. Logistic regression and "
        f"XGBoost, whose ROC-AUCs differ by only {f3(xmq.auc - lmq.auc)}, are led by the same two features, the change "
        f"in the TOP40 price-to-EBITDA ratio and South African real-yield-curve curvature, and the commodity index "
        f"component is in the top four of both. That a valuation measure leads both is in line with the debate on "
        f"valuation-based factor timing ",
        (["Arn16", "Asn17"], "(Arnott, Beck, Kalesnik, & West, 2016; Asness, Chandra, Ilmanen, & Israel, 2017)"),
        ", but with a ranking that reverses between periods (Section 4.2.2) it does not show that valuation times "
        "this pair reliably. The GAM, the weakest model for the pair, is led instead by factor-specific variables, "
        "the value–quality correlation and three relative-volatility measures, with the EBITDA measure seventh. "
        "XGBoost's SHAP values for this pair are small, the ten largest summing to less than a sixth of those for "
        "the other two pairs, consistent with the strong shrinkage it selected at the first two tuning points, which "
        "also kept its walk-forward probabilities in a narrow range (Section 4.2.2)."])
    note("4.4.2", "Paragraph rewritten from the new rankings (Figure " + str(shap_first + 1) + ") in analytical form: "
         "the two models closest in performance share the leading features; the valuation link is qualified by the "
         "ranking that reverses between periods; the GAM, the weakest model, uses different information. 'consistent "
         "with the near-constant forecasts of its initially selected setting' replaced by the strong shrinkage of "
         "the setting selected at the first two tuning points (the SHAP refit uses the first). The EBITDA measure "
         "was described as a 'local-versus-global' differential; it is the first difference of the TOP40 "
         "price-to-EBITDA ratio (Appendix A). Arnott et al. (2016) and Asness et al. (2017) kept, as citation "
         "fields.", rerun + " Appendix A. Analytical review, Section 4.4.")
    compose(e, find_para(body, startswith="The standardised value-versus-quality relative-performance measure"), [
        "Recent relative performance matters most for this pair, the one XGBoost predicts best. The three-month "
        "relative return of value and quality ranks first for XGBoost and second for logistic regression, and that "
        "of momentum and quality is in the top five of both, consistent with factor momentum ",
        (["Gup19", "Arn23"], "(Gupta & Kelly, 2019; Arnott, Kalesnik, & Linnainmaa, 2023)"),
        ", although global importance does not show whether recent outperformance predicted continuation or "
        "reversal. The value–quality volatility ratio is in the top four of all three models. Beyond these, logistic "
        "regression and the GAM share USDZAR entropy and global property volatility in their top three, whereas "
        "XGBoost draws on gold's rolling volatility and US and South African yield-curve curvature. Since logistic "
        "regression shares the relative-performance features but ranks well only in the first period (Section "
        "4.2.3), these features alone do not account for XGBoost's more persistent ranking."])
    note("4.4.3", "Paragraph rewritten from the new rankings (Figure " + str(shap_first + 2) + ") in analytical form: "
         "the factor-momentum reading is qualified (global importance has no direction), and the shared "
         "relative-performance features do not account for XGBoost's more persistent ranking, since logistic "
         "regression uses them too. Deleted: 'Logistic regression is the best-performing model for this pair, but "
         "its AUC of 0.585 remains modest. The feature rankings should therefore be interpreted as a description of "
         "the fitted decision function rather than evidence of stable economic causality.' (no longer true; the "
         "caveat is in Section 3.10).", rerun + " Analytical review, Section 4.4.")

    for old, new, pair in (("4.3", shap_first, mv), ("4.4", shap_first + 1, mq), ("4.5", shap_first + 2, vq)):
        name = PAIRS[pair].replace(' vs ', ' versus ').lower()
        caption = find_para(body, contains=f"SHAP importance for {name}")
        picture = caption.getprevious()
        while not list(picture.iter(A_NS + "blip")):
            picture = picture.getprevious()
        (run, part, _), = drawing_runs([picture], document)
        swap_picture(run, part, FIGURES / f"figure_final_shap_{pair.lower()}.png")
        e.replace(caption, f"Figure {old}. ", f"Figure {new}: ")
        e.replace(caption, f"for {name}.", f"for {name}")
        for r in caption.iter(qn("w:r")):
            e.mark(r)
    shap_range = f"{shap_first}–{shap_first + 2}"
    note(f"4.4 / Figures {shap_range}", "SHAP figures replaced by the renders from the rerun outputs, in the thesis "
         "colours with no title inside the figure, and renumbered (were 4.3–4.5) to follow Figures 9–14, in the "
         "caption format 'Figure N: Title' without a full stop.", rerun + " build_chapter4_corrected_figures.py.")

    # ---------------------------------------------------------------- 4.5 chapter conclusion
    xb = boot.loc["XGBoost mean AUC"]
    p = find_para(body, startswith="The final results provide modest")
    opening = "The final results provide modest and factor-pair-dependent evidence of predictability. "
    assert text(p).startswith(opening)
    compose(e, p, [
        opening + f"XGBoost has the highest accuracy and ROC-AUC and the lowest log-loss for every pair, and its mean "
        f"ROC-AUC across the three pairs ({f3(means.auc['XGBoost'])}) exceeds those of logistic regression "
        f"({f3(means.auc['Logistic'])}) and the GAM ({f3(means.auc['GAM'])}); it is the only model whose "
        f"block-bootstrap 95% interval for the mean ROC-AUC excludes 0.5 ({f3(xb.ci_low)} to {f3(xb.ci_high)}; "
        f"Section 3.9). The ordering does not carry equal weight in every pair. Only for value versus quality does "
        f"XGBoost's ranking remain above chance after adjustment for the nine model–pair results and persist across "
        f"the periods (Section 4.2.3); for the momentum pairs its lead rests on identifying the prevailing leader "
        f"and, for momentum versus quality, on predicting mainly one factor (Sections 4.2.1 and 4.2.2)."])
    note("4.5", "Model ranking rewritten in analytical form: XGBoost leads every pair on every measure and is the only "
         "model whose interval for the mean ROC-AUC excludes 0.5, but the ordering does not carry equal weight in "
         "every pair: only value vs quality is above chance after the adjustment and persistent; the momentum-pair "
         "leads rest on identifying the prevailing leader and, for momentum vs quality, on predicting mainly one "
         "factor. The Bonferroni interval and Harvey et al. (2016) moved to Section 4.2.3 (stated once). 'The result "
         "is therefore not merely the best of many tests' dropped: the adjustment covers the nine reported results, "
         "not the decisions taken while developing the study (analytical review). 'XGBoost's forecast was in effect "
         "constant for two of the three years' replaced (its class decision, not its probabilities, was constant). "
         "'several models produce strongly asymmetric classifications at the 0.5 threshold' deleted (Section 4.3). "
         "The author's first sentence is unchanged.",
         rerun + " Analytical review, Section 4.5.")
    p = find_para(body, startswith="The findings therefore qualify")
    head = ("The findings therefore qualify the proposition that increasing model flexibility necessarily improves "
            "out-of-sample prediction (Gu et al., 2020; Kelly et al., 2024). ")
    tail = "The evidence supports limited predictive structure rather than a strongly separable factor-timing signal."
    assert text(p).startswith(head) and text(p).rstrip().endswith(tail), text(p)
    compose(e, p, [
        head + f"Performance does not rise steadily with flexibility. The fitted GAMs do not improve consistently on "
        f"logistic regression, and on 156 weekly observations XGBoost's advantage in mean ROC-AUC over the "
        f"regularised linear model is not statistically established ({f3(xl.estimate)}, 95% interval {f3(xl.ci_low)} "
        f"to {f3(xl.ci_high)}), whereas its advantage over the GAM has an unadjusted paired interval that excludes "
        f"zero ({f3(xg.estimate)}, {f3(xg.ci_low)} to {f3(xg.ci_high)}). The two flexible models also differ in "
        f"more than nonlinearity: "
        f"XGBoost can represent interactions between predictors, which the additive GAM cannot ".replace("-0.", "−0."),
        (["Has86", "JHF01"], "(Hastie & Tibshirani, 1986; Friedman, 2001)"),
        ", and its tuning can shrink each tree's contribution, whereas the GAM's penalty leaves the linear part of "
        "each term unpenalised ", woo,
        ". The comparison cannot isolate either difference. A similar pattern has been reported for US stock "
        "returns, where an additive spline model fails to improve on a linear one and the gains of trees and neural "
        "networks are traced to interactions between predictors ", (["Shi20"], "(Gu, Kelly, & Xiu, 2020)"),
        ", and tree ensembles remain the strongest learners on tabular data of moderate size ",
        (["Gri22"], "(Grinsztajn, Oyallon, & Varoquaux, 2022)"),
        "; these studies offer an interpretation of XGBoost's advantage rather than evidence of its mechanism here. "
        "Across the three models, the specifications with the least shrinkage, logistic regression for momentum "
        "versus value and the GAM, produced overconfident probabilities and the poorest log-loss (Section 4.3), and "
        "the validation data gave little basis for choosing between penalties. " + tail])
    note("4.5", "Flexibility conclusion rewritten in analytical form: performance does not rise steadily with "
         "flexibility; XGBoost vs logistic regression uncertain, XGBoost vs GAM an unadjusted paired interval that "
         "excludes zero (the nine-result adjustment applies to the individual ROC-AUCs, not to these comparisons); "
         "the two flexible models differ in interactions and in regularisation, which the comparison cannot isolate. "
         "Narrowed (analytical review): 'smooth nonlinearity alone adds nothing' becomes 'the fitted GAMs do not "
         "improve consistently on logistic regression'; Gu, Kelly and Xiu (2020) and Grinsztajn et al. (2022) are "
         "presented as an interpretation, not as evidence of the mechanism here. The author's first and last "
         "sentences kept (the whole paragraph is highlighted because it now holds citation fields: Has86, JHF01, "
         "Woo17, Shi20, Gri22). 'Even the best AUC is below 0.60' deleted (no longer true). Gu, Kelly and Xiu (2020): "
         "their additive spline model 'fails to robustly outperform the linear specification' and the gains of trees "
         "and neural networks come from predictor interactions (pp. 3 and 27 of the NBER version). "
         "The typed citation '(Gu et al., 2020; Kelly et al., 2024)' in the author's first sentence should become a "
         "citation field (Gu: tag Shi20). Kelly et al. (2024) has no Source Manager entry: Kelly, B., Malamud, S., & "
         "Zhou, K. (2024). The virtue of complexity in return prediction. The Journal of Finance, 79(1), 459–503.",
         rerun + " results/model_outputs/bootstrap/auc_mean_across_pairs.csv. Author: add references.")
    p = find_para(body, startswith="The global SHAP analysis indicates")
    head, tail = text(p).split(" These patterns differ ")
    assert head.endswith("sentiment.") and tail.startswith("across model classes and factor pairs")
    compose(e, p, [
        head + " Only South African real-yield-curve curvature is among the ten leading features of almost every "
        "model and pair (eight of the nine), in line with evidence that yield-curve variables capture business-cycle "
        "variation in expected stock returns ", (["Fam89"], "(Fama & French, 1989)"),
        " and in momentum payoffs ", (["Cho02"], "(Chordia & Shivakumar, 2002)"),
        "; otherwise, the patterns differ " + tail])
    note("4.5", "SHAP summary: South African real-yield-curve curvature added as the one feature common to almost "
         f"all models and pairs (Figures {shap_range}), with the literature linking yield-curve variables to "
         "business-cycle variation in expected returns (Fama & French, 1989, tag Fam89) and in momentum payoffs "
         "(Chordia & Shivakumar, 2002, tag Cho02). The author's other sentences unchanged.",
         rerun + " Author: add references.")

    for tbl in (t42, t41, t44):
        keep_together(tbl._tbl)
    note(f"4.2–4.3 / Tables {t_perf}–{t_conf}", "Each table kept on one page with its caption (its rows are kept "
         "with the next); Table " + str(t_perf) + " was split after its first row.", "Layout.")

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
        note("4.2", f"{emptied} empty section(s) removed after Table {t_perf}. They held the duplicate ROC images deleted "
             "in the thesis and render as blank pages once that deletion is accepted.", "Layout.")

    # A section break after Table 4, left from the old combined ROC figure, would start 4.2.1 on a new page and
    # leave the rest of the page empty. Merge the two sections, keeping the header.
    final = body.find(qn("w:sectPr"))
    for para in section_ends:
        if para.getparent() is None:
            continue
        sect = para.find(qn("w:pPr") + "/" + qn("w:sectPr"))
        kind = sect.find(qn("w:type"))
        kind = "nextPage" if kind is None else kind.get(qn("w:val"))
        note("4.2", f"Section break after Table {t_perf} removed (left from the old combined ROC figure)"
             + ("; Section 4.2.1 no longer starts on a new page." if kind == "nextPage" else "."), "Layout.")
        assert list(body.iter(qn("w:sectPr"))) == [sect, final], "another section follows the old ROC figure"
        for tag in ("w:pgSz", "w:pgMar", "w:titlePg"):
            a, b = sect.find(qn(tag)), final.find(qn(tag))
            assert (a is None) == (b is None) and (a is None or dict(a.attrib) == dict(b.attrib)), tag
        assert final.find(qn("w:headerReference")) is None and final.find(qn("w:footerReference")) is None
        for i, ref in enumerate(sect.findall(qn("w:headerReference")) + sect.findall(qn("w:footerReference"))):
            final.insert(i, ref)
        para.getparent().remove(para)

    add_log(document, log)
    document.save(OUTPUT)
    return log


if __name__ == "__main__":
    log = build()
    print(OUTPUT.name, "changes logged:", len(log))
