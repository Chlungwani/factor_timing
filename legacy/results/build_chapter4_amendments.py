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
    """Replace a paragraph's whole content by highlighted text. `parts` holds strings and, for citations,
    (tags, shown) pairs, which become Word citation fields."""
    template = next(r for r in p.iter(qn("w:r")) if r.find(qn("w:t")) is not None)
    template = plain_run(template)
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    for part in parts:
        if isinstance(part, str):
            run = editor.new_run(template, part)
            rpr = run.find(qn("w:rPr"))
            for colour in ([] if rpr is None else rpr.findall(qn("w:color"))):
                rpr.remove(colour)
            p.append(run)
        else:
            p.append(citation(editor, template, *part))


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
    assert m["auc"].idxmax() == (vq, "XGBoost")
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
    cvq = ci.loc[(vq, "XGBoost")]
    assert xg.ci_low > 0 > xl.ci_low
    assert all(boot.loc[f"{k} mean AUC", "ci_low"] < 0.5 for k in ("Logistic", "GAM")) \
        and boot.loc["XGBoost mean AUC", "ci_low"] > 0.5

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
    unlinked = sum(unlink_seq(p) for p in list(body.iter(qn("w:p"))) if style(p) == "Caption")
    assert unlinked >= 6, unlinked                     # at least the six figure captions of Section 4.2

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

    # 4.1 is in the thesis as the author transplanted it (an outline of Sections 4.2-4.5); it still holds
    find_para(body, startswith="This chapter presents the out-of-sample results of the three models of Section 3.7")

    heads = {"Overview of the Evaluation Framework": "Overview of the evaluation framework",
             "Out-of-Sample predictive performance": "Out-of-sample predictive performance",
             "Momentum versus Value": "Momentum versus value",
             "Feature Importance": "Feature importance"}
    head_els = {old: heading(body, old) for old in heads}
    t42 = next(t for t in document.tables if text(t._tbl).startswith("Factor pairModelAccuracyROC-AUCLog-loss"))
    figs = {}
    for el in body.iter(qn("w:p")):
        hit = re.match(r"Figure (\d+): (Out-of-sample ROC curves for|Walk-forward predicted probability that) "
                       r"(\w+) (?:versus|leads) (\w+)", text(el))
        if hit:
            pair = f"{hit.group(3).capitalize()}_vs_{hit.group(4).capitalize()}"
            figs[("roc" if hit.group(2).startswith("Out") else "prob", pair)] = (int(hit.group(1)), el)
    assert sorted(n for n, _ in figs.values()) == list(range(numbered["Figure"] + 1, numbered["Figure"] + 7)), figs
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
    compose(e, find_para(body, startswith="Table 4 reports the accuracy"), [
        f"Table {t_perf} reports the accuracy, ROC-AUC and log-loss of each model for the three factor pairs. "
        f"XGBoost is the strongest model throughout: for every pair it has the highest accuracy and ROC-AUC and "
        f"the lowest log-loss. The GAM, which adds smooth nonlinear effects to the linear specification, does not "
        f"improve on logistic regression: its ROC-AUC is lower for two of the three pairs and its log-loss is the "
        f"highest for all three. The predictive relationships remain modest: the highest ROC-AUC in the study is "
        f"{f3(m['auc'].max())}, for value versus quality ",
        (["Hos13", "Isr20"], "(Hosmer, Lemeshow, & Sturdivant, 2013; Israel, Kelly, & Moskowitz, 2020)"), "."])
    note("4.2", "Opening rewritten for the new results: XGBoost leads every pair on every measure; the GAM does not "
         "improve on logistic regression (lower ROC-AUC for two pairs, highest log-loss for all three); the highest "
         f"ROC-AUC is now {f3(m['auc'].max())}.", rerun)

    conf_caption = find_para(body, startswith="Table 4.4. Confusion-matrix diagnostics")
    table_caption(e, conf_caption, t42._tbl,
                  f"Table {t_perf}: Out-of-sample accuracy, ROC-AUC and log-loss by factor pair and model")
    pair_of_row, current = {}, None
    for row in t42.rows[1:]:
        label = row.cells[0].text.strip()
        current = next((k for k, v in PAIRS.items() if v == label), current)
        pair_of_row[row] = (current, row.cells[1].text.strip())
    assert sorted(pair_of_row.values()) == sorted(m.index), pair_of_row.values()
    for row, (pair, model) in pair_of_row.items():
        r = m.loc[(pair, model)]
        for col, key, fmt, f in ((2, "acc", pct, max), (3, "auc", f3, max), (4, "ll", f3, min)):
            set_cell(e, row.cells[col], fmt(r[key]), bold=best(pair, key, f) == model)
    note(f"4.2 / Table {t_perf}", "Every value replaced; the best value per pair and measure in bold (XGBoost "
         "throughout). ROC-AUC shown on the zero-to-one scale (0.625) instead of as a percentage, as in the text "
         "and Section 3.9.", rerun)
    note(f"4.2 / Table {t_perf}", f"Caption added (the table had none): 'Table {t_perf}: Out-of-sample accuracy, "
         f"ROC-AUC and log-loss by factor pair and model'. Tables are numbered consecutively after Chapter 3 "
         f"(Tables 1–3), as the figures already are, so the class distribution becomes Table {t_class} (was 4.1) and "
         f"the confusion matrices Table {t_conf} (was 4.4). In the thesis, insert the captions with References > "
         f"Insert Caption so that they are numbered fields and appear in the List of Tables (in this extract all "
         f"caption numbers are typed text, because the fields would restart from 1 in a single chapter).",
         "Consistency with Chapter 3.")

    guide = find_para(body, startswith="The subsections that follow examine")
    e.replace(guide, "The first shows the ROC curves, the vertical axis", "The first shows the ROC curves: the vertical axis")
    note("4.2", "Reading guide after Table 4 kept as the author wrote it; one comma replaced by a colon ('The first "
         "shows the ROC curves: the vertical axis …').", "Punctuation.")
    section_ends = [el for el in between(body, t42._tbl, head_els["Momentum versus Value"])
                    if el.tag == qn("w:p") and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None]
    assert len(section_ends) <= 1 and all(not text(s).strip() for s in section_ends)

    # ---------------------------------------------------------------- 4.2.1 momentum versus value
    compose(e, find_para(body, startswith="Momentum versus value provides the strongest"), [
        f"For momentum versus value, XGBoost is the only model with a ROC-AUC well above 0.5 ({f3(xmv.auc)}), and "
        f"it has the highest accuracy ({pct(xmv.acc)}), consistent with evidence that tree-based models improve "
        f"style rotation on the JSE ", (["Pag241"], "(Page, McClelland, & Auret, 2024)"),
        f". Logistic regression and the GAM are close to random on ROC-AUC ({f3(lmv.auc)} and {f3(gmv.auc)}), and "
        f"their log-losses ({f3(lmv.ll)} and {f3(gmv.ll)}) are far above the uninformative 0.693 (Section 3.9). "
        f"Leadership shifted sharply during the evaluation period: value led in {pct_r(1 - share_y(mv, 2022))} of "
        f"the weeks of 2022, momentum in {pct_r(share_y(mv, 2023))} of those of 2023, and the two factors about "
        f"equally often in 2024."])
    figure_text("roc", mv, [
        f"Figure {fig('roc', mv)} shows that XGBoost's curve is the highest over most of the range, from a "
        f"false-positive rate of about 0.25 to 0.95, whereas logistic regression's lies below the diagonal over "
        f"about half of it. XGBoost's advantage lies in the weeks it ranks lowest: of its 30 lowest-probability "
        f"weeks, {x_bot_2022} fall in 2022 and {xmv.bottom30_neg} were led by value. Its ranking does not rest only "
        f"on telling the years apart, however. Computed separately within each of the three 52-week periods between "
        f"tuning points, its ROC-AUC is {f2(w_xmv[0])}, {f2(w_xmv[1])} and {f2(w_xmv[2])}, whereas logistic "
        f"regression's is below 0.5 in all three ", faw, "."])
    p = discussed("prob", mv)
    second = fresh_copy(p)
    figure_text("prob", mv, [
        f"Figure {fig('prob', mv)} shows how XGBoost's forecasts changed with its tuning. Until the first retune, "
        f"in January 2023, it used the smallest learning rate in its grid: its probabilities stayed between "
        f"{f2(x0.probability.min())} and {f2(x0.probability.max())}, and it predicted value leadership in all but "
        f"one week and was right in {pct_r(acc_of(x0))} of the weeks. The retune added the largest L2 penalty in its grid "
        f"(Table C.6). For the following year its probabilities stayed between {f2(x1.probability.min())} and "
        f"{f2(x1.probability.max())}, and it predicted momentum leadership in all but one week and was right in "
        f"{pct_r(acc_of(x1))}. Both were in effect constant forecasts that followed the dominant factor of the year, "
        f"and XGBoost's overall accuracy of {pct(xmv.acc)} is close to the {pct(hind[mv])} that would be achieved by "
        f"predicting, in every week, the factor that led most often in that calendar year, a rule that requires "
        f"hindsight. Only after the January 2024 retune, which selected its largest learning rate and smallest "
        f"penalty, did its probabilities spread, from {f2(x2.probability.min())} to {f2(x2.probability.max())}; its "
        f"{num(len(x_conf))} confidently wrong weeks all fall in this period."])
    set_ppr(second, body_ppr)
    compose(e, second, [
        "The same figure shows why logistic regression and the GAM have such poor log-loss. At every tuning point "
        "logistic regression selected the weakest penalty in its grid (λ = 0.01), which leaves its coefficients "
        "almost unshrunk ", (["Zou05"], "(Zou & Hastie, 2005)"),
        ", and the GAM its most flexible setting, λ = 1 with 10 splines ", woo,
        f". The probabilities of both therefore range from close to zero to close to one, and they were confidently "
        f"wrong in {len(l_conf)} and {len(g_conf)} weeks, which account for {pct_r(l_share)} and {pct_r(g_share)} of "
        f"their log-loss. In four weeks of May and June 2023, for example, both gave momentum leadership a "
        f"probability of almost one while value led. The validation offered little to choose between penalties: at "
        f"the initial tuning point, the mean ROC-AUC over the ten CPCV combinations ranged only from "
        f"{f3(lg.mean_auc.min())} to {f3(lg.mean_auc.max())} across the 18 logistic settings, and a penalty 10,000 "
        f"times stronger (λ = 100) scored within {f3(flat_gap)} of the one selected. For this pair, then, the gap "
        f"between XGBoost and logistic regression says more about the penalty selected on a nearly flat validation "
        f"surface than about nonlinearity."])
    p.addnext(second)
    note("4.2.1", "Opening rewritten: XGBoost is the only model with a ROC-AUC well above 0.5 for this pair; logistic "
         "regression and the GAM are close to random on ROC-AUC with log-loss far above 0.693. Deleted with the old "
         "results: 'provides the strongest classification result in the study', the GAM's accuracy, 'XGBoost's lead "
         "suggests that nonlinear structure contributes …', 'While results support … factor outperformance', 'The "
         "size of the effect remains moderate …', logistic regression's ROC-AUC and accuracy and its log-loss "
         "explanation, with the citation Hastie & Tibshirani (1986), which introduces GAMs and does not support the "
         "shrinkage claim.", rerun)
    note(f"4.2.1 / Figure {fig('roc', mv)}", "Paragraph rewritten: XGBoost's curve is highest from a false-positive "
         "rate of about 0.25; its advantage lies in the value-led weeks of 2022 it ranks lowest; its ROC-AUC within "
         "each 52-week period between tuning points is 0.55 or more, logistic regression's below 0.5 (Fawcett, 2006).",
         rerun)
    note(f"4.2.1 / Figure {fig('prob', mv)}", "Paragraph (styled as a caption) restyled as body text and rewritten "
         "as two paragraphs. (1) XGBoost's forecasts per tuning period: in effect constant forecasts in the first two "
         "years, so that its accuracy is close to that of predicting each calendar year's most frequent leader with "
         "hindsight (a rule defined here and referred to in Sections 4.2.2–4.2.3); its probabilities spread only "
         "after the January 2024 retune, when all its confident errors occur. (2) Logistic regression (λ = 0.01, "
         "Zou & Hastie, 2005) and the GAM (λ = 1, 10 splines; Wood, 2017) selected their weakest penalties, hence "
         "their confident errors; the CPCV surface for logistic regression was nearly flat, so the gap to XGBoost "
         "reflects the selected penalty more than nonlinearity.", rerun + " Tuning grid: results/model_outputs/"
         "artifacts/logistic_momentum_vs_value_tuning_grid.pkl (a CSV file).")

    # ---------------------------------------------------------------- 4.2.2 momentum versus quality
    compose(e, find_para(body, startswith="Logistic regression is the strongest model on all three measures"), [
        f"Momentum versus quality gives the weakest support for the more flexible models. XGBoost again leads on "
        f"all three measures, but its ROC-AUC ({f3(xmq.auc)}) exceeds that of logistic regression ({f3(lmq.auc)}) "
        f"by only {f3(xmq.auc - lmq.auc)}, and the GAM is last on all three. As for momentum versus value, "
        f"leadership changed between years: quality led in {pct_r(1 - share_y(mq, 2022))} of the weeks of 2022 and "
        f"{pct_r(1 - share_y(mq, 2024))} of those of 2024, momentum in {pct_r(share_y(mq, 2023))} of those of 2023."])
    figure_text("roc", mq, [
        f"Figure {fig('roc', mq)} shows that the two leading models are informative in different weeks. Logistic "
        f"regression's curve is the highest at false-positive rates up to about 0.55, and {lmq.top20_pos} of its 20 "
        f"highest-probability weeks were led by momentum. XGBoost's is the highest from there on, because "
        f"{xmq.bottom30_neg} of its 30 lowest-probability weeks, most of them in 2024, were led by quality. All "
        f"three models rank the weeks of the third period (from January 2024) well, with ROC-AUCs within it of "
        f"{f2(w_mq['Logistic'][2])} (logistic regression), {f2(w_mq['GAM'][2])} (GAM) and {f2(w_mq['XGBoost'][2])} "
        f"(XGBoost), whereas within the second period all three are below 0.4. Most of the ranking information in "
        f"this pair therefore comes from a single year."])
    figure_text("prob", mq, [
        f"Figure {fig('prob', mq)} shows that for two of the three years XGBoost's forecast was in effect constant. "
        f"At the first two tuning points it selected its smallest learning rate and largest L2 penalty (Table C.6): "
        f"its probabilities stayed between {f2(q01.min())} and {f2(q01.max())}, and it predicted momentum "
        f"leadership in every week until January 2024, wrongly in most weeks of the first period and correctly in "
        f"{pct_r(acc_of(q[1]))} of those of the second. Only after the January 2024 retune, which selected its "
        f"largest learning rate and smallest penalty, did its probabilities spread, from "
        f"{f2(q[2].probability.min())} to {f2(q[2].probability.max())}. Its accuracy is therefore below logistic "
        f"regression's in the first period and above it in the second, when the constant forecast matched the "
        f"dominant factor, and in the third; overall ({pct(xmq.acc)}) it is again close to the hindsight rule "
        f"({pct(hind[mq])}). Logistic regression, with a moderate penalty (λ = 10) at every tuning point, was "
        f"confidently wrong only once. The GAM's probabilities follow logistic regression's but are more extreme, "
        f"so that it was confidently wrong in {num(len(gq_conf))} weeks. It selected strong smoothing penalties "
        f"(λ = 1000 or 10000), which make its terms close to linear but leave their slopes unpenalised ", woo,
        ", so that, unlike in logistic regression, nothing shrinks its estimates ", hoe,
        ". Here too, what separates the models is the degree of shrinkage rather than the degree of nonlinearity."])
    note("4.2.2", "Opening rewritten: XGBoost leads on all three measures but its ROC-AUC exceeds logistic "
         "regression's by only " + f3(xmq.auc - lmq.auc) + "; the GAM is last. Deleted with the old results: the "
         "sentences on logistic regression as the strongest model, 'XGBoost again produces the strongest overall "
         "classification results', the log-loss of 0.687 and accuracy of 51.3%, the confusion-matrix sentence and the "
         "GAM's error pattern.", rerun)
    note(f"4.2.2 / Figure {fig('roc', mq)}", "Paragraph rewritten: logistic regression ranks best among the weeks it "
         "ranks highest, XGBoost among those it ranks lowest; all three models rank the third period well and the "
         "second poorly, so most of the information comes from one year. The bias–variance sentence and its "
         "citation (Hastie et al., 2009; Israel et al., 2020) no longer apply and are deleted.", rerun)
    note(f"4.2.2 / Figure {fig('prob', mq)}", "Paragraph restyled as body text and rewritten: XGBoost's forecast was "
         "in effect constant until January 2024; its accuracy is close to the hindsight rule; logistic regression "
         "(λ = 10) is confidently wrong once; the GAM follows logistic regression but, with its slopes unpenalised "
         "(Wood, 2017; Hoerl & Kennard, 1970), is more extreme. 'show log-loss probabilities' replaced.", rerun)

    # ---------------------------------------------------------------- 4.2.3 value versus quality
    compose(e, find_para(body, startswith="Logistic regression is also the strongest model for value versus quality"), [
        f"Value versus quality gives the strongest evidence of predictability in the study, and it comes from the "
        f"nonlinear model. XGBoost has the highest accuracy ({pct(xvq.acc)}), the highest ROC-AUC of any model and "
        f"pair ({f3(xvq.auc)}) and the lowest log-loss ({f3(xvq.ll)}). Logistic regression is second on ROC-AUC and "
        f"log-loss, and the GAM is close to random (accuracy {pct(gvq.acc)}, ROC-AUC {f3(gvq.auc)}). Leadership "
        f"swung less between years than in the momentum pairs: value led in {pct_r(share_y(vq, 2022))} of the "
        f"weeks of 2022, and quality in {pct_r(1 - share_y(vq, 2023))} and {pct_r(1 - share_y(vq, 2024))} of those "
        f"of 2023 and 2024."])
    figure_text("roc", vq, [
        f"Figure {fig('roc', vq)} shows that XGBoost's curve is the highest from a false-positive rate of about 0.08 "
        f"to 0.78 and never falls below the diagonal: at a false-positive rate of 0.1 it identifies "
        f"{pct_r(roc(vq, 'XGBoost')[10])} of the value-led weeks, against {pct_r(roc(vq, 'Logistic')[10])} for "
        f"logistic regression. Its ranking also holds within each period between tuning points, with ROC-AUCs of "
        f"{f2(w_xvq[0])}, {f2(w_xvq[1])} and {f2(w_xvq[2])}, whereas logistic regression ranks well only in the "
        f"first ({f2(w_lvq[0])}, against {f2(w_lvq[1])} and {f2(w_lvq[2])}). The GAM's curve lies below the "
        f"diagonal over about half of the range."])
    figure_text("prob", vq, [
        f"Figure {fig('prob', vq)} shows that, unlike in the momentum pairs, XGBoost's forecast was never constant. "
        f"With the shallowest trees in its grid at every tuning point (Table C.6), its probabilities ranged from "
        f"{f2(v[0].probability.min())} to {f2(v[0].probability.max())} in the first period and spread further after "
        f"the January 2024 retune, and it was confidently wrong only once. Its accuracy of {pct(xvq.acc)} is "
        f"{100 * margin_vq:.1f} points above the hindsight rule ({pct(hind[vq])}), whereas in the momentum pairs "
        f"XGBoost's accuracy was within {100 * margin_mom:.1f} points of that rule. Logistic regression is never "
        f"confidently wrong. The GAM was confidently wrong in {num(len(gv_conf))} weeks, which account for "
        f"{pct_r(gv_share)} of its log-loss; {num(len(may_aug))} of them fell between May and August 2023, when "
        f"value led but the GAM gave value leadership a probability below 0.1, under a strong smoothing penalty "
        f"(λ = 1000), as for momentum versus quality. For this pair, then, the tree ensemble ranks the weeks better "
        f"than the linear and additive models within every period, the clearest indication in the study that "
        f"nonlinearity adds value, whereas the GAM's smooth additive terms add none."])
    note("4.2.3", "Opening rewritten: the strongest evidence in the study, from XGBoost (highest ROC-AUC of any "
         "model and pair); logistic regression second; the GAM close to random. Deleted with the old results: "
         "logistic regression as the strongest model, XGBoost's lower accuracy and 'The evidence for predictable "
         "value versus quality … linear specification'.", rerun)
    note(f"4.2.3 / Figure {fig('roc', vq)}", "Paragraph restyled as body text and rewritten: XGBoost's curve is "
         "highest over most of the range and never below the diagonal; its ranking holds within every period, "
         "logistic regression's only in the first; the GAM's curve below the diagonal over half the range.", rerun)
    note(f"4.2.3 / Figure {fig('prob', vq)}", "Paragraph restyled as body text and rewritten: XGBoost's forecast "
         "never constant and its accuracy clearly above the hindsight rule; the GAM's confident errors, most of them "
         "in May–August 2023. The citation (Friedman, 2001; Chen, Pelger, & Zhu, 2019) belonged to the deleted "
         "sentence on XGBoost's shrinkage; Chen, Pelger and Zhu (2019) is a deep-learning paper in any case (XGBoost "
         "is Chen & Guestrin, 2016, tag Che16).", rerun)

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
    assert len(swapped) == 6, swapped
    note(f"4.2 / Figures {swapped[0]}–{swapped[-1]}", "Pictures replaced by the renders from the rerun outputs "
         "(captions highlighted; caption text unchanged).", rerun + " build_chapter4_corrected_figures.py.")
    note("4.2 / citations", "Every citation in the rewritten paragraphs is a Word citation field (Source Manager "
         "tags Hos13, Isr20, Pag241, Faw06, Zou05, Woo17, Hoe70), so the paragraphs can be copied into the thesis "
         "as they are and the reference list updates with them.", "Citations.")
    for old, new in heads.items():
        e.replace(head_els[old], old, new)
    note("4.1, 4.2, 4.2.1, 4.4 / headings", "Headings in sentence case, as in Chapters 1 and 3: 'Overview of the "
         "evaluation framework', 'Out-of-sample predictive performance', 'Momentum versus value', 'Feature "
         "importance'.", "Consistency.")

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

    rewrite(e, find_para(body, startswith="XGBoost provides the most balanced class treatment"),
            f"XGBoost provides the most balanced class treatment for momentum versus value and value versus quality, "
            f"and logistic regression for momentum versus quality (Table {t_conf}).")
    note("4.3", "Most balanced model per pair updated; the shares behind it deleted from the text (they are in "
         f"Table {t_conf}).", rerun)
    xr, lr_ = m.loc[(mq, "XGBoost")], m.loc[(vq, "Logistic")]
    rewrite(e, find_para(body, startswith="The extreme positive-class tendencies"),
            f"The two largest imbalances illustrate these inconsistencies. XGBoost predicted momentum "
            f"leadership against quality in {pct(xr.pred_pos)} of weeks, the result of its constant forecast until "
            f"January 2024 (Section 4.2.2), and so identified only {int(xr.TN)} of the {int(xr.TN + xr.FP)} "
            f"quality-led weeks. Logistic regression predicted value leadership against quality in only "
            f"{pct(lr_.pred_pos)} of weeks, against {pct(lr_.act_pos)} realised, so that its second-highest ROC-AUC "
            f"for that pair translates into an accuracy of only {pct(lr_.acc)}.")
    note("4.3", "Paragraph on positive-class tendencies rewritten: the two largest imbalances are now XGBoost for "
         "momentum vs quality (its constant forecast) and logistic regression for value vs quality (too few "
         "positive predictions), as examples of the inconsistencies between ROC-AUC and accuracy that the author's "
         "preceding sentence announces.", rerun)
    lq = [m.loc[(p, "Logistic"), "ll"] for p in (mq, vq)]
    rewrite(e, find_para(body, startswith="The probability results introduce"),
            f"The probability results show a clearer pattern. Only XGBoost has a log-loss below the uninformative "
            f"0.693 (Section 3.9), and only for momentum versus quality and value versus quality. Logistic regression "
            f"is within 0.05 of that value for those two pairs ({f3(min(lq))} and {f3(max(lq))}), where it selected "
            f"a moderate penalty, but reaches {f3(lmv.ll)} for momentum versus value, where it selected the weakest; "
            f"the GAM's log-loss lies between {f3(min(lls['GAM']))} and {f3(max(lls['GAM']))}. Probability quality "
            f"therefore depends more on the shrinkage each model selected than on its flexibility (Sections "
            f"4.2.1–4.2.3).")
    note("4.3", "Probability-quality paragraph rewritten: only XGBoost beats the uninformative 0.693, for two "
         "pairs; logistic regression's log-loss depends on the penalty it selected; the GAM's is the highest. "
         "Deleted: 'An uninformative forecast of 0.5 … approximately 0.693' (defined in Section 3.9) and 'better "
         "discrimination does not necessarily imply better probability quality', which no longer holds (XGBoost "
         "leads on both).", rerun)

    # ---------------------------------------------------------------- 4.4 feature importance
    rewrite(e, find_para(body, startswith="Global feature importance is measured"),
            f"Figures {shap_first}–{shap_first + 2} show the ten features with the largest global SHAP importance "
            f"for each model (Section 3.10).")
    note("4.4", "Opening reduced to a reference to Section 3.10, which explains how the figures are read (the ten "
         "leading features, ranks rather than magnitudes, colours by the feature categories of Table 1) and the "
         "limits of the interpretation. Deleted: the definition of global importance, the caveats (Kumar et al., "
         "2020; Molnar, 2022), the comparison and colour conventions, 'This distinction is important because all "
         "final ROC-AUC values remain below 0.60' (no longer true) and the reference to 'Table 1 of Section 3.6'.",
         "Author: say it once; Section 3.10 explains the results section.")
    compose(e, find_para(body, startswith="Logistic regression is dominated by currency-volatility"), [
        "A relative-volatility measure ranks first in all three models: that of momentum and value for XGBoost, of "
        "value and quality for logistic regression and of momentum and quality for the GAM. Logistic regression "
        "and the GAM then share three South African macroeconomic variables in their top six, medium-term USDZAR "
        "forward pricing, USDZAR entropy and rand PPP valuation, whereas XGBoost draws on the momentum–quality "
        "correlation, South African real-yield-curve curvature, the value–quality volatility ratio, global bond "
        "volatility and the commodity index component. The prominence of relative volatility is consistent with "
        "evidence that momentum's risk varies predictably over time and that its crashes follow volatile markets ",
        (["Bar15", "Dan16"], "(Barroso & Santa-Clara, 2015; Daniel & Moskowitz, 2016)"), "."])
    note("4.4.1", "Paragraph rewritten from the new rankings (Figure " + str(shap_first) + "). The citations Ehsani "
         "& Linnainmaa (2022) and Gupta & Kelly (2019) on factor momentum are dropped here, because short-horizon "
         "relative returns no longer rank among the leading features for this pair (factor momentum is now cited in "
         "Section 4.4.3). Barroso & Santa-Clara (2015) and Daniel & Moskowitz (2016) kept, as citation fields.",
         rerun)
    compose(e, find_para(body, startswith="Momentum-versus-quality predictions use"), [
        "Logistic regression and XGBoost are led by the same two features, the change in the TOP40 price-to-EBITDA "
        "ratio and South African real-yield-curve curvature, and the commodity index component is in the top four "
        "of both. That a valuation measure leads both is in line with the debate on valuation-based factor timing ",
        (["Arn16", "Asn17"], "(Arnott, Beck, Kalesnik, & West, 2016; Asness, Chandra, Ilmanen, & Israel, 2017)"),
        ". The GAM is led instead by factor-specific variables, the value–quality correlation and three "
        "relative-volatility measures, with the EBITDA measure seventh. XGBoost's SHAP values for this pair are "
        "small, the ten largest summing to less than a sixth of those for the other two pairs, consistent with the "
        "near-constant forecasts of its initially selected setting (Section 4.2.2)."])
    note("4.4.2", "Paragraph rewritten from the new rankings (Figure " + str(shap_first + 1) + "). The EBITDA measure "
         "was described as a 'local-versus-global' differential; it is the first difference of the TOP40 "
         "price-to-EBITDA ratio (Appendix A). Arnott et al. (2016) and Asness et al. (2017) kept, as citation "
         "fields.", rerun + " Appendix A.")
    compose(e, find_para(body, startswith="The standardised value-versus-quality relative-performance measure"), [
        "Recent relative performance matters most for this pair. The three-month relative return of value and "
        "quality ranks first for XGBoost and second for logistic regression, and that of momentum and quality is in "
        "the top five of both, consistent with factor momentum ",
        (["Gup19", "Arn23"], "(Gupta & Kelly, 2019; Arnott, Kalesnik, & Linnainmaa, 2023)"),
        ". The value–quality volatility ratio is in the top four of all three models. Beyond these, logistic "
        "regression and the GAM share USDZAR entropy and global property volatility in their top three, whereas "
        "XGBoost draws on gold's rolling volatility and US and South African yield-curve curvature."])
    note("4.4.3", "Paragraph rewritten from the new rankings (Figure " + str(shap_first + 2) + "): the standardised "
         "relative-performance measure no longer leads. Deleted: 'Logistic regression is the best-performing model "
         "for this pair, but its AUC of 0.585 remains modest. The feature rankings should therefore be interpreted "
         "as a description of the fitted decision function rather than evidence of stable economic causality.' (no "
         "longer true; the caveat is in Section 3.10).", rerun)

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
    e.replace(p, text(p)[len("The final results provide modest and factor-pair-dependent evidence of predictability. "):],
              f"XGBoost is the strongest model for every pair and on every measure, and its mean ROC-AUC across the "
              f"three pairs ({f3(means.auc['XGBoost'])}) exceeds those of logistic regression "
              f"({f3(means.auc['Logistic'])}) and the GAM ({f3(means.auc['GAM'])}). It is the only model whose "
              f"block-bootstrap 95% interval for the mean ROC-AUC excludes 0.5 ({f3(xb.ci_low)} to {f3(xb.ci_high)}; "
              f"Section 3.9). The evidence is strongest for value versus quality, the only pair for which XGBoost's ROC-AUC "
              f"interval excludes 0.5 ({f3(cvq.ci_low)} to {f3(cvq.ci_high)}) and where its ranking holds within "
              f"every period between tuning points, and weakest for momentum versus quality, where its forecast was in "
              f"effect constant for two of the three years.")
    note("4.5", "Model ranking rewritten: XGBoost leads every pair on every measure; mean ROC-AUC of each model; "
         "XGBoost's bootstrap interval is the only one to exclude 0.5, and per pair only for value vs quality "
         "(results/model_outputs/bootstrap/auc_ci.csv); where the evidence is strongest and weakest. "
         "'several models produce strongly asymmetric classifications at the 0.5 threshold' deleted (Section 4.3).",
         rerun + " results/model_outputs/bootstrap/auc_mean_across_pairs.csv.")
    p = find_para(body, startswith="The findings therefore qualify")
    head = ("The findings therefore qualify the proposition that increasing model flexibility necessarily improves "
            "out-of-sample prediction (Gu et al., 2020; Kelly et al., 2024). ")
    tail = "The evidence supports limited predictive structure rather than a strongly separable factor-timing signal."
    assert text(p).startswith(head) and text(p).rstrip().endswith(tail), text(p)
    compose(e, p, [
        head + f"XGBoost's advantage in mean ROC-AUC over the GAM is distinguishable from zero ({f3(xg.estimate)}, "
        f"95% interval {f3(xg.ci_low)} to {f3(xg.ci_high)}), but its advantage over logistic regression is not "
        f"({f3(xl.estimate)}, {f3(xl.ci_low)} to {f3(xl.ci_high)}), so on 156 weekly observations the value of "
        f"nonlinearity over a regularised linear model is not statistically established. The two flexible models "
        f"also differ in more than nonlinearity: XGBoost can represent interactions between predictors, which the "
        f"additive GAM cannot ".replace("-0.", "−0."),
        (["Has86", "JHF01"], "(Hastie & Tibshirani, 1986; Friedman, 2001)"),
        ", and its tuning can shrink each tree's contribution, whereas the GAM's penalty leaves the linear part of "
        "each term unpenalised ", woo,
        ". The results cannot separate these two differences, but they show that smooth nonlinearity alone adds "
        "nothing. The clearest regularity across the three models concerns regularisation: the lightly regularised "
        "specifications, logistic regression for momentum versus value and the GAM, produced overconfident "
        "probabilities and the poorest log-loss, and the validation data gave little basis for choosing between "
        "penalties. " + tail])
    note("4.5", "Flexibility conclusion rewritten: bootstrap intervals for the differences in mean ROC-AUC (method in "
         "Section 3.9): XGBoost vs GAM distinguishable from zero, XGBoost vs logistic regression not; the two "
         "flexible models differ in interactions and in regularisation, which the results cannot separate; smooth "
         "nonlinearity alone adds nothing; the regularisation finding. The author's first and last sentences kept (the "
         "whole paragraph is highlighted because it now holds citation fields: Has86, JHF01, Woo17). "
         "'Even the best AUC is below 0.60' deleted (no longer true).", rerun + " results/model_outputs/bootstrap/"
         "auc_mean_across_pairs.csv.")
    p = find_para(body, startswith="The global SHAP analysis indicates")
    e.replace(p, "sentiment. These patterns differ",
              "sentiment. Only South African real-yield-curve curvature is among the ten leading features of almost "
              "every model and pair (eight of the nine); otherwise, the patterns differ")
    note("4.5", "SHAP summary: South African real-yield-curve curvature added as the one feature common to almost "
         f"all models and pairs (Figures {shap_range}).", rerun)

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
