"""Build the abstract for the thesis.

The thesis has an Abstract page with a heading and no text. The abstract summarises the thesis as it now stands in
three paragraphs:
  - the problem, the aim and the design (Chapter 1 opening, Sections 3.1-3.9);
  - the findings (Chapter 4 and the summary of key findings in Chapter 5), opening with the main finding;
  - the implications and the scope (Chapter 5).
It has no citations, as is usual for an abstract, and quotes no sample counts that Chapter 3 has yet to update.
Every number is read from results/model_outputs or from the thesis's own Chapter 3, and the script stops if a
sentence's premise does not hold.

Starts from the current thesis (a snapshot, since Word keeps the original open), keeps only the Abstract page and
fills it, highlighted, with an amendment log at the end.

Usage: python build_abstract.py [out_dir] [thesis_snapshot.docx]
"""
import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn

from build_chapter2_amendments import new_paragraph
from build_chapter4_amendments import find_para, heading, metrics, text
from build_minimal_chapter3 import Editor, accept_revisions, add_log, strip_comments

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "legacy" / "results"
OUTPUTS = RESULTS / "model_outputs"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "legacy" / "27717801_ResearchAssignment_vf - EG Mar 2026.docx"
OUTPUT = OUT_DIR / "Abstract_Highlighted.docx"

MV, MQ, VQ = "Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"
LN2 = 0.6931

ABSTRACT = [
    "Momentum, value and quality earn long-term premia, but their relative performance varies over time. This study "
    "tests whether machine learning can predict which of these factors leads in the South African equity market, and "
    "whether more flexible models predict better. Regularised logistic regression, a generalised additive model "
    "(GAM) and XGBoost are trained on trend-scanned labels for three factor pairs, tuned by combinatorial purged "
    "cross-validation and evaluated on 156 weekly walk-forward predictions from 2022 to 2025.",

    "Relative leadership is only modestly predictable. XGBoost performs best on every measure for every pair, with "
    "areas under the ROC curve of 0.589 to 0.664. The GAM, however, ranks below logistic regression on this measure "
    "for two pairs, so the gain is associated with shallow boosted trees, which capture interactions, rather than "
    "with smooth nonlinear effects. Value versus quality is the most predictable pair, whereas XGBoost's accuracy "
    "for momentum versus quality comes mainly from predicting momentum in the great majority of weeks. "
    "Relative-volatility measures "
    "rank among the top ten features of every model and pair.",

    "Factor timing models should therefore be assessed pair by pair, against a regularised linear benchmark and on "
    "more than accuracy alone. Transaction costs and portfolio implementation are left for future research.",
]


def check(m, body):
    """The premises of the abstract, from the model outputs (Tables 4 and 6, Table C.6, Figures 15-17) and from the
    design that the thesis's Chapter 3 states."""
    v = lambda pair, model, col: m.loc[(pair, model), col]
    # paragraph 1: the evaluation as Chapter 3 states it
    chapter3 = " ".join(text(p) for p in body.iter(qn("w:p")))
    assert "156 weekly forecasts of variable-horizon trend labels from 31 January 2022 to 20 January 2025" in chapter3
    dates = set()
    for model in ("logistic", "gam", "xgboost"):
        for pair in (MV, MQ, VQ):
            d = pd.read_csv(OUTPUTS / "probs" / f"{model}_{pair.lower()}_probs.csv")
            assert len(d) == 156
            dates.add((d["Date"].iloc[0], d["Date"].iloc[-1]))
    assert dates == {("2022-01-31", "2025-01-20")}, dates
    # paragraph 2: XGBoost leads every pair on every measure; its ROC-AUC range
    for pair in (MV, MQ, VQ):
        assert all(v(pair, "XGBoost", "acc") > v(pair, k, "acc") for k in ("Logistic", "GAM"))
        assert all(v(pair, "XGBoost", "auc") > v(pair, k, "auc") for k in ("Logistic", "GAM"))
        assert all(v(pair, "XGBoost", "ll") < v(pair, k, "ll") for k in ("Logistic", "GAM"))
    xgb = m.xs("XGBoost", level="model")
    assert (f"{xgb.auc.min():.3f}", f"{xgb.auc.max():.3f}") == ("0.589", "0.664")
    assert [p for p in (MV, MQ, VQ) if v(p, "GAM", "auc") < v(p, "Logistic", "auc")] == [MQ, VQ]
    depths = {s["max_depth"] for p in (MV, MQ, VQ)
              for s in json.loads((OUTPUTS / "artifacts" / f"xgboost_{p.lower()}_walkforward_params.json").read_text())}
    assert depths <= {2, 3, 4}, depths                                         # "shallow boosted trees"
    # value vs quality the most predictable pair: XGBoost's highest ROC-AUC and lowest log-loss (below ln 2, the
    # log-loss of a constant 0.5 forecast), with most of both classes recognised
    assert v(VQ, "XGBoost", "auc") == xgb.auc.max() and v(VQ, "XGBoost", "ll") == xgb.ll.min() < LN2
    assert v(VQ, "XGBoost", "TP") / (v(VQ, "XGBoost", "TP") + v(VQ, "XGBoost", "FN")) > 0.5
    assert v(VQ, "XGBoost", "TN") / (v(VQ, "XGBoost", "TN") + v(VQ, "XGBoost", "FP")) > 0.5
    # momentum vs quality: momentum predicted in the great majority of weeks (89%), and most quality-led weeks missed
    assert v(MQ, "XGBoost", "pred_pos") > 0.85
    assert v(MQ, "XGBoost", "TN") < v(MQ, "XGBoost", "FP")
    # a relative-volatility feature among the ten largest mean |SHAP| values of every model and pair
    for model in ("logistic", "gam", "xgboost"):
        for pair in (MV, MQ, VQ):
            s = pd.read_csv(OUTPUTS / "shap" / f"{model}_{pair.lower()}_shap.csv", index_col=0).iloc[:, 0]
            top = s.sort_values(ascending=False).index[:10]
            assert any(f.startswith(("relvol", "volratio")) for f in top), (model, pair)


def build():
    document = Document(SOURCE)
    body = document.element.body
    accept_revisions(body)
    strip_comments(body)
    check(metrics(), body)

    # keep only the Abstract page: its heading, then the new text (the empty paragraphs under it are dropped)
    template = find_para(body, startswith="By submitting this research assignment")   # a Normal body paragraph
    title = heading(body, "Abstract")
    end = heading(body, "Acknowledgements")
    children = list(body)
    assert all(not text(el).strip() for el in children[children.index(title) + 1:children.index(end)]), \
        "the Abstract page already has text"
    e = Editor(True)
    paragraphs = [new_paragraph(e, template, [p]) for p in ABSTRACT]
    for el in children:
        if el is not title and el.tag != qn("w:sectPr"):
            body.remove(el)
    for p in reversed(paragraphs):
        title.addnext(p)

    words = sum(len(p.split()) for p in ABSTRACT)
    log = [
        ("Abstract", f"New abstract ({words} words) in three paragraphs: the problem, aim and design; the findings; "
         "the implications and scope. The empty paragraphs under the heading are replaced.",
         "The thesis Abstract page was empty."),
        ("Paragraph 1", "Summarises the Chapter 1 opening and the design in Chapter 3: the three models "
         "(Section 3.7), trend-scanned labels (Section 3.5), CPCV and the walk-forward evaluation (Section 3.8).",
         "The evaluation is given as 156 predictions from 2022 to 2025, so the abstract does not depend on the week "
         "counts in Section 3.4, which still read 944 weeks and 919 labelled weeks against 939 and 914 in the data."),
        ("Paragraph 2", "Opens with the main finding, then gives XGBoost's lead and ROC-AUC range (Table 4), the "
         "non-monotonic effect of flexibility (Table 4; depth two to four, Table C.6), the strongest and weakest "
         "pairs (Tables 4 and 6) and the SHAP result (Figures 15-17).",
         "Follows the order of the key findings in Chapter 5. The relative-volatility statement is checked on the "
         "SHAP outputs: such a feature is in the top ten of all nine reference fits."),
        ("Paragraph 3", "The implications (pair-by-pair assessment, the linear benchmark, evaluation beyond "
         "accuracy) and the main limitation (no transaction costs or portfolio implementation).",
         "Chapter 5, implications and limitations."),
        ("Not included", "Citations, percentages (the user's request; momentum is predicted in 89.1% of weeks, "
         "Table 6), the predictor set (Section 3.6), the metrics (Section 3.9), the accuracy range, "
         "the momentum versus value probabilities, the bootstrap comparison of models and the hyperparameter "
         "sensitivity analysis of Appendix C.5.", "Cut to about half of the first draft (379 words) at the user's request; abstracts "
         "usually carry no citations, and the bootstrap comparison is outside the study's scope."),
    ]
    add_log(document, log)
    document.save(OUTPUT)
    return log, words


if __name__ == "__main__":
    log, words = build()
    print(OUTPUT.name, "words:", words, "changes logged:", len(log))
