"""Build Appendix C (CPCV and walk-forward formulation) as a standalone Word document.

Usage: python build_appendix_c.py [out_dir] [thesis_snapshot.docx]

Takes the CPCV splitter from logistic_classifier_roc.ipynb, the tuning grids, walk-forward selections and
metrics from results/model_outputs, and the GAM exclusions from the GAM run logs in results/model_outputs/logs
(the rerun on the corrected data, rerun_pipeline.py), and writes
Appendix_C_CPCV_Walk_Forward.docx. Nothing is refitted: the only computed predictions are the no-feature
baselines (the training sample's positive share). Formatting follows Appendix B (build_appendix_b.py).
"""

import ast
import copy
import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches
from sklearn.metrics import roc_auc_score

from build_appendix_b import caption, paragraph, subheading, table
from build_minimal_chapter3 import accept_revisions, strip_comments

RESULTS = Path(__file__).resolve().parent
MODEL_TRAINING = RESULTS.parent / "model_training"
OUTPUTS = RESULTS / "model_outputs"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
# A copy of the thesis (Word keeps the original locked while it is open; snapshot it with cp first).
SOURCE = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS.parents[1] / ".tmp_chapter3_minimal" / "thesis_current.docx"
OUTPUT = OUT_DIR / "Appendix_C_CPCV_Walk_Forward.docx"
HEADING = "Appendix C: CPCV and Walk-forward Formulation"

PAIRS = {"Momentum_vs_Value": "Momentum vs value", "Momentum_vs_Quality": "Momentum vs quality",
         "Value_vs_Quality": "Value vs quality"}
MODELS = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}
LETTERS = "ABCDE"
RETUNES = (52, 104)


def text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def words(n):
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n]


def day(ts):
    return f"{ts.day} {ts:%B %Y}"


def f3(x):
    return f"{x:.3f}"


# ------------------------------------------------------------------ inputs --

def notebook_namespace():
    """Definitions from the logistic notebook's set-up cells (data loading and the CPCV splitter)."""
    ns, cwd = {}, os.getcwd()
    os.chdir(MODEL_TRAINING)
    try:
        cells = json.load(open("logistic_classifier_roc.ipynb", encoding="utf-8"))["cells"]
        for cell in cells[:6]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), ns)
    finally:
        os.chdir(cwd)
    return ns


def selected_paths(tag, pair):
    """Split and path ROC-AUCs of the setting selected at the initial tuning point."""
    grid = pd.read_csv(OUTPUTS / "artifacts" / f"{tag}_{pair.lower()}_tuning_grid.pkl")   # saved as CSV
    if "failed" in grid:
        grid = grid[grid["failed"].isna()]
    best = json.loads((OUTPUTS / "artifacts" / f"{tag}_{pair.lower()}_best_params.json").read_text())
    match = np.logical_and.reduce([np.isclose(grid[k].astype(float), float(v))
                                   for k, v in best.items() if not isinstance(v, bool)])
    top = grid[match].iloc[0]          # settings can tie (e.g. XGBoost depths 2-4); the notebook keeps the first
    assert np.isclose(top["mean_auc"], grid["mean_auc"].max()), (tag, pair)
    paths = [float(v) for v in re.findall(r"\d\.\d+", top["path_aucs"])]
    assert len(paths) == 4 and np.isclose(np.mean(paths), top["mean_path_auc"]), (tag, pair)
    return top["mean_auc"], paths


def gam_exclusions():
    """(pair, tuning step, setting, first failed combination) from the GAM run logs. Each search prints
    'Testing N parameter combinations' once, so the k-th such line opens tuning point k (0, then RETUNES)."""
    rows = []
    points = (0, *RETUNES)
    for pair in PAIRS:
        log = (OUTPUTS / "logs" / f"gam_{pair.lower()}.log").read_text(encoding="utf-8")
        starts = [m.start() for m in re.finditer(r"Testing \d+ parameter combinations", log)]
        assert len(starts) == len(points), (pair, len(starts))
        for m in re.finditer(r"Disqualified (\{.*?\}): split (\d+)", log):
            k = sum(st < m.start() for st in starts) - 1
            rows.append((pair, points[k], ast.literal_eval(m.group(1)), int(m.group(2))))
    return rows


def describe(model, params):
    if model == "Logistic":
        return f"α = {params['alpha']:g}, λ = {params['lambda_reg']:g}"
    if model == "GAM":
        return f"λ = {params['lam']:g}, {params['n_splines']} splines"
    return (f"η = {params['learning_rate']:g}, depth {params['max_depth']}, min. child weight "
            f"{params['min_child_weight']}, λ = {params['reg_lambda']:g}, subsample {params['subsample']:g}")


# ---------------------------------------------------------------- document --

def prepare():
    document = Document(SOURCE)
    body = document.element.body
    children = list(body)
    heading = next(el for el in children if el.tag == qn("w:p") and text(el).strip() == HEADING)
    start = children.index(heading)
    section = next(el for el in children[start:] if el.tag == qn("w:p") and el.find(qn("w:pPr")) is not None
                   and el.find(qn("w:pPr")).find(qn("w:sectPr")) is not None)
    accept_revisions(body)
    strip_comments(body)
    appendix_sect = copy.deepcopy(section.find(qn("w:pPr")).find(qn("w:sectPr")))
    for child in list(body):
        if child is not heading and child.tag != qn("w:sectPr"):
            body.remove(child)
    final = body.find(qn("w:sectPr"))
    final.addprevious(appendix_sect)
    body.remove(final)
    return document


def build():
    ns = notebook_namespace()
    cv, n_test, gap = ns["CombinatorialPurgedCV"](), ns["FINAL_TEST_POINTS"], ns["PURGE_GAP"]
    data = {pair: ns["load_data"](pair) for pair in PAIRS}
    dates = data["Momentum_vs_Value"].index
    assert all(d.index.equals(dates) for d in data.values())
    n_all = len(dates)
    n_tune = n_all - n_test
    labels = {pair: d.iloc[:, -1].values for pair, d in data.items()}

    # C.1: splits and combinations of the initial search
    bounds = cv.group_bounds(n_tune)
    combos = list(combinations(range(cv.n_splits), cv.n_test_splits))
    splits = list(cv.split(np.zeros((n_tune, 1))))
    assert len(splits) == len(combos)
    path_of = {(combo, group): j + 1 for j, path in enumerate(cv.backtest_paths()) for group, combo in enumerate(path)}
    share = {pair: [labels[pair][s:e].mean() for s, e in bounds] for pair in PAIRS}
    train_sizes = [len(train) for train, _ in splits]

    # C.2: path ROC-AUCs of the no-feature baseline and of the selected settings
    baseline, walk_forward = {}, {}
    for pair in PAIRS:
        y = labels[pair][:n_tune]
        probs = [ns["held_out_vector"](n_tune, test, np.full(len(test), y[train].mean())) for train, test in splits]
        baseline[pair] = ns["path_aucs"](cv, y, probs)
        y_all = labels[pair]
        walk_forward[pair] = roc_auc_score(y_all[-n_test:], [y_all[:i - gap].mean()
                                                             for i in range(n_all - n_test, n_all)])
    selected = {(model, pair): selected_paths(tag, pair) for model, tag in MODELS.items() for pair in PAIRS}
    margins = [np.mean(selected[(m, p)][1]) - np.mean(baseline[p]) for m in MODELS for p in PAIRS]
    above = sum(a > b for m in MODELS for p in PAIRS for a, b in zip(selected[(m, p)][1], baseline[p]))
    assert min(margins) > 0

    # C.3 and C.4: exclusions and the walk-forward schedule
    excluded = gam_exclusions()
    schedule = {(m, p): json.loads((OUTPUTS / "artifacts" / f"{t}_{p.lower()}_walkforward_params.json").read_text())
                for m, t in MODELS.items() for p in PAIRS}
    steps = [s["from_step"] for s in schedule[("Logistic", "Momentum_vs_Value")]]
    n_train = [s["n_train"] for s in schedule[("Logistic", "Momentum_vs_Value")]]
    assert steps == [0, *RETUNES] and all([s["from_step"] for s in v] == steps and [s["n_train"] for s in v] == n_train
                                          for v in schedule.values())
    assert n_train[0] == n_tune and n_train[1:] == [n_tune + s - gap for s in RETUNES]
    strip = lambda s: {k: v for k, v in s.items() if k not in ("from_step", "n_train", "fit_intercept", "max_iter")}
    changes = sum(strip(v[i]) != strip(v[i - 1]) for v in schedule.values() for i in (1, 2))

    doc = prepare()
    W = Inches
    paragraph(doc, "Review note: this appendix replaces the ‘work in progress’ placeholder; all of its content is "
                   "new.", size=10, highlight=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    paragraph(doc, "This appendix gives the details of the validation design summarised in Section 3.8: the splits "
                   "and combinations used by combinatorial purged cross-validation (CPCV) (C.1), the ROC-AUCs of its "
                   "back-test paths (C.2), the GAM settings excluded from the search (C.3), and the walk-forward "
                   "schedule with the hyperparameters selected at each tuning point (C.4). Splits and combinations "
                   f"are shown for the initial search, on the {n_tune} weekly observations from {day(dates[0])} to "
                   f"{day(dates[n_tune - 1])}; each retuning search divides its own, longer training window in the "
                   "same way.")

    # C.1 -----------------------------------------------------------------------------------------
    size = bounds[0][1] - bounds[0][0]
    shares = [v for p in PAIRS for v in share[p]]
    subheading(doc, "C.1: CPCV splits and combinations")
    paragraph(doc, f"The tuning sample is divided into five chronological splits of {size} observations, except the "
                   f"last, which has {bounds[-1][1] - bounds[-1][0]}. Table C.1 gives their dates "
                   f"and the share of positive labels in each, which ranges from {min(shares):.0%} to "
                   f"{max(shares):.0%} across splits and pairs. Table C.2 lists the ten combinations. Each training "
                   "set is what remains after removing the two validation splits and the 12 weeks before and after "
                   f"each of them, between {min(train_sizes)} and {max(train_sizes)} observations; the last column "
                   "gives the back-test path to which each validation split's predictions are assigned.")
    caption(doc, "Table C.1: The five splits of the initial tuning sample and the positive share of each pair's "
                 "labels")
    table(doc, ["Split", "Weeks", "Dates", "Observations", *PAIRS.values()],
          [[LETTERS[g], f"{s + 1}–{e}", f"{dates[s]:%d %b %Y} – {dates[e - 1]:%d %b %Y}", e - s,
            *(f"{share[p][g]:.1%}" for p in PAIRS)] for g, (s, e) in enumerate(bounds)],
          [W(0.45), W(0.75), W(1.75), W(0.85), W(0.8), W(0.8), W(0.8)], font=8, align_right=(3, 4, 5, 6))
    caption(doc, "Table C.2: The ten CPCV combinations of the initial search")
    table(doc, ["Validation splits", "Training observations", "Validation observations", "Back-test paths"],
          [["".join(LETTERS[g] for g in combo), train_sizes[i], len(splits[i][1]),
            ", ".join(f"{LETTERS[g]}: P{path_of[(i, g)]}" for g in combo)] for i, combo in enumerate(combos)],
          [W(1.3), W(1.5), W(1.5), W(1.9)], font=8, align_right=(1, 2))

    # C.2 -----------------------------------------------------------------------------------------
    subheading(doc, "C.2: Back-test paths and their ROC-AUCs")
    pair_list = list(PAIRS)
    paragraph(doc, "Each back-test path covers the tuning sample once, taking the predictions for each split from one "
                   "of the combinations that use it for validation (Table C.2), so its ROC-AUC ranks together the predictions of four "
                   "separately fitted models. Holding two splits out moves the positive share of the training "
                   "sample away from that of the splits being predicted: when splits with a high positive share are "
                   "held out, the training sample contains fewer positives and the predicted probabilities for those "
                   "splits are lower overall (Table C.1). A path's ROC-AUC therefore penalises differences in the "
                   "level of the predictions between combinations, not only their ranking within each. A predictor "
                   "without features, which always predicts the positive share of its training sample, shows the "
                   "effect: within each combination its ROC-AUC is exactly 0.5, but its mean path ROC-AUC is "
                   f"{f3(np.mean(baseline[pair_list[0]]))} for momentum versus value, "
                   f"{f3(np.mean(baseline[pair_list[1]]))} for momentum versus quality and "
                   f"{f3(np.mean(baseline[pair_list[2]]))} for value versus quality (Table C.3).")
    paragraph(doc, "Table C.3 compares the path ROC-AUCs of the settings selected at the initial tuning point with "
                   "this baseline. Every selected setting exceeds the baseline's mean path ROC-AUC, by "
                   f"{min(margins):.3f} to {max(margins):.3f}, and does so on {above} of the 36 individual paths. "
                   "Because the baseline lies below 0.5, the path ROC-AUCs are reported as a check rather than used "
                   "for selection; the mean ROC-AUC of the ten combinations, each computed from a single fitted "
                   "model, is not affected by these level differences (Section 3.8.1).")
    assert all(v < 0.45 for v in walk_forward.values()), walk_forward
    paragraph(doc, "In the walk-forward evaluation the training window grows by one week at a time, so the positive "
                   "share of successive training samples changes slowly, but the effect does not vanish. A no-feature "
                   "forecast, the positive share of the expanding training window, has a walk-forward ROC-AUC of "
                   f"{f3(walk_forward[pair_list[0]])} for momentum versus value, {f3(walk_forward[pair_list[1]])} for "
                   f"momentum versus quality and {f3(walk_forward[pair_list[2]])} for value versus quality: in every "
                   "pair the positive share of the training window moved against the labels of the evaluation block. "
                   "The walk-forward ROC-AUCs in Chapter 4 can therefore be compared with these values as well as "
                   "with 0.5.")
    rows = []
    for pair, name in PAIRS.items():
        rows.append([name, "No-feature baseline", "0.500", *map(f3, baseline[pair]), f3(np.mean(baseline[pair]))])
        for model in MODELS:
            split_mean, paths = selected[(model, pair)]
            rows.append(["", model, f3(split_mean), *map(f3, paths), f3(np.mean(paths))])
    caption(doc, "Table C.3: ROC-AUCs of the settings selected at the initial tuning point and of a no-feature "
                 "baseline, by back-test path")
    table(doc, ["Factor pair", "Model", "Mean of ten combinations", "P1", "P2", "P3", "P4", "Mean of paths"], rows,
          [W(1.2), W(1.3), W(0.95), W(0.55), W(0.55), W(0.55), W(0.55), W(0.8)], font=8,
          align_right=(2, 3, 4, 5, 6, 7))

    # C.3 -----------------------------------------------------------------------------------------
    subheading(doc, "C.3: GAM settings excluded from the search")
    runs = {(p, step) for p, step, _, _ in excluded}
    assert excluded and all(s["n_splines"] == 10 for _, _, s, _ in excluded)
    paragraph(doc, "A candidate setting is excluded if it cannot be fitted on all ten combinations (Section 3.8.1). "
                   f"This affected only the GAM, in {words(len(runs))} of its nine tuning runs, and only settings with "
                   "the largest basis in its grid (10 splines), whose penalised iteratively reweighted least-squares "
                   "fit diverged (Table C.4). The search stops at a setting's first failed combination, so the table "
                   "gives the first failure only.")
    point = {0: "Initial (before prediction 1)", 52: "After prediction 52", 104: "After prediction 104"}
    caption(doc, "Table C.4: GAM settings excluded from the search")
    table(doc, ["Factor pair", "Tuning point", "Training observations", "Excluded setting",
                "First failed combination"],
          [[PAIRS[p], point[step], n_train[[0, *RETUNES].index(step)], describe("GAM", s),
            "".join(LETTERS[g] for g in combos[k])] for p, step, s, k in excluded],
          [W(1.3), W(1.6), W(1.0), W(1.4), W(0.95)], font=8, align_right=(2,))

    # C.4 -----------------------------------------------------------------------------------------
    subheading(doc, "C.4: Walk-forward schedule and selected hyperparameters")
    paragraph(doc, "Table C.5 gives the tuning schedule. The initial search uses the observations that precede the "
                   "evaluation block, without a boundary purge (Section 3.8.1); each retuning search uses the purged "
                   "training window available at that point, which is also the training window of the next "
                   "prediction. Each selected setting is used for the following 52 predictions. Table C.6 lists the "
                   f"selections; the selected setting changes at {changes} of the 18 retuning points.")
    starts = [n_tune + s for s in (0, *RETUNES)]
    ends = [n_tune + s - 1 for s in (*RETUNES, n_test)]
    caption(doc, "Table C.5: Walk-forward tuning schedule")
    table(doc, ["Tuning point", "Predictions", "Prediction dates", "Tuning sample", "Training observations per "
                "prediction"],
          [[point[s], f"{a - n_tune + 1}–{b - n_tune + 1}", f"{dates[a]:%d %b %Y} – {dates[b]:%d %b %Y}",
            f"{n} (to {dates[n - 1]:%d %b %Y})", f"{a - gap}–{b - gap}"]
           for s, n, a, b in zip((0, *RETUNES), n_train, starts, ends)],
          [W(1.45), W(0.8), W(1.75), W(1.25), W(1.0)], font=8, align_right=(1, 4))
    caption(doc, "Table C.6: Hyperparameters selected at each tuning point")
    table(doc, ["Factor pair", "Model", point[0], point[52], point[104]],
          [[PAIRS[p] if m == "Logistic" else "", m, *(describe(m, s) for s in schedule[(m, p)])]
           for p in PAIRS for m in MODELS],
          [W(1.05), W(0.7), W(1.5), W(1.5), W(1.5)], font=7)

    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print("Wrote", build())
