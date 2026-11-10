"""Appendix C (CPCV and walk-forward formulation): the content written after the thesis's 'Appendix C' heading.

Used by build_appendices.py, which assembles Appendices A-C in one document. Takes the CPCV splitter from
logistic_classifier_roc.ipynb, the tuning grids, walk-forward selections and metrics from results/model_outputs,
the GAM exclusions from the GAM run logs in results/model_outputs/logs (the rerun on the corrected data,
rerun_pipeline.py), and the fixed-setting runs of sensitivity_fixed_settings.py (Section C.5, Figure C.1).
Nothing is refitted: the only computed predictions are the no-feature baselines (the training sample's
positive share). Formatting follows Appendix B (build_appendix_b.py).
"""

import ast
import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from build_appendix_b import caption, figure, paragraph, subheading, table
from thesis_style import GRID, INK, INK_2, LINE_THIN, MODEL_COLOURS, RC, TEXT_WIDTH

RESULTS = Path(__file__).resolve().parent
MODEL_TRAINING = RESULTS.parent / "model_training"
OUTPUTS = RESULTS / "model_outputs"
SENS = OUTPUTS / "sensitivity"
FIGURE_C1 = SENS / "figure_c1_fixed_settings.png"

PAIRS = {"Momentum_vs_Value": "Momentum vs value", "Momentum_vs_Quality": "Momentum vs quality",
         "Value_vs_Quality": "Value vs quality"}
MODELS = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}
KEYS = {"Logistic": ["alpha", "lambda_reg"], "GAM": ["lam", "n_splines"],
        "XGBoost": ["learning_rate", "max_depth", "min_child_weight", "reg_lambda", "subsample"]}
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


def sensitivity(schedule):
    """Walk-forward results of every grid setting held fixed for all predictions (sensitivity_fixed_settings.py),
    with the selected results of Chapter 4. Where the selected setting never changed, the fixed run of that
    setting must reproduce the selected result."""
    metrics = pd.read_csv(OUTPUTS / "metrics_summary.csv").set_index(["model", "factor_pair"])
    runs, reproduced = {}, []
    for model, tag in MODELS.items():
        for pair in PAIRS:
            grid = pd.read_csv(SENS / f"{tag}_{pair.lower()}_fixed.csv")
            ok = grid[grid["failed"].isna()]
            sel = metrics.loc[(model, pair)]
            used = {tuple(float(s[k]) for k in KEYS[model]) for s in schedule[(model, pair)]}
            if len(used) == 1:
                (values,) = used
                row = ok[np.logical_and.reduce([np.isclose(ok[k].astype(float), v) for k, v in zip(KEYS[model], values)])]
                assert len(row) == 1 and all(np.isclose(row[m].iloc[0], sel[m]) for m in ("roc_auc", "accuracy",
                                                                                          "log_loss")), (model, pair)
                reproduced.append((model, pair))
            runs[(model, pair)] = dict(grid=grid, auc=ok["roc_auc"].values, ll=ok["log_loss"].values,
                                       sel_auc=sel["roc_auc"], sel_ll=sel["log_loss"])
    assert reproduced == [("Logistic", "Value_vs_Quality"), ("GAM", "Momentum_vs_Value")], reproduced
    return runs


def first_period(schedule, model, pair, n_test):
    """ROC-AUC in the first 52 predictions of the setting selected at each tuning point, after checking that the
    selected forecasts equal the fixed run of the setting in use in every period."""
    tag = MODELS[model]
    probs = pd.read_csv(OUTPUTS / "probs" / f"{tag}_{pair.lower()}_probs.csv")
    y, q = probs["actual"].values, probs["probability"].values
    grid = pd.read_csv(SENS / f"{tag}_{pair.lower()}_fixed.csv")
    fixed = pd.read_csv(SENS / f"{tag}_{pair.lower()}_fixed_probs.csv")
    bounds = [0, *RETUNES, n_test]
    aucs = []
    for k, s in enumerate(schedule[(model, pair)]):
        match = np.logical_and.reduce([np.isclose(grid[c].astype(float), float(s[c])) for c in KEYS[model]])
        f = fixed[str(grid[match]["setting"].iloc[0])].values
        assert np.allclose(f[bounds[k]:bounds[k + 1]], q[bounds[k]:bounds[k + 1]], atol=1e-6), (model, pair, k)
        aucs.append(roc_auc_score(y[:RETUNES[0]], f[:RETUNES[0]]))
    return aucs


def figure_c1(runs):
    """Strip plot: the walk-forward ROC-AUC of every fixed setting, by pair and model, and the selected result."""
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(TEXT_WIDTH, 3.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (pair, name) in zip(axes, PAIRS.items()):
        ax.grid(axis="y", color=GRID, linewidth=LINE_THIN)
        ax.set_axisbelow(True)
        ax.axhline(0.5, color=INK_2, linewidth=LINE_THIN, linestyle="--", zorder=1)
        for x, model in enumerate(MODELS):
            r = runs[(model, pair)]
            ax.scatter(x + rng.uniform(-0.28, 0.28, len(r["auc"])), r["auc"], s=10, color=MODEL_COLOURS[model],
                       edgecolors="white", linewidths=0.4, zorder=2)
            ax.scatter([x], [r["sel_auc"]], marker="D", s=34, color=INK, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_xticks(range(len(MODELS)), list(MODELS))
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_title(name)
    axes[0].set_ylabel("Walk-forward ROC-AUC")
    handles = [Line2D([], [], marker="o", linestyle="", color=INK_2, markeredgecolor="white", markersize=5,
                      label="Setting held fixed for all 156 predictions"),
               Line2D([], [], marker="D", linestyle="", color=INK, markeredgecolor="white", markersize=6,
                      label="Selected by CPCV, retuned annually (Chapter 4)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_C1, dpi=300)
    plt.close(fig)
    return FIGURE_C1


def content(doc):
    """Append Appendix C after its heading."""
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

    W = Inches
    paragraph(doc, "Review note: this appendix replaces the ‘work in progress’ placeholder; all of its content is "
                   "new.", size=10, highlight=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    paragraph(doc, "This appendix gives the details of the validation design summarised in Section 3.8: the splits "
                   "and combinations used by combinatorial purged cross-validation (CPCV) (C.1), the ROC-AUCs of its "
                   "back-test paths (C.2), the GAM settings excluded from the search (C.3), the walk-forward "
                   "schedule with the hyperparameters selected at each tuning point (C.4), and the sensitivity of "
                   "the walk-forward results to the hyperparameters (C.5). Splits and combinations "
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

    # C.5 -----------------------------------------------------------------------------------------
    runs = sensitivity(schedule)
    f2, pct = (lambda x: f"{x:.3f}"), (lambda x: f"{100 * x:.0f}%")
    R = lambda m, p: runs[(m, p)]
    mv, mq, vq = PAIRS
    n_grid = {m: len(R(m, mv)["grid"]) for m in MODELS}
    assert all(len(R(m, p)["grid"]) == n_grid[m] for m in MODELS for p in PAIRS)
    failures = [(m, p, r) for (m, p), v in runs.items() for r in v["grid"][v["grid"]["failed"].notna()].itertuples()]
    assert len(failures) == 1 and failures[0][:2] == ("GAM", mq)
    fail = failures[0][2]
    below = lambda m, p: np.mean(R(m, p)["auc"] < R(m, p)["sel_auc"])     # share of fixed settings below selected
    under_ln2 = lambda m, p: np.mean(R(m, p)["ll"] < np.log(2))

    subheading(doc, "C.5: Sensitivity to the hyperparameters")
    paragraph(doc, "Chapter 4 reports the walk-forward results of the settings selected by CPCV at each tuning point. "
                   "To show how far these results depend on that selection, every setting in each model's search grid "
                   f"(Section 3.8.1) is also held fixed for all {n_test} predictions, without retuning, and evaluated by "
                   f"the same walk-forward procedure: {n_grid['Logistic']} settings of logistic regression, "
                   f"{n_grid['GAM']} of the GAM and {n_grid['XGBoost']} of XGBoost for each pair, "
                   f"{3 * sum(n_grid.values())} runs in all. One GAM setting "
                   f"({describe('GAM', {'lam': fail.lam, 'n_splines': fail.n_splines})}) could not be fitted in every "
                   "week for momentum versus quality and is excluded, as in the CPCV search. Where the selected "
                   "setting never changed (logistic regression for value versus quality and the GAM for momentum "
                   "versus value), holding it fixed reproduces the result in Chapter 4 exactly, so the fixed runs are "
                   "comparable with the main results. Table C.7 compares each selected result with the fixed settings, "
                   "and Figure C.1 shows the ROC-AUC of every fixed setting.", highlight=True)

    best_other = {p: max(R("Logistic", p)["auc"].max(), R("GAM", p)["auc"].max()) for p in PAIRS}
    x_vq, x_mv, x_mq = R("XGBoost", vq)["auc"], R("XGBoost", mv)["auc"], R("XGBoost", mq)["auc"]
    share_vq = np.mean(x_vq > best_other[vq])
    assert x_vq.min() > 0.5 and np.quantile(x_vq, 0.25) > best_other[vq]
    assert np.median(x_mv) > R("GAM", mv)["auc"].max() and np.median(x_mv) > np.median(R("Logistic", mv)["auc"])
    lmax_mv, lmax_mq, gmax_mq = R("Logistic", mv)["auc"].max(), R("Logistic", mq)["auc"].max(), R("GAM", mq)["auc"].max()
    assert lmax_mv < np.median(x_mv) and np.median(x_mq) > lmax_mq and gmax_mq > np.quantile(x_mq, 0.75)
    ln2 = {(m, p): under_ln2(m, p) for m in MODELS for p in PAIRS}
    assert [k for k, v in ln2.items() if v > 0.5] == [("XGBoost", mq), ("XGBoost", vq)]
    assert all(ln2[("GAM", p)] == 0 for p in PAIRS)
    paragraph(doc, "XGBoost's lead in ranking does not depend on its setting for value versus quality and momentum "
                   "versus value. For value versus quality every fixed XGBoost setting has a ROC-AUC above 0.5 "
                   f"({f2(x_vq.min())} to {f2(x_vq.max())}), and {pct(share_vq)} of them exceed the best fixed setting "
                   f"of both logistic regression ({f2(R('Logistic', vq)['auc'].max())}) and the GAM "
                   f"({f2(R('GAM', vq)['auc'].max())}). For momentum versus value {(x_mv > 0.5).sum()} of the "
                   f"{len(x_mv)} settings exceed 0.5, and their median ({f2(np.median(x_mv))}) exceeds every GAM "
                   f"setting and the median logistic setting ({f2(np.median(R('Logistic', mv)['auc']))}), although "
                   f"the best logistic setting ({f2(lmax_mv)}) is above {pct(np.mean(x_mv < lmax_mv))} of the XGBoost "
                   "settings. For momentum versus quality the ranking depends on the settings: the XGBoost median "
                   f"({f2(np.median(x_mq))}) only just exceeds the best logistic setting ({f2(lmax_mq)}), and the best "
                   f"GAM setting ({f2(gmax_mq)}) is above {pct(np.mean(x_mq < gmax_mq))} of the XGBoost settings. "
                   "XGBoost is also the only model whose fixed settings mostly have a log-loss below 0.693, for "
                   f"momentum versus quality ({pct(ln2[('XGBoost', mq)])}) and value versus quality "
                   f"({pct(ln2[('XGBoost', vq)])}); no GAM setting does so for any pair.", highlight=True)

    sel_x = {p: below("XGBoost", p) for p in PAIRS}
    assert sel_x[mv] > 0.9 and sel_x[vq] > 0.75 and 0.4 < sel_x[mq] < 0.6
    paragraph(doc, f"CPCV selection placed XGBoost above {pct(sel_x[mv])} and {pct(sel_x[vq])} of its fixed settings "
                   "for momentum versus value and value versus quality, and near the median for momentum versus "
                   f"quality ({pct(sel_x[mq])}), so the selected results do not overstate what its grid allows. The "
                   f"best fixed settings reach {f2(x_mv.max())}, {f2(x_mq.max())} and {f2(x_vq.max())}, but they could "
                   "be identified only with hindsight: the range across settings measures the dependence on the "
                   "hyperparameters, not an attainable improvement.", highlight=True)

    g_vq, g_mq, g_mv = R("GAM", vq), R("GAM", mq), R("GAM", mv)
    first = first_period(schedule, "GAM", vq, n_test)
    initial = schedule[("GAM", vq)][0]
    assert g_vq["sel_auc"] < g_vq["auc"].min() and (initial["lam"], initial["n_splines"]) == (1, 10)
    assert len({f2(a) for a in first[1:]}) == 1 and first[0] < 0.55 < first[1]
    q_gam = np.mean(g_vq["auc"] >= 0.58)
    assert q_gam >= 0.75 and below("GAM", mq) < 0.2 and g_mv["auc"].min() > 0.45 and g_mv["auc"].max() < 0.55
    paragraph(doc, "The GAM's results depend more on the selection. For value versus quality its selected ROC-AUC "
                   f"({f2(g_vq['sel_auc'])}) is below that of every fixed setting ({f2(g_vq['auc'].min())} to "
                   f"{f2(g_vq['auc'].max())}). In each period the selected forecasts coincide with the fixed run of "
                   "the setting then in use, so the difference arises in the first period: the weakest penalty "
                   f"({describe('GAM', initial)}), selected at the initial tuning point, ranks that period's weeks "
                   f"at {f2(first[0])}, whereas the strongly penalised settings selected at the later tuning points "
                   f"rank them at {f2(first[1])}. With {pct(q_gam)} of the fixed settings the GAM would have reached "
                   f"at least 0.580 for this pair, close to logistic regression ({f2(R('Logistic', vq)['sel_auc'])}). "
                   f"For momentum versus quality its selected result ({f2(g_mq['sel_auc'])}) is near the bottom of "
                   f"its range ({f2(g_mq['auc'].min())} to {f2(g_mq['auc'].max())}), and for momentum versus value "
                   f"every setting is close to 0.5 ({f2(g_mv['auc'].min())} to {f2(g_mv['auc'].max())}).",
              highlight=True)

    l_mv = R("Logistic", mv)
    lam_mv = {s["lambda_reg"] for s in schedule[("Logistic", mv)]}
    assert l_mv["auc"].max() < 0.56 and lam_mv == {0.01} and l_mv["sel_ll"] >= l_mv["ll"].max() - 1e-3
    paragraph(doc, "For logistic regression the selection matters mainly for its probabilities. For momentum versus "
                   "value no setting ranks the weeks reliably better than chance: the ROC-AUCs range from "
                   f"{f2(l_mv['auc'].min())} to {f2(l_mv['auc'].max())}, and {(l_mv['auc'] < 0.5).sum()} of the "
                   f"{len(l_mv['auc'])} settings fall below 0.5, so its weak ranking for this pair does not come from "
                   "the selection. The weakest penalty (λ = 0.01), selected at every tuning point for this pair, "
                   f"gives a log-loss of {f2(l_mv['sel_ll'])}, as high as the worst fixed setting, against a median "
                   f"of {f2(np.median(l_mv['ll']))}. For the other two pairs the selected results exceed "
                   f"{pct(below('Logistic', mq))} and {pct(below('Logistic', vq))} of the fixed settings.",
              highlight=True)
    assert g_vq["auc"].max() < np.quantile(x_vq, 0.25)
    paragraph(doc, "The sensitivity analysis therefore supports the finding that XGBoost ranks the weeks better than "
                   "the other models for value versus quality and momentum versus value whatever its setting, but not "
                   "for momentum versus quality. The GAM's weak results partly reflect the setting selected at the "
                   "initial tuning point rather than the model class: with a fixed, strongly penalised setting it is "
                   "comparable with logistic regression for value versus quality, but it does not approach XGBoost.",
              highlight=True)

    caption(doc, "Table C.7: Walk-forward ROC-AUC and log-loss of the selected settings and of every setting held "
                 "fixed")
    rows = []
    for pair, name in PAIRS.items():
        for model in MODELS:
            r = R(model, pair)
            n_ok, n_all = len(r["auc"]), len(r["grid"])
            rows.append([name if model == "Logistic" else "", model,
                         n_ok if n_ok == n_all else f"{n_ok} of {n_all}", f2(r["sel_auc"]),
                         f"{f2(np.median(r['auc']))} ({f2(r['auc'].min())}–{f2(r['auc'].max())})",
                         pct(below(model, pair)), f2(r["sel_ll"]),
                         f"{f2(np.median(r['ll']))} ({f2(r['ll'].min())}–{f2(r['ll'].max())})"])
    table(doc, ["Factor pair", "Model", "Fixed settings", "Selected ROC-AUC", "Fixed ROC-AUC: median (range)",
                "Fixed settings below selected", "Selected log-loss", "Fixed log-loss: median (range)"], rows,
          [W(1.0), W(0.65), W(0.55), W(0.6), W(1.05), W(0.7), W(0.6), W(1.05)], font=7.5,
          align_right=(2, 3, 4, 5, 6, 7))
    figure(doc, figure_c1(runs))
    caption(doc, "Figure C.1: Walk-forward ROC-AUC of every hyperparameter setting held fixed, by factor pair and "
                 "model, and of the settings selected by CPCV")
    return runs


if __name__ == "__main__":
    sys.exit("Appendix C is built with Appendices A and B by build_appendices.py.")
