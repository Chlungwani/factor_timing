"""Appendix C tables and quoted numbers after the purged rerun, as JSON for build_supervisor_corrections.py.

Same computations as legacy/results/build_appendix_c.py (its helpers are imported), except that the initial
tuning sample is the 746 observations left after the 12-week purge before the evaluation block, while each
retuning window still ends 12 weeks before its next prediction. Also redraws Figure C.1 with the new selections.

Usage: python appendix_c_values.py <out.json> <figure_c1.png>
"""
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LEGACY = Path(__file__).resolve().parents[3] / "legacy"
sys.path.insert(0, str(LEGACY / "results"))
import build_appendix_c as bc  # noqa: E402

PAIRS, MODELS, LETTERS, RETUNES = bc.PAIRS, bc.MODELS, bc.LETTERS, bc.RETUNES
f3 = bc.f3


def values():
    ns = bc.notebook_namespace()
    cv, n_test, gap = ns["CombinatorialPurgedCV"](), ns["FINAL_TEST_POINTS"], ns["PURGE_GAP"]
    data = {pair: ns["load_data"](pair) for pair in PAIRS}
    dates = data["Momentum_vs_Value"].index
    n_all = len(dates)
    n_pre = n_all - n_test
    n_tune = n_pre - gap                                   # the purged initial tuning sample
    labels = {pair: d.iloc[:, -1].values for pair, d in data.items()}
    V = {"n_tune": n_tune, "tune_first": f"{dates[0]:%d %B %Y}", "tune_last": f"{dates[n_tune - 1]:%d %B %Y}"}

    bounds = cv.group_bounds(n_tune)
    combos = list(combinations(range(cv.n_splits), cv.n_test_splits))
    splits = list(cv.split(np.zeros((n_tune, 1))))
    path_of = {(combo, group): j + 1 for j, path in enumerate(cv.backtest_paths()) for group, combo in enumerate(path)}
    share = {pair: [labels[pair][s:e].mean() for s, e in bounds] for pair in PAIRS}
    train_sizes = [len(train) for train, _ in splits]
    V["split_sizes"] = [e - s for s, e in bounds]
    V["train_range"] = [min(train_sizes), max(train_sizes)]
    V["table_c1"] = [[LETTERS[g], f"{s + 1}–{e}", f"{dates[s]:%d %b %Y} – {dates[e - 1]:%d %b %Y}", str(e - s),
                      *(f"{share[p][g]:.1%}" for p in PAIRS)] for g, (s, e) in enumerate(bounds)]
    V["table_c2"] = [["".join(LETTERS[g] for g in combo), str(train_sizes[i]), str(len(splits[i][1])),
                      ", ".join(f"{LETTERS[g]}: P{path_of[(i, g)]}" for g in combo)] for i, combo in enumerate(combos)]

    baseline = {}
    for pair in PAIRS:
        y = labels[pair][:n_tune]
        probs = [ns["held_out_vector"](n_tune, test, np.full(len(test), y[train].mean())) for train, test in splits]
        baseline[pair] = ns["path_aucs"](cv, y, probs)
    selected = {(model, pair): bc.selected_paths(tag, pair) for model, tag in MODELS.items() for pair in PAIRS}
    margins = [np.mean(selected[(m, p)][1]) - np.mean(baseline[p]) for m in MODELS for p in PAIRS]
    V["baseline_mean_paths"] = {p: float(np.mean(baseline[p])) for p in PAIRS}
    V["margin_range"] = [float(min(margins)), float(max(margins))]
    V["paths_above_baseline"] = int(sum(a > b for m in MODELS for p in PAIRS
                                        for a, b in zip(selected[(m, p)][1], baseline[p])))
    V["cpcv_mean"] = {f"{p}|{m}": float(selected[(m, p)][0]) for m in MODELS for p in PAIRS}
    rows = []
    for pair, name in PAIRS.items():
        rows.append([name, "No-feature baseline", "0.500", *map(f3, baseline[pair]), f3(np.mean(baseline[pair]))])
        for model in MODELS:
            split_mean, paths = selected[(model, pair)]
            rows.append(["", model, f3(split_mean), *map(f3, paths), f3(np.mean(paths))])
    V["table_c3"] = rows

    excluded = bc.gam_exclusions()
    schedule = {(m, p): json.loads((bc.OUTPUTS / "artifacts" / f"{t}_{p.lower()}_walkforward_params.json").read_text())
                for m, t in MODELS.items() for p in PAIRS}
    n_train = [s["n_train"] for s in schedule[("Logistic", "Momentum_vs_Value")]]
    assert n_train[0] == n_tune and n_train[1:] == [n_pre + s - gap for s in RETUNES], n_train
    point = {0: "Initial (before prediction 1)", 52: "After prediction 52", 104: "After prediction 104"}
    V["table_c4"] = [[PAIRS[p], point[step], str(n_train[[0, *RETUNES].index(step)]), bc.describe("GAM", s),
                      "".join(LETTERS[g] for g in combos[k])] for p, step, s, k in excluded]
    V["c4_runs"] = len({(p, step) for p, step, _, _ in excluded})
    strip = lambda s: {k: v for k, v in s.items() if k not in ("from_step", "n_train", "fit_intercept", "max_iter")}
    V["changes"] = int(sum(strip(v[i]) != strip(v[i - 1]) for v in schedule.values() for i in (1, 2)))
    starts = [n_pre + s for s in (0, *RETUNES)]
    ends = [n_pre + s - 1 for s in (*RETUNES, n_test)]
    V["table_c5"] = [[point[s], f"{a - n_pre + 1}–{b - n_pre + 1}", f"{dates[a]:%d %b %Y} – {dates[b]:%d %b %Y}",
                      f"{n} (to {dates[n - 1]:%d %b %Y})", f"{a - gap}–{b - gap}"]
                     for s, n, a, b in zip((0, *RETUNES), n_train, starts, ends)]
    V["table_c6"] = [[PAIRS[p] if m == "Logistic" else "", m, *(bc.describe(m, s) for s in schedule[(m, p)])]
                     for p in PAIRS for m in MODELS]
    V["schedule"] = {f"{p}|{m}": [strip(s) for s in schedule[(m, p)]] for m in MODELS for p in PAIRS}
    return V, schedule


def sensitivity_values(schedule):
    """C.5 numbers. bc.sensitivity() asserts which runs never changed setting; that list depends on the selection,
    so the comparison is recomputed here with the same rule."""
    metrics = pd.read_csv(bc.OUTPUTS / "metrics_summary.csv").set_index(["model", "factor_pair"])
    runs, reproduced = {}, []
    for model, tag in MODELS.items():
        for pair in PAIRS:
            grid = pd.read_csv(bc.SENS / f"{tag}_{pair.lower()}_fixed.csv")
            ok = grid[grid["failed"].isna()]
            sel = metrics.loc[(model, pair)]
            used = {tuple(float(s[k]) for k in bc.KEYS[model]) for s in schedule[(model, pair)]}
            if len(used) == 1:
                (vals,) = used
                row = ok[np.logical_and.reduce([np.isclose(ok[k].astype(float), v) for k, v in zip(bc.KEYS[model], vals)])]
                assert len(row) == 1 and all(np.isclose(row[m].iloc[0], sel[m]) for m in ("roc_auc", "accuracy",
                                                                                          "log_loss")), (model, pair)
                reproduced.append((model, pair))
            runs[(model, pair)] = dict(grid=grid, auc=ok["roc_auc"].values, ll=ok["log_loss"].values,
                                       sel_auc=sel["roc_auc"], sel_ll=sel["log_loss"])
    below = lambda m, p: float(np.mean(runs[(m, p)]["auc"] < runs[(m, p)]["sel_auc"]))
    S = {"reproduced": [f"{p}|{m}" for m, p in reproduced],
         "below": {f"{p}|{m}": below(m, p) for m in MODELS for p in PAIRS},
         "sel_auc": {f"{p}|{m}": float(runs[(m, p)]["sel_auc"]) for m in MODELS for p in PAIRS},
         "sel_ll": {f"{p}|{m}": float(runs[(m, p)]["sel_ll"]) for m in MODELS for p in PAIRS}}
    rows = []
    for pair, name in PAIRS.items():
        for model in MODELS:
            r = runs[(model, pair)]
            n_ok, n_all = len(r["auc"]), len(r["grid"])
            rows.append([name if model == "Logistic" else "", model, str(n_ok) if n_ok == n_all else f"{n_ok} of {n_all}",
                         f3(r["sel_auc"]), f"{f3(np.median(r['auc']))} ({f3(r['auc'].min())}–{f3(r['auc'].max())})",
                         f"{100 * below(model, pair):.0f}%", f3(r["sel_ll"]),
                         f"{f3(np.median(r['ll']))} ({f3(r['ll'].min())}–{f3(r['ll'].max())})"])
    S["table_c7"] = rows
    # first_period() asserts that, in every period, the selected forecasts equal the fixed run of the setting in use
    for model in MODELS:
        for pair in PAIRS:
            bc.first_period(schedule, model, pair, 156)
    S["period_reproduced"] = True
    # every fixed-setting result, for the C.5 and Chapter 5 text
    S["fixed"] = {f"{p}|{m}": {"auc": [float(x) for x in runs[(m, p)]["auc"]],
                               "ll": [float(x) for x in runs[(m, p)]["ll"]]} for m in MODELS for p in PAIRS}
    return S, runs


if __name__ == "__main__":
    V, schedule = values()
    S, runs = sensitivity_values(schedule)
    V.update(S)
    json.dump(V, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    shutil.copy(bc.figure_c1(runs), sys.argv[2])
    print("wrote", sys.argv[1], sys.argv[2])
