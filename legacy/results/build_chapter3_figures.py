"""Rebuild the Chapter 3 figures that describe the final pipeline.

Figures 3-5  trend-scanning labels on the cumulative relative return of each
             pair, using the labels the models are trained on (unresolved final
             labels are left unshaded rather than forward-filled). Saved as one
             figure per pair and as a combined figure_trend_scanning_labels.png.
Figure 6     Ward dendrogram of the mutual-information distances, cut at
             height 2 (rebuilt with appendix_b_feature_selection.py's pipeline).
Figure 7     CPCV as implemented: the ten combinations, the purge and embargo,
             the four back-test paths and the selection rule.
Figure 8     expanding-window walk-forward design with the 12-week purge and the
             52-step retuning schedule.

Figures 1-2 come from build_active_returns_insert.py. Colours and fonts come
from thesis_style.py. Reads only files in ``data_input``; writes PNGs to
``chapter3_final_figures``.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

import thesis_style as style
from thesis_style import BLUE as UP, ORANGE as DOWN, GRID, INK, INK_2, LINE, LINE_THIN, PURGE, PURGE_HATCH


RESULTS_DIR = Path(__file__).resolve().parent
DATA_DIR = RESULTS_DIR.parent / "data_input"
OUT_DIR = RESULTS_DIR / "chapter3_final_figures"

PAIRS = ["Momentum_vs_Value", "Value_vs_Quality", "Momentum_vs_Quality"]

N_TEST, PURGE_GAP, RETUNE_EVERY = 156, 12, 52
CV_SPLITS, CV_TEST_SPLITS = 5, 2

style.use()


def load_trend_inputs():
    returns = pd.read_csv(DATA_DIR / "factor_returns.csv", parse_dates=["Date"]).set_index("Date")
    labels = pd.read_csv(DATA_DIR / "labeled_relative_factor_returns.csv",
                         parse_dates=["Date"]).set_index("Date")
    return returns, labels


def modelling_dates():
    """Dates of the modelling sample (features joined to resolved labels)."""
    data = pd.read_csv(DATA_DIR / "lagged_data.csv", parse_dates=["Date"]).set_index("Date")
    labels = pd.read_csv(DATA_DIR / "labeled_relative_factor_returns.csv",
                         parse_dates=["Date"]).set_index("Date")
    return data.join(labels, how="inner").dropna(subset=["Momentum_vs_Value_trend"]).index


def draw_trend_panel(ax, pair, returns, labels):
    first, second = pair.split("_vs_")
    cumulative = (1 + returns[first] - returns[second]).cumprod() - 1
    label = labels[f"{pair}_trend"].reindex(cumulative.index)
    n_up, n_down = int((label == 1).sum()), int((label == 0).sum())

    # One touching weekly bar per labelled week; unresolved final labels stay unshaded.
    colours = label.map({1: UP, 0: DOWN})
    shaded = colours.notna()
    ax.bar(cumulative.index[shaded], cumulative[shaded], width=7, color=list(colours[shaded]),
           linewidth=0, antialiased=False)
    line, = ax.plot(cumulative.index, cumulative, color=INK, linewidth=LINE,
                    label="Cumulative relative return")
    ax.axhline(0, color=INK_2, linewidth=LINE_THIN)
    ax.set_title(f"{first} vs {second.lower()}", loc="left", fontsize=9, color=INK)
    ax.set_ylabel("Cumulative relative return")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [Patch(color=UP, label=f"Uptrend (n = {n_up})"),
               Patch(color=DOWN, label=f"Downtrend (n = {n_down})"), line]
    ax.legend(handles=handles, loc="lower left" if pair == "Value_vs_Quality" else "upper left",
              frameon=False, fontsize=7.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def figure_trend_scanning():
    returns, labels = load_trend_inputs()
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), sharex=True)
    for ax, pair in zip(axes, PAIRS):
        draw_trend_panel(ax, pair, returns, labels)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_trend_scanning_labels.png", dpi=300)
    plt.close(fig)


def figure_trend_scanning_individual():
    """The same three panels saved as separate figures, one per pair."""
    returns, labels = load_trend_inputs()
    for pair in PAIRS:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        draw_trend_panel(ax, pair, returns, labels)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"figure_trend_scanning_{pair.lower()}.png", dpi=300)
        plt.close(fig)


def figure_dendrogram():
    """Ward dendrogram of the MI distances; links below the cut are coloured by cluster."""
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

    from appendix_b_feature_selection import CUT_HEIGHT, cluster, nmi_matrix, screened_features

    _, _, data = screened_features()
    X = data.iloc[:, :-1]                     # as in the notebook (see appendix_b_feature_selection.py)
    D = 1 - nmi_matrix(data).loc[X.columns, X.columns]
    Z, _, reps = cluster(D, height=CUT_HEIGHT)

    set_link_color_palette([style.BLUE, style.ORANGE, style.GREEN, style.PURPLE, style.PINK, style.GOLD])
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 3.0))
    dendrogram(Z, no_labels=True, color_threshold=CUT_HEIGHT, above_threshold_color=INK_2, ax=ax)
    set_link_color_palette(None)
    ax.axhline(CUT_HEIGHT, color=INK, linestyle="--", linewidth=LINE_THIN,
               label=f"Cut height = {CUT_HEIGHT} ({len(reps)} clusters)")
    ax.set_ylabel("Linkage distance (Ward)")
    ax.set_xlabel(f"{len(D)} features")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_mi_dendrogram.png", dpi=300)
    plt.close(fig)
    return len(D), len(reps)


def figure_cpcv(n_obs):
    """CPCV as implemented in the three model notebooks (N = 5 groups, k = 2 held out)."""
    groups = "ABCDE"
    combos = list(combinations(range(CV_SPLITS), CV_TEST_SPLITS))
    # Path j takes each group's held-out predictions from the j-th combination that holds it out.
    holding = {g: [i for i, c in enumerate(combos) if g in c] for g in range(CV_SPLITS)}
    path_of = {(i, g): j + 1 for g, splits in holding.items() for j, i in enumerate(splits)}
    tuning_weeks = n_obs - N_TEST
    group_weeks = tuning_weeks // CV_SPLITS
    purge_w = PURGE_GAP / group_weeks         # purge width as a share of one group, drawn to scale

    fig = plt.figure(figsize=(style.TEXT_WIDTH, 3.45))
    ax = fig.add_axes([0.085, 0.14, 0.52, 0.8])
    flow = fig.add_axes([0.635, 0.14, 0.355, 0.8])

    gap = 0.06
    n_rows = len(combos)
    for row, held in enumerate(combos):
        y = n_rows - 1 - row
        for g in range(CV_SPLITS):
            x0, w = g + gap / 2, 1 - gap
            if g in held:
                ax.add_patch(Rectangle((x0, y + gap / 2), w, 1 - gap, color=DOWN, linewidth=0))
                ax.text(g + 0.5, y + 0.5, f"P{path_of[(row, g)]}", ha="center", va="center",
                        color="white", fontsize=7.5, fontweight="bold")
                continue
            ax.add_patch(Rectangle((x0, y + gap / 2), w, 1 - gap, color=UP, linewidth=0))
            # Purge: the weeks just before a held-out group; embargo: the weeks just after one.
            for side, neighbour in (("right", g + 1), ("left", g - 1)):
                if neighbour in held:
                    xs = x0 + w - purge_w * w if side == "right" else x0
                    ax.add_patch(Rectangle((xs, y + gap / 2), purge_w * w, 1 - gap, facecolor="white",
                                           edgecolor=PURGE, hatch=PURGE_HATCH, linewidth=0.5))
        ax.text(-0.12, y + 0.5, "".join(groups[g] for g in held), ha="right", va="center",
                fontsize=7.5, color=INK)

    for g in range(CV_SPLITS):
        ax.text(g + 0.5, n_rows + 0.25, f"Split {groups[g]}", ha="center", va="bottom",
                fontsize=7.5, color=INK, fontweight="bold")
    ax.text(-0.12, n_rows + 0.25, "Validation", ha="right", va="bottom", fontsize=7.5, color=INK,
            fontweight="bold")
    ax.set_xlim(-0.05, CV_SPLITS + 0.02)
    ax.set_ylim(-0.1, n_rows + 0.7)
    ax.axis("off")

    handles = [
        Patch(color=UP, label="Training split"),
        Patch(color=DOWN, label="Validation split"),
        Patch(facecolor="white", edgecolor=PURGE, hatch=PURGE_HATCH,
              label=f"{PURGE_GAP}-week purge / embargo"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2,
               fontsize=7, handlelength=1.6, columnspacing=1.2)

    # Selection flow (right). The path ROC-AUCs are reported in Appendix C, not used for selection.
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    bh = 0.16
    steps = [
        (0.90, "Fit each candidate set on the training\nsplits of all ten combinations", False),
        (0.633, "ROC-AUC on the two validation\nsplits of each combination", False),
        (0.367, "Mean of the ten ROC-AUCs", False),
        (0.10, "Selected hyperparameter set", True),
    ]
    for i, (yc, label, final) in enumerate(steps):
        flow.add_patch(FancyBboxPatch((0.03, yc - bh / 2), 0.94, bh, boxstyle="round,pad=0.006",
                                      facecolor=UP if final else "white", edgecolor=UP, linewidth=LINE_THIN))
        flow.text(0.5, yc, label, ha="center", va="center", fontsize=7.5,
                  color="white" if final else INK, fontweight="bold" if final else "normal")
        if i:
            flow.add_patch(FancyArrowPatch((0.5, steps[i - 1][0] - bh / 2 - 0.01), (0.5, yc + bh / 2 + 0.01),
                                           arrowstyle="-|>", mutation_scale=7, color=INK_2, linewidth=LINE_THIN))

    fig.savefig(OUT_DIR / "figure_cpcv_implementation.png", dpi=300)
    plt.close(fig)


def figure_walk_forward(dates):
    n_obs = len(dates)

    # One row per retuning segment, plus the final prediction step.
    steps = [0, RETUNE_EVERY, 2 * RETUNE_EVERY, N_TEST - 1]
    row_labels = ["Step 1\n(initial tuning)", "Step 53\n(retune)", "Step 105\n(retune)", "Step 156\n(final)"]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    height = 0.55
    for row, step in enumerate(steps):
        y = len(steps) - 1 - row
        test_idx = n_obs - N_TEST + step
        train_end = test_idx - PURGE_GAP
        ax.add_patch(Rectangle((0, y - height / 2), train_end, height, color=UP, linewidth=0))
        ax.add_patch(Rectangle((train_end, y - height / 2), PURGE_GAP, height, facecolor="white",
                               edgecolor=PURGE, hatch=PURGE_HATCH, linewidth=0.5))
        ax.add_patch(Rectangle((test_idx, y - height / 2), 1.6, height, color=DOWN, linewidth=0))
        ax.text(train_end / 2, y, f"Training: {train_end} observations", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")

    test_start = n_obs - N_TEST
    ax.axvspan(test_start, n_obs, color=style.SHADE, linewidth=0, zorder=0)
    ax.text(test_start + N_TEST / 2, len(steps) - 0.35,
            f"Evaluation block: {N_TEST} weeks\n{dates[test_start]:%d %b %Y} to {dates[-1]:%d %b %Y}",
            ha="center", va="bottom", fontsize=7.5, color=INK)

    ax.set_xlim(0, n_obs + 4)
    ax.set_ylim(-0.6, len(steps) + 0.25)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(row_labels[::-1], fontsize=7.5)
    tick_pos = [0, test_start, n_obs - 1]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{dates[i]:%b %Y}" for i in tick_pos], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    handles = [
        Patch(color=UP, label="Expanding training window"),
        Patch(facecolor="white", edgecolor=PURGE, hatch=PURGE_HATCH, label=f"{PURGE_GAP}-week purge"),
        Patch(color=DOWN, label="One-week-ahead prediction"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_walk_forward_expanding.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    figure_trend_scanning()
    figure_trend_scanning_individual()
    dates = modelling_dates()
    figure_walk_forward(dates)
    figure_cpcv(len(dates))
    n_features, n_clusters = figure_dendrogram()
    print(f"modelling sample: {len(dates)} weeks, {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}; "
          f"dendrogram: {n_features} features, {n_clusters} clusters")
