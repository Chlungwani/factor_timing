"""Build reproducible Chapter 4 figures from the final CPCV model outputs.

Usage: python build_chapter4_corrected_figures.py [outputs_dir] [figures_dir]

The script reads only files under ``results/model_outputs`` (or outputs_dir)
and writes figures to ``results/chapter4_final_figures`` (or figures_dir).  It
does not read from or modify the thesis Word document. Colours and fonts come
from thesis_style.py; figures carry no titles (the Word captions name them).
"""

import sys
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import thesis_style as style
from thesis_style import FEATURE_GROUP_COLOURS, GRID, INK_2, LINE, LINE_THIN, MODEL_COLOURS


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "model_outputs"
FIGURES_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_DIR / "chapter4_final_figures"

MODELS = ["Logistic", "GAM", "XGBoost"]
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]
PAIR_TITLES = {pair: f"{a} vs {b.lower()}" for pair in PAIRS for a, b in [pair.split("_vs_")]}
TAGS = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}

FEATURE_GROUPS = {
    "Price-based features": {
        "VolTermStructPC1",
        "VolTermStructPC2",
        "GLD_FracDiff",
        "GLD_RollSD",
        "USDZARV1M.Curncy_QS",
        "USDZARV1Y.Curncy_QS",
        "GlobalBondVol6",
        "GlobalPropertyVol12",
        "EURUSDV6M.Curncy_QS",
    },
    "Fundamental": {
        "MXWD_Fundamental_PC1",
        "JALSH_Fundamental_PC1",
        "TOP40.Index.Price.to.Book.Ratio_Diff",
        "MXWD.Index.PE.Ratio_QS",
        "TOP40.Index.Price.to.Ebitda_Diff",
        "JALSH.Index.Price.to.Sales.Ratio_Diff",
    },
    "Sentiment indicators": {
        "MichgnConcIndx",
        "GenComIndexPC1_QS",
        "GLD_StructBreak",
        "GLD_LempZiev",
        "GLD_RollSkew",
        "GLD_RollKurt",
        "GlobalEquityTS",
        "GlobalEquity_EntDyn",
    },
    "Macroeconomic": {
        "SA_RB_Curvature",
        "US_RB_Curvature",
        "SA_NB_Slope",
        "SA_CDS_Slope",
        "RandPPP_Factor_Inst_TS_ST",
        "PPP.ZA.Index_QS",
        "BCMPEBLS.Index",
        "ZAR18M.Curncy_QS",
        "USDZAR_EntDyn",
    },
    "Factor-specific performance": {
        "mom1m_momentum_vs_quality",
        "mom1m_momentum_vs_value",
        "mom1m_value_vs_quality",
        "mom3m_momentum_vs_quality",
        "mom3m_momentum_vs_value",
        "mom3m_value_vs_quality",
        "corr_momentum_quality_12",
        "corr_momentum_value_12",
        "corr_value_quality_12",
        "zrel_value_vs_quality_12",
        "zrel_momentum_vs_quality_12",
        "zrel_momentum_vs_value_12",
    },
    "Factor-specific volatility": {
        "relvol_momentum_vs_quality_12",
        "relvol_momentum_vs_value_12",
        "relvol_value_vs_quality_12",
        "volratio_momentum_quality_12",
        "volratio_momentum_value_12",
        "volratio_value_quality_12",
    },
}
FEATURE_TO_GROUP = {
    feature: group for group, features in FEATURE_GROUPS.items() for feature in features
}


def load_probabilities(model, pair):
    path = OUTPUTS_DIR / "probs" / f"{TAGS[model]}_{pair.lower()}_probs.csv"
    return pd.read_csv(path, parse_dates=["Date"])


def load_shap(model, pair):
    path = OUTPUTS_DIR / "shap" / f"{TAGS[model]}_{pair.lower()}_shap.csv"
    frame = pd.read_csv(path, index_col=0)
    return frame.iloc[:, 0].sort_values(ascending=False)


def style_axes(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_roc(ax, pair):
    for model in MODELS:
        data = load_probabilities(model, pair)
        false_positive_rate, true_positive_rate, _ = roc_curve(data["actual"], data["probability"])
        auc = roc_auc_score(data["actual"], data["probability"])
        ax.plot(false_positive_rate, true_positive_rate, label=f"{model} (AUC = {auc:.3f})",
                color=MODEL_COLOURS[model], linewidth=LINE)
    ax.plot([0, 1], [0, 1], "--", color=INK_2, linewidth=LINE_THIN)
    ax.set_title(PAIR_TITLES[pair])
    ax.set_xlabel("False-positive rate")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.4)
    style_axes(ax)


def build_roc_figure():
    fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 2.6), sharex=True, sharey=True)
    for ax, pair in zip(axes, PAIRS):
        draw_roc(ax, pair)
    axes[0].set_ylabel("True-positive rate")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGURES_DIR / "figure_final_roc_curves.png", dpi=300, facecolor="white")
    plt.close(fig)

    for pair in PAIRS:
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        draw_roc(ax, pair)
        ax.set_title("")
        ax.set_ylabel("True-positive rate")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"figure_final_roc_{pair.lower()}.png", dpi=300, facecolor="white")
        plt.close(fig)


def draw_probabilities(ax, pair):
    realised = None
    for model in MODELS:
        data = load_probabilities(model, pair)
        ax.plot(data["Date"], data["probability"], color=MODEL_COLOURS[model], linewidth=LINE,
                label=model)
        if realised is None:
            realised = data[["Date", "actual"]]
    ax.scatter(realised["Date"], realised["actual"], s=3, color=INK_2,
               label="Realised class", zorder=3, linewidths=0)
    ax.axhline(0.5, color=INK_2, linestyle="--", linewidth=LINE_THIN)
    ax.set_ylim(-0.06, 1.06)
    ax.set_ylabel("Probability / class")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_axes(ax, axis="y")


def build_probability_figures():
    fig, axes = plt.subplots(3, 1, figsize=(style.TEXT_WIDTH, 6.2), sharex=True, sharey=True)
    for ax, pair in zip(axes, PAIRS):
        draw_probabilities(ax, pair)
        ax.set_title(PAIR_TITLES[pair])
    axes[-1].set_xlabel("Prediction date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               markerscale=2.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2)
    fig.savefig(FIGURES_DIR / "figure_final_probabilities_consolidated.png", dpi=300, facecolor="white")
    plt.close(fig)

    # One figure per pair (Chapter 4 places each under its pair's discussion; the caption names the pair).
    for pair in PAIRS:
        fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 2.7))
        draw_probabilities(ax, pair)
        ax.set_xlabel("Prediction date")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
                   markerscale=2.5)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(FIGURES_DIR / f"figure_final_probabilities_{pair.lower()}.png", dpi=300, facecolor="white")
        plt.close(fig)


def wrap_feature(name, width=22):
    breakable = name.replace("_", "_ ").replace(".", ". ")
    lines = textwrap.wrap(
        breakable,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(line.replace(" ", "") for line in lines)


def build_shap_figures(top_n=10):
    for pair in PAIRS:
        fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 4.4))
        for ax, model in zip(axes, MODELS):
            importance = load_shap(model, pair).head(top_n).sort_values()
            missing = [name for name in importance.index if name not in FEATURE_TO_GROUP]
            if missing:
                raise ValueError(f"Unclassified SHAP features: {missing}")
            ax.barh(
                [wrap_feature(name) for name in importance.index],
                importance.values,
                color=[FEATURE_GROUP_COLOURS[FEATURE_TO_GROUP[name]] for name in importance.index],
                height=0.7,
            )
            ax.set_title(model)
            ax.set_xlabel("Mean |SHAP value|")
            ax.tick_params(axis="y", labelsize=6.5, length=0)
            ax.tick_params(axis="x", labelsize=7)
            style_axes(ax, axis="x")
        fig.legend(
            handles=[Patch(facecolor=colour, label=group) for group, colour in FEATURE_GROUP_COLOURS.items()],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            title="Feature category",
            title_fontsize=7.5,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.8)
        fig.savefig(FIGURES_DIR / f"figure_final_shap_{pair.lower()}.png", dpi=300, facecolor="white")
        plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    style.use()
    build_roc_figure()
    build_probability_figures()
    build_shap_figures()
    print(f"Wrote final figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
