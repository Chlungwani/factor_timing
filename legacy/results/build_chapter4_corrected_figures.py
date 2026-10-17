"""Build reproducible Chapter 4 figures from the final CPCV model outputs.

The script reads only files under ``results/model_outputs`` and writes figures
to ``results/chapter4_final_figures``.  It does not read from or modify the
thesis Word document.
"""

from pathlib import Path
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = RESULTS_DIR / "model_outputs"
FIGURES_DIR = RESULTS_DIR / "chapter4_final_figures"

MODELS = ["Logistic", "GAM", "XGBoost"]
PAIRS = ["Momentum_vs_Value", "Momentum_vs_Quality", "Value_vs_Quality"]
PAIR_TITLES = {
    "Momentum_vs_Value": "Momentum vs Value",
    "Momentum_vs_Quality": "Momentum vs Quality",
    "Value_vs_Quality": "Value vs Quality",
}
TAGS = {"Logistic": "logistic", "GAM": "gam", "XGBoost": "xgboost"}
COLORS = {"Logistic": "#0072B2", "GAM": "#E69F00", "XGBoost": "#009E73"}

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
    "Macro-economic": {
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
FEATURE_GROUP_COLORS = {
    "Price-based features": "#4E79A7",
    "Fundamental": "#E15759",
    "Sentiment indicators": "#F28E2B",
    "Macro-economic": "#B07AA1",
    "Factor-specific performance": "#76B7B2",
    "Factor-specific volatility": "#59A14F",
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


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.7)


def build_roc_figure():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.4), sharex=True, sharey=True)
    for ax, pair in zip(axes, PAIRS):
        for model in MODELS:
            data = load_probabilities(model, pair)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                data["actual"], data["probability"]
            )
            auc = roc_auc_score(data["actual"], data["probability"])
            ax.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model} (AUC = {auc:.2f})",
                color=COLORS[model],
                linewidth=2.5,
            )
        ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1.3)
        ax.set_title(PAIR_TITLES[pair], fontsize=13.5, fontweight="bold")
        ax.set_xlabel("False-positive rate", fontsize=11.5, fontweight="bold")
        ax.set_ylabel(
            "True-positive rate" if pair == PAIRS[0] else "",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=10.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#CCCCCC",
            fontsize=9.5,
            handlelength=1.5,
        )
        for label in legend.get_texts():
            label.set_fontweight("bold")
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)
    fig.suptitle(
        "Final out-of-sample ROC curves", fontsize=17, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.1)
    path = FIGURES_DIR / "figure_final_roc_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for pair in PAIRS:
        fig, ax = plt.subplots(figsize=(7.2, 5.7))
        for model in MODELS:
            data = load_probabilities(model, pair)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                data["actual"], data["probability"]
            )
            auc = roc_auc_score(data["actual"], data["probability"])
            ax.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model} (AUC = {auc:.3f})",
                color=COLORS[model],
                linewidth=2.2,
            )
        ax.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1)
        ax.set_title(
            f"Final out-of-sample ROC curves: {PAIR_TITLES[pair]}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.legend(loc="lower right", frameon=False)
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)
        fig.tight_layout()
        path = FIGURES_DIR / f"figure_final_roc_{pair.lower()}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def build_probability_figures():
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.2), sharex=True, sharey=True)
    for panel_index, (ax, pair) in enumerate(zip(axes, PAIRS)):
        realised = None
        for model in MODELS:
            data = load_probabilities(model, pair)
            ax.plot(
                data["Date"],
                data["probability"],
                color=COLORS[model],
                linewidth=1.9,
                label=model,
            )
            if realised is None:
                realised = data[["Date", "actual"]]

        ax.scatter(
            realised["Date"],
            realised["actual"],
            s=13,
            color="#4D4D4D",
            alpha=0.5,
            label="Realised class",
            zorder=3,
        )
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.2)
        ax.set_ylim(-0.06, 1.06)
        ax.set_title(
            f"{chr(65 + panel_index)}. {PAIR_TITLES[pair]}",
            loc="left",
            fontsize=13.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_ylabel("Probability / class", fontsize=11.5, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        style_axes(ax)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Prediction date", fontsize=12.5, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=11.5,
    )
    for label in legend.get_texts():
        label.set_fontweight("bold")
    fig.suptitle(
        "Final walk-forward predicted probabilities",
        fontsize=17.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93), h_pad=1.9)
    path = FIGURES_DIR / "figure_final_probabilities_consolidated.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
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
        fig, axes = plt.subplots(1, 3, figsize=(13, 8.6))
        for ax, model in zip(axes, MODELS):
            importance = load_shap(model, pair).head(top_n).sort_values()
            missing = [name for name in importance.index if name not in FEATURE_TO_GROUP]
            if missing:
                raise ValueError(f"Unclassified SHAP features: {missing}")
            ax.barh(
                [wrap_feature(name) for name in importance.index],
                importance.values,
                color=[
                    FEATURE_GROUP_COLORS[FEATURE_TO_GROUP[name]]
                    for name in importance.index
                ],
                alpha=0.95,
            )
            ax.set_title(model, fontsize=15, fontweight="bold")
            ax.set_xlabel("Mean |SHAP value|", fontsize=12.5, fontweight="bold")
            ax.tick_params(axis="x", labelsize=11)
            ax.tick_params(axis="y", labelsize=11)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            style_axes(ax)
        fig.suptitle(
            f"Final global feature importance: {PAIR_TITLES[pair]}",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )
        legend = fig.legend(
            handles=[
                Patch(facecolor=color, label=group)
                for group, color in FEATURE_GROUP_COLORS.items()
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
            title="Feature classification group",
            fontsize=12,
            title_fontsize=12.5,
        )
        legend.get_title().set_fontweight("bold")
        for label in legend.get_texts():
            label.set_fontweight("bold")
        fig.tight_layout(rect=(0, 0.17, 1, 0.93), w_pad=1.6)
        path = FIGURES_DIR / f"figure_final_shap_{pair.lower()}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlepad": 10,
            "figure.dpi": 120,
        }
    )
    build_roc_figure()
    build_probability_figures()
    build_shap_figures()
    print(f"Wrote final figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
