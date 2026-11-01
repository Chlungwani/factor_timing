"""Shared figure style for the thesis: one palette and one set of matplotlib defaults.

Every figure script imports its colours from here, so a series keeps the same
colour wherever it appears:

  factors   Momentum blue, Value orange, Quality green
  models    Logistic blue, GAM orange, XGBoost green
  pairs     momentum-value purple, momentum-quality pink, value-quality gold
  roles     training / uptrend blue, held-out / prediction / downtrend orange,
            purged weeks white with grey hatching

Colours are always drawn at full strength (no transparency), so blue and
orange have one shade everywhere. Data series use LINE; reference lines, box
edges and arrows use LINE_THIN.

Figures are drawn at the 6.5-inch text width and carry no titles; the Word
caption names them.
"""

import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE, PINK, GOLD = "#6250d6", "#d55181", "#c98500"
INK, INK_2, GRID, SHADE, PURGE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec", "#c3c2b7"

TEXT_WIDTH = 6.5   # inches
PURGE_HATCH = "////"
LINE, LINE_THIN = 0.9, 0.6   # points

FACTOR_COLOURS = {"Momentum": BLUE, "Value": ORANGE, "Quality": GREEN}
MODEL_COLOURS = {"Logistic": BLUE, "GAM": ORANGE, "XGBoost": GREEN}
PAIR_COLOURS = {"Momentum_vs_Value": PURPLE, "Momentum_vs_Quality": PINK, "Value_vs_Quality": GOLD}
FEATURE_GROUP_COLOURS = {
    "Price-based features": BLUE,
    "Fundamental": PINK,
    "Sentiment indicators": ORANGE,
    "Macroeconomic": PURPLE,
    "Factor-specific performance": GOLD,
    "Factor-specific volatility": GREEN,
}

RC = {
    "font.family": "Calibri",
    "font.size": 8,
    "lines.linewidth": LINE,
    "hatch.linewidth": 0.5,
    "axes.linewidth": LINE_THIN,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
}


def use():
    plt.rcParams.update(RC)
