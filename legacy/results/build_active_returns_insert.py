"""Rolling 12-month active returns and active-return correlations for Section 3.3.

Answers the supervisor's request for 'a graph of rolling 1 year active returns for
the actual factors ... and a correlation analysis'. Uses the monthly factor and
benchmark workbook (the only source with a benchmark series), on the common sample
in which all three factors are available.

Writes
  chapter3_final_figures/figure_rolling_active_returns.png
  chapter3_final_figures/figure_rolling_active_correlations.png
  chapter3_final_figures/active_return_correlations.csv
  chapter3_final_figures/active_return_summary.csv
  Chapter3_Section3.3_Active_Returns_Insert.docx   (to the folder given as argv[1])
"""

import copy
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

import thesis_style as style

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parent.parent
DATA = RESULTS.parent / "data_input" / "factor_benchmark_returns.xlsx"
FIG_DIR = RESULTS / "chapter3_final_figures"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
TEMPLATE = ROOT / ".tmp_chapter3_minimal" / "thesis_current.docx"

FACTORS = ["Momentum", "Value", "Quality"]
PAIRS = [("Momentum", "Value"), ("Momentum", "Quality"), ("Value", "Quality")]
COLOURS = style.FACTOR_COLOURS
# Pairs are different entities from the factors, so they get their own hues (violet, magenta, amber).
PAIR_COLOURS = {f"{a}–{b.lower()}": style.PAIR_COLOURS[f"{a}_vs_{b}"] for a, b in PAIRS}
INK, INK_2, GRID, SHADE = style.INK, style.INK_2, style.GRID, style.SHADE
WINDOW_RETURN, WINDOW_CORR = 12, 36
EVAL_START, EVAL_END = "2022-01-01", "2025-01-31"

plt.rcParams.update({k: v for k, v in style.RC.items() if not k.startswith(("axes.title", "legend."))})


# ------------------------------------------------------------------ data --

def load():
    df = pd.read_excel(DATA).rename(columns={"Data": "Date"}).set_index("Date").sort_index()
    return df.dropna(subset=FACTORS + ["Benchmark"])       # common sample: Quality starts Feb 2007


def analyse(df):
    compounded = (1 + df).rolling(WINDOW_RETURN).apply(np.prod, raw=True) - 1
    active_12m = compounded[FACTORS].sub(compounded["Benchmark"], axis=0).dropna()
    active = df[FACTORS].sub(df["Benchmark"], axis=0)
    down = df["Benchmark"] < 0

    rolling_corr = pd.DataFrame({f"{a}–{b.lower()}": active[a].rolling(WINDOW_CORR).corr(active[b])
                                 for a, b in PAIRS}).dropna()
    corr_rows = []
    for (a, b), name in zip(PAIRS, rolling_corr.columns):
        rc = rolling_corr[name]
        corr_rows.append({
            "pair": name,
            "raw_returns": df[a].corr(df[b]),
            "active_returns": active[a].corr(active[b]),
            "active_benchmark_down": active[a][down].corr(active[b][down]),
            "active_benchmark_up": active[a][~down].corr(active[b][~down]),
            "rolling_36m_min": rc.min(),
            "rolling_36m_max": rc.max(),
            "rolling_36m_sign_changes": int((np.sign(rc).diff().abs() > 0).sum()),
        })
    corr = pd.DataFrame(corr_rows)
    summary = pd.DataFrame({
        "annualised_active_return": active.mean() * 12,
        "tracking_error": active.std() * np.sqrt(12),
        "share_12m_windows_positive": (active_12m > 0).mean(),
        "best_12m": active_12m.max(),
        "worst_12m": active_12m.min(),
    })
    raw_bench = {f: df[f].corr(df["Benchmark"]) for f in FACTORS}
    return active_12m, rolling_corr, corr, summary, down.sum(), raw_bench


# ---------------------------------------------------------------- figure --

def _frame(ax):
    """Shared styling: evaluation-block shading, zero line, light grid, two-year ticks."""
    ax.axvspan(pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END), color=SHADE, linewidth=0, zorder=0)
    ax.axhline(0, color=INK_2, linewidth=style.LINE_THIN, zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlim(pd.Timestamp("2007-07-01"), pd.Timestamp("2025-12-31"))   # same span in both figures
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def figure_active_returns(active_12m, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    _frame(ax)
    for f in FACTORS:
        ax.plot(active_12m.index, active_12m[f] * 100, color=COLOURS[f], linewidth=style.LINE, label=f, zorder=2)
    ax.set_ylabel("Rolling 12-month active return (%)")
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    ax.text(pd.Timestamp("2023-07-01"), ax.get_ylim()[1] * 0.97, "Evaluation\nblock", ha="center", va="top",
            fontsize=7, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_correlations(rolling_corr, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    _frame(ax)
    for name in rolling_corr.columns:
        ax.plot(rolling_corr.index, rolling_corr[name], color=PAIR_COLOURS[name], linewidth=style.LINE,
                label=name, zorder=2)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"{WINDOW_CORR}-month correlation")
    ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=8)
    ax.text(pd.Timestamp("2023-07-01"), 0.97, "Evaluation\nblock", ha="center", va="top", fontsize=7, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------- document --

def new_document():
    doc = Document(TEMPLATE)
    body = doc.element.body
    sect = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is not sect:
            body.remove(child)
    return doc


def para(doc, text, style="Normal", highlight=True, size=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, keep=False):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    p.paragraph_format.keep_with_next = keep
    run = p.add_run(text)
    if size:
        run.font.size = Pt(size)
    if highlight:
        hl = OxmlElement("w:highlight"); hl.set(qn("w:val"), "yellow"); run._r.get_or_add_rPr().append(hl)
    return p


def table(doc, header, rows, widths):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = doc.styles["Table Grid4"]
    for cell, text in zip(t.rows[0].cells, header):
        cell.paragraphs[0].add_run(text).bold = True
    for values in rows:
        for cell, value in zip(t.add_row().cells, values):
            cell.paragraphs[0].add_run(value)
    for r, row in enumerate(t.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for c, (cell, width) in enumerate(zip(row.cells, widths)):
            cell.width = width
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == 0 or r == 0 else WD_ALIGN_PARAGRAPH.RIGHT
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_with_next = r < len(t.rows) - 1
                for run in p.runs:
                    run.font.size = Pt(8.5)
    return t


def num(x, digits=2):
    """Format a number with a true minus sign and no negative zero."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


def picture(doc, path):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(6.27))


def build_doc(df, active_12m, rolling_corr, corr, summary, n_down, raw_bench, fig_paths):
    doc = new_document()
    start, end = df.index.min(), df.index.max()
    pct = lambda x: num(x * 100, 1) + "%"
    s = summary
    mv, mq, vq = (corr.set_index("pair").loc[p] for p in rolling_corr.columns)

    para(doc, "Placement: Section 3.3, after the paragraph ending ‘… (Bartram, Lohre, Pope, & "
              "Ranganathan, 2021; Ilmanen & Kizer, 2012).’ Figure and table numbers are placeholders; "
              "inserting them with Word’s caption feature renumbers the later figures and tables. Name the "
              "benchmark where marked.", size=9, align=WD_ALIGN_PARAGRAPH.LEFT, highlight=False)

    para(doc, f"Figure X illustrates this rotation for the three factors studied here. It plots rolling "
              f"12-month active returns relative to the [benchmark name] benchmark, measured as the compounded "
              f"12-month factor return less the compounded 12-month benchmark return, using monthly data from "
              f"{start:%B %Y} to {end:%B %Y}. Leadership changes repeatedly. Momentum earns a positive 12-month "
              f"active return in {pct(s.loc['Momentum', 'share_12m_windows_positive'])} of windows, value in "
              f"{pct(s.loc['Value', 'share_12m_windows_positive'])} and quality in "
              f"{pct(s.loc['Quality', 'share_12m_windows_positive'])}, and 12-month active returns range from "
              f"{pct(s['worst_12m'].min())} to {pct(s['best_12m'].max())}. Over the full sample, the annualised "
              f"active return is {pct(s.loc['Momentum', 'annualised_active_return'])} for momentum, "
              f"{pct(s.loc['Value', 'annualised_active_return'])} for value and "
              f"{pct(s.loc['Quality', 'annualised_active_return'])} for quality, with tracking errors of "
              f"{pct(s['tracking_error'].min())} to {pct(s['tracking_error'].max())}.")

    picture(doc, fig_paths[0])
    para(doc, "Figure X: Rolling 12-month active returns", style="Caption", align=WD_ALIGN_PARAGRAPH.CENTER)

    raw_lo, raw_hi = corr["raw_returns"].min(), corr["raw_returns"].max()
    act_lo, act_hi = corr["active_returns"].min(), corr["active_returns"].max()
    para(doc, f"Table X reports the correlations of monthly returns. In raw terms the three factors are highly "
              f"correlated ({num(raw_lo)} to {num(raw_hi)}), because each is a long-only portfolio that shares the "
              f"market’s exposure (correlations of {num(min(raw_bench.values()))} to "
              f"{num(max(raw_bench.values()))} with the benchmark). Measured against the benchmark, the correlations "
              f"fall to between {num(act_lo)} and {num(act_hi)}: momentum and value active returns are negatively "
              f"correlated ({num(mv.active_returns)}), while quality is close to uncorrelated with both. Figure Y "
              f"shows that these relationships are not stable. The {WINDOW_CORR}-month correlation between "
              f"momentum and value active returns ranges from {num(mv.rolling_36m_min)} to {num(mv.rolling_36m_max)}, "
              f"and the correlations involving quality range from {num(min(mq.rolling_36m_min, vq.rolling_36m_min))} "
              f"to {num(max(mq.rolling_36m_max, vq.rolling_36m_max))} and change sign repeatedly. Splitting the "
              f"sample by the sign of the benchmark return ({n_down} of {len(df)} months are negative) changes the "
              "correlations little, so the convergence of styles in falling markets described above is not "
              "evident in these monthly South African data.")
    picture(doc, fig_paths[1])
    para(doc, f"Figure Y: Rolling {WINDOW_CORR}-month correlations of active returns", style="Caption",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Table X: Correlations of monthly factor returns", style="Caption", highlight=True, keep=True,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    table(doc, ["Factor pair", "Raw returns", "Active returns", "Active, benchmark down",
                "Active, benchmark up", f"Rolling {WINDOW_CORR}-month range"],
          [[r.pair, f"{num(r.raw_returns)}", f"{num(r.active_returns)}", f"{num(r.active_benchmark_down)}",
            f"{num(r.active_benchmark_up)}", f"{num(r.rolling_36m_min)} to {num(r.rolling_36m_max)}"]
           for r in corr.itertuples()],
          [Inches(1.5), Inches(0.8), Inches(0.85), Inches(1.0), Inches(0.95), Inches(1.15)])
    para(doc, f"Note: monthly data, {start:%B %Y} to {end:%B %Y}. Active returns are factor returns less the "
              "benchmark return; down and up months are defined by the sign of the monthly benchmark return. In "
              "Figures X and Y, the shaded period is the walk-forward evaluation block.", size=9,
         align=WD_ALIGN_PARAGRAPH.LEFT)
    para(doc, "Because each modelling target is a difference between two factor returns, these benchmark-relative "
              "descriptives do not change the targets themselves, which are identical in raw and active terms.")
    out = OUT_DIR / "Chapter3_Section3.3_Active_Returns_Insert.docx"
    doc.save(out)
    return out


def main():
    df = load()
    active_12m, rolling_corr, corr, summary, n_down, raw_bench = analyse(df)
    FIG_DIR.mkdir(exist_ok=True)
    fig_paths = (FIG_DIR / "figure_rolling_active_returns.png",
                 FIG_DIR / "figure_rolling_active_correlations.png")
    figure_active_returns(active_12m, fig_paths[0])
    figure_correlations(rolling_corr, fig_paths[1])
    corr.round(3).to_csv(FIG_DIR / "active_return_correlations.csv", index=False)
    summary.round(4).to_csv(FIG_DIR / "active_return_summary.csv")
    out = build_doc(df, active_12m, rolling_corr, corr, summary, n_down, raw_bench, fig_paths)
    pd.set_option("display.width", 200)
    print(f"sample {df.index.min():%b %Y} to {df.index.max():%b %Y}, {len(df)} months, {n_down} down months")
    print(corr.round(2).to_string(index=False))
    print(summary.round(3).to_string())
    print("raw corr with benchmark:", {k: round(v, 2) for k, v in raw_bench.items()})
    print(out)


if __name__ == "__main__":
    main()
