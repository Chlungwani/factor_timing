"""Appendix B (Feature Selection): the content written after the thesis's 'Appendix B' heading.

Reads the outputs of appendix_b_feature_selection.py. Used by build_appendices.py, which assembles
Appendices A-C in one document; Table B.8 (the economic classification of the 50 modelling features) is
taken from the Chapter 3 amendment document, where it was moved out of Section 3.6. Formatting follows
Appendix A of the thesis: 'Appendix Heading 1' title, bold 'B.n:' subheadings and 8-point grid tables.
"""

import copy
import json
import sys
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

RESULTS = Path(__file__).resolve().parent
DATA = RESULTS / "appendix_b"
TEXT_WIDTH = Inches(6.27)

S = json.loads((DATA / "summary.json").read_text())
b1 = pd.read_csv(DATA / "b1_clusters.csv")
b2 = pd.read_csv(DATA / "b2_screening.csv")
b3 = pd.read_csv(DATA / "b3_cut_heights.csv")
b4 = pd.read_csv(DATA / "b4_stability.csv")
b5 = pd.read_csv(DATA / "b5_high_mi_low_correlation.csv")


def words(n):
    """Capitalised number word for sentence starts (1-12)."""
    return ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven",
            "Twelve"][n]


def fmt_date(iso):
    return pd.Timestamp(iso).strftime("%-d %B %Y") if sys.platform != "win32" else pd.Timestamp(iso).strftime("%#d %B %Y")


# --------------------------------------------------------------- document --

def paragraph(document, text, bold=False, size=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6,
              highlight=False, keep_next=False):
    p = document.add_paragraph(style="Normal")
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = keep_next
    run = p.add_run(text)
    run.bold = bold
    if size:
        run.font.size = Pt(size)
    if highlight:
        rpr = run._r.get_or_add_rPr()
        hl = OxmlElement("w:highlight"); hl.set(qn("w:val"), "yellow"); rpr.append(hl)
    return p


def subheading(document, text):
    p = paragraph(document, text, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=4, keep_next=True)
    p.paragraph_format.space_before = Pt(10)
    return p


def caption(document, text):
    p = document.add_paragraph(style="Caption")
    p.add_run(text)
    p.paragraph_format.keep_with_next = True
    return p


def table(document, header, rows, widths, font=8, align_right=(), keep_together=True):
    t = document.add_table(rows=1, cols=len(header))
    t.style = document.styles["Table Grid4"]
    for cell, text in zip(t.rows[0].cells, header):
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
    for values in rows:
        cells = t.add_row().cells
        for cell, value in zip(cells, values):
            cell.paragraphs[0].add_run(str(value))
    for r_index, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if r_index == 0:
            hdr = OxmlElement("w:tblHeader"); trpr.append(hdr)
        for c_index, (cell, width) in enumerate(zip(row.cells, widths)):
            cell.width = width
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                p.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if c_index in align_right and r_index > 0
                               else WD_ALIGN_PARAGRAPH.LEFT)
                p.paragraph_format.keep_with_next = keep_together and r_index < len(t.rows) - 1
                for run in p.runs:
                    run.font.size = Pt(font)
    document.add_paragraph(style="Normal").paragraph_format.space_after = Pt(2)
    return t


def figure(document, path):
    p = document.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=TEXT_WIDTH)


# ---------------------------------------------------------------- content --

def classification(table_el):
    """Feature -> category, from Table B.8 (the category is given on the first row of each group)."""
    mapping, current = {}, None
    for tr in table_el.findall(qn("w:tr"))[1:]:
        cells = ["".join(t.text or "" for t in tc.iter(qn("w:t"))).strip() for tc in tr.findall(qn("w:tc"))]
        current = cells[0] or current
        mapping[cells[1]] = current
    return mapping


def content(doc, table_b8):
    """Append Appendix B after its heading. `table_b8` is the w:tbl of Table B.8 (copied, not moved)."""
    W = Inches
    categories = classification(table_b8)

    paragraph(doc, "Review note: this appendix replaces the ‘work in progress’ placeholder; all of its "
                   "content is new.", size=10, highlight=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    paragraph(doc, "This appendix documents the mutual-information (MI) feature selection described in Section 3.6 "
                   "and tests how sensitive its outcome is to the main estimation choices. All results are "
                   "recomputed from the candidate feature file with the procedure implemented in the "
                   "feature-selection notebook. The recomputation recovers all 32 representative features used in "
                   "the models.")

    # B.1 ------------------------------------------------------------------
    subheading(doc, "B.1: Screening of the candidate features")
    paragraph(doc, "Table B.1 traces the candidate features from the initial universe to the final predictor set. "
                   f"The missing-value screen is applied before the clustering; Table B.2 lists the "
                   f"{S['removed_missing']} features it removes. The MI distances are estimated on the "
                   f"{S['mi_weeks']} weekly observations from {fmt_date(S['mi_start'])} to {fmt_date(S['mi_end'])}, "
                   "the period in which every remaining feature is available after forward filling.")
    caption(doc, "Table B.1: From candidate features to the final predictor set")
    table(doc, ["Step", "Features"], [
        ["Candidate features in the feature file", S["candidates"]],
        ["Removed: more than 30% of weekly observations missing (Table B.2)", f"−{S['removed_missing']}"],
        ["Passing the missing-value screen", S["screened"]],
        [f"Entering the MI distance matrix ({S['omitted_feature']} omitted by the implementation; "
         "Section 3.6)", S["clustered"]],
        ["Clusters at a cut height of 2, one representative each (Table B.3)", S["baseline_clusters"]],
        ["Relative-factor features added by design (Section 3.6)", 18],
        ["Final predictor set", S["baseline_clusters"] + 18],
    ], [W(5.2), W(1.0)], font=9, align_right=(1,))
    caption(doc, "Table B.2: Features removed by the missing-value screen")
    table(doc, ["Feature", "Share of weeks missing", "First available"],
          [[r.feature, f"{r.missing_share:.0%}", pd.Timestamp(r.first_available).strftime("%d %b %Y")]
           for r in b2.itertuples()],
          [W(3.2), W(1.5), W(1.5)], align_right=(1,))
    const = S["constant_features"]
    paragraph(doc, f"The screen does not remove features without variation. {words(len(const))} of the clustered features "
                   "are constant over the clustering sample: the Nelson–Siegel decay parameters (Lambda) of the "
                   "South African and US nominal and real yield curves and of the South African CDS curve, in "
                   f"level and quantile-scaled form ({', '.join(const)}). Their entropy is zero, so their "
                   "normalised MI with every other feature is zero and they lie at the maximum distance of one "
                   "from all other features. They carry no information and should be removed before estimation; "
                   "Section B.2 shows how they affect one selected representative.")

    # B.2 ------------------------------------------------------------------
    subheading(doc, "B.2: Cluster membership and representatives")
    n_duplicates = int(((b1["size"] == 2) & (b1["within_mean_distance"] == 0)).sum())
    paragraph(doc, "Table B.3 lists the 32 clusters formed at a cut height of 2, ordered by size. For each cluster "
                   "it reports the representative retained for modelling, its category in Table 2, the cluster "
                   "size, the representative’s mean MI distance to the members of its cluster, the "
                   "mean distance between all pairs of members, and the remaining members in order of increasing "
                   "mean distance. Clusters with low within-cluster distances group closely related series, such "
                   "as the same instrument at different tenors (ZAR forward points, USDZAR implied volatility and "
                   f"the VIX term structure). {words(n_duplicates)} two-member clusters have a distance of zero because "
                   "they pair duplicated series, such as gold ETF statistics (GLD_*) and the corresponding gold "
                   "spot statistics (XAU*). The largest clusters have within-cluster distances close to the median "
                   f"of {S['median_distance']:.2f} across all feature pairs: their members are only weakly related, "
                   "so their representatives summarise a heterogeneous group rather than a set of near-duplicates.")
    ppp_values = ", ".join(f"{v:.2f}" for v in S["ppp_values"])
    paragraph(doc, "One selected representative is itself nearly uninformative. PPP.ZA.Index_QS takes only the "
                   f"values {ppp_values}; it equals 1.00 in every week until "
                   f"{fmt_date(S['ppp_first_change'])} and varies only within the evaluation block. It is "
                   "selected because it is the only member of its cluster with any variation: the other ten "
                   "members are the constant Lambda features (within-cluster distance 1.00). It therefore carries "
                   "no information in the initial training window and is not among the ten most influential "
                   "features of any model in Chapter 4.")
    caption(doc, "Table B.3: Clusters, representatives and members (cut height 2)")
    table(doc, ["Representative", "Category", "Size", "Rep. mean dist.", "Within-cluster dist.", "Other members"],
          [[r.representative, categories.get(r.representative, ""), r.size, f"{r.rep_mean_distance:.2f}",
            f"{r.within_mean_distance:.2f}", r.other_members if isinstance(r.other_members, str) else "–"]
           for r in b1.itertuples()],
          [W(1.35), W(0.8), W(0.35), W(0.5), W(0.55), W(2.75)], font=7, align_right=(2, 3, 4),
          keep_together=False)

    # B.3 ------------------------------------------------------------------
    subheading(doc, "B.3: Choice of cut height")
    sil32, silmax, kmax = S["silhouette_at_32"], S["silhouette_max"], S["silhouette_argmax"]
    paragraph(doc, "Table B.4 reports the number of clusters and the mean silhouette score for a range of cut "
                   "heights, and Figure B.1 plots the silhouette score for partitions of 5 to 100 clusters. The "
                   "silhouette score compares each feature’s mean distance to the other members of its cluster "
                   "with its mean distance to the nearest other cluster; values near zero indicate overlapping "
                   f"clusters. The scores remain below 0.1 throughout, at {sil32:.3f} for the 32 clusters used. "
                   f"Beyond about 20 clusters they increase with the number of clusters, reaching {silmax:.3f} at "
                   f"{kmax} clusters, the largest partition considered, without an interior maximum. The data therefore do not suggest a natural number of clusters, "
                   "and the cut height of 2 reflects a judgement that balances a compact predictor set against "
                   "merging dissimilar features.")
    caption(doc, "Table B.4: Number of clusters and silhouette score by cut height")
    table(doc, ["Cut height", "Clusters", "Mean silhouette score"],
          [[f"{r.cut_height:g}", r.clusters, f"{r.silhouette:.3f}"] for r in b3.itertuples()],
          [W(1.4), W(1.4), W(1.8)], font=9, align_right=(0, 1, 2))
    figure(doc, DATA / "figure_b1_silhouette.png")
    caption(doc, "Figure B.1: Mean silhouette score by number of clusters (MI distances, Ward linkage)")

    # B.4 ------------------------------------------------------------------
    subheading(doc, "B.4: Stability of the selected features")
    pre = b4[b4["variant"].str.startswith("Pre-evaluation")].iloc[0]
    fix = b4[b4["variant"].str.startswith("All 344")].iloc[0]
    others = b4[~b4["variant"].str.startswith("All 344")]
    changed = 32 - others["kept_at_32"]
    covered = others["kept_at_32"] + others["same_cluster_substitute_at_32"]
    paragraph(doc, "Table B.5 repeats the selection under four alternative choices: correcting the omission of "
                   "the final feature column; 10 and 30 quantile bins instead of 20; and estimating the MI "
                   f"distances on the {S['pre_eval_weeks']} weeks before the evaluation block only (to "
                   f"{fmt_date(S['pre_eval_end'])}), which removes evaluation-period information from the selection. "
                   "Changing the number of bins or the sample also changes the scale of the linkage heights, so "
                   "each alternative is compared with the baseline at the same number of clusters (32). A "
                   "representative is counted as substituted when the alternative selection replaces it with "
                   "another member of its baseline cluster. The adjusted Rand index (ARI) measures agreement "
                   "between the baseline and alternative cluster memberships (one for identical partitions, zero "
                   "for chance agreement).")
    caption(doc, "Table B.5: Representatives retained under alternative estimation choices (32 clusters)")
    table(doc, ["Alternative", "Clusters at cut height 2", "Unchanged", "Substituted from same cluster",
                "Not represented", "ARI"],
          [[r.variant, r.clusters_at_height_2, r.kept_at_32, r.same_cluster_substitute_at_32,
            32 - r.kept_at_32 - r.same_cluster_substitute_at_32, f"{r.ari_at_32:.2f}"] for r in b4.itertuples()],
          [W(2.3), W(0.8), W(0.75), W(0.95), W(0.8), W(0.5)], font=8, align_right=(1, 2, 3, 4, 5))
    paragraph(doc, f"Correcting the omitted column changes one representative: {fix.dropped_at_32} is replaced by "
                   f"{fix.added_at_32}, a near tie in the largest cluster. The other choices change between "
                   f"{changed.min()} and {changed.max()} of the 32 representatives. Most replacements come from the "
                   f"same baseline cluster, so between {covered.min()} and {covered.max()} of the 32 baseline "
                   "clusters remain represented. The specific features selected are therefore sensitive to "
                   "reasonable estimation choices, while the groups of information they represent are largely "
                   f"stable. Restricting the estimation to pre-evaluation data changes {32 - pre.kept_at_32} "
                   f"representatives, {pre.same_cluster_substitute_at_32} of which are replaced from the same "
                   "cluster. The full-sample estimation disclosed in Section 3.6 therefore influenced which "
                   "specific features were selected. Whether this affects out-of-sample performance could only be "
                   "established by re-estimating the models with the pre-evaluation selection. Table B.6 lists the "
                   "changes. With 10 bins, one cluster of constant Lambda features is represented by one of them "
                   "(SA_NB_Lambda), which illustrates why constant features should be removed before clustering.")
    caption(doc, "Table B.6: Representatives removed and added under each alternative (32 clusters)")
    table(doc, ["Alternative", "Removed from the selection", "Added to the selection"],
          [[r.variant, r.dropped_at_32.replace(";", ";"), r.added_at_32] for r in b4.itertuples()],
          [W(1.5), W(2.35), W(2.35)], font=7)

    # B.5 ------------------------------------------------------------------
    subheading(doc, "B.5: Mutual information compared with correlation")
    paragraph(doc, "Section 3.6 motivates MI by its ability to capture nonlinear dependence that Pearson "
                   "correlation misses. To test whether the choice of dependence measure matters, the clustering "
                   "is repeated with the distance 1 − |r|, using the same sample, Ward linkage and number of "
                   f"clusters. Only {S['corr_kept']} of the 32 MI representatives are also selected under "
                   f"correlation, a further {S['corr_same_cluster_substitutes']} are replaced by a member of the "
                   f"same MI cluster, and the ARI is {S['corr_ari']:.2f}. The choice of dependence measure therefore "
                   "changes the selection materially.")
    paragraph(doc, f"Figure B.2 compares the two measures across all {S['pairs']:,} feature pairs. They are related "
                   f"(Spearman rank correlation {S['spearman_nmi_abs_r']:.2f}) but diverge in two systematic ways. "
                   f"First, the {S['hi_mi_low_r_pairs']} pairs with high MI (NMI ≥ 0.4) and low correlation "
                   "(|r| < 0.2) all involve slow-moving series, with a median week-to-week autocorrelation of "
                   f"{S['hi_mi_low_r_median_autocorr']:.2f}. When MI is recomputed on weekly changes, their median "
                   f"NMI falls from {S['hi_mi_low_r_median_nmi_levels']:.2f} to "
                   f"{S['hi_mi_low_r_median_nmi_changes']:.2f} (Table B.7). Their high MI reflects series that "
                   "occupy similar ranges during the same periods, rather than a stable nonlinear relationship. "
                   f"Second, the {S['strong_r_pairs']} strongly correlated pairs (|r| ≥ 0.8) have a median NMI "
                   f"of {S['strong_r_median_nmi']:.2f}, and {S['strong_r_nmi_below_0_3']} of them fall below 0.3: "
                   "with 20 bins, normalising by entropy places strong linear dependence well below the top of the "
                   "NMI scale, so pairs that correlation treats as largely redundant can remain far apart in MI "
                   "distance.")
    figure(doc, DATA / "figure_b2_mi_vs_correlation.png")
    caption(doc, f"Figure B.2: Normalised MI against absolute Pearson correlation for all {S['pairs']:,} pairs "
                 "of clustered features")
    caption(doc, "Table B.7: Feature pairs with high MI but low correlation")
    table(doc, ["Feature pair", "NMI (levels)", "NMI (weekly changes)", "|r| (levels)", "Lag-1 autocorrelation"],
          [[f"{r.feature_a} – {r.feature_b}", f"{r.nmi:.2f}", f"{r.nmi_weekly_changes:.2f}",
            f"{r.abs_pearson:.2f}", f"{r.autocorr_a:.2f} / {r.autocorr_b:.2f}"] for r in b5.itertuples()],
          [W(2.9), W(0.75), W(0.85), W(0.7), W(1.0)], font=7.5, align_right=(1, 2, 3, 4))
    paragraph(doc, "MI-based and correlation-based clustering therefore produce materially different selections in "
                   "this dataset, but the evidence does not show that the difference arises from nonlinear "
                   "redundancy. Much of the dependence that MI detects and correlation does not is shared "
                   "persistence between slow-moving series.")

    # B.6 ------------------------------------------------------------------
    subheading(doc, "B.6: Economic classification of the modelling features")
    assert len(categories) == 50 and set(b1.representative) <= set(categories), "Table B.8 does not match"
    paragraph(doc, "Table B.8 lists the 50 modelling features, the 32 cluster representatives and the 18 "
                   "relative-factor features added by design, by the categories of Table 2, with the economic "
                   "rationale for including each and the empirical evidence that supports it. Appendix A defines "
                   "each feature.", highlight=True)
    caption(doc, "Table B.8: Economic classification of the 50 modelling features")
    doc.element.body.find(qn("w:sectPr")).addprevious(copy.deepcopy(table_b8))
    doc.add_paragraph(style="Normal").paragraph_format.space_after = Pt(2)

    # B.7 ------------------------------------------------------------------
    subheading(doc, "B.7: Summary")
    paragraph(doc, "The documented procedure reproduces the selected features exactly. The selection is sensitive "
                   "to the bin count, the estimation sample and the dependence measure: about a third of the "
                   "representatives change under each reasonable alternative, although most are replaced by "
                   "features from the same cluster. The clustering is weakly structured, and part of the MI it "
                   "relies on reflects shared persistence rather than nonlinear dependence. The candidate universe "
                   "also contains ten constant features, which leave one nearly uninformative representative "
                   "(PPP.ZA.Index_QS) in the predictor set. The selected features "
                   "should therefore be read as one reasonable, non-redundant subset of the candidate universe, not "
                   "as a uniquely identified set of informative predictors.")



if __name__ == "__main__":
    sys.exit("Appendix B is built with Appendices A and C by build_appendices.py.")
