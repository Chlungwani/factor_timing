"""Appendix B analysis: mutual-information feature selection, reported and stress-tested.

Reproduces the clustering in feature_importance/feature_selection.ipynb exactly
(including its omission of the last feature column), then writes:

  appendix_b/b1_clusters.csv           cluster membership and representatives
  appendix_b/b2_screening.csv          features removed by the 30% missing-value screen
  appendix_b/b3_cut_heights.csv        clusters and silhouette by cut height
  appendix_b/b3_silhouette.csv         silhouette score for 5-100 clusters
  appendix_b/b4_stability.csv          representative overlap under alternative choices
  appendix_b/b5_mi_vs_correlation.csv  NMI and |Pearson r| for every feature pair
  appendix_b/figure_b1_silhouette.png
  appendix_b/figure_b2_mi_vs_correlation.png
  appendix_b/summary.json              numbers quoted in the appendix text
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, mutual_info_score, silhouette_score
from sklearn.preprocessing import KBinsDiscretizer

from thesis_style import BLUE, GRID, INK, INK_2, ORANGE, RC

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent
DATA = RESULTS.parent / "data_input"
OUT = RESULTS / "appendix_b"
EVALUATION_START = "2022-01-31"
CUT_HEIGHT, N_BINS = 2, 20

plt.rcParams.update({k: v for k, v in RC.items() if not k.startswith(("axes.title", "legend."))})


# ----------------------------------------------------------------- pipeline --

def screened_features():
    raw = pd.read_csv(DATA / "featureset.csv", parse_dates=["Date"], na_values=["NA"]).set_index("Date")
    missing = raw.isnull().mean()
    removed = missing[missing > 0.30].sort_values(ascending=False)
    data = raw.drop(columns=removed.index).ffill().dropna()
    return raw, removed, data


def bins_for(X, n_bins):
    return np.column_stack([KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
                            .fit_transform(X.iloc[:, [j]].values)[:, 0] for j in range(X.shape[1])])


def nmi_matrix(X, n_bins=N_BINS):
    b = bins_for(X, n_bins)
    n = X.shape[1]
    H = np.array([mutual_info_score(b[:, j], b[:, j]) for j in range(n)])
    N = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            denom = max(H[i], H[j])
            N[i, j] = N[j, i] = 0 if denom == 0 else mutual_info_score(b[:, i], b[:, j]) / denom
    return pd.DataFrame(N, index=X.columns, columns=X.columns)


def cluster(D, height=None, k=None):
    Z = linkage(D, method="ward")            # rows of the square matrix, as in the notebook
    labels = fcluster(Z, t=height, criterion="distance") if k is None else fcluster(Z, t=k, criterion="maxclust")
    labels = pd.Series(labels, index=D.index)
    reps = {c: D.loc[f, f].mean(axis=1).idxmin()
            for c in sorted(labels.unique()) for f in [labels[labels == c].index]}
    return Z, labels, reps


def compare(base_labels, base_reps, labels, reps):
    common = base_labels.index.intersection(labels.index)
    kept = set(base_reps.values()) & set(reps.values())
    return {"clusters": len(reps), "representatives_kept": len(kept),
            "ari": round(adjusted_rand_score(base_labels[common], labels[common]), 3),
            "dropped": sorted(set(base_reps.values()) - set(reps.values())),
            "added": sorted(set(reps.values()) - set(base_reps.values()))}


# --------------------------------------------------------------------- main --

def main():
    OUT.mkdir(exist_ok=True)
    raw, removed, data = screened_features()
    X = data.iloc[:, :-1]                     # notebook behaviour: last column treated as a target
    omitted = data.columns[-1]
    summary = {"candidates": raw.shape[1], "removed_missing": len(removed), "screened": data.shape[1],
               "clustered": X.shape[1], "omitted_feature": omitted, "mi_weeks": len(X),
               "mi_start": str(X.index.min().date()), "mi_end": str(X.index.max().date())}

    # B.2 screening
    pd.DataFrame({"feature": removed.index, "missing_share": removed.round(3).values,
                  "first_available": [raw[c].first_valid_index().date() for c in removed.index]}
                 ).to_csv(OUT / "b2_screening.csv", index=False)

    # baseline (reproduces the saved selection)
    nmi_full = nmi_matrix(data)
    D = 1 - nmi_full.loc[X.columns, X.columns]
    Z, labels, reps = cluster(D, height=CUT_HEIGHT)
    saved = pd.read_csv(DATA / "representatives.csv")["Representative"]
    assert set(reps.values()) == set(saved), "baseline does not reproduce representatives.csv"
    summary["baseline_clusters"] = len(reps)
    off = D.values[np.triu_indices(len(D), 1)]
    summary["median_distance"] = round(float(np.median(off)), 3)

    # B.1 clusters
    rows = []
    for c, rep in reps.items():
        members = labels[labels == c].index
        mean_d = D.loc[members, members].mean(axis=1)
        others = mean_d.drop(rep).sort_values().index.tolist()
        rows.append({"cluster": c, "representative": rep, "size": len(members),
                     "rep_mean_distance": round(mean_d[rep], 3),
                     "within_mean_distance": round(float(D.loc[members, members].values[
                         np.triu_indices(len(members), 1)].mean()), 3),
                     "other_members": "; ".join(others)})
    b1 = pd.DataFrame(rows).sort_values("size", ascending=False)
    b1.to_csv(OUT / "b1_clusters.csv", index=False)
    summary["cluster_size_min"], summary["cluster_size_max"] = int(b1["size"].min()), int(b1["size"].max())
    summary["cluster_size_median"] = float(b1["size"].median())

    # B.3 cut height and silhouette
    sil = {k: silhouette_score(D, fcluster(Z, t=k, criterion="maxclust"), metric="precomputed")
           for k in range(5, 101)}
    pd.Series(sil, name="silhouette").rename_axis("clusters").to_csv(OUT / "b3_silhouette.csv")
    cut_rows = []
    for h in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        lab = fcluster(Z, t=h, criterion="distance")
        cut_rows.append({"cut_height": h, "clusters": len(set(lab)),
                         "silhouette": round(silhouette_score(D, lab, metric="precomputed"), 3)})
    pd.DataFrame(cut_rows).to_csv(OUT / "b3_cut_heights.csv", index=False)
    summary["silhouette_at_32"] = round(sil[32], 3)
    summary["silhouette_max"] = round(max(sil.values()), 3)
    summary["silhouette_argmax"] = max(sil, key=sil.get)

    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.plot(list(sil), list(sil.values()), color=BLUE, linewidth=1.6)
    ax.scatter([32], [sil[32]], color=ORANGE, s=28, zorder=3)
    ax.annotate(f"32 clusters (cut height 2): {sil[32]:.3f}", (32, sil[32]), xytext=(38, sil[32] - 0.03),
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.6))
    ax.axhline(0, color=INK_2, linewidth=0.6)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Mean silhouette score")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "figure_b1_silhouette.png", dpi=300)
    plt.close(fig)

    # B.4 stability
    variants = {}
    _, lab_v, rep_v = cluster(1 - nmi_full, height=CUT_HEIGHT)
    variants["All 344 screened features (omission corrected)"] = (lab_v, rep_v, 1 - nmi_full)
    for nb in (10, 30):
        Dv = 1 - nmi_matrix(X, n_bins=nb)
        variants[f"{nb} quantile bins instead of 20"] = (*cluster(Dv, height=CUT_HEIGHT)[1:], Dv)
    pre = X[X.index < EVALUATION_START]
    summary["pre_eval_weeks"], summary["pre_eval_end"] = len(pre), str(pre.index.max().date())
    Dp = 1 - nmi_matrix(pre)
    variants[f"Pre-evaluation weeks only ({pre.index.min():%b %Y} to {pre.index.max():%b %Y})"] = (
        *cluster(Dp, height=CUT_HEIGHT)[1:], Dp)
    def substitutes(rep_k):
        """Dropped baseline representatives replaced by a new representative from the same baseline cluster."""
        new = set(rep_k.values()) - set(reps.values())
        dropped = set(reps.values()) - set(rep_k.values())
        return {d for d in dropped if any(labels.get(n) == labels[d] for n in new)}

    stab = []
    for name, (lab_v, rep_v, Dv) in variants.items():
        at_h = compare(labels, reps, lab_v, rep_v)
        _, lab_k, rep_k = cluster(Dv, k=len(reps))
        at_k = compare(labels, reps, lab_k, rep_k)
        stab.append({"variant": name, "clusters_at_height_2": at_h["clusters"],
                     "kept_at_32": at_k["representatives_kept"], "same_cluster_substitute_at_32": len(substitutes(rep_k)),
                     "ari_at_32": at_k["ari"],
                     "dropped_at_32": "; ".join(at_k["dropped"]), "added_at_32": "; ".join(at_k["added"])})
    pd.DataFrame(stab).to_csv(OUT / "b4_stability.csv", index=False)
    summary["baseline_within_cluster_median_distance"] = round(float(b1["within_mean_distance"].median()), 3)

    # B.5 mutual information vs correlation
    constant = X.columns[X.nunique() == 1].tolist()   # std alone misses floating-point noise
    summary["constant_features"] = constant
    ppp = pd.read_csv(DATA / "lagged_data.csv", parse_dates=["Date"]).set_index("Date")["PPP.ZA.Index_QS"]
    summary["ppp_values"] = sorted(ppp.unique().round(2).tolist())
    summary["ppp_first_change"] = str(ppp[ppp != ppp.iloc[0]].index.min().date())
    r = X.corr().abs().fillna(0)
    r.loc[constant, :] = 0                       # constant series: uncorrelated, as NMI treats them
    r.loc[:, constant] = 0
    np.fill_diagonal(r.values, 1)
    _, lab_c, rep_c = cluster(1 - r, k=len(reps))
    corr_cmp = compare(labels, reps, lab_c, rep_c)
    summary["corr_kept"], summary["corr_ari"] = corr_cmp["representatives_kept"], corr_cmp["ari"]
    summary["corr_same_cluster_substitutes"] = len(substitutes(rep_c))
    nmi = nmi_full.loc[X.columns, X.columns]
    iu = np.triu_indices(len(X.columns), 1)
    pairs = pd.DataFrame({"feature_a": X.columns[iu[0]], "feature_b": X.columns[iu[1]],
                          "nmi": nmi.values[iu].round(4), "abs_pearson": r.values[iu].round(4)})
    summary["pairs"] = len(pairs)
    summary["spearman_nmi_abs_r"] = round(float(pairs["nmi"].corr(pairs["abs_pearson"], method="spearman")), 3)

    # Where the two measures disagree: persistence on one side, NMI compression on the other.
    autocorr = X.apply(lambda s: s.autocorr(1))
    hi_mi = pairs[(pairs["nmi"] >= 0.4) & (pairs["abs_pearson"] < 0.2)].copy()
    changes = X.diff().dropna()
    nmi_changes = nmi_matrix(changes[sorted(set(hi_mi["feature_a"]) | set(hi_mi["feature_b"]))])
    hi_mi["nmi_weekly_changes"] = [round(nmi_changes.loc[a_, b_], 3) for a_, b_ in zip(hi_mi["feature_a"], hi_mi["feature_b"])]
    hi_mi["autocorr_a"] = autocorr[hi_mi["feature_a"]].round(3).values
    hi_mi["autocorr_b"] = autocorr[hi_mi["feature_b"]].round(3).values
    hi_mi.sort_values("nmi", ascending=False).to_csv(OUT / "b5_high_mi_low_correlation.csv", index=False)
    summary["hi_mi_low_r_pairs"] = len(hi_mi)
    summary["hi_mi_low_r_median_autocorr"] = round(float(np.minimum(hi_mi["autocorr_a"], hi_mi["autocorr_b"]).median()), 3)
    summary["hi_mi_low_r_median_nmi_levels"] = round(float(hi_mi["nmi"].median()), 3)
    summary["hi_mi_low_r_median_nmi_changes"] = round(float(hi_mi["nmi_weekly_changes"].median()), 3)
    strong_r = pairs[pairs["abs_pearson"] >= 0.8]
    summary["strong_r_pairs"] = len(strong_r)
    summary["strong_r_median_nmi"] = round(float(strong_r["nmi"].median()), 3)
    summary["strong_r_nmi_below_0_3"] = int((strong_r["nmi"] < 0.3).sum())
    pairs.to_csv(OUT / "b5_mi_vs_correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ax.scatter(pairs["abs_pearson"], pairs["nmi"], s=2, color=BLUE, alpha=0.18, linewidths=0)
    ax.set_xlabel("Absolute Pearson correlation")
    ax.set_ylabel("Normalised mutual information")
    ax.set_xlim(0, 1)
    ax.grid(color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "figure_b2_mi_vs_correlation.png", dpi=300)
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(cut_rows).to_string(index=False))
    print(pd.DataFrame(stab).drop(columns=["dropped_at_32", "added_at_32"]).to_string(index=False))
    for s in stab:
        print(s["variant"], "| dropped:", s["dropped_at_32"], "| added:", s["added_at_32"])
    print("correlation clustering:", corr_cmp)


if __name__ == "__main__":
    main()
