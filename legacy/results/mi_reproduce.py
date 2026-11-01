"""Reproduce the MI clustering in feature_selection.ipynb with precomputed bins."""
import warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore")
DATA = "C:/Users/ccwis/OneDrive/Documents/1. MEng - Data Science/Project/factor_timing/legacy/data_input/"
data = pd.read_csv(DATA + "featureset.csv", parse_dates=["Date"], na_values=["NA"]).set_index("Date")
miss = data.isnull().mean()
data = data.drop(columns=miss[miss > 0.30].index)
print("after screen", data.shape[1], "last column (excluded as y):", data.columns[-1])
data = data.ffill().dropna()
X = data.iloc[:, :-1]
print("MI sample", X.shape, X.index.min().date(), X.index.max().date())

bins = np.column_stack([KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="quantile")
                        .fit_transform(X.iloc[:, [j]].values)[:, 0] for j in range(X.shape[1])])
n = X.shape[1]
H = np.array([mutual_info_score(bins[:, j], bins[:, j]) for j in range(n)])
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        denom = max(H[i], H[j])
        nmi = 0 if denom == 0 else mutual_info_score(bins[:, i], bins[:, j]) / denom
        D[i, j] = D[j, i] = 1 - nmi
D = pd.DataFrame(D, index=X.columns, columns=X.columns)
Z = linkage(D, method="ward")
labels = pd.Series(fcluster(Z, t=2, criterion="distance"), index=D.index)
print("clusters at height 2:", labels.nunique())
reps = []
for c in sorted(labels.unique()):
    f = labels[labels == c].index
    reps.append(D.loc[f, f].mean(axis=1).idxmin())
saved = set(pd.read_csv(DATA + "representatives.csv")["Representative"])
print("representatives reproduced:", len(set(reps) & saved), "of", len(saved))
sizes = labels.value_counts()
print("cluster sizes min/median/max:", sizes.min(), sizes.median(), sizes.max(), "singletons:", (sizes == 1).sum())
print("median off-diagonal distance:", round(float(np.median(D.values[np.triu_indices(n, 1)])), 3))
sil = {k: silhouette_score(D, fcluster(Z, t=k, criterion="maxclust"), metric="precomputed") for k in [5, 10, 20, 32, 50, 75, 100]}
print("silhouette:", {k: round(v, 3) for k, v in sil.items()})
for h in [1.5, 2, 2.5, 3]:
    print("height", h, "->", len(set(fcluster(Z, t=h, criterion="distance"))), "clusters")
