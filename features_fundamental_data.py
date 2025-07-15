import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Load data
df = pd.read_excel('FeatureSheetWeekly.xlsx')  # Replace with your file name

# Define features
features = [
    "RepSouCDSCurve_QS_PC1", "RepSouCDSCurve_QS_PC2", "RepSouCDSCurve_QS_PC3",
    "RepSouCDSCurve_PC1", "RepSouCDSCurve_PC2", "RepSouCDSCurve_PC3",
    "REPSOU_5YR_CDS", "US_NB_Level", "US_NB_Slope", "US_NB_Curvature", "US_NB_Lambda", "US_NB_R2",
    "US_NB_Level_QS", "US_NB_Slope_QS", "US_NB_Curvature_QS", "US_NB_Lambda_QS", "US_NB_R2_QS",
    "US_RB_Level", "US_RB_Slope", "US_RB_QS_Level", "US_RB_QS_Slope",
    "SA_NB_Level", "SA_NB_Slope", "SA_NB_Curvature", "SA_NB_Lambda", "SA_NB_R2",
    "SA_NB_Level_QS", "SA_NB_Slope_QS", "SA_NB_Curvature_QS", "SA_NB_Lambda_QS", "SA_NB_R2_QS",
    "SA_RB_Level", "SA_RB_Slope", "SA_RB_Curvature", "SA_RB_Lambda", "SA_RB_R2",
    "SA_RB_Level_QS", "SA_RB_Slope_QS", "SA_RB_Curvature_QS", "SA_RB_Lambda_QS", "SA_RB_R2_QS",
    "SAPMI_YoY", "SACWC_YoY", "OEZAA013_YoY",
    "CESIUSD.Index", "CESIEM.Index", "CESIGL.Index", "EcoSrpCITI_PC1", "EcoSrpCITI_PC2",
    "InflGrowthSurpPC1", "InflGrowthSurpPC2", "InflGrowthSurpPC1_QS", "InflGrowthSurpPC2_QS",
    "BCMPEBLS.Index", "LEI.TOTL.Index", "CONCCONF.Index", "SSICCONF.Index", "TINSALNL.Index", "OPCVTPCR.Index",
    "MichgnConcIndx_QuantileScale", "MichgnConcIndx_TS", "MichgnConcIndx",
    "RandPPP_QuantileScale", "RandPPP_Factor_Inst_TS_ST", "RandPPP_Factor_Inst_TS_LT", "RandPPP_Factor_Inst"
]

# Clean data
df = df[features].dropna()
df = df.loc[:, df.nunique() > 1]
corr = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Set threshold and find columns to drop
threshold = 0.75  # <<-- you can change this value
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop highly correlated features
df_filtered = df.drop(columns=to_drop)

# Inform user of dropped features
print("Features dropped due to correlation threshold > {:.2f}:".format(threshold))
for f in to_drop:
    print(f)

# Correlation matrix of filtered features
corr_filtered = df_filtered.corr().abs()

# Distance matrix and linkage
distance_matrix = 1 - corr_filtered
condensed_dist = squareform(distance_matrix.values, checks=False)
Z = linkage(condensed_dist, method='ward')

# Plot clustermap
sns.clustermap(corr_filtered, row_linkage=Z, col_linkage=Z, cmap="coolwarm", figsize=(18, 18))
plt.suptitle("Correlation Clustermap After Filtering", y=1.02, fontsize=16)
plt.show()

# Auto-select one feature per cluster
max_clusters = 10
cluster_labels = fcluster(Z, max_clusters, criterion='maxclust')

# Map features to clusters
feature_cluster_map = dict(zip(corr_filtered.columns, cluster_labels))

# Select representative feature per cluster
selected_features = []
for cluster in np.unique(cluster_labels):
    members = [f for f, c in feature_cluster_map.items() if c == cluster]
    if len(members) == 1:
        selected_features.append(members[0])
    else:
        sub_corr = corr_filtered.loc[members, members]
        avg_corr = sub_corr.mean().sort_values(ascending=False)
        selected_features.append(avg_corr.index[0])

# Output selected features
print("\nSelected features per cluster:")
for f in selected_features:
    print(f)

# Save selected features
features_economic_data = df_filtered[selected_features]
features_economic_data.to_csv("featureseconomic_data.csv", index=False)
