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
    "US_NB_Level_TS", "US_NB_Slope_TS", "US_NB_Curvature_TS", "US_RB_Level_TS", "US_RB_Slope_TS",
    "SA_NB_Level_TS", "SA_NB_Slope_TS", "SA_NB_Curvature_TS", "SA_RB_Level_TS", "SA_RB_Slope_TS",
    "SA_RB_Curvature_TS", "VIXUSDZARVolSpread", "SaviUSDZARVolSpread", "USDEURVolTermStruct_PC1",
    "USDEURVolTermStruct_PC2", "USDZARVolTermStruct_PC1", "USDZARVolTermStruct_PC2",
    "USDZAR_EntDyn", "USDZAR_Ent", "MultiAssetEnt", "LocalBonds_EntDyn", "LocalBonds_Ent",
    "GlobalBonds_EntDyn", "GlobalBonds_Ent", "LocalEquity_EntDyn", "LocalEquity_Ent",
    "GlobalEquity_EntDyn", "GlobalEquity_Ent", "LocalEquityTS", "LocalEquityGret1",
    "LocalEquityGret3", "LocalEquityGret6", "LocalEquityGret12", "LocalEquityVol1",
    "LocalEquityVol3", "LocalEquityVol6", "LocalEquityVol12", "LocalEquitySB", "LocalBondsTS",
    "LocalBondsGret1", "LocalBondsGret3", "LocalBondsGret6", "LocalBondsGret12", "LocalBondsVol1",
    "LocalBondsVol3", "LocalBondsVol6", "LocalBondsVol12", "LocalBondsSB", "LocalPropertyTS",
    "LocalPropertyGret1", "LocalPropertyGret3", "LocalPropertyGret6", "LocalPropertyGret12",
    "LocalPropertyVol1", "LocalPropertyVol3", "LocalPropertyVol6", "LocalPropertyVol12",
    "LocalPropertySB", "GlobalEquityTS", "GlobalEquityGret1", "GlobalEquityGret3",
    "GlobalEquityGret6", "GlobalEquityGret12", "GlobalEquityVol1", "GlobalEquityVol3",
    "GlobalEquityVol6", "GlobalEquityVol12", "GlobalEquitySB", "GlobalBondTS", "GlobalBondGret1",
    "GlobalBondGret3", "GlobalBondGret6", "GlobalBondGret12", "GlobalBondVol1", "GlobalBondVol3",
    "GlobalBondVol6", "GlobalBondVol12", "GlobalBondSB", "GlobalPropertyTS", "GlobalPropertyGret1",
    "GlobalPropertyGret3", "GlobalPropertyGret6", "GlobalPropertyGret12", "GlobalPropertyVol1",
    "GlobalPropertyVol3", "GlobalPropertyVol6", "GlobalPropertyVol12", "GlobalBondSB.1",
    "USDZARTS", "USDZARGret1", "USDZARGret3", "USDZARGret6", "USDZARGret12", "USDZARVol1",
    "USDZARVol3", "USDZARVol6", "USDZARVol12", "USDZARSB", "LE_LB_Corr", "GE_GB_Corr",
    "GE_LE_Corr", "GB_LB_Corr", "LagRank_J803TR.Index", "LagRank_JALSHTR.Index",
    "LagRank_ALBTR.Index", "LagRank_STEFI.Index", "LagRank_EPRA.Index", "LagRank_SPXT.Index",
    "LagRank_W0G1.Index", "LagRank_G0O1.Index", "USNomBndStructBreak_Level",
    "USNomBndStructBreak_Slope", "USRealBndStructBreak_Level", "USRealBndStructBreak_Slope",
    "SANomBndStructBreak_Level", "SANomBndStructBreak_Slope", "RepSouCDSStructBreak",
    "SPXIVSructBreak_PC1", "SPXIVSructBreak_PC2", "MOVEStructBreak", "SwaptionVolStructBreak_PC1",
    "SwaptionVolStructBreak_PC2", "IRSwapImpVolDecomp_PC1", "IRSwapImpVolDecomp_PC2",
    "IRSwapImpVolDecompTS_PC1", "IRSwapImpVolDecompTS_PC2", "VIX.Index", "VIX3M.Index",
    "VIX6M.Index", "VIX1Y.Index", "VolTermStructPC1", "VolTermStructPC2", "VolTermStructPC1_QS",
    "VolTermStructPC2_QS", "VolSpread1M", "VolSpread6M", "OVX.Index", "GVZ.Index", "SKEW",
    "SkewVixSelf_PC1", "SkewVixSelf_PC2", "VixSaviSpread", "VixMoveSpread", "GLD_FracDiff",
    "GLD_StructBreak", "GLD_LempZiev", "GLD_RollSD", "GLD_RollSkew", "GLD_RollKurt",
    "CL1.Comdty_Diff_QS", "NG1.Comdty_Diff_QS", "HG1.Comdty_Diff_QS", "LA1.Comdty_Diff_QS",
    "XAU.Comdty_Diff_QS", "XAG.Comdty_Diff_QS", "GenComIndexPC1", "GenComIndexPC2",
    "GenComIndexPC1_QS", "GenComIndexPC2_QS", "CL1FracDiff", "CL1StructBreak", "CL1LempZiev",
    "CL1RollSD", "CL1RollSkew", "CL1RollKurt", "XAUFracDiff", "XAUStructBreak", "XAULempZiev",
    "XAURollSD", "XAURollSkew", "XAURollKurt", "USDZAR_SwapBasisSpreadPC1",
    "USDZAR_SwapBasisSpreadPC2", "ZARForwardPointsPC1", "ZARForwardPointsPC2", "ZARBasisIY12M",
    "USDZAR_SwapBasisSpreadPC1_QS", "USDZAR_SwapBasisSpreadPC2_QS", "ZARForwardPointsPC1_QS",
    "ZARForwardPointsPC2_QS", "EM_CurrencyBask_RV_12", "EM_CurrencyBask_RV_3", "DXY_RV_12",
    "DXY_RV_3", "WOG1_Ent_TS", "ACWI_Ent_TS", "JALSH_Ent_TS", "ACWI_RV_3_TS", "ACWI_RV_12_TS",
    "JALSH_RV_3_TS", "JALSH_RV_12_TS", "LEI.Totl_TS", "AAII_Bull_TS", "SA_PMI_TS",
    "SA_US_BEInfl_Sprd_QS", "SA_BE_Infl_QS", "USGGBE10.Index_QS", "SA_US_BEInfl_Sprd",
    "SA_BE_Infl", "ZAR_FwdPts_PC2_TS", "ZAR_FwdPts_PC1_TS", "DXY_QuantScale", "DXY_TS",
    "LB_GB_Corr", "SubIndex_PC1_TS", "SubIndex_PC1", "SubIndex_PC2"
]
# Drop non-numeric or irrelevant columns (e.g., Date)
df = df.select_dtypes(include=[np.number])
df = df.dropna()  # drop rows with NaNs — alternatively, df.fillna(method='ffill')

# Optional: drop constant columns
df = df.loc[:, df.nunique() > 1]

# === Step 2: Compute Absolute Correlation Matrix ===
corr = df.corr().abs()

# === Step 3: Convert Correlation to Distance Matrix ===
distance_matrix = 1 - corr
condensed_distance = squareform(distance_matrix.values, checks=False)

# === Step 4: Hierarchical Clustering ===
Z = linkage(condensed_distance, method='average')  # or 'ward', 'complete', etc.

# Choose number of clusters or distance threshold
max_clusters = 45
clusters = fcluster(Z, t=max_clusters, criterion='maxclust')

# === Step 5: Select One Feature per Cluster ===
clustered_features = pd.DataFrame({
    'feature': corr.columns,
    'cluster': clusters
})

# Select the feature in each cluster with the **highest average correlation** to others in the same cluster
selected_features = []
for cluster_id in clustered_features['cluster'].unique():
    cluster_members = clustered_features[clustered_features['cluster'] == cluster_id]['feature'].values
    if len(cluster_members) == 1:
        selected_features.append(cluster_members[0])
    else:
        sub_corr = corr.loc[cluster_members, cluster_members]
        avg_corr = sub_corr.mean().sort_values(ascending=False)
        selected_features.append(avg_corr.index[0])

# === Output: Filtered DataFrame ===
df_selected = df[selected_features]

# === Save to CSV ===
df_selected.to_csv("features_market_data.csv", index=False)
print(f"Selected {len(selected_features)} features across {max_clusters} clusters.")
