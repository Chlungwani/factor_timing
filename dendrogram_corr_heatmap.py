import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

from sklearn.preprocessing import StandardScaler


# Assuming your data is in a pandas DataFrame called 'data' with 300 columns/features
df = pd.read_csv("fundamental_data.csv") 
X = df.drop(columns=['Date'])  # modify as needed
data = X.dropna()
# Assuming your data is in a pandas DataFrame called 'data' with 300 columns/features
# 1. Check for and handle missing/infinite values
print("Missing values per column before handling:")
print(data.isna().sum())

# Option 1: Drop columns with too many missing values (adjust threshold as needed)
data = data.dropna(axis=1, thresh=0.8*len(data)) 

# 1. Normalize the features
scaler = StandardScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# 1. Calculate correlation matrix
corr = normalized_data.corr()
# 4. Check for NaN or infinite values in correlation matrix
if np.isnan(corr.values).any() or np.isinf(corr.values).any():
    print("Warning: Correlation matrix still contains NaN or infinite values")
    # Replace any remaining NaN/inf values with 0 (no correlation)
    corr = corr.fillna(0)
    corr = corr.replace([np.inf, -np.inf], 0)

# 5. Calculate distance matrix for hierarchical clustering
distance = 1 - np.abs(corr)

# 6. Verify distance matrix contains no NaN/inf values
assert not np.isnan(distance.values).any(), "Distance matrix contains NaN values"
assert not np.isinf(distance.values).any(), "Distance matrix contains infinite values"
# 2. Calculate distance matrix for hierarchical clustering
# We convert correlation to distance: distance = 1 - |correlation|
distance = 1 - np.abs(corr)

# 3. Perform hierarchical clustering
linkage = hierarchy.linkage(distance, method='complete')

# 4. Create the heatmap with dendrogram
plt.figure(figsize=(20, 18))  # Adjust size as needed

# Create a gridspec to add dendrogram on top and side
grid_kws = {"height_ratios": (.15, .85), "hspace": 0.2}
f, (dendro_ax, heatmap_ax) = plt.subplots(2, 1, gridspec_kw=grid_kws)

# Dendrogram on top
dendro = hierarchy.dendrogram(
    linkage,
    ax=dendro_ax,
    orientation='top',
    labels=distance.columns,
    color_threshold=0.7  # Adjust this to change cluster separation
)
dendro_ax.spines['top'].set_visible(False)
dendro_ax.spines['right'].set_visible(False)
dendro_ax.spines['bottom'].set_visible(False)
dendro_ax.spines['left'].set_visible(False)
dendro_ax.tick_params(axis='x', which='both', length=0)
dendro_ax.set_xticks([])  # Remove x-axis labels for dendrogram

# Reorder the data based on the clustering
reordered_idx = dendro['leaves']
reordered_corr = corr.iloc[reordered_idx, reordered_idx]

# Heatmap
heatmap = sns.heatmap(
    reordered_corr,
    ax=heatmap_ax,
    cmap='coolwarm',
    center=0,
    xticklabels=reordered_corr.columns,
    yticklabels=reordered_corr.columns,
    cbar_kws={"shrink": 0.5}
)

# Rotate x-axis labels for better readability
heatmap_ax.set_xticklabels(
    heatmap_ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='center'
)

plt.tight_layout()
plt.show()