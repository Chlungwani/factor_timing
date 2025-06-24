
import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

df = pd.read_excel('FeatureSheetWeekly.xlsx')
df = df.drop(['Date'], axis=1)
X=df.dropna()


# Correlation matrix (absolute value)
corr = np.abs(X.corr())

# Convert to distance matrix
distance_matrix = 1 - corr

# Hierarchical clustering
linkage = sch.linkage(distance_matrix, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
sch.dendrogram(linkage, labels=X.columns, leaf_rotation=90, leaf_font_size=8)
plt.title("Dendrogram of Feature Similarities")
plt.show()
