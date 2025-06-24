import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel("FeatureSheetWeekly.xlsx", index_col="Date")
X=df.dropna()
# Ensure date column is properly parsed
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.02)  # Adjust based on inspection
X_filtered = selector.fit_transform(X)

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_filtered = X.drop(columns=to_drop)

import xgboost as xgb
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)
importance = pd.DataFrame({
    'feature': X.columns,
    'gain': model.feature_importances_
}).sort_values('gain', ascending=False)

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'shap': np.abs(shap_values).mean(0)
}).sort_values('shap', ascending=False)

from gudhi import RipsComplex
from sklearn.preprocessing import MinMaxScaler

# Normalize data
X_normalized = MinMaxScaler().fit_transform(X)

# Generate persistence diagrams
rips = RipsComplex(points=X_normalized, max_edge_length=1)
persistence = rips.persistence()

# Analyze topological features (0D=connected components, 1D=loops)
# Features contributing to persistent structures may be more important

from kmapper import KeplerMapper

mapper = KeplerMapper()
projection = PCA(n_components=2).fit_transform(X_normalized)
graph = mapper.map(projection, X_normalized, clusterer=DBSCAN())

# Visualize with KeplerMapper's interactive plot
mapper.visualize(graph, path_html="feature_mapper.html")

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    LassoCV(cv=5, max_iter=10000),
    threshold="1.25*median"  # Tune this
).fit(X, y)
selected_features = X.columns[selector.get_support()]