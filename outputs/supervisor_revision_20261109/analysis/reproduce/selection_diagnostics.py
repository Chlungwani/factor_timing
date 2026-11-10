import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score,silhouette_score,adjusted_rand_score
TASK=Path(__file__).resolve().parent
OUT=TASK.parent
data=np.load(OUT/'clustering_arrays.npz')
X=data['values']; features=data['features']; Z=data['Z']; baseline=data['labels']
D=pd.read_csv(OUT/'mi_distance_matrix.csv',index_col=0)
def representatives(distance,labels):
    result=[]
    for c in sorted(set(labels)):
        members=features[labels==c]
        result.append(distance.loc[members,members].mean(axis=1).idxmin())
    return result
base=set(representatives(D,baseline))
rows=[]
for h in [1.5,1.75,2,2.25,2.5]:
    lab=fcluster(Z,h,criterion='distance')
    reps=set(representatives(D,lab))
    score=silhouette_score(D.to_numpy(),lab,metric='precomputed')
    rows.append({'choice':'Ward cut height','value':h,'clusters':len(reps),'representatives_in_common':len(base&reps),'adjusted_rand':float(adjusted_rand_score(baseline,lab)),'silhouette':float(score)})
for n in [10,30]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); bins=KBinsDiscretizer(n_bins=n,encode='ordinal',strategy='quantile',subsample=None).fit_transform(X)
    H=[mutual_info_score(bins[:,i],bins[:,i]) for i in range(len(features))]
    N=np.eye(len(features))
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            N[i,j]=N[j,i]=mutual_info_score(bins[:,i],bins[:,j])/max(H[i],H[j])
    new=pd.DataFrame(1-N,index=features,columns=features)
    lab=fcluster(linkage(new.to_numpy(),method='ward'),2,criterion='distance')
    reps=set(representatives(new,lab))
    rows.append({'choice':'Quantile bins','value':n,'clusters':len(reps),'representatives_in_common':len(base&reps),'adjusted_rand':float(adjusted_rand_score(baseline,lab)),'silhouette':float(silhouette_score(new.to_numpy(),lab,metric='precomputed'))})
    print('Completed bins',n,flush=True)
pd.DataFrame(rows).to_csv(OUT/'selection_sensitivity.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False),flush=True)
