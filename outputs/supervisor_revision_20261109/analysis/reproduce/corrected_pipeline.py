"""Supervisor revision: causal imputation, pre-evaluation selection, purged tuning.

Run with the existing scientific Python environment. All outputs are isolated.
The target labels and the 156 evaluation dates are retained from the verified run.
"""
import os
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[v]='1'
os.environ['MPLBACKEND']='Agg'
import sys, json, time, warnings, itertools, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import mutual_info_score, roc_auc_score, accuracy_score, log_loss, confusion_matrix, silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

TASK=Path(__file__).resolve().parent
OUT=TASK.parent
ROOT=OUT.parents[2]
DATA=OUT/'inputs'
PAIRS=['Momentum_vs_Value','Momentum_vs_Quality','Value_vs_Quality']
MODELS=['Logistic','GAM','XGBoost']
NTEST=156; GAP=12
EVAL_START=pd.Timestamp('2022-01-31')
SELECT_END=EVAL_START-pd.Timedelta(weeks=GAP+1)

def dump(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(DATA/'featureset.csv',parse_dates=['Date']).set_index('Date')
    prior=raw.loc[:SELECT_END]
    miss=prior.isna().mean()
    retained=miss[miss<=.3].index.tolist()
    # Forward filling never takes an observation from a later date.
    complete=prior[retained].ffill().dropna()
    constants=complete.nunique()[lambda x:x<=1].index.tolist()
    eligible=[c for c in retained if c not in constants]
    X=complete[eligible]
    binner=KBinsDiscretizer(n_bins=20,encode='ordinal',strategy='quantile',subsample=None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bins=binner.fit_transform(X).astype(int)
    H=np.array([mutual_info_score(bins[:,i],bins[:,i]) for i in range(len(eligible))])
    N=np.eye(len(eligible))
    for i in range(len(eligible)):
        for j in range(i+1,len(eligible)):
            N[i,j]=N[j,i]=mutual_info_score(bins[:,i],bins[:,j])/max(H[i],H[j])
    D=pd.DataFrame(1-N,index=eligible,columns=eligible)
    # Preserve the thesis's Ward clustering of distance profiles.
    Z=linkage(D.to_numpy(),method='ward')
    labs=fcluster(Z,t=2,criterion='distance')
    reps=[]; clusters=[]
    for c in sorted(set(labs)):
        members=[eligible[i] for i in np.flatnonzero(labs==c)]
        means=D.loc[members,members].mean(axis=1)
        rep=means.idxmin(); reps.append(rep)
        clusters.append({'cluster':int(c),'representative':rep,'size':len(members),'mean_distance':float(means[rep]),'members':members})
    pd.DataFrame({'Representative':reps}).to_csv(OUT/'representatives.csv',index=False)
    D.to_csv(OUT/'mi_distance_matrix.csv')
    dump(clusters,OUT/'clusters.json')
    # Missingness is assessed before filling. Drop pre-inception rows instead of backfilling.
    rel=pd.read_csv(DATA/'factor_relative_features.csv',parse_dates=['Date']).set_index('Date')
    joined=pd.concat([raw[reps].ffill(),rel],axis=1,join='inner')
    lagged=joined.shift(4).dropna()
    assert np.isfinite(lagged.to_numpy()).all()
    labels=pd.read_csv(DATA/'labeled_relative_factor_returns.csv',parse_dates=['Date']).set_index('Date')
    merged=pd.concat([lagged,labels],axis=1,join='inner').dropna()
    assert merged.index[-NTEST]==EVAL_START
    assert (merged.index[-1]==pd.Timestamp('2025-01-20'))
    assert lagged.index.is_monotonic_increasing and lagged.index.is_unique
    lagged.to_csv(OUT/'lagged_features.csv')
    merged.to_csv(OUT/'model_input.csv')
    removed=[{'feature':c,'pre_selection_missing_share':float(miss[c]),'first_available':str(raw[c].first_valid_index().date())} for c in miss[miss>.3].index]
    dump(removed,OUT/'screened_out.json')
    summary={'selection_cutoff':str(SELECT_END.date()),'selection_start':str(X.index[0].date()),'selection_rows':len(X),'candidates':len(raw.columns),'missing_removed':len(removed),'constant_removed':len(constants),'constants':constants,'eligible':len(eligible),'clusters':len(reps),'relative_features':len(rel.columns),'n_features':len(lagged.columns),'model_start':str(merged.index[0].date()),'model_end':str(merged.index[-1].date()),'n_labelled':len(merged),'initial_training_n':len(merged)-NTEST-GAP,'initial_training_end':str(merged.index[-NTEST-GAP-1].date()),'evaluation_start':str(EVAL_START.date()),'evaluation_end':str(merged.index[-1].date()),'backfilled_cells':0,'feature_names':lagged.columns.tolist(),'representatives':reps}
    dump(summary,OUT/'data_summary.json')
    np.savez_compressed(OUT/'clustering_arrays.npz',Z=Z,bins=bins,labels=labs,values=X.to_numpy(),features=np.array(eligible))
    print(json.dumps(summary,indent=2),flush=True)

def splits(n):
    size=n//5; bounds=[(i*size,(i+1)*size if i<4 else n) for i in range(5)]
    result=[]
    for pair in itertools.combinations(range(5),2):
        test=np.zeros(n,dtype=bool)
        for g in pair:
            a,b=bounds[g]; test[a:b]=True
        train=~test
        for g in pair:
            a,b=bounds[g]; train[max(0,a-GAP):min(n,b+GAP)]=False
        result.append((np.flatnonzero(train),np.flatnonzero(test)))
    return bounds,result

def paths(y,split_probs):
    bounds,_=splits(len(y)); combos=list(itertools.combinations(range(5),2))
    holding=[[i for i,p in enumerate(combos) if g in p] for g in range(5)]
    out=[]
    for j in range(4):
        p=np.full(len(y),np.nan)
        for g,(a,b) in enumerate(bounds):
            p[a:b]=split_probs[holding[g][j]][a:b]
        assert np.isfinite(p).all()
        out.append(float(roc_auc_score(y,p)))
    return out

def grid_for(model):
    if model=='Logistic':
        return list(ParameterGrid({'alpha':[0.,.5,1.],'lambda_reg':[.01,.1,1.,10.,100.,1000.]}))
    if model=='GAM':
        return list(ParameterGrid({'lam':[1,10,100,1000,10000],'n_splines':[4,5,7,10]}))
    return list(ParameterGrid({'learning_rate':[.01,.03,.1],'max_depth':[2,3,4],'min_child_weight':[5,10,20],'reg_lambda':[1,10,100,1000],'subsample':[.6,.8]}))

def fit(model,params,X,y):
    if model=='Logistic':
        m=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=params['alpha'],C=1/params['lambda_reg'],tol=1e-6,max_iter=100000,random_state=0)
    elif model=='GAM':
        from pygam import LogisticGAM,s
        terms=s(0,n_splines=params['n_splines'])
        for j in range(1,X.shape[1]): terms+=s(j,n_splines=params['n_splines'])
        m=LogisticGAM(terms,lam=params['lam'],max_iter=1000,fit_intercept=True)
    else:
        from xgboost import XGBClassifier
        m=XGBClassifier(objective='binary:logistic',eval_metric='logloss',n_estimators=500,n_jobs=1,random_state=0,**params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X,y)
    if model=='Logistic' and np.any(m.n_iter_>=100000):
        raise RuntimeError('Logistic solver did not converge')
    if model=='GAM' and (not m.logs_.get('diffs') or m.logs_['diffs'][-1]>=m.tol):
        raise RuntimeError('GAM solver did not converge')
    return m

def probability(model,m,X):
    return m.predict_proba(X) if model=='GAM' else m.predict_proba(X)[:,1]

def tune(model,X,y,pair,step):
    _,cv=splits(len(y))
    folds=[]
    for tr,te in cv:
        a,b=X[tr],X[te]
        if model!='XGBoost':
            scaler=StandardScaler().fit(a); a=scaler.transform(a); b=scaler.transform(b)
        folds.append((tr,te,a,b))
    grid=grid_for(model); rows=[]; best=None; bestscore=-np.inf
    for i,params in enumerate(grid):
        aucs=[]; predictions=[]; fail=None
        for tr,te,a,b in folds:
            try:
                m=fit(model,params,a,y[tr]); p=probability(model,m,b)
                if not np.isfinite(p).all(): raise ValueError('Nonfinite probabilities')
                aucs.append(float(roc_auc_score(y[te],p)))
                vector=np.full(len(y),np.nan); vector[te]=p; predictions.append(vector)
            except Exception as exc:
                fail=f'{type(exc).__name__}: {exc}'; break
        row={**params,'mean_auc':float(np.mean(aucs)) if fail is None else None,'fold_aucs':aucs,'path_aucs':paths(y,predictions) if fail is None else [],'failed':fail}
        rows.append(row)
        if fail is None and row['mean_auc']>bestscore:
            bestscore=row['mean_auc']; best=params.copy()
        if (i+1)%5==0 or i==len(grid)-1:
            print(f'{pair} {model} tuning {step}: {i+1}/{len(grid)}, best={bestscore:.4f}',flush=True)
    dump(rows,OUT/f'artifacts/{model.lower()}_{pair.lower()}_grid_step{step}.json')
    if best is None: raise RuntimeError('No valid settings')
    return best,bestscore

def shap_exact(model,m,X,names):
    if model=='Logistic':
        phi=(X-X.mean(axis=0))*m.coef_[0]
    elif model=='GAM':
        # An additive model has exact interventional attributions per smooth term.
        contributions=np.column_stack([np.asarray(m.partial_dependence(term=j,X=X)) for j in range(len(names))])
        phi=contributions-contributions.mean(axis=0)
        raw=np.asarray(m._modelmat(X).dot(m.coef_)).reshape(-1)
        reconstruction=raw.mean()+phi.sum(axis=1)
        assert np.max(np.abs(reconstruction-raw))<1e-7
    else:
        import shap
        phi=shap.TreeExplainer(m,model_output='raw').shap_values(X)
    imp=pd.DataFrame({'feature':names,'mean_abs_shap':np.abs(phi).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    return imp

def run_model(model,pair):
    log=OUT/f'logs/{model.lower()}_{pair.lower()}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    import contextlib
    with log.open('w',encoding='utf-8',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f),threadpool_limits(limits=1):
        t0=time.time(); df=pd.read_csv(OUT/'model_input.csv',parse_dates=['Date']).set_index('Date')
        summary=json.loads((OUT/'data_summary.json').read_text())
        names=summary['feature_names']; X=df[names].to_numpy(); y=df[f'{pair}_trend'].to_numpy().astype(int)
        first=len(df)-NTEST; selections=[]; predictions=[]; reference=None
        for step in range(NTEST):
            end=first+step-GAP
            assert df.index[end-1]+pd.Timedelta(weeks=10)<df.index[first+step]
            if step%52==0:
                current,score=tune(model,X[:end],y[:end],pair,step)
                selections.append({'from_step':step,'n_train':end,'train_end':str(df.index[end-1].date()),'mean_auc':score,**current})
                dump(selections,OUT/f'artifacts/{model.lower()}_{pair.lower()}_walkforward_params.json')
            a,b=X[:end],X[first+step:first+step+1]
            if model!='XGBoost':
                scaler=StandardScaler().fit(a); a=scaler.transform(a); b=scaler.transform(b)
            m=fit(model,current,a,y[:end]); p=float(probability(model,m,b)[0])
            predictions.append({'Date':str(df.index[first+step].date()),'actual':int(y[first+step]),'prediction':int(p>=.5),'probability':p})
            if step==0: reference=(m,a)
            if step%13==0 or step==NTEST-1:
                print(f'{pair} {model} walk-forward {step+1}/156 elapsed={(time.time()-t0)/60:.1f} minutes',flush=True)
            outpath=OUT/f'probs/{model.lower()}_{pair.lower()}_probs.csv'; outpath.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(predictions).to_csv(outpath,index=False)
        pr=pd.DataFrame(predictions); yp=pr.prediction; p=pr.probability; yy=pr.actual
        tn,fp,fn,tp=confusion_matrix(yy,yp).ravel()
        metric={'model':model,'factor_pair':pair,'n_test':len(pr),'accuracy':accuracy_score(yy,yp),'roc_auc':roc_auc_score(yy,p),'log_loss':log_loss(yy,np.clip(p,1e-12,1-1e-12)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'elapsed_minutes':(time.time()-t0)/60}
        dump(metric,OUT/f'artifacts/{model.lower()}_{pair.lower()}_metrics.json')
        imp=shap_exact(model,reference[0],reference[1],names)
        sp=OUT/f'shap/{model.lower()}_{pair.lower()}_shap.csv'; sp.parent.mkdir(parents=True,exist_ok=True); imp.to_csv(sp,index=False)
        print('DONE',json.dumps(metric),flush=True)
        return metric

def run_all():
    order=[(m,p) for m in ['GAM','XGBoost','Logistic'] for p in PAIRS]
    results=[]
    with ProcessPoolExecutor(max_workers=9) as pool:
        jobs={pool.submit(run_model,m,p):(m,p) for m,p in order}
        for fut in as_completed(jobs):
            name=jobs[fut]
            try:
                result=fut.result(); results.append(result)
                pd.DataFrame(results).to_csv(OUT/'metrics_summary.csv',index=False)
                print('COMPLETED',json.dumps(result),flush=True)
            except Exception:
                print('FAILED',name,traceback.format_exc(),flush=True)
    assert len(results)==9,f'Only {len(results)} of 9 completed'

if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='models': run_all()
    elif sys.argv[1]=='one': print(run_model(sys.argv[2],sys.argv[3]))
