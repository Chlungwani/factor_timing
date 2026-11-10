"""Verify corrected predictions, compute paired block inference and redraw figures."""
import os
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[v]='1'
os.environ['MPLBACKEND']='Agg'
import json,sys,itertools,hashlib,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss,confusion_matrix,roc_curve
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from corrected_pipeline import OUT,DATA,PAIRS,MODELS,splits,paths
DEFINITIONS=json.loads((OUT/'feature_definitions.json').read_text(encoding='utf-8'))
sys.stdout.reconfigure(encoding='utf-8')
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COLOURS={'Logistic':'#2a78d6','GAM':'#eb6834','XGBoost':'#1b8d67'}
CATEGORY_COLOURS={'Price-based indicators':'#2a78d6','Fundamental valuation':'#b95783','Sentiment and risk appetite':'#d86434','Macroeconomic':'#6956a8','Factor relative performance':'#b18400','Factor relative volatility':'#1b8d67'}
plt.rcParams.update({'font.family':'Calibri','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'black','text.color':'black','xtick.labelsize':8,'ytick.labelsize':8,'legend.frameon':False,'savefig.facecolor':'white'})

def block_indices(n,L,B=10000):
    starts=np.random.default_rng(0).integers(0,n,size=(B,int(np.ceil(n/L))))
    return ((starts[:,:,None]+np.arange(L))%n).reshape(B,-1)[:,:n]

def inference():
    frames={};rows=[];pairs=[];sensitivity=[]
    original=pd.read_csv(DATA/'labeled_relative_factor_returns.csv',parse_dates=['Date']).set_index('Date')
    for pair in PAIRS:
        for model in MODELS:
            path=OUT/f'probs/{model.lower()}_{pair.lower()}_probs.csv'
            f=pd.read_csv(path,parse_dates=['Date'])
            assert len(f)==156 and f.Date.is_unique
            assert f.Date.iloc[0]==pd.Timestamp('2022-01-31') and f.Date.iloc[-1]==pd.Timestamp('2025-01-20')
            assert np.array_equal(f.actual,original.loc[f.Date,f'{pair}_trend'].to_numpy())
            assert ((f.probability>=.5).astype(int)==f.prediction).all()
            assert np.isfinite(f.probability).all() and f.probability.between(0,1).all()
            frames[pair,model]=f
    B=10000;N=156
    for L in [6,12,24]:
        ix=block_indices(N,L,B)
        for pair in PAIRS:
            y=frames[pair,'Logistic'].actual.to_numpy().astype(int);by=y[ix];n1=by.sum(axis=1);n0=N-n1
            aucs={};losses={}
            for model in MODELS:
                f=frames[pair,model];p=f.probability.to_numpy();pred=f.prediction.to_numpy()
                ranks=rankdata(p[ix],axis=1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    aucdraw=((ranks*by).sum(axis=1)-n1*(n1+1)/2)/(n1*n0)
                accdraw=(pred[ix]==by).mean(axis=1)
                auc=roc_auc_score(y,p);acc=accuracy_score(y,pred);ll=log_loss(y,np.clip(p,1e-12,1-1e-12))
                low,high=np.nanpercentile(aucdraw,[2.5,97.5]);aclo,achi=np.percentile(accdraw,[2.5,97.5])
                sensitivity.append({'block_length':L,'factor_pair':pair,'model':model,'auc_low':low,'auc_high':high,'accuracy_low':aclo,'accuracy_high':achi,'valid_auc_resamples':int(np.isfinite(aucdraw).sum())})
                if L==12:
                    bonlo,bonhi=np.nanpercentile(aucdraw,[100*.05/18,100*(1-.05/18)])
                    tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
                    rows.append({'factor_pair':pair,'model':model,'accuracy':acc,'accuracy_low':aclo,'accuracy_high':achi,'roc_auc':auc,'auc_low':low,'auc_high':high,'auc_bonf_low':bonlo,'auc_bonf_high':bonhi,'log_loss':ll,'majority_accuracy':max(y.mean(),1-y.mean()),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'actual_positive':y.mean(),'predicted_positive':pred.mean(),'min_probability':p.min(),'max_probability':p.max(),'confident_wrong':int((((p>.9)&(y==0))|((p<.1)&(y==1))).sum())})
                aucs[model]=aucdraw
                losses[model]=-(y*np.log(np.clip(p,1e-12,1-1e-12))+(1-y)*np.log(np.clip(1-p,1e-12,1-1e-12)))
            if L==12:
                delta=roc_auc_score(y,frames[pair,'XGBoost'].probability)-roc_auc_score(y,frames[pair,'Logistic'].probability)
                draw=aucs['XGBoost']-aucs['Logistic'];lo,hi=np.nanpercentile(draw,[2.5,97.5]);pval=(1+np.sum(np.abs(draw-delta)>=abs(delta)))/(1+np.isfinite(draw).sum())
                lossdiff=losses['XGBoost']-losses['Logistic'];lossdraw=lossdiff[ix].mean(axis=1);ll,lh=np.percentile(lossdraw,[2.5,97.5]);lp=(1+np.sum(np.abs(lossdraw-lossdiff.mean())>=abs(lossdiff.mean())))/(B+1)
                pairs.append({'factor_pair':pair,'auc_difference':delta,'auc_diff_low':lo,'auc_diff_high':hi,'auc_p':pval,'log_loss_difference':lossdiff.mean(),'log_loss_diff_low':ll,'log_loss_diff_high':lh,'log_loss_p':lp})
    metrics=pd.DataFrame(rows);paired=pd.DataFrame(pairs)
    for key in ['auc_p','log_loss_p']:
        values=paired[key].to_numpy();order=np.argsort(values);adjust=np.maximum.accumulate([(len(values)-i)*values[o] for i,o in enumerate(order)]);out=np.empty(len(values));out[order]=np.minimum(adjust,1);paired[key+'_holm']=out
    metrics.to_csv(OUT/'metrics_with_intervals.csv',index=False);paired.to_csv(OUT/'paired_inference.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'bootstrap_block_sensitivity.csv',index=False)
    source=pd.read_csv(OUT/'metrics_summary.csv')
    for row in metrics.itertuples():
        old=source[(source.model==row.model)&(source.factor_pair==row.factor_pair)].iloc[0]
        assert all(np.isclose(getattr(row,key),old[key]) for key in ['accuracy','roc_auc','log_loss'])
        assert row.tn+row.fp+row.fn+row.tp==156
    print(metrics.round(3).to_string(index=False));print(paired.round(3).to_string(index=False))
    return frames,metrics

def figures(frames,metrics):
    returns=pd.read_csv(DATA/'factor_returns.csv',parse_dates=['Date']).set_index('Date')
    labels=pd.read_csv(DATA/'labeled_relative_factor_returns.csv',parse_dates=['Date']).set_index('Date')
    for number,pair in [(3,'Momentum_vs_Value'),(4,'Value_vs_Quality'),(5,'Momentum_vs_Quality')]:
        a,b=pair.split('_vs_');cumulative=(1+returns[a]-returns[b]).cumprod()-1;target=labels[pair+'_trend'].reindex(cumulative.index)
        fig,ax=plt.subplots(figsize=(6.25,3.5));mask=target.notna();colours=target[mask].map({1:'#2a78d6',0:'#eb6834'})
        ax.bar(cumulative.index[mask],cumulative[mask],width=7,color=colours,linewidth=0)
        ax.plot(cumulative.index,cumulative,color='black',lw=.9);ax.axhline(0,color='#777777',lw=.5)
        ax.legend(handles=[Patch(color='#2a78d6',label=f'Uptrend (n = {int((target==1).sum())})'),Patch(color='#eb6834',label=f'Downtrend (n = {int((target==0).sum())})')],loc='upper left',fontsize=8)
        ax.set(ylabel='Cumulative relative return',xlabel='Weekly observation date');ax.grid(axis='y',color='#e5e5e5',lw=.5);ax.set_axisbelow(True)
        fig.tight_layout();fig.savefig(FIG/f'figure_{number:02}.png',dpi=250);plt.close(fig)
    arrays=np.load(OUT/'clustering_arrays.npz')
    fig,ax=plt.subplots(figsize=(6.25,3.4));dendrogram(arrays['Z'],truncate_mode='level',p=4,ax=ax,no_labels=True,color_threshold=2,above_threshold_color='#777777');ax.axhline(2,color='black',ls='--',lw=.9);ax.set(xlabel='Pre-evaluation candidate-feature distance profiles',ylabel='Ward linkage height');fig.tight_layout();fig.savefig(FIG/'figure_06.png',dpi=250);plt.close(fig)
    summary=json.loads((OUT/'data_summary.json').read_text());n=summary['initial_training_n'];bounds,cv=splits(n)
    combos=list(itertools.combinations(range(5),2))
    fig,ax=plt.subplots(figsize=(6.25,4.0))
    for i,((tr,te),groups) in enumerate(zip(cv,combos)):
        kinds=np.full(n,2);kinds[tr]=0;kinds[te]=1
        for kind,colour in [(0,'#2a78d6'),(1,'#eb6834'),(2,'#f6f6f6')]:
            mask=kinds==kind;starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
            for a,b in zip(starts,ends):ax.add_patch(Rectangle((a,i-.32),b-a,.64,facecolor=colour,edgecolor='#8a8a8a' if kind==2 else 'none',hatch='////' if kind==2 else None,lw=.3))
    for a,b in bounds:ax.axvline(a,color='white',lw=.8)
    ax.set(xlim=(0,n),ylim=(9.7,-.7),xlabel='Observation within the initial purged tuning sample');ax.set_yticks(range(10),[''.join('ABCDE'[j] for j in pair) for pair in combos]);ax.set_ylabel('Held-out groups')
    ax.legend(handles=[Patch(color='#2a78d6',label='Training'),Patch(color='#eb6834',label='Validation'),Patch(facecolor='white',edgecolor='#8a8a8a',hatch='////',label='Purge or embargo')],loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,fontsize=8);fig.tight_layout();fig.savefig(FIG/'figure_07.png',dpi=250);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.25,2.7))
    for i,step in enumerate([0,52,104]):
        end=n+step;forecast=end+12
        ax.barh(i,end,color='#2a78d6',height=.5);ax.barh(i,12,left=end,color='white',edgecolor='#999999',hatch='////',height=.5)
        ax.barh(i,52,left=forecast,color='#eb6834',height=.5);ax.text(end/2,i,f'{end} training observations',ha='center',va='center',fontsize=8,color='white');ax.text(forecast+26,i,'52',ha='center',va='center',fontsize=8,color='white')
    ax.set_yticks(range(3),['Initial settings','Retune after 52','Retune after 104']);ax.set(xlabel='Observation within the complete labelled modelling sample',xlim=(0,summary['n_labelled']),ylim=(2.7,-.7));ax.legend(handles=[Patch(color='#2a78d6',label='Training'),Patch(facecolor='white',edgecolor='#999999',hatch='////',label='12-week purge'),Patch(color='#eb6834',label='Forecast block')],loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3,fontsize=8);fig.tight_layout();fig.savefig(FIG/'figure_08.png',dpi=250);plt.close(fig)
    if frames is None:return
    for i,pair in enumerate(PAIRS):
        fig,ax=plt.subplots(figsize=(6.25,3.7))
        for model in MODELS:
            f=frames[pair,model];fp,tp,_=roc_curve(f.actual,f.probability);auc=roc_auc_score(f.actual,f.probability);ax.plot(fp,tp,color=COLOURS[model],lw=1.3,label=f'{model} (AUC {auc:.3f})')
        ax.plot([0,1],[0,1],color='#777777',ls='--',lw=.8);ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1));ax.legend(loc='lower right');ax.grid(color='#e5e5e5',lw=.5);fig.tight_layout();fig.savefig(FIG/f'figure_{9+2*i:02}.png',dpi=250);plt.close(fig)
        fig,axs=plt.subplots(3,1,figsize=(6.25,4.7),sharex=True,sharey=True)
        for ax,model in zip(axs,MODELS):
            f=frames[pair,model];ax.plot(f.Date,f.probability,color=COLOURS[model],lw=1);ax.scatter(f.Date,f.actual,c='#777777',s=4,alpha=.6);ax.axhline(.5,color='black',ls='--',lw=.6);ax.set_ylim(-.04,1.04);ax.set_ylabel(model);ax.grid(axis='y',color='#e5e5e5',lw=.5)
        axs[-1].set_xlabel('Weekly forecast date');fig.text(.01,.5,'Probability that the first factor leads',rotation=90,va='center',fontsize=8);fig.tight_layout(rect=(.04,0,1,1));fig.savefig(FIG/f'figure_{10+2*i:02}.png',dpi=250);plt.close(fig)
        fig,axs=plt.subplots(3,1,figsize=(6.25,7.0))
        for ax,model in zip(axs,MODELS):
            f=pd.read_csv(OUT/f'shap/{model.lower()}_{pair.lower()}_shap.csv').head(10).iloc[::-1]
            colors=[CATEGORY_COLOURS[DEFINITIONS[name][0]] for name in f.feature]
            ax.barh(range(10),f.mean_abs_shap,color=colors,height=.75);ax.set_yticks(range(10),f.feature,fontsize=8);ax.set_title(model,loc='left',fontsize=10);ax.set_xlabel('Mean absolute attribution in log-odds',fontsize=8);ax.grid(axis='x',color='#e5e5e5',lw=.5);ax.set_axisbelow(True)
        fig.subplots_adjust(left=.50,right=.98,bottom=.06,top=.96,hspace=.6);fig.savefig(FIG/f'figure_{15+i:02}.png',dpi=250);plt.close(fig)

def save_provenance():
    import scipy,sklearn,pygam,xgboost,shap,matplotlib
    files=['featureset.csv','factor_returns.csv','factor_relative_features.csv','labeled_relative_factor_returns.csv']
    record={'python':sys.version,'packages':{m.__name__:m.__version__ for m in [np,pd,scipy,sklearn,pygam,xgboost,shap,matplotlib]},'input_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in files},'bootstrap':{'resamples':10000,'block_length':12,'sensitivity_blocks':[6,24],'seed':0},'verification':'All 9 metric rows independently reproduced from 156 saved predictions; targets match the supplied label file; class decisions match p>=0.5.'}
    (OUT/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

if __name__=='__main__':
    frames,metrics=inference();figures(frames,metrics);save_provenance()
    print('Verified nine corrected runs; inference and figures complete.')
