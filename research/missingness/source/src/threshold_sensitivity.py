"""Validation-only decision-threshold sensitivity with unchanged v0.6 scores."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pilot import ROOT
from replication import split_rows,retain,metrics
from robustness import cleaned_data,check_feature_isolation,LAMBDAS,VARIANTS,CONDITIONS,seed_base

OUT=ROOT/'outputs'/'threshold_sensitivity_v08'


def choose_threshold(y,p):
    """Exact balanced accuracy at each candidate, O(n log n) after sorting."""
    distinct=np.unique(p)
    candidates=np.concatenate([[0.],(distinct[:-1]+distinct[1:])/2,[np.nextafter(1.,2.)]])
    pos=np.sort(p[y==1]);neg=np.sort(p[y==0])
    assert len(pos)>0 and len(neg)>0
    tpr=1-np.searchsorted(pos,candidates,side='left')/len(pos)
    tnr=np.searchsorted(neg,candidates,side='left')/len(neg)
    ba=(tpr+tnr)/2
    best=np.flatnonzero(np.isclose(ba,ba.max(),atol=1e-12,rtol=0))
    index=sorted(best,key=lambda i:(abs(candidates[i]-.5),candidates[i]))[0]
    return float(candidates[index]),float(ba[index])


def checks():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.3,.4])
    t,ba=choose_threshold(y,p)
    assert np.isclose(t,.25) and ba==1
    assert balanced_accuracy_score(y,p>=t)==ba
    y=np.array([0,1,0,1]);p=np.array([.1,.1,.9,.9])
    t,ba=choose_threshold(y,p)
    assert np.isclose(t,.5) and ba==.5


def main():
    checks();OUT.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running threshold sensitivity','started_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'threshold_sensitivity_plan.md').read_bytes()).hexdigest(),
        'lambda_grid':LAMBDAS,'base_models':'unchanged v0.6 locally generated models',
        'note':'same validation rows select base threshold and selector lambda; no evaluation tuning'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows,chosen,fits,thresholds,refs=[],[],[],[],[]
    for dataset in ['bank','wdbc','spambase','online_shoppers']:
        X,y,_=cleaned_data(dataset)
        for seed in [17,42,83]:
            splits=split_rows(y,seed);check_feature_isolation(X,splits)
            _,sel,val,test=splits
            original=ROOT/'outputs/robustness_v06'/f'{dataset}_seed{seed}'
            subdir=OUT/f'{dataset}_seed{seed}';subdir.mkdir(exist_ok=True)
            with np.load(original/'meta_rep0.npz') as z:
                clean_p=z['validation_confidence'][:len(val),1]
            threshold,score=choose_threshold(y[val],clean_p)
            thresholds.append({'dataset':dataset,'seed':seed,'threshold':threshold,'validation_balanced_accuracy':score,
                'fixed_0.5_validation_balanced_accuracy':balanced_accuracy_score(y[val],clean_p>=.5),'validation_rows':len(val)})
            budgets=[30,len(sel)] if dataset=='wdbc' else [100,len(sel)]
            for rep in range(3):
                print(f'{dataset}/{seed}/rep{rep}: threshold={threshold:.4f}',flush=True)
                with np.load(original/f'meta_rep{rep}.npz') as z:
                    cache={k:z[k].copy() for k in z.files}
                source_index=pd.Index(cache['source_indices'])
                assert np.array_equal(source_index.to_numpy(),X.index[sel].to_numpy())
                error=((cache['development_confidence'][:,1]>=threshold)!=np.tile(y[sel],3)).astype(int)
                val_error=((cache['validation_confidence'][:,1]>=threshold)!=np.tile(y[val],3)).astype(int)
                for ci,condition in enumerate(CONDITIONS):
                    p=cache[f'eval_c{ci}_probability'];pred=(p>=threshold).astype(int)
                    keep=retain(np.minimum(p,1-p),pred,seed_base(seed,rep)+90+ci,'global')
                    refs.append({'dataset':dataset,'seed':seed,'repetition':rep,'condition':condition,'threshold':threshold,
                        **metrics(y[test],pred,keep),'retained_predicted_positive_fraction':float(pred[keep].mean())})
                for budget in budgets:
                    ids=json.loads((original/f'n{budget}_source_ids.json').read_text())
                    positions=source_index.get_indexer(ids)
                    ix=np.concatenate([positions+j*len(sel) for j in range(3)])
                    saved={'source_index':X.index[test].to_numpy(),'label':y[test]}
                    for ci in range(3):saved[f'prediction_c{ci}']=(cache[f'eval_c{ci}_probability']>=threshold).astype(int)
                    for variant in VARIANTS:
                        local=[];criteria=[];fallback=len(np.unique(error[ix]))<2
                        for li,lam in enumerate(LAMBDAS):
                            if fallback:
                                constant=(error[ix].sum()+.5)/(len(ix)+1)
                                vp=np.full(len(val_error),constant)
                            else:
                                estimator=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(
                                    C=1/(lam*len(ix)),max_iter=2000))]).fit(cache['development_'+variant][ix],error[ix])
                                vp=estimator.predict_proba(cache['validation_'+variant])[:,1]
                            criterion=log_loss(val_error,np.column_stack([1-vp,vp]),labels=[0,1]);criteria.append(criterion)
                            fits.append({'dataset':dataset,'seed':seed,'repetition':rep,'budget':budget,'variant':variant,
                                'lambda_index':li,'lambda':lam,'validation_log_loss':criterion,'constant_fallback':fallback})
                            for ci,condition in enumerate(CONDITIONS):
                                pred=saved[f'prediction_c{ci}']
                                s=np.full(len(test),constant) if fallback else estimator.predict_proba(cache[f'eval_c{ci}_{variant}'])[:,1]
                                saved[f'l{li}_{variant}_c{ci}']=s
                                keep=retain(s,pred,seed_base(seed,rep)+90+ci,'global')
                                row={'dataset':dataset,'seed':seed,'repetition':rep,'budget':budget,'variant':variant,
                                    'lambda_index':li,'lambda':lam,'condition':condition,'threshold':threshold,
                                    **metrics(y[test],pred,keep),
                                    'retained_predicted_positive_fraction':float(pred[keep].mean())}
                                rows.append(row);local.append(row)
                        selected=int(np.argmin(criteria));saved['selected_'+variant]=np.array(selected)
                        chosen.extend(row.copy() for row in local if row['lambda_index']==selected)
                    np.savez_compressed(subdir/f'scores_rep{rep}_n{budget}.npz',**saved)
                pd.DataFrame(rows).to_csv(OUT/'grid_metrics.csv',index=False)
                pd.DataFrame(chosen).to_csv(OUT/'tuned_metrics.csv',index=False)
                pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
                pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
                pd.DataFrame(refs).to_csv(OUT/'raw_confidence.csv',index=False)
    manifest['status']='completed threshold sensitivity'
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed validation-only threshold sensitivity.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
