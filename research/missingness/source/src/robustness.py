"""Versioned cleaning correction, repeated masks and validation-selected penalties."""
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pilot import ROOT
from replication import ResearchClassifier,load_named,split_rows,inject,retain,metrics
from external_validation import load_extra
from sample_size import nested_order,features

OUT=ROOT/'outputs'/'robustness_v06'
LAMBDAS=[.001,.01,.1,1.]
VARIANTS=['confidence','count','mask','placebo']
CONDITIONS=['independent','block_a','block_b']


def cleaned_data(dataset):
    X,y,meta=load_named(dataset) if dataset in ['bank','wdbc'] else load_extra(dataset)
    keep=~X.duplicated()
    removed=int((~keep).sum())
    X,y=X.loc[keep].copy(),y[keep]
    assert not X.duplicated().any()
    return X,y,{**meta,'v06_postexclusion_feature_duplicates_removed':removed,'v06_rows':len(X),
        'v06_table_sha256':hashlib.sha256(X.assign(_target=y).to_csv(index=True).encode()).hexdigest()}


def seed_base(split_seed,repetition):
    return 1_000_000+split_seed*10000+repetition*100


def check_feature_isolation(X,splits):
    hashes=pd.util.hash_pandas_object(X,index=False).to_numpy()
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            assert not set(hashes[a]) & set(hashes[b])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').is_file() and 'completed' in json.loads((OUT/'manifest.json').read_text()).get('status',''):
        raise FileExistsError('Completed outputs preserved; create a new version for new experiments.')
    manifest={'status':'running exploratory robustness','started_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'lambda_grid':LAMBDAS,'repetitions':3,'seeds':[17,42,83],'datasets':{},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'robustness_plan.md').read_bytes()).hexdigest(),
        'helper_hashes':{name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                         for name in ['pilot.py','replication.py','sample_size.py','external_validation.py']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows,chosen,fit_rows,refs=[],[],[],[]
    for dataset in ['bank','wdbc','spambase','online_shoppers']:
        X,y,source=cleaned_data(dataset)
        manifest['datasets'][dataset]=source
        print(f'{dataset}: {len(X)} cleaned rows',flush=True)
        for seed in [17,42,83]:
            splits=split_rows(y,seed)
            train,sel,val,test=splits
            check_feature_isolation(X,splits)
            subdir=OUT/f'{dataset}_seed{seed}'
            subdir.mkdir(exist_ok=True)
            (subdir/'splits.json').write_text(json.dumps({k:v.tolist() for k,v in
                zip(['train','selector','validation','evaluation'],splits)}),encoding='utf-8')
            model=CalibratedClassifierCV(ResearchClassifier('native_boost_augmented',seed),method='sigmoid',
                cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1).fit(X.iloc[train],y[train])
            joblib.dump(model,subdir/'base_model.joblib',compress=3)
            budgets=[30,len(sel)] if dataset=='wdbc' else [100,len(sel)]
            order=nested_order(y[sel],seed*100)
            for budget in budgets:
                (subdir/f'n{budget}_source_ids.json').write_text(json.dumps(X.index[sel[order[:budget]]].tolist()),encoding='utf-8')
            for rep in range(3):
                print(f'{dataset}/{seed}: corruption repetition {rep}',flush=True)
                base=seed_base(seed,rep)
                dev={v:[] for v in VARIANTS};vfeatures={v:[] for v in VARIANTS}
                errors=[];verrors=[]
                for i,rate in enumerate([0.,.1,.3]):
                    damaged,_,_=inject(X.iloc[sel],'independent',rate,base+i)
                    p,fmap=features(model,damaged,base+60+i)
                    errors.append(((p>=.5)!=y[sel]).astype(int))
                    for v in VARIANTS:dev[v].append(fmap[v])
                    damaged,_,_=inject(X.iloc[val],'independent',rate,base+20+i)
                    p,fmap=features(model,damaged,base+70+i)
                    verrors.append(((p>=.5)!=y[val]).astype(int))
                    for v in VARIANTS:vfeatures[v].append(fmap[v])
                dev={k:np.vstack(v) for k,v in dev.items()}
                vfeatures={k:np.vstack(v) for k,v in vfeatures.items()}
                errors,verrors=np.concatenate(errors),np.concatenate(verrors)
                evals=[]
                cache={'source_indices':X.index[sel].to_numpy(),'error':errors,'validation_error':verrors}
                cache.update({'development_'+k:v for k,v in dev.items()})
                cache.update({'validation_'+k:v for k,v in vfeatures.items()})
                for ci,kind in enumerate(CONDITIONS):
                    damaged,added,newly=inject(X.iloc[test],kind,.3,base+40+ci)
                    p,fmap=features(model,damaged,base+80+ci)
                    pred=(p>=.5).astype(int)
                    evals.append((ci,kind,p,pred,fmap,added,newly))
                    keep=retain(np.minimum(p,1-p),pred,base+90+ci,'global')
                    refs.append({'dataset':dataset,'seed':seed,'repetition':rep,'condition':kind,**metrics(y[test],pred,keep)})
                    cache[f'eval_c{ci}_probability']=p
                    for v in VARIANTS:cache[f'eval_c{ci}_{v}']=fmap[v]
                np.savez_compressed(subdir/f'meta_rep{rep}.npz',**cache)
                for budget in budgets:
                    pos=order[:budget]
                    ix=np.concatenate([pos+j*len(sel) for j in range(3)])
                    target=errors[ix]
                    saved={'source_index':X.index[test].to_numpy(),'label':y[test]}
                    for ci,_,p,pred,_,_,_ in evals:
                        saved[f'probability_c{ci}']=p
                        saved[f'prediction_c{ci}']=pred
                    for variant in VARIANTS:
                        criteria=[];local_rows=[]
                        fallback=len(np.unique(target))<2
                        for li,lam in enumerate(LAMBDAS):
                            if fallback:
                                constant=float((target.sum()+.5)/(len(target)+1))
                                vp=np.full(len(verrors),constant)
                            else:
                                estimator=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(
                                    C=1/(lam*len(ix)),max_iter=2000))]).fit(dev[variant][ix],target)
                                vp=estimator.predict_proba(vfeatures[variant])[:,1]
                            criterion=log_loss(verrors,np.column_stack([1-vp,vp]),labels=[0,1])
                            criteria.append(criterion)
                            fit_rows.append({'dataset':dataset,'seed':seed,'repetition':rep,'budget':budget,
                                'variant':variant,'lambda_index':li,'lambda':lam,'validation_log_loss':criterion,
                                'constant_fallback':fallback,'error_observations':int(target.sum())})
                            for ci,kind,p,pred,fmap,added,newly in evals:
                                score=np.full(len(test),constant) if fallback else estimator.predict_proba(fmap[variant])[:,1]
                                saved[f'l{li}_{variant}_c{ci}']=score
                                keep=retain(score,pred,base+90+ci,'global')
                                row={'dataset':dataset,'seed':seed,'repetition':rep,'budget':budget,'variant':variant,
                                    'lambda_index':li,'lambda':lam,'condition':kind,'added_mask_fraction':added,
                                    'newly_lost_fraction':newly,**metrics(y[test],pred,keep)}
                                rows.append(row);local_rows.append(row)
                        selected=int(np.argmin(criteria))
                        saved['selected_'+variant]=np.array(selected)
                        for row in local_rows:
                            if row['lambda_index']==selected:chosen.append(row.copy())
                    np.savez_compressed(subdir/f'scores_rep{rep}_n{budget}.npz',**saved)
                pd.DataFrame(rows).to_csv(OUT/'grid_metrics.csv',index=False)
                pd.DataFrame(chosen).to_csv(OUT/'tuned_metrics.csv',index=False)
                pd.DataFrame(fit_rows).to_csv(OUT/'selector_fits.csv',index=False)
                pd.DataFrame(refs).to_csv(OUT/'raw_confidence.csv',index=False)
    manifest['status']='completed exploratory robustness'
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed corrected-data, repeated-mask and regularization study.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
