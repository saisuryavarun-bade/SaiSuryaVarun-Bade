"""Prospectively specified additional datasets; see external_validation_plan.md."""
import hashlib
import json
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pilot import ROOT
from replication import ResearchClassifier,split_rows,inject,retain,metrics
from sample_size import nested_order,features

OUT=ROOT/'outputs'/'additional_datasets_v05'


def load_extra(name):
    archive=ROOT/'data/raw'/('spambase.zip' if name=='spambase' else 'online_shoppers.zip')
    with zipfile.ZipFile(archive) as z:
        if name=='spambase':
            names=[line.split(':')[0].strip() for line in z.read('spambase.names').decode().splitlines()
                   if ':' in line and line.split(':',1)[1].strip()=='continuous.']
            assert len(names)==57
            frame=pd.read_csv(z.open('spambase.data'),header=None,names=names+['_target'],na_values=['?'])
            frame[names]=frame[names].astype(float)
            excluded=[]
            doi='10.24432/C53G6X'
        else:
            frame=pd.read_csv(z.open('online_shoppers_intention.csv'))
            revenue=frame.pop('Revenue')
            if revenue.dtype==bool:
                frame['_target']=revenue.astype(int)
            else:
                frame['_target']=revenue.astype(str).str.lower().map({'true':1,'false':0})
            frame=frame.drop(columns=['PageValues'])
            cat=['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
            for col in cat:
                frame[col]=frame[col].map(lambda x:str(x) if pd.notna(x) else np.nan).astype(object)
            num=[c for c in frame if c not in cat+['_target']]
            frame[num]=frame[num].astype(float)
            excluded=['PageValues']
            doi='10.24432/C5F88Q'
    assert frame._target.isin([0,1]).all()
    original=len(frame)
    frame=frame.drop_duplicates()
    duplicates=original-len(frame)
    covariates=[c for c in frame if c!='_target']
    conflicts=frame.groupby(covariates,dropna=False,sort=False)._target.transform('nunique')>1
    n_conflicts=int(conflicts.sum())
    frame=frame.loc[~conflicts].copy()
    assert not frame[covariates].duplicated().any()
    y=frame.pop('_target').to_numpy(int)
    meta={'source_doi':doi,'license':'CC BY 4.0','raw_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'clean_table_sha256':hashlib.sha256(frame.assign(_target=y).to_csv(index=True).encode()).hexdigest(),
        'original_rows':original,'rows':len(frame),'features':list(frame.columns),'excluded':excluded,
        'duplicates_after_exclusions':duplicates,'conflicting_feature_rows_removed':n_conflicts,
        'parsed_missing_cells':int(frame.isna().to_numpy().sum())}
    return frame,y,meta


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running prospective extension','started_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'external_validation_plan.md').read_bytes()).hexdigest(),
        'datasets':{},'seeds':[17,42,83],'models':['logistic','native_boost_calibrated']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    records,refs,fit_rows=[],[],[]
    for dataset in ['spambase','online_shoppers']:
        X,y,meta=load_extra(dataset)
        manifest['datasets'][dataset]=meta
        for seed in [17,42,83]:
            train,sel,val,test=split_rows(y,seed)
            subdir=OUT/f'{dataset}_seed{seed}'
            subdir.mkdir(exist_ok=True)
            (subdir/'splits.json').write_text(json.dumps({k:v.tolist() for k,v in
                zip(['train','selector','validation','evaluation'],[train,sel,val,test])}),encoding='utf-8')
            order=nested_order(y[sel],seed*100)
            for model_name in ['logistic','native_boost_calibrated']:
                print(f'{dataset}/{seed}/{model_name}',flush=True)
                if model_name=='logistic': model=ResearchClassifier('logistic',seed)
                else: model=CalibratedClassifierCV(ResearchClassifier('native_boost_augmented',seed),
                    method='sigmoid',cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1)
                model.fit(X.iloc[train],y[train])
                pools={k:[] for k in ['confidence','count','mask','placebo']}
                errors=[]
                for i,rate in enumerate([0.,.1,.3]):
                    damaged,_,_=inject(X.iloc[sel],'independent',rate,seed*1000+i)
                    p,fmap=features(model,damaged,70000+seed*10+i)
                    errors.append(((p>=.5)!=y[sel]).astype(int))
                    for k in pools:pools[k].append(fmap[k])
                pools={k:np.vstack(v) for k,v in pools.items()}
                errors=np.concatenate(errors)
                evals=[]
                conditions=[('independent',0.),('independent',.1),('independent',.3)]
                conditions += [(k,r) for k in ['block_a','block_b'] for r in [.1,.3]]
                for i,(kind,rate) in enumerate(conditions):
                    damaged,added,newly=inject(X.iloc[test],kind,rate,seed*3000+i)
                    p,fmap=features(model,damaged,90000+seed*10+i)
                    pred=(p>=.5).astype(int)
                    condition=f'{kind}_{rate:.1f}'
                    keep=retain(np.minimum(p,1-p),pred,seed*4000+i,'global')
                    refs.append({'dataset':dataset,'seed':seed,'model':model_name,'condition':condition,
                        **metrics(y[test],pred,keep)})
                    evals.append((condition,p,pred,fmap,i,added,newly))
                saved=[]
                for budget in [100,300,len(sel)]:
                    positions=order[:budget]
                    ix=np.concatenate([positions+j*len(sel) for j in range(3)])
                    (subdir/f'{model_name}_n{budget}_source_ids.json').write_text(
                        json.dumps(X.index[sel[positions]].tolist()),encoding='utf-8')
                    for variant in pools:
                        fallback=len(np.unique(errors[ix]))<2
                        if fallback: constant=(errors[ix].sum()+.5)/(len(ix)+1)
                        else:
                            selector=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(
                                C=1/(.01*len(ix)),max_iter=2000))]).fit(pools[variant][ix],errors[ix])
                        fit_rows.append({'dataset':dataset,'seed':seed,'model':model_name,'budget':budget,
                            'variant':variant,'constant_fallback':fallback,'augmented_rows':len(ix),
                            'dimensions':pools[variant].shape[1],'error_observations':int(errors[ix].sum())})
                        for condition,p,pred,fmap,i,added,newly in evals:
                            score=np.full(len(test),constant) if fallback else selector.predict_proba(fmap[variant])[:,1]
                            keep=retain(score,pred,seed*4000+i,'global')
                            records.append({'dataset':dataset,'seed':seed,'model':model_name,'budget':budget,
                                'variant':variant,'condition':condition,'added_mask_fraction':added,
                                'newly_lost_fraction':newly,**metrics(y[test],pred,keep)})
                            saved.append(pd.DataFrame({'source_index':X.index[test],'label':y[test],
                                'prediction':pred,'base_probability':p,'budget':budget,'variant':variant,
                                'condition':condition,'score':score}))
                pd.concat(saved).to_csv(subdir/f'{model_name}_scores.csv',index=False)
                pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False)
                pd.DataFrame(refs).to_csv(OUT/'raw_confidence.csv',index=False)
                pd.DataFrame(fit_rows).to_csv(OUT/'fits.csv',index=False)
    manifest['status']='completed prospective extension'
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed additional dataset extension.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
