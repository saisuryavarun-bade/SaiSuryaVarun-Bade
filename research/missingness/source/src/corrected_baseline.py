"""Bounded corrected-data adaptation check; seed42, rep0, independent .3."""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from threadpoolctl import threadpool_limits
from pilot import ROOT
from replication import split_rows,inject,retain,metrics
from robustness import cleaned_data,check_feature_isolation,seed_base
from risk_advisor_baseline import decomposition,preprocessor,self_checks,SOURCE_COMMIT

OUT=ROOT/'outputs'/'corrected_baseline_v07'


def main():
    self_checks();OUT.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running corrected adaptation spot check','started_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':SOURCE_COMMIT,'split_seed':42,'corruption_repetition':0,'condition':'independent_0.3',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'corrected_baseline_plan.md').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]
    for dataset in ['bank','wdbc','spambase','online_shoppers']:
        X,y,_=cleaned_data(dataset)
        splits=split_rows(y,42);check_feature_isolation(X,splits)
        train,sel,_,test=splits
        directory=ROOT/'outputs/robustness_v06'/f'{dataset}_seed42'
        model=joblib.load(directory/'base_model.joblib') # Only our own locally generated model.
        cache=np.load(directory/'meta_rep0.npz')
        assert np.array_equal(cache['source_indices'],X.index[sel].to_numpy())
        probability=cache['development_confidence'][:,1]
        error=cache['error']
        assert np.array_equal(error,((probability>=.5)!=np.tile(y[sel],3)).astype(int))
        trans=preprocessor(X.iloc[train]);frames=[];masks=[];base=seed_base(42,0)
        for i,rate in enumerate([0.,.1,.3]):
            damaged,_,_=inject(X.iloc[sel],'independent',rate,base+i)
            frames.append(trans.transform(damaged));masks.append(damaged.isna().to_numpy(float))
        development=np.column_stack([np.vstack(frames),1-probability,probability])
        development_mask=np.column_stack([development,np.vstack(masks)])
        damaged,_,_=inject(X.iloc[test],'independent',.3,base+40)
        p=model.predict_proba(damaged)[:,1]
        assert np.allclose(p,cache['eval_c0_probability'],atol=1e-12)
        evaluation=np.column_stack([trans.transform(damaged),1-p,p])
        evaluation_mask=np.column_stack([evaluation,damaged.isna().to_numpy(float)])
        budgets=[30,len(sel)] if dataset=='wdbc' else [100,len(sel)]
        for budget in budgets:
            ids=json.loads((directory/f'n{budget}_source_ids.json').read_text())
            positions=pd.Index(X.index[sel]).get_indexer(ids)
            assert (positions>=0).all() and len(set(positions))==budget
            ix=np.concatenate([positions+j*len(sel) for j in range(3)])
            for variant,train_x,test_x in [('covariates',development,evaluation),
                                         ('covariates_mask',development_mask,evaluation_mask)]:
                print(f'{dataset} n={budget} {variant}: 10 members',flush=True)
                started=time.perf_counter();probs=[]
                for member_seed in range(1,11):
                    estimator=GradientBoostingClassifier(n_estimators=100,max_depth=4,subsample=.5,
                        learning_rate=.1,random_state=member_seed).fit(train_x[ix],error[ix])
                    probs.append(estimator.predict_proba(test_x)[:,1])
                parts=decomposition(np.array(probs));pred=(p>=.5).astype(int)
                save=pd.DataFrame({'source_index':X.index[test],'label':y[test],'base_probability':p,**parts})
                save.to_csv(OUT/f'{dataset}_n{budget}_{variant}_scores.csv',index=False)
                for scoring in ['prob_error','published_risk']:
                    keep=retain(parts[scoring],pred,base+90,'global')
                    rows.append({'dataset':dataset,'seed':42,'repetition':0,'budget':budget,'input_variant':variant,
                        'scoring':scoring,**metrics(y[test],pred,keep),'fit_seconds':time.perf_counter()-started})
                pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    manifest['status']='completed corrected adaptation spot check'
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed all corrected baseline checks.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
