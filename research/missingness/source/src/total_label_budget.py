"""Budget includes all selector fit/tuning labels. Existing-data exploration only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pilot import ROOT
from robustness import cleaned_data, check_feature_isolation, CONDITIONS, LAMBDAS
from decision_aligned_smoke import aligned_inputs, record, VARIANTS

OUT = ROOT/'outputs'/'total_label_budget_v11'
DATASETS = ['bank','wdbc','spambase','online_shoppers']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed/partial results; use a new version for changed runs.')
    OUT.mkdir(parents=True)
    manifest = {'status':'running exploratory total-label-budget check',
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':sha(Path(__file__)), 'plan_sha256':sha(ROOT/'total_label_budget_design.md'),
        'helpers':{n:sha(ROOT/'src'/n) for n in ['decision_aligned_smoke.py','robustness.py','replication.py','pilot.py']},
        'seed':42,'repetition':0,'threshold':.5,'lambda_grid':LAMBDAS,
        'allocation_seed_base':20261027,'datasets':{},
        'warning':'Previously inspected evaluation data; one allocation/split/mask. No confirmatory claim.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows, fits, accounting = [],[],[]
    for di,dataset in enumerate(DATASETS):
        X,y,_ = cleaned_data(dataset)
        original = ROOT/'outputs/robustness_v06'/f'{dataset}_seed42'
        splits = json.loads((original/'splits.json').read_text())
        train,sel,val,test = [np.asarray(splits[n]) for n in ['train','selector','validation','evaluation']]
        check_feature_isolation(X,[train,sel,val,test])
        source = original/'meta_rep0.npz'
        # Old validation features/labels are deliberately not read from the cache.
        with np.load(source) as z:
            cache = {k:z[k].copy() for k in z.files if k.startswith('development_') or k.startswith('eval_c') or k=='source_indices'}
        assert np.array_equal(cache['source_indices'],X.index[sel].to_numpy())
        _,hdev,development = aligned_inputs(cache,'development_',.5)
        evaluations = [aligned_inputs(cache,f'eval_c{ci}_',.5) for ci in range(3)]
        order = np.random.default_rng(20261027+di).permutation(len(sel))
        budgets = [b for b in [50,100,300,1000,3000] if b<=len(sel)]
        if len(sel) not in budgets: budgets.append(len(sel))
        allocation = {'dataset':dataset,'allocation_seed':20261027+di,'budgets':budgets,
            'source_order':X.index[sel[order]].tolist(),
            'position_order':order.tolist(),'rule':'1-based positions divisible by five tune; all others fit'}
        (OUT/f'{dataset}_allocation.json').write_text(json.dumps(allocation,indent=2))
        manifest['datasets'][dataset] = {'cache_sha256':sha(source),'splits_sha256':sha(original/'splits.json'),
            'allocation_sha256':sha(OUT/f'{dataset}_allocation.json'),'budgets':budgets,
            'base_training_calibration_labels':len(train),'old_validation_source_ids':X.index[val].tolist()}
        for ci,(p,h,_) in enumerate(evaluations):
            rows.append({'dataset':dataset,'budget':0,'variant':'decision_aware','condition':CONDITIONS[ci],
                **record(y[test],h,np.where(h==1,1-p,p),ci)})
        accounting.append({'dataset':dataset,'budget':0,'variant':'decision_aware','fit_labels':0,'tuning_labels':0,
            'threshold_labels':0,'unique_post_classifier_labels':0,'base_training_calibration_labels':len(train)})
        for budget in budgets:
            positions = order[:budget]
            is_tune = (np.arange(budget)+1)%5==0
            fitpos,tunepos = positions[~is_tune],positions[is_tune]
            fi = np.concatenate([fitpos+j*len(sel) for j in range(3)])
            ti = np.concatenate([tunepos+j*len(sel) for j in range(3)])
            err = (hdev[fi]!=np.tile(y[sel[fitpos]],3)).astype(int)
            terr = (hdev[ti]!=np.tile(y[sel[tunepos]],3)).astype(int)
            saved = {'label':y[test],'source_index':X.index[test].to_numpy(),
                'fit_source_index':X.index[sel[fitpos]].to_numpy(),
                'tuning_source_index':X.index[sel[tunepos]].to_numpy(),
                'fit_position':fitpos,'tuning_position':tunepos,
                'fit_error':err,'tuning_error':terr}
            for ci,(p,h,_) in enumerate(evaluations):
                saved[f'p_c{ci}'],saved[f'h_c{ci}'] = p,h
                saved[f'aware_c{ci}'] = np.where(h==1,1-p,p)
            for variant in VARIANTS:
                losses=[]
                fallback = len(np.unique(err))<2
                for li,lam in enumerate(LAMBDAS):
                    if fallback:
                        constant = (err.sum()+.5)/(len(err)+1)
                        vp = np.full(len(terr),constant)
                    else:
                        estimator = make_pipeline(StandardScaler(),LogisticRegression(
                            C=1/(lam*len(fi)),max_iter=2000)).fit(development[variant][fi],err)
                        assert estimator[-1].n_iter_.max()<2000
                        saved[f'{variant}_scale_mean'] = estimator[0].mean_
                        vp = estimator.predict_proba(development[variant][ti])[:,1]
                    criterion = log_loss(terr,np.column_stack([1-vp,vp]),labels=[0,1])
                    losses.append(criterion)
                    saved[f'{variant}_l{li}_tuning'] = vp
                    fits.append({'dataset':dataset,'budget':budget,'variant':variant,'lambda_index':li,'lambda':lam,
                        'tuning_log_loss':criterion,'fit_error_views':int(err.sum()),'tuning_error_views':int(terr.sum()),
                        'fit_error_source_rows':int(np.any(err.reshape(3,-1),axis=0).sum()),
                        'tuning_error_source_rows':int(np.any(terr.reshape(3,-1),axis=0).sum()),
                        'constant_fallback':fallback,'single_class_tuning':len(np.unique(terr))<2})
                    for ci,(_,h,ef) in enumerate(evaluations):
                        saved[f'{variant}_l{li}_c{ci}'] = np.full(len(test),constant) if fallback else estimator.predict_proba(ef[variant])[:,1]
                selected=int(np.argmin(losses))
                saved[f'{variant}_selected']=np.array(selected)
                for ci,(_,h,_) in enumerate(evaluations):
                    rows.append({'dataset':dataset,'budget':budget,'variant':variant,'condition':CONDITIONS[ci],
                        'lambda_index':selected,'lambda':LAMBDAS[selected],
                        **record(y[test],h,saved[f'{variant}_l{selected}_c{ci}'],ci)})
                accounting.append({'dataset':dataset,'budget':budget,'variant':variant,'fit_labels':len(fitpos),
                    'tuning_labels':len(tunepos),'threshold_labels':0,'unique_post_classifier_labels':budget,
                    'base_training_calibration_labels':len(train)})
            np.savez_compressed(OUT/f'{dataset}_B{budget}.npz',**saved)
            print(f'{dataset}: B={budget}, fit={len(fitpos)}, tune={len(tunepos)}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'selected_metrics.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
    pd.DataFrame(accounting).to_csv(OUT/'label_accounting.csv',index=False)
    assert len(fits)==288 and len(rows)==228
    manifest.update(status='completed exploratory total-label-budget check',candidate_fits=len(fits),records=len(rows),
        completed_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
