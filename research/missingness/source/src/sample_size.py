"""Fixed-prediction, nested unique-row budget study. See sample_size_plan.md."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pilot import ROOT, meta_features
from replication import ResearchClassifier, load_named, split_rows, inject, retain, metrics

OUT = ROOT/'outputs'/'sample_size_v03'
SEEDS = [17, 42, 83]
MODELS = ['logistic','native_boost_calibrated']
VARIANTS = ['confidence','count','mask','placebo']


def nested_order(y, seed):
    rng = np.random.default_rng(seed)
    queues = [rng.permutation(np.flatnonzero(y==c)).tolist() for c in [0,1]]
    # Sequential proportional allocation creates one nested order for all budgets.
    sizes = np.bincount(y, minlength=2)
    used = [0,0]
    order = []
    for n in range(1,len(y)+1):
        deficits = n*sizes/len(y)-used
        for c in [0,1]:
            if used[c] == sizes[c]:
                deficits[c] = -np.inf
        c = int(np.argmax(deficits))
        order.append(queues[c][used[c]])
        used[c] += 1
    assert len(set(order)) == len(y)
    return np.asarray(order)


def features(model, X, shuffle_seed):
    prob, meta = meta_features(model, X)
    conf = meta['learned_confidence']
    mask = meta['learned_missingness']
    placebo = mask[np.random.default_rng(shuffle_seed).permutation(len(mask))]
    return prob, {'confidence':conf, 'count':np.column_stack([conf,mask[:,-1]]),
                  'mask':np.column_stack([conf,mask]),
                  'placebo':np.column_stack([conf,placebo])}


def checks():
    y = np.array([0]*80+[1]*20)
    a = nested_order(y,1)
    assert len(set(a[:30]) & set(a[:60])) == 30
    assert (y[a[:30]]==1).sum() == 6
    assert (y[a[:60]]==1).sum() == 12


def main():
    checks()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = {'status':'running exploratory','started_utc':datetime.now(timezone.utc).isoformat(),
        'seeds':SEEDS,'models':MODELS,'subsets':2,'regularization_lambda':.01,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256((ROOT/'src/replication.py').read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'sample_size_plan.md').read_bytes()).hexdigest(),
        'classifier_reproduction_checks':[],'datasets':{}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows, fits, fixed_rows, references = [], [], [], []
    for dataset in ['bank','wdbc']:
        X,y,source = load_named(dataset)
        manifest['datasets'][dataset] = source
        for seed in SEEDS:
            train,sel,val,test = split_rows(y,seed)
            budgets = ([100,300,1000,3000,len(sel)] if dataset=='bank' else [30,60,len(sel)])
            subdir = OUT/f'{dataset}_seed{seed}'
            subdir.mkdir(exist_ok=True)
            for model_name in MODELS:
                print(f'{dataset} seed={seed} model={model_name}: fixed classifier',flush=True)
                if model_name == 'native_boost_calibrated':
                    model = CalibratedClassifierCV(ResearchClassifier('native_boost_augmented',seed),
                        method='sigmoid',cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1)
                else:
                    model = ResearchClassifier(model_name,seed)
                model.fit(X.iloc[train],y[train])
                development, labels = {v:[] for v in VARIANTS}, []
                for i,rate in enumerate([0.,.1,.3]):
                    damaged,_,_ = inject(X.iloc[sel],'independent',rate,seed*1000+i)
                    prob,feature_map = features(model,damaged,70000+seed*10+i)
                    labels.append(((prob>=.5)!=y[sel]).astype(int))
                    for key in VARIANTS:
                        development[key].append(feature_map[key])
                err = np.concatenate(labels)
                all_features = {key:np.vstack(value) for key,value in development.items()}
                validation = {v:[] for v in VARIANTS}
                for i,rate in enumerate([0.,.1,.3]):
                    damaged,_,_ = inject(X.iloc[val],'independent',rate,seed*2000+i)
                    _,feature_map = features(model,damaged,80000+seed*10+i)
                    for key in VARIANTS:
                        validation[key].append(feature_map[key])
                validation = {key:np.vstack(value) for key,value in validation.items()}
                conditions = [('independent',0.),('independent',.1),('independent',.3)]
                conditions += [(k,r) for k in ['block_a','block_b'] for r in [.1,.3]]
                evaluation = []
                old = pd.read_csv(ROOT/'outputs/replication_v02'/f'{dataset}_seed{seed}'/f'{model_name}_predictions.csv')
                for i,(kind,rate) in enumerate(conditions):
                    condition = f'{kind}_{rate:.1f}'
                    damaged,added,newly = inject(X.iloc[test],kind,rate,seed*3000+i)
                    prob,feature_map = features(model,damaged,90000+seed*10+i)
                    prior = old[old.condition==condition]
                    assert np.array_equal(prior.source_index.to_numpy(),X.index[test].to_numpy())
                    diff = float(np.max(np.abs(prob-prior.probability.to_numpy())))
                    assert diff < 1e-10, (dataset,seed,model_name,condition,diff)
                    manifest['classifier_reproduction_checks'].append({'dataset':dataset,'seed':seed,
                        'model':model_name,'condition':condition,'max_probability_difference':diff})
                    evaluation.append((condition,(prob>=.5).astype(int),feature_map,i,added,newly))
                    ref_keep = retain(np.minimum(prob,1-prob),(prob>=.5).astype(int),seed*4000+i,'global')
                    references.append({'dataset':dataset,'seed':seed,'model':model_name,'condition':condition,
                        **metrics(y[test],(prob>=.5).astype(int),ref_keep)})
                cache = {'source_indices':X.index[sel].to_numpy(),'development_error':err}
                cache.update({'development_'+k:v for k,v in all_features.items()})
                np.savez_compressed(subdir/f'{model_name}_development.npz',**cache)
                for draw in [0,1]:
                    order = nested_order(y[sel],seed*100+draw)
                    for budget in budgets:
                        if budget==len(sel) and draw==1:
                            continue
                        draw_label = -1 if budget==len(sel) else draw
                        source_ix = order[:budget]
                        ix = np.concatenate([source_ix+j*len(sel) for j in range(3)])
                        selected_errors = err[ix]
                        base = {'dataset':dataset,'seed':seed,'model':model_name,'draw':draw_label,
                            'budget':budget,'full_budget':len(sel),'unique_error_rows':int(
                            np.column_stack(labels)[source_ix].any(axis=1).sum()),
                            'error_observations':int(selected_errors.sum()),'augmented_rows':len(ix)}
                        subset_file = subdir/f'{model_name}_draw{draw_label}_n{budget}_indices.json'
                        subset_file.write_text(json.dumps(X.index[sel[source_ix]].tolist()),encoding='utf-8')
                        for variant in VARIANTS:
                            fallback = len(np.unique(selected_errors)) < 2
                            if fallback:
                                constant = float((selected_errors.sum()+.5)/(len(selected_errors)+1))
                                train_prob = np.full(len(ix),constant)
                                val_score = np.full(len(validation[variant]),constant)
                                estimator = None
                            else:
                                estimator = Pipeline([('scale',StandardScaler()),('model',LogisticRegression(
                                    C=1/(.01*len(ix)),max_iter=2000))])
                                estimator.fit(all_features[variant][ix],selected_errors)
                                train_prob = estimator.predict_proba(all_features[variant][ix])[:,1]
                                val_score = estimator.predict_proba(validation[variant])[:,1]
                            threshold = float(np.quantile(val_score,.8))
                            fits.append({**base,'variant':variant,'dimensions':all_features[variant].shape[1],
                                'constant_fallback':fallback,'C':1/(.01*len(ix)),
                                'training_log_loss':log_loss(selected_errors,np.column_stack([1-train_prob,train_prob]),labels=[0,1])})
                            for condition,pred,fmap,i,added,newly in evaluation:
                                score = (np.full(len(test),constant) if fallback else estimator.predict_proba(fmap[variant])[:,1])
                                keep = retain(score,pred,seed*4000+i,'global')
                                rows.append({**base,'variant':variant,'condition':condition,
                                    'added_mask_fraction':added,'newly_lost_fraction':newly,
                                    **metrics(y[test],pred,keep)})
                                fixed_rows.append({**base,'variant':variant,'condition':condition,'threshold':threshold,
                                    **metrics(y[test],pred,score<=threshold)})
                pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
                pd.DataFrame(fits).to_csv(OUT/'selector_fits.csv',index=False)
                pd.DataFrame(fixed_rows).to_csv(OUT/'fixed_threshold_metrics.csv',index=False)
                pd.DataFrame(references).to_csv(OUT/'raw_confidence_reference.csv',index=False)
                print(f'Completed {dataset}/{seed}/{model_name} budgets={budgets}',flush=True)
    manifest['status'] = 'completed exploratory'
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('All fixed-prediction sample-size experiments completed.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
