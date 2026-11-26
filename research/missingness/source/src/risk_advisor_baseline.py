"""Adaptation of published Risk Advisor ensemble/score; see plan for differences."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from pilot import ROOT, order_with_ties
from replication import load_named, split_rows, inject, retain, metrics

OUT = ROOT/'outputs'/'risk_advisor_v04'
SOURCE_COMMIT = '0a2b5b0473dabecf13e296dc7605bf081c309adb'


def decomposition(member_probability):
    """Input shape members x rows; error probability refers to class 1."""
    average = member_probability.mean(axis=0)
    def binary_h(p):
        p = np.asarray(p)
        out = np.zeros_like(p)
        for term in [p,1-p]:
            positive = term>0
            out[positive] -= term[positive]*np.log2(term[positive])
        return out
    total = binary_h(average)
    aleatoric = binary_h(member_probability).mean(axis=0)
    epistemic = total-aleatoric
    assert epistemic.min() > -1e-10
    return {'prob_error':average,'total_entropy':total,'aleatoric':aleatoric,
            'epistemic':epistemic,'published_risk':average+total}


def self_checks():
    p = np.array([[0.,.2,.6,1.],[0.,.4,.8,1.]])
    got = decomposition(p)
    avg = p.mean(axis=0)
    expected_total = entropy(np.column_stack([1-avg,avg]),axis=1,base=2)
    expected_alea = np.array([entropy(np.column_stack([1-row,row]),axis=1,base=2) for row in p]).mean(axis=0)
    assert np.allclose(got['total_entropy'],expected_total)
    assert np.allclose(got['aleatoric'],expected_alea)
    assert np.allclose(got['published_risk'],avg+expected_alea+(expected_total-expected_alea))
    # Validate member averaging against sklearn soft voting using fitted members.
    X = np.arange(80.).reshape(40,2)
    y = np.tile([0,0,1,1],10)
    ensemble = VotingClassifier([(str(i),GradientBoostingClassifier(n_estimators=5,max_depth=2,
        subsample=.5,random_state=i)) for i in [1,2]],voting='soft').fit(X,y)
    mean = np.array([m.predict_proba(X)[:,1] for m in ensemble.estimators_]).mean(axis=0)
    assert np.allclose(mean,ensemble.predict_proba(X)[:,1])


def preprocessor(X):
    num = X.select_dtypes(include='number').columns.tolist()
    cat = [c for c in X if c not in num]
    transforms = [('num',Pipeline([('impute',SimpleImputer(strategy='median')),
        ('scale',StandardScaler())]),num)]
    if cat:
        transforms.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    return ColumnTransformer(transforms).fit(X)


def main():
    self_checks()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = {'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running exploratory adaptation',
        'source_commit':SOURCE_COMMIT,'ensemble_members':10,'member_seeds':list(range(1,11)),
        'member_params':{'n_estimators':100,'max_depth':4,'subsample':.5,'learning_rate':.1},
        'base_classifier':'native_boost_calibrated','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'risk_advisor_plan.md').read_bytes()).hexdigest(),
        'input_provenance':{},'self_checks':'passed entropy and soft voting parity'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows,fit_rows = [],[]
    for dataset in ['bank','wdbc']:
        X,y,source = load_named(dataset)
        manifest['input_provenance'][dataset] = source
        for seed in [17,42,83]:
            train,sel,_,test = split_rows(y,seed)
            transformer = preprocessor(X.iloc[train])
            cache_path = ROOT/'outputs/sample_size_v03'/f'{dataset}_seed{seed}'/'native_boost_calibrated_development.npz'
            cache = np.load(cache_path)
            assert np.array_equal(cache['source_indices'],X.index[sel].to_numpy())
            prob_development = cache['development_confidence'][:,1]
            development_error = cache['development_error']
            assert np.array_equal(development_error,((prob_development>=.5)!=np.tile(y[sel],3)).astype(int))
            covariates, masks = [],[]
            for i,rate in enumerate([0.,.1,.3]):
                damaged,_,_ = inject(X.iloc[sel],'independent',rate,seed*1000+i)
                covariates.append(transformer.transform(damaged))
                masks.append(damaged.isna().to_numpy(float))
            base_development = np.column_stack([np.vstack(covariates),1-prob_development,prob_development])
            mask_development = np.column_stack([base_development,np.vstack(masks)])
            old_path = ROOT/'outputs/replication_v02'/f'{dataset}_seed{seed}'/'native_boost_calibrated_predictions.csv'
            old = pd.read_csv(old_path)
            evaluation = []
            conditions = [('independent',0.),('independent',.1),('independent',.3)]
            conditions += [(k,r) for k in ['block_a','block_b'] for r in [.1,.3]]
            for i,(kind,rate) in enumerate(conditions):
                condition = f'{kind}_{rate:.1f}'
                prior = old[old.condition==condition]
                assert np.array_equal(prior.source_index.to_numpy(),X.index[test].to_numpy())
                prob = prior.probability.to_numpy()
                damaged,_,_ = inject(X.iloc[test],kind,rate,seed*3000+i)
                inputs = np.column_stack([transformer.transform(damaged),1-prob,prob])
                inputs_mask = np.column_stack([inputs,damaged.isna().to_numpy(float)])
                evaluation.append((condition,prob,inputs,inputs_mask,i))
            budgets = [100,1000,len(sel)] if dataset=='bank' else [30,len(sel)]
            subdir = OUT/f'{dataset}_seed{seed}'
            subdir.mkdir(exist_ok=True)
            for budget in budgets:
                draw = -1 if budget==len(sel) else 0
                ids_path = ROOT/'outputs/sample_size_v03'/f'{dataset}_seed{seed}'/f'native_boost_calibrated_draw{draw}_n{budget}_indices.json'
                ids = json.loads(ids_path.read_text())
                positions = pd.Index(X.index[sel]).get_indexer(ids)
                assert len(set(positions))==budget and (positions>=0).all()
                ix = np.concatenate([positions+j*len(sel) for j in range(3)])
                for input_variant,development in [('covariates',base_development),('covariates_mask',mask_development)]:
                    print(f'{dataset}/{seed} budget={budget} {input_variant}: fitting 10 members',flush=True)
                    started = time.perf_counter()
                    members = []
                    for member_seed in range(1,11):
                        member = GradientBoostingClassifier(n_estimators=100,max_depth=4,
                            subsample=.5,learning_rate=.1,random_state=member_seed)
                        member.fit(development[ix],development_error[ix])
                        members.append(member)
                    fit_rows.append({'dataset':dataset,'seed':seed,'budget':budget,'draw':draw,
                        'input_variant':input_variant,'dimensions':development.shape[1],
                        'unique_source_rows':budget,'augmented_rows':len(ix),
                        'fit_seconds':time.perf_counter()-started,
                        'subset_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
                    score_frames = []
                    for condition,prob,inputs,inputs_mask,i in evaluation:
                        evaluation_input = inputs if input_variant=='covariates' else inputs_mask
                        member_prob = np.array([m.predict_proba(evaluation_input)[:,1] for m in members])
                        components = decomposition(member_prob)
                        score_frame = pd.DataFrame({'source_index':X.index[test],'label':y[test],
                            'base_probability':prob,'condition':condition,**components})
                        pred = (prob>=.5).astype(int)
                        err = (pred!=y[test]).astype(int)
                        for scoring in ['prob_error','published_risk']:
                            score = components[scoring]
                            keep = retain(score,pred,seed*4000+i,'global')
                            order = order_with_ties(score,seed*4000+i)
                            rows.append({'dataset':dataset,'seed':seed,'budget':budget,'draw':draw,
                                'input_variant':input_variant,'scoring':scoring,'condition':condition,
                                **metrics(y[test],pred,keep),
                                'aurc':float(np.mean(np.cumsum(err[order])/np.arange(1,len(err)+1))),
                                'error_detection_auc':roc_auc_score(err,score) if len(np.unique(err))==2 else np.nan})
                        score_frames.append(score_frame)
                    pd.concat(score_frames).to_csv(subdir/f'n{budget}_{input_variant}_scores.csv',index=False)
                    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
                    pd.DataFrame(fit_rows).to_csv(OUT/'fits.csv',index=False)
                    print(f'Finished in {time.perf_counter()-started:.1f}s',flush=True)
    manifest['status'] = 'completed exploratory adaptation'
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed all adapted baseline fits.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
