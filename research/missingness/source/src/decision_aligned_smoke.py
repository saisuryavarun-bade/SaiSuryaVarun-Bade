"""Bounded exploratory implementation check; see decision_aligned_smoke_plan.md."""
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
from replication import split_rows, retain, metrics
from robustness import cleaned_data, check_feature_isolation, CONDITIONS, LAMBDAS, seed_base
from threshold_sensitivity import choose_threshold

OUT = ROOT/'outputs'/'decision_aligned_v10'
VARIANTS = ['confidence', 'count', 'mask', 'placebo']


def aligned_inputs(cache, prefix, threshold):
    old = cache[prefix+'confidence']
    p, entropy = old[:, 1], old[:, 2]
    h = (p >= threshold).astype(int)
    correctness = np.where(h == 1, p, 1-p)
    conf = np.column_stack([p, h, correctness, entropy])
    return p, h, {
        'confidence': conf,
        'count': np.column_stack([conf, cache[prefix+'count'][:, 3:]]),
        'mask': np.column_stack([conf, cache[prefix+'mask'][:, 3:]]),
        'placebo': np.column_stack([conf, cache[prefix+'placebo'][:, 3:]])}


def record(y, h, score, ci):
    keep = retain(score, h, seed_base(42, 0)+90+ci, 'global')
    return {**metrics(y, h, keep), **{
        f'{name}': int(np.sum(keep & (y == actual) & (h == pred)))
        for name, actual, pred in [('tn',0,0), ('fp',0,1), ('fn',1,0), ('tp',1,1)]}}


def main():
    if OUT.exists():
        raise FileExistsError('Preserve existing v10 outputs; use a new version for a changed run.')
    OUT.mkdir(parents=True)
    manifest = {'status': 'running exploratory smoke check',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256': hashlib.sha256((ROOT/'decision_aligned_smoke_plan.md').read_bytes()).hexdigest(),
        'lambda_grid': LAMBDAS, 'seed':42, 'repetition':0,
        'confidence_columns':['p','assigned_label','assigned_label_probability','entropy'],
        'inputs':{}, 'no_base_refitting':True,
        'warning':'Existing evaluation data; smoke check only; total label cost not equalized.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, fit_rows, counts = [], [], []
    for dataset in ['bank','wdbc','spambase','online_shoppers']:
        X, y, _ = cleaned_data(dataset)
        splits = split_rows(y, 42)
        train, sel, val, test = splits
        check_feature_isolation(X, splits)
        original = ROOT/'outputs'/'robustness_v06'/f'{dataset}_seed42'
        source = original/'meta_rep0.npz'
        manifest['inputs'][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source) as z:
            cache = {k:z[k].copy() for k in z.files}
        assert np.array_equal(cache['source_indices'], X.index[sel].to_numpy())
        learned_budgets = [30, len(sel)] if dataset == 'wdbc' else [100, len(sel)]
        adaptive, _ = choose_threshold(y[val], cache['validation_confidence'][:len(val),1])
        thresholds = {'fixed_0.5': .5, 'validation_balanced': adaptive}
        for regime, threshold in thresholds.items():
            pdev, hdev, dev = aligned_inputs(cache, 'development_', threshold)
            pval, hval, vf = aligned_inputs(cache, 'validation_', threshold)
            err = (hdev != np.tile(y[sel],3)).astype(int)
            verr = (hval != np.tile(y[val],3)).astype(int)
            evaluations = [aligned_inputs(cache, f'eval_c{ci}_', threshold) for ci in range(3)]
            # Independently stored probabilities must match the confidence-cache column.
            for ci, (p, h, _) in enumerate(evaluations):
                assert np.array_equal(p, cache[f'eval_c{ci}_probability'])
                rows.append({'dataset':dataset,'regime':regime,'threshold':threshold,'budget':0,
                    'variant':'decision_aware','condition':CONDITIONS[ci],
                    **record(y[test],h,np.where(h==1,1-p,p),ci)})
            common = {'dataset':dataset,'regime':regime,'base_training_calibration_labels':len(train),
                'threshold_labels':len(val) if regime=='validation_balanced' else 0}
            counts.append({**common,'budget':0,'variant':'decision_aware','fit_labels':0,'tuning_labels':0,
                'unique_post_classifier_labels':common['threshold_labels']})
            for budget in learned_budgets:
                ids = json.loads((original/f'n{budget}_source_ids.json').read_text())
                pos = pd.Index(cache['source_indices']).get_indexer(ids)
                assert len(set(ids)) == budget and np.all(pos >= 0)
                ix = np.concatenate([pos+j*len(sel) for j in range(3)])
                saved = {'label':y[test], 'source_index':X.index[test].to_numpy(),
                    'fit_source_index':np.asarray(ids), 'validation_source_index':X.index[val].to_numpy(),
                    'validation_error':verr,'threshold':np.array(threshold)}
                for ci,(p,h,_) in enumerate(evaluations):
                    saved[f'p_c{ci}'],saved[f'h_c{ci}'] = p,h
                for variant in VARIANTS:
                    losses=[]
                    for li, lam in enumerate(LAMBDAS):
                        fallback = len(np.unique(err[ix])) < 2
                        if fallback:
                            constant = (err[ix].sum()+.5)/(len(ix)+1)
                            vp = np.full(len(verr), constant)
                        else:
                            model = make_pipeline(StandardScaler(), LogisticRegression(
                                C=1/(lam*len(ix)),max_iter=2000)).fit(dev[variant][ix],err[ix])
                            assert model[-1].n_iter_.max() < 2000
                            vp = model.predict_proba(vf[variant])[:,1]
                        loss = log_loss(verr,np.column_stack([1-vp,vp]),labels=[0,1])
                        losses.append(loss)
                        saved[f'{variant}_l{li}_validation'] = vp
                        fit_rows.append({'dataset':dataset,'regime':regime,'budget':budget,'variant':variant,
                            'lambda_index':li,'lambda':lam,'validation_log_loss':loss,
                            'development_error_count':int(err[ix].sum()),'constant_fallback':fallback})
                        for ci, (_,h,ef) in enumerate(evaluations):
                            saved[f'{variant}_l{li}_c{ci}'] = np.full(len(test),constant) if fallback else model.predict_proba(ef[variant])[:,1]
                    selected = int(np.argmin(losses))
                    saved[f'{variant}_selected'] = np.array(selected)
                    for ci,(_,h,_) in enumerate(evaluations):
                        score = saved[f'{variant}_l{selected}_c{ci}']
                        rows.append({'dataset':dataset,'regime':regime,'threshold':threshold,'budget':budget,
                            'variant':variant,'condition':CONDITIONS[ci],'lambda_index':selected,'lambda':LAMBDAS[selected],
                            **record(y[test],h,score,ci)})
                    counts.append({**common,'budget':budget,'variant':variant,'fit_labels':budget,
                        'tuning_labels':len(val),'unique_post_classifier_labels':budget+len(val)})
                np.savez_compressed(OUT/f'{dataset}_{regime}_n{budget}.npz',**saved)
            print(f'Completed {dataset}/{regime}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'selected_metrics.csv',index=False)
    pd.DataFrame(fit_rows).to_csv(OUT/'fits.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'label_accounting.csv',index=False)
    assert len(fit_rows)==256 and len(rows)==216
    manifest.update(status='completed exploratory smoke check',fits=len(fit_rows),records=len(rows),
        completed_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
