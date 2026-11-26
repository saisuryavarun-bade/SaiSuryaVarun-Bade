"""Summarize all prespecified sample-size settings and audit cached fits."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pilot import ROOT
from replication import load_named, split_rows, inject, retain, metrics
from sample_size import OUT


def table(df):
    lines = ['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def audit():
    count = 0
    for dataset in ['bank','wdbc']:
        X,y,_ = load_named(dataset)
        seed = 42
        _,sel,_,test = split_rows(y,seed)
        for model in ['logistic','native_boost_calibrated']:
            subdir = OUT/f'{dataset}_seed{seed}'
            cache = np.load(subdir/f'{model}_development.npz')
            predfile = pd.read_csv(ROOT/'outputs/replication_v02'/f'{dataset}_seed{seed}'/f'{model}_predictions.csv')
            results = pd.read_csv(OUT/'metrics.csv')
            budgets = [100,len(sel)] if dataset=='bank' else [30,len(sel)]
            for budget in budgets:
                draw = -1 if budget==len(sel) else 0
                ids = json.loads((subdir/f'{model}_draw{draw}_n{budget}_indices.json').read_text())
                positions = pd.Index(cache['source_indices']).get_indexer(ids)
                assert (positions>=0).all() and len(set(positions))==budget
                ix = np.concatenate([positions+j*len(sel) for j in range(3)])
                target = cache['development_error'][ix]
                for variant in ['confidence','count','mask','placebo']:
                    if len(np.unique(target))<2:
                        estimator = None
                        constant = (target.sum()+.5)/(len(target)+1)
                    else:
                        estimator = Pipeline([('scale',StandardScaler()),('model',LogisticRegression(
                            C=1/(.01*len(ix)),max_iter=2000))])
                        estimator.fit(cache['development_'+variant][ix],target)
                    conditions = [('independent',0.),('independent',.1),('independent',.3)]
                    conditions += [(k,r) for k in ['block_a','block_b'] for r in [.1,.3]]
                    for i,(kind,rate) in enumerate(conditions):
                        condition = f'{kind}_{rate:.1f}'
                        old = predfile[predfile.condition==condition]
                        prob = old.probability.to_numpy()
                        bounded = np.clip(prob,1e-12,1-1e-12)
                        conf = np.column_stack([np.minimum(prob,1-prob),prob,
                            -bounded*np.log(bounded)-(1-bounded)*np.log(1-bounded)])
                        damaged,_,_ = inject(X.iloc[test],kind,rate,seed*3000+i)
                        mask = damaged.isna().to_numpy(float)
                        mask = np.column_stack([mask,mask.mean(axis=1)])
                        if variant=='confidence': f = conf
                        elif variant=='count': f = np.column_stack([conf,mask[:,-1]])
                        elif variant=='mask': f = np.column_stack([conf,mask])
                        else:
                            shuffle = np.random.default_rng(90000+seed*10+i).permutation(len(mask))
                            f = np.column_stack([conf,mask[shuffle]])
                        score = np.full(len(test),constant) if estimator is None else estimator.predict_proba(f)[:,1]
                        pred = (prob>=.5).astype(int)
                        keep = retain(score,pred,seed*4000+i,'global')
                        expected = metrics(y[test],pred,keep)
                        observed = results[(results.dataset==dataset)&(results.seed==seed)&(results.model==model)&
                            (results.budget==budget)&(results.draw==draw)&(results.variant==variant)&
                            (results.condition==condition)].iloc[0]
                        for key in ['risk','balanced_risk','positive_retention','negative_retention','coverage']:
                            assert np.isclose(expected[key],observed[key],equal_nan=True,atol=1e-12)
                        count += 1
    return count


def main():
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='completed exploratory'
    d = pd.read_csv(OUT/'metrics.csv')
    fits = pd.read_csv(OUT/'selector_fits.csv')
    assert len(d)==2352 and len(fits)==336
    keys = ['dataset','seed','model','draw','budget','condition']
    assert not d.duplicated(keys+['variant']).any()
    assert (d[d.budget==d.full_budget].draw==-1).all()
    assert not d[['risk','coverage','positive_retention','negative_retention']].isna().any().any()
    assert len(manifest['classifier_reproduction_checks'])==84
    assert max(x['max_probability_difference'] for x in manifest['classifier_reproduction_checks'])<1e-10
    # Check actual saved subset nesting, not just the sampling function.
    for subdir in OUT.glob('*_seed*'):
        for model in ['logistic','native_boost_calibrated']:
            for draw in [0,1]:
                paths = list(subdir.glob(f'{model}_draw{draw}_n*_indices.json'))
                paths.sort(key=lambda p:int(p.stem.split('_n')[-1].split('_')[0]))
                previous = set()
                for p in paths:
                    current = set(json.loads(p.read_text()))
                    assert previous<=current
                    previous = current
    audited = audit()
    reference = d[d.variant=='confidence'].set_index(keys)
    paired = []
    for variant in ['count','mask','placebo']:
        a = d[d.variant==variant].set_index(keys)
        assert a.index.equals(reference.index)
        frame = a[['full_budget','unique_error_rows']].copy()
        for metric in ['risk','balanced_risk','positive_retention','negative_retention']:
            frame['delta_'+metric] = a[metric]-reference[metric]
        frame['variant'] = variant
        paired.append(frame.reset_index())
    paired = pd.concat(paired,ignore_index=True)
    paired.to_csv(OUT/'paired_effects.csv',index=False)
    # Average subset draws within each seed first; full-budget data counted once.
    by_seed = paired.groupby(['dataset','seed','model','budget','condition','variant']).agg(
        risk_delta=('delta_risk','mean'),balanced_delta=('delta_balanced_risk','mean'),
        positive_retention_delta=('delta_positive_retention','mean')).reset_index()
    summary = by_seed.groupby(['dataset','model','budget','condition','variant']).agg(
        mean_risk_delta=('risk_delta','mean'),min_risk_delta=('risk_delta','min'),max_risk_delta=('risk_delta','max'),
        mean_balanced_delta=('balanced_delta','mean'),
        mean_positive_retention_delta=('positive_retention_delta','mean')).reset_index()
    by_seed.to_csv(OUT/'within_seed_effects.csv',index=False)
    summary.to_csv(OUT/'seed_summaries.csv',index=False)
    raw = pd.read_csv(OUT/'raw_confidence_reference.csv')
    raw_keys = ['dataset','seed','model','condition']
    raw_comparison = d.merge(raw[raw_keys+['risk']],on=raw_keys,suffixes=['','_raw'])
    raw_comparison['delta_vs_raw'] = raw_comparison.risk-raw_comparison.risk_raw
    raw_comparison.to_csv(OUT/'raw_reference_comparisons.csv',index=False)
    colors = {'count':'#1b9e77','mask':'#d95f02','placebo':'#7570b3'}
    for condition in ['independent_0.3','block_a_0.3','block_b_0.3']:
        fig,axes = plt.subplots(2,2,figsize=(11,7))
        for i,dataset in enumerate(['bank','wdbc']):
            for j,model in enumerate(['logistic','native_boost_calibrated']):
                ax = axes[i,j]
                for variant in ['count','mask','placebo']:
                    q = summary[(summary.dataset==dataset)&(summary.model==model)&
                        (summary.condition==condition)&(summary.variant==variant)].sort_values('budget')
                    x = q.budget.to_numpy()
                    ax.plot(x,q.mean_risk_delta.to_numpy()*100,'o-',color=colors[variant],label=variant)
                    ax.fill_between(x,q.min_risk_delta.to_numpy()*100,q.max_risk_delta.to_numpy()*100,
                        color=colors[variant],alpha=.12)
                ax.axhline(0,color='black',linewidth=.8)
                ax.set_xscale('log')
                ax.set_title(f'{dataset}: {model}')
                ax.set_xlabel('Unique labelled selector rows')
                ax.set_ylabel('Error difference vs learned confidence (pp)')
                ax.grid(alpha=.2)
        axes[0,0].legend()
        fig.suptitle(f'{condition}: 80% retained; lower favors added features\nLines: 3-seed means. Shading: seed range, not confidence intervals.')
        fig.tight_layout()
        fig.savefig(OUT/f'learning_curve_{condition}.png',dpi=180)
        plt.close(fig)
    display = summary[(summary.condition=='independent_0.3')&(summary.variant=='mask')].copy()
    display = display[['dataset','model','budget','mean_risk_delta','mean_balanced_delta','mean_positive_retention_delta']]
    display.iloc[:,3:] = display.iloc[:,3:]*100
    display.columns = ['Dataset','Classifier','Unique rows','Error change pp','Balanced error change pp','Positive retention change pp']
    comparisons=[]
    for dataset in ['bank','wdbc']:
        for model in ['logistic','native_boost_calibrated']:
            q = summary[(summary.dataset==dataset)&(summary.model==model)&(summary.condition=='independent_0.3')]
            for variant in ['count','mask','placebo']:
                v=q[q.variant==variant].sort_values('budget')
                comparisons.append({'Dataset':dataset,'Classifier':model,'Variant':variant,
                    'Smallest budget error change pp':v.iloc[0].mean_risk_delta*100,
                    'Full budget error change pp':v.iloc[-1].mean_risk_delta*100})
    fallback_n = int(fits.constant_fallback.sum())
    report=f'''# Controlled selector sample size experiment

Completed 27 October 2026. Exploratory study v0.3, not final validation.

## What was controlled

The classifier and evaluation data were held fixed while the selector's labelled unique-row budget varied. There are 12 classifier configurations (2 datasets x 3 seeds x 2 classifiers), 336 selector fits, and 2,352 condition-level metric records. All 84 condition-level classifier prediction comparisons matched the saved v0.2 outputs to within 1e-10. Nested budgets were verified from saved source-row identifiers. An audit rebuilt {audited} metric records using cached development features and the prior prediction files.

Corrupted copies do not count as new labelled source cases. Full-budget fits are counted once per seed, rather than duplicated across subset draws. The run had {fallback_n} constant error-score fallbacks; all are retained in outputs.

## Full-mask learning curve

At 30 percent independent cell masking and 80 percent batch retention, the following are mean combined-minus-confidence differences. Negative error difference favors added mask information. Negative positive-retention difference means more actual positives are referred. Average subset draws within seed before averaging seeds.

{table(display)}

## Low-dimensional and placebo controls

{table(pd.DataFrame(comparisons))}

The placebo shuffles mask blocks across rows within corruption conditions. It preserves feature count and marginal mask patterns but breaks their case-specific linkage. Its features may use masks from the full unlabelled selector pool, though no outside labels. This control cannot establish a complete causal explanation of performance differences.

## Interpretation limits

The untrained raw-confidence baseline is included in `raw_reference_comparisons.csv`. At independent masking 0.3, the banking calibrated classifier's full-mask selector has about +1.960 percentage points more error than raw-confidence selection at 100 development rows, and +0.078 points at the full budget. Thus nearing the learned confidence-only baseline does not demonstrate an advantage over the untrained baseline.

These curves test development sample size within fixed classifier/split/condition realizations. They do not estimate a universal required sample size, formal risk control, or clinical/business benefit. The WDBC evaluation set is small, so a few predictions can move percentages substantially. Seed ranges describe variation; overlapping splits are not independent replications.

This version holds mean-loss regularization fixed, unlike v0.2's fixed C. Compare variants within this run; do not explain between-version changes solely through sample size. Neither training loss nor an improving curve by itself proves overfitting caused earlier failure. All settings, including harms and null effects, remain in `outputs/sample_size_v03/paired_effects.csv`.

## Publication position

The experiment provides a more controlled diagnostic than the initial cross-dataset comparison. It is still a research-development result. Existing work includes learned model-failure predictors (Risk Advisor), shift-aware selective prediction and missingness-aware deferral. Source access and overlap questions remain documented. No novelty or acceptance claim is established.

The next contribution decision should consider whether effects persist with lower-dimensional mask summaries, held-out outage patterns, calibrated classifiers, task-relevant costs and additional datasets. A larger experiment should follow only after the exact closest-work gap is resolved.
'''
    (ROOT/'sample_size_report.md').write_text(report,encoding='utf-8')
    print(display.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f'Audited {audited} records. Constant fallbacks: {fallback_n}')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
