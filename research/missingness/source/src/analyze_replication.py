"""Audit saved predictions, summarize paired effects, and write exploratory report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from replication import OUT, ROOT, SEEDS, MODELS, retain, metrics


def md_table(frame, digits=3):
    cols = list(frame.columns)
    lines = ['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(f'{v:.{digits}f}' if isinstance(v, (float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['status'].startswith('completed')
    d = pd.read_csv(OUT/'selective_metrics.csv')
    assert len(d) == 2*3*4*7*9*2
    keys = ['dataset','seed','model','condition','selection_mode']
    assert not d.duplicated(keys+['selector']).any()
    assert d[['risk','coverage','positive_retention','negative_retention']].notna().all().all()
    # Independently recompute metrics for a fixed saved condition for every fit.
    verified = 0
    for dataset in ['bank','wdbc']:
        for seed in SEEDS:
            subdir = OUT/f'{dataset}_seed{seed}'
            splits = json.loads((subdir/'splits.json').read_text())
            arrays = list(splits.values())
            assert len(set(sum(arrays, []))) == sum(map(len, arrays))
            for model in MODELS:
                p = pd.read_csv(subdir/f'{model}_predictions.csv')
                assert not p.duplicated(['source_index','condition']).any()
                assert len(p) == len(splits['evaluation'])*7
                for condition in ['independent_0.3']:
                    q = p[p.condition == condition]
                    for selector in ['linear_confidence','linear_combined']:
                        for mode in ['global','predicted_class']:
                            keep = retain(q['score_'+selector].to_numpy(), q.prediction.to_numpy(), seed*4000+2, mode)
                            expected = metrics(q.label.to_numpy(), q.prediction.to_numpy(), keep)
                            observed = d[(d.dataset==dataset)&(d.seed==seed)&(d.model==model)&
                                (d.condition==condition)&(d.selector==selector)&(d.selection_mode==mode)].iloc[0]
                            for key in ['risk','balanced_risk','coverage','positive_retention','negative_retention']:
                                assert np.isclose(expected[key], observed[key], equal_nan=True)
                            verified += 1
    pairs = []
    for family in ['linear','nonlinear','classweighted']:
        a = d[d.selector==family+'_combined'].set_index(keys)
        b = d[d.selector==family+'_confidence'].set_index(keys)
        assert a.index.equals(b.index)
        pair = pd.DataFrame(index=a.index)
        for col in ['risk','balanced_risk','positive_retention','negative_retention','unreviewed_fn_fraction_of_all_positives']:
            pair['delta_'+col] = a[col]-b[col]
        pair['selector_family'] = family
        pairs.append(pair.reset_index())
    pair = pd.concat(pairs, ignore_index=True)
    pair.to_csv(OUT/'paired_effects.csv', index=False)
    summary = pair.groupby(['dataset','model','condition','selection_mode','selector_family']).agg(
        mean_risk_delta=('delta_risk','mean'), min_risk_delta=('delta_risk','min'),
        max_risk_delta=('delta_risk','max'), mean_balanced_delta=('delta_balanced_risk','mean'),
        mean_positive_retention_delta=('delta_positive_retention','mean')).reset_index()
    summary.to_csv(OUT/'seed_summaries.csv', index=False)
    # Conditional paired bootstrap, prespecified seed and corruption condition.
    # Resample the same original evaluation-row indices for both selectors; rank
    # again within each resampled batch. No cross-condition replicas are pooled.
    bootstrap = []
    for dataset in ['bank','wdbc']:
        for model in MODELS:
            p = pd.read_csv(OUT/f'{dataset}_seed42'/f'{model}_predictions.csv')
            q = p[p.condition=='independent_0.3']
            y, pred = q.label.to_numpy(), q.prediction.to_numpy()
            a, b = q.score_linear_combined.to_numpy(), q.score_linear_confidence.to_numpy()
            rng = np.random.default_rng(17483)
            samples = []
            for i in range(200):
                ix = rng.integers(0, len(q), len(q))
                ka, kb = retain(a[ix], pred[ix], 10000+i, 'global'), retain(b[ix], pred[ix], 10000+i, 'global')
                ma, mb = metrics(y[ix], pred[ix], ka), metrics(y[ix], pred[ix], kb)
                samples.append([ma['risk']-mb['risk'], ma['balanced_risk']-mb['balanced_risk']])
            samples = np.array(samples)
            bootstrap.append({'dataset':dataset,'model':model,'seed':42,'condition':'independent_0.3',
                'risk_delta_low':np.quantile(samples[:,0],.025),'risk_delta_high':np.quantile(samples[:,0],.975),
                'balanced_delta_low':np.nanquantile(samples[:,1],.025),
                'balanced_delta_high':np.nanquantile(samples[:,1],.975),'resamples':200})
    pd.DataFrame(bootstrap).to_csv(OUT/'conditional_bootstrap.csv',index=False)
    core = summary[(summary.condition=='independent_0.3')&(summary.selection_mode=='global')&
                   (summary.selector_family=='linear')].copy()
    display = core[['dataset','model','mean_risk_delta','mean_balanced_delta','mean_positive_retention_delta']].copy()
    display.columns = ['Dataset','Classifier','Error change pp','Balanced error change pp','Positive retention change pp']
    display.iloc[:,2:] = display.iloc[:,2:]*100
    fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
    colors = {'bank':'#2166ac','wdbc':'#b35806'}
    for j,(column,title) in enumerate([('mean_risk_delta','Ordinary error'),('mean_balanced_delta','Balanced error')]):
        for i,row in enumerate(core.itertuples()):
            axes[j].scatter(getattr(row,column)*100,i,color=colors[row.dataset],s=45)
        axes[j].axvline(0,color='black',linewidth=.8)
        axes[j].set_title(title)
        axes[j].set_xlabel('Combined minus confidence-only (percentage points)')
        axes[j].grid(axis='x',alpha=.2)
    axes[0].set_yticks(np.arange(len(core)),[f'{r.dataset}: {r.model}' for r in core.itertuples()])
    axes[0].invert_yaxis()
    fig.suptitle('30% independent masking, 80% batch retention: descriptive mean over 3 seeds\nNegative values favor combined selection; two development datasets')
    fig.tight_layout()
    fig.savefig(OUT/'replication_effects.png',dpi=180)
    plt.close(fig)
    linear_global = pair[(pair.selector_family=='linear')&(pair.selection_mode=='global')]
    tally = linear_global.assign(lower_error=linear_global.delta_risk < -1e-12,
        lower_balanced=linear_global.delta_balanced_risk < -1e-12,
        lower_both=(linear_global.delta_risk < -1e-12)&(linear_global.delta_balanced_risk < -1e-12)).groupby('dataset').agg(
        settings=('delta_risk','size'),lower_error=('lower_error','sum'),
        lower_balanced=('lower_balanced','sum'),lower_both=('lower_both','sum')).reset_index()
    # Describe local evidence; do not treat counts as independent replications.
    report = f'''# Exploratory replication report

Completed 27 October 2026. The initial pilot is preserved separately.

## Scope and verification

Ran two datasets, three fixed seeds, four classifier configurations, seven input conditions, nine selectors, and two batch-selection protocols. Saved {len(d):,} selective-evaluation rows, all 24 fit-level prediction files, fixed-threshold results, classifier metrics, source/split metadata, and a versioned plan.

Checked partition separation and independently recomputed {verified} saved metric records from predictions. Classifier calibration uses only the training partition, with augmentation inside each fold. Both datasets are now development data; this is not final validation or evidence of clinical utility.

## Fixed example chosen in the replication plan

The following compares logistic error selectors (combined versus confidence-only) under 30 percent independent cell masking at 80 percent global batch retention. Numbers are descriptive means across three overlapping split seeds. Negative error change favors combined; negative positive-retention change means more true positives are referred.

{md_table(display)}

Balanced error averages the conditional error rates among retained negatives and retained positives; if either class is absent it is undefined. It is not a guarantee against all forms of selection bias. The retained subset changes between selectors, so report class retention alongside it.

## All planned conditions, without selecting winners

{md_table(tally,0)}

These are counts of correlated exploratory settings (4 classifiers x 3 seeds x 7 conditions per dataset), not independent experiments, a hypothesis-test denominator, or a claim of superiority. Ties are not counted as improvements. The full per-setting effects for every selector family and both selection protocols are in `outputs/replication_v02/paired_effects.csv`.

## Uncertainty

`conditional_bootstrap.csv` gives paired percentile intervals from 200 source-row resamples for seed 42, independent masking 0.3, and linear selectors. Both selectors use the same resampled rows and reallocate the same batch capacity. These intervals condition on the trained models, fixed split, and realized corruption; they omit training and corruption-generation uncertainty. They are exploratory and unadjusted for multiple comparisons. Three seed ranges are descriptive, not confidence intervals.

## Literature and publication position

See `closest_work_audit.md`. Missingness-aware prediction, abstention under shift, and class-aware selective evaluation are established. Recently found repositories and papers overlap closely with the simple proposal, some with unresolved access or review status. We have not established a new method or completed reproduction of published selective-classification baselines.

## Next decision

Use the saved paired effects to decide whether a narrowly framed empirical question is worthwhile, especially under correlated outages and stronger selectors. A lower aggregate error alone does not qualify as success. Add larger, diverse datasets and published baselines only after resolving the closest-work overlap. Do not submit the original simple combined-selector idea as a novel algorithm.

No external upload, submission, acceptance, or publication has occurred.
'''
    (ROOT/'replication_report.md').write_text(report,encoding='utf-8')
    print(display.to_string(index=False))
    print(tally.to_string(index=False))
    print('Verified records:',verified)


if __name__ == '__main__':
    main()
