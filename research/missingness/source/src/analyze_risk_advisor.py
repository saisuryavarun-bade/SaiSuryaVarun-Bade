"""Audit adaptation scores and compare with saved matched-budget baselines."""
import json
import numpy as np
import pandas as pd
from pilot import ROOT,order_with_ties
from replication import retain,metrics
from risk_advisor_baseline import OUT,decomposition


def table(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='completed exploratory adaptation'
    d=pd.read_csv(OUT/'metrics.csv')
    fits=pd.read_csv(OUT/'fits.csv')
    assert len(d)==420 and len(fits)==30
    assert not d.duplicated(['dataset','seed','budget','input_variant','scoring','condition']).any()
    checked=0
    for fit in fits.itertuples():
        scores=pd.read_csv(OUT/f'{fit.dataset}_seed{fit.seed}'/f'n{fit.budget}_{fit.input_variant}_scores.csv')
        assert not scores.duplicated(['source_index','condition']).any()
        assert np.allclose(scores.published_risk,scores.prob_error+scores.total_entropy)
        assert np.allclose(scores.total_entropy,scores.aleatoric+scores.epistemic)
        for i,condition in enumerate(['independent_0.0','independent_0.1','independent_0.3',
                                     'block_a_0.1','block_a_0.3','block_b_0.1','block_b_0.3']):
            q=scores[scores.condition==condition]
            pred=(q.base_probability.to_numpy()>=.5).astype(int)
            for scoring in ['prob_error','published_risk']:
                keep=retain(q[scoring].to_numpy(),pred,fit.seed*4000+i,'global')
                got=metrics(q.label.to_numpy(),pred,keep)
                expected=d[(d.dataset==fit.dataset)&(d.seed==fit.seed)&(d.budget==fit.budget)&
                    (d.input_variant==fit.input_variant)&(d.scoring==scoring)&(d.condition==condition)].iloc[0]
                for metric in ['risk','balanced_risk','positive_retention','negative_retention']:
                    assert np.isclose(got[metric],expected[metric],equal_nan=True)
                checked+=1
    linear=pd.read_csv(ROOT/'outputs/sample_size_v03/metrics.csv')
    linear=linear[(linear.model=='native_boost_calibrated')&linear.draw.isin([0,-1])]
    raw=pd.read_csv(ROOT/'outputs/sample_size_v03/raw_confidence_reference.csv')
    raw=raw[raw.model=='native_boost_calibrated']
    joined=d.merge(raw[['dataset','seed','condition','risk','balanced_risk']],on=['dataset','seed','condition'],
        suffixes=['','_raw'])
    joined['delta_vs_raw']=joined.risk-joined.risk_raw
    joined['delta_balanced_vs_raw']=joined.balanced_risk-joined.balanced_risk_raw
    joined.to_csv(OUT/'raw_comparisons.csv',index=False)
    summary=joined.groupby(['dataset','budget','input_variant','scoring','condition']).agg(
        risk=('risk','mean'),risk_delta_vs_raw=('delta_vs_raw','mean'),
        balanced_delta_vs_raw=('delta_balanced_vs_raw','mean'),positive_retention=('positive_retention','mean')).reset_index()
    summary.to_csv(OUT/'seed_summaries.csv',index=False)
    keys=['dataset','seed','budget','scoring','condition']
    no=d[d.input_variant=='covariates'].set_index(keys)
    yes=d[d.input_variant=='covariates_mask'].set_index(keys)
    paired=pd.DataFrame(index=no.index)
    for col in ['risk','balanced_risk','positive_retention']:
        paired['delta_'+col]=yes[col]-no[col]
    paired.reset_index().to_csv(OUT/'mask_ablation.csv',index=False)
    selected=summary[summary.condition=='independent_0.3'][['dataset','budget','input_variant','scoring',
        'risk_delta_vs_raw','balanced_delta_vs_raw']].copy()
    selected.iloc[:,4:]*=100
    selected.columns=['Dataset','Unique rows','Inputs','Score','Error change vs raw pp','Balanced error change vs raw pp']
    full=summary[(summary.condition=='independent_0.3')&(((summary.dataset=='bank')&(summary.budget==9042))|
        ((summary.dataset=='wdbc')&(summary.budget==114)))].copy()
    comparison=[]
    for dataset,budget in [('bank',9042),('wdbc',114)]:
        for variant in ['confidence','count','mask']:
            a=linear[(linear.dataset==dataset)&(linear.budget==budget)&(linear.condition=='independent_0.3')&
                (linear.variant==variant)].risk.mean()
            comparison.append({'Dataset':dataset,'Method':'linear_'+variant,'Error percent':a*100})
        a=raw[(raw.dataset==dataset)&(raw.condition=='independent_0.3')].risk.mean()
        comparison.append({'Dataset':dataset,'Method':'raw_confidence','Error percent':a*100})
        for row in full[full.dataset==dataset].itertuples():
            comparison.append({'Dataset':dataset,'Method':row.input_variant+'_'+row.scoring,'Error percent':row.risk*100})
    report=f'''# Adapted published-baseline comparison

Completed 27 October 2026. This is an adaptation of the Risk Advisor ensemble and score, not reproduction of its reported experimental results.

## Implementation and verification

Inspected and pinned the author's implementation at commit `{manifest['source_commit']}`. Preserved read-only source files and its license in `references/risk_advisor`. The adaptation uses ten stochastic gradient-boosted members and the documented risk score: estimated error probability plus entropy of the ensemble-mean error distribution. Also reported error probability alone.

Passed entropy parity checks against SciPy and member-average checks against sklearn soft voting. Recomputed all {checked} condition-level metric records from saved scores and checked the entropy decomposition. Completed 30 ensemble fits, representing 300 gradient-boosting members, across two datasets, three seeds and matched labelled budgets for the calibrated base classifier.

## Explicit changes from the source experiment

We train the error model on held-out selector rows rather than base-classifier training errors, use a fixed published-range hyperparameter setting instead of the original tuning grid, add imputation for degraded inputs, and evaluate our fixed missingness conditions and data splits. These changes support matched exploratory comparisons but prohibit claiming a faithful reproduction or refutation of the published method. Adding masks is our ablation of the adaptation, not the original algorithm.

## Comparison against untrained confidence

The following uses 30 percent independent cell masking, 80 percent retained batch capacity, and descriptive means over three split seeds. Negative differences favor the adapted method. All other conditions are saved as well.

{table(selected)}

## Full-budget comparison with earlier linear probes

{table(pd.DataFrame(comparison))}

Methods differ in covariates and hypothesis family. Cross-family differences do not isolate mask value; the within-ensemble mask ablation in `mask_ablation.csv` is the controlled input comparison. Review class retention and balanced error alongside ordinary error.

## Evidence limits

These are two development datasets, overlapping split seeds, fixed masks, fixed ensemble parameters and one selected base-classifier configuration. The small WDBC sample has coarse error increments. Neither row counts nor ensemble members are independent scientific replications. There is no confirmatory test, sample-size guarantee, clinical validation or established novel algorithm.

Next: assess whether a distinct question remains after this comparator and closest-work audit, then use prospectively specified additional datasets and evaluation. No paper has been submitted or published.
'''
    (ROOT/'risk_advisor_report.md').write_text(report,encoding='utf-8')
    print(selected.to_string(index=False))
    print(pd.DataFrame(comparison).to_string(index=False))
    print('Audited records:',checked)


if __name__=='__main__':
    main()
