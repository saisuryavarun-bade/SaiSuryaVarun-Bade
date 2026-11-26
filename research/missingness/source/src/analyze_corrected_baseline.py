"""Matched single-split baseline audit and report; not a split-averaged result."""
import json
import numpy as np
import pandas as pd
from pilot import ROOT
from replication import retain,metrics
from robustness import seed_base
from corrected_baseline import OUT
from analyze_robustness import table


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='completed corrected adaptation spot check'
    d=pd.read_csv(OUT/'metrics.csv');assert len(d)==32
    for row in d.itertuples():
        saved=pd.read_csv(OUT/f'{row.dataset}_n{row.budget}_{row.input_variant}_scores.csv')
        assert not saved.source_index.duplicated().any()
        assert np.allclose(saved.published_risk,saved.prob_error+saved.total_entropy)
        assert np.allclose(saved.total_entropy,saved.aleatoric+saved.epistemic)
        pred=(saved.base_probability.to_numpy()>=.5).astype(int)
        keep=retain(saved[row.scoring].to_numpy(),pred,seed_base(42,0)+90,'global')
        got=metrics(saved.label.to_numpy(),pred,keep)
        for key in ['risk','balanced_risk','positive_retention','negative_retention']:
            assert np.isclose(got[key],getattr(row,key),equal_nan=True)
    path=ROOT/'outputs/robustness_v06'
    raw=pd.read_csv(path/'raw_confidence.csv')
    raw=raw[(raw.seed==42)&(raw.repetition==0)&(raw.condition=='independent')]
    linear=pd.read_csv(path/'tuned_metrics.csv')
    linear=linear[(linear.seed==42)&(linear.repetition==0)&(linear.condition=='independent')]
    rows=[]
    for dataset,budget in d[['dataset','budget']].drop_duplicates().itertuples(index=False,name=None):
        r=raw[raw.dataset==dataset].iloc[0]
        rows.append({'dataset':dataset,'budget':budget,'method':'raw_confidence','risk':r.risk,
            'balanced_risk':r.balanced_risk,'positive_retention':r.positive_retention,'delta_vs_raw':0.})
        for s in linear[(linear.dataset==dataset)&(linear.budget==budget)&linear.variant.isin(['confidence','count','mask'])].itertuples():
            rows.append({'dataset':dataset,'budget':budget,'method':'tuned_'+s.variant,'risk':s.risk,
                'balanced_risk':s.balanced_risk,'positive_retention':s.positive_retention,'delta_vs_raw':s.risk-r.risk})
        for s in d[(d.dataset==dataset)&(d.budget==budget)].itertuples():
            rows.append({'dataset':dataset,'budget':budget,'method':s.input_variant+'_'+s.scoring,'risk':s.risk,
                'balanced_risk':s.balanced_risk,'positive_retention':s.positive_retention,'delta_vs_raw':s.risk-r.risk})
    comparison=pd.DataFrame(rows)
    comparison.to_csv(OUT/'matched_comparisons.csv',index=False)
    shown=comparison[comparison.method.isin(['raw_confidence','tuned_confidence','tuned_mask',
        'covariates_published_risk','covariates_mask_published_risk'])].copy()
    for col in ['risk','balanced_risk','positive_retention','delta_vs_raw']:shown[col]*=100
    shown.columns=['Dataset','Unique rows','Method','Error percent','Balanced error percent','Positive retention percent','Error change vs raw pp']
    report=f'''# Corrected-data published-baseline spot check

Completed 27 October 2026. Fixed split seed 42, masking repetition 0, independent masking rate 0.3 and 80 percent global batch retention. These are single-split, single-mask results; do not compare them numerically with the main three-split, three-mask means as if they were matched estimates.

## Scope

The same documented Risk Advisor adaptation was re-evaluated using corrected data on four tasks at small/full labelled budgets. It reuses exact v0.6 classifier models, subset rows and input masks. Sixteen ten-member ensembles produced 32 scoring records. All 32 records were recomputed from saved score files. Feature-vector partition isolation, base-output consistency and entropy decomposition were checked.

## Matched comparisons

{table(shown)}

The full table also retains error-probability-only scoring, count-only selectors and all class-retention values. Input covariates and ensemble hypothesis families differ from the linear probes, so across-family differences do not identify mask value. The covariates versus covariates-plus-mask comparison is the within-ensemble ablation.

## Limits

This replaces the old potentially overlapping banking split only at this spot-check scope. It does not establish the cleaning correction's effect by itself because the mask realization differs from v0.4. It is a fixed-setting adaptation, not reproduction of published benchmark results or a fully tuned baseline across multiple splits/masks. Results are exploratory; no superiority, formal risk control or publication claim follows.
'''
    (ROOT/'corrected_baseline_report.md').write_text(report,encoding='utf-8')
    print(shown.to_string(index=False));print('Verified all 32 baseline records.')


if __name__=='__main__':main()
