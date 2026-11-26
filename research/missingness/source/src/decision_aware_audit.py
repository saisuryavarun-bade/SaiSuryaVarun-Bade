"""No fitting: align the reference confidence with the actual predicted label."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from pilot import ROOT
from replication import retain,metrics
from robustness import CONDITIONS,seed_base
from analyze_robustness import table

OUT=ROOT/'outputs'/'decision_aware_v09'


def decision_error_probability(p,pred):
    return np.where(pred==1,1-p,p)


def main():
    p=np.array([.05,.15,.8]);pred=(p>=.1).astype(int)
    assert np.allclose(decision_error_probability(p,pred),[.05,.85,.2])
    assert not np.array_equal(np.argsort(np.minimum(p,1-p)),np.argsort(decision_error_probability(p,pred)))
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for directory in (ROOT/'outputs/robustness_v06').glob('*_seed*'):
        dataset,seed=directory.name.rsplit('_seed',1);seed=int(seed)
        newdir=ROOT/'outputs/threshold_sensitivity_v08'/directory.name
        for rep in range(3):
            with np.load(directory/f'meta_rep{rep}.npz') as z:
                probs=[z[f'eval_c{ci}_probability'].copy() for ci in range(3)]
            files=sorted(directory.glob(f'scores_rep{rep}_n*.npz'))
            assert files
            same_name=files[0].name
            with np.load(files[0]) as z:
                label=z['label'].copy();fixed=[z[f'prediction_c{ci}'].copy() for ci in range(3)]
            with np.load(newdir/same_name) as z:
                assert np.array_equal(label,z['label'])
                selected=[z[f'prediction_c{ci}'].copy() for ci in range(3)]
            for regime,predictions in [('fixed_0.5',fixed),('validation_balanced',selected)]:
                for ci,condition in enumerate(CONDITIONS):
                    pred=predictions[ci];p=probs[ci]
                    aware=decision_error_probability(p,pred);old=np.minimum(p,1-p)
                    if regime=='fixed_0.5':assert np.array_equal(aware,old)
                    ka=retain(aware,pred,seed_base(seed,rep)+90+ci,'global')
                    ko=retain(old,pred,seed_base(seed,rep)+90+ci,'global')
                    if regime=='fixed_0.5':assert np.array_equal(ka,ko)
                    rows.append({'dataset':dataset,'seed':seed,'repetition':rep,'condition':condition,'regime':regime,
                        **metrics(label,pred,ka),'retained_predicted_positive_fraction':float(pred[ka].mean()),
                        'old_raw_risk':metrics(label,pred,ko)['risk'],
                        'changed_retained_fraction':float(np.mean(ka!=ko))})
    reference=pd.DataFrame(rows);assert len(reference)==216
    reference.to_csv(OUT/'decision_aware_reference.csv',index=False)
    comparisons=[]
    for regime,path in [('fixed_0.5','robustness_v06'),('validation_balanced','threshold_sensitivity_v08')]:
        d=pd.read_csv(ROOT/'outputs'/path/'tuned_metrics.csv')
        d=d[d.variant.isin(['confidence','count','mask'])]
        r=reference[reference.regime==regime]
        j=d.merge(r[['dataset','seed','repetition','condition','risk','balanced_risk']],
            on=['dataset','seed','repetition','condition'],suffixes=['','_aware'])
        j['regime']=regime
        j['delta_vs_decision_aware']=j.risk-j.risk_aware
        j['delta_balanced_vs_decision_aware']=j.balanced_risk-j.balanced_risk_aware
        comparisons.append(j)
    comp=pd.concat(comparisons,ignore_index=True);assert len(comp)==1296
    comp.to_csv(OUT/'learned_vs_decision_aware.csv',index=False)
    summary=comp.groupby(['dataset','budget','variant','condition','regime']).agg(
        error_delta=('delta_vs_decision_aware','mean'),
        balanced_delta=('delta_balanced_vs_decision_aware','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    shown=summary[(summary.variant=='mask')&(summary.condition=='independent')].copy()
    shown=shown[['dataset','budget','regime','error_delta','balanced_delta']]
    shown.iloc[:,3:]*=100
    shown.columns=['Dataset','Unique labels','Decision regime','Mask error minus decision-aware pp','Balanced error difference pp']
    report=f'''# Decision-aware reference audit

Completed 27 October 2026. Standard baseline diagnostic, not a proposed algorithm.

The earlier raw confidence used min(p,1-p) as an error score. For a classifier with an arbitrary decision threshold, the probability assigned to an incorrect actual prediction is instead p for a negative prediction and 1-p for a positive prediction. The two scores coincide exactly at threshold 0.5. Probability calibration under degradation remains an assumption; no actual-risk guarantee follows.

Computed 216 dataset/split/mask/condition/regime reference records and 1,296 matched learned-selector comparisons without fitting any model. Verified exact old/new score and retention equality throughout the 0.5 regime, and tested an explicit nonstandard-threshold counterexample.

## Full-mask comparison at independent masking 0.3 and 80 percent retention

{table(shown)}

These differences compare an existing learned full-mask selector to the new untrained reference, rather than the earlier learned confidence-only comparator. Negative values favor masks on that metric. Label budgets refer to the learned selector, while the reference has no fitted error-selector parameters. All class metrics and conditions are preserved in the output tables.

## Consequence for the proposed paper

The v0.8 threshold-sensitivity analysis held its earlier confidence representation fixed; it did not establish improvement over decision-aware confidence. A final comparison would need decision-aligned learned features for both confidence-only and mask variants, plus a justified loss objective. The construction used here is established, including in definitions of a fixed classifier's correctness posterior, and must not be claimed as new.

This diagnostic constrains our interpretation. It cannot turn an exploratory result into a novel contribution, establish utility of human review, or validate hypothetical task costs. Original results remain recorded rather than replaced with whichever baseline yields a preferred headline.
'''
    (ROOT/'decision_aware_report.md').write_text(report,encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'status':'completed decision-aware audit',
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plan_sha256':hashlib.sha256((ROOT/'decision_aware_audit_plan.md').read_bytes()).hexdigest(),
        'no_new_fitting':True,'reference_records':216,'comparison_records':1296},indent=2),encoding='utf-8')
    print(shown.to_string(index=False))


if __name__=='__main__':main()
