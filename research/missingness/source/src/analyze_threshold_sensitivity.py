"""Audit sensitivity choices and compare the two explicitly separate regimes."""
import json
import numpy as np
import pandas as pd
from pilot import ROOT
from replication import split_rows,retain,metrics
from robustness import cleaned_data,seed_base,CONDITIONS
from threshold_sensitivity import OUT,choose_threshold
from analyze_robustness import table


def main():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['status']=='completed threshold sensitivity'
    grid=pd.read_csv(OUT/'grid_metrics.csv');d=pd.read_csv(OUT/'tuned_metrics.csv')
    fits=pd.read_csv(OUT/'fits.csv');thresholds=pd.read_csv(OUT/'thresholds.csv')
    assert len(grid)==3456 and len(d)==864 and len(fits)==1152 and len(thresholds)==12
    checked=0
    for dataset in ['bank','wdbc','spambase','online_shoppers']:
        X,y,_=cleaned_data(dataset)
        for seed in [17,42,83]:
            _,sel,val,test=split_rows(y,seed)
            old=ROOT/'outputs/robustness_v06'/f'{dataset}_seed{seed}'
            with np.load(old/'meta_rep0.npz') as z:
                threshold,_=choose_threshold(y[val],z['validation_confidence'][:len(val),1])
            observed=thresholds[(thresholds.dataset==dataset)&(thresholds.seed==seed)].iloc[0]
            assert np.isclose(threshold,observed.threshold,atol=1e-12)
            directory=OUT/f'{dataset}_seed{seed}'
            for file in directory.glob('scores_rep*_n*.npz'):
                rep=int(file.stem.split('_')[1].replace('rep',''));budget=int(file.stem.split('_n')[-1])
                with np.load(file) as z:
                    assert np.array_equal(z['source_index'],X.index[test].to_numpy())
                    for variant in ['confidence','count','mask','placebo']:
                        group=fits[(fits.dataset==dataset)&(fits.seed==seed)&(fits.repetition==rep)&
                            (fits.budget==budget)&(fits.variant==variant)].sort_values('lambda_index')
                        li=int(group.iloc[np.argmin(group.validation_log_loss)].lambda_index)
                        assert li==int(z['selected_'+variant])
                        for ci,condition in enumerate(CONDITIONS):
                            pred=z[f'prediction_c{ci}'];score=z[f'l{li}_{variant}_c{ci}']
                            keep=retain(score,pred,seed_base(seed,rep)+90+ci,'global')
                            got=metrics(z['label'],pred,keep)
                            q=d[(d.dataset==dataset)&(d.seed==seed)&(d.repetition==rep)&(d.budget==budget)&
                                (d.variant==variant)&(d.condition==condition)].iloc[0]
                            for key in ['risk','balanced_risk','positive_retention','negative_retention']:
                                assert np.isclose(got[key],q[key],equal_nan=True)
                            assert np.isclose(pred[keep].mean(),q.retained_predicted_positive_fraction)
                            checked+=1
    effects=[]
    for regime,df in [('fixed_threshold_0.5',pd.read_csv(ROOT/'outputs/robustness_v06/tuned_metrics.csv')),
                      ('validation_balanced_threshold',d)]:
        keys=['dataset','seed','repetition','budget','condition']
        ref=df[df.variant=='confidence'].set_index(keys)
        for variant in ['count','mask','placebo']:
            a=df[df.variant==variant].set_index(keys)
            assert a.index.equals(ref.index)
            pair=pd.DataFrame(index=a.index)
            for metric in ['risk','balanced_risk','positive_retention']:
                pair['delta_'+metric]=a[metric]-ref[metric]
            pair['variant']=variant;pair['threshold_regime']=regime
            effects.append(pair.reset_index())
    paired=pd.concat(effects,ignore_index=True);paired.to_csv(OUT/'paired_regime_effects.csv',index=False)
    summary=paired.groupby(['dataset','budget','condition','variant','threshold_regime']).agg(
        mean_risk_delta=('delta_risk','mean'),mean_balanced_delta=('delta_balanced_risk','mean'),
        mean_positive_retention_delta=('delta_positive_retention','mean')).reset_index()
    summary.to_csv(OUT/'regime_summary.csv',index=False)
    shown=summary[(summary.variant=='mask')&(summary.condition=='independent')][[
        'dataset','budget','threshold_regime','mean_risk_delta','mean_balanced_delta']].copy()
    shown.iloc[:,3:]*=100
    shown.columns=['Dataset','Unique rows','Threshold regime','Error change pp','Balanced error change pp']
    composition=d[d.condition=='independent'].groupby(['dataset','budget','variant']).agg(
        predicted_positive_fraction=('retained_predicted_positive_fraction','mean'),risk=('risk','mean'),
        balanced_risk=('balanced_risk','mean')).reset_index()
    composition.to_csv(OUT/'retained_composition.csv',index=False)
    threshold_table=thresholds.groupby('dataset').agg(min_threshold=('threshold','min'),
        max_threshold=('threshold','max')).reset_index()
    report=f'''# Decision-threshold sensitivity

Completed 27 October 2026. Follow-up sensitivity motivated by a post-hoc retained-class diagnostic, not replacement of the original equal-error-cost primary regime.

## Procedure and checks

Reused v0.6's calibrated probability models, splits, masks, labelled subsets and feature inputs. Chose each base threshold from only clean validation probabilities and labels to maximize balanced accuracy, then froze it across corruptions. Recomputed all error-selector labels and refitted the fixed lambda grid with validation-only log-loss selection. The same validation partition chooses threshold and selector regularization, which adds tuning variance and uses labels beyond the counted selector-training budget.

Completed 1,152 selector fits, 3,456 grid records and 864 validation-selected records. Recomputed all {checked} selected records from saved scores, including class composition, and checked all threshold and lambda decisions against validation-only inputs.

## Threshold ranges across three splits

{table(threshold_table)}

## Full-mask versus learned-confidence comparison

Independent masking 0.3, 80 percent batch retention, each selector using its own validation-selected regularization. Means average three mask repetitions and three split seeds. Negative difference favors masks for that metric.

{table(shown)}

## Interpretation limits

Changing the decision threshold changes base predictions, selector error labels and the decision task's trade-off. It is not legitimate to choose whichever regime produces the desired headline. The original threshold regime is preserved and both must be explained. No new confidence intervals or multiplicity-adjusted tests are claimed here; v0.6 intervals do not apply to this phase.

Confidence feature inputs remain fixed to the earlier representation. In particular the raw max-probability ranking is not necessarily optimal for an altered threshold. The experiment therefore assesses sensitivity of these specified selectors, not all possible confidence functions or an optimal cost-sensitive system. New data, realistic costs and a completed closest-work assessment remain necessary before a final publication claim.

The detailed retained predicted-class fractions are in `retained_composition.csv`; the older all-negative diagnostic is in `retention_diagnostic.md`.
'''
    (ROOT/'threshold_sensitivity_report.md').write_text(report,encoding='utf-8')
    print(shown.to_string(index=False));print(threshold_table.to_string(index=False))
    print(composition[composition.variant.isin(['confidence','mask'])].to_string(index=False))
    print('Verified',checked,'selected records.')


if __name__=='__main__':main()
