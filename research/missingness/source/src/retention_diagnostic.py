"""Predicted-class composition diagnostic for the corrected primary analysis."""
import numpy as np
import pandas as pd
from pilot import ROOT
from robustness import OUT,CONDITIONS,VARIANTS,seed_base
from replication import retain
from analyze_robustness import table


def main():
    results=[]
    for directory in OUT.glob('*_seed*'):
        dataset,seed=directory.name.rsplit('_seed',1);seed=int(seed)
        for file in directory.glob('scores_rep*_n*.npz'):
            rep=int(file.stem.split('_')[1].replace('rep',''));budget=int(file.stem.split('_n')[-1])
            with np.load(file) as z:
                y=z['label']
                for ci,condition in enumerate(CONDITIONS):
                    pred=z[f'prediction_c{ci}'];p=z[f'probability_c{ci}']
                    scores={'raw_confidence':np.minimum(p,1-p)}
                    for variant in ['confidence','count','mask']:
                        li=int(z['selected_'+variant])
                        scores['tuned_'+variant]=z[f'l{li}_{variant}_c{ci}']
                    for method,score in scores.items():
                        keep=retain(score,pred,seed_base(seed,rep)+90+ci,'global')
                        tp=int((keep&(y==1)&(pred==1)).sum())
                        fp=int((keep&(y==0)&(pred==1)).sum())
                        fn=int((keep&(y==1)&(pred==0)).sum())
                        tn=int((keep&(y==0)&(pred==0)).sum())
                        assert tp+fp+fn+tn==keep.sum()
                        results.append({'dataset':dataset,'seed':seed,'repetition':rep,'budget':budget,
                            'condition':condition,'method':method,'retained_n':int(keep.sum()),
                            'retained_tp':tp,'retained_fp':fp,'retained_fn':fn,'retained_tn':tn,
                            'predicted_positive_fraction':(tp+fp)/keep.sum(),
                            'all_retained_predictions_negative':tp+fp==0,
                            'retained_positive_prevalence':(tp+fn)/keep.sum(),
                            'full_predicted_positive_fraction':float(pred.mean()),
                            'full_actual_positive_prevalence':float(y.mean())})
    d=pd.DataFrame(results);assert len(d)==864
    d.to_csv(OUT/'predicted_class_diagnostic.csv',index=False)
    s=d[d.condition=='independent'].groupby(['dataset','budget','method']).agg(
        settings=('seed','size'),all_negative_settings=('all_retained_predictions_negative','sum'),
        mean_predicted_positive_fraction=('predicted_positive_fraction','mean'),
        mean_actual_positive_prevalence=('retained_positive_prevalence','mean')).reset_index()
    s.to_csv(OUT/'predicted_class_summary.csv',index=False)
    shown=s[s.method.isin(['raw_confidence','tuned_confidence','tuned_mask'])].copy()
    for col in ['mean_predicted_positive_fraction','mean_actual_positive_prevalence']:shown[col]*=100
    shown.columns=['Dataset','Budget','Method','Split/mask settings','All-negative settings','Predicted positive percent','Actual positive percent']
    report=f'''# Retained predicted-class composition

27 October 2026. Post-hoc interpretation diagnostic, not a newly selected primary endpoint.

Computed the retained confusion counts for raw confidence and validation-tuned confidence/count/mask selectors across the corrected v0.6 splits, mask repetitions and conditions. Counts reconcile to retained batch size. These are correlated setting counts, not independent replications.

## Independent masking 0.3, 80 percent batch retention

{table(shown)}

## Interpretation

When all retained predictions are negative, the ordinary retained error equals the actual positive prevalence in that retained subset. If both actual classes are retained, balanced error is exactly one half: false-positive rate is zero and false-negative rate is one. A change in ordinary error can then be explained by which actual positives are referred, rather than demonstrating correct automatic positive decisions.

This is not automatically a defect under equal-cost ordinary selective risk: referring likely positive cases may be a legitimate policy for that narrow objective. It does prevent claiming broad practical utility, improved minority-class decisions or successful human review from ordinary error alone. Actual costs, acceptable class-specific workloads and the base classification threshold would need a prospectively designed study.

This diagnostic supports retaining class-composition and confusion counts in the paper. Do not silently optimize a new decision threshold against the existing evaluation labels to rescue a preferred result. A threshold/cost sensitivity analysis, if pursued, needs its own recorded design and validation-only tuning.
'''
    (ROOT/'retention_diagnostic.md').write_text(report,encoding='utf-8')
    print(shown.to_string(index=False))


if __name__=='__main__':main()
