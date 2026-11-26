"""Rebuild selected outcomes from saved scores without experiment metric helpers."""
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pilot import ROOT

OUT = ROOT/'outputs'/'decision_aligned_v10'
CONDITIONS = ['independent','block_a','block_b']


def rebuild(y, h, score, ci):
    assert np.isfinite(score).all()
    # Same specified seeded tie rule, reconstructed without retain()/metrics().
    jitter = np.random.default_rng(1_000_000+42*10000+90+ci).random(len(y))
    indices = np.lexsort((jitter,score))[:int(np.ceil(.8*len(y)))]
    kept = np.zeros(len(y),dtype=bool)
    kept[indices] = True
    tn,fp,fn,tp = [int(np.sum(kept & (y==a) & (h==p))) for a,p in [(0,0),(0,1),(1,0),(1,1)]]
    neg, pos = tn+fp, fn+tp
    return {'tn':tn,'fp':fp,'fn':fn,'tp':tp,'retained_n':len(indices),
        'coverage':len(indices)/len(y),'risk':(fp+fn)/len(indices),
        'balanced_risk':.5*(fp/neg+fn/pos) if neg and pos else np.nan,
        'positive_retention':pos/np.sum(y==1),'negative_retention':neg/np.sum(y==0)}


def main():
    m = pd.read_csv(OUT/'selected_metrics.csv')
    fits = pd.read_csv(OUT/'fits.csv')
    counts = pd.read_csv(OUT/'label_accounting.csv')
    checked, selected_checks = 0,0
    for dataset in m.dataset.unique():
        for regime in ['fixed_0.5','validation_balanced']:
            group = m[(m.dataset==dataset)&(m.regime==regime)]
            budgets = sorted(group.loc[group.budget>0,'budget'].unique())
            subsets = []
            for budget in budgets:
                with np.load(OUT/f'{dataset}_{regime}_n{budget}.npz') as z:
                    y = z['label']; threshold = float(z['threshold'])
                    source = z['fit_source_index']; validation = z['validation_source_index']
                    assert len(set(source)) == budget
                    assert not set(source)&set(validation)
                    assert not set(source)&set(z['source_index'])
                    assert not set(validation)&set(z['source_index'])
                    subsets.append(set(source))
                    for variant in ['confidence','count','mask','placebo']:
                        f = fits[(fits.dataset==dataset)&(fits.regime==regime)&(fits.budget==budget)&(fits.variant==variant)].sort_values('lambda_index')
                        assert len(f)==4
                        losses = []
                        for li in range(4):
                            p = np.clip(z[f'{variant}_l{li}_validation'],np.finfo(float).eps,1-np.finfo(float).eps)
                            labels = z['validation_error']
                            loss = -np.mean(labels*np.log(p)+(1-labels)*np.log1p(-p))
                            assert np.isclose(loss,f.iloc[li].validation_log_loss,rtol=1e-10,atol=1e-12)
                            losses.append(loss)
                        selected = int(np.argmin(losses))
                        assert selected == int(z[f'{variant}_selected'])
                        selected_checks += 1
                        for ci,condition in enumerate(CONDITIONS):
                            h,p = z[f'h_c{ci}'],z[f'p_c{ci}']
                            assert np.array_equal(h,(p>=threshold).astype(int))
                            row = group[(group.budget==budget)&(group.variant==variant)&(group.condition==condition)].iloc[0]
                            assert int(row.lambda_index) == selected
                            computed = rebuild(y,h,z[f'{variant}_l{selected}_c{ci}'],ci)
                            for key,value in computed.items():
                                assert np.isclose(row[key],value,equal_nan=True), (dataset,regime,budget,variant,key)
                            checked += 1
                    if budget == budgets[0]:
                        for ci,condition in enumerate(CONDITIONS):
                            h,p = z[f'h_c{ci}'],z[f'p_c{ci}']
                            computed = rebuild(y,h,np.where(h==1,1-p,p),ci)
                            row = group[(group.budget==0)&(group.condition==condition)].iloc[0]
                            for key,value in computed.items():
                                assert np.isclose(row[key],value,equal_nan=True)
                            checked += 1
                    lc = counts[(counts.dataset==dataset)&(counts.regime==regime)&(counts.budget==budget)]
                    assert (lc.unique_post_classifier_labels==budget+len(validation)).all()
                    assert (lc.tuning_labels==len(validation)).all()
            assert subsets[0].issubset(subsets[1])
    assert checked==216 and selected_checks==64
    refs = m[m.variant=='decision_aware'][['dataset','regime','condition','risk','balanced_risk']]
    learned = m[m.variant!='decision_aware'].merge(refs,on=['dataset','regime','condition'],suffixes=['','_aware'])
    conf = learned[learned.variant=='confidence'][['dataset','regime','condition','budget','risk']]
    learned = learned.merge(conf,on=['dataset','regime','condition','budget'],suffixes=['','_learned_confidence'])
    learned['delta_vs_aware_pp'] = 100*(learned.risk-learned.risk_aware)
    learned['delta_vs_learned_confidence_pp'] = 100*(learned.risk-learned.risk_learned_confidence)
    learned.to_csv(OUT/'paired_comparisons.csv',index=False)
    shown = learned[(learned.variant=='mask')&(learned.condition=='independent')][
        ['dataset','regime','budget','risk','risk_aware','risk_learned_confidence','delta_vs_aware_pp','delta_vs_learned_confidence_pp']].copy()
    shown[['risk','risk_aware','risk_learned_confidence']] *= 100
    shown.to_csv(OUT/'independent_mask_summary.csv',index=False)
    audit = {'status':'passed score reconstruction','checked_selected_records':checked,
        'checked_validation_choices':selected_checks,'checked_candidate_validation_losses':256,
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'auditor_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))
    print(shown.to_string(index=False))


if __name__=='__main__':
    main()
