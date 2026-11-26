"""Score audit, validation-choice checks and paired row/mask bootstrap."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import ROOT
from replication import retain,metrics,split_rows
from robustness import OUT,LAMBDAS,VARIANTS,CONDITIONS,seed_base,cleaned_data,check_feature_isolation


def table(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def paired_bootstrap(labels,preds,a,b,draws=400,seed=60642):
    """Paired source-row sample is shared across repeated corruption conditions."""
    rng=np.random.default_rng(seed)
    out=[]
    for sample in range(draws):
        repetitions=rng.integers(0,len(a),len(a))
        ix=rng.integers(0,len(labels),len(labels))
        effects=[]
        for j,rep in enumerate(repetitions):
            tie_seed=seed+sample*10+j
            ka=retain(a[rep][ix],preds[rep][ix],tie_seed,'global')
            kb=retain(b[rep][ix],preds[rep][ix],tie_seed,'global')
            ma=metrics(labels[ix],preds[rep][ix],ka)
            mb=metrics(labels[ix],preds[rep][ix],kb)
            effects.append([ma['risk']-mb['risk'],ma['balanced_risk']-mb['balanced_risk']])
        out.append(np.nanmean(effects,axis=0))
    return np.asarray(out)


def checks():
    y=np.array([0,0,1,1,0,1,0,1])
    pred=[y.copy(),1-y,y.copy()]
    scores=[np.arange(8.),np.arange(8.)[::-1],np.zeros(8)]
    z=paired_bootstrap(y,pred,scores,scores,draws=20,seed=1)
    assert np.nanmax(np.abs(z))==0


def main():
    checks()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='completed exploratory robustness'
    grid=pd.read_csv(OUT/'grid_metrics.csv')
    tuned=pd.read_csv(OUT/'tuned_metrics.csv')
    fits=pd.read_csv(OUT/'selector_fits.csv')
    assert len(grid)==3456 and len(tuned)==864 and len(fits)==1152
    keys=['dataset','seed','repetition','budget','variant','lambda_index','condition']
    assert not grid.duplicated(keys).any()
    assert not grid[['risk','coverage','positive_retention','negative_retention']].isna().any().any()
    for key,group in fits.groupby(['dataset','seed','repetition','budget','variant']):
        expected=group.sort_values('lambda_index').sort_values('validation_log_loss',kind='stable').iloc[0].lambda_index
        matched=tuned[(tuned.dataset==key[0])&(tuned.seed==key[1])&(tuned.repetition==key[2])&
            (tuned.budget==key[3])&(tuned.variant==key[4])]
        assert (matched.lambda_index==expected).all() and len(matched)==3
    audited=0
    for dataset in ['bank','wdbc','spambase','online_shoppers']:
        X,y,_=cleaned_data(dataset)
        for seed in [17,42,83]:
            check_feature_isolation(X,split_rows(y,seed))
            subdir=OUT/f'{dataset}_seed{seed}'
            for file in subdir.glob('scores_rep*_n*.npz'):
                rep=int(file.stem.split('_')[1].replace('rep',''))
                budget=int(file.stem.split('_n')[-1])
                with np.load(file) as cache:
                    for li in range(4):
                        for variant in VARIANTS:
                            # Audit all independent-mask results; other conditions remain serialized.
                            ci=0
                            score=cache[f'l{li}_{variant}_c{ci}']
                            pred=cache['prediction_c0']
                            keep=retain(score,pred,seed_base(seed,rep)+90,'global')
                            got=metrics(cache['label'],pred,keep)
                            q=grid[(grid.dataset==dataset)&(grid.seed==seed)&(grid.repetition==rep)&
                                (grid.budget==budget)&(grid.variant==variant)&(grid.lambda_index==li)&
                                (grid.condition=='independent')].iloc[0]
                            for metric in ['risk','balanced_risk','positive_retention','negative_retention','coverage']:
                                assert np.isclose(got[metric],q[metric],equal_nan=True)
                            audited+=1
    compare=[]
    for policy,df in [('fixed_0.01',grid[grid.lambda_index==1]),('validation_selected',tuned)]:
        index=['dataset','seed','repetition','budget','condition']
        reference=df[df.variant=='confidence'].set_index(index)
        for variant in ['count','mask','placebo']:
            a=df[df.variant==variant].set_index(index)
            assert a.index.equals(reference.index)
            p=pd.DataFrame(index=a.index)
            for col in ['risk','balanced_risk','positive_retention']:
                p['delta_'+col]=a[col]-reference[col]
            p['variant']=variant;p['policy']=policy
            compare.append(p.reset_index())
    pair=pd.concat(compare,ignore_index=True)
    pair.to_csv(OUT/'paired_effects.csv',index=False)
    # Independent mask draws are averaged inside each split before the split summary.
    split_summary=pair.groupby(['dataset','seed','budget','condition','variant','policy']).agg(
        risk_delta=('delta_risk','mean'),balanced_delta=('delta_balanced_risk','mean'),
        positive_retention_delta=('delta_positive_retention','mean')).reset_index()
    summary=split_summary.groupby(['dataset','budget','condition','variant','policy']).agg(
        mean_risk_delta=('risk_delta','mean'),min_risk_delta=('risk_delta','min'),max_risk_delta=('risk_delta','max'),
        mean_balanced_delta=('balanced_delta','mean'),mean_positive_retention_delta=('positive_retention_delta','mean')).reset_index()
    split_summary.to_csv(OUT/'within_split_summary.csv',index=False)
    summary.to_csv(OUT/'seed_summaries.csv',index=False)
    intervals=[]
    for dataset in ['bank','wdbc','spambase','online_shoppers']:
        subdir=OUT/f'{dataset}_seed42'
        budgets=sorted(grid[grid.dataset==dataset].budget.unique())
        for budget in budgets:
            print(f'Bootstrap {dataset} n={budget}',flush=True)
            reps=[]
            for rep in range(3):
                with np.load(subdir/f'scores_rep{rep}_n{budget}.npz') as z:
                    reps.append({k:z[k].copy() for k in z.files})
            for policy in ['fixed_0.01','validation_selected']:
                for variant in ['count','mask']:
                    a=[];b=[];pred=[]
                    for r in reps:
                        ai=1 if policy=='fixed_0.01' else int(r['selected_'+variant])
                        bi=1 if policy=='fixed_0.01' else int(r['selected_confidence'])
                        a.append(r[f'l{ai}_{variant}_c0']);b.append(r[f'l{bi}_confidence_c0'])
                        pred.append(r['prediction_c0'])
                    samples=paired_bootstrap(reps[0]['label'],pred,a,b)
                    estimate=pair[(pair.dataset==dataset)&(pair.seed==42)&(pair.budget==budget)&
                        (pair.condition=='independent')&(pair.variant==variant)&(pair.policy==policy)]
                    intervals.append({'dataset':dataset,'seed':42,'budget':int(budget),'policy':policy,'variant':variant,
                        'risk_delta':estimate.delta_risk.mean(),'risk_low':np.nanquantile(samples[:,0],.025),
                        'risk_high':np.nanquantile(samples[:,0],.975),
                        'balanced_delta':estimate.delta_balanced_risk.mean(),
                        'balanced_low':np.nanquantile(samples[:,1],.025),'balanced_high':np.nanquantile(samples[:,1],.975),
                        'resamples':400,'corruption_draws':3})
    ci=pd.DataFrame(intervals)
    ci.to_csv(OUT/'conditional_bootstrap.csv',index=False)
    display=summary[(summary.condition=='independent')&(summary.variant=='mask')][[
        'dataset','budget','policy','mean_risk_delta','mean_balanced_delta','mean_positive_retention_delta']].copy()
    display.iloc[:,3:]*=100
    display.columns=['Dataset','Unique rows','Penalty policy','Error change pp','Balanced error change pp','Positive retention change pp']
    choices=tuned[tuned.condition=='independent'].groupby(['variant','lambda']).size().unstack(fill_value=0)
    choices.to_csv(OUT/'validation_lambda_choices.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,dataset in zip(axes.flat,['bank','wdbc','spambase','online_shoppers']):
        for policy,color in [('fixed_0.01','#d95f02'),('validation_selected','#1b9e77')]:
            q=summary[(summary.dataset==dataset)&(summary.condition=='independent')&
                (summary.variant=='mask')&(summary.policy==policy)].sort_values('budget')
            x=q.budget.to_numpy()
            ax.plot(x,q.mean_risk_delta.to_numpy()*100,'o-',label=policy,color=color)
            ax.fill_between(x,q.min_risk_delta.to_numpy()*100,q.max_risk_delta.to_numpy()*100,color=color,alpha=.15)
        ax.axhline(0,color='black',linewidth=.8);ax.set_xscale('log')
        ax.set_title(dataset);ax.set_xlabel('Unique labelled selector rows');ax.set_ylabel('Mask minus confidence error (pp)')
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Corrected data, 3 split seeds, 3 mask draws: 80% retained\nShading is the split range, not a confidence interval.')
    fig.tight_layout();fig.savefig(OUT/'regularization_check.png',dpi=180);plt.close(fig)
    report=f'''# Cleaning correction and robustness analysis

Completed 27 October 2026. Follow-up exploratory analysis; no untouched final test claim.

## Corrections and isolation

Removed 16 duplicate feature vectors from Bank Marketing after excluding duration, leaving 45,195 rows. Verified feature-hash isolation across every partition pair for all four datasets and three split seeds. Original banking results remain historical; the updated results do not isolate cleaning alone because mask draws and analysis settings also differ. The earlier adapted published-baseline comparison still needs a corrected banking rerun.

## Experiment

Four datasets, three split seeds, three independently regenerated masking repetitions, two source-row budgets, four lambda values and four selector feature sets. Completed 1,152 selector fits, 3,456 grid metric records and 864 validation-selected metric records. Recomputed {audited} independent-mask records from serialized scores and verified all validation-only penalty selections. Constant fallbacks: {int(fits.constant_fallback.sum())}.

The regularization grid was fixed before the run. Each selector chose lambda using validation error-prediction log loss, not evaluation retained error. All grid values, selected choices and raw-confidence references remain saved. The calibrated base classifier was fitted once per dataset/split and reused across selector settings and mask repetitions.

## Main descriptive condition

Independent cell masking 0.3; 80 percent global batch retention. Compare full masks against learned confidence at the same budget, either with fixed lambda 0.01 or each selector's own validation-selected lambda. Means average mask repetitions within each split, then average the three splits. Lower error difference favors masks.

{table(display)}

Balanced error and class retention remain co-reported. Mask penalties that disappear after tuning must not be described as regularization-robust. A validation-choice result is still subject to finite validation sample variation and does not establish the evaluation-best or globally optimal setting.

## Conditional uncertainty

`conditional_bootstrap.csv` reports split-42 paired percentile intervals for mask/count versus confidence, under both penalty policies and both budgets. Four hundred resamples jointly resample source rows and the three evaluated mask repetitions, reranking each paired batch. The same source-row resample is shared across sampled repetitions to preserve cross-mask dependence. The identity-method fixture returns zero differences.

These intervals condition on the fitted base model, development labels and fitted selectors. They omit training/split uncertainty and do not justify population-wide or multiplicity-adjusted significance claims. Three mask draws provide limited Monte Carlo information. Split ranges are descriptive because partitions overlap.

## Scientific status

This run strengthens the controls and corrects an identified leakage risk; it does not complete independent code review or establish novelty. The appropriate paper claim must be revised to match these results. The published-baseline adaptation and literature gaps remain separate obligations before submission. No submission or publication has occurred.
'''
    (ROOT/'robustness_report.md').write_text(report,encoding='utf-8')
    print(display.to_string(index=False));print(choices.to_string())
    print('Audited records:',audited)


if __name__=='__main__':main()
