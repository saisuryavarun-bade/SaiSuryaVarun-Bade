"""Audit and summarize the prospectively specified added-dataset extension."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import ROOT
from replication import retain,metrics
from external_validation import OUT,load_extra


def table(df):
    s=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        s.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(s)


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='completed prospective extension'
    d=pd.read_csv(OUT/'metrics.csv')
    fits=pd.read_csv(OUT/'fits.csv')
    assert len(d)==1008 and len(fits)==144
    keys=['dataset','seed','model','budget','variant','condition']
    assert not d.duplicated(keys).any()
    assert not d[['risk','positive_retention','negative_retention','coverage']].isna().any().any()
    checked=0
    conditions=['independent_0.0','independent_0.1','independent_0.3','block_a_0.1','block_a_0.3','block_b_0.1','block_b_0.3']
    for dataset in ['spambase','online_shoppers']:
        X,y,source=load_extra(dataset)
        assert source['clean_table_sha256']==manifest['datasets'][dataset]['clean_table_sha256']
        for seed in [17,42,83]:
            subdir=OUT/f'{dataset}_seed{seed}'
            split=json.loads((subdir/'splits.json').read_text())
            assert len(set(sum(split.values(),[])))==sum(map(len,split.values()))==len(X)
            for model in ['logistic','native_boost_calibrated']:
                saved=pd.read_csv(subdir/f'{model}_scores.csv')
                assert not saved.duplicated(['source_index','budget','variant','condition']).any()
                ids_by_budget=[]
                for path in subdir.glob(f'{model}_n*_source_ids.json'):
                    ids_by_budget.append(set(json.loads(path.read_text())))
                ids_by_budget.sort(key=len)
                assert all(a<=b for a,b in zip(ids_by_budget,ids_by_budget[1:]))
                for (budget,variant,condition),q in saved.groupby(['budget','variant','condition'],sort=False):
                    i=conditions.index(condition)
                    keep=retain(q.score.to_numpy(),q.prediction.to_numpy(),seed*4000+i,'global')
                    expected=metrics(q.label.to_numpy(),q.prediction.to_numpy(),keep)
                    observed=d[(d.dataset==dataset)&(d.seed==seed)&(d.model==model)&(d.budget==budget)&
                        (d.variant==variant)&(d.condition==condition)].iloc[0]
                    for key in ['risk','balanced_risk','positive_retention','negative_retention','coverage']:
                        assert np.isclose(expected[key],observed[key],equal_nan=True)
                    checked+=1
    pairkeys=['dataset','seed','model','budget','condition']
    ref=d[d.variant=='confidence'].set_index(pairkeys)
    paired=[]
    for variant in ['count','mask','placebo']:
        a=d[d.variant==variant].set_index(pairkeys)
        assert a.index.equals(ref.index)
        frame=pd.DataFrame(index=a.index)
        for key in ['risk','balanced_risk','positive_retention','negative_retention']:
            frame['delta_'+key]=a[key]-ref[key]
        frame['variant']=variant
        paired.append(frame.reset_index())
    pair=pd.concat(paired)
    pair.to_csv(OUT/'paired_effects.csv',index=False)
    summary=pair.groupby(['dataset','model','budget','condition','variant']).agg(
        mean_risk_delta=('delta_risk','mean'),min_risk_delta=('delta_risk','min'),max_risk_delta=('delta_risk','max'),
        mean_balanced_delta=('delta_balanced_risk','mean'),mean_positive_retention_delta=('delta_positive_retention','mean')).reset_index()
    summary.to_csv(OUT/'seed_summaries.csv',index=False)
    raw=pd.read_csv(OUT/'raw_confidence.csv')
    joined=d.merge(raw[['dataset','seed','model','condition','risk']],on=['dataset','seed','model','condition'],suffixes=['','_raw'])
    joined['delta_vs_raw']=joined.risk-joined.risk_raw
    joined.to_csv(OUT/'raw_comparisons.csv',index=False)
    main=summary[(summary.condition=='independent_0.3')&(summary.variant=='mask')].copy()
    display=main[['dataset','model','budget','mean_risk_delta','mean_balanced_delta','mean_positive_retention_delta']].copy()
    display.iloc[:,3:]*=100
    display.columns=['Dataset','Classifier','Unique rows','Error change pp','Balanced error change pp','Positive retention change pp']
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    colors={'count':'#1b9e77','mask':'#d95f02','placebo':'#7570b3'}
    for i,dataset in enumerate(['spambase','online_shoppers']):
        for j,model in enumerate(['logistic','native_boost_calibrated']):
            ax=axes[i,j]
            for variant in ['count','mask','placebo']:
                q=summary[(summary.dataset==dataset)&(summary.model==model)&
                    (summary.condition=='independent_0.3')&(summary.variant==variant)].sort_values('budget')
                x=q.budget.to_numpy()
                ax.plot(x,q.mean_risk_delta.to_numpy()*100,'o-',color=colors[variant],label=variant)
                ax.fill_between(x,q.min_risk_delta.to_numpy()*100,q.max_risk_delta.to_numpy()*100,color=colors[variant],alpha=.12)
            ax.axhline(0,color='black',linewidth=.8)
            ax.set_xscale('log')
            ax.set_title(f'{dataset}: {model}')
            ax.set_xlabel('Unique labelled selector rows')
            ax.set_ylabel('Error difference vs confidence (pp)')
            ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Two added datasets: 30% cell masking, 80% retained\nLines: 3-seed means; shading: seed range, not confidence intervals.')
    fig.tight_layout()
    fig.savefig(OUT/'additional_learning_curves.png',dpi=180)
    plt.close(fig)
    data_table=pd.DataFrame([{'Dataset':name,'Original rows':v['original_rows'],'Analysed rows':v['rows'],
        'Duplicates removed':v['duplicates_after_exclusions'],'Conflicting rows removed':v['conflicting_feature_rows_removed'],
        'Missing cells':v['parsed_missing_cells']} for name,v in manifest['datasets'].items()])
    report=f'''# Additional dataset validation

Completed 27 October 2026 under the plan written before inspecting these outcome results. This is an additional-data extension of an exploratory study, not a publicly preregistered final validation.

## Data audit

{table(data_table)}

Source: UCI Spambase, https://doi.org/10.24432/C53G6X; Online Shoppers Purchasing Intention, https://doi.org/10.24432/C5F88Q. Both are CC BY 4.0. Archive hashes, cleaned-table hashes, exclusions and feature lists are preserved in the output manifest. Spambase's parsed missingness is recorded independently of the repository's metadata flag. PageValues is excluded from the shopping data due to prediction-time availability ambiguity; this remains a session-level benchmark, not early intervention evidence.

## Experiment and checks

Two additional datasets, three fixed split seeds, two base-classifier configurations, three unique-source-row budgets and four linear selector variants. Completed 144 selector fits and 1,008 setting-level metric records. Recomputed all {checked} records from saved scores; checked source-row split separation, nested subsets and duplicate handling. Constant fallback fits: {int(fits.constant_fallback.sum())}.

## Prespecified primary descriptive condition

At 30 percent independent cell masking and 80 percent global batch retention, full-mask minus learned-confidence results were:

{table(display)}

Negative error change favors full masks. Negative positive-retention change means more actual positive cases are referred. All other condition results, the count-only and shuffled-mask controls, and raw-confidence comparisons are saved. A result under this condition is not a claim across all missingness processes.

## Interpretation

These datasets broaden the tasks beyond banking and a small diagnostic sample. They still share benchmark-style artificial corruptions and arbitrary feature-block groups, and have no guaranteed relevance to real operational outages. Three overlapping seed splits do not provide independent population replications. The next scientific conclusion must jointly consider ordinary error, class-conditioned error and class retention rather than choose whichever metric favors a method.

Published-baseline adaptation is reported separately in `risk_advisor_report.md`; it was not run on these two added datasets. A manuscript must not imply every comparator covers every dataset or that every phase used identical budgets. The initial exploratory phases, hypothesis refinement, and prospective added-data extension should be described separately.

No submission, publication, formal guarantee or universal sample-budget recommendation follows from these results.
'''
    (ROOT/'external_validation_report.md').write_text(report,encoding='utf-8')
    print(data_table.to_string(index=False))
    print(display.to_string(index=False))
    print('Verified',checked,'records')


if __name__=='__main__':main()
