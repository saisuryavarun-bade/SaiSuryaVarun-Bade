"""Independent source-allocation, tuning, scaling and retained-score audit."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import ROOT
from analyze_decision_aligned_smoke import rebuild

OUT=ROOT/'outputs/total_label_budget_v11'
VARIANTS=['confidence','count','mask','placebo']
CONDITIONS=['independent','block_a','block_b']


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='completed exploratory total-label-budget check'
    assert hashlib.sha256((ROOT/'src/total_label_budget.py').read_bytes()).hexdigest()==manifest['script_sha256']
    m=pd.read_csv(OUT/'selected_metrics.csv')
    f=pd.read_csv(OUT/'fits.csv')
    accounts=pd.read_csv(OUT/'label_accounting.csv')
    checked,choices,losses_checked=0,0,0
    for di,dataset in enumerate(['bank','wdbc','spambase','online_shoppers']):
        old=ROOT/'outputs/robustness_v06'/f'{dataset}_seed42'
        assert hashlib.sha256((old/'meta_rep0.npz').read_bytes()).hexdigest()==manifest['datasets'][dataset]['cache_sha256']
        with np.load(old/'meta_rep0.npz') as z:
            source=z['source_indices'].copy();conf=z['development_confidence'].copy()
            old_errors=z['error'].copy()
            extra={v:z['development_'+v][:,3:].copy() for v in VARIANTS if v!='confidence'}
        splits=json.loads((old/'splits.json').read_text())
        assert len(source)==len(splits['selector'])
        allocation=json.loads((OUT/f'{dataset}_allocation.json').read_text())
        order=np.random.default_rng(20261027+di).permutation(len(source))
        assert np.array_equal(order,allocation['position_order'])
        assert np.array_equal(source[order],allocation['source_order'])
        prior_fit,prior_tune=set(),set()
        reference_scores=None
        for budget in allocation['budgets']:
            with np.load(OUT/f'{dataset}_B{budget}.npz') as z:
                positions=order[:budget]
                tune=(np.arange(budget)+1)%5==0
                fp,tp=positions[~tune],positions[tune]
                assert np.array_equal(fp,z['fit_position']) and np.array_equal(tp,z['tuning_position'])
                fitids,tuneids=set(z['fit_source_index']),set(z['tuning_source_index'])
                assert len(fitids)==budget-budget//5 and len(tuneids)==budget//5
                assert fitids==set(source[fp]) and tuneids==set(source[tp])
                assert len(fitids|tuneids)==budget and not fitids&tuneids
                assert prior_fit.issubset(fitids) and prior_tune.issubset(tuneids)
                prior_fit,prior_tune=fitids,tuneids
                forbidden=set(z['source_index'])|set(manifest['datasets'][dataset]['old_validation_source_ids'])
                assert not (fitids|tuneids)&forbidden
                fi=np.concatenate([fp+j*len(source) for j in range(3)])
                ti=np.concatenate([tp+j*len(source) for j in range(3)])
                assert np.array_equal(z['fit_error'],old_errors[fi])
                assert np.array_equal(z['tuning_error'],old_errors[ti])
                p=conf[:,1];h=(p>=.5).astype(int)
                aligned=np.column_stack([p,h,np.where(h==1,p,1-p),conf[:,2]])
                fallback=len(np.unique(z['fit_error']))<2
                for variant in VARIANTS:
                    features=aligned if variant=='confidence' else np.column_stack([aligned,extra[variant]])
                    if not fallback:
                        assert np.allclose(z[f'{variant}_scale_mean'],features[fi].mean(axis=0),rtol=1e-10,atol=1e-12)
                    fitrows=f[(f.dataset==dataset)&(f.budget==budget)&(f.variant==variant)].sort_values('lambda_index')
                    assert len(fitrows)==4
                    losses=[]
                    for li in range(4):
                        vp=z[f'{variant}_l{li}_tuning']
                        if fallback:
                            assert np.all(vp==(z['fit_error'].sum()+.5)/(len(fi)+1))
                        v=np.clip(vp,np.finfo(float).eps,1-np.finfo(float).eps)
                        err=z['tuning_error']
                        loss=-np.mean(err*np.log(v)+(1-err)*np.log1p(-v))
                        assert np.isclose(loss,fitrows.iloc[li].tuning_log_loss,rtol=1e-10,atol=1e-12)
                        assert bool(fitrows.iloc[li].constant_fallback)==fallback
                        losses.append(loss);losses_checked+=1
                    selected=int(np.argmin(losses))
                    assert selected==int(z[f'{variant}_selected']);choices+=1
                    for ci,condition in enumerate(CONDITIONS):
                        y,h=z['label'],z[f'h_c{ci}']
                        assert np.array_equal(h,(z[f'p_c{ci}']>=.5).astype(int))
                        row=m[(m.dataset==dataset)&(m.budget==budget)&(m.variant==variant)&(m.condition==condition)].iloc[0]
                        computed=rebuild(y,h,z[f'{variant}_l{selected}_c{ci}'],ci)
                        for k,v in computed.items():assert np.isclose(row[k],v,equal_nan=True)
                        assert int(row.lambda_index)==selected
                        checked+=1
                aware=[z[f'aware_c{ci}'] for ci in range(3)]
                for ci in range(3):
                    assert np.array_equal(aware[ci],np.minimum(z[f'p_c{ci}'],1-z[f'p_c{ci}']))
                if reference_scores is None:
                    reference_scores=[v.copy() for v in aware]
                    for ci,condition in enumerate(CONDITIONS):
                        computed=rebuild(z['label'],z[f'h_c{ci}'],aware[ci],ci)
                        row=m[(m.dataset==dataset)&(m.variant=='decision_aware')&(m.condition==condition)].iloc[0]
                        for k,v in computed.items():assert np.isclose(row[k],v,equal_nan=True)
                        checked+=1
                else:
                    assert all(np.array_equal(a,b) for a,b in zip(aware,reference_scores))
                ac=accounts[(accounts.dataset==dataset)&(accounts.budget==budget)]
                assert len(ac)==4 and (ac.fit_labels==len(fp)).all() and (ac.tuning_labels==len(tp)).all()
                assert (ac.threshold_labels==0).all() and (ac.unique_post_classifier_labels==budget).all()
    assert checked==228 and choices==72 and losses_checked==288
    refs=m[m.variant=='decision_aware'][['dataset','condition','risk','balanced_risk']]
    learned=m[m.variant!='decision_aware'].merge(refs,on=['dataset','condition'],suffixes=['','_aware'])
    conf=m[m.variant=='confidence'][['dataset','condition','budget','risk']]
    learned=learned.merge(conf,on=['dataset','condition','budget'],suffixes=['','_confidence'])
    learned['delta_vs_aware_pp']=100*(learned.risk-learned.risk_aware)
    learned['delta_vs_confidence_pp']=100*(learned.risk-learned.risk_confidence)
    learned.to_csv(OUT/'paired_comparisons.csv',index=False)
    primary=learned[learned.condition=='independent']
    primary.to_csv(OUT/'independent_summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.85,hspace=.65,wspace=.30)
    colors={'confidence':'#526d82','count':'#d47b1f','mask':'#147d68','placebo':'#9e84ad'}
    for ax,dataset in zip(axes.flat,['bank','wdbc','spambase','online_shoppers']):
        for variant in VARIANTS:
            d=primary[(primary.dataset==dataset)&(primary.variant==variant)].sort_values('budget')
            ax.plot(d.budget,d.delta_vs_aware_pp,marker='o',markersize=4,label=variant,color=colors[variant],alpha=.6 if variant=='placebo' else 1)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xscale('log')
        ticks=sorted(primary.loc[primary.dataset==dataset,'budget'].unique())
        ax.set_xticks(ticks,labels=[f'{int(v):,}' for v in ticks])
        ax.minorticks_off()
        ax.set_title({'bank':'Banking','wdbc':'Diagnostic','spambase':'Spam','online_shoppers':'Shopping'}[dataset])
        ax.set_xlabel('Total labelled rows: fit + tuning (log scale)')
        ax.set_ylabel('Error minus untrained reference (pp)')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Exploratory label-budget check: one split, one mask\nIndependent 30% masking; 80% retained; negative favors learned selector',fontsize=12)
    fig.savefig(OUT/'label_budget_curves.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/'label_budget_curves.svg',bbox_inches='tight')
    plt.close(fig)
    result={'status':'passed allocation/scaling/tuning/score audit','checked_records':checked,
        'checked_choices':choices,'checked_tuning_losses':losses_checked,
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    print(primary[primary.variant=='mask'][['dataset','budget','risk','risk_aware','delta_vs_aware_pp','delta_vs_confidence_pp']].to_string(index=False))
    print('Constant-fallback candidate configurations:',int(f.constant_fallback.sum()))
    print('Single-class tuning candidate configurations:',int(f.single_class_tuning.sum()))


if __name__=='__main__':main()
