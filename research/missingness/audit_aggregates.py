"""Checks released aggregate arithmetic, not fitted models or individual predictions."""
from pathlib import Path
import csv
import json
import math
ROOT=Path(__file__).resolve().parent


def audit(root=ROOT):
    with (root/'results/selected_metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==228
    keys=set()
    for r in rows:
        key=tuple(r[k] for k in ['dataset','budget','variant','condition'])
        assert key not in keys;keys.add(key)
        tn,fp,fn,tp=[int(r[k]) for k in ['tn','fp','fn','tp']]
        n=int(r['retained_n'])
        assert tn+fp+fn+tp==n and n>0
        assert math.isclose((fp+fn)/n,float(r['risk']),abs_tol=1e-12)
        if tp+fn and tn+fp:
            assert math.isclose(.5*(fn/(tp+fn)+fp/(tn+fp)),float(r['balanced_risk']),abs_tol=1e-12)
    with (root/'results/label_accounting.csv').open(newline='',encoding='utf-8') as f:
        accounts=list(csv.DictReader(f))
    assert len(accounts)==76
    for r in accounts:
        budget=int(r['budget']);fit=int(r['fit_labels']);tune=int(r['tuning_labels'])
        assert fit+tune+int(r['threshold_labels'])==int(r['unique_post_classifier_labels'])==budget
        if r['variant']!='decision_aware':assert tune==budget//5 and fit==budget-tune
    comparisons=[]
    for r in rows:
        if r['variant']!='mask' or r['condition']!='independent':continue
        base=next(x for x in rows if x['dataset']==r['dataset'] and x['variant']=='decision_aware' and x['condition']=='independent')
        comparisons.append(dict(dataset=r['dataset'],total_labels=int(r['budget']),
                                delta_error_percentage_points=100*(float(r['risk'])-float(base['risk'])),
                                retained_errors=int(r['fp'])+int(r['fn']),
                                reference_errors=int(base['fp'])+int(base['fn']),retained_n=int(r['retained_n'])))
    result=dict(scope='Aggregate arithmetic and label-accounting only; not model re-execution',
                verified_metric_rows=len(rows),verified_budget_rows=len(accounts),comparisons=comparisons)
    return result


if __name__=='__main__':print(json.dumps(audit(),indent=2))
