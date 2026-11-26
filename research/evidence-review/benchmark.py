"""Synthetic evidence-integrity fixtures, not a model or a human study.

Python 3.11+, standard library only. Integer cents prevent rounding ambiguity.
Policy functions receive visible evidence only, never hidden truth or fault labels.
"""
from pathlib import Path
import csv
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parent
KINDS = ('supported', 'wrong_total', 'wrong_sign', 'missing_citation',
         'wrong_period', 'unsupported_cause')


def make_cases(seed=20261126, per_kind=12):
    rng = random.Random(seed)
    cases = []
    for i in range(per_kind):
        period = '2026-09'
        values = [rng.randrange(80000, 400001) for _ in range(3)]
        actual = sum(values)
        # Alternate overspend/underspend so a sign check is not one-sided.
        budget = actual + (-1 if i % 2 else 1) * rng.randrange(10000, 80001)
        ledger = [dict(id=f'T{i:02d}-{j}', period=period, amount_cents=v)
                  for j, v in enumerate(values)]
        for kind in KINDS:
            evidence = [dict(r) for r in ledger]
            claim = dict(actual_cents=actual, variance_cents=actual-budget,
                         citation_ids=[r['id'] for r in evidence],
                         cause=None)
            expected = 'accept'
            if kind == 'wrong_total':
                claim['actual_cents'] += 10000
                claim['variance_cents'] += 10000
                expected = 'reject'
            elif kind == 'wrong_sign':
                claim['variance_cents'] *= -1
                expected = 'reject'
            elif kind == 'missing_citation':
                claim['citation_ids'][-1] = 'UNAVAILABLE'
                expected = 'defer'
            elif kind == 'wrong_period':
                evidence[-1]['period'] = '2026-08'
                expected = 'defer'
            elif kind == 'unsupported_cause':
                claim['cause'] = 'Supplier price changes caused the variance.'
                expected = 'defer'
            cases.append(dict(case_id=f'{i:02d}-{kind}', fault_type=kind,
                              expected_action=expected,
                              truth=dict(actual_cents=actual, ledger=ledger),
                              visible=dict(period=period, budget_cents=budget,
                                           claim=claim, records=evidence)))
    rng.shuffle(cases)
    return cases


def accept_all(visible):
    return 'accept'


def citation_presence(visible):
    """Intentionally weak baseline: existence is not evidentiary support."""
    return 'accept' if visible['claim']['citation_ids'] else 'defer'


def linked_arithmetic(visible):
    claim = visible['claim']
    records = {r['id']: r for r in visible['records']}
    ids = claim['citation_ids']
    if not ids or len(ids) != len(set(ids)) or any(i not in records for i in ids):
        return 'defer'
    linked = [records[i] for i in ids]
    if any(r['period'] != visible['period'] for r in linked):
        return 'defer'
    total = sum(r['amount_cents'] for r in linked)
    if total != claim['actual_cents'] or total-visible['budget_cents'] != claim['variance_cents']:
        return 'reject'
    return 'accept'


def conservative_review(visible):
    result = linked_arithmetic(visible)
    if result != 'accept':
        return result
    # The schema contains amounts, dates and IDs, but no causal evidence.
    return 'defer' if visible['claim']['cause'] else 'accept'


POLICIES = dict(accept_all=accept_all, citation_presence=citation_presence,
                linked_arithmetic=linked_arithmetic, conservative_review=conservative_review)


def evaluate(cases):
    rows = []
    for name, policy in POLICIES.items():
        for case in cases:
            action = policy(case['visible'])
            rows.append(dict(policy=name, case_id=case['case_id'],
                             fault_type=case['fault_type'],
                             expected=case['expected_action'], action=action,
                             rubric_match=int(action == case['expected_action']),
                             unsupported_accept=int(action == 'accept' and case['expected_action'] != 'accept')))
    return rows


def write_outputs():
    cases = make_cases()
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    text = json.dumps(cases, indent=2, sort_keys=True)+'\n'
    (out/'cases.json').write_text(text, encoding='utf-8')
    rows = evaluate(cases)
    with (out/'policy_decisions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for name in POLICIES:
        selected = [r for r in rows if r['policy'] == name]
        summary[name] = dict(cases=len(selected), rubric_matches=sum(r['rubric_match'] for r in selected),
                             unsupported_accepts=sum(r['unsupported_accept'] for r in selected),
                             deferrals=sum(r['action']=='defer' for r in selected))
    report = dict(status='Synthetic fixture diagnostic; no LLM or participants',
                  seed=20261126, base_scenarios=12, related_variants_per_scenario=6,
                  independent_samples=False, results=summary,
                  cases_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    write_outputs()
