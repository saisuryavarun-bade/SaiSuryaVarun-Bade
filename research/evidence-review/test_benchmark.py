"""Adversarial checks and independently specified accounting examples."""
import copy
import unittest
from benchmark import make_cases, linked_arithmetic, conservative_review, evaluate


class EvidenceTests(unittest.TestCase):
    def example(self):
        return dict(period='2026-09', budget_cents=10000,
                    claim=dict(actual_cents=12000, variance_cents=2000,
                               citation_ids=['a','b'], cause=None),
                    records=[dict(id='a',period='2026-09',amount_cents=7000),
                             dict(id='b',period='2026-09',amount_cents=5000)])

    def test_known_sum_and_sign(self):
        v=self.example()
        self.assertEqual(conservative_review(v),'accept')
        v['claim']['variance_cents']=-2000
        self.assertEqual(conservative_review(v),'reject')

    def test_internal_consistency_does_not_prove_total(self):
        v=self.example();v['claim'].update(actual_cents=14000,variance_cents=4000)
        self.assertEqual(conservative_review(v),'reject')

    def test_duplicate_and_unresolved_links(self):
        for ids in (['a','a'], ['a','missing'], []):
            v=self.example();v['claim']['citation_ids']=ids
            self.assertEqual(conservative_review(v),'defer')

    def test_wrong_period_is_incomplete_evidence(self):
        v=self.example();v['records'][0]['period']='2026-08'
        self.assertEqual(conservative_review(v),'defer')

    def test_numbers_do_not_identify_cause(self):
        v=self.example();v['claim']['cause']='Prices increased.'
        self.assertEqual(linked_arithmetic(v),'accept')
        self.assertEqual(conservative_review(v),'defer')

    def test_ground_truth_not_visible(self):
        cases=make_cases()
        for c in cases:
            self.assertNotIn('expected_action',c['visible'])
            self.assertNotIn('fault_type',c['visible'])
        altered=copy.deepcopy(cases)
        for c in altered:c['expected_action']='anything';c['fault_type']='hidden';c['truth']={}
        self.assertEqual([x['action'] for x in evaluate(cases)], [x['action'] for x in evaluate(altered)])

    def test_balanced_deterministic_fixtures(self):
        cases=make_cases();self.assertEqual(cases,make_cases())
        self.assertEqual(len({c['case_id'] for c in cases}),72)
        for k in set(c['fault_type'] for c in cases):
            self.assertEqual(sum(c['fault_type']==k for c in cases),12)


if __name__ == '__main__':unittest.main()
