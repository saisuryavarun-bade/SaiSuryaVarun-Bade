# Evidence review for financial explanations

**Question:** What can a reviewer verify from linked source records, and what remains unsupported even when the arithmetic is correct?

This is a reproducible, standard-library Python diagnostic with synthetic cases. It supports the design of a future human-AI decision study. It is **not** a trained model, an LLM benchmark, a completed human study, or an evaluation of the existing VarianceIQ agent. AI assisted implementation and documentation; author review of the new material is pending.

## Run

From this repository's root, with Python 3.11 or later:

```bash
python research/evidence-review/benchmark.py
python -m unittest discover -s research/evidence-review -p "test_*.py" -v
```

No API key, network access or third-party package is needed. The first command regenerates the committed fixtures, individual decisions and summary in `results/`. The second checks accounting and adverse evidence conditions using hand-specified examples.

## Design

Twelve base financial scenarios each have six related versions: supported claim, incorrect total, reversed variance sign, unresolved citation, out-of-period evidence, and unsupported causal explanation. These 72 fixtures are correlated variants, not 72 independent observations. All amounts use integer cents. Correctness labels follow an explicit three-action rubric:

- **Accept:** linked records support the numeric claim and no unsupported causal claim is present.
- **Reject:** complete in-period cited records contradict the amount or variance sign.
- **Defer:** evidence is unavailable, from the wrong period, or insufficient to establish a causal claim.

Four deterministic policies illustrate different verification requirements. `accept_all` accepts everything. `citation_presence` checks only for nonempty references. `linked_arithmetic` resolves those references, checks periods and recomputes numbers. `conservative_review` also defers causal claims because this record schema cannot establish causality. Policies receive only `visible`, never hidden truth or the fault category.

## Diagnostic output

| Policy | Rubric matches / 72 | Unsupported accepts | Deferrals |
|---|---:|---:|---:|
| Accept all | 12 | 60 | 0 |
| Citation presence | 12 | 60 | 0 |
| Linked arithmetic | 60 | 12 | 24 |
| Conservative review | 72 | 0 | 36 |

These are deterministic fixture counts. The policies and faults were designed together; this is not a fair comparison against a trained AI, a statistical result, or a measured effect of UI design.

## Interpretation

The conservative rule is designed to match this fixture rubric, so perfect performance would be a **software check, not evidence of general intelligence or research novelty**. Citation presence can look reassuring while providing no protection in this fixture set. Arithmetic checks cannot establish causation. These facts motivate an experimental question; they do not answer whether real people review more accurately.

These fixtures assume the linked records are the complete relevant ledger when in-period and available. They do not detect omitted uncited transactions, dishonest source records, ambiguously defined budgets, tax treatment, accounting policies, or errors in natural-language parsing. An external held-out case set and expert review are required before extending the claims.

See [the proposed study](STUDY_PROTOCOL.md) and [machine-readable diagnostic results](results/summary.json).
