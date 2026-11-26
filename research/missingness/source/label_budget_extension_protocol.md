# Candidate replication and label-budget extension

Recorded 27 October 2026 after inspecting v0.6–v0.9 results. This is a prospective development plan, not a preregistration of the results already seen. No confirmatory data or venue has been selected.

## Question and intended contribution

When does a learned missingness-aware error ranker improve on a classifier's decision-aware confidence enough to justify collecting its development labels? The intended output is a reproducible learning-curve and break-even analysis, if supported, plus a faithful account of failure cases. No new score formula is proposed. A break-even point may be absent or too uncertain to estimate.

## Reproduction boundary

First document the closest deferral artifact's data availability, preprocessing, split construction, corruption, rejector scaling, labels, C parameter, coverage and ties. Reproduce an identified author-reported table only if the necessary data and exact settings are available and reproducible. Merely porting its logistic rejector to our four tasks is an adaptation. Keep that adaptation clearly separate from any successful original-setting reproduction. External code requires dependency and execution review before running; the downloaded archive has only been read.

## Bounded development comparison

Reuse the corrected classifier and cached inputs for an initial implementation check. Keep threshold regimes separate: fixed 0.5 and the already explored validation-balanced threshold. Do not choose whichever regime gives a favorable result.

Compare the following at identical evaluation coverage and with shared source rows:

1. Untrained decision-aware score: `p` for a negative assigned label, `1-p` for a positive assigned label.
2. Learned decision-aligned confidence using probability, assigned label, their correctness-probability interaction and entropy. Provide the same confidence channels to every augmented variant; avoid a weaker confidence-only representation.
3. Those channels plus scalar missing fraction.
4. Those channels plus full missing indicators.
5. Those channels plus shuffled indicators as a diagnostic control.

Match scaling, error labels, regularization candidates, tuning criterion and constant-error fallback across learned variants. The feature list is fixed before fitting this extension; do not select representations using evaluation risk. Use a published error-auditor comparator if making claims about learned selectors generally; the existing Risk Advisor spot check cannot stand in for a fully tuned multi-split comparison.

For a bounded smoke check, use one existing split and mask repetition, two existing budgets and both decision regimes. Its purpose is software and design validation. It cannot demonstrate a learning curve or establish the final contribution. Save predictions, source identifiers, validation choices and exact configuration; independently recompute coverage and retained confusion counts from scores.

## Label accounting for a subsequent learning curve

Count distinct labelled source rows rather than corruption copies. Report base-classifier training/calibration labels, decision-threshold labels, selector-fit labels and selector-tuning labels separately, plus their union. The current nominal selector budget excludes the validation labels and must not be called the total adaptation cost.

For fair cost comparisons, hold the base training resource fixed and compare policies at the same *total post-classifier labelled budget*. Allocate that budget to fitting and tuning using a rule fixed before evaluation. Cross-fitting is a possible allocation design, not an assumed free source of labels. A tuned decision threshold itself costs labelled data; report an untrained score at the same fixed classifier and decision rule.

Use nested unique-row budgets covering a meaningful range, chosen for feasible sample/error counts before inspecting new evaluation results. Report single-class error-target fallbacks and effective observed errors. A count of corrupted copies is not an effective independent sample size. Never extrapolate a universal label requirement from two endpoints.

## Outcomes and uncertainty

For this development step, preserve ordinary retained error at 80 percent coverage as the specified comparison, alongside balanced retained error, per-class retention and retained confusion counts. These describe a statistical benchmark. A deployment claim would require a justified task loss and review policy. No hypothetical human-review benefit or arbitrary cost ratio is inferred.

A later final design should fix one primary estimand, coverage regime and missingness question before touching new evaluation outcomes. Use additional coverages as declared sensitivity checks. Distinguish a batch rank policy from future-case thresholds calibrated on development data.

Uncertainty must distinguish evaluation-row variation from classifier, selector, tuning and corruption variation. Existing conditional bootstrap intervals omit several of these and cannot be recycled for the extension. Repeated overlapping splits are not independent datasets. A crossover claim needs uncertainty on the paired improvement across budgets and acknowledgement of absent or nonmonotone crossovers.

## Completion criteria

- Source-to-implementation mapping establishes an honest replication/adaptation boundary.
- Correct decision-aware and learned baselines pass score-to-metric checks.
- Label accounting includes all development/tuning labels.
- Closest-work review supports a specific useful extension; an unresolved absence claim does not count.
- Final evaluation is separated from the repeatedly inspected exploratory benchmarks.
- The human author can explain the design, interpret negative results and verify claims before submission; extensive AI assistance is accurately disclosed.

Only then choose a publication venue for the actual contribution. Acceptance and timing cannot be promised.
