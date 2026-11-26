# Robustness and cleaning correction protocol version 0.6

Recorded before v0.6 outcomes on 27 October 2026. Follow-up analysis of already inspected datasets; no untouched final test or public preregistration claim.

## Cleaning correction

The banking loader removed full-record duplicates before excluding duration. An audit after duration exclusion found 16 duplicate feature rows with no conflicting labels. At seeds 17/42/83, classifier-training and exploratory evaluation partitions shared respectively 6/4/2 identical feature hashes. These may be distinct source people, but independence cannot be established, and identical vectors can bias comparisons. Prior code-level split-disjointness checks only checked row identifiers and missed this issue.

Preserve original outputs as historical. For v0.6, drop duplicate feature vectors after final exclusions, then split. Check feature hashes are disjoint across every pair of partitions for all four datasets. Earlier bank comparisons, including the adapted published baseline, must not support final conclusions until recomputed under corrected cleaning. Do not claim the effect of this correction is known before rerunning.

WDBC receives the same feature-vector audit. Added Spambase and shopping cleaning already excluded duplicate vectors and conflicting labels; keep their source policy unchanged. Log all changes and corrected data hashes.

## Fixed experiment

Datasets: cleaned Bank Marketing, WDBC, Spambase, Online Shoppers. Split seeds: 17, 42, 83. Base classifier: the same calibrated, augmentation-trained boosting configuration as v0.2. All source-row copies remain inside training calibration folds. Save the locally trained model to support later baseline re-evaluation without refitting.

Selector budgets: 100 and full for bank, spam and shopping; 30 and full for WDBC. Nested draw 0. Four linear feature sets: confidence, count, full mask, shuffled-mask placebo. Mean-loss L2 lambda grid fixed at 0.001, 0.01, 0.1, 1.0. Evaluate the full grid; do not pick lambda using evaluation outcomes.

Independent corruption repetitions 0, 1, 2 independently regenerate the development mixture, validation mixture and target masks using predeclared new seed formulas. Each development and validation mixture contains zero, 0.1 and 0.3 independent masking. Test conditions: independent masking 0.3, first-half-block 0.3, second-half-block 0.3, with target cell-loss probabilities as before. Different partitions and repetitions use disjoint deterministic seed ranges.

Additionally select lambda for each feature set/budget/repetition by held-out validation error-prediction log loss averaged over its fixed mixture; ties favor the smaller lambda in grid order. This tuning criterion differs from retained error, and uses a small validation pool on WDBC. Report fixed lambda 0.01 and validation-selected results side by side, not the evaluation-best setting.

Report 80 percent global batch retained ordinary error, balanced retained error, actual-class retention and unreviewed false negatives. Save all grid values, chosen validation criteria, raw-confidence references, source IDs, exact evaluation scores and meta-input caches. Constant fallbacks stay in records.

## Uncertainty and reporting

Report each dataset and split separately; summarize the three split seeds descriptively. For split seed 42, at lambda 0.01 and validation-chosen lambda, compare full-mask and count selectors against confidence-only under independent masking 0.3 at both budgets. Use paired two-level bootstrap: sample the three corruption repetitions with replacement, then sample evaluation source rows with replacement using the same source-row index resample across all sampled repetitions and both methods; re-rank at matched batch capacity within each resampled repetition and average risks. Use 400 resamples and seed 60642. Record percentile intervals as conditional on trained classifiers, split, development labels and fitted selectors; these do not cover training variation or all missingness mechanisms. Three corruption repetitions are a limited Monte Carlo sample, not a full uncertainty model. No multiplicity-adjusted or confirmatory significance claim.

Check uncertainty code against known identical-method and no-difference fixtures. Audit serialized score metrics, validation-only selection, duplicate-feature isolation, finite values, expected row counts and source provenance. No claim of independent external code review.

## Decisions

Determine whether small-budget ordinary-error penalties persist with corrected data, repeated masks and validation-based regularization. Report effects that disappear. Do not state that more data or adding indicators is universally beneficial or harmful; class-retention trade-offs remain co-reported. This run does not resolve novelty or replace the outstanding published-baseline coverage and author scientific review.
