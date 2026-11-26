# Total post-classifier label-budget design

27 October 2026. Proposed exploratory development stage after the v10 implementation check. Not executed and not a confirmatory preregistration. No new holdout is claimed.

## Fixed scope and resource definition

Hold the corrected classifier and its base training/calibration data fixed. Use only fixed decision threshold 0.5 in the first stage, so no labels outside the declared post-classifier budget are used to select a threshold. This is an explicitly narrow statistical setting; it does not resolve imbalanced-task utility or the adaptive-threshold case.

Define B as all distinct source rows whose labels are available for selector fitting **or tuning**. Use the existing selector pool only; do not use the old validation labels. Evaluation labels are used only to compute outcomes. The untrained decision-aware score requires zero post-classifier labels and serves as the fixed reference across B. Fitted methods may use B labels but do not receive any others. This measures whether spending B labels improves on that reference; equal access does not require an untrained method to consume labels it does not need.

For a bounded next stage use split 42, mask repetition 0, all four corrected tasks and all three pre-existing corruption conditions. Primary description remains independent 0.3 masking at 80 percent batch retention; report all other conditions as sensitivities. Do not add seeds or select budgets because evaluation results look favorable.

## Budget allocation

Candidate budgets fixed before fitting: B in [50,100,300,1000,3000], retaining only values no larger than the task's selector pool, and appending its full pool size if absent. Thus diagnostic has [50,100,114], spam [50,100,300,840], shopping [50,100,300,1000,2441], banking [50,100,300,1000,3000,9039]. Two endpoints alone are insufficient for a learning-curve claim; this remains a feasibility check.

To avoid deciding which rows to label by consulting all pool labels, create an outcome-independent seeded random source-row order. Use a deterministic position rule to reserve every fifth row for tuning and the remainder for fitting. At each B take the first B unique source rows, giving exactly floor(B/5) tuning labels and B-floor(B/5) fit labels. Both subsets are nested, disjoint and shared across methods. Record the order and seed before fitting. Do not stratify the ordering on labels that are supposed to be unobserved.

For each labelled row reuse all three cached development corruptions (0,0.1,0.3). These are views of the same labelled row, not three independent acquisitions. If the tuning error target has a single class, log loss remains defined with explicit binary labels; if fitting targets have a single class, use the specified smoothed constant fallback. Report counts, without retrying a more favorable ordering. Do not report a universal label requirement from one random draw.

## Methods and scoring

Use the v10 decision-aligned channels and identical confidence/count/mask/placebo variants. Fit StandardScaler only on fit views. Use the same four mean-loss L2 candidates. Select each method's penalty using only tuning-view error log loss, with the earliest grid entry breaking exact ties. No development row serves both roles. Do not use old validation data to choose features, thresholds, penalties or a stopping budget.

Save all candidate validation scores, chosen indices, evaluation scores, source-row allocations and cache hashes. Verify that B equals the fit/tuning union and no old-validation or evaluation source row appears there. Confirm that the untrained score is identical across all B. Reconstruct retained counts and class metrics independently from saved scores.

## Interpretation before any expanded run

Report paired risk differences versus the untrained score and aligned learned confidence, along with retained confusion counts and class retention. Fallbacks, zero empirical improvement, absent/nonmonotone crossovers and unstable tuning are valid outcomes. Plotting a curve does not establish a smooth population relationship.

A cost-aware extension with adaptive thresholds would have to allocate threshold-selection labels within B too, specify its allocation before fitting and revisit the appropriate decision loss. It must not silently reuse the old validation labels. Final confirmation still needs a distinct useful contribution, fuller closest-work checking, a reproduction/adaptation boundary and genuinely unexamined evaluation data.
