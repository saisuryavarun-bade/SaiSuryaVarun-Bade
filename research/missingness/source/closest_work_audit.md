# Closest work audit update

**Latest status, 27 October 2026:** the access limitations for the completeness-deferral artifact and the label-missingness repository described below were subsequently resolved. The deferral source includes learned confidence/completeness and scalar missing-count comparisons. MissingnessShiftBench source components have also now been inspected. See `literature_gap_assessment.md` for the superseding comparison, source provenance and decision to retire the new-method framing. Earlier entries below preserve the search history and must not be read as current access status.

27 October 2026. Reading status is explicit; a related-work absence claim is not justified. This update narrows the candidate contribution and does not claim full-paper reproduction.

## Work checked beyond the initial abstracts

**Liang, Peng and Sun, Selective Classification Under Distribution Shifts.** Read the HTML methods and experimental setup, especially Sections 3.2 and 4.1: https://arxiv.org/html/2405.05160v1. The inspected evaluation uses multiclass neural models on image and text tasks. Our binary tabular missingness experiments differ in scope, but difference in scope alone is not novelty. The paper's multiclass margin argument should not be presented as an independent binary baseline without checking rank equivalence. Final publication is identified by the author's PMC manuscript as TMLR, October 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC12470254/. No published results were reproduced here.

**Rockenschaub et al., Robust prediction under missingness shifts.** Inspected Sections 6 and 7, including missingness scenarios, data, estimators, and evaluation: https://arxiv.org/html/2406.16484v1. Its inspected experiments use continuous generated outcomes, including on real covariates, and MSE. Our endpoint is ranking errors from fixed binary classifiers under changed masks. This is a candidate distinction, not proof of a new contribution. Formal invariance statements from that work do not automatically justify our selector transfer.

## Newly found overlap

**Jones et al., Selective Classification Can Magnify Disparities Across Groups.** Abstract inspected: https://arxiv.org/abs/2010.14134. Aggregate improvement can obscure subgroup degradation. Our actual-label retention audit is a performance diagnostic; it is not a new fairness theorem or a protected-group audit.

**Lee et al., Fair Selective Classification Via Sufficiency, ICML 2021.** Publisher abstract inspected: https://proceedings.mlr.press/v139/lee21b.html. Existing methods already address disparities across coverage levels. Full methods need review if a fairness contribution is proposed.

**Schreuder and Chzhen, Classification with abstention but without disparities, UAI 2021.** Publisher abstract inspected: https://proceedings.mlr.press/v161/schreuder21a.html. Abstention with constraints is established. Our predicted-class-stratified selection is a diagnostic, not a novel fairness algorithm.

**Selective classification under imbalance in multiclass settings: A novel metric for bias-aware risk-coverage evaluation (2026).** Publisher search excerpts inspected; full page fetch blocked (403): https://www.sciencedirect.com/science/article/pii/S1532046426001085. The retrieved description includes class-aware evaluation and class-conditional selection on tabular data. Treat as a close-overlap alert, verify authors, DOI, full text and code before a manuscript claim. Do not claim that class-balanced selective metrics are new.

**MissingnessShiftBench, Kovari (2026 software repository).** Author README inspected: https://github.com/kovariati/MissingnessShiftBench. Describes broad binary tabular experiments on model-selection transfer across missingness mechanisms, with selective prediction as a supporting analysis. We have not independently verified its reported results or inspected/reproduced source code. Before paper-scale runs, inspect its selection analyses and companion article. Its reported broad coverage makes a generic benchmark-only claim weak.

**Completeness-aware selective deferral for at-risk student prediction under degraded data.** Search metadata only: https://zenodo.org/records/23612495. Direct retrieval failed. This appears especially close to the simple combined-selector idea; inspect the artifact and associated paper before a novelty decision. Publication/review status unverified.

**Selective Cost-Aware Random Forests for Unreliable ...** Search excerpt only: https://openreview.net/pdf?id=McQ73Hv4Ru. Retrieval encountered browser verification. Title, authors, venue, and full methods unverified. Excerpt mentions train/test MCAR changes; record as a lead, not a verified comparator.

## Position after this search

### Follow-up during the sample-size experiment

**Lahoti, Gummadi and Weikum, Responsible model deployment via model-agnostic uncertainty learning.** Machine Learning 112:939-970 (2023 issue; online 2022), DOI 10.1007/s10994-022-06248-y. Inspected introduction, proposed approach and implementation passages: https://pmc.ncbi.nlm.nih.gov/articles/PMC9988805/. Risk Advisor is an established post-hoc error meta-learner using an ensemble of stochastic gradient-boosted trees and includes selective abstention. Its published implementation and tuning differ from our simple logistic selectors; do not label our code a reproduction. This adds a necessary stronger baseline before a novelty claim based on learning error scores. Author code: https://github.com/plahoti-lgtm/RiskAdvisor.

The targeted search for missingness, selective prediction and sample size did not resolve a distinct new contribution. It returned theoretical selective-learning sample complexity and several application studies; none should be cited as proof that this precise question is new. A software record apparently discussing completeness-aware deferral remains inaccessible through direct retrieval as of this search. A repository search excerpt about label-missingness-aware abstention was also found, but its returned URL used a non-GitHub mirror; metadata and source identity remain unverified, so it is not accepted as a primary citation. Label missingness and missing covariates must be distinguished.

### Author-code audit and adapted baseline

Retrieved the Risk Advisor author's code through the connected GitHub tool and pinned commit `0a2b5b0473dabecf13e296dc7605bf081c309adb`. Inspected ensemble construction, training-error targets, optional base-probability inputs, entropy decomposition and final risk aggregation. Archived reference files with the Apache 2.0 license. Implemented a clearly labelled adaptation with held-out error labels, fixed in-range hyperparameters and missing-input preprocessing; see `risk_advisor_plan.md` and `risk_advisor_report.md`. All adaptations are material and must be stated in any manuscript. Numerical score checks and comparison in our setting are not reproduction of the paper's published benchmarks.

The initial idea is too broad to market as a new method. Continue the exploratory replication to learn whether a more specific question is worth pursuing: whether mask-augmented error ranking transfers across unseen correlated outages when evaluated jointly for error, class retention, and coverage stability. A publishable contribution would need stronger closest-work comparisons and an informative, replicated finding or justified method.

No source access failure should be interpreted as evidence that a study does not exist. No legal or clinical conclusion is drawn from these sources or our benchmarks.
