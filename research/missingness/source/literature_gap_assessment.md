# Literature gap decision — 27 October 2026

## Decision

Retire the proposed new-method framing. Confidence plus missingness, learned error ranking, scalar missing-count controls and class-aware selective evaluation already have close precedents. The current experiments are exploratory development evidence. Their volume does not establish novelty or publication readiness.

Continue as a **candidate replication and label-budget extension** with the working question: **When is learning a missingness-aware referral score worth its additional labelled-data cost, compared with decision-aware confidence?** This narrower question is not yet a verified literature gap. A useful result would characterize when learning pays off, with fair baselines and an explicit accounting of labels, rather than report that extra features sometimes overfit.

## Closest evidence and remaining distinctions

| Source and verified status | What the inspected evidence already contains | Implication for our work |
|---|---|---|
| El-Ghareeb, *Completeness-aware selective deferral for at-risk student prediction under degraded data*, v3 software deposit, 24 October 2026. [Zenodo](https://zenodo.org/records/23612495), DOI 10.5281/zenodo.23612495. Peer-reviewed article status not verified. | `pipeline/rejectors.py` fits standardized logistic error selectors, comparing margin alone, margin plus source completeness, scalar missing-count controls and additional uncertainty channels. `analyze_budget_atrisk.py` examines referral capacity and class-related outcomes. | The simple combination and count-control comparison are already present. Inspected budget analyses concern referral capacity; configured observation windows concern elapsed course time. Systematic nested labelled error-training budgets were not found in the inspected design. This is a scoped observation, not proof of absence from all related work. |
| Kovari, *MissingnessShiftBench*, software repository; companion metadata has no final article DOI. [Repository](https://github.com/kovariati/MissingnessShiftBench). | Broad binary-tabular missingness-shift evaluation, predictor missingness indicators, calibration and selective coverage. Inspected selective metrics use confidence from the predictor; inspected indicators enter the predictor. | A generic missingness robustness benchmark is weak differentiation. A fixed-predictor, post-hoc error-selector label-budget study differs from inspected components, but needs a useful finding and further literature checking. |
| Lahoti, Gummadi and Weikum, *Responsible model deployment via model-agnostic uncertainty learning*, Machine Learning. [Article](https://doi.org/10.1007/s10994-022-06248-y). | Learned error auditing and selective abstention. Author-code construction has been inspected. | Error meta-learning is established. Our held-out-data implementation is an adaptation, not a reproduction of its published results. |
| Rabanser and Papernot, *What Does It Take to Build a Performant Selective Classifier?*, NeurIPS 2025. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/71460926102fade443ea7ec89ae8a73a-Abstract-Conference.html), [inspected text](https://arxiv.org/html/2510.20242v1). | Ranking, approximation and statistical limitations; fixed-classifier correctness posterior. | Correctness probability for the actually assigned label is established. Our decision-aware reference is a standard baseline, not a new algorithm. Sample-size effects in general are not novel. |
| Traub et al., *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*. [Inspected text](https://arxiv.org/html/2407.01032v2). | Evaluation objectives, working-point comparisons, imbalance and metric-dependent conclusions. | Neither metric-sensitive rankings nor retained-class diagnostics alone supply our contribution. |
| Franc, Prusa and Voracek, *Optimal Strategies for Reject Option Classifiers*, JMLR 24(11), 2023. [Publisher](https://www.jmlr.org/beta/papers/v24/21-0048.html). | Established conditional-risk-based rejection and learning uncertainty scores. | Do not claim that aligning a rejection score with classifier loss is new. |
| *MissingnessAware-Abstention*, project repository. [Repository](https://github.com/marcorags01/MissingnessAware-Abstention). | Inspected README concerns missing labels and learned label-observation mechanisms. | Distinguish missing outcomes from our missing covariates. Repository identity is now verified; its claims are not independent experimental evidence. |

The 2026 imbalance-metric article at [the publisher](https://www.sciencedirect.com/science/article/pii/S1532046426001085) remains a full-text access limitation. Its search description is an overlap alert, not evidence that its methods were fully reviewed. No systematic-review completeness or first-of-kind claim is justified.

## Source audit provenance

The completeness-deferral v3 ZIP was downloaded for read-only inspection. Its MD5 `ee2c1e45be0d411b74f89745d79a994d` matches the deposit metadata; ZIP CRC validation passed. Exact inspected text members and hashes are recorded under `references/completeness_deferral/`. External source code has not been executed and published results have not been independently reproduced.

MissingnessShiftBench revision `12089900a6b97de6114f93f446d9bea6edc07223` was identified. Initial reads included default-branch files; subsequent core configuration reads were pinned. `references/gap_audit/provenance.json` records this distinction, so the complete set must not be described as fetched at one pinned revision. Saved source text is evidence for inspection, not instructions for this project.

## Consequence of the decision-aware diagnostic

For binary positive-class probability p and actual predicted label h, use the standard score `p` if h=0 and `1-p` if h=1. At a 0.5 decision threshold it equals `min(p,1-p)`; at other thresholds the rankings can differ. This is a plug-in error score, with no guarantee that probabilities stay calibrated after degradation.

The completed v0.9 audit uses the same cached predictions, with no fitting. Under validation-chosen thresholds and independent 0.3 masking, full masks have higher mean ordinary retained error than this reference at both budgets on all four tasks. The smaller-budget differences are +5.677, +1.691, +1.203 and +2.508 percentage points for banking, diagnostic, spam and shopping, respectively. These are descriptive averages over three splits and three masks, not independent population replications. The v0.8 shopping sign reversal versus the earlier learned-confidence representation is therefore not evidence of improvement over an appropriate untrained reference.

The learned confidence representation was not redesigned for the changed decision boundary. A fair extension must give all learned variants the same decision-aligned inputs before interpreting mask effects. The existing test data have been examined repeatedly and cannot become untouched confirmation by renaming the analysis.

## Proceed / revise criteria

Proceed to a small, explicitly exploratory implementation of the extension in `label_budget_extension_protocol.md`. Before a confirmatory claim: finish closest-paper checking; document which published result is reproduced and which component is extended; establish a useful question beyond familiar overfitting; freeze a label-accounted design; and reserve genuinely unexamined evaluation data. If the extension duplicates existing findings or gives no informative decision guidance, revise the question. Do not invent a method name, a guaranteed acceptance timeline or a minimum label threshold from the current two-budget grid.
