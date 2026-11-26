# Scientific corrections log

## 27 October 2026 banking post-exclusion duplicate audit

The original Bank Marketing loader dropped exact full-record duplicates before removing the call-duration feature. The resulting feature table contained 16 duplicate feature rows with consistent labels. A hash audit found respectively 6, 4 and 2 identical feature hashes shared by classifier-training and exploratory evaluation sets at split seeds 17, 42 and 83. Row-index partition checks had passed but were insufficient to catch this.

These vectors may represent distinct source people; the dataset does not establish independence. We are therefore conservatively removing duplicate feature vectors after exclusions before splitting in v0.6. Original outputs remain preserved as historical evidence, not silently edited. Until a clean rerun is complete, earlier Bank Marketing gains and the adapted Risk Advisor comparison cannot support final conclusions. The added datasets already used post-exclusion feature-vector deduplication; WDBC will receive the same hash-isolation check.

The v0.6 experiment also changes mask realizations and adds regularization checks, so a difference from earlier tables cannot be attributed solely to the cleaning correction. A standalone original-mask rerun would be needed for that attribution. No effect magnitude from this correction is assumed in advance.

The manuscript will distinguish historical exploration from the corrected analyses. Previously stated project reports are retained with this explicit correction notice rather than rewritten as if the issue had never occurred.

The v0.6 corrected analysis is now complete: feature-hash isolation passes across every partition pair in four datasets and three split seeds. The revised manuscript uses corrected repeated-mask/validation-selected comparisons as its main evidence. The v0.7 baseline recheck is completed only at split seed 42, repetition 0, on all four cleaned tasks. It must not be presented as the former three-split baseline average. Historical reports now carry visible correction notices.
