# Report: fraud and credit risk (P3)

Written 2026-10-23. Every number below comes from outputs/part_a_* unless marked otherwise. The plan and its dated deviations are in docs/PLAN.md. Part b has NO real results (BLOCKED-ON-TRISTAN).

## Result in one paragraph
On ULB card transactions split by time (train on the first ~33 h, test on the last ~8 h), gradient boosting beats logistic regression on two of the three primary metrics, and both wins survive Holm correction:
- PR-AUC: 0.813 vs 0.688, delta +0.125, 95% CI [0.073, 0.187].
- Expected cost at a validation-chosen cutoff: $2,988 vs $3,439 on test, delta -$451, CI [-1,024, -93].

The primary capacity metric is a NULL. Precision in the top 0.5% is 0.226 vs 0.215 (65 vs 62 of 75 frauds), delta +0.010, CI [0, 0.024], Holm p = 0.096. That null is mostly low power: precision@0.5% cannot exceed 75/288 = 0.260 here. It is not evidence that the two models are equal.

The validation-chosen cost cutoff FAILS to beat the simple top-0.5% rule for either model (H4(ii)). The plain Bayes rule p*Amount > $5 gives a lower test cost than the chosen cutoff for both models.

GBM is better calibrated by Brier score (CI excludes 0). The ECE difference is not significant. All of this sits on top of upstream PCA features that were fit on the full two days. Part b (Freddie Mac) is built and tested on a SYNTHETIC fixture only.

## Eval protocol
- Data: OpenML 1597, parsed from the cached ARFF (md5 178bcf9b...), 284,807 rows. Time is not a feature.
- Split on Time only, with cut points taken from row quantiles before labels were looked at:
  - train: Time < 120000. 170,067 rows, 360 frauds, prevalence 0.212%, 640 duplicate rows.
  - val: 120000 <= Time < 145000. 57,190 rows, 57 frauds, 0.100%, 200 duplicates.
  - test: Time >= 145000. 57,550 rows, 75 frauds, 0.130%, 241 duplicates.
- LR: L2, C = 1, features V1..V28 + log1p(Amount), scaler fit on train only.
- GBM: HistGradientBoosting, learning rate 0.05, 300 iterations, 31 leaves, no early stopping. No tuning for either model.
- Validation is used only to pick each model's cost cutoff. The test set is scored once.
- Review capacity: k = ceil(0.005n), which is 288 on test.
- Cost model (assumed): $5 per flag, plus the Amount of each missed fraud. Sensitivity runs at $1 and $25.
- Inference: paired bootstrap on the same test rows, B = 5000, seed 20261023. k is recomputed in each resample and the cutoffs stay frozen.
- Multiple testing: Holm over H1-H3.
- Sensitivity analyses: a 15-minute block bootstrap and a deduplicated test set.
- A p printed as 0 means p < 0.0004 raw and p < 0.0012 after Holm.

## Results (test set)
| Metric | LR | GBM |
|---|---|---|
| precision@0.5% | 0.2153 (62) | 0.2257 (65) |
| recall@0.5% | 0.827 | 0.867 |
| PR-AUC | 0.688 | 0.813 |
| ROC-AUC | 0.972 | 0.979 |
| Brier | 0.000667 | 0.000447 |
| Brier skill | 0.488 | 0.656 |
| ECE (10 equal-mass bins) | 0.000289 (22% of prevalence) | 0.000215 (17% of prevalence) |
| top-0.5% predicted vs observed | 0.205 vs 0.215 | 0.216 vs 0.226 |
| val-chosen threshold (C=$5) | 0.0939 | 0.1039 |
| test flag rate | 0.153% | 0.122% |
| test cost at threshold | $3,439 ($59.8 per 1k transactions) | $2,988 ($51.9 per 1k transactions) |
| cost of top-0.5% rule | $3,823 | $3,173 |
| cost of flagging nothing | $7,729 | $7,729 |
| cost of Bayes rule p*Amount>$5 | $3,140 | $2,846 |

Validation PR-AUC: LR 0.612, GBM 0.760.

## Paired contrasts, GBM - LR
| Contrast | Family | Delta | iid CI | p raw | p Holm | block CI | dedup CI |
|---|---|---|---|---|---|---|---|
| H1 precision@0.5% | primary | +0.0104 | [0, 0.0243] | 0.096 | 0.096 | [0, 0.0215] | [0, 0.0244] |
| H2 PR-AUC | primary | +0.125 | [0.073, 0.187] | <0.0004 | <0.0012 | [0.063, 0.209] | [0.068, 0.185] |
| H3 cost | primary | -451 | [-1024, -93] | <0.0004 | <0.0012 | [-1025, -102] | [-1030, -83] |
| recall@0.5% | secondary | +0.040 | [0, 0.0875] | 0.096 | | [0, 0.0877] | [0, 0.0921] |
| ROC-AUC | secondary | +0.0073 | [-0.0098, 0.0289] | 0.459 | | [-0.0111, 0.0305] | [-0.0096, 0.0292] |
| Brier | secondary | -0.000220 | [-0.000333, -0.000114] | <0.0004 | | [-0.000352, -0.000110] | [-0.000325, -0.000110] |
| ECE | secondary | -0.000074 | [-0.000414, 0.000151] | 0.456 | | [-0.000423, 0.000145] | [-0.000391, 0.000146] |

## H4: chosen cutoff minus simple rule, in dollars (secondary)
| Comparison | Delta | 95% CI | p | Result |
|---|---|---|---|---|
| LR vs flag nothing | -4,290 | [-8,633, -1,133] | <0.0004 | pass |
| LR vs top-0.5% | -384 | [-963, +397] | 0.28 | FAIL |
| GBM vs flag nothing | -4,741 | [-9,126, -1,541] | <0.0004 | pass |
| GBM vs top-0.5% | -185 | [-1,100, +1,343] | 0.70 | FAIL |

The block and dedup bootstraps give the same verdicts: p = 0.25 / 0.67 (block) and 0.26 / 0.71 (dedup).

## Cost sensitivity (descriptive)
Test costs at each review cost C:
| C | Model | Chosen cutoff | Top-0.5% | Bayes rule | Flag nothing |
|---|---|---|---|---|---|
| $1 | LR | 2,779 | 2,671 | 2,943 | 7,729 |
| $1 | GBM | 2,121 | 2,021 | 2,479 | 7,729 |
| $5 | LR | 3,439 | 3,823 | 3,140 | 7,729 |
| $5 | GBM | 2,988 | 3,173 | 2,846 | 7,729 |
| $25 | LR | 6,053 | 9,583 | 4,049 | 7,729 |
| $25 | GBM | 5,179 | 8,933 | 3,322 | 7,729 |

The chosen cutoff never wins outright. The top-0.5% rule is cheaper at $1, and the Bayes rule is cheaper at $5 and $25. With only 57 fraud cases in validation, a threshold chosen there is noisy.

## Falsification
- H1-H3 would be unsupported if Holm p >= 0.05, and falsified if a significant delta favored LR. Result: H1 is unsupported, H2 and H3 survive, and no significant delta favors LR.
- H4(ii) failed for both models.
- Leakage alarms did not trip:
  - Test minus validation PR-AUC: +0.076 (LR) and +0.053 (GBM), against a limit of 0.10.
  - Permuted-label PR-AUC: 0.00089 and 0.00087, against a limit of 0.0026.
- The leakage audits are pytest tests. They pass on the real split and raise on leaky fixtures. 33 passed (with data/raw present; on a fresh clone 28 pass and the 5 real-data tests are skipped with a printed reason).
- Audit 4 (no shared rows) now ignores Time (PLAN deviation 9). Train has feature-identical twins for 639 of 57,550 test rows (1.1%) and 546 of 57,190 validation rows. All are non-fraud on both sides, so the audit passes; twins are reported and tolerated, and any fraud-labelled twin fails it. Non-fraud twins could make both models' test numbers slightly optimistic; this was not quantified.

## Caveats
- Upstream PCA leakage: V1..V28 were computed over all 48 h, including the test period.
- 48 h only, so time of day is confounded with the split.
- There are no card IDs, and label latency is ignored.
- Prevalence drifts: 0.21% in train, 0.10% in validation, 0.13% in test. Test has only 75 frauds.
- ECE has poor resolution: 69 of the 75 test frauds land in the top bin. LR over-predicts in bins 1-9.
- Deviations 2 and 3 (commits 3361b38 and 26a63e9, 06:05) were made AFTER test scoring (05:48-06:04), so treat them as post-hoc. Deviation 1 was made after looking at the real train data.
- The cost figures are assumptions.

## Pass criteria (reviewer statuses)
- PASS: leakage audit runs as a failing test.
- PASS: calibration reported along with discrimination.
- PASS: temporal split.
- PASS: precision@0.5%.
- PASS: cost cutoff.
- PASS: paired bootstrap.
- PASS: pre-registration.
- FAIL: every comparison reported. REPORT.md was missing; this file fixes that, pending re-review.
- BLOCKED-ON-TRISTAN: part b.
- PASS: repo hygiene.

## Resolved reviewer issues (2026-10-23)
1. Audit 4 was vacuous: its hash included Time (0 shared all-column hashes train/test on real data). It now hashes every column except Time and Class, tolerates and counts non-fraud twins (639 test, 546 validation, re-counted by the new audit), and fails on any fraud-labelled twin (0 of 75 test, 0 of 57 validation frauds have one). The bite test plants twins that differ only in Time. PLAN deviation 9; post-hoc, made after test scoring.
2. Real-data audits skipped silently on a fresh clone. The skip reason now names the download command, pytest prints skip reasons (`-rs`), and the README says a run with "5 skipped" has not audited the real split.

Results are unaffected. Audit 4 is a gate, and nothing downstream reads it. The new version passes on the real split (checked with `uv run pytest`, 33 passed), so the runner would not halt. No model, metric, split or bootstrap code changed. outputs/ was not regenerated, so part_a_run.log does not yet show the twin-count line that run_part_a.py now logs.

## Unresolved reviewer issues
1. Audit 3 would miss an additive leak with AUC below 0.999.
2. In part b, freddie.build() concatenates whole vintages as str, which is a memory risk.
3. The part b column positions have not been checked against the guide. vintage_curves may be off by one month.
4. The PR curves are drawn with linear interpolation (cosmetic).

## Tristan's steps
See tristan_steps.
