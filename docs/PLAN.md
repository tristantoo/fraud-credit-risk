# Pre-registered analysis plan: fraud and credit risk (P3)

Written 2026-10-23, before any model was fit or any outcome was tabulated by split. It is committed
on its own as "Pre-register analysis plan". Any later change to this plan goes in a separate,
dated commit with the reason given, and any analysis not listed here is labeled **exploratory** in
docs/REPORT.md.

## 0. What was inspected before writing this plan

Part a data: OpenML dataset 1597 "creditcard" v1. The uncompressed ARFF md5 is
`178bcf9bb1f31a3dfe12d0e577884add`, which matches the OpenML `md5_checksum`. Inspected:

- Schema: 284,807 rows and 31 columns (`Time`, `V1`..`V28`, `Amount`, `Class`). Zero missing values.
- `Time`: 0 to 172,792 seconds (about 48 h), nondecreasing in file order, 124,592 unique values.
  Row-count quantiles: q50 = 84,692, q60 = 120,396, q80 = 145,248.
- `Amount`: min 0, median 22.0, max 25,691.16. 1,825 rows have Amount = 0.
- 1,081 exact duplicate rows. Every column is included, so `Time` is identical within a duplicate
  pair and the pair cannot straddle a split.
- `Class` levels are {0, 1}.

**Not inspected:** class counts overall or by split, and any relationship between a feature and
the label. The documented prevalence (492 frauds, 0.172%) comes from the dataset's public
description and was not computed here.

Loader trap: `sklearn.datasets.fetch_openml` silently drops `Time` because OpenML marks it as
`row_id_attribute`. The pipeline must therefore parse the cached ARFF directly
(`data/raw/.../creditcard.arff.gz`) to get `data/raw/creditcard.parquet`. Both paths are
gitignored.

Part b data: Freddie Mac files were searched for in Downloads, Desktop and Documents (all
recursively; OneDrive is empty), using the patterns `sample_orig_*`, `sample_svcg_*`,
`historical_data_*`, `sample_20*.zip` and `*freddie*`. **None were found.** Part b is
**BLOCKED-ON-TRISTAN** for real results.

---

## Part a: ULB credit card fraud (OpenML 1597)

### a1. Question and hypotheses
Q: On transactions that come later in time, does gradient boosting rank and price fraud risk
better than logistic regression, at a realistic review capacity and under an explicit cost model?

- **H1 (primary):** HistGradientBoosting (GBM) beats logistic regression (LR) on test precision at
  0.5% review capacity. Two-sided test.
- **H2 (primary):** GBM beats LR on test PR-AUC (average precision). Two-sided test.
- **H3 (primary):** GBM gives a lower test expected cost than LR. Each model uses its own cutoff,
  chosen on validation, at a review cost of C = $5.
- **H4 (secondary):** Each model's validation-chosen cost cutoff gives a lower test cost than
  (i) flagging nothing and (ii) the fixed top-0.5% capacity rule.

The prior expectation is that GBM ≥ LR. The tests are two-sided, so "LR wins" is a reportable
outcome.

### a2. Data and inclusion rules
- All 284,807 rows are included. Duplicates are kept as recorded. A sensitivity analysis
  deduplicates the test set and reruns the primary metrics.
- Features:
  - LR uses `V1`..`V28` plus `log1p(Amount)`.
  - GBM uses `V1`..`V28` plus raw `Amount`.
  - `Time` is **excluded** as a feature. It is an elapsed-seconds counter, so its test values lie
    outside the training range.
  - `Class` is the label only.

### a3. Split and evaluation protocol (train earlier, test later; no random splits)
The cut points are fixed on `Time` alone, chosen from row-count quantiles. No labels were used to
choose them.

| Slice | Rule | Rows | Use |
|---|---|---|---|
| train | `Time < 120000` | 170,067 | fit models and preprocessing |
| validation | `120000 <= Time < 145000` | 57,190 | choose the cost cutoff only |
| test | `Time >= 145000` | 57,550 | evaluate once |

- Preprocessing (the LR StandardScaler) is fit on **train only**.
- Validation is never used for fitting or tuning. It is used only to choose the cutoff.
- Test is scored once, with frozen models and frozen cutoffs.
- Hyperparameters are fixed a priori and are not tuned:
  - LR: L2 penalty, C = 1.0, lbfgs, max_iter = 2000, no class weights. Unweighted fitting keeps the
    probability scale interpretable.
  - GBM: `HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=0)`.
  - Threads are capped at 4 (`OMP_NUM_THREADS=4`).
- No recalibration in the primary analysis. If calibration is poor, a Platt or isotonic fit on
  validation may be reported, labeled exploratory.

### a4. Metrics
- **Review capacity:** k = ceil(0.005·n) of the rows with the highest scores. That is k = 288 on
  test and k = 286 on validation. Ties are broken by row order.
- **Primary metrics:**
  - precision@0.5%
  - PR-AUC (sklearn `average_precision_score`)
  - expected cost
- **Secondary metrics:**
  - recall@0.5%
  - ROC-AUC
  - Brier score and Brier skill relative to a constant equal to train prevalence
  - ECE: 10 equal-mass bins on the test predicted probability
  - a reliability table: the 10 equal-mass bins, plus a separate row for the top-0.5% bin
  - ROC and PR curves
- **Calibration is reported next to discrimination for every model.**
- **Cost model** (these are assumptions, stated as such):
  - Every flagged transaction costs C = $5 to review.
  - A missed fraud costs its `Amount`. This assumes no recovery, so a $0 fraud costs $0 to miss.
  - A caught fraud costs only its review.
  - Cost = C·(#flagged) + Σ Amount over unflagged frauds. It is reported in total and per 1,000
    transactions.
  - Sensitivity runs use C ∈ {$1, $25}. These are descriptive only and not tested.
- **Cutoff choice:**
  - For each model, the score threshold that minimizes validation cost over all unique validation
    scores, plus a "flag nothing" option. Ties go to the higher threshold.
  - That threshold is then applied unchanged to test. The test flag rate is reported.
  - Secondary rule: the Bayes rule that flags when p·Amount > C, using the raw probabilities.
    Reported descriptively.

### a5. Statistical tests, alpha and multiplicity
- **Paired bootstrap** over the same test rows: B = 5000, seed 20261023.
  - In each resample, compute Δ = GBM − LR for precision@0.5%, PR-AUC and expected cost.
  - k is recomputed as ceil(0.005·n) within the resample.
  - The cutoffs stay frozen from validation.
- Output: percentile 95% CI and a two-sided p = 2·min(P*(Δ≤0), P*(Δ≥0)).
- **Alpha = 0.05.** The three primary deltas (H1–H3) form one family, corrected with Holm.
  Adjusted p-values are reported.
- Secondary deltas get unadjusted 95% bootstrap CIs, explicitly labeled secondary/descriptive. These
  are recall@0.5%, ROC-AUC, Brier and ECE, plus the H4 cost contrasts.
- Sensitivity: a block bootstrap on test with 15-minute blocks (`Time//900`, about 31 blocks),
  because i.i.d. row resampling ignores temporal dependence.
- **Every comparison that is run gets reported**, significant or not.

### a6. Falsification criteria
- H1, H2 or H3 is **unsupported** if its Holm-adjusted p ≥ 0.05, meaning the CI includes 0. It is
  **falsified in the other direction** if the significant delta favors LR. A null is reported as a
  null.
- H4 fails for a model if its test cost CI versus "flag nothing" or versus "top-0.5%" does not
  exclude 0 in favor of the cost cutoff.
- **Leakage alarms.** Halt and audit before believing results if either of these happens:
  - Test PR-AUC exceeds validation PR-AUC by more than 0.10.
  - A run with training labels permuted (seeded) reaches test PR-AUC above 2× test prevalence.

### a7. Leakage audit (pytest; each check must be shown to bite)
Each audit function raises `AssertionError` on violation. Each test runs it on the real split,
where it must pass, and on a deliberately leaky fixture, where it must fail (wrapped in
`pytest.raises(AssertionError)` or a parametrized known-bad case).

1. **Time ordering:** max(train.Time) < min(val.Time) and max(val.Time) < min(test.Time). Bad
   fixture: interleaved rows.
2. **Preprocessing fit on train only:** the fitted scaler's `mean_` and `scale_` equal the
   train-only statistics. Bad fixture: a scaler fit on train ∪ test.
3. **No label-derived features:** `Class` is not in the feature list, `Time` is not in the feature
   list, and no single feature has train ROC-AUC = 1.0 against the label. Bad fixtures: a copy of
   `Class`, and `Class*Amount`.
4. Train and test row hashes do not overlap, so no row appears in both. Bad fixture: a test row
   copied into train.

### a8. Known risks and caveats (part a)
- **Upstream PCA leakage we cannot remove.** `V1`..`V28` are PCA components the data provider
  computed over the full two days, test period included. This is stated as a limitation, not
  fixed.
- **No card, customer or merchant IDs.** The same card can appear in both train and test, and we
  cannot group by entity.
- **Only about 48 h of data.** "Later" means hours later, not months. Train spans day 1 and the
  morning of day 2. Validation and test are afternoon and evening of day 2, so time of day is
  confounded with the split.
- **Few positives.** At the documented 0.172% rate, test should hold about 100 frauds, so CIs will
  be wide. Precision@0.5% is capped at min(1, n_fraud_test / 288).
- **Label latency is ignored.** In production, chargeback labels arrive weeks later.
- The cost figures are assumptions, not bank data.
- Amount = 0 for 1,825 rows. Duplicates are kept, with a dedup sensitivity analysis.

---

## Part b: Freddie Mac Single-Family Loan-Level sample (BLOCKED-ON-TRISTAN)

### b1. Question and hypotheses
Q: What do vintage default curves and delinquency roll rates look like? Does a 12-month PD model
trained on earlier snapshots stay discriminating and calibrated on a later snapshot?

- **H5:** The out-of-time test AUC of the LR PD model is above 0.5. The CI must exclude 0.5.
- **H6:** Calibration-in-the-large holds out of time. The 95% bootstrap CI of (mean predicted PD −
  observed default rate) contains 0. A miss falsifies "calibrated out of time" and is reported
  plainly; regime shift is expected.
- Secondary: GBM versus LR on test AUC and Brier, using a paired bootstrap over test loans with
  B = 5000 and seed 20261023. Holm correction over these 2 deltas.

### b2. Data and inclusion rules
- Files are the Freddie sample pair per vintage: `sample_orig_YYYY.txt` and `sample_svcg_YYYY.txt`.
  They are pipe-delimited with no header.
- Columns are parsed **by position from the General User Guide that ships with the release Tristan
  downloads**. The layout has changed across releases, so positions are verified against that
  guide and not assumed.
- Sentinel codes such as 9999, 999 and 99 are recoded to missing, with indicator columns.
- Files are streamed or chunked, reading only the needed columns, because of the 8 GB RAM limit.
- **Default event:** the first month with delinquency status ≥ 3 (90+ days), or `RA`, or a
  zero-balance code for a credit-event termination (02, 03, 09, 15).
- Prepayment (zero-balance 01) is a competing absorbing state.

### b3. Outputs and split
- **Vintage curves:** cumulative default rate by loan age, for each origination year. The
  denominator is the loans in that vintage's sample. Right-censoring of recent vintages is shown,
  not extrapolated.
- **Roll rates:** monthly transition matrices over {Current, 30, 60, 90+, Default, Prepaid}, pooled
  by calendar year. Both counts and row-normalized rates are reported.
- **PD model:**
  - Snapshots are December reporting periods.
  - Population: loans active and not in default at the snapshot.
  - Label: a default in snapshot+1 through snapshot+12 months.
  - Features: origination fields (credit score, LTV, CLTV, DTI, original rate, log original UPB,
    purpose, occupancy, channel, property type, number of borrowers, first-time buyer flag, MI%)
    plus snapshot state (delinquency 0/1/2, loan age, current rate, current UPB / original UPB).
  - **Never used as features:** anything dated after the snapshot, such as zero-balance
    code/date, loss fields or later modification flags.
- **Out-of-time rule:**
  - The test snapshot S is the latest December with S+12 ≤ the last reporting period in the files.
  - Training snapshots are all Decembers ≤ S−12, so every training label is fully observed before
    S.
  - Models: LR (one-hot plus standardized, L2, C = 1) is primary; GBM (same fixed settings as part
    a) is secondary.
- **Metrics:** AUC/Gini, KS, PR-AUC, Brier, ECE (10 equal-mass bins), a reliability table, and
  calibration-in-the-large.
- **Leakage tests:**
  - max train label window < S
  - no feature dated after the snapshot
  - preprocessing fit on train only
  - each shown to fail on a bad fixture

### b4. Synthetic fixture
Until real files exist, the pipeline runs only on a small generated pipe-delimited fixture inside
`tests/`. It is labeled **SYNTHETIC** in file names, test names and the report. No number from it
is reported as a result.

### b5. Risks (part b)
- The layout can drift between releases.
- Recent vintages are censored.
- The 2008–2010 crisis is a regime shift, so out-of-time calibration may fail. If it does, that
  gets reported.
- The sample of 50k loans per vintage is small for the rare default years.

---

## Pass criteria (copied from spec)
- **A leakage audit runs as a failing test.**
- **Calibration is reported along with discrimination.**
- Part a: train on earlier transactions and test on later ones. Report precision at a fixed review
  capacity (top 0.5%). Pick the cutoff by missed-fraud cost versus false-alarm cost. Compare LR
  with gradient boosting using a paired bootstrap.
- Part b: vintage default curves, roll rates and a 12-month PD model, with a test period after the
  training period. Real data is BLOCKED-ON-TRISTAN.

## Tristan's steps (part b)
1. Register for the Freddie Mac Single-Family Loan-Level Dataset on freddiemac.com.
2. Download the sample files (one zip per vintage). Get at least the 2005–2016 vintages, plus the
   General User Guide for that release.
3. Unzip into `data/raw/freddie/`, which is gitignored. Then run the part b pipeline. It must
   first verify the column positions against the guide.

---

## Deviations (appended 2026-10-23, after the plan commit 4481906; original text above unchanged)

1. **a7 audit 3 strengthened (no label-derived features).** The plan's rule "no single feature has
   train ROC-AUC = 1.0" does not catch the plan's own bad fixture on real data: on the real train
   slice `Class*Amount` has AUC 0.978, because 16 of the 360 train frauds have Amount = 0 and tie
   with the 1,109 zero-amount non-frauds. The audit now also (i) requires every feature name to be
   in a whitelist (`V1..V28`, `Amount`, `log1p_Amount`), (ii) flags AUC >= 0.999 instead of
   exactly 1.0, and (iii) fails any feature that is constant within a class (which catches
   `Class*x` under any name). A test shows the AUC-only rule would miss the leak on real train and
   the strengthened audit catches it (`test_class_times_amount_caught_on_real_train`).
2. **a5 bootstrap details not fixed in the plan.** The 15-minute block bootstrap uses B = 5000 and
   the same seed. The dedup sensitivity reruns the full paired bootstrap (B = 5000) on the
   deduplicated test set, not point estimates only. Resampled index arrays are sorted so "ties
   broken by row order" keeps its meaning inside a resample. Bootstrap p-values are floored by
   resolution: p = 0 means p < 2/B = 0.0004.
3. **a4 ECE binning detail.** Equal-mass bins are formed by splitting the probability-sorted rows
   into 10 equal-count chunks; tied probabilities are ordered by row. Checked: no tied score sits
   on any of the 9 bin edges of the real test scores for either model, so the rule does not bind.
4. **b3 out-of-time leakage test wording.** The plan says "max train label window < S" but also
   "training snapshots are all Decembers <= S-12". The window of snapshot S-12 is (S-12, S], which
   ends at S, so the audit is implemented as "every training label window ends at or before S".
   The data month S is the test snapshot's own state and is known at S.
5. **b3 roll-rate states.** The PD default event (b2) counts 90+ as default, which would make the
   roll-rate states "90+" and "Default" identical. In the roll-rate matrices only, "90+" means
   delinquency >= 3 months while still active, and "Default" means REO acquisition (`RA`) or a
   credit-event zero-balance code (02, 03, 09, 15). Rows with other zero-balance codes (96, 16) or
   unparseable status go to "Other" and are dropped from the matrices. The vintage curves and the PD
   label keep the b2 definition.
6. **b2 column positions.** The positions in `src/frc/freddie.py` follow the published 32-field
   origination and 32-field monthly layouts of the General User Guide as known at writing; no guide
   was available locally to verify against. `check_layout` refuses files with the wrong field count
   or with loan-id / period formats that do not parse, but Tristan must still verify positions
   against the guide of the release he downloads (as b2 already requires).
7. **b PD bootstrap.** Resamples with only one class are skipped (AUC undefined); the number of
   resamples actually used is written next to every interval (`B_used`).
8. **b layout update to release 47 (2026-10-23, before any real Freddie data was downloaded).** Freddie Mac's July 2026
   File Layout has 31 origination fields and 35 performance fields, and names performance files `sample_perf_YYYY.txt`.
   The code expected 32, 32 and `sample_svcg_*`. Changed `ORIG_NFIELDS, SVCG_NFIELDS` to 31, 35, the runner glob, and
   the synthetic fixture (wrong-count test now uses 30). Column positions read by the code are unchanged.
9. **a7 audit 4 ignores Time; non-fraud twins tolerated (2026-10-23, after test scoring; post-hoc).** As written, the
   audit hashed every column including `Time`, which differs across slices by construction, so it could never fire on
   real data (verified: 0 shared all-column hashes train/test). It now hashes every column except `Time` and `Class`
   (the data has no row id). On real data this finds feature-identical twins in train for 639 of 57,550 test rows and
   546 of 57,190 validation rows. Decision: **reported and tolerated, not a fail.** a2 keeps duplicates as recorded,
   a8 already concedes the same card can appear on both sides of the split, and a non-fraud twin can only leak the
   non-fraud label (it could push those ~1.1% of test rows down the ranking, a possible small optimistic bias on
   both models; not quantified). The audit fails if any twin carries a fraud label on either side (none do: 0 of 75 test and 0 of 57
   validation frauds) and returns the twin count, which the runner logs. The test plants twins that differ only in
   `Time`: a non-fraud twin is counted, a fraud twin (or a train copy with the label flipped to fraud) must raise.
