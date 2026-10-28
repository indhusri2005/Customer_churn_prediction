# FreshBasket loyalty churn analysis

## Executive summary

The churn model uses customer profile and monthly activity available only through **March 2024** to predict the supplied April-June 2024 churn label. 1,384 adequately observed customers from older signup cohorts were used for training and 843 customers from 2023-24 signup cohorts were held out. The best validation PR-AUC was **0.939** from **Random forest (all features)** (precision 0.929, recall 0.827, F1 0.875, accuracy 0.951 at a fixed 0.50 threshold).

Use the score to prioritize outreach, not as a definitive churn decision. A small model-based coupon scenario is explicitly non-causal; validate it with a randomized pilot before funding a broad rollout.

## Data quality and preparation

| check                                                   |   count | handling                                                                                                                              |
|:--------------------------------------------------------|--------:|:--------------------------------------------------------------------------------------------------------------------------------------|
| Customer profile rows                                   |    2608 | Original profile table row count.                                                                                                     |
| Unique customers in profile                             |    2600 | Expected one row per customer.                                                                                                        |
| Exact duplicate profile rows                            |       8 | Removed before joining.                                                                                                               |
| Duplicate profile customer keys                         |      16 | Repeated key rows; deduplicated to one profile.                                                                                       |
| Monthly activity rows                                   |   28073 | Original monthly table row count.                                                                                                     |
| Exact duplicate activity rows                           |      58 | Removed before monthly-key deduplication.                                                                                             |
| Duplicate customer-month keys                           |     120 | Resolved to one record per customer-month.                                                                                            |
| Conflicting duplicate customer-month groups             |       2 | Prefer nonnegative-spend copy for duplicated records.                                                                                 |
| Negative spend rows                                     |      24 | Set negative spend to zero after resolving duplicate sign-flip copies.                                                                |
| Zero-spend rows                                         |    2355 | Retained; zero spend is valid for no-purchase months.                                                                                 |
| Zero-transaction rows                                   |    2335 | Checked against spend and labels.                                                                                                     |
| Zero transactions with positive spend                   |       0 | Potential transaction/spend inconsistency.                                                                                            |
| Positive transactions with nonpositive spend            |      44 | Potential transaction/spend inconsistency; negative/zero spend retained as zero.                                                      |
| Monthly spend outliers (IQR rule)                       |     673 | Raw positive spend above $381.84; model input clipped at raw positive 99th percentile.                                                |
| Unrecorded customer-months through Mar 2024             |   26138 | Expected from signup month through the pre-label cutoff; treated as zero recorded activity in features, not asserted to be data loss. |
| Missing AGE values                                      |      78 | Imputed inside model pipelines.                                                                                                       |
| Missing GENDER values                                   |      52 | Imputed as most frequent category inside model pipelines.                                                                             |
| Missing HOME_STORE_PRICE_TIER values                    |     130 | Imputed as most frequent category inside model pipelines.                                                                             |
| Duplicate churn-label rows                              |       5 | Removed before joining.                                                                                                               |
| Insufficient-history customers                          |     373 | Excluded from model fitting and validation.                                                                                           |
| Churn-label mismatches among adequate-history customers |       0 | Flagged and excluded from supervised fitting; provided label otherwise retained.                                                      |

Exact duplicates are removed; duplicated customer-months are reduced to one row, preferring nonnegative spend when the duplicate differs by sign. Negative spend is floored at zero and positive monthly spend is clipped at the raw positive 99th percentile for feature construction. Missing customer-months through March 2024 are treated as no recorded activity in monthly aggregates. This is an assumption, not proof of zero activity. Categorical case/whitespace is normalized; missing values are imputed within each training fold's pipeline.

The monthly activity data is sparse, and an absent row is treated as zero recorded activity for feature construction. The label-consistency check uses the supplied `LAST_PURCHASE_DATE`, since sparse monthly rows cannot independently prove whether an absent month contained a transaction. Customers with sufficient history whose label conflicts with this date-based check are excluded rather than silently relabeled.

## Validation and model comparison

The split is by signup cohort, not random: signup before 2023-01-01 is training, and signup on/after that date is validation. Only features dated January 2023 through March 2024 are used. The four metrics at threshold 0.50 are reported alongside PR-AUC to make class imbalance visible.

| model                              |   accuracy |   precision |   recall |    f1 |   pr_auc |   threshold |
|:-----------------------------------|-----------:|------------:|---------:|------:|---------:|------------:|
| Logistic regression (all features) |      0.930 |       0.774 |    0.931 | 0.845 |    0.833 |       0.500 |
| Random forest (all features)       |      0.951 |       0.929 |    0.827 | 0.875 |    0.939 |       0.500 |

### Ablation

| model                                 |   accuracy |   precision |   recall |    f1 |   pr_auc |
|:--------------------------------------|-----------:|------------:|---------:|------:|---------:|
| Demographics only                     |      0.606 |       0.225 |    0.376 | 0.281 |    0.225 |
| Demographics + behaviour              |      0.936 |       0.870 |    0.809 | 0.838 |    0.919 |
| Demographics + behaviour + engagement |      0.950 |       0.928 |    0.821 | 0.871 |    0.932 |
| All feature groups                    |      0.953 |       0.935 |    0.827 | 0.877 |    0.939 |

### Main feature signals

The table below reports validation permutation importance (mean reduction in average precision). Importance is predictive association, not causality.

| feature                       |   importance_mean |   importance_std |
|:------------------------------|------------------:|-----------------:|
| recency_months                |             0.008 |            0.005 |
| app_sessions_recent3_mean     |             0.006 |            0.004 |
| coupons_redeemed_recent3_mean |             0.006 |            0.001 |
| spend_recent3_mean            |             0.005 |            0.005 |
| transactions_recent3_mean     |             0.005 |            0.005 |
| transactions_slope            |             0.004 |            0.002 |
| spend_slope                   |             0.003 |            0.002 |
| support_tickets_slope         |             0.003 |            0.001 |

## Segment comparison (held-out cohort)

### Membership tier

| segment   |   customers |   observed_churn_rate |   mean_predicted_risk |
|:----------|------------:|----------------------:|----------------------:|
| Platinum  |      95.000 |                 0.295 |                 0.313 |
| Silver    |     488.000 |                 0.215 |                 0.303 |
| Gold      |     260.000 |                 0.154 |                 0.252 |

### Cities with the highest observed churn rates

| segment   |   customers |   observed_churn_rate |   mean_predicted_risk |
|:----------|------------:|----------------------:|----------------------:|
| Boise     |      79.000 |                 0.253 |                 0.325 |
| Austin    |      83.000 |                 0.253 |                 0.310 |
| Columbus  |      93.000 |                 0.226 |                 0.302 |

### Signup cohorts

| segment   |   customers |   observed_churn_rate |   mean_predicted_risk |
|:----------|------------:|----------------------:|----------------------:|
| 2023Q1    |     176.000 |                 0.341 |                 0.372 |
| 2023Q2    |     182.000 |                 0.319 |                 0.337 |
| 2023Q3    |     177.000 |                 0.181 |                 0.250 |
| 2023Q4    |     187.000 |                 0.118 |                 0.222 |
| 2024Q1    |     121.000 |                 0.008 |                 0.251 |

Segment rates and average risks are calculated on the held-out signup cohort only. They are descriptive, not causal. Small cohorts can be noisy; use customer counts and confidence intervals before targeting a segment.

## Retention scenario

The scenario targets the top quartile of validation predicted risk and increases recent promotion transaction share by 0.15 (capped at 1.0) and coupon redemptions by one per month. Targeted population: 211 customers (25.0%). Mean predicted risk changes from 0.756 to 0.631 (change -0.125).

**Important:** this is a model-based feature perturbation, not a treatment-effect estimate. The model learned historical associations and cannot establish that offering a coupon would cause the probability change. Run a randomized controlled pilot with an untreated holdout and measure incremental retention, margin, redemption, and cannibalization.

## Recommendations

1. Use the model as a ranked outreach queue and compare capacity-based top-k targeting with the current blanket offer policy.
2. Pilot proactive support recovery for customers with rising support tickets or recent complaints; track complaint resolution and subsequent purchase activity.
3. Use targeted, margin-aware coupons only in a randomized test. Measure incremental gross margin and retained customers, not redemptions alone.
4. Monitor score calibration, precision/recall, PR-AUC, cohort performance, and data completeness monthly; retrain only after label maturation.
5. Improve month-level activity capture and validate the label-generation logic with the data owner, even though the supplied label and last-purchase date are consistent in this extract.

## Limitations and assumptions

- The dataset is synthetic and spans one observation window; external validity is unknown.
- Validation uses different signup cohorts from the same observation period, not a later outcome period. Tenure and data completeness differ between cohorts.
- Sparse months are treated as zero recorded activity; alternative missingness assumptions may materially change the model.
- The target definition depends on prior purchases; inadequate-history customers and any definition/label conflicts are excluded from evaluation.
- Threshold 0.50 is a transparent reporting default, not an optimized business operating point. Threshold selection should reflect offer costs and capacity.
- Permutation importance and per-customer indicator flags are explanatory aids, not causal attributions.
- Coupon scenario changes selected feature values only; it does not estimate intervention response or profit.

## Reproducibility

See `README.md` for setup and execution. Outputs are produced in `outputs/`; this report is `reports/final_report.md`.
