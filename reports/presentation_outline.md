# FreshBasket Loyalty Churn — 10–15 Minute Presentation

## Slide 1 — The business question (1 minute)
- Can FreshBasket identify members at risk of stopping purchases before the April-June 2024 label window?
- Goal: prioritize retention outreach, understand signals, avoid blanket discounts.
- Speaker note: Emphasize that the model supports prioritization; it does not make final customer-treatment decisions.

## Slide 2 — Data and target (1–2 minutes)
- 2,600 loyalty profiles, monthly activity from January 2023 to June 2024, and a customer-level churn label.
- Churn: no purchases in April-June despite prior purchase history.
- Features stop at March 2024 to prevent label-window leakage.
- Speaker note: The activity table is sparse and the treatment of missing months is an explicit assumption.

## Slide 3 — Data quality and preparation (1–2 minutes)
- Duplicate profile and label rows; duplicated customer-month keys.
- Negative and zero spend, missing profile values, sparse customer-month coverage, and a small set of label-definition conflicts.
- Exact duplicates removed; negative spend zeroed; duplicated months resolved; outliers capped for feature use.
- Speaker note: Show `outputs/data_quality_summary.csv`; confirm the missing-month interpretation with the data owner.

## Slide 4 — Validation design and model comparison (2 minutes)
- Signup-cohort holdout: customers who joined before 2023 train; signups from 2023 onward validate.
- Compare class-weighted logistic regression with a class-weighted random forest.
- Report precision, recall, F1, accuracy and PR-AUC; PR-AUC is particularly useful with imbalanced labels.
- Chart: `outputs/charts/model_comparison.png`.
- Speaker note: This is cohort-based, not a later-time evaluation; both cohorts share the same label period.

## Slide 5 — What correlates with churn risk? (1–2 minutes)
- Validation permutation importance highlights the signals that change predictive performance.
- Compare observed churn rates across tier, city and signup quarter.
- Charts: `outputs/charts/feature_importance.png` and `outputs/charts/tier_churn.png`.
- Speaker note: Associations are not causes; avoid interpreting a feature as a retention lever without an experiment.

## Slide 6 — Feature ablation (1 minute)
- Compare demographics alone with behavior, then engagement and support added.
- Use the ablation results to decide whether richer activity data contributes useful ranking signal.
- Speaker note: Validation differences may reflect cohort shift and sampling noise; avoid overclaiming small gains.

## Slide 7 — Retention scenario (1–2 minutes)
- Target top-quartile validation risk.
- Counterfactual: recent promotion share +0.15 and coupon redemption +1 per month.
- Show predicted change from `outputs/retention_scenario_summary.json`.
- Speaker note: This is a model perturbation, not estimated causal uplift. Coupon response and margin are unobserved.

## Slide 8 — Recommended pilot and next steps (1–2 minutes)
- Run a randomized outreach/coupon pilot with an untreated holdout.
- Track incremental retention, gross margin, redemption, complaint resolution and customer experience.
- Improve month-level activity completeness; verify label conflicts; monitor calibration and performance by tier/city/cohort.
- Speaker note: Scale only if incremental value is positive and service capacity remains adequate.
