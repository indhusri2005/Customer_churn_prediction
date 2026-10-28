# FreshBasket Loyalty Churn Prediction

This project analyzes the provided FreshBasket loyalty dataset and builds a leakage-aware customer churn prediction baseline. It is designed to be reproducible from the original Excel workbook.

## Setup

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep `FreshBasket_Loyalty_Churn_Dataset.xlsx` in the project root (or pass another file with `--input`).

## Run

```powershell
python src\churn_analysis.py
```

Optional paths:

```powershell
python src\churn_analysis.py --input path\to\workbook.xlsx --output-dir outputs
```

The run creates:

- `outputs/customer_modeling_dataset.csv` — customer-level profile, target and engineered pre-label features.
- `outputs/data_quality_summary.csv` — quality checks and handling notes.
- `outputs/model_comparison.csv` — logistic regression and random-forest cohort-validation metrics.
- `outputs/ablation_study.csv` — demographic-only and incremental feature-group comparisons.
- `outputs/feature_importance.csv` — validation permutation importance.
- `outputs/segment_churn_summary.csv` — observed rate and model risk by tier, city and signup quarter.
- `outputs/churn_predictions.csv` — customer-level churn score, threshold label and risk-indicator flags. The `prediction_cohort` column distinguishes held-out from in-sample predictions.
- `outputs/retention_scenario_*.csv/json` — individual and aggregate coupon-feature counterfactual.
- `outputs/charts/` — validation and segment plots.
- `reports/final_report.md` — results, recommendations, assumptions and limitations.

## Modeling and data assumptions

- Churn is the supplied `CHURNED` label: no purchase transactions during April-June 2024 despite prior purchase history.
- Features use only monthly activity from January 2023 through March 2024. Target-window activity and label-window dates are never used as predictors.
- The cohort validation split trains on signups before 2023-01-01 and validates on signups on/after that date. It is not a random split, but both cohorts share the same outcome window; it is not a future-time evaluation.
- Exact duplicate rows are removed. Repeated customer-month keys are resolved to one record, preferring a nonnegative spend copy. Negative spend is floored at zero; positive spend is capped at the raw-data 99th percentile for features.
- Missing customer-months are treated as no recorded activity when features are generated. This assumption should be revisited with the data owner.
- Customers flagged for insufficient history and customers whose provided labels conflict with the written definition (checked against `LAST_PURCHASE_DATE`, not sparse monthly rows) are excluded from model fitting and validation. Their supplied values remain in the customer-level file.
- Categorical/numeric imputation occurs inside each model pipeline. The logistic-regression and random-forest models use class weighting. PR-AUC is included because accuracy alone can obscure churn-class performance.
- The prediction threshold is 0.50 for consistent metric reporting. Operational thresholding should account for offer capacity, cost, margin and retention value.
- The coupon scenario is an associational counterfactual; it is not a causal estimate. A randomized pilot is needed to measure incremental impact.

## EDA notebook

Open `notebooks/data_quality_and_eda.ipynb` from the project root. It summarizes source data, data-quality results, churn prevalence, validation metrics and segment patterns. The notebook reads the same data and outputs as the reproducible analysis script.

