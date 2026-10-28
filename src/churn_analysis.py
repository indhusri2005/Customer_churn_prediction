"""Reproducible customer churn analysis for the FreshBasket loyalty dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


START = pd.Timestamp("2023-01-01")
CUTOFF = pd.Timestamp("2024-03-01")
LABEL_START = pd.Timestamp("2024-04-01")
LABEL_END = pd.Timestamp("2024-06-01")
RANDOM_STATE = 42

PROFILE_COLUMNS = [
    "AGE",
    "GENDER",
    "CITY",
    "MEMBERSHIP_TIER",
    "MARKETING_OPT_IN",
    "PREFERRED_CATEGORY",
    "HOME_STORE_PRICE_TIER",
]

BEHAVIOUR_COLUMNS = [
    "tenure_months",
    "activity_months_observed",
    "months_missing",
    "transactions_total",
    "transactions_mean_month",
    "transactions_recent3_mean",
    "transactions_previous3_mean",
    "transactions_recent_vs_previous",
    "spend_total",
    "spend_mean_month",
    "spend_recent3_mean",
    "spend_previous3_mean",
    "spend_recent_vs_previous",
    "average_basket_pre_label",
    "recency_months",
    "active_month_rate",
    "transactions_slope",
    "spend_slope",
    "distinct_categories_mean",
]

ENGAGEMENT_COLUMNS = [
    "promo_txn_pct_mean",
    "promo_txn_pct_recent3_mean",
    "app_sessions_mean",
    "app_sessions_recent3_mean",
    "emails_opened_mean",
    "emails_opened_recent3_mean",
    "coupons_redeemed_mean",
    "coupons_redeemed_recent3_mean",
]

SUPPORT_COLUMNS = [
    "support_tickets_mean",
    "support_tickets_recent3_mean",
    "support_tickets_slope",
    "complaint_month_rate",
    "complaint_recent3_rate",
]

ALL_FEATURES = PROFILE_COLUMNS + BEHAVIOUR_COLUMNS + ENGAGEMENT_COLUMNS + SUPPORT_COLUMNS


def load_tables(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three data tables; row 1 is the header in each worksheet."""
    profile = pd.read_excel(input_path, sheet_name="fb Customer Profile", header=1)
    activity = pd.read_excel(input_path, sheet_name="fb Monthly Activity", header=1)
    labels = pd.read_excel(input_path, sheet_name="fb Churn Label", header=1)
    return profile, activity, labels


def _normalize_profile(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.drop_duplicates().drop_duplicates("CUSTOMER_ID", keep="first").copy()
    for column in ["GENDER"]:
        profile[column] = profile[column].astype("string").str.strip().str.upper()
    for column in ["CITY", "MEMBERSHIP_TIER", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"]:
        profile[column] = (
            profile[column].astype("string").str.strip().str.title().replace({"<NA>": pd.NA})
        )
    for column in ["GENDER", "CITY", "MEMBERSHIP_TIER", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"]:
        profile[column] = profile[column].astype(object).where(profile[column].notna(), np.nan)
    profile["SIGNUP_DATE"] = pd.to_datetime(profile["SIGNUP_DATE"], errors="coerce")
    return profile


def _normalize_activity(activity: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    activity = activity.drop_duplicates().copy()
    activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    duplicate_keys = activity.duplicated(["CUSTOMER_ID", "MONTH"], keep=False)
    conflicting_duplicate_rows = int(
        activity.loc[duplicate_keys].groupby(["CUSTOMER_ID", "MONTH"]).ngroups
    )
    # Conflicting duplicate keys in this file are negative-spend copies of otherwise
    # identical rows. Prefer a nonnegative-spend copy, then retain one row per month.
    activity["_invalid_spend_copy"] = activity["TOTAL_SPEND"].lt(0)
    activity = activity.sort_values(
        ["_invalid_spend_copy", "CUSTOMER_ID", "MONTH"], kind="stable"
    ).drop_duplicates(["CUSTOMER_ID", "MONTH"], keep="first")
    activity = activity.drop(columns="_invalid_spend_copy")
    activity["TOTAL_SPEND"] = pd.to_numeric(activity["TOTAL_SPEND"], errors="coerce")
    activity.loc[activity["TOTAL_SPEND"] < 0, "TOTAL_SPEND"] = 0
    activity["TOTAL_SPEND"] = activity["TOTAL_SPEND"].fillna(0)
    activity["TRANSACTIONS"] = pd.to_numeric(activity["TRANSACTIONS"], errors="coerce").fillna(0)
    activity["AVG_BASKET_VALUE"] = pd.to_numeric(
        activity["AVG_BASKET_VALUE"], errors="coerce"
    ).fillna(0)
    positive_spend = activity.loc[activity["TOTAL_SPEND"] > 0, "TOTAL_SPEND"]
    spend_cap = float(positive_spend.quantile(0.99)) if len(positive_spend) else 0.0
    activity["TOTAL_SPEND"] = activity["TOTAL_SPEND"].clip(upper=spend_cap)
    return activity, conflicting_duplicate_rows


def _normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.drop_duplicates().drop_duplicates("CUSTOMER_ID", keep="first").copy()


def quality_summary(
    raw_profile: pd.DataFrame,
    raw_activity: pd.DataFrame,
    raw_labels: pd.DataFrame,
    activity_clean: pd.DataFrame,
    profile_clean: pd.DataFrame,
    labels_clean: pd.DataFrame,
    conflicting_activity_keys: int,
) -> pd.DataFrame:
    """Summarize input-quality checks without hiding the original issue counts."""
    expected_months = (
        raw_profile[["CUSTOMER_ID", "SIGNUP_DATE"]]
        .drop_duplicates("CUSTOMER_ID")
        .copy()
    )
    expected_months["SIGNUP_DATE"] = pd.to_datetime(expected_months["SIGNUP_DATE"])
    expected_months["first_month"] = expected_months["SIGNUP_DATE"].dt.to_period("M")
    expected_months["last_month"] = CUTOFF.to_period("M")
    expected_months["expected_month_count"] = (
        expected_months["last_month"] - expected_months["first_month"]
    ).apply(lambda value: max(0, value.n + 1))
    expected_months = expected_months.loc[expected_months["expected_month_count"] > 0]
    activity_pre = activity_clean.loc[activity_clean["MONTH"].between(START, CUTOFF)]
    recorded_months = activity_pre.groupby("CUSTOMER_ID")["MONTH"].nunique()
    expected_months["recorded_month_count"] = (
        expected_months["CUSTOMER_ID"].map(recorded_months).fillna(0).astype(int)
    )
    total_missing_months = int(
        (expected_months["expected_month_count"] - expected_months["recorded_month_count"])
        .clip(lower=0)
        .sum()
    )
    zero_txn = raw_activity["TRANSACTIONS"].eq(0)
    zero_txn_positive_spend = (
        zero_txn & pd.to_numeric(raw_activity["TOTAL_SPEND"], errors="coerce").gt(0)
    ).sum()
    positive_txn_nonpositive_spend = (
        raw_activity["TRANSACTIONS"].gt(0)
        & pd.to_numeric(raw_activity["TOTAL_SPEND"], errors="coerce").le(0)
    ).sum()
    positive_spend = pd.to_numeric(raw_activity["TOTAL_SPEND"], errors="coerce")
    q1, q3 = positive_spend[positive_spend > 0].quantile([0.25, 0.75])
    outlier_threshold = float(q3 + 1.5 * (q3 - q1))
    outliers = (positive_spend > outlier_threshold).sum()

    observed = labels_clean.copy()
    last_purchase = pd.to_datetime(observed["LAST_PURCHASE_DATE"], errors="coerce")
    expected_label = (last_purchase.notna() & last_purchase.lt(LABEL_START)).astype(int)
    mismatch = (
        observed["INSUFFICIENT_HISTORY_FLAG"].eq(0)
        & observed["CHURNED"].ne(expected_label)
    ).sum()

    rows = [
        ("Customer profile rows", len(raw_profile), "Original profile table row count."),
        ("Unique customers in profile", raw_profile["CUSTOMER_ID"].nunique(), "Expected one row per customer."),
        ("Exact duplicate profile rows", int(raw_profile.duplicated().sum()), "Removed before joining."),
        ("Duplicate profile customer keys", int(raw_profile.duplicated("CUSTOMER_ID", keep=False).sum()), "Repeated key rows; deduplicated to one profile."),
        ("Monthly activity rows", len(raw_activity), "Original monthly table row count."),
        ("Exact duplicate activity rows", int(raw_activity.duplicated().sum()), "Removed before monthly-key deduplication."),
        ("Duplicate customer-month keys", int(raw_activity.duplicated(["CUSTOMER_ID", "MONTH"], keep=False).sum()), "Resolved to one record per customer-month."),
        ("Conflicting duplicate customer-month groups", conflicting_activity_keys, "Prefer nonnegative-spend copy for duplicated records."),
        ("Negative spend rows", int(pd.to_numeric(raw_activity["TOTAL_SPEND"], errors="coerce").lt(0).sum()), "Set negative spend to zero after resolving duplicate sign-flip copies."),
        ("Zero-spend rows", int(pd.to_numeric(raw_activity["TOTAL_SPEND"], errors="coerce").eq(0).sum()), "Retained; zero spend is valid for no-purchase months."),
        ("Zero-transaction rows", int(zero_txn.sum()), "Checked against spend and labels."),
        ("Zero transactions with positive spend", int(zero_txn_positive_spend), "Potential transaction/spend inconsistency."),
        ("Positive transactions with nonpositive spend", int(positive_txn_nonpositive_spend), "Potential transaction/spend inconsistency; negative/zero spend retained as zero."),
        ("Monthly spend outliers (IQR rule)", int(outliers), f"Raw positive spend above ${outlier_threshold:.2f}; model input clipped at raw positive 99th percentile."),
        ("Unrecorded customer-months through Mar 2024", total_missing_months, "Expected from signup month through the pre-label cutoff; treated as zero recorded activity in features, not asserted to be data loss."),
        ("Missing AGE values", int(raw_profile["AGE"].isna().sum()), "Imputed inside model pipelines."),
        ("Missing GENDER values", int(raw_profile["GENDER"].isna().sum()), "Imputed as most frequent category inside model pipelines."),
        ("Missing HOME_STORE_PRICE_TIER values", int(raw_profile["HOME_STORE_PRICE_TIER"].isna().sum()), "Imputed as most frequent category inside model pipelines."),
        ("Duplicate churn-label rows", int(raw_labels.duplicated().sum()), "Removed before joining."),
        ("Insufficient-history customers", int(labels_clean["INSUFFICIENT_HISTORY_FLAG"].eq(1).sum()), "Excluded from model fitting and validation."),
        ("Churn-label mismatches among adequate-history customers", int(mismatch), "Flagged and excluded from supervised fitting; provided label otherwise retained."),
    ]
    return pd.DataFrame(rows, columns=["check", "count", "handling"])


def build_customer_dataset(
    profile_raw: pd.DataFrame, activity_raw: pd.DataFrame, labels_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    profile = _normalize_profile(profile_raw)
    labels = _normalize_labels(labels_raw)
    activity, conflicting_keys = _normalize_activity(activity_raw)
    quality = quality_summary(
        profile_raw, activity_raw, labels_raw, activity, profile, labels, conflicting_keys
    )

    features_activity = activity.loc[activity["MONTH"].between(START, CUTOFF)].copy()
    activity_groups = {
        customer_id: group.set_index("MONTH").sort_index()
        for customer_id, group in features_activity.groupby("CUSTOMER_ID")
    }

    records: list[dict] = []
    for row in profile.itertuples(index=False):
        customer_id = row.CUSTOMER_ID
        signup_date = row.SIGNUP_DATE
        if pd.isna(signup_date):
            continue
        first_month = max(START, signup_date.to_period("M").to_timestamp())
        months = pd.date_range(first_month, CUTOFF, freq="MS") if first_month <= CUTOFF else pd.DatetimeIndex([])
        source = activity_groups.get(customer_id, pd.DataFrame())
        monthly = source.reindex(months) if len(months) else pd.DataFrame(index=months)
        for column in [
            "TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE", "DISTINCT_CATEGORIES",
            "PROMO_TXN_PCT", "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED",
            "SUPPORT_TICKETS", "COMPLAINT_FLAG",
        ]:
            if column not in monthly:
                monthly[column] = 0
        monthly = monthly.fillna(0)
        values = monthly
        recent = values.tail(3)
        previous = values.iloc[-6:-3]
        purchase_months = values.index[values["TRANSACTIONS"].gt(0)]
        recency = int((CUTOFF.to_period("M") - purchase_months[-1].to_period("M")).n) if len(purchase_months) else len(values) + 1
        record = {
            "CUSTOMER_ID": customer_id,
            "tenure_months": max(0, (CUTOFF.to_period("M") - signup_date.to_period("M")).n),
            "activity_months_observed": len(source),
            "months_missing": max(0, len(months) - len(source)),
            "transactions_total": values["TRANSACTIONS"].sum(),
            "transactions_mean_month": values["TRANSACTIONS"].mean() if len(values) else 0,
            "transactions_recent3_mean": recent["TRANSACTIONS"].mean() if len(recent) else 0,
            "transactions_previous3_mean": previous["TRANSACTIONS"].mean() if len(previous) else 0,
            "transactions_recent_vs_previous": (
                recent["TRANSACTIONS"].mean() - previous["TRANSACTIONS"].mean()
                if len(recent) and len(previous) else 0
            ),
            "spend_total": values["TOTAL_SPEND"].sum(),
            "spend_mean_month": values["TOTAL_SPEND"].mean() if len(values) else 0,
            "spend_recent3_mean": recent["TOTAL_SPEND"].mean() if len(recent) else 0,
            "spend_previous3_mean": previous["TOTAL_SPEND"].mean() if len(previous) else 0,
            "spend_recent_vs_previous": (
                recent["TOTAL_SPEND"].mean() - previous["TOTAL_SPEND"].mean()
                if len(recent) and len(previous) else 0
            ),
            "average_basket_pre_label": (
                values.loc[values["TRANSACTIONS"].gt(0), "AVG_BASKET_VALUE"].mean()
                if values["TRANSACTIONS"].gt(0).any() else 0
            ),
            "recency_months": recency,
            "active_month_rate": (
                values["TRANSACTIONS"].gt(0).mean() if len(values) else 0
            ),
            "transactions_slope": (
                float(np.polyfit(np.arange(len(values)), values["TRANSACTIONS"], 1)[0])
                if len(values) > 1 else 0
            ),
            "spend_slope": (
                float(np.polyfit(np.arange(len(values)), values["TOTAL_SPEND"], 1)[0])
                if len(values) > 1 else 0
            ),
            "distinct_categories_mean": values["DISTINCT_CATEGORIES"].mean() if len(values) else 0,
            "promo_txn_pct_mean": values["PROMO_TXN_PCT"].mean() if len(values) else 0,
            "promo_txn_pct_recent3_mean": recent["PROMO_TXN_PCT"].mean() if len(recent) else 0,
            "app_sessions_mean": values["APP_SESSIONS"].mean() if len(values) else 0,
            "app_sessions_recent3_mean": recent["APP_SESSIONS"].mean() if len(recent) else 0,
            "emails_opened_mean": values["EMAILS_OPENED"].mean() if len(values) else 0,
            "emails_opened_recent3_mean": recent["EMAILS_OPENED"].mean() if len(recent) else 0,
            "coupons_redeemed_mean": values["COUPONS_REDEEMED"].mean() if len(values) else 0,
            "coupons_redeemed_recent3_mean": recent["COUPONS_REDEEMED"].mean() if len(recent) else 0,
            "support_tickets_mean": values["SUPPORT_TICKETS"].mean() if len(values) else 0,
            "support_tickets_recent3_mean": recent["SUPPORT_TICKETS"].mean() if len(recent) else 0,
            "support_tickets_slope": (
                float(np.polyfit(np.arange(len(values)), values["SUPPORT_TICKETS"], 1)[0])
                if len(values) > 1 else 0
            ),
            "complaint_month_rate": values["COMPLAINT_FLAG"].mean() if len(values) else 0,
            "complaint_recent3_rate": recent["COMPLAINT_FLAG"].mean() if len(recent) else 0,
        }
        records.append(record)

    engineered = pd.DataFrame(records).merge(profile, on="CUSTOMER_ID", how="left")
    engineered["SIGNUP_COHORT"] = engineered["SIGNUP_DATE"].dt.to_period("Q").astype(str)
    engineered["SIGNUP_YEAR"] = engineered["SIGNUP_DATE"].dt.year

    # Validate labels against the provided last-purchase date; sparse monthly
    # activity rows cannot establish whether an absent month is zero.
    last_purchase = pd.to_datetime(labels["LAST_PURCHASE_DATE"], errors="coerce")
    labels["_definition_label"] = (
        last_purchase.notna() & last_purchase.lt(LABEL_START)
    ).astype(int)
    labels["_label_mismatch"] = (
        labels["INSUFFICIENT_HISTORY_FLAG"].eq(0)
        & labels["CHURNED"].ne(labels["_definition_label"])
    )
    engineered = engineered.merge(
        labels[
            [
                "CUSTOMER_ID", "CHURNED", "MONTHS_OBSERVED",
                "INSUFFICIENT_HISTORY_FLAG", "OBSERVATION_END_DATE", "_label_mismatch",
            ]
        ],
        on="CUSTOMER_ID",
        how="left",
        validate="one_to_one",
    )
    engineered["SIGNUP_COHORT"] = engineered["SIGNUP_COHORT"].replace("NaT", "Unknown")
    run_info = {
        "activity_conflicting_duplicate_keys": conflicting_keys,
        "observation_cutoff": CUTOFF.strftime("%Y-%m-%d"),
        "label_window_start": LABEL_START.strftime("%Y-%m-%d"),
        "label_window_end": "2024-06-30",
        "spend_clip_quantile": 0.99,
        "random_state": RANDOM_STATE,
    }
    return engineered, quality, run_info


def _make_pipeline(model, numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    if isinstance(model, LogisticRegression):
        numeric.steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _score_row(name: str, y_true: pd.Series, probability: np.ndarray) -> dict:
    predicted = (probability >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "pr_auc": average_precision_score(y_true, probability),
        "threshold": 0.5,
    }


def _feature_profile(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [column for column in columns if frame[column].dtype == "object" or str(frame[column].dtype).startswith("string")]
    numeric = [column for column in columns if column not in categorical]
    return numeric, categorical


def _risk_indicators(row: pd.Series, feature_importance: pd.DataFrame) -> str:
    drivers: list[str] = []
    if row["recency_months"] >= 2:
        drivers.append("recent purchase recency")
    if row["transactions_recent_vs_previous"] < 0:
        drivers.append("declining transactions")
    if row["spend_recent_vs_previous"] < 0:
        drivers.append("declining spend")
    if row["support_tickets_recent3_mean"] > 0 or row["complaint_recent3_rate"] > 0:
        drivers.append("recent support/complaint activity")
    if row["app_sessions_recent3_mean"] == 0 and row["emails_opened_recent3_mean"] == 0:
        drivers.append("low recent digital engagement")
    if not drivers:
        drivers.append("no single elevated indicator; review overall model score")
    return "; ".join(drivers[:3])


def _write_charts(
    output_dir: Path,
    metrics: pd.DataFrame,
    segment_summary: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    if not metrics.empty:
        ax = metrics.set_index("model")[["precision", "recall", "f1", "pr_auc"]].plot(
            kind="bar", figsize=(10, 5), ylim=(0, 1), rot=20
        )
        ax.set_title("Cohort validation model comparison")
        ax.set_ylabel("Score")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(charts_dir / "model_comparison.png", dpi=150)
        plt.close()
    if not segment_summary.empty:
        tiers = segment_summary.loc[segment_summary["segment_type"].eq("MEMBERSHIP_TIER")]
        if not tiers.empty:
            ax = tiers.set_index("segment")[["observed_churn_rate", "mean_predicted_risk"]].plot(
                kind="bar", figsize=(8, 5), ylim=(0, 1), rot=0
            )
            ax.set_title("Churn by membership tier")
            ax.set_ylabel("Rate")
            plt.tight_layout()
            plt.savefig(charts_dir / "tier_churn.png", dpi=150)
            plt.close()
    if not importance.empty:
        top = importance.head(12).sort_values("importance_mean")
        ax = top.set_index("feature")["importance_mean"].plot(kind="barh", figsize=(9, 6))
        ax.set_title("Permutation feature importance (validation PR-AUC)")
        ax.set_xlabel("Mean decrease in average precision")
        plt.tight_layout()
        plt.savefig(charts_dir / "feature_importance.png", dpi=150)
        plt.close()


def run_analysis(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_profile, raw_activity, raw_labels = load_tables(input_path)
    customer_df, quality, run_info = build_customer_dataset(raw_profile, raw_activity, raw_labels)

    quality.to_csv(output_dir / "data_quality_summary.csv", index=False)
    customer_df.to_csv(output_dir / "customer_modeling_dataset.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    # A cohort holdout separates older signup cohorts from 2023-24 signups.
    valid_target = (
        customer_df["CHURNED"].notna()
        & customer_df["INSUFFICIENT_HISTORY_FLAG"].eq(0)
        & ~customer_df["_label_mismatch"].fillna(True)
    )
    train_mask = valid_target & (customer_df["SIGNUP_DATE"] < pd.Timestamp("2023-01-01"))
    validation_mask = valid_target & (customer_df["SIGNUP_DATE"] >= pd.Timestamp("2023-01-01"))
    train = customer_df.loc[train_mask].copy()
    validation = customer_df.loc[validation_mask].copy()
    if train.empty or validation.empty or train["CHURNED"].nunique() < 2 or validation["CHURNED"].nunique() < 2:
        raise ValueError(
            "Cohort split must contain both target classes in train and validation; "
            f"got train={len(train)}, validation={len(validation)}."
        )

    metrics_rows: list[dict] = []
    ablation_rows: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    model_specs = [
        (
            "Logistic regression (all features)",
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            ALL_FEATURES,
        ),
        (
            "Random forest (all features)",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            ALL_FEATURES,
        ),
    ]
    y_train = train["CHURNED"].astype(int)
    y_validation = validation["CHURNED"].astype(int)
    for name, estimator, columns in model_specs:
        numeric, categorical = _feature_profile(train, columns)
        model = _make_pipeline(estimator, numeric, categorical)
        model.fit(train[columns], y_train)
        probability = model.predict_proba(validation[columns])[:, 1]
        metrics_rows.append(_score_row(name, y_validation, probability))
        fitted_models[name] = model

    selected_name = max(metrics_rows, key=lambda row: row["pr_auc"])["model"]
    selected_model = fitted_models[selected_name]
    ablation_features = {
        "Demographics only": PROFILE_COLUMNS,
        "Demographics + behaviour": PROFILE_COLUMNS + BEHAVIOUR_COLUMNS,
        "Demographics + behaviour + engagement": PROFILE_COLUMNS + BEHAVIOUR_COLUMNS + ENGAGEMENT_COLUMNS,
        "All feature groups": ALL_FEATURES,
    }
    for name, columns in ablation_features.items():
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        numeric, categorical = _feature_profile(train, columns)
        model = _make_pipeline(estimator, numeric, categorical)
        model.fit(train[columns], y_train)
        probability = model.predict_proba(validation[columns])[:, 1]
        ablation_rows.append(_score_row(name, y_validation, probability))

    metrics = pd.DataFrame(metrics_rows)
    ablation = pd.DataFrame(ablation_rows)
    metrics.to_csv(output_dir / "model_comparison.csv", index=False)
    ablation.to_csv(output_dir / "ablation_study.csv", index=False)

    # Validation-only permutation importance; avoids interpreting training fit.
    permutation = permutation_importance(
        selected_model,
        validation[ALL_FEATURES],
        y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    # Keep segment risk comparisons on held-out customers to avoid in-sample scores.
    segment_frame = validation.copy()
    segment_frame["mean_predicted_risk"] = selected_model.predict_proba(
        validation[ALL_FEATURES]
    )[:, 1]
    segment_rows = []
    for column, segment_type in [
        ("MEMBERSHIP_TIER", "MEMBERSHIP_TIER"),
        ("CITY", "CITY"),
        ("SIGNUP_COHORT", "SIGNUP_COHORT"),
    ]:
        grouped = segment_frame.groupby(column, dropna=False).agg(
            customers=("CUSTOMER_ID", "size"),
            observed_churn_rate=("CHURNED", "mean"),
            mean_predicted_risk=("mean_predicted_risk", "mean"),
        )
        for segment, row in grouped.iterrows():
            segment_rows.append(
                {
                    "segment_type": segment_type,
                    "segment": "Unknown" if pd.isna(segment) else str(segment),
                    **row.to_dict(),
                }
            )
    segment_summary = pd.DataFrame(segment_rows)
    segment_summary.to_csv(output_dir / "segment_churn_summary.csv", index=False)

    # Scenario is a model-based counterfactual, not a causal treatment estimate.
    validation_predictions = selected_model.predict_proba(validation[ALL_FEATURES])[:, 1]
    scenario_base = validation.copy()
    scenario_base["baseline_probability"] = validation_predictions
    cutoff_risk = float(scenario_base["baseline_probability"].quantile(0.75))
    targeted = scenario_base["baseline_probability"] >= cutoff_risk
    scenario_counterfactual = validation.copy()
    scenario_counterfactual.loc[targeted, "promo_txn_pct_recent3_mean"] = (
        scenario_counterfactual.loc[targeted, "promo_txn_pct_recent3_mean"] + 0.15
    ).clip(upper=1.0)
    scenario_counterfactual.loc[targeted, "coupons_redeemed_recent3_mean"] += 1.0
    scenario_probability = selected_model.predict_proba(
        scenario_counterfactual[ALL_FEATURES]
    )[:, 1]
    scenario = pd.DataFrame(
        {
            "CUSTOMER_ID": validation["CUSTOMER_ID"].to_numpy(),
            "CHURNED": y_validation.to_numpy(),
            "baseline_probability": validation_predictions,
            "scenario_probability": scenario_probability,
            "probability_change": scenario_probability - validation_predictions,
            "targeted_for_scenario": targeted.to_numpy(),
        }
    )
    scenario.to_csv(output_dir / "retention_scenario_customer_estimates.csv", index=False)
    scenario_summary = {
        "scenario": "For the top-quartile validation risk group, increase recent promotion transaction share by 0.15 (capped at 1.0) and recent coupon redemptions by 1 per month.",
        "targeted_customers": int(targeted.sum()),
        "targeted_share": float(targeted.mean()),
        "targeted_baseline_mean_risk": float(scenario.loc[targeted.to_numpy(), "baseline_probability"].mean()),
        "targeted_scenario_mean_risk": float(scenario.loc[targeted.to_numpy(), "scenario_probability"].mean()),
        "targeted_mean_risk_change": float(scenario.loc[targeted.to_numpy(), "probability_change"].mean()),
        "all_validation_baseline_mean_risk": float(validation_predictions.mean()),
        "all_validation_scenario_mean_risk": float(scenario_probability.mean()),
        "all_validation_mean_risk_change": float((scenario_probability - validation_predictions).mean()),
        "interpretation": "Associational model counterfactual only; it is not evidence that coupons cause churn reduction. A randomized pilot is required to estimate treatment effect.",
    }
    (output_dir / "retention_scenario_summary.json").write_text(
        json.dumps(scenario_summary, indent=2), encoding="utf-8"
    )

    # Scores for all customer records; validation-cohort scores are held out.
    all_probability = selected_model.predict_proba(customer_df[ALL_FEATURES])[:, 1]
    predictions = customer_df[
        [
            "CUSTOMER_ID", "CHURNED", "MEMBERSHIP_TIER", "CITY", "SIGNUP_COHORT",
            "INSUFFICIENT_HISTORY_FLAG", "_label_mismatch",
        ]
    ].copy()
    predictions["churn_probability"] = all_probability
    predictions["predicted_label_at_0_5"] = (all_probability >= 0.5).astype(int)
    predictions["prediction_cohort"] = np.where(
        validation_mask, "held-out signup cohort",
        np.where(train_mask, "training cohort; in-sample", "excluded from evaluation")
    )
    predictions["key_drivers"] = customer_df.apply(
        lambda row: _risk_indicators(row, importance), axis=1
    )
    predictions["selected_model"] = selected_name
    predictions.to_csv(output_dir / "churn_predictions.csv", index=False)

    metadata = {
        **run_info,
        "selected_model_by_validation_pr_auc": selected_name,
        "train_customers": int(len(train)),
        "validation_customers": int(len(validation)),
        "train_churn_rate": float(y_train.mean()),
        "validation_churn_rate": float(y_validation.mean()),
        "excluded_insufficient_or_mismatched_labels": int((~valid_target).sum()),
        "validation_method": "Signup-cohort holdout: signups before 2023-01-01 train; signups on/after 2023-01-01 validate.",
        "feature_cutoff": "2024-03-31; only Jan 2023-Mar 2024 activity is used.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_charts(output_dir, metrics, segment_summary, importance)
    _write_report(output_dir, quality, metrics, ablation, importance, segment_summary, scenario_summary, metadata)
    return {
        "metrics": metrics,
        "ablation": ablation,
        "importance": importance,
        "segments": segment_summary,
        "scenario": scenario_summary,
        "metadata": metadata,
        "quality": quality,
    }


def _write_report(
    output_dir: Path,
    quality: pd.DataFrame,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    importance: pd.DataFrame,
    segments: pd.DataFrame,
    scenario: dict,
    metadata: dict,
) -> None:
    best = metrics.loc[metrics["pr_auc"].idxmax()]
    top_features = importance.head(8)
    top_tier = segments.loc[segments["segment_type"].eq("MEMBERSHIP_TIER")].sort_values(
        "observed_churn_rate", ascending=False
    )
    top_city = segments.loc[segments["segment_type"].eq("CITY")].sort_values(
        "observed_churn_rate", ascending=False
    ).head(3)
    top_cohort = segments.loc[segments["segment_type"].eq("SIGNUP_COHORT")].sort_values(
        "observed_churn_rate", ascending=False
    )

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame[columns].to_markdown(index=False, floatfmt=".3f")

    report = f"""# FreshBasket loyalty churn analysis

## Executive summary

The churn model uses customer profile and monthly activity available only through **March 2024** to predict the supplied April-June 2024 churn label. {metadata['train_customers']:,} adequately observed customers from older signup cohorts were used for training and {metadata['validation_customers']:,} customers from 2023-24 signup cohorts were held out. The best validation PR-AUC was **{best['pr_auc']:.3f}** from **{best['model']}** (precision {best['precision']:.3f}, recall {best['recall']:.3f}, F1 {best['f1']:.3f}, accuracy {best['accuracy']:.3f} at a fixed 0.50 threshold).

Use the score to prioritize outreach, not as a definitive churn decision. A small model-based coupon scenario is explicitly non-causal; validate it with a randomized pilot before funding a broad rollout.

## Data quality and preparation

{table(quality, ['check', 'count', 'handling'])}

Exact duplicates are removed; duplicated customer-months are reduced to one row, preferring nonnegative spend when the duplicate differs by sign. Negative spend is floored at zero and positive monthly spend is clipped at the raw positive 99th percentile for feature construction. Missing customer-months through March 2024 are treated as no recorded activity in monthly aggregates. This is an assumption, not proof of zero activity. Categorical case/whitespace is normalized; missing values are imputed within each training fold's pipeline.

The monthly activity data is sparse, and an absent row is treated as zero recorded activity for feature construction. The label-consistency check uses the supplied `LAST_PURCHASE_DATE`, since sparse monthly rows cannot independently prove whether an absent month contained a transaction. Customers with sufficient history whose label conflicts with this date-based check are excluded rather than silently relabeled.

## Validation and model comparison

The split is by signup cohort, not random: signup before 2023-01-01 is training, and signup on/after that date is validation. Only features dated January 2023 through March 2024 are used. The four metrics at threshold 0.50 are reported alongside PR-AUC to make class imbalance visible.

{table(metrics, ['model', 'accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'threshold'])}

### Ablation

{table(ablation, ['model', 'accuracy', 'precision', 'recall', 'f1', 'pr_auc'])}

### Main feature signals

The table below reports validation permutation importance (mean reduction in average precision). Importance is predictive association, not causality.

{table(top_features, ['feature', 'importance_mean', 'importance_std'])}

## Segment comparison (held-out cohort)

### Membership tier

{table(top_tier, ['segment', 'customers', 'observed_churn_rate', 'mean_predicted_risk'])}

### Cities with the highest observed churn rates

{table(top_city, ['segment', 'customers', 'observed_churn_rate', 'mean_predicted_risk'])}

### Signup cohorts

{table(top_cohort, ['segment', 'customers', 'observed_churn_rate', 'mean_predicted_risk'])}

Segment rates and average risks are calculated on the held-out signup cohort only. They are descriptive, not causal. Small cohorts can be noisy; use customer counts and confidence intervals before targeting a segment.

## Retention scenario

The scenario targets the top quartile of validation predicted risk and increases recent promotion transaction share by 0.15 (capped at 1.0) and coupon redemptions by one per month. Targeted population: {scenario['targeted_customers']:,} customers ({scenario['targeted_share']:.1%}). Mean predicted risk changes from {scenario['targeted_baseline_mean_risk']:.3f} to {scenario['targeted_scenario_mean_risk']:.3f} (change {scenario['targeted_mean_risk_change']:+.3f}).

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
"""
    report_path = output_dir.parent / "reports" / "final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("FreshBasket_Loyalty_Churn_Dataset.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    result = run_analysis(args.input, args.output_dir)
    print("Selected model:", result["metadata"]["selected_model_by_validation_pr_auc"])
    print("Model comparison:")
    print(result["metrics"].to_string(index=False))
    print("Outputs written to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
