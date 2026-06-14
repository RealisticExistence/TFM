from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config import FEATURE_ENGINEERING_OUTPUT_FULL_DIR, INVOICE_FORECASTER_DIR
from util import create_dir, write_csv
from models.clearing_model.clearing_date_model_lib import parse_dates, datetime_ordinal_days, safe_day_delta

RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.20
MAX_INVOICE_SEQUENCE = 8
SALESFORCE_MODEL_NAME = "salesforce_only"
SALESFORCE_SAP_MODEL_NAME = "salesforce_plus_sap"

RAW_SAP_COLUMNS = {
    "client_number",
    "so_date",
    "incoterms",
    "terms_of_payment",
    "wbs_element",
    "req_deliv_date",
    "net_value",
}
SAP_RELATED_PREFIXES = (
    "so_date_",
    "req_deliv_date_",
    "incoterms_",
    "terms_of_payment_",
    "hist_client__",
    "hist_terms_of_payment__",
)
SAP_RELATED_EXACT = {
    "sap_so_num",
    "sales_order_id",
    "hist_sales_order_amount",
    "hist_sales_order_net_value",
    "hist_sales_order_amount_converted",
    "hist_sales_order_expected_revenue",
}
ID_OR_TEXT_COLS = {
    "sales_order_id",
    "sap_so_num",
    "opportunity_id",
    "account_id",
    "sfdc_quote_num",
    "reference_no",
    "customer_po_num",
    "ifa_number",
    "wbs_element",
    "project",
    "description",
    "sales_comments",
    "opportunity_name",
}
TARGET_AND_OUTPUT_COLS = {
    "invoice_num",
    "invoice_type",
    "invoice_type_clean",
    "invoice_date",
    "invoice_date_parsed",
    "amount",
    "amount_numeric",
    "clearing_date",
    "payment_due_date",
    "actual_invoice_count",
    "target_invoice_count",
    "target_invoice_sequence",
    "target_days_to_invoice",
    "target_amount_abs_log1p",
    "target_amount_signed",
    "target_amount_abs",
    "target_amount_sign",
    "target_amount_abs_to_reference_value",
    "target_amount_abs_to_reference_value_log1p",
}
OUTPUT_LEAKAGE_PREFIXES = (
    "invoice_date_",
    "clearing_date_",
    "invoice_is_",
    "invoice_amount_",
    "invoice_cash_flow_",
    "days__",
    "cleared_after_",
)
DATE_LIKE_COLUMNS = {
    "so_date",
    "req_deliv_date",
    "close_date",
    "created_date",
    "bid_submission_date",
    "bid_validity",
}


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_full_input_output() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = FEATURE_ENGINEERING_OUTPUT_FULL_DIR / "input.csv"
    output_path = FEATURE_ENGINEERING_OUTPUT_FULL_DIR / "output.csv"
    if not input_path.exists() or not output_path.exists():
        raise FileNotFoundError(
            "Feature-engineered full dataset not found. Run feature_engineering/convert_feature_engineering_into_output.py first."
        )
    return pd.read_csv(input_path), pd.read_csv(output_path)


def model_output_dir(model_name: str) -> Path:
    return INVOICE_FORECASTER_DIR / model_name


def is_sap_related_column(col: str) -> bool:
    return col in RAW_SAP_COLUMNS or col in SAP_RELATED_EXACT or any(col.startswith(prefix) for prefix in SAP_RELATED_PREFIXES)


def select_salesforce_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not is_sap_related_column(c)]


def select_salesforce_sap_columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def reference_date_for_orders(df: pd.DataFrame, feature_set: str) -> pd.Series:
    if feature_set == SALESFORCE_SAP_MODEL_NAME and "so_date" in df.columns:
        so_date = parse_dates(df["so_date"])
        if so_date.notna().any():
            return so_date
    for col in ["created_date", "close_date", "so_date"]:
        if col in df.columns:
            parsed = parse_dates(df[col])
            if parsed.notna().any():
                return parsed
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def clean_amount(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.replace("EUR", "", regex=False).str.strip()
    text = text.str.replace("_", ".", regex=False)
    text = text.str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")




def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def amount_reference_value(df: pd.DataFrame, feature_set: str) -> pd.Series:
    """Return a non-leaking order-value denominator for amount normalization.

    Salesforce-only models cannot use SAP-only values such as net_value as the
    amount denominator. The SAP-enhanced model may use net_value because it is
    available at prediction time. The returned value is an absolute scale, not a
    target derived from invoice rows.
    """
    if feature_set == SALESFORCE_SAP_MODEL_NAME:
        candidates = ["net_value", "salesforce_amount", "amount_converted", "expected_revenue_converted", "expected_revenue"]
    else:
        candidates = ["salesforce_amount", "amount_converted", "expected_revenue_converted", "expected_revenue"]

    ref = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            values = clean_amount(df[col]).abs()
            ref = ref.fillna(values)
    return ref.replace(0, np.nan)


def add_duration_feature(df: pd.DataFrame, start_col: str, end_col: str, output_col: str) -> None:
    if start_col in df.columns and end_col in df.columns:
        df[output_col] = safe_day_delta(parse_dates(df[start_col]), parse_dates(df[end_col]))


def add_ratio_feature(df: pd.DataFrame, numerator_col: str, denominator_col: str, output_col: str) -> None:
    if numerator_col in df.columns and denominator_col in df.columns:
        df[output_col] = safe_divide(clean_amount(df[numerator_col]), clean_amount(df[denominator_col]))

def add_invoice_sequence(output: pd.DataFrame) -> pd.DataFrame:
    out = output.copy()
    out["invoice_date_parsed"] = parse_dates(out["invoice_date"])
    out["amount_numeric"] = clean_amount(out["amount"])
    out = out.sort_values(["sales_order_id", "invoice_date_parsed", "invoice_num"], kind="mergesort")
    out["target_invoice_sequence"] = out.groupby("sales_order_id").cumcount() + 1
    out["actual_invoice_count"] = out.groupby("sales_order_id")["sales_order_id"].transform("size")
    return out


def build_order_table(input_df: pd.DataFrame, output_df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    orders = input_df.copy()
    orders["sales_order_id"] = orders["sales_order_id"].astype(str)
    if "amount" in orders.columns and "salesforce_amount" not in orders.columns:
        orders["salesforce_amount"] = orders["amount"]
    counts = output_df.groupby(output_df["sales_order_id"].astype(str)).size().rename("target_invoice_count")
    orders = orders.merge(counts, left_on="sales_order_id", right_index=True, how="left")
    orders["target_invoice_count"] = orders["target_invoice_count"].fillna(0).astype(int)
    orders["model_reference_date"] = reference_date_for_orders(orders, feature_set)
    return orders


def build_invoice_training_table(input_df: pd.DataFrame, output_df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    orders = build_order_table(input_df, output_df, feature_set)
    invoices = add_invoice_sequence(output_df)
    merged = invoices.merge(orders, on="sales_order_id", how="left", suffixes=("", "__order"))
    if "amount__order" in merged.columns and "salesforce_amount" not in merged.columns:
        merged["salesforce_amount"] = merged["amount__order"]
    merged["model_reference_date"] = parse_dates(merged["model_reference_date"])
    merged["target_days_to_invoice"] = safe_day_delta(merged["model_reference_date"], merged["invoice_date_parsed"])
    merged["target_amount_signed"] = pd.to_numeric(merged["amount_numeric"], errors="coerce")
    merged["target_amount_abs"] = merged["target_amount_signed"].abs()
    merged["target_amount_abs_log1p"] = np.log1p(merged["target_amount_abs"].clip(lower=0))
    merged["target_amount_sign"] = np.sign(merged["target_amount_signed"].fillna(0))
    merged["model_amount_reference_value"] = amount_reference_value(merged, feature_set)
    merged["model_amount_reference_value_log1p"] = np.log1p(merged["model_amount_reference_value"].abs())
    merged["target_amount_abs_to_reference_value"] = safe_divide(
        merged["target_amount_abs"],
        merged["model_amount_reference_value"],
    )
    merged["target_amount_abs_to_reference_value_log1p"] = np.log1p(
        merged["target_amount_abs_to_reference_value"].clip(lower=0)
    )
    return merged


def temporal_order_split(orders: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dates = parse_dates(orders["model_reference_date"])
    fallback = pd.Series(pd.Timestamp("1900-01-01"), index=orders.index, dtype="datetime64[ns]")
    dates = dates.fillna(fallback)
    order = np.argsort(dates.to_numpy(dtype="datetime64[ns]"))
    n_test = max(1, int(math.ceil(len(orders) * HOLDOUT_FRACTION)))
    return order[:-n_test], order[-n_test:]


def add_common_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Numeric value transforms.
    for col in ["salesforce_amount", "amount_converted", "expected_revenue", "expected_revenue_converted", "net_value"]:
        if col in out.columns:
            num = clean_amount(out[col])
            out[f"model_{col}_numeric"] = num
            out[f"model_{col}_abs_log1p"] = np.log1p(num.abs())
            out[f"model_{col}_is_missing"] = num.isna().astype(int)

    # Commercial and SAP value ratios. These are scale-normalized and tend to help
    # amount models more than raw currency values alone.
    add_ratio_feature(out, "expected_revenue", "salesforce_amount", "model_expected_revenue_to_salesforce_amount")
    add_ratio_feature(out, "expected_revenue_converted", "amount_converted", "model_expected_revenue_to_amount_converted")
    add_ratio_feature(out, "net_value", "salesforce_amount", "model_net_value_to_salesforce_amount")
    add_ratio_feature(out, "net_value", "amount_converted", "model_net_value_to_amount_converted")
    add_ratio_feature(out, "salesforce_amount", "net_value", "model_salesforce_amount_to_net_value")

    # Opportunity/order lifecycle durations.
    add_duration_feature(out, "created_date", "close_date", "model_days_created_to_close")
    add_duration_feature(out, "created_date", "so_date", "model_days_created_to_so")
    add_duration_feature(out, "close_date", "so_date", "model_days_close_to_so")
    add_duration_feature(out, "so_date", "req_deliv_date", "model_days_so_to_req_delivery")
    add_duration_feature(out, "close_date", "req_deliv_date", "model_days_close_to_req_delivery")
    add_duration_feature(out, "bid_submission_date", "bid_validity", "model_days_bid_submission_to_validity")
    add_duration_feature(out, "created_date", "req_deliv_date", "model_days_created_to_req_delivery")

    # Date availability/maturity flags.
    for col in ["created_date", "close_date", "so_date", "req_deliv_date", "bid_submission_date", "bid_validity"]:
        if col in out.columns:
            parsed = parse_dates(out[col])
            out[f"model_{col}_is_missing"] = parsed.isna().astype(int)
            out[f"model_{col}_month"] = parsed.dt.month
            out[f"model_{col}_quarter"] = parsed.dt.quarter
            out[f"model_{col}_dayofweek"] = parsed.dt.dayofweek
            out[f"model_{col}_is_month_end"] = parsed.dt.is_month_end.fillna(False).astype(int)
            out[f"model_{col}_is_quarter_end"] = parsed.dt.is_quarter_end.fillna(False).astype(int)

    # Invoice sequence context. During holdout generation this uses the predicted
    # count; during conditional invoice-row validation, only sequence-local values are used.
    if "target_invoice_sequence" in out.columns:
        seq = pd.to_numeric(out["target_invoice_sequence"], errors="coerce")
        out["model_invoice_sequence"] = seq
        out["model_invoice_sequence_log1p"] = np.log1p(seq)
        out["model_invoice_is_first"] = (seq == 1).astype(int)
        out["model_invoice_sequence_bucket"] = pd.cut(seq, bins=[0, 1, 2, 4, 8, np.inf], labels=["1", "2", "3-4", "5-8", "9+"]).astype("string")
    count_col = None
    if "predicted_invoice_count" in out.columns:
        count_col = "predicted_invoice_count"
    elif "sf_predicted_invoice_count" in out.columns:
        count_col = "sf_predicted_invoice_count"
    if count_col is not None:
        count = pd.to_numeric(out[count_col], errors="coerce")
        out["model_predicted_invoice_count"] = count
        out["model_predicted_invoice_count_log1p"] = np.log1p(count.clip(lower=0))
        if "target_invoice_sequence" in out.columns:
            seq = pd.to_numeric(out["target_invoice_sequence"], errors="coerce")
            out["model_invoice_sequence_fraction"] = safe_divide(seq, count)
            out["model_invoice_sequence_remaining"] = count - seq
            out["model_invoice_is_predicted_last"] = (seq == count).astype(int)
    return out

def build_feature_frame(df: pd.DataFrame, feature_set: str, extra_drop: set[str] | None = None) -> pd.DataFrame:
    out = add_common_model_features(df)
    out["model_amount_reference_value"] = amount_reference_value(out, feature_set)
    out["model_amount_reference_value_log1p"] = np.log1p(out["model_amount_reference_value"].abs())
    if feature_set == SALESFORCE_MODEL_NAME:
        allowed = set(select_salesforce_columns(out))
        # Keep model helper features, even when they are derived after column selection.
        allowed.update(c for c in out.columns if c.startswith("model_") or c.startswith("sf_predicted_"))
        out = out[[c for c in out.columns if c in allowed]].copy()

    drop_cols = set(TARGET_AND_OUTPUT_COLS) | ID_OR_TEXT_COLS
    if extra_drop:
        drop_cols |= set(extra_drop)
    X = out.drop(
        columns=[
            c for c in out.columns
            if c in drop_cols
            or c.startswith("Unnamed:")
            or c.endswith("__order")
            or any(c.startswith(prefix) for prefix in OUTPUT_LEAKAGE_PREFIXES)
        ],
        errors="ignore",
    ).copy()

    for col in list(X.columns):
        if col in DATE_LIKE_COLUMNS or col.endswith("_date") or col.endswith("_date__order") or col == "model_reference_date":
            parsed = parse_dates(X[col])
            if parsed.notna().any():
                X[f"{col}__ordinal_day"] = datetime_ordinal_days(parsed)
                X[f"{col}__month"] = parsed.dt.month
                X[f"{col}__dayofweek"] = parsed.dt.dayofweek
                X.drop(columns=[col], inplace=True)

    X = X.drop(columns=[c for c in X.columns if X[c].isna().all()], errors="ignore")
    for col in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])):
            X[col] = X[col].where(X[col].notna(), "__MISSING__").astype(str)
    return X


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


def make_pipeline(estimator: BaseEstimator, X: pd.DataFrame) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(X)), ("model", estimator)])


def count_model_spec() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=40, max_features=0.85, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1)


def date_model_specs() -> dict[str, dict[str, Any]]:
    """Candidate invoice-date offset models.

    Unlike clearing delays, invoice offsets can occasionally be negative because
    an invoice may pre-date the chosen reference date. Therefore the comparison
    includes:
    - raw days, with no target transform;
    - shifted log days, analogous to the clearing-date log target;
    - signed-log days, which compresses large offsets while preserving negative values.
    """
    return {
        "extra_trees_smooth_raw_offset": {
            "estimator": ExtraTreesRegressor(
                n_estimators=40,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "raw",
        },
        "extra_trees_flexible_raw_offset": {
            "estimator": ExtraTreesRegressor(
                n_estimators=50,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "raw",
        },
        "gradient_boosting_raw_offset": {
            "estimator": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            "target_transform": "raw",
        },
        "extra_trees_smooth_log_offset": {
            "estimator": ExtraTreesRegressor(
                n_estimators=40,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "shifted_log",
        },
        "extra_trees_flexible_log_offset": {
            "estimator": ExtraTreesRegressor(
                n_estimators=50,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "shifted_log",
        },
        "gradient_boosting_log_offset": {
            "estimator": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            "target_transform": "shifted_log",
        },
        "extra_trees_smooth_signed_log_offset": {
            "estimator": ExtraTreesRegressor(
                n_estimators=40,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "signed_log",
        },
        "extra_trees_flexible_signed_log_offset": {
            "estimator": ExtraTreesRegressor(
                n_estimators=50,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "signed_log",
        },
        "gradient_boosting_signed_log_offset": {
            "estimator": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            "target_transform": "signed_log",
        },
        "decision_tree_log_offset": {
            "estimator": DecisionTreeRegressor(
                max_depth=8,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            ),
            "target_transform": "shifted_log",
        },
    }


def fit_transformed_regressor(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    target_transform: str,
) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    """Fit a regressor under a named target transform and return predictions in days."""
    y_train = np.asarray(y_train, dtype=float)
    metadata: dict[str, Any] = {"target_transform": target_transform}
    if target_transform == "raw":
        y_model = y_train
    elif target_transform == "shifted_log":
        shift = max(0.0, -float(np.nanmin(y_train)))
        metadata["target_shift"] = shift
        y_model = np.log1p(np.clip(y_train + shift, 0, None))
    elif target_transform == "signed_log":
        y_model = np.sign(y_train) * np.log1p(np.abs(y_train))
    else:
        raise ValueError(f"Unknown target transform: {target_transform}")

    model = make_pipeline(clone(estimator), X_train)
    model.fit(X_train, y_model)
    pred_model = model.predict(X_test)
    pred = inverse_transformed_prediction(pred_model, metadata)
    return model, pred, metadata


def inverse_transformed_prediction(prediction: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    transform = metadata.get("target_transform", "raw")
    pred = np.asarray(prediction, dtype=float)
    if transform == "raw":
        return pred
    if transform == "shifted_log":
        return np.expm1(pred) - float(metadata.get("target_shift", 0.0))
    if transform == "signed_log":
        return np.sign(pred) * np.expm1(np.abs(pred))
    raise ValueError(f"Unknown target transform: {transform}")


def amount_model_specs() -> dict[str, dict[str, Any]]:
    """Candidate invoice-amount models.

    The *_log_amount models predict log(1 + absolute invoice amount). The
    *_log_amount_ratio models normalize the target by an order-value reference
    before fitting, then multiply the predicted ratio back by that reference.
    That is the amount-side equivalent of the log-delay transformation used in
    the clearing-date model.
    """
    return {
        "extra_trees_smooth_log_amount": {
            "estimator": ExtraTreesRegressor(
                n_estimators=40,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "log_abs",
        },
        "extra_trees_flexible_log_amount": {
            "estimator": ExtraTreesRegressor(
                n_estimators=50,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "log_abs",
        },
        "gradient_boosting_log_amount": {
            "estimator": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            "target_transform": "log_abs",
        },
        "extra_trees_smooth_log_amount_ratio": {
            "estimator": ExtraTreesRegressor(
                n_estimators=40,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "log_ratio_to_order_value",
        },
        "extra_trees_flexible_log_amount_ratio": {
            "estimator": ExtraTreesRegressor(
                n_estimators=50,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "target_transform": "log_ratio_to_order_value",
        },
        "gradient_boosting_log_amount_ratio": {
            "estimator": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            "target_transform": "log_ratio_to_order_value",
        },
        "decision_tree_log_amount_ratio": {
            "estimator": DecisionTreeRegressor(
                max_depth=8,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            ),
            "target_transform": "log_ratio_to_order_value",
        },
    }


def fit_amount_regressor(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    train_rows: pd.DataFrame,
    X_test: pd.DataFrame,
    test_rows: pd.DataFrame,
    feature_set: str,
    target_transform: str,
) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    y_abs = pd.to_numeric(train_rows["target_amount_abs"], errors="coerce")
    metadata: dict[str, Any] = {"target_transform": target_transform}

    if target_transform == "log_abs":
        valid = y_abs.notna()
        y_model = np.log1p(y_abs.loc[valid].clip(lower=0))
    elif target_transform == "log_ratio_to_order_value":
        ref_train = amount_reference_value(train_rows, feature_set)
        ratio = safe_divide(y_abs, ref_train)
        valid = ratio.notna() & np.isfinite(ratio)
        y_model = np.log1p(ratio.loc[valid].clip(lower=0))
        metadata["reference_median"] = float(ref_train.loc[valid].median()) if valid.any() else np.nan
    else:
        raise ValueError(f"Unknown amount target transform: {target_transform}")

    model = make_pipeline(clone(estimator), X_train.loc[valid])
    model.fit(X_train.loc[valid], y_model)
    pred_model = model.predict(X_test)
    pred = inverse_amount_prediction(pred_model, test_rows, metadata, feature_set)
    return model, pred, metadata


def inverse_amount_prediction(
    prediction: np.ndarray,
    rows: pd.DataFrame,
    metadata: dict[str, Any],
    feature_set: str,
) -> np.ndarray:
    transform = metadata.get("target_transform", "log_abs")
    pred = np.asarray(prediction, dtype=float)
    if transform == "log_abs":
        return np.clip(np.expm1(pred), 0, None)
    if transform == "log_ratio_to_order_value":
        ref = amount_reference_value(rows, feature_set)
        fallback = float(metadata.get("reference_median", np.nan))
        ref = ref.fillna(fallback)
        return np.clip(np.expm1(pred), 0, None) * ref.to_numpy(dtype=float)
    raise ValueError(f"Unknown amount target transform: {transform}")


def type_model_specs() -> dict[str, BaseEstimator]:
    return {
        "extra_trees_classifier": ExtraTreesClassifier(
            n_estimators=40,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "decision_tree_classifier": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
   }

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "bias_mean_error": float(np.mean(err)),
    }


def fit_log_regressor(estimator: BaseEstimator, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    model = make_pipeline(clone(estimator), X_train)
    model.fit(X_train, np.log1p(np.clip(y_train, 0, None)))
    pred = np.expm1(model.predict(X_test))
    return model, np.clip(pred, 0, None)


def fit_count_model(orders: pd.DataFrame, train_orders: pd.Series, test_orders: pd.Series, feature_set: str, sf_count_features: pd.DataFrame | None = None) -> tuple[Pipeline, pd.DataFrame, dict[str, float]]:
    data = orders.copy()
    if sf_count_features is not None:
        data = data.merge(sf_count_features, on="sales_order_id", how="left")
    X = build_feature_frame(data, feature_set)
    y = orders["target_invoice_count"].to_numpy(dtype=float)
    train_mask = orders["sales_order_id"].astype(str).isin(train_orders.astype(str)).to_numpy()
    test_mask = orders["sales_order_id"].astype(str).isin(test_orders.astype(str)).to_numpy()
    model, pred = fit_log_regressor(count_model_spec(), X.loc[train_mask], y[train_mask], X.loc[test_mask])
    pred_round = np.clip(np.rint(pred), 0, MAX_INVOICE_SEQUENCE).astype(int)
    actual = y[test_mask]
    metrics = evaluate_regression(actual, pred)
    metrics.update({
        "exact_count_accuracy": float(np.mean(pred_round == actual)),
        "within_one_count_accuracy": float(np.mean(np.abs(pred_round - actual) <= 1)),
        "n_test_orders": int(test_mask.sum()),
    })
    holdout = orders.loc[test_mask, ["sales_order_id", "target_invoice_count"]].copy()
    holdout["predicted_invoice_count_raw"] = pred
    holdout["predicted_invoice_count"] = pred_round
    return model, holdout, metrics


def train_attribute_models(
    invoice_rows: pd.DataFrame,
    orders: pd.DataFrame,
    train_orders: pd.Series,
    test_orders: pd.Series,
    feature_set: str,
    sf_sequence_features: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows = invoice_rows.copy()
    if sf_sequence_features is not None:
        keys = ["sales_order_id", "target_invoice_sequence"]
        rows = rows.merge(sf_sequence_features, on=keys, how="left")
    valid = rows["target_days_to_invoice"].notna() & rows["target_amount_abs_log1p"].notna() & rows["invoice_type"].notna()
    rows = rows.loc[valid].copy()
    train_mask = rows["sales_order_id"].astype(str).isin(train_orders.astype(str)).to_numpy()
    test_mask = rows["sales_order_id"].astype(str).isin(test_orders.astype(str)).to_numpy()

    X = build_feature_frame(rows, feature_set)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_date = rows["target_days_to_invoice"].to_numpy(dtype=float)
    y_date_test = y_date[test_mask]
    y_amount_log = rows["target_amount_abs_log1p"].to_numpy(dtype=float)
    y_amount_abs = rows["target_amount_abs"].to_numpy(dtype=float)
    y_type = rows["invoice_type"].astype(str).to_numpy()

    metric_rows: list[dict[str, Any]] = []
    fitted_date: dict[str, Pipeline] = {}
    date_predictions: dict[str, np.ndarray] = {}
    date_model_metadata: dict[str, dict[str, Any]] = {}
    for name, spec in date_model_specs().items():
        model, pred, metadata = fit_transformed_regressor(
            spec["estimator"],
            X_train,
            y_date[train_mask],
            X_test,
            spec["target_transform"],
        )
        fitted_date[name] = model
        date_predictions[name] = pred
        date_model_metadata[name] = metadata
        metrics = evaluate_regression(y_date_test, pred)
        metrics.update({
            "target": "invoice_date_offset_days",
            "model": name,
            "target_transform": spec["target_transform"],
        })
        metric_rows.append(metrics)

    fitted_amount: dict[str, Pipeline] = {}
    amount_predictions: dict[str, np.ndarray] = {}
    amount_model_metadata: dict[str, dict[str, Any]] = {}
    train_rows_for_amount = rows.loc[train_mask].copy()
    test_rows_for_amount = rows.loc[test_mask].copy()
    for name, spec in amount_model_specs().items():
        model, pred, metadata = fit_amount_regressor(
            spec["estimator"],
            X_train,
            train_rows_for_amount,
            X_test,
            test_rows_for_amount,
            feature_set,
            spec["target_transform"],
        )
        pred = np.clip(pred, 0, None)
        fitted_amount[name] = model
        amount_predictions[name] = pred
        amount_model_metadata[name] = metadata
        metrics = evaluate_regression(y_amount_abs[test_mask], pred)
        metrics.update({
            "target": "invoice_amount_abs",
            "model": name,
            "target_transform": spec["target_transform"],
        })
        metric_rows.append(metrics)

    fitted_type: dict[str, Pipeline] = {}
    type_predictions: dict[str, np.ndarray] = {}
    for name, estimator in type_model_specs().items():
        model = make_pipeline(clone(estimator), X_train)
        model.fit(X_train, y_type[train_mask])
        pred = model.predict(X_test)
        fitted_type[name] = model
        type_predictions[name] = pred
        metric_rows.append({
            "target": "invoice_type",
            "model": name,
            "accuracy": float(accuracy_score(y_type[test_mask], pred)),
            "macro_f1": float(f1_score(y_type[test_mask], pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_type[test_mask], pred, average="weighted", zero_division=0)),
            "n_test_invoices": int(test_mask.sum()),
        })

    metric_df = pd.DataFrame(metric_rows)
    best_date_model = metric_df.loc[metric_df["target"] == "invoice_date_offset_days"].sort_values(["rmse", "mae"]).iloc[0]["model"]
    best_amount_model = metric_df.loc[metric_df["target"] == "invoice_amount_abs"].sort_values(["rmse", "mae"]).iloc[0]["model"]
    best_type_model = metric_df.loc[metric_df["target"] == "invoice_type"].sort_values(["weighted_f1", "accuracy"], ascending=[False, False]).iloc[0]["model"]

    holdout = rows.loc[test_mask, ["sales_order_id", "invoice_num", "invoice_type", "invoice_date", "amount", "target_invoice_sequence", "target_days_to_invoice", "target_amount_abs"]].copy()
    holdout["predicted_days_to_invoice"] = date_predictions[best_date_model]
    holdout["predicted_amount_abs"] = amount_predictions[best_amount_model]
    holdout["predicted_invoice_type"] = type_predictions[best_type_model]
    holdout["date_error_days"] = holdout["predicted_days_to_invoice"] - holdout["target_days_to_invoice"]
    holdout["amount_error_abs"] = holdout["predicted_amount_abs"] - holdout["target_amount_abs"]

    fitted = {
        "date_models": fitted_date,
        "amount_models": fitted_amount,
        "type_models": fitted_type,
        "type_model": fitted_type[best_type_model],
        "best_date_model": str(best_date_model),
        "best_amount_model": str(best_amount_model),
        "best_type_model": str(best_type_model),
        "date_model_metadata": date_model_metadata,
        "amount_model_metadata": amount_model_metadata,
        "feature_columns": list(X.columns),
    }
    return fitted, metric_df, holdout


def build_sequence_prediction_features(
    orders: pd.DataFrame,
    count_predictions: pd.DataFrame,
    feature_set: str,
    sf_sequence_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    counts = count_predictions[["sales_order_id", "predicted_invoice_count"]].copy()
    counts["predicted_invoice_count"] = counts["predicted_invoice_count"].fillna(0).astype(int).clip(0, MAX_INVOICE_SEQUENCE)
    base = orders.merge(counts, on="sales_order_id", how="left")
    base["predicted_invoice_count"] = base["predicted_invoice_count"].fillna(0).astype(int)
    for _, order in base.iterrows():
        n = int(order["predicted_invoice_count"])
        for seq in range(1, n + 1):
            row = order.to_dict()
            row["target_invoice_sequence"] = seq
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    pred_rows = pd.DataFrame(rows)
    if sf_sequence_features is not None:
        pred_rows = pred_rows.merge(sf_sequence_features, on=["sales_order_id", "target_invoice_sequence"], how="left")
    return pred_rows


def predict_invoice_attributes(rows: pd.DataFrame, fitted: dict[str, Any], feature_set: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["sales_order_id", "predicted_invoice_sequence", "predicted_invoice_type", "predicted_invoice_date", "predicted_amount"])
    X = build_feature_frame(rows, feature_set)
    date_model = fitted["date_models"][fitted["best_date_model"]]
    amount_model = fitted["amount_models"][fitted["best_amount_model"]]
    type_model = fitted["type_model"]
    date_metadata = fitted.get("date_model_metadata", {}).get(fitted["best_date_model"], {"target_transform": "shifted_log", "target_shift": fitted.get("date_target_shift", 0.0)})
    pred_days = inverse_transformed_prediction(date_model.predict(X), date_metadata)
    amount_metadata = fitted.get("amount_model_metadata", {}).get(fitted["best_amount_model"], {"target_transform": "log_abs"})
    pred_amount = inverse_amount_prediction(amount_model.predict(X), rows, amount_metadata, feature_set)
    pred_type = type_model.predict(X)
    ref = parse_dates(rows["model_reference_date"])
    pred_date = ref + pd.to_timedelta(np.rint(pred_days).astype("int64"), unit="D")
    out = pd.DataFrame({
        "sales_order_id": rows["sales_order_id"].astype(str).to_numpy(),
        "predicted_invoice_sequence": rows["target_invoice_sequence"].astype(int).to_numpy(),
        "predicted_invoice_type": pred_type,
        "predicted_invoice_date": pd.to_datetime(pred_date, errors="coerce").dt.strftime("%Y-%m-%d"),
        "predicted_amount": np.round(np.clip(pred_amount, 0, None), 2),
    })
    return out


def train_salesforce_model(input_df: pd.DataFrame, output_df: pd.DataFrame) -> dict[str, Any]:
    return train_invoice_model(input_df, output_df, SALESFORCE_MODEL_NAME)


def make_salesforce_prediction_features(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    train_orders: pd.Series,
    test_orders: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # Prefer the previously saved Salesforce-only forecast when it exists. This
    # keeps the Salesforce+SAP step sequential without retraining the Salesforce
    # model inside every downstream script.
    sf_dir = model_output_dir(SALESFORCE_MODEL_NAME)
    saved_predictions = sf_dir / "predicted_invoices.csv"
    saved_report = sf_dir / "invoice_forecaster_report.json"
    if saved_predictions.exists():
        seq_preds = pd.read_csv(saved_predictions)
        order_ids = input_df[["sales_order_id"]].copy()
        order_ids["sales_order_id"] = order_ids["sales_order_id"].astype(str)
        if seq_preds.empty:
            counts = order_ids.copy()
            counts["sf_predicted_invoice_count"] = 0
        else:
            seq_preds["sales_order_id"] = seq_preds["sales_order_id"].astype(str)
            counts = seq_preds.groupby("sales_order_id").size().rename("sf_predicted_invoice_count").reset_index()
            counts = order_ids.merge(counts, on="sales_order_id", how="left")
            counts["sf_predicted_invoice_count"] = counts["sf_predicted_invoice_count"].fillna(0)
        counts["sf_predicted_invoice_count_raw"] = counts["sf_predicted_invoice_count"]
        sf_seq_features = seq_preds.rename(columns={
            "predicted_invoice_sequence": "target_invoice_sequence",
            "predicted_invoice_type": "sf_predicted_invoice_type",
            "predicted_invoice_date": "sf_predicted_invoice_date",
            "predicted_amount": "sf_predicted_amount",
        })
        if not sf_seq_features.empty:
            sf_seq_features["target_invoice_sequence"] = pd.to_numeric(sf_seq_features["target_invoice_sequence"], errors="coerce").astype("Int64")
            sf_seq_features["sf_predicted_invoice_date_ordinal_day"] = datetime_ordinal_days(parse_dates(sf_seq_features["sf_predicted_invoice_date"]))
            sf_seq_features["sf_predicted_amount_log1p"] = np.log1p(pd.to_numeric(sf_seq_features["sf_predicted_amount"], errors="coerce").abs())
        report = {}
        if saved_report.exists():
            with open(saved_report, "r", encoding="utf-8") as f:
                report = json.load(f)
        return counts, sf_seq_features, {"report": report, "source": "saved_salesforce_predictions"}

    sf_result = train_invoice_model(input_df, output_df, SALESFORCE_MODEL_NAME, train_orders=train_orders, test_orders=test_orders, write_outputs=False)
    orders = sf_result["orders"]
    count_preds = sf_result["all_order_count_predictions"]
    seq_rows = build_sequence_prediction_features(orders, count_preds, SALESFORCE_MODEL_NAME)
    seq_preds = predict_invoice_attributes(seq_rows, sf_result["attribute_models"], SALESFORCE_MODEL_NAME)
    sf_count_features = count_preds[["sales_order_id", "predicted_invoice_count", "predicted_invoice_count_raw"]].copy()
    sf_count_features.rename(columns={
        "predicted_invoice_count": "sf_predicted_invoice_count",
        "predicted_invoice_count_raw": "sf_predicted_invoice_count_raw",
    }, inplace=True)
    sf_seq_features = seq_preds.rename(columns={
        "predicted_invoice_sequence": "target_invoice_sequence",
        "predicted_invoice_type": "sf_predicted_invoice_type",
        "predicted_invoice_date": "sf_predicted_invoice_date",
        "predicted_amount": "sf_predicted_amount",
    })
    sf_seq_features["sf_predicted_invoice_date_ordinal_day"] = datetime_ordinal_days(parse_dates(sf_seq_features["sf_predicted_invoice_date"]))
    sf_seq_features["sf_predicted_amount_log1p"] = np.log1p(pd.to_numeric(sf_seq_features["sf_predicted_amount"], errors="coerce").abs())
    return sf_count_features, sf_seq_features, sf_result

def train_invoice_model(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    feature_set: str,
    train_orders: pd.Series | None = None,
    test_orders: pd.Series | None = None,
    sf_count_features: pd.DataFrame | None = None,
    sf_sequence_features: pd.DataFrame | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    orders = build_order_table(input_df, output_df, feature_set)
    if train_orders is None or test_orders is None:
        train_pos, test_pos = temporal_order_split(orders)
        train_orders = orders.iloc[train_pos]["sales_order_id"].astype(str)
        test_orders = orders.iloc[test_pos]["sales_order_id"].astype(str)
    invoices = build_invoice_training_table(input_df, output_df, feature_set)

    count_model, count_holdout, count_metrics = fit_count_model(orders, train_orders, test_orders, feature_set, sf_count_features)
    # Refit count model on all orders for final invoice generation.
    count_feature_data = orders.copy()
    if sf_count_features is not None:
        count_feature_data = count_feature_data.merge(sf_count_features, on="sales_order_id", how="left")
    X_count_all = build_feature_frame(count_feature_data, feature_set)
    y_count_all = orders["target_invoice_count"].to_numpy(dtype=float)
    count_model_all, count_all_raw = fit_log_regressor(count_model_spec(), X_count_all, y_count_all, X_count_all)
    all_count_predictions = orders[["sales_order_id", "target_invoice_count"]].copy()
    all_count_predictions["predicted_invoice_count_raw"] = count_all_raw
    all_count_predictions["predicted_invoice_count"] = np.clip(np.rint(count_all_raw), 0, MAX_INVOICE_SEQUENCE).astype(int)

    attribute_models, attribute_metrics, attribute_holdout = train_attribute_models(
        invoices, orders, train_orders, test_orders, feature_set, sf_sequence_features=sf_sequence_features
    )
    # Use the temporally validated attribute models for the generated benchmark output.
    # This keeps the script fast and keeps the generated rows consistent with the validation setup.
    attribute_models_all = attribute_models

    seq_rows = build_sequence_prediction_features(orders, all_count_predictions, feature_set, sf_sequence_features=sf_sequence_features)
    predicted_invoices = predict_invoice_attributes(seq_rows, attribute_models_all, feature_set)

    best_date_metrics = attribute_metrics[(attribute_metrics["target"] == "invoice_date_offset_days") & (attribute_metrics["model"] == attribute_models["best_date_model"])].iloc[0].to_dict()
    best_amount_metrics = attribute_metrics[(attribute_metrics["target"] == "invoice_amount_abs") & (attribute_metrics["model"] == attribute_models["best_amount_model"])].iloc[0].to_dict()
    best_type_metrics = attribute_metrics[(attribute_metrics["target"] == "invoice_type") & (attribute_metrics["model"] == attribute_models["best_type_model"])].iloc[0].to_dict()

    report = {
        "model_name": feature_set,
        "n_orders": int(len(orders)),
        "n_invoices": int(len(output_df)),
        "n_train_orders": int(len(train_orders)),
        "n_test_orders": int(len(test_orders)),
        "holdout_fraction": HOLDOUT_FRACTION,
        "max_invoice_sequence": MAX_INVOICE_SEQUENCE,
        "count_metrics": count_metrics,
        "best_date_model": attribute_models["best_date_model"],
        "best_amount_model": attribute_models["best_amount_model"],
        "best_type_model": attribute_models.get("best_type_model"),
        "best_date_metrics": best_date_metrics,
        "best_amount_metrics": best_amount_metrics,
        "best_type_metrics": best_type_metrics,
        "date_r2_note": (
            "R2 for invoice-date offsets is reference-date dependent. "
            "Salesforce-only and Salesforce+SAP models may use different reference dates, so date MAE/RMSE are the primary cross-model comparison metrics."
        ),
        "amount_target_note": (
            "Amount candidates include both log absolute amount and log amount/share of an order-value denominator. "
            "Tree models are not feature-scaled, but amount targets are normalized in the ratio candidates."
        ),
        "feature_set_description": "Salesforce columns only" if feature_set == SALESFORCE_MODEL_NAME else "Salesforce + SAP columns, with Salesforce-model predictions when provided",
    }

    if write_outputs:
        out_dir = model_output_dir(feature_set)
        create_dir(out_dir)
        write_csv(count_holdout, out_dir / "invoice_count_holdout_predictions.csv")
        write_csv(attribute_metrics, out_dir / "invoice_attribute_metrics.csv")
        write_csv(attribute_holdout, out_dir / "invoice_attribute_holdout_predictions.csv")
        write_csv(predicted_invoices, out_dir / "predicted_invoices.csv")
        save_json(report, out_dir / "invoice_forecaster_report.json")

    return {
        "orders": orders,
        "invoices": invoices,
        "count_model": count_model,
        "count_model_all": count_model_all,
        "count_holdout": count_holdout,
        "count_metrics": count_metrics,
        "all_order_count_predictions": all_count_predictions,
        "attribute_models": attribute_models,
        "attribute_models_all": attribute_models_all,
        "attribute_metrics": attribute_metrics,
        "attribute_holdout": attribute_holdout,
        "predicted_invoices": predicted_invoices,
        "report": report,
        "train_orders": train_orders,
        "test_orders": test_orders,
    }
