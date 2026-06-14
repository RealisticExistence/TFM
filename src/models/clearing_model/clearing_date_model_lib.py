from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config import DATA_CLEANING_OUTPUT_FULL_DIR, FEATURE_ENGINEERING_OUTPUT_FULL_DIR
from util import create_dir, write_csv

RANDOM_STATE = 42
FAST_CLEARING_DAYS = 7
MEDIUM_CLEARING_DAYS = 45
DELAY_REGIME_LABELS = ["fast", "medium", "long"]
HOLDOUT_FRACTION = 0.20
DATE_LOWER_BOUND = pd.Timestamp("1900-01-01")
DATE_UPPER_BOUND = pd.Timestamp("2100-12-31")

EXACT_LEAKAGE_COLUMNS = {
    "clearing_date",
    "clearing_date_original",
    "clearing_date_forecasted",
    "clearing_date_model_predicted",
    "clearing_date_source",
    "clearing_date_was_forecasted",
    "clearing_date_was_cleaned_negative_delay",
    "cleared_after_payment_due_date",
    "days__invoice_date__to__clearing_date",
    "days__payment_due_date__to__clearing_date",
    "days__so_date__to__clearing_date",
    "days__created_date__to__clearing_date",
    "hist_has_clearing_date",
    "hist_missing_clearing_date",
}

LEAKAGE_PREFIXES = ("clearing_date_",)
LEAKAGE_SUBSTRINGS = ("__to__clearing_date",)

DATE_LIKE_KEEP_COLUMNS = {
    "invoice_date",
    "payment_due_date",
    "so_date",
    "created_date",
    "req_deliv_date",
    "close_date",
    "bid_submission_date",
    "bid_validity",
}

ID_OR_TEXT_COLS = (
    "invoice_num",
    "sales_order_id",
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
)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_full_output() -> pd.DataFrame:
    path = FEATURE_ENGINEERING_OUTPUT_FULL_DIR / "output.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature-engineered full output not found: {path}. "
            "Run feature_engineering/convert_feature_engineering_into_output.py first."
        )
    return pd.read_csv(path)


def coerce_datetime64ns(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert(None)
    except AttributeError:
        if getattr(out, "tz", None) is not None:
            out = out.tz_convert(None)
    out = pd.Series(out.to_numpy(dtype="datetime64[ns]"), index=getattr(values, "index", None))
    out = out.mask(out.lt(DATE_LOWER_BOUND) | out.gt(DATE_UPPER_BOUND))
    return out


def parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return coerce_datetime64ns(s)

    text = s.astype("string").str.strip()
    text = text.mask(text.isin(["", "nan", "NaN", "NaT", "None", "NULL", "null"]))
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    non_null = text.notna()

    year_first = non_null & text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$", na=False)
    if year_first.any():
        parsed = pd.to_datetime(text.loc[year_first], errors="coerce", yearfirst=True, dayfirst=False)
        out.loc[year_first] = coerce_datetime64ns(parsed).to_numpy()

    remaining = non_null & out.isna()
    day_first = remaining & text.str.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:[ T].*)?$", na=False)
    if day_first.any():
        parsed = pd.to_datetime(text.loc[day_first], errors="coerce", dayfirst=True, yearfirst=False)
        out.loc[day_first] = coerce_datetime64ns(parsed).to_numpy()

    remaining = non_null & out.isna()
    numeric = pd.to_numeric(text.where(remaining), errors="coerce")
    excel_serial = remaining & numeric.between(20000, 80000)
    if excel_serial.any():
        parsed = pd.to_datetime(numeric.loc[excel_serial], unit="D", origin="1899-12-30", errors="coerce")
        out.loc[excel_serial] = coerce_datetime64ns(parsed).to_numpy()

    remaining = non_null & out.isna()
    if remaining.any():
        try:
            parsed = pd.to_datetime(text.loc[remaining], errors="coerce", format="mixed", dayfirst=True)
        except TypeError:
            parsed = pd.to_datetime(text.loc[remaining], errors="coerce", dayfirst=True)
        out.loc[remaining] = coerce_datetime64ns(parsed).to_numpy()

    out = out.mask(out.lt(DATE_LOWER_BOUND) | out.gt(DATE_UPPER_BOUND))
    return pd.Series(out.to_numpy(dtype="datetime64[ns]"), index=s.index)


def datetime_ordinal_days(s: pd.Series) -> pd.Series:
    dt = parse_dates(s)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    valid = dt.notna()
    if valid.any():
        out.loc[valid] = dt.loc[valid].to_numpy(dtype="datetime64[D]").astype("int64")
    return out


def safe_day_delta(start: pd.Series, end: pd.Series) -> pd.Series:
    start_dt = parse_dates(start)
    end_dt = parse_dates(end)
    out = pd.Series(np.nan, index=start.index, dtype="float64")
    valid = start_dt.notna() & end_dt.notna()
    if valid.any():
        start_days = start_dt.loc[valid].to_numpy(dtype="datetime64[D]").astype("int64")
        end_days = end_dt.loc[valid].to_numpy(dtype="datetime64[D]").astype("int64")
        out.loc[valid] = end_days - start_days
    return out


def validate_no_negative_clearing_delays(df: pd.DataFrame, context: str) -> None:
    invoice_date = parse_dates(df["invoice_date"])
    clearing_date = parse_dates(df["clearing_date"])
    delay = safe_day_delta(invoice_date, clearing_date)
    negative = invoice_date.notna() & clearing_date.notna() & delay.lt(0)
    if not negative.any():
        return
    sample_cols = [c for c in ["sales_order_id", "invoice_num", "invoice_type", "invoice_date", "clearing_date"] if c in df.columns]
    sample = df.loc[negative, sample_cols].head(10).to_dict(orient="records")
    raise ValueError(
        f"{context}: found {int(negative.sum())} negative clearing delays. "
        "These should be cleaned in data_cleaning/04_clean_dates.py. "
        f"Sample rows: {sample}"
    )


def add_model_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "invoice_date" in out.columns:
        invoice_date = parse_dates(out["invoice_date"])
        out["model_invoice_dayofweek"] = invoice_date.dt.dayofweek
        out["model_invoice_is_month_end"] = invoice_date.dt.is_month_end.astype("float")
        out["model_invoice_is_month_start"] = invoice_date.dt.is_month_start.astype("float")
        out["model_invoice_is_quarter_end"] = invoice_date.dt.is_quarter_end.astype("float")
        out["model_invoice_quarter"] = invoice_date.dt.quarter
        out["model_invoice_ordinal_day"] = datetime_ordinal_days(invoice_date)

    if "payment_due_date" in out.columns:
        due_date = parse_dates(out["payment_due_date"])
        out["model_due_dayofweek"] = due_date.dt.dayofweek
        out["model_due_is_month_end"] = due_date.dt.is_month_end.astype("float")
        out["model_due_ordinal_day"] = datetime_ordinal_days(due_date)

    amount_col = None
    for candidate in ["invoice_amount_company_signed", "invoice_amount_raw_signed", "amount"]:
        if candidate in out.columns:
            amount_col = candidate
            break
    if amount_col is not None:
        amount = pd.to_numeric(out[amount_col], errors="coerce")
        out["model_invoice_amount_abs_log1p"] = np.log1p(amount.abs())
        out["model_invoice_amount_sign"] = np.sign(amount)

    if "invoice_num" in out.columns:
        invoice_num = out["invoice_num"].astype("string")
        counts = invoice_num.map(invoice_num.value_counts())
        out["model_invoice_num_count"] = pd.to_numeric(counts, errors="coerce")
        out["model_invoice_num_is_duplicate"] = (out["model_invoice_num_count"] > 1).astype("float")

    if {"sales_order_id", "invoice_date"}.issubset(out.columns):
        amount_abs = pd.to_numeric(out.get("invoice_amount_abs", pd.Series(np.nan, index=out.index)), errors="coerce")
        tmp = pd.DataFrame({
            "sales_order_id": out["sales_order_id"].astype("string"),
            "invoice_date_sort": parse_dates(out["invoice_date"]),
            "amount_abs": amount_abs,
        }, index=out.index)
        tmp = tmp.sort_values(["sales_order_id", "invoice_date_sort"], kind="mergesort")
        seq = tmp.groupby("sales_order_id").cumcount() + 1
        n = tmp.groupby("sales_order_id")["sales_order_id"].transform("size")
        out.loc[tmp.index, "model_invoice_sequence_in_so"] = seq.to_numpy()
        out.loc[tmp.index, "model_invoice_count_in_so"] = n.to_numpy()
        out["model_invoice_sequence_pct_in_so"] = out["model_invoice_sequence_in_so"] / out["model_invoice_count_in_so"].replace(0, np.nan)
        so_total = tmp.groupby("sales_order_id")["amount_abs"].transform("sum")
        out.loc[tmp.index, "model_invoice_abs_amount_share_in_so"] = tmp["amount_abs"].to_numpy() / so_total.replace(0, np.nan).to_numpy()

    if "terms_of_payment_days" in out.columns:
        out["model_terms_days_effective"] = pd.to_numeric(out["terms_of_payment_days"], errors="coerce")
    else:
        out["model_terms_days_effective"] = np.nan

    if "terms_of_payment" in out.columns:
        terms = out["terms_of_payment"].astype("string").str.upper().str.strip()
        short_hint = terms.isin(["32-C", "32C", "0000", "0", "IMMEDIATE", "CONTADO"])
        out.loc[out["model_terms_days_effective"].isna() & short_hint, "model_terms_days_effective"] = 0
        out["model_terms_code_has_dash"] = terms.str.contains("-", regex=False).astype("float")
        out["model_terms_code_digits"] = pd.to_numeric(terms.str.extract(r"(\d+)", expand=False), errors="coerce")

    out["model_terms_days_missing"] = out["model_terms_days_effective"].isna().astype("float")
    out["model_terms_is_short_hint"] = (out["model_terms_days_effective"].fillna(9999) <= FAST_CLEARING_DAYS).astype("float")
    return out


def is_leakage_column(col: str) -> bool:
    if col in EXACT_LEAKAGE_COLUMNS:
        return True
    if any(col.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
        return True
    if not col.startswith("hist_") and any(substr in col for substr in LEAKAGE_SUBSTRINGS):
        return True
    if re.match(r"^clearing_date_(day|month|year|day_sin|day_cos|month_sin|month_cos)$", col):
        return True
    return False


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_model_specific_features(df)
    drop_cols = [c for c in out.columns if is_leakage_column(c)]
    X = out.drop(columns=drop_cols, errors="ignore").copy()
    X = X.drop(columns=[c for c in X.columns if c in ID_OR_TEXT_COLS], errors="ignore")

    for col in list(X.columns):
        if col in DATE_LIKE_KEEP_COLUMNS or col.endswith("_date"):
            parsed = parse_dates(X[col])
            if parsed.notna().any():
                X[f"{col}__ordinal_day"] = datetime_ordinal_days(parsed)
                X[f"{col}__dayofweek"] = parsed.dt.dayofweek
                X[f"{col}__month"] = parsed.dt.month
                X.drop(columns=[col], inplace=True)

    X = X.drop(columns=[c for c in X.columns if c.startswith("Unnamed:")], errors="ignore")
    all_null_cols = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=all_null_cols, errors="ignore")

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
    return Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("model", estimator),
    ])


def prepare_known_clearing_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    out = df.copy()
    out["invoice_date"] = parse_dates(out["invoice_date"])
    out["clearing_date"] = parse_dates(out["clearing_date"])
    validate_no_negative_clearing_delays(out, "Clearing-date model input")
    delay = safe_day_delta(out["invoice_date"], out["clearing_date"])
    valid = out["invoice_date"].notna() & out["clearing_date"].notna() & delay.notna()
    return out.loc[valid].copy(), delay.loc[valid].astype(float), out["clearing_date"].isna()


def temporal_train_test_split(model_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sort_dates = parse_dates(model_df["invoice_date"])
    order = np.argsort(sort_dates.to_numpy(dtype="datetime64[ns]"))
    n_test = max(1, int(math.ceil(len(model_df) * HOLDOUT_FRACTION)))
    return order[:-n_test], order[-n_test:]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_mean_error": float(np.mean(err)),
        "n_test": int(len(y_true)),
    }


def get_single_regressor_specs() -> dict[str, dict[str, Any]]:
    return {
        "extra_trees_raw_target": {
            "target_transform": "raw",
            "estimator": ExtraTreesRegressor(
                n_estimators=160,
                max_features=0.55,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "extra_trees_smooth_log_target": {
            "target_transform": "log1p",
            "estimator": ExtraTreesRegressor(
                n_estimators=180,
                max_features=0.55,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "extra_trees_flexible_log_target": {
            "target_transform": "log1p",
            "estimator": ExtraTreesRegressor(
                n_estimators=180,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "gradient_boosting_log_target": {
            "target_transform": "log1p",
            "estimator": GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.045,
                max_depth=2,
                min_samples_leaf=12,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        },
        "gradient_boosting_flexible_log_target": {
            "target_transform": "log1p",
            "estimator": GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=3,
                min_samples_leaf=8,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        },
    }


def fit_predict_regressor(spec: dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    model = make_pipeline(clone(spec["estimator"]), X_train)
    if spec["target_transform"] == "log1p":
        model.fit(X_train, np.log1p(y_train))
        pred = np.expm1(model.predict(X_test))
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    return model, np.clip(pred, 0, None)


def run_single_model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], dict[str, Any]]:
    model_df, delay, missing = prepare_known_clearing_data(df)
    X = build_feature_frame(model_df)
    train_pos, test_pos = temporal_train_test_split(model_df)
    X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
    y = delay.to_numpy()
    y_train, y_test = y[train_pos], y[test_pos]
    holdout_index = model_df.index[test_pos]

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}

    baseline_pred = np.full_like(y_test, fill_value=float(np.median(y_train)), dtype=float)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred)
    baseline_metrics.update({"model": "median_baseline", "kind": "baseline", "target_transform": "none"})
    rows.append(baseline_metrics)
    predictions["median_baseline"] = baseline_pred

    for name, spec in get_single_regressor_specs().items():
        model, pred = fit_predict_regressor(spec, X_train, y_train, X_test)
        metrics = evaluate_predictions(y_test, pred)
        metrics.update({"model": name, "kind": "single_regressor", "target_transform": spec["target_transform"]})
        rows.append(metrics)
        fitted[name] = model
        predictions[name] = pred

    comparison = pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=True).reset_index(drop=True)
    best_single_name = comparison.loc[comparison["kind"] == "single_regressor"].sort_values(["rmse", "mae"]).iloc[0]["model"]

    holdout = model_df.loc[holdout_index].copy()
    holdout["actual_days_to_clear"] = y_test
    holdout["is_fast_clearing"] = (holdout["actual_days_to_clear"] <= FAST_CLEARING_DAYS).astype(int)
    holdout["delay_regime"] = assign_delay_regime(holdout["actual_days_to_clear"])
    holdout["best_single_model"] = best_single_name
    for name, pred in predictions.items():
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", name)
        holdout[f"pred__{safe}"] = pred
        holdout[f"error__{safe}"] = pred - y_test
        holdout[f"squared_error__{safe}"] = (pred - y_test) ** 2
    holdout["best_single_prediction_days"] = holdout[f"pred__{best_single_name}"]
    holdout["best_single_error_days"] = holdout[f"error__{best_single_name}"]
    holdout["best_single_squared_error"] = holdout[f"squared_error__{best_single_name}"]

    summary = {
        "n_rows_total": int(len(df)),
        "n_known_clearing_dates": int(len(model_df)),
        "n_missing_clearing_dates": int(missing.sum()),
        "n_holdout_train": int(len(train_pos)),
        "n_holdout_test": int(len(test_pos)),
        "holdout_fraction": HOLDOUT_FRACTION,
        "holdout_start_invoice_date": str(parse_dates(model_df["invoice_date"]).iloc[test_pos].min().date()),
        "holdout_end_invoice_date": str(parse_dates(model_df["invoice_date"]).iloc[test_pos].max().date()),
        "fast_clearing_days": FAST_CLEARING_DAYS,
        "best_single_model": str(best_single_name),
        "n_model_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
    }
    return comparison, holdout, fitted, summary


def get_best_single_model_name(comparison: pd.DataFrame) -> str:
    single = comparison.loc[comparison["kind"] == "single_regressor"].copy()
    return str(single.sort_values(["rmse", "mae"]).iloc[0]["model"])


def assign_delay_regime(days: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(days, dtype=float)
    return np.select(
        [values <= FAST_CLEARING_DAYS, values <= MEDIUM_CLEARING_DAYS],
        ["fast", "medium"],
        default="long",
    )


def sharpen_multiclass_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    if abs(temperature - 1.0) < 1e-12:
        return p / p.sum(axis=1, keepdims=True)
    p = p ** (1.0 / temperature)
    return p / p.sum(axis=1, keepdims=True)


def delay_regime_counts(days: np.ndarray | pd.Series) -> dict[str, int]:
    regimes = assign_delay_regime(days)
    return {label: int((regimes == label).sum()) for label in DELAY_REGIME_LABELS}


def sharpen_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if abs(temperature - 1.0) < 1e-12:
        return p
    logit = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-logit / temperature))


def fit_soft_fast_classifier_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    temperature: float,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    classifier = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.055,
        max_depth=2,
        min_samples_leaf=12,
        subsample=0.85,
        random_state=RANDOM_STATE,
    )
    fast_estimator = ExtraTreesRegressor(
        n_estimators=160,
        max_features=0.75,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    regular_estimator = ExtraTreesRegressor(
        n_estimators=180,
        max_features=0.55,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    fast_target = (y_train <= FAST_CLEARING_DAYS).astype(int)
    classifier_model = make_pipeline(classifier, X_train)
    classifier_model.fit(X_train, fast_target)
    p_fast = classifier_model.predict_proba(X_test)[:, 1]

    fast_mask = y_train <= FAST_CLEARING_DAYS
    if fast_mask.sum() >= 20 and (~fast_mask).sum() >= 20:
        fast_X, fast_y = X_train.loc[fast_mask], y_train[fast_mask]
        regular_X, regular_y = X_train.loc[~fast_mask], y_train[~fast_mask]
    else:
        fast_X, fast_y = X_train, y_train
        regular_X, regular_y = X_train, y_train

    fast_model, fast_pred = fit_predict_regressor({"target_transform": "log1p", "estimator": fast_estimator}, fast_X, fast_y, X_test)
    regular_model, regular_pred = fit_predict_regressor({"target_transform": "log1p", "estimator": regular_estimator}, regular_X, regular_y, X_test)
    fast_weight = sharpen_probabilities(p_fast, temperature)
    pred = fast_weight * fast_pred + (1 - fast_weight) * regular_pred

    model = {
        "kind": "tempered_soft_fast_classifier",
        "classifier": classifier_model,
        "fast_regressor": fast_model,
        "regular_regressor": regular_model,
        "temperature": float(temperature),
    }
    aux = {
        "temperature": float(temperature),
        "p_fast_mean": float(np.mean(p_fast)),
        "p_fast_median": float(np.median(p_fast)),
        "fast_weight_mean": float(np.mean(fast_weight)),
        "fast_weight_median": float(np.median(fast_weight)),
    }
    return model, np.clip(pred, 0, None), aux


def fit_soft_delay_regime_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    temperature: float,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    classifier = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    regressor_specs = {
        "fast": {
            "target_transform": "log1p",
            "estimator": ExtraTreesRegressor(
                n_estimators=160,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "medium": {
            "target_transform": "log1p",
            "estimator": ExtraTreesRegressor(
                n_estimators=180,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        "long": {
            "target_transform": "log1p",
            "estimator": ExtraTreesRegressor(
                n_estimators=180,
                max_features=0.55,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
    }

    y_regime = assign_delay_regime(y_train)
    classifier_model = make_pipeline(classifier, X_train)
    classifier_model.fit(X_train, y_regime)
    p_regime_raw = classifier_model.predict_proba(X_test)
    p_regime = sharpen_multiclass_probabilities(p_regime_raw, temperature)

    regime_predictions: dict[str, np.ndarray] = {}
    regressors: dict[str, Pipeline] = {}
    for label in DELAY_REGIME_LABELS:
        mask = y_regime == label
        if int(mask.sum()) >= 20:
            reg_X = X_train.loc[mask]
            reg_y = y_train[mask]
        else:
            reg_X = X_train
            reg_y = y_train
        regressor, pred = fit_predict_regressor(regressor_specs[label], reg_X, reg_y, X_test)
        regressors[label] = regressor
        regime_predictions[label] = pred

    pred = np.zeros(len(X_test), dtype=float)
    class_to_col = {class_label: i for i, class_label in enumerate(classifier_model.classes_)}
    for label in DELAY_REGIME_LABELS:
        if label in class_to_col:
            pred += p_regime[:, class_to_col[label]] * regime_predictions[label]

    model = {
        "kind": "tempered_soft_delay_regime_classifier",
        "classifier": classifier_model,
        "regressors": regressors,
        "temperature": float(temperature),
        "regime_labels": DELAY_REGIME_LABELS,
    }
    aux = {
        "temperature": float(temperature),
        "p_fast_mean": float(p_regime_raw[:, class_to_col.get("fast", 0)].mean()) if "fast" in class_to_col else np.nan,
        "p_medium_mean": float(p_regime_raw[:, class_to_col.get("medium", 0)].mean()) if "medium" in class_to_col else np.nan,
        "p_long_mean": float(p_regime_raw[:, class_to_col.get("long", 0)].mean()) if "long" in class_to_col else np.nan,
        "fast_weight_mean": float(p_regime[:, class_to_col.get("fast", 0)].mean()) if "fast" in class_to_col else np.nan,
        "medium_weight_mean": float(p_regime[:, class_to_col.get("medium", 0)].mean()) if "medium" in class_to_col else np.nan,
        "long_weight_mean": float(p_regime[:, class_to_col.get("long", 0)].mean()) if "long" in class_to_col else np.nan,
        **{f"n_train_{label}": int((y_regime == label).sum()) for label in DELAY_REGIME_LABELS},
    }
    return model, np.clip(pred, 0, None), aux


def delay_regime_classifier_metrics(y_true_days: np.ndarray, p_regime: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    true_regime = assign_delay_regime(y_true_days)
    pred_regime = np.asarray(classes)[np.argmax(p_regime, axis=1)]
    rows = {
        "regime_classifier_accuracy": float(accuracy_score(true_regime, pred_regime)),
        "regime_classifier_balanced_accuracy": float(balanced_accuracy_score(true_regime, pred_regime)),
        "regime_classifier_macro_f1": float(f1_score(true_regime, pred_regime, average="macro", zero_division=0)),
    }
    for label in DELAY_REGIME_LABELS:
        binary_true = (true_regime == label).astype(int)
        binary_pred = (pred_regime == label).astype(int)
        rows[f"regime_{label}_precision"] = float(precision_score(binary_true, binary_pred, zero_division=0))
        rows[f"regime_{label}_recall"] = float(recall_score(binary_true, binary_pred, zero_division=0))
        rows[f"regime_{label}_f1"] = float(f1_score(binary_true, binary_pred, zero_division=0))
    return rows


def fast_classifier_metrics(y_true_days: np.ndarray, p_fast: np.ndarray) -> dict[str, float]:
    true_fast = (y_true_days <= FAST_CLEARING_DAYS).astype(int)
    pred_fast = (p_fast >= 0.5).astype(int)
    return {
        "fast_classifier_precision": float(precision_score(true_fast, pred_fast, zero_division=0)),
        "fast_classifier_recall": float(recall_score(true_fast, pred_fast, zero_division=0)),
        "fast_classifier_f1": float(f1_score(true_fast, pred_fast, zero_division=0)),
        "fast_classifier_roc_auc": float(roc_auc_score(true_fast, p_fast)) if len(np.unique(true_fast)) > 1 else np.nan,
    }


def load_clean_output_template(expected_rows: int) -> pd.DataFrame | None:
    path = DATA_CLEANING_OUTPUT_FULL_DIR / "output.csv"
    if not path.exists():
        return None
    template = pd.read_csv(path)
    if len(template) != expected_rows:
        return None
    return template


def complete_output_with_model(df: pd.DataFrame, model: dict[str, Any] | Pipeline, kind: str) -> tuple[pd.DataFrame, dict[str, int]]:
    working = df.copy()
    clean_template = load_clean_output_template(len(df))
    working["invoice_date"] = parse_dates(working["invoice_date"])
    working["clearing_date"] = parse_dates(working["clearing_date"])
    validate_no_negative_clearing_delays(working, "Output completion")

    X = build_feature_frame(working)
    if kind == "single_log":
        pred_days = np.expm1(model.predict(X))
    elif kind == "single_raw":
        pred_days = model.predict(X)
    elif kind == "tempered_soft_fast_classifier":
        p_fast = model["classifier"].predict_proba(X)[:, 1]
        fast_weight = sharpen_probabilities(p_fast, float(model["temperature"]))
        fast_pred = np.expm1(model["fast_regressor"].predict(X))
        regular_pred = np.expm1(model["regular_regressor"].predict(X))
        pred_days = fast_weight * fast_pred + (1 - fast_weight) * regular_pred
    elif kind == "tempered_soft_delay_regime_classifier":
        p_raw = model["classifier"].predict_proba(X)
        weights = sharpen_multiclass_probabilities(p_raw, float(model["temperature"]))
        pred_days = np.zeros(len(X), dtype=float)
        class_to_col = {class_label: i for i, class_label in enumerate(model["classifier"].classes_)}
        for label, regressor in model["regressors"].items():
            if label in class_to_col:
                pred_days += weights[:, class_to_col[label]] * np.expm1(regressor.predict(X))
    else:
        raise ValueError(f"Unsupported completion model kind: {kind}")

    pred_days = np.clip(pred_days, 0, None)
    missing = working["clearing_date"].isna()
    predicted_dates = working["invoice_date"] + pd.to_timedelta(np.round(pred_days).astype("int64"), unit="D")

    completed = clean_template.copy() if clean_template is not None else df.copy()
    completed_dates = parse_dates(completed["clearing_date"]) if "clearing_date" in completed.columns else working["clearing_date"].copy()
    completed_dates.loc[missing] = predicted_dates.loc[missing]
    completed["clearing_date"] = pd.to_datetime(completed_dates, errors="coerce").dt.strftime("%Y-%m-%d")
    stats = {
        "n_missing_clearing_dates_before_forecast": int(missing.sum()),
        "n_missing_clearing_dates_after_forecast": int(pd.to_datetime(completed["clearing_date"], errors="coerce").isna().sum()),
        "n_rows_forecasted": int(missing.sum()),
    }
    return completed, stats


def refit_single_model_on_all_known(df: pd.DataFrame, model_name: str) -> tuple[Pipeline, str]:
    model_df, delay, _missing = prepare_known_clearing_data(df)
    X = build_feature_frame(model_df)
    y = delay.to_numpy()
    spec = get_single_regressor_specs()[model_name]
    model = make_pipeline(clone(spec["estimator"]), X)
    if spec["target_transform"] == "log1p":
        model.fit(X, np.log1p(y))
        return model, "single_log"
    model.fit(X, y)
    return model, "single_raw"


def refit_fast_classifier_model_on_all_known(df: pd.DataFrame, temperature: float) -> dict[str, Any]:
    model_df, delay, _missing = prepare_known_clearing_data(df)
    X = build_feature_frame(model_df)
    y = delay.to_numpy()
    model, _pred, _aux = fit_soft_fast_classifier_model(X, y, X, temperature=temperature)
    return model


def refit_delay_regime_model_on_all_known(df: pd.DataFrame, temperature: float) -> dict[str, Any]:
    model_df, delay, _missing = prepare_known_clearing_data(df)
    X = build_feature_frame(model_df)
    y = delay.to_numpy()
    model, _pred, _aux = fit_soft_delay_regime_model(X, y, X, temperature=temperature)
    return model


def write_model_table(df: pd.DataFrame, path: Path) -> None:
    write_csv(df, path)
