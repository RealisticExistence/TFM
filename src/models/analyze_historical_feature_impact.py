"""Analyze whether time-aware historical aggregate features help clearing-date forecasting.

This script is intentionally separate from ``clearing_date_forecaster.py``. It
performs an ablation study on the same temporal holdout used by the forecaster:

1. Build the normal model feature frame, including ``hist_*`` columns.
2. Build an ablated feature frame where all ``hist_*`` columns are removed.
3. Train the same candidate regressors on both feature sets.
4. Compare validation metrics and write feature-importance diagnostics.

Run after ``feature_engineering/convert_feature_engineering_into_output.py``::

    PYTHONPATH=. python models/analyze_historical_feature_impact.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from config import FEATURE_ENGINEERING_OUTPUT_FULL_DIR, CLEARING_DATE_FORECASTER_DIR
from util import create_dir, log, write_csv
from models.clearing_model.clearing_date_forecaster import (
    HOLDOUT_FRACTION,
    RANDOM_STATE,
    build_feature_frame,
    evaluate_predictions,
    log1p_fit_predict,
    parse_dates,
    raw_fit_predict,
    safe_day_delta,
    validate_no_negative_clearing_delays,
)

OUTPUT_DIR = CLEARING_DATE_FORECASTER_DIR / "historical_feature_impact"


def _temporal_split_positions(invoice_dates: pd.Series, holdout_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(parse_dates(invoice_dates).to_numpy(dtype="datetime64[ns]"))
    n = len(order)
    n_test = max(1, int(math.ceil(n * holdout_fraction)))
    return order[:-n_test], order[-n_test:]


def _candidate_estimators() -> dict[str, tuple[str, Any]]:
    return {
        "extra_trees_raw_target": (
            "raw",
            ExtraTreesRegressor(
                n_estimators=160,
                max_features=0.55,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "extra_trees_smooth_log_target": (
            "log",
            ExtraTreesRegressor(
                n_estimators=180,
                max_features=0.55,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "extra_trees_flexible_log_target": (
            "log",
            ExtraTreesRegressor(
                n_estimators=180,
                max_features=0.75,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "gradient_boosting_log_target": (
            "log",
            GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.045,
                max_depth=2,
                min_samples_leaf=12,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        ),
        "gradient_boosting_flexible_log_target": (
            "log",
            GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=3,
                min_samples_leaf=8,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def _evaluate_feature_set(
    feature_set_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}

    X_train = X.iloc[train_pos]
    X_test = X.iloc[test_pos]
    y_train = y[train_pos]
    y_test = y[test_pos]

    n_hist_features = len([c for c in X.columns if c.startswith("hist_")])

    baseline_pred = np.full_like(y_test, float(np.median(y_train)), dtype=float)
    metrics = evaluate_predictions(y_test, baseline_pred)
    metrics.update(
        {
            "feature_set": feature_set_name,
            "model": "median_baseline",
            "kind": "baseline",
            "n_features": int(X.shape[1]),
            "n_hist_features": int(n_hist_features),
        }
    )
    rows.append(metrics)
    preds["median_baseline"] = baseline_pred

    for model_name, (target_type, estimator) in _candidate_estimators().items():
        if target_type == "log":
            model, pred = log1p_fit_predict(clone(estimator), X_train, y_train, X_test)
        else:
            model, pred = raw_fit_predict(clone(estimator), X_train, y_train, X_test)

        metrics = evaluate_predictions(y_test, pred)
        metrics.update(
            {
                "feature_set": feature_set_name,
                "model": model_name,
                "kind": f"single_regressor_{target_type}",
                "n_features": int(X.shape[1]),
                "n_hist_features": int(n_hist_features),
            }
        )
        rows.append(metrics)
        preds[model_name] = pred
        fitted[model_name] = model

    return rows, preds, fitted


def _feature_importance(pipe: Any) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(len(model.feature_importances_))])

    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame()

    out = pd.DataFrame({"feature": names, "importance": importance})
    out["is_historical"] = out["feature"].astype(str).str.contains("hist_")
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def _bucket_diagnostics(y_test: np.ndarray, pred_with: np.ndarray, pred_without: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "actual_days_to_clear": y_test,
            "pred_with_historicals": pred_with,
            "pred_without_historicals": pred_without,
        }
    )
    df["abs_err_with"] = (df["pred_with_historicals"] - df["actual_days_to_clear"]).abs()
    df["abs_err_without"] = (df["pred_without_historicals"] - df["actual_days_to_clear"]).abs()
    df["se_with"] = (df["pred_with_historicals"] - df["actual_days_to_clear"]) ** 2
    df["se_without"] = (df["pred_without_historicals"] - df["actual_days_to_clear"]) ** 2

    bins = [-0.1, 0, 7, 15, 30, 45, 60, 90, 120, np.inf]
    labels = ["same-day", "1-7", "8-15", "16-30", "31-45", "46-60", "61-90", "91-120", ">120"]
    df["actual_delay_bucket"] = pd.cut(df["actual_days_to_clear"], bins=bins, labels=labels)

    grouped = (
        df.groupby("actual_delay_bucket", observed=True)
        .agg(
            n=("actual_days_to_clear", "size"),
            rmse_with=("se_with", lambda s: float(np.sqrt(s.mean()))),
            rmse_without=("se_without", lambda s: float(np.sqrt(s.mean()))),
            mae_with=("abs_err_with", "mean"),
            mae_without=("abs_err_without", "mean"),
            mse_sum_with=("se_with", "sum"),
            mse_sum_without=("se_without", "sum"),
        )
        .reset_index()
    )
    grouped["rmse_delta_without_minus_with"] = grouped["rmse_without"] - grouped["rmse_with"]
    grouped["mse_share_with"] = grouped["mse_sum_with"] / grouped["mse_sum_with"].sum()
    grouped["mse_share_without"] = grouped["mse_sum_without"] / grouped["mse_sum_without"].sum()
    return grouped


def main() -> None:
    log("==================== HISTORICAL FEATURE IMPACT ANALYSIS ====================")
    create_dir(OUTPUT_DIR)

    full_output_path = FEATURE_ENGINEERING_OUTPUT_FULL_DIR / "output.csv"
    if not full_output_path.exists():
        raise FileNotFoundError(
            f"Feature-engineered full output not found: {full_output_path}. "
            "Run feature_engineering/convert_feature_engineering_into_output.py first."
        )

    full_output = pd.read_csv(full_output_path)
    full_output["invoice_date"] = parse_dates(full_output["invoice_date"])
    full_output["clearing_date"] = parse_dates(full_output["clearing_date"])
    validate_no_negative_clearing_delays(full_output, "Historical-feature impact analysis")

    delay = safe_day_delta(full_output["invoice_date"], full_output["clearing_date"])
    valid = full_output["invoice_date"].notna() & full_output["clearing_date"].notna() & delay.ge(0)
    model_df = full_output.loc[valid].copy()
    y = delay.loc[valid].astype(float).to_numpy()

    X_with = build_feature_frame(model_df)
    hist_cols = [c for c in X_with.columns if c.startswith("hist_")]
    X_without = X_with.drop(columns=hist_cols)

    train_pos, test_pos = _temporal_split_positions(model_df["invoice_date"], HOLDOUT_FRACTION)

    all_rows: list[dict[str, Any]] = []
    all_preds: dict[tuple[str, str], np.ndarray] = {}
    all_fitted: dict[tuple[str, str], Any] = {}

    for feature_set_name, X in [("with_historicals", X_with), ("without_historicals", X_without)]:
        rows, preds, fitted = _evaluate_feature_set(feature_set_name, X, y, train_pos, test_pos)
        all_rows.extend(rows)
        all_preds.update({(feature_set_name, k): v for k, v in preds.items()})
        all_fitted.update({(feature_set_name, k): v for k, v in fitted.items()})

    comparison = pd.DataFrame(all_rows).sort_values(["feature_set", "rmse", "mae"]).reset_index(drop=True)
    comparison_path = OUTPUT_DIR / "historical_ablation_model_comparison.csv"
    write_csv(comparison, comparison_path)

    selected_model_name = "extra_trees_flexible_log_target"
    if ("with_historicals", selected_model_name) not in all_fitted:
        selected_model_name = comparison.query("feature_set == 'with_historicals' and kind != 'baseline'").iloc[0]["model"]

    importance = _feature_importance(all_fitted[("with_historicals", selected_model_name)])
    importance_path = OUTPUT_DIR / "historical_ablation_feature_importance.csv"
    write_csv(importance, importance_path)

    y_test = y[test_pos]
    bucket_diag = _bucket_diagnostics(
        y_test,
        all_preds[("with_historicals", selected_model_name)],
        all_preds[("without_historicals", selected_model_name)],
    )
    bucket_path = OUTPUT_DIR / "historical_ablation_by_delay_bucket.csv"
    write_csv(bucket_diag, bucket_path)

    with_best = comparison.query("feature_set == 'with_historicals' and kind != 'baseline'").sort_values("rmse").iloc[0]
    without_best = comparison.query("feature_set == 'without_historicals' and kind != 'baseline'").sort_values("rmse").iloc[0]
    selected_with = comparison.query("feature_set == 'with_historicals' and model == @selected_model_name").iloc[0]
    selected_without = comparison.query("feature_set == 'without_historicals' and model == @selected_model_name").iloc[0]

    summary = {
        "n_valid_known_clearings": int(len(model_df)),
        "n_holdout_train": int(len(train_pos)),
        "n_holdout_test": int(len(test_pos)),
        "n_features_with_historicals": int(X_with.shape[1]),
        "n_features_without_historicals": int(X_without.shape[1]),
        "n_historical_features_removed": int(len(hist_cols)),
        "best_with_historicals": with_best.to_dict(),
        "best_without_historicals": without_best.to_dict(),
        "selected_model_name_for_direct_ablation": selected_model_name,
        "selected_model_with_historicals": selected_with.to_dict(),
        "selected_model_without_historicals": selected_without.to_dict(),
        "rmse_delta_without_minus_with_for_selected_model": float(selected_without["rmse"] - selected_with["rmse"]),
        "mae_delta_without_minus_with_for_selected_model": float(selected_without["mae"] - selected_with["mae"]),
        "historical_feature_importance_total_for_selected_model": float(
            importance.loc[importance["is_historical"], "importance"].sum()
        ) if not importance.empty else None,
    }
    with open(OUTPUT_DIR / "historical_feature_impact_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"Ablation comparison written to: {comparison_path}")
    log(f"Feature importance written to: {importance_path}")
    log(f"Delay-bucket diagnostics written to: {bucket_path}")
    log(
        "Selected-model RMSE with historicals vs without: "
        f"{selected_with['rmse']:.3f} vs {selected_without['rmse']:.3f}"
    )


if __name__ == "__main__":
    main()
