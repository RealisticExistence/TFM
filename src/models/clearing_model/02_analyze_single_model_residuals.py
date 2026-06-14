from config import CLEARING_DATE_FORECASTER_DIR
from util import create_dir, log, write_csv
from models.clearing_model.clearing_date_model_lib import (
    FAST_CLEARING_DAYS,
    MEDIUM_CLEARING_DAYS,
    assign_delay_regime,
    save_json,
)
import numpy as np
import pandas as pd

log("="*20 + " ANALYZE SINGLE MODEL RESIDUALS " + "="*20)

SINGLE_MODELS_DIR = CLEARING_DATE_FORECASTER_DIR / "01_single_models"
RESIDUALS_DIR = CLEARING_DATE_FORECASTER_DIR / "02_single_model_residual_analysis"


def add_delay_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [-0.1, 0, FAST_CLEARING_DAYS, 15, 30, MEDIUM_CLEARING_DAYS, 60, 90, 120, np.inf]
    labels = [
        "same_day",
        f"1_{FAST_CLEARING_DAYS}_days",
        "8_15_days",
        "16_30_days",
        f"31_{MEDIUM_CLEARING_DAYS}_days",
        "46_60_days",
        "61_90_days",
        "91_120_days",
        "gt_120_days",
    ]
    out["actual_delay_bucket"] = pd.cut(out["actual_days_to_clear"], bins=bins, labels=labels)
    out["delay_regime"] = assign_delay_regime(out["actual_days_to_clear"])
    return out


def group_residuals(df: pd.DataFrame, group_col) -> pd.DataFrame:
    rows = []
    total_mse = df["best_single_squared_error"].sum()
    group_cols = group_col if isinstance(group_col, list) else [group_col]
    for value, group in df.groupby(group_cols, dropna=False, observed=False):
        errors = group["best_single_error_days"]
        squared = group["best_single_squared_error"]
        row = {}
        if isinstance(value, tuple):
            for col, item in zip(group_cols, value):
                row[col] = item
        else:
            row[group_cols[0]] = value
        row.update({
            "rows": int(len(group)),
            "mae": float(errors.abs().mean()),
            "median_absolute_error": float(errors.abs().median()),
            "rmse": float(np.sqrt(squared.mean())),
            "bias_mean_error": float(errors.mean()),
            "mse_share": float(squared.sum() / total_mse) if total_mse else 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mse_share", ascending=False)


def analyze_error_direction(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.copy()
    out["error_direction"] = np.where(
        out["best_single_error_days"] > 0,
        "over_prediction_model_too_late",
        np.where(out["best_single_error_days"] < 0, "under_prediction_model_too_early", "exact"),
    )
    return group_residuals(out, ["error_direction", group_col])


def top_n_regime_summary(ranked: pd.DataFrame, total_mse: float) -> dict:
    summary = {}
    for n in [1, 2, 5, 10, 20, 50]:
        top = ranked.head(n)
        summary[f"top_{n}_mse_share"] = float(top["best_single_squared_error"].sum() / total_mse) if total_mse else 0.0
        summary[f"top_{n}_fast_share"] = float((top["delay_regime"] == "fast").mean()) if len(top) else 0.0
        summary[f"top_{n}_medium_share"] = float((top["delay_regime"] == "medium").mean()) if len(top) else 0.0
        summary[f"top_{n}_long_share"] = float((top["delay_regime"] == "long").mean()) if len(top) else 0.0
    return summary


def analyze_residuals() -> None:
    create_dir(RESIDUALS_DIR)
    holdout_path = SINGLE_MODELS_DIR / "single_model_holdout_predictions.csv"
    comparison_path = SINGLE_MODELS_DIR / "single_model_comparison.csv"

    if not holdout_path.exists() or not comparison_path.exists():
        raise FileNotFoundError(
            "Single-model outputs were not found. Run models/01_clearing_date_single_models.py first."
        )

    holdout = pd.read_csv(holdout_path)
    comparison = pd.read_csv(comparison_path)
    holdout = add_delay_bucket(holdout)
    ranked = holdout.sort_values("best_single_squared_error", ascending=False).reset_index(drop=True)
    total_mse = float(ranked["best_single_squared_error"].sum())
    ranked["cum_mse_share"] = ranked["best_single_squared_error"].cumsum() / total_mse if total_mse else 0.0

    write_csv(ranked, RESIDUALS_DIR / "single_model_residuals_ranked.csv")
    write_csv(group_residuals(holdout, "is_fast_clearing"), RESIDUALS_DIR / "residuals_by_fast_clearing.csv")
    write_csv(group_residuals(holdout, "delay_regime"), RESIDUALS_DIR / "residuals_by_delay_regime.csv")
    write_csv(group_residuals(holdout, "actual_delay_bucket"), RESIDUALS_DIR / "residuals_by_actual_delay_bucket.csv")
    write_csv(analyze_error_direction(holdout, "is_fast_clearing"), RESIDUALS_DIR / "residuals_by_error_direction_and_fast_clearing.csv")
    write_csv(analyze_error_direction(holdout, "delay_regime"), RESIDUALS_DIR / "residuals_by_error_direction_and_delay_regime.csv")

    for col in ["invoice_type", "invoice_type_clean", "terms_of_payment", "account_name", "client_number"]:
        if col in holdout.columns:
            write_csv(group_residuals(holdout, col).head(30), RESIDUALS_DIR / f"residuals_by_{col}.csv")

    fast_rows = holdout.loc[holdout["delay_regime"] == "fast"]
    medium_rows = holdout.loc[holdout["delay_regime"] == "medium"]
    long_rows = holdout.loc[holdout["delay_regime"] == "long"]
    over_rows = holdout.loc[holdout["best_single_error_days"] > 0]
    under_rows = holdout.loc[holdout["best_single_error_days"] < 0]
    best_model = str(comparison.loc[comparison["kind"] == "single_regressor"].sort_values(["rmse", "mae"]).iloc[0]["model"])

    summary = {
        "analyzed_model": best_model,
        "fast_clearing_days": FAST_CLEARING_DAYS,
        "medium_clearing_days": MEDIUM_CLEARING_DAYS,
        "regime_definition": f"fast <= {FAST_CLEARING_DAYS}; medium {FAST_CLEARING_DAYS + 1}-{MEDIUM_CLEARING_DAYS}; long > {MEDIUM_CLEARING_DAYS}",
        "n_holdout_rows": int(len(holdout)),
        "fast_holdout_rate": float(len(fast_rows) / len(holdout)) if len(holdout) else 0.0,
        "medium_holdout_rate": float(len(medium_rows) / len(holdout)) if len(holdout) else 0.0,
        "long_holdout_rate": float(len(long_rows) / len(holdout)) if len(holdout) else 0.0,
        "fast_mse_share": float(fast_rows["best_single_squared_error"].sum() / total_mse) if total_mse else 0.0,
        "medium_mse_share": float(medium_rows["best_single_squared_error"].sum() / total_mse) if total_mse else 0.0,
        "long_mse_share": float(long_rows["best_single_squared_error"].sum() / total_mse) if total_mse else 0.0,
        "over_prediction_mse_share": float(over_rows["best_single_squared_error"].sum() / total_mse) if total_mse else 0.0,
        "under_prediction_mse_share": float(under_rows["best_single_squared_error"].sum() / total_mse) if total_mse else 0.0,
        **top_n_regime_summary(ranked, total_mse),
    }
    save_json(summary, RESIDUALS_DIR / "single_model_residual_analysis_report.json")

    log(f"Residual analysis for: {best_model}")
    log(f"Fast / medium / long MSE shares: {summary['fast_mse_share']:.1%} / {summary['medium_mse_share']:.1%} / {summary['long_mse_share']:.1%}")
    log(f"Top 10 residual regime shares: fast {summary['top_10_fast_share']:.1%}; medium {summary['top_10_medium_share']:.1%}; long {summary['top_10_long_share']:.1%}")
    log(f"Residual-analysis outputs written to: {RESIDUALS_DIR.resolve()}")


if __name__ == "__main__":
    analyze_residuals()
