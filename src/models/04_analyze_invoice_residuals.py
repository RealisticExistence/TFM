from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import INVOICE_FORECASTER_DIR
from util import create_dir, log, write_csv
from models.invoice_forecaster_lib import SALESFORCE_MODEL_NAME, SALESFORCE_SAP_MODEL_NAME

log("="*20 + " INVOICE FORECASTER RESIDUAL ANALYSIS " + "="*20)

RESIDUAL_DIR = INVOICE_FORECASTER_DIR / "residual_analysis"


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_abs_error_days"] = out["date_error_days"].abs()
    out["date_squared_error_days"] = out["date_error_days"] ** 2
    out["amount_abs_error"] = out["amount_error_abs"].abs()
    out["amount_squared_error"] = out["amount_error_abs"] ** 2
    out["invoice_date_regime"] = pd.cut(
        out["target_days_to_invoice"],
        bins=[-np.inf, 0, 120, 360, np.inf],
        labels=["pre_reference", "early_1_120", "standard_121_360", "late_361_plus"],
    ).astype(str)
    out["invoice_amount_regime"] = pd.cut(
        out["target_amount_abs"],
        bins=[-np.inf, 10_000, 50_000, 150_000, np.inf],
        labels=["small_le_10k", "medium_10k_50k", "large_50k_150k", "very_large_150k_plus"],
    ).astype(str)
    out["type_is_correct"] = (out["invoice_type"].astype(str) == out["predicted_invoice_type"].astype(str)).astype(int)
    return out


def summarize_regime(df: pd.DataFrame, group_col: str, error_col: str, squared_col: str) -> pd.DataFrame:
    total_sq = df[squared_col].sum()
    grouped = df.groupby(group_col, dropna=False).agg(
        rows=(squared_col, "size"),
        mae=(error_col, "mean"),
        median_ae=(error_col, "median"),
        rmse=(squared_col, lambda x: float(np.sqrt(np.mean(x)))),
        squared_error_sum=(squared_col, "sum"),
    ).reset_index()
    grouped["mse_share"] = grouped["squared_error_sum"] / total_sq if total_sq else np.nan
    return grouped.sort_values("squared_error_sum", ascending=False)


def top_error_summary(df: pd.DataFrame, squared_col: str, regime_col: str) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values(squared_col, ascending=False).reset_index(drop=True)
    total = ranked[squared_col].sum()
    for n in [5, 10, 20, 50]:
        top = ranked.head(n)
        row = {
            "top_n": n,
            "squared_error_share": float(top[squared_col].sum() / total) if total else np.nan,
        }
        for regime, share in top[regime_col].value_counts(normalize=True).items():
            row[f"share_{regime}"] = float(share)
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_model(model_name: str) -> dict:
    model_dir = INVOICE_FORECASTER_DIR / model_name
    holdout = pd.read_csv(model_dir / "invoice_attribute_holdout_predictions.csv")
    residuals = add_error_columns(holdout)

    out_dir = RESIDUAL_DIR / model_name
    create_dir(out_dir)
    write_csv(residuals.sort_values("date_squared_error_days", ascending=False), out_dir / "invoice_residuals_ranked_by_date_error.csv")
    write_csv(summarize_regime(residuals, "invoice_date_regime", "date_abs_error_days", "date_squared_error_days"), out_dir / "date_error_by_invoice_date_regime.csv")
    write_csv(summarize_regime(residuals, "invoice_amount_regime", "amount_abs_error", "amount_squared_error"), out_dir / "amount_error_by_amount_regime.csv")
    write_csv(top_error_summary(residuals, "date_squared_error_days", "invoice_date_regime"), out_dir / "top_date_error_regime_mix.csv")

    type_by_actual = residuals.groupby("invoice_type", dropna=False).agg(
        rows=("type_is_correct", "size"),
        type_accuracy=("type_is_correct", "mean"),
        date_mae=("date_abs_error_days", "mean"),
        amount_mae=("amount_abs_error", "mean"),
    ).reset_index().sort_values("rows", ascending=False)
    write_csv(type_by_actual, out_dir / "error_by_invoice_type.csv")

    date_regime = summarize_regime(residuals, "invoice_date_regime", "date_abs_error_days", "date_squared_error_days")
    report = {
        "model_name": model_name,
        "n_holdout_invoices": int(len(residuals)),
        "date_rmse": float(np.sqrt(np.mean(residuals["date_squared_error_days"]))),
        "date_mae": float(residuals["date_abs_error_days"].mean()),
        "amount_rmse": float(np.sqrt(np.mean(residuals["amount_squared_error"]))),
        "amount_mae": float(residuals["amount_abs_error"].mean()),
        "type_accuracy": float(residuals["type_is_correct"].mean()),
        "date_mse_share_by_regime": dict(zip(date_regime["invoice_date_regime"], date_regime["mse_share"])),
    }
    save_json(report, out_dir / "invoice_residual_analysis_report.json")
    return report


def analyze_invoice_residuals() -> None:
    create_dir(RESIDUAL_DIR)
    reports = [analyze_model(name) for name in [SALESFORCE_MODEL_NAME, SALESFORCE_SAP_MODEL_NAME]]
    save_json({"models": reports}, RESIDUAL_DIR / "invoice_residual_analysis_summary.json")
    log(f"Residual analysis written to: {RESIDUAL_DIR.resolve()}")


if __name__ == "__main__":
    analyze_invoice_residuals()
