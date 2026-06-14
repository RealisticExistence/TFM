import json

import pandas as pd

from config import INVOICE_FORECASTER_DIR
from util import create_dir, log, write_csv
from models.invoice_forecaster_lib import SALESFORCE_MODEL_NAME, SALESFORCE_SAP_MODEL_NAME

log("="*20 + " INVOICE FORECASTER COMPARISON " + "="*20)


def read_report(model_name: str) -> dict:
    path = INVOICE_FORECASTER_DIR / model_name / "invoice_forecaster_report.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_attribute_metrics(model_name: str) -> pd.DataFrame:
    path = INVOICE_FORECASTER_DIR / model_name / "invoice_attribute_metrics.csv"
    metrics = pd.read_csv(path)
    metrics["model_name"] = model_name
    return metrics


def best_metric_row(metrics: pd.DataFrame, target: str, model: str) -> dict:
    rows = metrics[(metrics["target"] == target) & (metrics["model"] == model)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def compare_invoice_forecasters() -> None:
    create_dir(INVOICE_FORECASTER_DIR)
    rows = []
    attribute_metrics = []

    for model_name in [SALESFORCE_MODEL_NAME, SALESFORCE_SAP_MODEL_NAME]:
        report = read_report(model_name)
        metrics = read_attribute_metrics(model_name)
        attribute_metrics.append(metrics)
        count = report["count_metrics"]
        date = best_metric_row(metrics, "invoice_date_offset_days", report["best_date_model"])
        amount = best_metric_row(metrics, "invoice_amount_abs", report["best_amount_model"])
        inv_type = best_metric_row(metrics, "invoice_type", report["best_type_model"])
        rows.append({
            "model_name": model_name,
            "n_orders": report["n_orders"],
            "n_invoices": report["n_invoices"],
            "count_mae": count["mae"],
            "count_rmse": count["rmse"],
            "exact_count_accuracy": count["exact_count_accuracy"],
            "within_one_count_accuracy": count["within_one_count_accuracy"],
            "best_date_model": report["best_date_model"],
            "date_target_transform": date.get("target_transform"),
            "date_mae": date.get("mae"),
            "date_median_ae": date.get("median_absolute_error"),
            "date_rmse": date.get("rmse"),
            "date_r2_reference_specific": date.get("r2"),
            "best_amount_model": report["best_amount_model"],
            "amount_target_transform": amount.get("target_transform"),
            "amount_mae": amount.get("mae"),
            "amount_median_ae": amount.get("median_absolute_error"),
            "amount_rmse": amount.get("rmse"),
            "amount_r2": amount.get("r2"),
            "best_type_model": report["best_type_model"],
            "type_accuracy": inv_type.get("accuracy"),
            "type_weighted_f1": inv_type.get("weighted_f1"),
            "note": report.get("date_r2_note"),
        })

    comparison = pd.DataFrame(rows)
    all_attribute_metrics = pd.concat(attribute_metrics, ignore_index=True)
    write_csv(comparison, INVOICE_FORECASTER_DIR / "invoice_forecaster_comparison.csv")
    write_csv(all_attribute_metrics, INVOICE_FORECASTER_DIR / "invoice_attribute_metrics_comparison.csv")

    log(f"Comparison written to: {INVOICE_FORECASTER_DIR.resolve()}")


if __name__ == "__main__":
    compare_invoice_forecasters()
