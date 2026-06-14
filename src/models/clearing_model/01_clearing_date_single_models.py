from config import CLEARING_DATE_FORECASTER_DIR
from util import create_dir, log, write_csv
from models.clearing_model.clearing_date_model_lib import (
    load_full_output,
    run_single_model_comparison,
    save_json,
)

log("="*20 + " CLEARING DATE SINGLE MODELS " + "="*20)

SINGLE_MODELS_DIR = CLEARING_DATE_FORECASTER_DIR / "01_single_models"


def run_single_models() -> None:
    create_dir(SINGLE_MODELS_DIR)
    full_output = load_full_output()
    comparison, holdout, _fitted, summary = run_single_model_comparison(full_output)

    write_csv(comparison, SINGLE_MODELS_DIR / "single_model_comparison.csv")
    write_csv(holdout, SINGLE_MODELS_DIR / "single_model_holdout_predictions.csv")
    save_json(summary, SINGLE_MODELS_DIR / "single_model_report.json")

    best = summary["best_single_model"]
    best_row = comparison.loc[comparison["model"] == best].iloc[0]
    log(f"Best single model: {best}")
    log(f"Validation RMSE: {best_row['rmse']:.4f}; MAE: {best_row['mae']:.4f}; median AE: {best_row['median_absolute_error']:.4f}")
    log(f"Single-model outputs written to: {SINGLE_MODELS_DIR.resolve()}")


run_single_models()
