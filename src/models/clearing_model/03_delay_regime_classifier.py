import pandas as pd

from config import (
    CLEARING_DATE_FORECASTER_DIR,
    DATA_CLEANING_OUTPUT_FULL_FOLDER,
    FEATURE_ENGINEERING_OUTPUT_FULL_DIR,
)
from util import create_dir, log, write_csv
from models.clearing_model.clearing_date_model_lib import (
    DELAY_REGIME_LABELS,
    FAST_CLEARING_DAYS,
    MEDIUM_CLEARING_DAYS,
    build_feature_frame,
    complete_output_with_model,
    delay_regime_classifier_metrics,
    evaluate_predictions,
    fast_classifier_metrics,
    fit_soft_delay_regime_model,
    fit_soft_fast_classifier_model,
    get_best_single_model_name,
    load_full_output,
    prepare_known_clearing_data,
    refit_delay_regime_model_on_all_known,
    refit_fast_classifier_model_on_all_known,
    refit_single_model_on_all_known,
    run_single_model_comparison,
    save_json,
    sharpen_multiclass_probabilities,
    temporal_train_test_split,
)

log("="*20 + " DELAY REGIME CLASSIFIER " + "="*20)

SINGLE_MODELS_DIR = CLEARING_DATE_FORECASTER_DIR / "01_single_models"
REGIME_MODEL_DIR = CLEARING_DATE_FORECASTER_DIR / "03_delay_regime_classifier"
TEMPERATURES = [1.0, 0.8, 0.6, 0.5]
MEDIAN_AE_TOLERANCE_DAYS = 1.0


def build_validation_matrices(full_output: pd.DataFrame):
    model_df, delay, missing = prepare_known_clearing_data(full_output)
    X = build_feature_frame(model_df)
    train_pos, test_pos = temporal_train_test_split(model_df)
    y = delay.to_numpy()
    return {
        "model_df": model_df,
        "missing": missing,
        "X": X,
        "X_train": X.iloc[train_pos],
        "X_test": X.iloc[test_pos],
        "y": y,
        "y_train": y[train_pos],
        "y_test": y[test_pos],
        "train_pos": train_pos,
        "test_pos": test_pos,
        "test_index": model_df.index[test_pos],
    }


def append_single_model_rows(single_comparison: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in single_comparison.iterrows():
        row_dict = row.to_dict()
        if row_dict["kind"] == "single_regressor":
            row_dict["source_step"] = "01_single_models"
            rows.append(row_dict)
        elif row_dict["kind"] == "baseline":
            row_dict["source_step"] = "baseline"
            rows.append(row_dict)
    return rows


def add_fast_classifier_candidates(data: dict, rows: list[dict], pred_by_model: dict, fitted_models: dict) -> list[dict]:
    diagnostics = []
    for temperature in TEMPERATURES:
        model, pred, aux = fit_soft_fast_classifier_model(
            data["X_train"],
            data["y_train"],
            data["X_test"],
            temperature=temperature,
        )
        p_fast = model["classifier"].predict_proba(data["X_test"])[:, 1]
        metrics = evaluate_predictions(data["y_test"], pred)
        metrics.update(fast_classifier_metrics(data["y_test"], p_fast))
        model_name = (
            "soft_two_stage_fast_classifier_extra_trees_log"
            if abs(temperature - 1.0) < 1e-12
            else f"tempered_soft_two_stage_fast_classifier_extra_trees_log_T{temperature:g}"
        )
        metrics.update({
            "model": model_name,
            "kind": "binary_fast_classifier_two_stage",
            "source_step": "03_delay_regime_classifier",
            **aux,
        })
        rows.append(metrics)
        diagnostics.append(metrics)
        pred_by_model[model_name] = pred
        fitted_models[model_name] = {"kind": "tempered_soft_fast_classifier", "temperature": temperature}
    return diagnostics


def add_delay_regime_candidates(data: dict, rows: list[dict], pred_by_model: dict, fitted_models: dict) -> list[dict]:
    diagnostics = []
    for temperature in TEMPERATURES:
        model, pred, aux = fit_soft_delay_regime_model(
            data["X_train"],
            data["y_train"],
            data["X_test"],
            temperature=temperature,
        )
        raw_probabilities = model["classifier"].predict_proba(data["X_test"])
        weights = sharpen_multiclass_probabilities(raw_probabilities, temperature)
        metrics = evaluate_predictions(data["y_test"], pred)
        metrics.update(delay_regime_classifier_metrics(data["y_test"], raw_probabilities, model["classifier"].classes_))
        model_name = (
            "soft_three_regime_classifier_extra_trees_log"
            if abs(temperature - 1.0) < 1e-12
            else f"tempered_soft_three_regime_classifier_extra_trees_log_T{temperature:g}"
        )
        metrics.update({
            "model": model_name,
            "kind": "delay_regime_classifier_three_stage",
            "source_step": "03_delay_regime_classifier",
            **aux,
        })
        for label in DELAY_REGIME_LABELS:
            if label in model["classifier"].classes_:
                idx = list(model["classifier"].classes_).index(label)
                metrics[f"raw_p_{label}_mean"] = float(raw_probabilities[:, idx].mean())
                metrics[f"tempered_weight_{label}_mean"] = float(weights[:, idx].mean())
        rows.append(metrics)
        diagnostics.append(metrics)
        pred_by_model[model_name] = pred
        fitted_models[model_name] = {"kind": "tempered_soft_delay_regime_classifier", "temperature": temperature}
    return diagnostics


def select_final_model(comparison: pd.DataFrame) -> tuple[str, dict]:
    single = comparison.loc[comparison["kind"] == "single_regressor"].copy()
    best_single = single.sort_values(["rmse", "mae"]).iloc[0]
    best_single_medae = float(best_single["median_absolute_error"])
    max_allowed_medae = best_single_medae + MEDIAN_AE_TOLERANCE_DAYS

    eligible = comparison.loc[
        (comparison["kind"] != "baseline")
        & (comparison["median_absolute_error"] <= max_allowed_medae)
    ].copy()
    if eligible.empty:
        return str(best_single["model"]), {
            "reason": "no_model_met_median_ae_constraint",
            "best_single_median_absolute_error": best_single_medae,
            "max_allowed_median_absolute_error": max_allowed_medae,
        }
    selected = eligible.sort_values(["rmse", "mae"]).iloc[0]
    return str(selected["model"]), {
        "reason": "lowest_rmse_with_median_ae_constraint",
        "best_single_model": str(best_single["model"]),
        "best_single_median_absolute_error": best_single_medae,
        "max_allowed_median_absolute_error": max_allowed_medae,
        "median_ae_tolerance_days": MEDIAN_AE_TOLERANCE_DAYS,
    }


def build_holdout_predictions(data: dict, selected_name: str, pred_by_model: dict) -> pd.DataFrame:
    holdout = data["model_df"].loc[data["test_index"]].copy()
    holdout["actual_days_to_clear"] = data["y_test"]
    holdout["selected_model"] = selected_name
    for model_name, pred in pred_by_model.items():
        holdout[f"pred__{model_name}"] = pred
        holdout[f"error__{model_name}"] = pred - data["y_test"]
        holdout[f"squared_error__{model_name}"] = (pred - data["y_test"]) ** 2
    holdout["selected_prediction_days"] = pred_by_model[selected_name]
    holdout["selected_error_days"] = holdout["selected_prediction_days"] - data["y_test"]
    holdout["selected_squared_error"] = holdout["selected_error_days"] ** 2
    return holdout


def refit_selected_model(full_output: pd.DataFrame, selected_name: str, fitted_model_info: dict):
    if selected_name in fitted_model_info:
        info = fitted_model_info[selected_name]
        if info["kind"] == "tempered_soft_fast_classifier":
            return refit_fast_classifier_model_on_all_known(full_output, temperature=float(info["temperature"])), info["kind"]
        if info["kind"] == "tempered_soft_delay_regime_classifier":
            return refit_delay_regime_model_on_all_known(full_output, temperature=float(info["temperature"])), info["kind"]
    return refit_single_model_on_all_known(full_output, selected_name)


def run_delay_regime_classifier() -> None:
    create_dir(REGIME_MODEL_DIR)
    full_output = load_full_output()

    single_comparison, single_holdout, _fitted, single_summary = run_single_model_comparison(full_output)
    best_single_name = get_best_single_model_name(single_comparison)
    data = build_validation_matrices(full_output)

    rows = append_single_model_rows(single_comparison)
    pred_by_model = {best_single_name: single_holdout[f"pred__{best_single_name}"].to_numpy()}
    fitted_model_info = {}

    fast_diagnostics = add_fast_classifier_candidates(data, rows, pred_by_model, fitted_model_info)
    regime_diagnostics = add_delay_regime_candidates(data, rows, pred_by_model, fitted_model_info)

    comparison = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    selected_name, selection_info = select_final_model(comparison)
    holdout = build_holdout_predictions(data, selected_name, pred_by_model)

    write_csv(comparison, REGIME_MODEL_DIR / "delay_regime_model_comparison.csv")
    write_csv(pd.DataFrame(fast_diagnostics), REGIME_MODEL_DIR / "binary_fast_classifier_diagnostics.csv")
    write_csv(pd.DataFrame(regime_diagnostics), REGIME_MODEL_DIR / "delay_regime_classifier_diagnostics.csv")
    write_csv(holdout, REGIME_MODEL_DIR / "delay_regime_holdout_predictions.csv")

    final_model, final_kind = refit_selected_model(full_output, selected_name, fitted_model_info)
    completed_output, completion_stats = complete_output_with_model(full_output, final_model, final_kind)
    final_out_dir = CLEARING_DATE_FORECASTER_DIR / DATA_CLEANING_OUTPUT_FULL_FOLDER
    create_dir(final_out_dir)
    write_csv(completed_output, final_out_dir / "output.csv")

    input_path = FEATURE_ENGINEERING_OUTPUT_FULL_DIR / "input.csv"
    if input_path.exists():
        write_csv(pd.read_csv(input_path), final_out_dir / "input.csv")

    selected_row = comparison.loc[comparison["model"] == selected_name].iloc[0].to_dict()
    summary = {
        "selected_model": selected_name,
        "selected_model_kind": final_kind,
        "selected_model_metrics": selected_row,
        "selection_info": selection_info,
        "best_single_model": best_single_name,
        "fast_clearing_days": FAST_CLEARING_DAYS,
        "medium_clearing_days": MEDIUM_CLEARING_DAYS,
        "regime_definition": f"fast <= {FAST_CLEARING_DAYS}; medium {FAST_CLEARING_DAYS + 1}-{MEDIUM_CLEARING_DAYS}; long > {MEDIUM_CLEARING_DAYS}",
        "n_rows_total": int(len(full_output)),
        "n_known_clearing_dates": int(len(data["model_df"])),
        "n_missing_clearing_dates_original": int(data["missing"].sum()),
        "n_holdout_train": int(len(data["train_pos"])),
        "n_holdout_test": int(len(data["test_pos"])),
        "n_model_features": int(data["X"].shape[1]),
        **completion_stats,
    }
    save_json(summary, REGIME_MODEL_DIR / "delay_regime_model_report.json")
    save_json(summary, CLEARING_DATE_FORECASTER_DIR / "clearing_date_forecaster_report.json")
    write_csv(comparison, CLEARING_DATE_FORECASTER_DIR / "clearing_date_model_comparison.csv")
    write_csv(holdout, CLEARING_DATE_FORECASTER_DIR / "holdout_diagnostics.csv")

    log(f"Best single model: {best_single_name}")
    log(f"Selected final model: {selected_name}")
    log(f"Final output written to: {(final_out_dir / 'output.csv').resolve()}")


if __name__ == "__main__":
    run_delay_regime_classifier()
