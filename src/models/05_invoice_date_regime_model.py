from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from config import INVOICE_FORECASTER_DIR
from util import create_dir, log, write_csv
from models.invoice_forecaster_lib import (
    RANDOM_STATE,
    SALESFORCE_MODEL_NAME,
    SALESFORCE_SAP_MODEL_NAME,
    build_feature_frame,
    build_invoice_training_table,
    build_order_table,
    date_model_specs,
    evaluate_regression,
    fit_log_regressor,
    fit_transformed_regressor,
    inverse_transformed_prediction,
    load_full_input_output,
    make_pipeline,
    make_salesforce_prediction_features,
    temporal_order_split,
)

log("="*20 + " INVOICE DATE REGIME MODEL " + "="*20)

REGIME_DIR = INVOICE_FORECASTER_DIR / "date_regime_model"
REGIME_LABELS = ["early_le_120", "standard_121_360", "late_gt_360"]


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def assign_date_regime(days: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(days), errors="coerce")
    out = pd.Series("standard_121_360", index=values.index, dtype="object")
    out.loc[values <= 120] = "early_le_120"
    out.loc[values > 360] = "late_gt_360"
    return out.to_numpy(dtype=str)


def sharpen_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.clip(probabilities, 1e-8, 1.0)
    logits = np.log(probs)
    scaled = np.exp(logits / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_regime_regressors(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    regimes_train: np.ndarray,
    target_transform: str,
) -> dict[str, Any]:
    regressors = {}
    global_estimator = ExtraTreesRegressor(
        n_estimators=50,
        max_features=0.75,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    global_model, _, global_metadata = fit_transformed_regressor(global_estimator, X_train, y_train, X_train, target_transform)
    for regime in REGIME_LABELS:
        mask = regimes_train == regime
        if mask.sum() < 20:
            regressors[regime] = {"model": global_model, "metadata": global_metadata}
            continue
        estimator = ExtraTreesRegressor(
            n_estimators=50,
            max_features=0.75,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model, _, metadata = fit_transformed_regressor(estimator, X_train.loc[mask], y_train[mask], X_train.loc[mask], target_transform)
        regressors[regime] = {"model": model, "metadata": metadata}
    return regressors


def predict_regime_blend(regressors: dict[str, Any], X_test: pd.DataFrame, probabilities: np.ndarray, classes: np.ndarray, temperature: float) -> np.ndarray:
    class_to_pos = {c: i for i, c in enumerate(classes)}
    full_probs = np.zeros((len(X_test), len(REGIME_LABELS)))
    for j, regime in enumerate(REGIME_LABELS):
        if regime in class_to_pos:
            full_probs[:, j] = probabilities[:, class_to_pos[regime]]
    weights = sharpen_probabilities(full_probs, temperature)
    pred = np.zeros(len(X_test), dtype=float)
    for j, regime in enumerate(REGIME_LABELS):
        fitted = regressors[regime]
        pred_regime = inverse_transformed_prediction(fitted["model"].predict(X_test), fitted["metadata"])
        pred += weights[:, j] * pred_regime
    return pred


def prepare_model_data(input_df: pd.DataFrame, output_df: pd.DataFrame, feature_set: str):
    orders = build_order_table(input_df, output_df, feature_set)
    train_pos, test_pos = temporal_order_split(orders)
    train_orders = orders.iloc[train_pos]["sales_order_id"].astype(str)
    test_orders = orders.iloc[test_pos]["sales_order_id"].astype(str)
    sf_sequence_features = None
    if feature_set == SALESFORCE_SAP_MODEL_NAME:
        sf_count_features, sf_sequence_features, _ = make_salesforce_prediction_features(input_df, output_df, train_orders, test_orders)
    rows = build_invoice_training_table(input_df, output_df, feature_set)
    if sf_sequence_features is not None:
        rows = rows.merge(sf_sequence_features, on=["sales_order_id", "target_invoice_sequence"], how="left")
    rows = rows.loc[rows["target_days_to_invoice"].notna()].copy()
    train_mask = rows["sales_order_id"].astype(str).isin(train_orders.astype(str)).to_numpy()
    test_mask = rows["sales_order_id"].astype(str).isin(test_orders.astype(str)).to_numpy()
    X = build_feature_frame(rows, feature_set)
    y = rows["target_days_to_invoice"].to_numpy(dtype=float)
    return rows, X, y, train_mask, test_mask


def run_regime_model_for_feature_set(feature_set: str) -> dict[str, Any]:
    input_df, output_df = load_full_input_output()
    rows, X, y, train_mask, test_mask = prepare_model_data(input_df, output_df, feature_set)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_test = y[test_mask]
    regimes_train = assign_date_regime(y[train_mask])
    regimes_test = assign_date_regime(y_test)

    # Single-model baseline using a small subset of the date candidates. The full
    # raw/log/signed-log comparison is written by the main invoice scripts; this
    # script focuses on whether a regime classifier adds value on top of the best
    # log-style baselines.
    candidate_rows = []
    baseline_predictions = {}
    baseline_names = {"gradient_boosting_log_offset", "extra_trees_smooth_log_offset"}
    for name, spec in date_model_specs().items():
        if name not in baseline_names:
            continue
        model, pred, metadata = fit_transformed_regressor(
            spec["estimator"],
            X_train,
            y[train_mask],
            X_test,
            spec["target_transform"],
        )
        baseline_predictions[name] = pred
        metrics = evaluate_regression(y_test, pred)
        metrics.update({
            "feature_set": feature_set,
            "model": name,
            "model_family": "single_regressor",
            "target_transform": spec["target_transform"],
        })
        candidate_rows.append(metrics)
    best_single = pd.DataFrame(candidate_rows).sort_values(["rmse", "mae"]).iloc[0]

    classifier = make_pipeline(
        ExtraTreesClassifier(
            n_estimators=80,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        X_train,
    )
    classifier.fit(X_train, regimes_train)
    regime_pred = classifier.predict(X_test)
    regime_prob = classifier.predict_proba(X_test)
    regime_regressor_sets = {
        "shifted_log": fit_regime_regressors(X_train, y[train_mask], regimes_train, "shifted_log"),
    }

    diagnostics = {
        "feature_set": feature_set,
        "regime_accuracy": float(accuracy_score(regimes_test, regime_pred)),
        "regime_balanced_accuracy": float(balanced_accuracy_score(regimes_test, regime_pred)),
        "regime_macro_f1": float(f1_score(regimes_test, regime_pred, average="macro", zero_division=0)),
    }
    for target_transform, regressors in regime_regressor_sets.items():
        for temperature in [1.0, 0.8, 0.6, 0.5]:
            pred = predict_regime_blend(regressors, X_test, regime_prob, classifier.named_steps["model"].classes_, temperature)
            metrics = evaluate_regression(y_test, pred)
            metrics.update({
                "feature_set": feature_set,
                "model": f"three_regime_extra_trees_{target_transform}_T{temperature}",
                "model_family": "three_regime_soft_blend",
                "target_transform": target_transform,
                "temperature": temperature,
            })
            candidate_rows.append(metrics)

    comparison = pd.DataFrame(candidate_rows).sort_values(["rmse", "mae"])
    out_dir = REGIME_DIR / feature_set
    create_dir(out_dir)
    write_csv(comparison, out_dir / "invoice_date_regime_model_comparison.csv")
    save_json({"diagnostics": diagnostics, "best_single_model": dict(best_single), "best_overall_model": comparison.iloc[0].to_dict()}, out_dir / "invoice_date_regime_model_report.json")
    return {"feature_set": feature_set, "diagnostics": diagnostics, "comparison": comparison}


def run_invoice_date_regime_models() -> None:
    create_dir(REGIME_DIR)
    requested_feature_set = os.environ.get("INVOICE_DATE_REGIME_FEATURE_SET")
    if requested_feature_set:
        log(f"Running invoice-date regime model for: {requested_feature_set}")
        run_regime_model_for_feature_set(requested_feature_set)
        log(f"Finished invoice-date regime model for: {requested_feature_set}")
        return

    # Run each feature set in a fresh Python process. This avoids occasional long
    # sklearn cleanup/parallel-state stalls when repeatedly fitting many tree
    # pipelines in one interpreter.
    for feature_set in [SALESFORCE_MODEL_NAME, SALESFORCE_SAP_MODEL_NAME]:
        env = os.environ.copy()
        env["INVOICE_DATE_REGIME_FEATURE_SET"] = feature_set
        subprocess.run([sys.executable, __file__], check=True, env=env)

    comparisons = []
    reports = []
    for feature_set in [SALESFORCE_MODEL_NAME, SALESFORCE_SAP_MODEL_NAME]:
        out_dir = REGIME_DIR / feature_set
        comparisons.append(pd.read_csv(out_dir / "invoice_date_regime_model_comparison.csv"))
        with open(out_dir / "invoice_date_regime_model_report.json", "r", encoding="utf-8") as f:
            reports.append(json.load(f)["diagnostics"])
    write_csv(pd.concat(comparisons, ignore_index=True), REGIME_DIR / "invoice_date_regime_model_comparison_all.csv")
    save_json({"classifier_diagnostics": reports}, REGIME_DIR / "invoice_date_regime_model_summary.json")
    log(f"Invoice date regime model results written to: {REGIME_DIR.resolve()}")

if __name__ == "__main__":
    run_invoice_date_regime_models()
