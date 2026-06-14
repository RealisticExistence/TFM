from config import INVOICE_FORECASTER_DIR
from util import create_dir, log
from models.invoice_forecaster_lib import (
    SALESFORCE_SAP_MODEL_NAME,
    load_full_input_output,
    build_order_table,
    make_salesforce_prediction_features,
    temporal_order_split,
    train_invoice_model,
)

log("="*20 + " INVOICE FORECASTER - SALESFORCE + SAP " + "="*20)


def run_salesforce_sap_invoice_model() -> None:
    create_dir(INVOICE_FORECASTER_DIR)
    input_df, output_df = load_full_input_output()
    orders = build_order_table(input_df, output_df, SALESFORCE_SAP_MODEL_NAME)
    train_pos, test_pos = temporal_order_split(orders)
    train_orders = orders.iloc[train_pos]["sales_order_id"].astype(str)
    test_orders = orders.iloc[test_pos]["sales_order_id"].astype(str)

    sf_count_features, sf_sequence_features, _sf_result = make_salesforce_prediction_features(
        input_df, output_df, train_orders, test_orders
    )
    result = train_invoice_model(
        input_df,
        output_df,
        SALESFORCE_SAP_MODEL_NAME,
        train_orders=train_orders,
        test_orders=test_orders,
        sf_count_features=sf_count_features,
        sf_sequence_features=sf_sequence_features,
    )
    report = result["report"]
    log(f"Salesforce+SAP model written to: {(INVOICE_FORECASTER_DIR / SALESFORCE_SAP_MODEL_NAME).resolve()}")
    log(f"Count MAE: {report['count_metrics']['mae']:.4f}; within-one count accuracy: {report['count_metrics']['within_one_count_accuracy']:.4f}")
    log(f"Best date model: {report['best_date_model']}")
    log(f"Best amount model: {report['best_amount_model']}")


if __name__ == "__main__":
    run_salesforce_sap_invoice_model()
