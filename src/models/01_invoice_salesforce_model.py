from config import INVOICE_FORECASTER_DIR
from util import create_dir, log
from models.invoice_forecaster_lib import SALESFORCE_MODEL_NAME, load_full_input_output, train_invoice_model

log("="*20 + " INVOICE FORECASTER - SALESFORCE ONLY " + "="*20)


def run_salesforce_invoice_model() -> None:
    create_dir(INVOICE_FORECASTER_DIR)
    input_df, output_df = load_full_input_output()
    result = train_invoice_model(input_df, output_df, SALESFORCE_MODEL_NAME)
    report = result["report"]
    log(f"Salesforce-only model written to: {(INVOICE_FORECASTER_DIR / SALESFORCE_MODEL_NAME).resolve()}")
    log(f"Count MAE: {report['count_metrics']['mae']:.4f}; within-one count accuracy: {report['count_metrics']['within_one_count_accuracy']:.4f}")
    log(f"Best date model: {report['best_date_model']}")
    log(f"Best amount model: {report['best_amount_model']}")


if __name__ == "__main__":
    run_salesforce_invoice_model()
