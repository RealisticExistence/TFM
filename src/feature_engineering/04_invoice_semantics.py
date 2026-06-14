from config import *
from util import log, run_func_on_outputs
import pandas as pd
import numpy as np
import re

log("="*20 + " INVOICE SEMANTICS " + "="*20)


SALES_INVOICE_TYPES = {
    "final_invoice",
    "down_payment_invoice",
    "down_payment_invoice_recification",
    "partial_invoice",
    "cancelled_invoice",
    "cancelled_partial_invoice",
}

PURCHASE_INVOICE_TYPES = {
    "po_invoice",
}


def _clean_invoice_type(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()
    if text in {"", "nan", "nat", "none", "null"}:
        return pd.NA

    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else pd.NA


def _parse_amount(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if text in {"", "nan", "NaN", "NaT", "None", "NULL"}:
        return np.nan

    # Handle accounting-style negatives: (1,234.56)
    negative_parentheses = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("€", "").replace("EUR", "")
    text = text.replace(" ", "")

    # Existing processed data uses decimal points and optional commas as
    # thousands separators. This also handles simple European decimal commas.
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")

    try:
        number = float(text)
    except ValueError:
        return np.nan

    return -abs(number) if negative_parentheses else number


def process_invoice_semantics(input_df: pd.DataFrame, output_df: pd.DataFrame, dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds invoice-type flags and explicit amount columns for modelling.

    Important modelling choice:
    - raw amount keeps the sign as scraped/processed from SAP.
    - company-signed amount changes purchase invoices into outflows by
      multiplying purchase invoice amounts by -1. Sales invoices remain inflows.
    - negative sales invoices/credits remain negative because the raw sign is
      preserved before applying the sales/purchase direction multiplier.
    """

    input_mod = input_df.copy()
    output_mod = output_df.copy()

    if "invoice_type" not in output_mod.columns:
        return input_mod, output_mod

    invoice_type = output_mod["invoice_type"].apply(_clean_invoice_type).astype("string")
    output_mod["invoice_type_clean"] = invoice_type

    output_mod["invoice_is_purchase"] = invoice_type.isin(PURCHASE_INVOICE_TYPES).astype("boolean")
    output_mod["invoice_is_sales"] = invoice_type.isin(SALES_INVOICE_TYPES).astype("boolean")
    output_mod["invoice_is_final"] = invoice_type.eq("final_invoice").astype("boolean")
    output_mod["invoice_is_down_payment"] = invoice_type.str.contains("down_payment", regex=False, na=False).astype("boolean")
    output_mod["invoice_is_partial"] = invoice_type.str.contains("partial", regex=False, na=False).astype("boolean")
    output_mod["invoice_is_cancelled"] = invoice_type.str.contains("cancelled", regex=False, na=False).astype("boolean")
    output_mod["invoice_is_rectification"] = (
        invoice_type.str.contains("rectification", regex=False, na=False)
        | invoice_type.str.contains("recification", regex=False, na=False)
    ).astype("boolean")

    if "amount" in output_mod.columns:
        raw_amount = output_mod["amount"].apply(_parse_amount)
        output_mod["invoice_amount_raw_signed"] = pd.to_numeric(raw_amount, errors="coerce")
        output_mod["invoice_amount_abs"] = output_mod["invoice_amount_raw_signed"].abs()
        output_mod["invoice_is_credit_or_negative"] = (output_mod["invoice_amount_raw_signed"] < 0).astype("boolean")

        direction = pd.Series(np.nan, index=output_mod.index, dtype="float64")
        direction = direction.mask(output_mod["invoice_is_sales"].fillna(False), 1.0)
        direction = direction.mask(output_mod["invoice_is_purchase"].fillna(False), -1.0)

        output_mod["invoice_cash_flow_direction_multiplier"] = direction
        output_mod["invoice_cash_flow_direction"] = pd.Series(pd.NA, index=output_mod.index, dtype="string")
        output_mod.loc[output_mod["invoice_cash_flow_direction_multiplier"].eq(1.0), "invoice_cash_flow_direction"] = "inflow"
        output_mod.loc[output_mod["invoice_cash_flow_direction_multiplier"].eq(-1.0), "invoice_cash_flow_direction"] = "outflow"
        output_mod["invoice_amount_company_signed"] = (
            output_mod["invoice_amount_raw_signed"] * output_mod["invoice_cash_flow_direction_multiplier"]
        )
        output_mod["invoice_amount_company_abs"] = output_mod["invoice_amount_company_signed"].abs()

    return input_mod, output_mod


run_func_on_outputs(INCOTERMS_DIR, INVOICE_SEMANTICS_DIR, process_invoice_semantics)
