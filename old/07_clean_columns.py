from math import isnan

from pyexpat import errors

from config import *
from util import log, create_dir, expand_date_with_cyclics, parse_net_value, warn, file_not_empty
import pandas as pd

log("="*20 + " CLEAN COLUMNS " + "="*20)

df = pd.read_csv(DATA_CLEANING_OUTPUT_DIR / "input.csv", dtype={"terms_of_payment": str})

create_dir(PROCESSED_CLEAN_COLUMNS_SALES_ORDERS_DIR)

for sales_order_dir in PROCESSED_SEPARATE_X_Y_SALES_ORDERS_DIR.iterdir():
    if sales_order_dir.is_dir():

        new_sales_order_dir = PROCESSED_CLEAN_COLUMNS_SALES_ORDERS_DIR / sales_order_dir.name


        if not file_not_empty(sales_order_dir / "invoices.csv"):
            warn(f"Dropping {sales_order_dir.name}, its empty")
            continue


        create_dir(new_sales_order_dir)

        input_data = pd.read_csv(sales_order_dir / "input.csv", dtype={"terms_of_payment": str})
        invoices = pd.read_csv(sales_order_dir / "invoices.csv", dtype={"clearing_date": str})

        expand_date_with_cyclics("invoice_date", invoices)
        expand_date_with_cyclics("clearing_date", invoices)

        expand_date_with_cyclics("so_date", input_data)
        expand_date_with_cyclics("req_deliv_date", input_data)
        expand_date_with_cyclics("close_date", input_data, day_dict=[2, 1, 3])
        expand_date_with_cyclics("created_date", input_data, day_dict=[2, 1, 3])
        input_data = input_data.drop("Unnamed: 0", axis=1, errors="ignore")

        input_data["net_value"] = input_data["net_value"].apply(parse_net_value)

        input_data.to_csv(new_sales_order_dir / "input.csv", index=False)
        invoices.to_csv(new_sales_order_dir / "invoices.csv", index=False)

        if file_not_empty(sales_order_dir / "clearing_invoices.csv"):
            clearing_invoices = pd.read_csv(sales_order_dir / "clearing_invoices.csv", dtype={"clearing_date": str})
            expand_date_with_cyclics("invoice_date", clearing_invoices)
            expand_date_with_cyclics("clearing_date", clearing_invoices)

            clearing_invoices.to_csv(new_sales_order_dir / "clearing_invoices.csv", index=False)

            if len(invoices.index) != len(clearing_invoices.index):
                warn(f"Cleared and uncleared dont match for {sales_order_dir.name}")
        else: warn(f"No clearing invoices for {sales_order_dir.name}")

log("SF Report integrated with SAP SO data")