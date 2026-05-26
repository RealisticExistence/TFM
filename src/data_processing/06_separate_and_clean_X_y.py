from math import isnan

from config import *
from util import copy_file, log, create_dir
import pandas as pd

log("="*20 + " SEPARATE X AND Y " + "="*20)

so_info_columns = [
    "client_number",
    "so_date",
    "incoterms",
    "terms_of_payment",
    "wbs_element",
    "req_deliv_date",
    "net_value"
]

create_dir(PROCESSED_SEPARATE_X_Y_SALES_ORDERS_DIR)

sf_report = pd.read_csv(PROCESSED_MERGE_INVOICES_SF_REPORT)

for sales_order_dir in PROCESSED_MERGE_INVOICES_SALES_ORDERS_DIR.iterdir():
    if sales_order_dir.is_dir():

        sales_order_new_dir = PROCESSED_SEPARATE_X_Y_SALES_ORDERS_DIR / sales_order_dir.name
        create_dir(sales_order_new_dir)

        copy_file(sales_order_dir / "invoices.csv", sales_order_new_dir / "invoices.csv")
        copy_file(sales_order_dir / "clearing_invoices.csv", sales_order_new_dir / "clearing_invoices.csv")

        so_id = sales_order_dir.name.replace("so-", "")
        so_info = pd.read_csv(sales_order_dir / "so_info.csv", usecols=so_info_columns)
        sf_report_row = sf_report[sf_report["sap_so_num"] == so_id]
        final_so_info = pd.concat([so_info.reset_index(drop=True), sf_report_row.reset_index(drop=True)], axis=1)
        final_so_info.to_csv(sales_order_new_dir / "input.csv", index=False)

log("SF Report integrated with SAP SO data")