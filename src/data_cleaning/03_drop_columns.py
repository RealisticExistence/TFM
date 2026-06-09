from config import *
from util import apply_csv_files, copy_dir, create_dir, log, drop_constant_columns, warn
import pandas as pd

log("="*20 + " DROP COLUMNS " + "="*20)

sf_report = pd.read_csv(PROCESSED_NO_EMPTY_SF_REPORT)

drop_cols = pd.read_csv(PROCESSED_DROP_COLS_LIST)["drop_cols"].tolist() + ["booking_are", "awarded_to_epc", "user_country", "ifa_number", "opportunity_id", "bid_submission_date", "bid_validity"]

initial_cols = len(sf_report.columns)

log(f"Starting with SF report {initial_cols} columns")

sf_report = sf_report.drop(columns=drop_cols, errors="ignore")

log(f"Dropped {initial_cols-len(drop_cols)}, left with {len(sf_report.columns)} columns")

create_dir(PROCESSED_DROP_COLS_DIR)

po_drop_cols = ["mvt", "sh_text", "trans_event_type"]

log(f"Start dropping columns from purchase order invoices {po_drop_cols}")

copy_dir(PROCESSED_NO_EMPTY_SALES_ORDERS_DIR, PROCESSED_DROP_COLS_DIR)

for sales_order_dir in PROCESSED_DROP_COLS_SALES_ORDERS_DIR.iterdir():
    if sales_order_dir.is_dir():
        apply_csv_files(sales_order_dir / "purchase_orders", lambda df: df.drop(columns=po_drop_cols))

    else: warn(f"{sales_order_dir.name} is not a directory, skipping")

sf_report.to_csv(PROCESSED_DROP_COLS_SF_REPORT, index=False)

