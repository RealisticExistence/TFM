from math import isnan

from config import *
from util import copy_file, log, create_dir, file_not_empty, warn
import pandas as pd

log("="*20 + " CONVERT PROCESSED INTO OUTPUT " + "="*20)

last_processed_step_dir = sorted([path for path in PROCESSED_DIR.iterdir()], key=lambda p: p.name)[-1] / "sales_orders"

log(f"Last processed step: {last_processed_step_dir.resolve()}")

create_dir(OUTPUT_DIR)

output_df = pd.DataFrame()
output_clearing_df = pd.DataFrame()
input_df = pd.DataFrame()

for sales_order_dir in last_processed_step_dir.iterdir():
    if sales_order_dir.is_dir():
        so_id = sales_order_dir.name.replace("so-", "")

        invoices_file = sales_order_dir / "invoices.csv"
        clearing_invoices_file = sales_order_dir / "clearing_invoices.csv"
        input_file = sales_order_dir / "input.csv"

        if not file_not_empty(invoices_file):
            warn(f"Skipping {so_id}, no invoices found")
            continue

        if not file_not_empty(input_file):
            warn(f"Skipping {so_id}, no input found")
            continue

        if file_not_empty(clearing_invoices_file):
            clearing_invoices = pd.read_csv(clearing_invoices_file)
            clearing_invoices["sales_order_id"] = so_id
            output_clearing_df = pd.concat([output_clearing_df, clearing_invoices], ignore_index=True)
        else: warn(f"No invoices with clearing found for {so_id}")

        invoices = pd.read_csv(invoices_file)
        input_data = pd.read_csv(input_file)

        invoices["sales_order_id"] = so_id
        input_data["sales_order_id"] = so_id

        output_df = pd.concat([output_df, invoices], ignore_index=True)
        input_df = pd.concat([input_df, input_data], ignore_index=True)

sales_orders = input_df["sales_order_id"].tolist()

output_df.to_csv(OUTPUT_DIR / "output.csv", index=False)
output_clearing_df.to_csv(OUTPUT_DIR / "output_clearing.csv", index=False)
input_df.to_csv(OUTPUT_DIR / "input.csv", index=False)

log(f"Final output:")
log(f"Total sales orders: {len(sales_orders)}")
log(f"Total invoices: {len(output_df.index)}")
log(f"Total invoices w/ clearing: {len(output_clearing_df.index)} ({len(output_clearing_df.index)/len(output_df.index)*100}%)")
