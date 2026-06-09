from config import *
from util import sales_order_is_empty, copy_dir, create_dir, log, count_dirs, warn, copy_file
import pandas as pd

log("="*20 + " CLEAN EMPTY " + "="*20)
num_reg = count_dirs(SALES_ORDERES_RAW_DIR)
log(f"{num_reg} sales orders present")

sf_report = pd.read_csv(PROCESSED_RENAMED_SF_REPORT)

create_dir(PROCESSED_NO_EMPTY_SALES_ORDERS_DIR)

for sales_order_dir in SALES_ORDERES_RAW_DIR.iterdir():
    if sales_order_dir.is_dir():
        if not sales_order_is_empty(sales_order_dir):
            copy_dir(sales_order_dir, PROCESSED_NO_EMPTY_SALES_ORDERS_DIR)
        else:
            so_num = sales_order_dir.name.replace("so-","")
            sf_report = sf_report[sf_report["sap_so_num"] != so_num]
            log(f"{sales_order_dir.name} is empty")

    else: warn(f"{sales_order_dir.name} is not a directory, skipping")

sf_report.to_csv(PROCESSED_NO_EMPTY_SF_REPORT, index=False)

new_num_reg = count_dirs(PROCESSED_NO_EMPTY_SALES_ORDERS_DIR)
log(f"{new_num_reg} non-empty sales orders present. Removed {num_reg - new_num_reg} ({(num_reg - new_num_reg)/num_reg*100}%)")