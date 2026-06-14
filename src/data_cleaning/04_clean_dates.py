from datetime import datetime

from config import *
from util import copy_file, copy_dir, log, warn, parse_date
import pandas as pd

log("="*20 + " CLEAN DATES " + "="*20)

log("Searching invalid dates coming from OCR (i.e. so_info/so_date, so_info/req_deliv_date and both clearing dates)")

copy_dir(PROCESSED_DROP_COLS_SALES_ORDERS_DIR, PROCESSED_PARSE_DATE_DIR)

sf_report = pd.read_csv(PROCESSED_DROP_COLS_SF_REPORT)
sf_report["created_date"] = sf_report["created_date"].str.replace("/", ".")
sf_report["close_date"] = sf_report["close_date"].str.replace("/", ".")
sf_report["bid_submission_date"] = sf_report["bid_submission_date"].str.replace("/", ".")
sf_report["bid_validity"] = sf_report["bid_validity"].str.replace("/", ".")
sf_report.to_csv(PROCESSED_PARSE_DATE_SF_REPORT)

total_po_invoice_count = 0
missing_clearing_po_invoice_count = 0

total_sales_invoice_count = 0
missing_clearing_sales_invoice_count = 0

for sales_order_dir in PROCESSED_PARSE_DATE_SALES_ORDERS_DIR.iterdir():
    if sales_order_dir.is_dir():
        so_info = pd.read_csv(sales_order_dir / "so_info.csv", dtype={"terms_of_payment": str, "so_date": str, "req_deliv_date": str})
        so_info["so_date"] = so_info["so_date"].apply(parse_date)
        so_info["req_deliv_date"] = so_info["req_deliv_date"].apply(parse_date)
        so_info.to_csv(sales_order_dir / "so_info.csv")

        if so_info["so_date"].any() == "":
            warn(f"Sales order data {sales_order_dir.resolve()} doesnt have date")

        if so_info["req_deliv_date"].any() == "":
            warn(f"Sales order data {sales_order_dir.resolve()} doesnt have requested delivery date")

        for po in (sales_order_dir / "purchase_orders").iterdir():
            if po.is_dir():
                for po_invoice in po.iterdir():
                    if po_invoice.is_file():
                        po_invoice_info = pd.read_csv(po_invoice)
                        total_po_invoice_count += 1
                        if po_invoice_info.empty:
                            missing_clearing_po_invoice_count += 1
                            warn(f"PO data {po_invoice.resolve()} is empty")
                            continue
                        if not "clearing_date" in po_invoice_info.columns:
                            missing_clearing_po_invoice_count += 1
                            warn(f"PO data {po_invoice.resolve()} doesnt have clearing date")
                            continue
                        po_invoice_info["clearing_date"] = po_invoice_info["clearing_date"].apply(parse_date)
                        if po_invoice_info["clearing_date"].any() == "":
                            warn(f"PO data {po_invoice.resolve()} doesnt have clearing date")
                            missing_clearing_po_invoice_count += 1
                        elif datetime.strptime(po_invoice_info["clearing_date"].tolist()[0],"%d.%m.%Y") < datetime.strptime(po_invoice_info["document_date"].tolist()[0],"%d.%m.%Y"):
                            warn(f"PO data {po_invoice.resolve()} has an invalid clearing date")
                            missing_clearing_po_invoice_count += 1
                            po_invoice_info["clearing_date"] = None
                        po_invoice_info.to_csv(po_invoice)

        for sales_invoice in (sales_order_dir / "sales_invoices").iterdir():
            if sales_invoice.is_file():
                sales_invoice_info = pd.read_csv(sales_invoice)
                total_sales_invoice_count += 1
                if sales_invoice_info.empty:
                    missing_clearing_sales_invoice_count += 1
                    warn(f"Sales invoice {sales_invoice.resolve()} is empty")
                    continue
                if not "clearing_date" in sales_invoice_info.columns:
                    missing_clearing_sales_invoice_count += 1
                    warn(f"Sales invoice {sales_invoice.resolve()} doesnt have clearing date")
                    continue
                sales_invoice_info["clearing_date"] = sales_invoice_info["clearing_date"].apply(parse_date)

                if sales_invoice_info["clearing_date"].any() == "":
                    warn(f"Sales invoice {sales_invoice.resolve()} doesnt have clearing date")
                    missing_clearing_sales_invoice_count += 1
                elif datetime.strptime(sales_invoice_info["clearing_date"].tolist()[0], "%d.%m.%Y") < datetime.strptime(
                        sales_invoice_info["on"].tolist()[0], "%d.%m.%Y"):
                    warn(f"Sales invoice {sales_invoice.resolve()} has an invalid clearing date")
                    missing_clearing_sales_invoice_count += 1
                    sales_invoice_info["clearing_date"] = None
                sales_invoice_info.to_csv(sales_invoice)


    else: warn(f"{sales_order_dir.name} is not a directory, skipping")

log(f"Total PO line entries: {total_po_invoice_count}")
log(f"Total sales invoices: {total_sales_invoice_count}")
log(f"Missing clearings on PO: {missing_clearing_po_invoice_count/total_po_invoice_count*100}%")
log(f"Missing clearings on sales invoices: {missing_clearing_sales_invoice_count/total_sales_invoice_count*100}%")
