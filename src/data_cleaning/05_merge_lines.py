from math import isnan

from config import *
from config import PROCESSED_MERGE_INVOICES_SALES_ORDERS_DIR
from util import copy_file, parse_monetary_amount, log, warn, parse_date, create_dir
import pandas as pd
import re

log("="*20 + " MERGE INVOICES " + "="*20)

create_dir(PROCESSED_MERGE_INVOICES_SALES_ORDERS_DIR)

copy_file(PROCESSED_PARSE_DATE_SF_REPORT, PROCESSED_MERGE_INVOICES_SF_REPORT)

total_invoices = 0
total_clearing_invoices = 0

for sales_order_dir in PROCESSED_PARSE_DATE_SALES_ORDERS_DIR.iterdir():
    if sales_order_dir.is_dir():

        sales_order_new_dir = PROCESSED_MERGE_INVOICES_SALES_ORDERS_DIR / sales_order_dir.name
        create_dir(sales_order_new_dir)

        copy_file(sales_order_dir / "so_info.csv", sales_order_new_dir / "so_info.csv")

        invoices = {}  # All data
        invoices_clearings = {}  # For free cash flow

        for po in (sales_order_dir / "purchase_orders").iterdir():
            if po.is_dir():
                for po_item in po.iterdir():
                    if po_item.is_file():
                        po_item_info = pd.read_csv(po_item)
                        for i, po_item_row in po_item_info.iterrows():
                            mat_doc = po_item_row["material_document"]
                            if not mat_doc in invoices:
                                log(f"New PO invoice, material document {mat_doc}")
                                invoices[mat_doc] = {"invoice_num": mat_doc, "invoice_type": "po_invoice", "invoice_date": None, "amount": 0, "clearing_date": None}

                            amount = parse_monetary_amount(str(po_item_row["amount"]))
                            if isnan(amount): amount = 0

                            invoices[mat_doc]["amount"] += amount

                            if po_item_row["document_date"] == "" or po_item_row["document_date"] is None: continue

                            if invoices[mat_doc]["invoice_date"] is None:
                                invoices[mat_doc]["invoice_date"] = po_item_row["document_date"]
                            elif invoices[mat_doc]["invoice_date"] != po_item_row["document_date"]:
                                warn(f"Invoice date mismatch for material document {mat_doc}")

                            if "clearing_date" in po_item_row.keys():
                                if invoices[mat_doc]["clearing_date"] is None:
                                    invoices[mat_doc]["clearing_date"] = po_item_row["clearing_date"]
                                elif invoices[mat_doc]["clearing_date"] != po_item_row["clearing_date"]:
                                    warn(f"Clearing date mismatch for material document {mat_doc}")

                            if invoices[mat_doc]["clearing_date"] is not None:
                                if re.search(r"\d{2}\.\d{2}\.\d{4}", str(invoices[mat_doc]["clearing_date"])) is not None:
                                    invoices_clearings[mat_doc] = invoices[mat_doc]

        for sales_invoice in (sales_order_dir / "sales_invoices").iterdir():
            if sales_invoice.is_file():
                sales_invoice_info = pd.read_csv(sales_invoice)
                for i, so_item_row in sales_invoice_info.iterrows():
                    doc_num = so_item_row["doc_no"]
                    if not doc_num in invoices:
                        log(f"New sales invoice, document number {doc_num}")
                        invoices[doc_num] = {"invoice_num": doc_num, "invoice_type": so_item_row["doc_type"], "invoice_date": None, "amount": 0, "clearing_date": None}

                    amount = parse_monetary_amount(str(so_item_row["ref_value"]))
                    if isnan(amount): amount = 0

                    invoices[doc_num]["amount"] += amount

                    if so_item_row["on"] == "" or so_item_row["on"] is None: continue

                    if invoices[doc_num]["invoice_date"] is None:
                        invoices[doc_num]["invoice_date"] = so_item_row["on"]
                    elif invoices[doc_num]["invoice_date"] != so_item_row["on"]:
                        warn(f"Invoice date mismatch for material document {doc_num}")

                    if "clearing_date" in so_item_row.keys():
                        if invoices[doc_num]["clearing_date"] is None:
                            invoices[doc_num]["clearing_date"] = so_item_row["clearing_date"]
                        elif invoices[doc_num]["clearing_date"] != so_item_row["clearing_date"]:
                            warn(f"Clearing date mismatch for material document {doc_num}")

                    if invoices[doc_num]["clearing_date"] is not None:
                        if re.search(r"\d{2}\.\d{2}\.\d{4}", str(invoices[doc_num]["clearing_date"])) is not None:
                            invoices_clearings[doc_num] = invoices[doc_num]


        pd.DataFrame(invoices.values()).to_csv(sales_order_new_dir / "invoices.csv", index=False)
        pd.DataFrame(invoices_clearings.values()).to_csv(sales_order_new_dir / "clearing_invoices.csv", index=False)

        total_invoices += len(invoices.keys())
        total_clearing_invoices += len(invoices_clearings.keys())

log(f"Total invoices: {total_invoices}")
log(f"Total invoices w/ clearing date: {total_clearing_invoices} ({total_clearing_invoices/total_invoices*100}%)")