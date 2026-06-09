sf_rename = {
    "SFDC Quote #": "sfdc_quote_num",
    "Id. de la oportunidad": "opportunity_id",
    "Nombre de la oportunidad": "opportunity_name",
    "Descripción": "description",
    "Nombre de la cuenta": "account_name",
    "Country of Installation": "installation_country",
    "Reference No.": "reference_no",
    "Sales Channel": "sales_channel",
    "End User Customer": "end_user_customer",
    "Planning Category": "planning_category",
    "Etapa": "stage",
    "Phase": "phase",
    "Reason for Closure": "reason_for_closure",
    "Winner": "winner",
    "Fecha de cierre": "close_date",
    "Año fiscal": "fiscal_year",
    "Periodo fiscal": "fiscal_period",
    "Sales Comments": "sales_comments",
    "Importe Divisa": "amount_currency",
    "Importe": "amount",
    "Importe (convertido) Divisa": "amount_converted_currency",
    "Importe (convertido)": "amount_converted",
    "Ingresos previstos Divisa": "expected_revenue_currency",
    "Ingresos previstos": "expected_revenue",
    "Ingresos previstos (convertido) Divisa": "expected_revenue_converted_currency",
    "Ingresos previstos (convertido)": "expected_revenue_converted",
    "Winners Price Divisa": "winners_price_currency",
    "Winners Price": "winners_price",
    "Winners Price (convertido) Divisa": "winners_price_converted_currency",
    "Winners Price (convertido)": "winners_price_converted",
    "Propietario de oportunidad": "opportunity_owner",
    "Market Segment": "market_segment",
    "Market Subsegment": "market_subsegment",
    "Product Type": "product_type",
    "SAP SO#": "sap_so_num",
    "Número de contrato": "contract_number",
    "Responsible Business Unit": "responsible_business_unit",
    "Business Segment": "business_segment",
    "Execution Unit": "execution_unit",
    "Execution Unit Function": "execution_unit_function",
    "Id. de la cuenta": "account_id",
    "IFA Number": "ifa_number",
    "Awarded to EPC": "awarded_to_epc",
    "Tipo de cuenta": "account_type",
    "Account Status": "account_status",
    "Country of Customer": "customer_country",
    "Ultimate Parent Name": "ultimate_parent_name",
    "Cuenta principal": "parent_account",
    "Legal Entity (Onshore)": "legal_entity_onshore",
    "Booking ARE": "booking_are",
    "Master Opportunity": "master_opportunity",
    "Divisa de la oportunidad": "opportunity_currency",
    "User Country": "user_country",
    "Country Code": "country_code",
    "Bid Submission Date": "bid_submission_date",
    "Bid Validity": "bid_validity",
    "Proposal Type": "proposal_type",
    "Project Execution Type": "project_execution_type",
    "Origen del candidato": "lead_source",
    "Customer PO #": "customer_po_num",
    "Project": "project",
    "Fiscal Year of Close Date": "close_date_fiscal_year",
    "G2M": "g2m",
    "Business": "business",
    "SE Region Name": "se_region_name",
    "SE Sub Region": "se_sub_region",
    "Tipo de registro de la oportunidad": "opportunity_record_type",
    "Creado por": "created_by",
    "Fecha de creación": "created_date",
}

from config import *
from util import copy_dir, create_dir, log
import pandas as pd

log("="*20 + " RENAME SF REPORT " + "="*20)

create_dir(PROCESSED_RENAMED_SALES_ORDERS_DIR)

df = pd.read_csv(SF_REPORT_RAW, encoding="cp1252")

df = df.rename(columns=sf_rename)

df.to_csv(PROCESSED_RENAMED_SF_REPORT, index=False, encoding="utf-8")

copy_dir(SALES_ORDERES_RAW_DIR, PROCESSED_RENAMED_DIR)