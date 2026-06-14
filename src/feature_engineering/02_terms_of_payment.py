from config import *
from util import log, run_func_on_outputs
import pandas as pd
import numpy as np
import re

log("="*20 + " TERMS OF PAYMENT " + "="*20)


def process_terms_of_payment(input_df: pd.DataFrame, output_df: pd.DataFrame, dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Interprets the SAP terms_of_payment code without hard-coding company-specific
    accounting rules. The safest generic interpretation is the numeric payment
    period at the end of the code, for example:
      - 3E60 -> 60 days
      - 3E30 -> 30 days
      - 3220 -> 20 days, 3205 -> 5 days
      - 32 6 -> 6 days
      - 32-C -> unknown days, suffix C

    The derived columns are added to the input table and copied to output rows
    through sales_order_id. Output rows also receive invoice-based due-date
    columns when invoice_date is available.
    """

    ORDER_ID = "sales_order_id"
    TERMS_COL = "terms_of_payment"

    def clean_code(value):
        if pd.isna(value):
            return pd.NA

        raw = str(value).strip().upper()
        if raw in {"", "NAN", "NAT", "NONE", "NULL"}:
            return pd.NA

        return re.sub(r"\s+", " ", raw)

    def compact_code(cleaned):
        if pd.isna(cleaned):
            return pd.NA
        return re.sub(r"[\s\-_/]+", "", str(cleaned).upper())

    def suffix_letter(cleaned):
        if pd.isna(cleaned):
            return pd.NA

        match = re.search(r"[-\s]?([A-Z])$", str(cleaned).upper())
        if match:
            return match.group(1)

        return pd.NA

    def family_code(cleaned):
        if pd.isna(cleaned):
            return pd.NA

        text = str(cleaned).upper().strip()
        compact = compact_code(text)

        # Keep explicit families such as "32" in "32 6" or "32-C".
        match = re.match(r"^(\d{2})(?:\s|[-_/])", text)
        if match:
            return match.group(1)

        # SAP-like numeric codes in this dataset appear as 32xx, 31x, 34x.
        if isinstance(compact, str) and len(compact) >= 2 and compact[:2].isdigit():
            return compact[:2]

        # Alphanumeric families such as 3E60 or DO60.
        match = re.match(r"^([0-9A-Z]+?)(\d{1,3})$", str(compact))
        if match:
            return match.group(1)

        return compact

    def extract_days(cleaned) -> float:
        if pd.isna(cleaned):
            return np.nan

        text = str(cleaned).upper().strip()
        compact = compact_code(text)

        # Codes with a separated final numeric token: "32 6", "31 3", "34 1".
        match = re.search(r"(?:\s|[-_/])(\d{1,3})$", text)
        if match:
            days = int(match.group(1))
            return float(days) if 0 <= days <= 365 else np.nan

        # Alphanumeric codes ending in the day count: "3E60", "3E30", "DO60".
        match = re.search(r"[A-Z]+(\d{1,3})$", str(compact))
        if match:
            days = int(match.group(1))
            return float(days) if 0 <= days <= 365 else np.nan

        # Numeric SAP-like codes: use the last two digits as the payment days.
        # Examples in the data: 3220 -> 20, 3205 -> 5, 3231 -> 31.
        if isinstance(compact, str) and compact.isdigit():
            if len(compact) >= 4:
                days = int(compact[-2:])
            elif len(compact) == 3:
                days = int(compact[-1:])
            else:
                days = int(compact)
            return float(days) if 0 <= days <= 365 else np.nan

        return np.nan

    def bucket_days(days) -> str:
        if pd.isna(days):
            return "unknown"

        days = int(days)
        if days == 0:
            return "immediate"
        if days <= 15:
            return "001_015_days"
        if days <= 30:
            return "016_030_days"
        if days <= 60:
            return "031_060_days"
        if days <= 90:
            return "061_090_days"
        return "gt_090_days"

    def parse_mixed_date(s: pd.Series) -> pd.Series:
        s = s.astype("string").str.strip()
        s = s.mask(s.isin(["", "nan", "NaN", "NaT", "None", "NULL"]))

        try:
            return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)
        except TypeError:
            return pd.to_datetime(s, errors="coerce", dayfirst=True)

    def add_terms_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if TERMS_COL not in df.columns:
            return df

        clean = df[TERMS_COL].apply(clean_code)
        compact = clean.apply(compact_code)
        days = clean.apply(extract_days)
        suffix = clean.apply(suffix_letter)

        df["terms_of_payment_code_clean"] = clean
        df["terms_of_payment_code_compact"] = compact
        df["terms_of_payment_family"] = clean.apply(family_code)
        df["terms_of_payment_suffix_letter"] = suffix
        df["terms_of_payment_days"] = pd.Series(days, index=df.index).round().astype("Int64")
        df["terms_of_payment_bucket"] = df["terms_of_payment_days"].apply(bucket_days)
        df["terms_of_payment_is_interpretable"] = df["terms_of_payment_days"].notna()
        df["terms_of_payment_is_numeric_code"] = compact.astype("string").str.fullmatch(r"\d+").fillna(False)
        df["terms_of_payment_has_alpha"] = compact.astype("string").str.contains(r"[A-Z]", regex=True).fillna(False)
        df["terms_of_payment_has_e_marker"] = compact.astype("string").str.contains("E", regex=False).fillna(False)
        df["terms_of_payment_has_suffix_letter"] = suffix.notna()

        return df

    def add_due_date_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "invoice_date" not in df.columns or "terms_of_payment_days" not in df.columns:
            return df

        invoice_date = parse_mixed_date(df["invoice_date"])
        payment_days = pd.to_numeric(df["terms_of_payment_days"], errors="coerce")
        df["payment_due_date"] = invoice_date + pd.to_timedelta(payment_days, unit="D")
        df["days__invoice_date__to__payment_due_date"] = payment_days

        if "clearing_date" in df.columns:
            clearing_date = parse_mixed_date(df["clearing_date"])
            df["days__payment_due_date__to__clearing_date"] = (clearing_date - df["payment_due_date"]).dt.days
            df["cleared_after_payment_due_date"] = (df["days__payment_due_date__to__clearing_date"] > 0).astype("boolean")
            df.loc[df["days__payment_due_date__to__clearing_date"].isna(), "cleared_after_payment_due_date"] = pd.NA

        return df

    input_mod = add_terms_columns(input_df)
    output_mod = output_df.copy()

    if TERMS_COL in input_mod.columns and ORDER_ID in input_mod.columns and ORDER_ID in output_mod.columns:
        terms_cols = [
            ORDER_ID,
            TERMS_COL,
            "terms_of_payment_code_clean",
            "terms_of_payment_code_compact",
            "terms_of_payment_family",
            "terms_of_payment_suffix_letter",
            "terms_of_payment_days",
            "terms_of_payment_bucket",
            "terms_of_payment_is_interpretable",
            "terms_of_payment_is_numeric_code",
            "terms_of_payment_has_alpha",
            "terms_of_payment_has_e_marker",
            "terms_of_payment_has_suffix_letter",
        ]
        terms_cols = [c for c in terms_cols if c in input_mod.columns]

        lookup = (
            input_mod[terms_cols]
            .drop_duplicates(subset=[ORDER_ID], keep="first")
            .copy()
        )

        output_mod["__terms_row_id"] = np.arange(len(output_mod))
        output_mod = output_mod.drop(columns=[c for c in terms_cols if c != ORDER_ID and c in output_mod.columns], errors="ignore")
        output_mod = output_mod.merge(lookup, on=ORDER_ID, how="left")
        output_mod = output_mod.sort_values("__terms_row_id").drop(columns=["__terms_row_id"]).reset_index(drop=True)
        output_mod = add_due_date_columns(output_mod)

    return input_mod, output_mod


run_func_on_outputs(EXPAND_DATES_DIR, TERMS_OF_PAYMENT_DIR, process_terms_of_payment)
