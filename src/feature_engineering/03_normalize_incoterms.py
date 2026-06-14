from config import *
from util import log, run_func_on_outputs
import pandas as pd
import numpy as np
import re
import unicodedata

log("="*20 + " NORMALIZE INCOTERMS " + "="*20)


STANDARD_INCOTERM_CODES = {
    "EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"
}

INCOTERM_GROUPS = {
    "EXW": "E",
    "FCA": "F", "FAS": "F", "FOB": "F",
    "CFR": "C", "CIF": "C", "CPT": "C", "CIP": "C",
    "DAP": "D", "DPU": "D", "DDP": "D",
}

# Broad commercial interpretation for model features. These are not intended to
# replace legal Incoterms analysis; they are compact encodings of the main
# logistics responsibility implied by the code.
SELLER_PAYS_MAIN_TRANSPORT = {"CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"}
SELLER_DELIVERS_TO_DESTINATION = {"DAP", "DPU", "DDP"}
SELLER_PAYS_IMPORT_DUTIES = {"DDP"}

SPAIN_LOCATION_PATTERNS = {
    "SPAIN", "ESPANA", "ESPANYA", "ESPAÑA", "ESPAFA", "DESTINO",
    "MADRID", "VALENCIA", "NAVARRA", "CUENCA", "ALBACETE", "VALLADOLID",
    "LEON", "GALICIA", "SORIA", "GUADALAJARA", "MIAJADAS", "TABERNAS",
    "JUMILLA", "MURCIA", "BARAJAS", "TORDESILLAS", "CABRERIZA", "NIJAR",
    "CARRION", "MARCHAMALO", "HUERTA", "CARMONA", "CARMON", "PENARR",
}


def _strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def _clean_text(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().upper()
    if text in {"", "NAN", "NAT", "NONE", "NULL"}:
        return pd.NA

    text = _strip_accents(text)
    text = re.sub(r"[\u00A0\t\r\n]+", " ", text)
    text = re.sub(r"[^0-9A-Z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA


def _extract_code(cleaned):
    if pd.isna(cleaned):
        return pd.NA

    text = str(cleaned).upper().strip()
    first_token = text.split(" ", 1)[0]

    # Normal case: "DAP SPAIN", "DDP Destination", "CIF Valencia".
    if first_token[:3] in STANDARD_INCOTERM_CODES:
        return first_token[:3]

    # Handle occasional text where the code is preceded by noise.
    match = re.search(r"\b(EXW|FCA|FAS|FOB|CFR|CIF|CPT|CIP|DAP|DPU|DDP)\b", text)
    if match:
        return match.group(1)

    return pd.NA


def _extract_unknown_prefix(cleaned):
    if pd.isna(cleaned):
        return pd.NA

    text = str(cleaned).upper().strip()
    token = text.split(" ", 1)[0]
    return token if token else pd.NA


def _extract_location(cleaned, code):
    if pd.isna(cleaned):
        return pd.NA

    text = str(cleaned).upper().strip()

    if not pd.isna(code):
        location = re.sub(rf"^.*?\b{re.escape(str(code))}\b\s*", "", text, count=1).strip()
    else:
        parts = text.split(" ", 1)
        location = parts[1].strip() if len(parts) > 1 else ""

    location = re.sub(r"\s+", " ", location).strip()
    return location if location else pd.NA


def _normalize_location(location):
    if pd.isna(location):
        return pd.NA

    text = _clean_text(location)
    if pd.isna(text):
        return pd.NA

    replacements = {
        "ESPANA": "SPAIN",
        "ESPANYA": "SPAIN",
        "ESPAÑA": "SPAIN",
        "ESPAFA": "SPAIN",
        "DESTINATION": "DESTINATION",
        "DESTINO": "DESTINATION",
    }
    tokens = [replacements.get(tok, tok) for tok in str(text).split()]
    normalized = " ".join(tokens)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if normalized else pd.NA


def _country_or_region(normalized_location):
    if pd.isna(normalized_location):
        return pd.NA

    text = str(normalized_location).upper()
    tokens = set(text.split())

    if tokens & SPAIN_LOCATION_PATTERNS:
        return "SPAIN"

    return "OTHER_OR_UNKNOWN"


def _safe_bool(series):
    return series.astype("boolean")


def add_incoterm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = "incoterms"

    if col not in df.columns:
        return df

    clean = df[col].apply(_clean_text)
    code = clean.apply(_extract_code)
    location_raw = [_extract_location(cleaned, extracted_code) for cleaned, extracted_code in zip(clean, code)]
    location_raw = pd.Series(location_raw, index=df.index, dtype="string")
    location_norm = location_raw.apply(_normalize_location)
    country_region = location_norm.apply(_country_or_region)
    unknown_prefix = clean.apply(_extract_unknown_prefix)

    df["incoterms_raw"] = df[col]
    df["incoterms_clean"] = clean
    df["incoterms_code"] = code.astype("string")
    df["incoterms_unknown_prefix"] = unknown_prefix.astype("string").where(code.isna(), pd.NA)
    df["incoterms_location_raw"] = location_raw
    df["incoterms_location_normalized"] = location_norm.astype("string")
    df["incoterms_country_or_region"] = country_region.astype("string")

    df["incoterms_is_standard_code"] = _safe_bool(df["incoterms_code"].isin(STANDARD_INCOTERM_CODES))
    df["incoterms_has_location"] = _safe_bool(df["incoterms_location_normalized"].notna())
    df["incoterms_group"] = df["incoterms_code"].map(INCOTERM_GROUPS).astype("string")
    df["incoterms_seller_pays_main_transport"] = _safe_bool(df["incoterms_code"].isin(SELLER_PAYS_MAIN_TRANSPORT))
    df["incoterms_seller_delivers_to_destination"] = _safe_bool(df["incoterms_code"].isin(SELLER_DELIVERS_TO_DESTINATION))
    df["incoterms_seller_pays_import_duties"] = _safe_bool(df["incoterms_code"].isin(SELLER_PAYS_IMPORT_DUTIES))
    df["incoterms_is_domestic_spain"] = _safe_bool(df["incoterms_country_or_region"].eq("SPAIN"))

    return df


def process_incoterms(input_df: pd.DataFrame, output_df: pd.DataFrame, dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalizes free-text incoterms into lower-cardinality model features.

    Input rows contain the sales-order incoterm. Output invoice rows receive the
    same normalized incoterm columns through sales_order_id, so downstream
    historical features can group by incoterms_code instead of the noisy raw text.
    """
    ORDER_ID = "sales_order_id"

    input_mod = add_incoterm_columns(input_df)
    output_mod = output_df.copy()

    incoterm_cols = [
        ORDER_ID,
        "incoterms",
        "incoterms_raw",
        "incoterms_clean",
        "incoterms_code",
        "incoterms_unknown_prefix",
        "incoterms_location_raw",
        "incoterms_location_normalized",
        "incoterms_country_or_region",
        "incoterms_is_standard_code",
        "incoterms_has_location",
        "incoterms_group",
        "incoterms_seller_pays_main_transport",
        "incoterms_seller_delivers_to_destination",
        "incoterms_seller_pays_import_duties",
        "incoterms_is_domestic_spain",
    ]
    incoterm_cols = [c for c in incoterm_cols if c in input_mod.columns]

    if ORDER_ID in input_mod.columns and ORDER_ID in output_mod.columns and len(incoterm_cols) > 1:
        lookup = input_mod[incoterm_cols].drop_duplicates(subset=[ORDER_ID], keep="first").copy()
        output_mod["__incoterms_row_id"] = np.arange(len(output_mod))
        output_mod = output_mod.drop(columns=[c for c in incoterm_cols if c != ORDER_ID and c in output_mod.columns], errors="ignore")
        output_mod = output_mod.merge(lookup, on=ORDER_ID, how="left")
        output_mod = output_mod.sort_values("__incoterms_row_id").drop(columns=["__incoterms_row_id"]).reset_index(drop=True)

    return input_mod, output_mod


run_func_on_outputs(TERMS_OF_PAYMENT_DIR, INCOTERMS_DIR, process_incoterms)
