from config import *
from util import log, create_dir, run_func_on_outputs
import pandas as pd
import numpy as np

log("="*20 + " HISTORICAL AGGREGATES " + "="*20)

def process_hist_aggregates(input_df: pd.DataFrame, output_df: pd.DataFrame, dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    ORDER_ID = "sales_order_id"

    GROUPING_SETS = {
        "client": ["client_number"],
        "incoterms": ["incoterms"],
        "terms_of_payment": ["terms_of_payment"],
        #"client__incoterms": ["client_number", "incoterms"],
        #"client__terms_of_payment": ["client_number", "terms_of_payment"],
        #"incoterms__terms_of_payment": ["incoterms", "terms_of_payment"],
        #"client__incoterms__terms_of_payment": [
        #    "client_number",
        #    "incoterms",
        #    "terms_of_payment",
        #],
    }

    INPUT_DATE_COLS = [
        "so_date",
        "created_date",
        "req_deliv_date",
        "close_date",
        "bid_submission_date",
        "bid_validity",
    ]

    OUTPUT_DATE_COLS = [
        "invoice_date",
        "clearing_date",
    ]

    DATE_PAIRS = [
        ("so_date", "invoice_date"),
        ("so_date", "clearing_date"),
        ("created_date", "invoice_date"),
        ("created_date", "clearing_date"),
        ("invoice_date", "clearing_date"),
        #("req_deliv_date", "invoice_date"),
        #("req_deliv_date", "clearing_date"),
        #("close_date", "invoice_date"),
        #("close_date", "clearing_date"),
        #("bid_submission_date", "invoice_date"),
        #("bid_submission_date", "clearing_date"),
        #("bid_validity", "invoice_date"),
        #("bid_validity", "clearing_date"),
    ]

    def parse_mixed_date(s: pd.Series) -> pd.Series:
        s = s.astype("string").str.strip()
        s = s.mask(s.isin(["", "nan", "NaN", "NaT", "None"]))

        try:
            return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)
        except TypeError:
            return pd.to_datetime(s, errors="coerce", dayfirst=True)

    def rebuild_date(df: pd.DataFrame, base_col: str) -> pd.Series:
        day_col = f"{base_col}_day"
        month_col = f"{base_col}_month"
        year_col = f"{base_col}_year"

        out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

        if {day_col, month_col, year_col}.issubset(df.columns):
            parts = pd.DataFrame({
                "year": pd.to_numeric(df[year_col], errors="coerce"),
                "month": pd.to_numeric(df[month_col], errors="coerce"),
                "day": pd.to_numeric(df[day_col], errors="coerce"),
            })

            out = pd.to_datetime(parts, errors="coerce")

        if base_col in df.columns:
            out = out.fillna(parse_mixed_date(df[base_col]))

        return out

    def add_clean_dates(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
        df = df.copy()

        for col in date_cols:
            has_raw_col = col in df.columns
            has_split_cols = {
                f"{col}_day",
                f"{col}_month",
                f"{col}_year",
            }.issubset(df.columns)

            if has_raw_col or has_split_cols:
                df[col] = rebuild_date(df, col)

        return df

    def add_date_deltas(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for start, end in DATE_PAIRS:
            if start in df.columns and end in df.columns:
                df[f"days__{start}__to__{end}"] = (df[end] - df[start]).dt.days

        return df

    def winsorized_mean(x: pd.Series, lower: float = 0.05, upper: float = 0.95) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()

        if len(x) == 0:
            return np.nan

        lo, hi = x.quantile([lower, upper])
        return x.clip(lo, hi).mean()

    def make_agg_spec(df: pd.DataFrame, delta_cols: list[str], amount_cols: list[str]) -> dict:
        agg = {
            "n_rows": (ORDER_ID, "size"),
            "n_sales_orders": (ORDER_ID, "nunique"),
        }

        if "invoice_num" in df.columns:
            agg["n_invoices"] = ("invoice_num", "nunique")

        for col in delta_cols:
            agg.update({
                f"{col}__count": (col, "count"),
                f"{col}__mean": (col, "mean"),
                #f"{col}__median": (col, "median"),
                f"{col}__std": (col, "std"),
                #f"{col}__min": (col, "min"),
                #f"{col}__p25": (col, lambda s: s.quantile(0.25)),
                #f"{col}__p75": (col, lambda s: s.quantile(0.75)),
                #f"{col}__p90": (col, lambda s: s.quantile(0.90)),
                #f"{col}__max": (col, "max"),
                #f"{col}__winsor_mean_5_95": (col, winsorized_mean),
                #f"{col}__missing_pct": (col, lambda s: s.isna().mean()),
            })

        for col in amount_cols:
            agg.update({
                f"{col}__count": (col, "count"),
                #f"{col}__sum": (col, "sum"),
                f"{col}__mean": (col, "mean"),
                #f"{col}__median": (col, "median"),
                f"{col}__std": (col, "std"),
                #f"{col}__min": (col, "min"),
                #f"{col}__p25": (col, lambda s: s.quantile(0.25)),
                #f"{col}__p75": (col, lambda s: s.quantile(0.75)),
                #f"{col}__p90": (col, lambda s: s.quantile(0.90)),
                #f"{col}__max": (col, "max"),
            })

        return agg

    def build_group_aggregates(df: pd.DataFrame, group_cols: list[str], grouping_name: str) -> pd.DataFrame:
        if not set(group_cols).issubset(df.columns):
            return pd.DataFrame()

        delta_cols = [c for c in df.columns if c.startswith("days__")]

        amount_cols = [
            c for c in [
                "hist_invoice_amount",
                "hist_abs_invoice_amount",
                "hist_sales_order_amount",
                "hist_sales_order_net_value",
                "hist_sales_order_amount_converted",
                "hist_sales_order_expected_revenue",
            ]
            if c in df.columns
        ]

        agg_spec = make_agg_spec(df, delta_cols, amount_cols)

        result = (
            df.groupby(group_cols, dropna=False)
              .agg(**agg_spec)
              .reset_index()
        )

        feature_cols = [c for c in result.columns if c not in group_cols]
        result = result.rename(
            columns={c: f"hist_{grouping_name}__{c}" for c in feature_cols}
        )

        return result

    def merge_features(base_df: pd.DataFrame, features_df: pd.DataFrame, on: list[str]) -> pd.DataFrame:
        if features_df.empty:
            return base_df

        if not set(on).issubset(base_df.columns):
            return base_df

        feature_cols = [c for c in features_df.columns if c not in on]

        base_df = base_df.drop(
            columns=[c for c in feature_cols if c in base_df.columns],
            errors="ignore",
        )

        return base_df.merge(features_df, on=on, how="left")

    input_mod = input_df.copy()
    output_mod = output_df.copy()

    input_mod = add_clean_dates(input_mod, INPUT_DATE_COLS)
    output_mod = add_clean_dates(output_mod, OUTPUT_DATE_COLS)

    if "amount" in input_mod.columns:
        input_mod["hist_sales_order_amount"] = pd.to_numeric(
            input_mod["amount"],
            errors="coerce",
        )

    if "net_value" in input_mod.columns:
        input_mod["hist_sales_order_net_value"] = pd.to_numeric(
            input_mod["net_value"],
            errors="coerce",
        )

    if "amount_converted" in input_mod.columns:
        input_mod["hist_sales_order_amount_converted"] = pd.to_numeric(
            input_mod["amount_converted"],
            errors="coerce",
        )

    if "expected_revenue" in input_mod.columns:
        input_mod["hist_sales_order_expected_revenue"] = pd.to_numeric(
            input_mod["expected_revenue"],
            errors="coerce",
        )

    if "amount" in output_mod.columns:
        output_mod["hist_invoice_amount"] = pd.to_numeric(
            output_mod["amount"],
            errors="coerce",
        )
        output_mod["hist_abs_invoice_amount"] = output_mod["hist_invoice_amount"].abs()
        output_mod["hist_is_credit_or_negative_invoice"] = (
            output_mod["hist_invoice_amount"] < 0
        )

    if "clearing_date" in output_mod.columns:
        output_mod["hist_has_clearing_date"] = output_mod["clearing_date"].notna()

    input_meta_cols = [
        ORDER_ID,
        "client_number",
        "account_name",
        "incoterms",
        "terms_of_payment",
        *INPUT_DATE_COLS,
        "hist_sales_order_amount",
        "hist_sales_order_net_value",
        "hist_sales_order_amount_converted",
        "hist_sales_order_expected_revenue",
    ]

    input_meta_cols = [c for c in input_meta_cols if c in input_mod.columns]

    input_meta = (
        input_mod[input_meta_cols]
        .drop_duplicates(subset=[ORDER_ID], keep="first")
        .copy()
    )

    output_mod["__hist_row_id"] = np.arange(len(output_mod))

    hist_base = output_mod.merge(
        input_meta,
        on=ORDER_ID,
        how="left",
        suffixes=("", "__from_input"),
    )

    # If output already had some metadata columns, fill missing values from input
    for col in input_meta_cols:
        if col == ORDER_ID:
            continue

        from_input_col = f"{col}__from_input"

        if from_input_col in hist_base.columns:
            if col in hist_base.columns:
                hist_base[col] = hist_base[col].combine_first(hist_base[from_input_col])
                hist_base = hist_base.drop(columns=[from_input_col])
            else:
                hist_base = hist_base.rename(columns={from_input_col: col})

    hist_base = add_date_deltas(hist_base)
    hist_base = hist_base.sort_values("__hist_row_id").reset_index(drop=True)

    columns_to_copy_to_output = [
        "client_number",
        "account_name",
        "incoterms",
        "terms_of_payment",
        *INPUT_DATE_COLS,
        "hist_sales_order_amount",
        "hist_sales_order_net_value",
        "hist_sales_order_amount_converted",
        "hist_sales_order_expected_revenue",
    ]

    delta_cols = [c for c in hist_base.columns if c.startswith("days__")]

    columns_to_copy_to_output.extend(delta_cols)

    for col in columns_to_copy_to_output:
        if col in hist_base.columns:
            output_mod[col] = hist_base[col].to_numpy()

    for grouping_name, group_cols in GROUPING_SETS.items():
        group_agg = build_group_aggregates(
            hist_base,
            group_cols=group_cols,
            grouping_name=grouping_name,
        )

        input_mod = merge_features(input_mod, group_agg, on=group_cols)
        output_mod = merge_features(output_mod, group_agg, on=group_cols)

    # -----------------------------
    # Restore output row order and remove helper column
    # -----------------------------
    output_mod = (
        output_mod.sort_values("__hist_row_id")
        .drop(columns=["__hist_row_id"], errors="ignore")
        .reset_index(drop=True)
    )

    input_mod = input_mod.reset_index(drop=True)

    return input_mod, output_mod

run_func_on_outputs(EXPAND_DATES_DIR, HISTORICAL_AGGREGATES_DIR, process_hist_aggregates)