from config import *
from util import log, create_dir, run_func_on_outputs, expand_date_with_cyclics
import pandas as pd
import numpy as np

log("="*20 + " EXPAND DATES " + "="*20)

def expand_dates(input_df: pd.DataFrame, output_df: pd.DataFrame, dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mod_input = input_df.copy()
    mod_output = output_df.copy()
    expand_date_with_cyclics("invoice_date", mod_output)
    expand_date_with_cyclics("clearing_date", mod_output)
    expand_date_with_cyclics("so_date", mod_input)
    expand_date_with_cyclics("req_deliv_date", mod_input)
    expand_date_with_cyclics("close_date", mod_input)
    expand_date_with_cyclics("created_date", mod_input)

    return mod_input, mod_output

run_func_on_outputs(DATA_CLEANING_OUTPUT_DIR, EXPAND_DATES_DIR, expand_dates)