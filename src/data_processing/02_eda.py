from config import *
from util import log, create_dir
import pandas as pd
import numpy as np

log("="*20 + " EDA " + "="*20)

sf_report = pd.read_csv(PROCESSED_NO_EMPTY_SF_REPORT)

sf_report.info()

consider_drop_cols = []
for col in sf_report.columns:
    rel_val_count = sf_report[col].value_counts(dropna=False)/len(sf_report.index)
    if np.max(rel_val_count) > 0.9: consider_drop_cols += [col]
    if "winner" in col: consider_drop_cols += [col]

log(f"Consider dropping {len(consider_drop_cols)} columns of {len(sf_report.columns)}: {consider_drop_cols}")

create_dir(PROCESSED_DROP_COLS_LIST_DIR)
pd.DataFrame({"drop_cols": consider_drop_cols}).to_csv(PROCESSED_DROP_COLS_LIST, index=False)