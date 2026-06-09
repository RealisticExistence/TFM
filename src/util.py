import math
import re
import time
from collections.abc import Callable
from typing import Dict

import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import __main__
from enum import Enum
from config import LOG_DIR, DATA_CLEANING_OUTPUT_COMPLETE_FOLDER, DATA_CLEANING_OUTPUT_CLEARING_FOLDER, \
    DATA_CLEANING_OUTPUT_FULL_FOLDER
from termcolor import colored

class LogLevel(Enum):
    INFO = 1
    WARN = 2
    ERROR = 3

log_file = None
log_suppress = 0 # 1 doesnt print infos, 2 only prints errors, 3 prints nothing

def err(msg: str) -> None: log(msg, LogLevel.ERROR)
def warn(msg: str) -> None: log(msg, LogLevel.WARN)

def log(msg: str, level: LogLevel = LogLevel.INFO) -> None:
    global log_file
    if log_file is None:
        script_name = Path(__main__.__file__).name
        log_path = LOG_DIR / script_name
        create_dir(log_path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file_path = log_path / f"{timestamp}.log"
        log_file = create_file(log_file_path)

    timestamp = time.strftime("%Y/%m/%d-%H:%M:%S")
    level_str = ["INFO", "WARN", "ERROR"][level.value-1]
    formatted_msg = f"[{timestamp}] [{level_str}] {msg}"
    log_file.write(formatted_msg+"\n")
    if level.value > log_suppress: print(colored(formatted_msg, ["blue", "yellow", "red"][level.value-1]))

def read_csv(path: Path) -> pd.DataFrame:
    """
    Reads csv file
    :param path:
    :return:
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Writes df to path
    :param df:
    :param path:
    :return:
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def dir_has_files(path: Path) -> bool:
    """
    Checks whether a directory has any files. Returns false if the dir doesnt exist
    :param path:
    :return:
    """
    if not path.exists(): return False
    result = False
    for elm in path.iterdir():
        if elm.is_dir():
            result = result or dir_has_files(elm)
        if elm.is_file():
            result = True
    return result

def sales_order_is_empty(path: Path) -> bool:
    purchase_order_path = path / "purchase_orders"
    sales_invoice_path = path / "sales_invoices"
    return (
            not dir_has_files(purchase_order_path) and
            not dir_has_files(sales_invoice_path)
    )

def create_dir(path: Path) -> None: path.mkdir(parents=True, exist_ok=True)
def create_file(path: Path): return open(path.resolve(), "x")
def copy_dir(path_from: Path, path_to: Path) -> None:
    """
    Copies the directory path_from to path_to (e.g. dir/1/ to dir2/ gives dir2/1/)
    :param path_from:
    :param path_to:
    :return:
    """
    if not path_from.is_dir():
        raise FileNotFoundError(f"Directory not found: {path_from}")

    target_dir = path_to / path_from.name

    create_dir(target_dir)
    shutil.copytree(path_from.resolve(), target_dir, dirs_exist_ok=True)

def copy_file(path_from: Path, path_to: Path): shutil.copyfile(path_from.resolve(), path_to.resolve())

def count_dirs(path: Path) -> int:
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return sum(1 for p in path.iterdir() if p.is_dir())

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame: return df.loc[:, (df != df.iloc[0]).any()]

def apply_csv_files(path: Path, function: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    for csv_path in path.rglob("*.csv"):
        function(pd.read_csv(csv_path)).to_csv(csv_path, index=False)

def parse_date(string: str) -> str:
    string = str(string)
    res = re.search(r"\d{2}\.\d{2}\.\d{4}", string)  # search pattern

    if res:
        return res.group()
    else:
        numbers = "".join(re.findall(r"\d+", string))
        if len(numbers) < 6: return ""
        return numbers[:2]+"."+numbers[2:4]+"."+numbers[4:]

def parse_monetary_amount(string: str) -> float:
    sign = -1 if '-' in string else 1
    abs_amount = float(string.replace("-", "").replace(",", ""))
    return sign * abs_amount

def file_not_empty(path: Path) -> bool:
    if not path.exists(): return False
    return len(path.read_text(encoding="utf-8").strip()) > 0

def expand_date_with_cyclics(column: str, df: pd.DataFrame, day_dict=None, sep=".") -> None:
    if day_dict is None: day_dict = [1, 2, 3]
    def day_extract(date_string):
        match = re.search(r"(\d{1,2})"+sep+r"(\d{1,2})"+sep+r"(\d{2,4})", str(date_string))
        if match is None: return None
        return int(match.group(day_dict[0]))
    def month_extract(date_string):
        match = re.search(r"(\d{1,2})"+sep+r"(\d{1,2})"+sep+r"(\d{2,4})", str(date_string))
        if match is None: return None
        return int(match.group(day_dict[1]))
    def year_extract(date_string):
        match = re.search(r"(\d{1,2})"+sep+r"(\d{1,2})"+sep+r"(\d{2,4})", str(date_string))
        if match is None: return None
        return int(match.group(day_dict[2]))

    df[f"{column}_day"] = pd.to_numeric(df[column].apply(day_extract), errors="coerce")
    df[f"{column}_month"] = pd.to_numeric(df[column].apply(month_extract))
    df[f"{column}_year"] = pd.to_numeric(df[column].apply(year_extract))

    df[f"{column}_day_sin"] = np.sin(df[f"{column}_day"]/31*2*math.pi)
    df[f"{column}_day_cos"] = np.cos(df[f"{column}_day"]/31*2*math.pi)
    df[f"{column}_month_sin"] = np.sin(df[f"{column}_month"]/13*2*math.pi)
    df[f"{column}_month_cos"] = np.cos(df[f"{column}_month"]/13*2*math.pi)

def parse_net_value(string: str) -> float:
    index_EUR = string.find("EUR")
    currencyless_val = string[:index_EUR]
    clean_currencyless = currencyless_val.replace("_", ".").strip()
    if not "." in clean_currencyless: clean_currencyless = clean_currencyless[:-3] + "." + clean_currencyless[-2:]
    return clean_currencyless

def run_func_on_outputs(parent_dir: Path, output_dir: Path, func: Callable[[pd.DataFrame, pd.DataFrame, str], tuple[pd.DataFrame, pd.DataFrame]]):
    complete_input_clearing = pd.read_csv(parent_dir / DATA_CLEANING_OUTPUT_COMPLETE_FOLDER / "complete_input_clearing.csv")
    complete_output_clearing = pd.read_csv(parent_dir / DATA_CLEANING_OUTPUT_COMPLETE_FOLDER / "complete_output_clearing.csv")
    input_clearing = pd.read_csv(parent_dir / DATA_CLEANING_OUTPUT_CLEARING_FOLDER / "input_clearing.csv")
    output_clearing = pd.read_csv(parent_dir / DATA_CLEANING_OUTPUT_CLEARING_FOLDER / "output_clearing.csv")
    input = pd.read_csv(parent_dir / DATA_CLEANING_OUTPUT_FULL_FOLDER / "input.csv")
    output = pd.read_csv(parent_dir / DATA_CLEANING_OUTPUT_FULL_FOLDER / "output.csv")

    complete_input_clearing, complete_output_clearing = func(complete_input_clearing, complete_output_clearing, "complete")
    input_clearing, output_clearing = func(input_clearing, output_clearing, "clearing")
    input, output = func(input, output, "full")

    create_dir(output_dir / DATA_CLEANING_OUTPUT_COMPLETE_FOLDER)
    create_dir(output_dir / DATA_CLEANING_OUTPUT_CLEARING_FOLDER)
    create_dir(output_dir / DATA_CLEANING_OUTPUT_FULL_FOLDER)

    complete_input_clearing.to_csv(output_dir / DATA_CLEANING_OUTPUT_COMPLETE_FOLDER / "complete_input_clearing.csv", index=False)
    complete_output_clearing.to_csv(output_dir / DATA_CLEANING_OUTPUT_COMPLETE_FOLDER / "complete_output_clearing.csv", index=False)
    input_clearing.to_csv(output_dir / DATA_CLEANING_OUTPUT_CLEARING_FOLDER / "input_clearing.csv", index=False)
    output_clearing.to_csv(output_dir / DATA_CLEANING_OUTPUT_CLEARING_FOLDER / "output_clearing.csv", index=False)
    input.to_csv(output_dir / DATA_CLEANING_OUTPUT_FULL_FOLDER / "input.csv", index=False)
    output.to_csv(output_dir / DATA_CLEANING_OUTPUT_FULL_FOLDER / "output.csv", index=False)