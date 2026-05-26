import time
from typing import List, Tuple, Optional

import pyautogui
import pyperclip
import ctypes
from pywinauto import Desktop
from pywinauto.keyboard import send_keys
import os
from datetime import datetime
import easyocr
import numpy as np
import csv
import io
import re
import pandas as pd
from io import StringIO
import threading
from pynput import keyboard

_OCR_READER = None
_STOP_REQUESTED = threading.Event()

LOG_FILE = None
CONFIDENCE = 0.8
POLL_INTERVAL = 0.25
INFO = 0
WARN = 1
ERROR = 2
alert_levels = ["INFO", "WARN", "ERROR"]

doc_types = {
    "final invoice": "final invoice",
    "partial invoice": "partial invoice",
    "cancel. invoice": "cancelled invoice",
    "cancel. partial inv.": "cancelled partial invoice",
    "purchase order": "purchase order",
    "down payment invoice": "down payment invoice",
    "down p. inv. rect": "down payment invoice recification",
    "cancel.downpmt inv": "cancelled downpayment invoice"
    }

po_doc_types = {
    "invoice": "invoice",
    "credit memo": "credit memo",
    "subsequent debit": "subsequent debit",
    "subsequent credit": "subsequent credit"
    }
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def get_image_positions(image_path: str) -> List[Tuple[int, int]]:
    """
    Find all instances of an image on screen.
    Returns a list of center positions: [(x, y), ...]
    """
    positions = []

    try:
        matches = pyautogui.locateAllOnScreen(
            image_path,
            confidence=CONFIDENCE
        )

        for match in matches:
            center = pyautogui.center(match)
            positions.append((center.x, center.y))

    except:
        pass

    return positions


def wait_for_image(image_path: str, timeout: float = 5) -> bool:
    """
    Wait until image appears on screen.
    Returns True if found, False if timeout expires.
    """
    start = time.time()

    while time.time() - start < timeout:
        if get_image_positions(image_path):
            return True

        time.sleep(POLL_INTERVAL)

    return False



def right_click_image(image_path: str, offset_x: int = 0, offset_y: int = 0) -> bool:
    """
    Right click at an offset from the center of the first matching image.
    Returns True if clicked, False if image was not found.
    """
    positions = get_image_positions(image_path)

    if not positions:
        return False

    x, y = positions[0]
    pyautogui.rightClick(x + offset_x, y + offset_y)

    return True

def click_image(image_path: str, offset_x: int = 0, offset_y: int = 0) -> bool:
    """
    Click at an offset from the center of the first matching image.
    Returns True if clicked, False if image was not found.
    """
    positions = get_image_positions(image_path)

    if not positions:
        return False

    x, y = positions[0]
    pyautogui.click(x + offset_x, y + offset_y)

    return True

def click_image_n(n_clicks: int, image_path: str, offset_x: int = 0, offset_y: int = 0) -> bool:
    """
    Click n times at an offset from the center of the first matching image.
    Returns True if clicked, False if image was not found.
    """
    for i in range(n_clicks):
        if not click_image(image_path, offset_x, offset_y): return False
        time.sleep(0.2)
    return True

def type_text(text: str) -> None:
    """
    Paste text using the clipboard.
    More reliable than typing character-by-character.
    """
    pyperclip.copy(str(text))
    pyautogui.hotkey("ctrl", "v")


def type_enter(text: str) -> None:
    """
    Paste text, then press Enter.
    """
    type_text(text)
    pyautogui.press("enter")

def get_ocr_reader():
    """
    Lazy-load EasyOCR once.
    First run may take a little while.
    """
    global _OCR_READER

    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader(["en"], gpu=False)

    return _OCR_READER


def read_text(
    topleft_x: int,
    topleft_y: int,
    width: int,
    height: int
) -> str:
    """
    OCR text from a screen rectangle using EasyOCR.
    Does not require Tesseract.
    """
    screenshot = pyautogui.screenshot(
        region=(int(topleft_x), int(topleft_y), int(width), int(height))
    )

    image = np.array(screenshot)

    reader = get_ocr_reader()
    results = reader.readtext(image, detail=0)

    return " ".join(results).strip()

def read_text_image(
    image_path: str,
    offset_x: int,
    offset_y: int,
    width: int,
    height: int
) -> str:
    """
    Find an image, then OCR a rectangle positioned at an offset
    from that image's center.

    The rectangle's top-left corner is:
    image_center_x + offset_x, image_center_y + offset_y
    """
    positions = get_image_positions(image_path)

    if not positions:
        return ""

    center_x, center_y = positions[0]

    topleft_x = center_x + offset_x
    topleft_y = center_y + offset_y

    return read_text(topleft_x, topleft_y, width, height)


def run_transaction(text: str) -> bool:
    """
    Run a transaction from the SAP Business Client command/search bar.

    Example:
        run_transaction("VA03")
        run_transaction("/nVA03")
        run_transaction("/oVA03")

    Returns True if it could find and use the SAP command bar.
    """

    # Get the currently focused SAP Business Client window
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    desktop = Desktop(backend="uia")
    sap_spec = desktop.window(handle=hwnd)

    # This is the SAP Business Client command/search box from your dumps
    speed_text = sap_spec.child_window(
        auto_id="_speedText",
        control_type="Edit"
    )

    if not speed_text.exists(timeout=2):
        return False

    speed_text.set_focus()
    time.sleep(0.2)

    # Normalize transaction code
    transaction = str(text).strip()

    if not transaction.startswith("/"):
        transaction = "/n" + transaction

    # Clear command bar and run transaction
    send_keys("^a")
    type_enter(transaction)

    return True


def create_log(logs_dir: str = "logs") -> str:
    """
    Create a new log file for this Python process.
    Returns the log file path.
    """
    global LOG_FILE

    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = os.path.join(logs_dir, f"run_{timestamp}.log")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] [INFO] Log created\n")

    return LOG_FILE


def log(message: str, alert_level: int = INFO) -> None:
    """
    Append a timestamped message to the current log file.
    Call create_log() once at the start of the process.
    """
    global LOG_FILE

    if LOG_FILE is None:
        create_log()

    timestamp = datetime.now().isoformat(timespec="seconds")

    level = alert_levels[alert_level]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")

def double_click_image(image_path: str, offset_x: int = 0, offset_y: int = 0) -> bool:
    """
    Double-click at an offset from the center of the first matching image.

    Returns True if double-clicked, False if image was not found.
    """
    positions = get_image_positions(image_path)

    if not positions:
        return False

    x, y = positions[0]
    pyautogui.doubleClick(x + offset_x, y + offset_y)

    return True

def wait_and_click_image(image_path: str, timeout: float = 5, offset_x: int = 0, offset_y: int = 0) -> bool:
    """
    Wait until image appears on screen, click if it does and return true. Otherwise return false.
    """
    if not wait_for_image(image_path, timeout): return False
    click_image(image_path, offset_x, offset_y)
    return True

def save_to_file(text: str, path: str) -> None:
    """
    Save text to a file, creating parent folders if needed.
    Overwrites the file if it already exists.
    """
    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def create_folder(dir: str) -> None: os.makedirs(dir, exist_ok=True)


def normalize_po_doc_type(doc_type: str) -> str:
    """
    Converts raw PO doc type to a normalized version
    """
    for doc_type_it in po_doc_types.keys():
        if doc_type_it in doc_type.lower(): return po_doc_types[doc_type_it].replace(" ", "_")
    
    return "unknown"

def normalize_doc_type(doc_type: str) -> str:
    """
    Converts raw doc type to a normalized version
    """
    for doc_type_it in doc_types.keys():
        if doc_type_it in doc_type.lower(): return doc_types[doc_type_it].replace(" ", "_")
    
    return "unknown"

def to_snake_case(name: str) -> str:
    name = name.strip().lower()
    name = name.replace(".", " ")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def make_unique_columns(columns: List[str]) -> List[str]:
    seen = {}
    result = []

    for col in columns:
        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")

    return result


def parse_pipe_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip("|").strip().split("|")]


def raw_to_csv(text: str, doc_type: str) -> str:
    """
    Convert SAP pipe-table text into a CSV string.

    Adds:
        doc_type

    Example:
        csv_text = raw_to_csv(raw_text, "final_invoice")
    """
    lines = text.splitlines()

    header = None
    rows = []

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        cells = parse_pipe_row(stripped)

        # Detect header row
        if "Doc.no." in cells:
            header = cells
            continue

        # Skip invalid rows before header
        if header is None:
            continue

        # Skip separator-like or malformed rows
        if len(cells) != len(header):
            continue

        rows.append(cells)

    if header is None:
        raise ValueError("Could not find table header row.")

    columns = [to_snake_case(col) for col in header]
    columns = make_unique_columns(columns)

    # Add metadata columns
    columns = ["doc_type"] + columns

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    writer.writerow(columns)

    for row in rows:
        writer.writerow([doc_type] + row)

    return output.getvalue()

def export_to_clipboard(return_image: str):
    if not wait_and_click_image("control_images/export_icon.png"): return False

    if not wait_and_click_image("control_images/local_file.png"): 
        if not wait_and_click_image("control_images/export_icon.png"): return False # Try again
        if not wait_and_click_image("control_images/local_file.png"): return False

    if not wait_and_click_image("control_images/in_clipboard.png"): return False
    pyautogui.press("enter")

    if not wait_for_image(return_image): return False

    return True

def classify_receipt_type(row: List[str]) -> Optional[str]:
    """
    Classify a PO history row as Goods receipt or Invoice receipt.

    Uses the first column, usually Sh. Text:
        GR   -> Goods
        IR-* -> Invoice
    """
    if not row:
        return None

    sh_text = row[0].strip().upper()

    if sh_text.startswith("GR"):
        return "Goods"

    if sh_text.startswith("IR"):
        return "Invoice"

    return None


def raw_po_history_to_csv(text: str):
    """
    Convert SAP PO History pipe-table text into a CSV string.

    Adds:
        receipt_type

    Example:
        csv_text, n_invoices = raw_po_history_to_csv(raw_text)
    """
    lines = text.splitlines()

    header = None
    rows = []
    num_invoices = 0

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        cells = parse_pipe_row(stripped)

        # Detect header row
        if "Material Document" in cells and "Trans./event type" in cells:
            header = cells
            continue

        if header is None:
            continue

        # Skip subtotal rows like:
        # |*Tr./Ev. Goods receipt ...
        # |*Tr./Ev. Invoice receipt ...
        if cells and cells[0].startswith("*Tr./Ev."):
            continue

        # Skip malformed rows
        if len(cells) != len(header):
            continue

        receipt_type = classify_receipt_type(cells)

        # Skip rows that are neither Goods nor Invoice receipts
        if receipt_type is None:
            continue

        if receipt_type == "Invoice": 
            num_invoices += 1
            rows.append(cells)

    if header is None:
        raise ValueError("Could not find PO History table header row.")

    columns = [to_snake_case(col) for col in header]
    columns = make_unique_columns(columns)

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    writer.writerow(columns)
    writer.writerows(rows)

    return output.getvalue(), num_invoices


def csv_str_to_df(text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(text))

def extract_clearing_date(text: str) -> Optional[str]:
    """
    Extract the first clearing date from an SAP pipe-table text.

    Returns:
        Clearing date as a string, e.g. "21.12.2021"
        None if no clearing date is found.
    """
    lines = text.splitlines()
    header = None
    clearing_index = None

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]

        # Find header row
        if "Clearing" in cells:
            header = cells
            clearing_index = header.index("Clearing")
            continue

        if header is None or clearing_index is None:
            continue

        if len(cells) <= clearing_index:
            continue

        clearing_value = cells[clearing_index].strip()

        # Skip subtotal rows like "* 3000055161" or "**"
        first_cell = cells[0].strip()
        if first_cell.startswith("*") or first_cell.startswith("**"):
            continue

        # Validate date format DD.MM.YYYY
        if re.match(r"^\d{2}\.\d{2}\.\d{4}$", clearing_value):
            return clearing_value

    return None
def get_clearing_date_by_profit_center(text: str, profit_center: str) -> Optional[str]:
    """
    Extract the first clearing date for a given Profit Ctr from SAP pipe-table text.

    Returns:
        Clearing date as string, e.g. "07.05.2025"
        None if no match is found.
    """
    header = None
    profit_center_index = None
    clearing_index = None

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]

        # Find header row
        if "Profit Ctr" in cells and "Clearing" in cells:
            header = cells
            profit_center_index = header.index("Profit Ctr")
            clearing_index = header.index("Clearing")
            continue

        if header is None:
            continue

        if len(cells) <= max(profit_center_index, clearing_index):
            continue

        # Skip subtotal rows
        if any(cell.startswith("*") for cell in cells[:2]):
            continue

        row_profit_center = cells[profit_center_index].strip()
        clearing_date = cells[clearing_index].strip()

        if row_profit_center == "":
            continue

        if re.match(r"^\d{2}\.\d{2}\.\d{4}$", clearing_date):
            return clearing_date

    return None

def start_esc_listener() -> threading.Event:
    """
    Starts a background keyboard listener.

    Press ESC to request the script to stop.

    Returns:
        threading.Event you can check with .is_set()
    """

    def on_press(key):
        if key == keyboard.Key.space:
            print("Space pressed. Requesting stop...")
            _STOP_REQUESTED.set()
            return False  # stops the listener

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    return _STOP_REQUESTED


def stop_requested() -> bool:
    """
    Returns True if ESC has been pressed.
    """
    return _STOP_REQUESTED.is_set()

def stop_transaction(tab_image: str) -> bool: 
    if right_click_image(tab_image):
        if click_image("control_images/stop_transaction.png"): return True
    return False

def window_exists(hwnd: int) -> bool:
    """
    Returns True if the window handle still exists.
    """
    return bool(ctypes.windll.user32.IsWindow(hwnd))


def window_is_visible(hwnd: int) -> bool:
    """
    Returns True if the window exists and is visible.
    """
    return bool(ctypes.windll.user32.IsWindowVisible(hwnd))


def get_focused_window_handle() -> int:
    """
    Returns the handle of the currently focused / foreground window.
    """
    return ctypes.windll.user32.GetForegroundWindow()


def is_window_focused(hwnd: int) -> bool:
    """
    Returns True if the given window is currently focused.
    """
    return get_focused_window_handle() == hwnd


def is_sap_window_alive_and_focused(hwnd: int) -> bool:
    """
    Returns True if the SAP window still exists, is visible, and is focused.
    """
    return (
        window_exists(hwnd)
        and window_is_visible(hwnd)
        and is_window_focused(hwnd)
    )