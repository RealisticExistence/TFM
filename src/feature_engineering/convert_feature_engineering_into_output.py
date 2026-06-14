from pathlib import Path
import shutil

from config import *
from util import create_dir, log

log("="*20 + " CONVERT FEATURE ENGINEERING INTO OUTPUT " + "="*20)

DATASET_FOLDERS = [
    DATA_CLEANING_OUTPUT_COMPLETE_FOLDER,
    DATA_CLEANING_OUTPUT_CLEARING_FOLDER,
    DATA_CLEANING_OUTPUT_FULL_FOLDER,
]


def is_feature_step_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) >= 2 and path.name[:2].isdigit()


def find_latest_feature_step() -> Path:
    if not FEATURE_ENGINEERING_DIR.exists():
        raise FileNotFoundError(f"Feature-engineering directory not found: {FEATURE_ENGINEERING_DIR}")

    step_dirs = sorted([p for p in FEATURE_ENGINEERING_DIR.iterdir() if is_feature_step_dir(p)], key=lambda p: p.name)
    if not step_dirs:
        raise FileNotFoundError(f"No numbered feature-engineering steps found under: {FEATURE_ENGINEERING_DIR}")
    return step_dirs[-1]


def copy_feature_engineering_output() -> None:
    latest_step_dir = find_latest_feature_step()
    log(f"Latest feature-engineering step: {latest_step_dir.resolve()}")

    if FEATURE_ENGINEERING_OUTPUT_DIR.exists():
        shutil.rmtree(FEATURE_ENGINEERING_OUTPUT_DIR)
    create_dir(FEATURE_ENGINEERING_OUTPUT_DIR)

    copied_files = 0
    for dataset_folder in DATASET_FOLDERS:
        source_dir = latest_step_dir / dataset_folder
        target_dir = FEATURE_ENGINEERING_OUTPUT_DIR / dataset_folder
        if not source_dir.exists():
            log(f"Skipping missing dataset folder: {source_dir}")
            continue
        shutil.copytree(source_dir, target_dir)
        copied_files += sum(1 for p in target_dir.rglob("*.csv") if p.is_file())

    log(f"Feature-engineering output written to: {FEATURE_ENGINEERING_OUTPUT_DIR.resolve()}")
    log(f"Copied CSV files: {copied_files}")


copy_feature_engineering_output()
