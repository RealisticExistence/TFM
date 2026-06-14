from pathlib import Path
import runpy

from util import log

log("="*20 + " CLEARING DATE FORECASTER PIPELINE " + "="*20)

SCRIPT_DIR = Path(__file__).resolve().parent


def run_model_pipeline() -> None:
    scripts = [
        "01_clearing_date_single_models.py",
        "02_analyze_single_model_residuals.py",
        "03_delay_regime_classifier.py",
    ]
    for script in scripts:
        log(f"Running models/{script}")
        runpy.run_path(str(SCRIPT_DIR / script), run_name="__main__")


if __name__ == "__main__":
    run_model_pipeline()
