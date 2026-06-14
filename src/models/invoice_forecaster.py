from pathlib import Path
import subprocess
import sys

from util import log

log("="*20 + " INVOICE FORECASTER ORCHESTRATOR " + "="*20)

SCRIPT_DIR = Path(__file__).resolve().parent


def run_invoice_forecaster() -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / "01_invoice_salesforce_model.py")], check=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "02_invoice_salesforce_sap_model.py")], check=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "03_compare_invoice_forecasters.py")], check=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "04_analyze_invoice_residuals.py")], check=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "05_invoice_date_regime_model.py")], check=True)


if __name__ == "__main__":
    run_invoice_forecaster()
