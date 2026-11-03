from pathlib import Path

import matplotlib

# Backend sem janela, definido antes de qualquer import que use pyplot.
matplotlib.use("Agg")

from src.bank_marketing_analysis import run_analysis  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

if __name__ == "__main__":
    run_analysis(DATA_PATH, FIGURES_DIR)
