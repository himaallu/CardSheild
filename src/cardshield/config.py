"""Paths, constants and cost settings shared across CardShield."""

from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
DATA_DIR: Final = ROOT / "data"
RAW_CSV: Final = DATA_DIR / "creditcard.csv"
ARTIFACTS_DIR: Final = ROOT / "artifacts"
MODEL_DIR: Final = ARTIFACTS_DIR / "model"
REPORTS_DIR: Final = ROOT / "reports"

# Dataset: ULB Credit Card Fraud Detection, original CSV with the Time column.
# OpenML id 1597 has identical V1..V28/Amount/Class values but drops Time, which the
# time-based split and the 30-field API need, so we use the TensorFlow-hosted mirror.
DATA_URL: Final = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_SHA256: Final = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
EXPECTED_ROWS: Final = 284_807
EXPECTED_FRAUDS: Final = 492

TIME_COL: Final = "Time"
AMOUNT_COL: Final = "Amount"
LABEL_COL: Final = "Class"
V_COLS: Final[tuple[str, ...]] = tuple(f"V{i}" for i in range(1, 29))
# All 30 input fields, in canonical order. The label is never among them.
INPUT_COLS: Final[tuple[str, ...]] = (TIME_COL, *V_COLS, AMOUNT_COL)
RAW_COLS: Final[tuple[str, ...]] = (*INPUT_COLS, LABEL_COL)

# Time-based split fractions (train / validation / test).
TRAIN_FRAC: Final = 0.6
VAL_FRAC: Final = 0.2

# Cost model: a missed fraud costs its Amount; a false alarm costs REVIEW_COST.
REVIEW_COST: Final = 5.0

MLFLOW_TRACKING_URI: Final = "sqlite:///mlflow.db"
MODEL_NAME: Final = "cardshield"
CHAMPION_ALIAS: Final = "champion"
