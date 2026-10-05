"""Load, validate schema, time-based split."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from cardshield.config import (
    AMOUNT_COL,
    LABEL_COL,
    RAW_COLS,
    RAW_CSV,
    TIME_COL,
    TRAIN_FRAC,
    V_COLS,
    VAL_FRAC,
)

# Model features: V1..V28 + Amount. Time is excluded because absolute seconds since the
# first transaction never repeat between train and test; Class is the label only.
FEATURE_COLS: Final[tuple[str, ...]] = (*V_COLS, AMOUNT_COL)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    return validate_schema(pd.read_csv(path))


def validate_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Check columns, nulls and label values; return the frame in canonical column order."""
    missing = set(RAW_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = frame.loc[:, list(RAW_COLS)]
    if out.isna().any().any():
        raise ValueError("dataset contains nulls")
    if not set(out[LABEL_COL].unique()) <= {0, 1}:
        raise ValueError(f"{LABEL_COL} must be 0 or 1")
    return out


def _cut_index(times: np.ndarray, frac: float) -> int:
    """Row index at `frac`, moved forward past any rows sharing the boundary timestamp."""
    idx = int(len(times) * frac)
    if idx == 0 or idx >= len(times):
        return idx
    return int(np.searchsorted(times, times[idx - 1], side="right"))


def time_split(
    frame: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC
) -> Split:
    """Split by Time: first ~60% train, next ~20% validation, rest test. No timestamp
    appears in two splits, so max(train.Time) < min(val.Time) < min(test.Time)."""
    ordered = frame.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    times = ordered[TIME_COL].to_numpy()
    a = _cut_index(times, train_frac)
    b = max(a, _cut_index(times, train_frac + val_frac))
    return Split(ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:])


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, list(FEATURE_COLS)]


def data_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
