"""Download the ULB credit card fraud dataset from OpenML and verify it.

Saves data/creditcard.csv with columns Time, V1..V28, Amount, Class.
Exits non-zero if the row or fraud counts do not match the published dataset.
"""

import sys

import pandas as pd

from cardshield.config import (
    DATA_DIR,
    EXPECTED_FRAUDS,
    EXPECTED_ROWS,
    LABEL_COL,
    OPENML_DATA_ID,
    RAW_COLS,
    RAW_CSV,
)


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical column order; label cast to int (OpenML returns it as a category)."""
    missing = set(RAW_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = frame.loc[:, list(RAW_COLS)].copy()
    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL].astype(str)).astype("int64")
    return out


def verify(frame: pd.DataFrame) -> tuple[int, int]:
    """Return (rows, frauds); raise if they differ from the published counts."""
    rows, frauds = len(frame), int(frame[LABEL_COL].sum())
    if (rows, frauds) != (EXPECTED_ROWS, EXPECTED_FRAUDS):
        raise ValueError(
            f"dataset mismatch: got {rows:,} rows / {frauds} frauds, "
            f"expected {EXPECTED_ROWS:,} / {EXPECTED_FRAUDS}"
        )
    return rows, frauds


def fetch() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    frame: pd.DataFrame = bunch.frame
    return frame


def main() -> int:
    if RAW_CSV.exists():
        print(f"Found {RAW_CSV}, verifying...")
        frame = pd.read_csv(RAW_CSV)
    else:
        print(f"Downloading OpenML dataset {OPENML_DATA_ID}...")
        frame = normalise(fetch())
    try:
        rows, frauds = verify(frame)
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    if not RAW_CSV.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(RAW_CSV, index=False)
    print(f"OK: {rows:,} rows, {frauds} frauds ({frauds / rows:.3%}) -> {RAW_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
