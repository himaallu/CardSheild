"""Download the ULB credit card fraud dataset and verify it.

Saves data/creditcard.csv with columns Time, V1..V28, Amount, Class.
Exits non-zero if the checksum, columns, time order or row/fraud counts are wrong.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from cardshield.config import (
    DATA_SHA256,
    DATA_URL,
    EXPECTED_FRAUDS,
    EXPECTED_ROWS,
    LABEL_COL,
    RAW_COLS,
    RAW_CSV,
    TIME_COL,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical column order; label cast to int (some mirrors store it as a category)."""
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
    if TIME_COL in frame and not frame[TIME_COL].is_monotonic_increasing:
        raise ValueError(f"{TIME_COL} is not sorted ascending")
    return rows, frauds


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, tmp)
    actual = sha256(tmp)
    if actual != DATA_SHA256:
        tmp.unlink()
        raise ValueError(f"checksum mismatch: got {actual}, expected {DATA_SHA256}")
    tmp.replace(dest)


def main() -> int:
    try:
        if RAW_CSV.exists():
            print(f"Found {RAW_CSV}, verifying...")
        else:
            download(RAW_CSV)
        rows, frauds = verify(normalise(pd.read_csv(RAW_CSV)))
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"OK: {rows:,} rows, {frauds} frauds ({frauds / rows:.3%}) -> {RAW_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
