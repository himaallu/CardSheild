import pandas as pd
import pytest

from cardshield.config import EXPECTED_FRAUDS, EXPECTED_ROWS, LABEL_COL, RAW_COLS
from download_data import normalise, verify


def test_normalise_orders_columns_and_casts_category_label(synthetic_raw: pd.DataFrame) -> None:
    shuffled = synthetic_raw[list(reversed(synthetic_raw.columns))].copy()
    shuffled[LABEL_COL] = shuffled[LABEL_COL].astype(str).astype("category")
    out = normalise(shuffled)
    assert tuple(out.columns) == RAW_COLS
    assert out[LABEL_COL].dtype == "int64"
    assert out[LABEL_COL].sum() == synthetic_raw[LABEL_COL].sum()


def test_normalise_rejects_missing_columns(synthetic_raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        normalise(synthetic_raw.drop(columns=["V7"]))


def test_verify_accepts_published_counts() -> None:
    labels = [1] * EXPECTED_FRAUDS + [0] * (EXPECTED_ROWS - EXPECTED_FRAUDS)
    assert verify(pd.DataFrame({LABEL_COL: labels})) == (EXPECTED_ROWS, EXPECTED_FRAUDS)


def test_verify_rejects_wrong_counts(synthetic_raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="dataset mismatch"):
        verify(synthetic_raw)


def test_verify_rejects_unsorted_time() -> None:
    labels = [1] * EXPECTED_FRAUDS + [0] * (EXPECTED_ROWS - EXPECTED_FRAUDS)
    frame = pd.DataFrame({LABEL_COL: labels, "Time": range(EXPECTED_ROWS, 0, -1)})
    with pytest.raises(ValueError, match="not sorted"):
        verify(frame)
