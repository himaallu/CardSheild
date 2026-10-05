import pandas as pd
import pytest

from cardshield.config import LABEL_COL, TIME_COL
from cardshield.data import FEATURE_COLS, features, time_split, validate_schema


def test_split_has_no_time_overlap(synthetic_raw: pd.DataFrame) -> None:
    s = time_split(synthetic_raw)
    assert s.train[TIME_COL].max() < s.val[TIME_COL].min()
    assert s.val[TIME_COL].max() < s.test[TIME_COL].min()


def test_split_covers_all_rows_in_time_order(synthetic_raw: pd.DataFrame) -> None:
    s = time_split(synthetic_raw)
    joined = pd.concat([s.train, s.val, s.test])
    assert len(joined) == len(synthetic_raw)
    assert joined[TIME_COL].is_monotonic_increasing


def test_split_fractions_are_close_to_60_20_20(synthetic_raw: pd.DataFrame) -> None:
    s = time_split(synthetic_raw)
    n = len(synthetic_raw)
    assert abs(len(s.train) / n - 0.6) < 0.01
    assert abs(len(s.val) / n - 0.2) < 0.01


def test_split_moves_cut_past_tied_timestamps() -> None:
    frame = pd.DataFrame({TIME_COL: [0, 1, 2, 3, 5, 5, 5, 7, 8, 9], LABEL_COL: [0] * 10})
    s = time_split(frame)  # naive 60% cut lands inside the run of 5s
    assert s.train[TIME_COL].tolist() == [0, 1, 2, 3, 5, 5, 5]
    assert s.val[TIME_COL].max() < s.test[TIME_COL].min()


def test_split_ignores_input_row_order(synthetic_raw: pd.DataFrame) -> None:
    shuffled = synthetic_raw.sample(frac=1.0, random_state=1)
    a, b = time_split(synthetic_raw), time_split(shuffled)
    assert a.test[TIME_COL].tolist() == b.test[TIME_COL].tolist()


def test_features_exclude_label_and_time(synthetic_raw: pd.DataFrame) -> None:
    cols = list(features(synthetic_raw).columns)
    assert cols == list(FEATURE_COLS)
    assert LABEL_COL not in cols
    assert TIME_COL not in cols
    assert len(cols) == 29


def test_validate_schema_rejects_missing_column(synthetic_raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(synthetic_raw.drop(columns=["Amount"]))


def test_validate_schema_rejects_bad_labels(synthetic_raw: pd.DataFrame) -> None:
    bad = synthetic_raw.copy()
    bad.loc[0, LABEL_COL] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        validate_schema(bad)
