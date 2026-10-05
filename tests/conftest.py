"""Shared fixtures. Tests never need the real dataset: they use this synthetic frame."""

import numpy as np
import pandas as pd
import pytest

from cardshield.config import AMOUNT_COL, LABEL_COL, TIME_COL, V_COLS


def make_synthetic(n_rows: int = 2_000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """A frame with the real schema: Time ascending, V1..V28, Amount >= 0, Class in {0, 1}."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_rows) < fraud_rate).astype("int64")
    v = rng.normal(size=(n_rows, len(V_COLS)))
    v[labels == 1, :3] += 3.0  # make frauds separable so tiny models learn something
    frame = pd.DataFrame(v, columns=list(V_COLS))
    frame.insert(0, TIME_COL, np.sort(rng.integers(0, 172_800, n_rows)).astype("float64"))
    frame[AMOUNT_COL] = rng.exponential(80.0, n_rows).round(2)
    frame[LABEL_COL] = labels
    return frame


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic()
