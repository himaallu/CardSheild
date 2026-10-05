"""Shared fixtures. Tests never need the real dataset: they use this synthetic frame."""

import socket
from collections.abc import Iterator
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cardshield.config import AMOUNT_COL, INPUT_COLS, LABEL_COL, TIME_COL, V_COLS
from cardshield.data import FEATURE_COLS, features
from cardshield.export import export_model
from cardshield.serve.app import create_app

TEST_THRESHOLD = 0.5


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


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to reach the internet (e.g. downloading the dataset)."""
    real_connect = socket.socket.connect

    def guarded(sock: socket.socket, address: object) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"network access blocked in tests: {address!r}")
        real_connect(sock, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny LightGBM trained on synthetic data, exported like the real champion."""
    frame = make_synthetic(seed=1)
    params = {"objective": "binary", "verbose": -1, "num_leaves": 8, "seed": 0}
    booster = lgb.train(params, lgb.Dataset(features(frame), frame[LABEL_COL]), 30)
    out = tmp_path_factory.mktemp("model")
    meta = {"features": list(FEATURE_COLS), "threshold": TEST_THRESHOLD, "model_version": "test"}
    return export_model(booster, meta, out)


@pytest.fixture
def client(model_dir: Path, tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(model_dir, tmp_path / "predictions.db")) as c:
        yield c


def _row(frame: pd.DataFrame, label: int) -> dict[str, float]:
    row = frame[frame[LABEL_COL] == label].iloc[0]
    return {c: float(row[c]) for c in INPUT_COLS}


@pytest.fixture
def legit_tx() -> dict[str, float]:
    return _row(make_synthetic(seed=2), 0)


@pytest.fixture
def fraud_tx() -> dict[str, float]:
    return _row(make_synthetic(seed=2), 1)
