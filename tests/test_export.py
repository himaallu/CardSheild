from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from cardshield.config import LABEL_COL
from cardshield.data import FEATURE_COLS, features
from cardshield.export import METADATA_FILE, MODEL_FILE, export_model, load_metadata


def _tiny_booster(frame: pd.DataFrame) -> lgb.Booster:
    params = {"objective": "binary", "verbose": -1, "num_leaves": 4, "seed": 0}
    return lgb.train(params, lgb.Dataset(features(frame), frame[LABEL_COL]), num_boost_round=5)


def test_export_writes_model_and_metadata(synthetic_raw: pd.DataFrame, tmp_path: Path) -> None:
    booster = _tiny_booster(synthetic_raw)
    meta = {"features": list(FEATURE_COLS), "threshold": 0.42, "model_version": "1"}
    export_model(booster, meta, tmp_path)
    assert (tmp_path / MODEL_FILE).is_file()
    assert (tmp_path / METADATA_FILE).is_file()
    assert load_metadata(tmp_path) == meta


def test_exported_model_round_trips(synthetic_raw: pd.DataFrame, tmp_path: Path) -> None:
    booster = _tiny_booster(synthetic_raw)
    export_model(booster, {"features": list(FEATURE_COLS)}, tmp_path)
    loaded = lgb.Booster(model_file=str(tmp_path / MODEL_FILE))
    x = features(synthetic_raw.head(20))
    assert loaded.feature_name() == list(FEATURE_COLS)
    assert np.array_equal(np.asarray(loaded.predict(x)), np.asarray(booster.predict(x)))


def test_export_rejects_mismatched_feature_order(
    synthetic_raw: pd.DataFrame, tmp_path: Path
) -> None:
    booster = _tiny_booster(synthetic_raw)

    with pytest.raises(ValueError, match="feature order"):
        export_model(booster, {"features": list(reversed(FEATURE_COLS))}, tmp_path)
