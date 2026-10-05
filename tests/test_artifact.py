"""The committed artifact the Docker image ships loads and scores, without the dataset."""

import json
import socket

import pytest

from cardshield.config import INPUT_COLS, MODEL_DIR
from cardshield.data import FEATURE_COLS
from cardshield.export import METADATA_FILE, MODEL_FILE
from cardshield.serve.model import FraudModel

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / MODEL_FILE).exists(), reason="no exported model in artifacts/model/"
)


def test_committed_artifact_loads_and_scores(legit_tx: dict[str, float]) -> None:
    model = FraudModel(MODEL_DIR)
    meta = json.loads((MODEL_DIR / METADATA_FILE).read_text())
    assert model.features == list(FEATURE_COLS)
    assert model.version == str(meta["model_version"])
    assert 0.0 < model.threshold < 1.0
    [score] = model.score([{c: legit_tx[c] for c in INPUT_COLS}])
    assert 0.0 <= score.probability <= 1.0
    assert len(score.reasons) == 3


def test_network_is_blocked_in_tests() -> None:
    with pytest.raises(RuntimeError, match="network access"):
        socket.create_connection(("storage.googleapis.com", 443), timeout=1)
