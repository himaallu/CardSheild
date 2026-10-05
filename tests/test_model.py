import random
from pathlib import Path

import numpy as np

from cardshield.data import FEATURE_COLS
from cardshield.serve.model import FraudModel, top_reasons


def test_loads_feature_order_and_threshold_from_metadata(model_dir: Path) -> None:
    model = FraudModel(model_dir)
    assert model.features == list(FEATURE_COLS)
    assert model.threshold == 0.5
    assert model.version == "test"


def test_feature_order_ignores_request_dict_order(
    model_dir: Path, fraud_tx: dict[str, float]
) -> None:
    model = FraudModel(model_dir)
    keys = list(fraud_tx)
    random.Random(0).shuffle(keys)
    shuffled = {k: fraud_tx[k] for k in keys}
    assert model.score([shuffled])[0].probability == model.score([fraud_tx])[0].probability
    assert model.matrix([shuffled])[0].tolist() == [fraud_tx[f] for f in FEATURE_COLS]


def test_reasons_match_lightgbm_contributions(model_dir: Path, fraud_tx: dict[str, float]) -> None:
    model = FraudModel(model_dir)
    s = model.score([fraud_tx])[0]
    contrib = model.booster.predict(model.matrix([fraud_tx]), pred_contrib=True)[0]
    assert len(contrib) == len(FEATURE_COLS) + 1  # trailing bias column
    by_feature = dict(zip(FEATURE_COLS, contrib[:-1], strict=True))
    assert len(s.reasons) == 3
    for r in s.reasons:
        assert r.feature in by_feature  # never the bias term
        assert r.contribution == by_feature[r.feature]
        assert r.value == fraud_tx[r.feature]
    top3 = sorted(by_feature, key=lambda f: -abs(by_feature[f]))[:3]
    assert [r.feature for r in s.reasons] == top3


def test_fraud_scores_higher_and_decision_uses_threshold(
    model_dir: Path, fraud_tx: dict[str, float], legit_tx: dict[str, float]
) -> None:
    model = FraudModel(model_dir)
    fraud, legit = model.score([fraud_tx, legit_tx])
    assert fraud.probability > legit.probability
    for s in (fraud, legit):
        assert s.decision == ("block" if s.probability >= model.threshold else "approve")


def test_top_reasons_ranks_by_absolute_contribution() -> None:
    contrib = np.array([0.1, -2.0, 0.5, 1.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    reasons = top_reasons(contrib, values, ["a", "b", "c", "d"])
    assert [r.feature for r in reasons] == ["b", "d", "c"]
    assert reasons[0].contribution == -2.0
