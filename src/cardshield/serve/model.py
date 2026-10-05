"""Artifact loading, predict, explain."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from cardshield.export import METADATA_FILE, MODEL_FILE

FloatMatrix = npt.NDArray[np.float64]
N_REASONS = 3


@dataclass(frozen=True)
class Reason:
    feature: str
    value: float
    contribution: float


@dataclass(frozen=True)
class Score:
    probability: float
    decision: Literal["block", "approve"]
    reasons: list[Reason]


def top_reasons(
    contrib: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    features: Sequence[str],
    k: int = N_REASONS,
) -> list[Reason]:
    """Top-k features by |contribution|. `contrib` must exclude LightGBM's bias column."""
    order = np.argsort(-np.abs(contrib), kind="stable")[:k]
    return [Reason(features[i], float(values[i]), float(contrib[i])) for i in order]


class FraudModel:
    """The exported champion: LightGBM booster plus metadata.json."""

    def __init__(self, model_dir: Path) -> None:
        meta = json.loads((model_dir / METADATA_FILE).read_text())
        self.booster = lgb.Booster(model_file=str(model_dir / MODEL_FILE))
        # Serve-time feature order comes from metadata.json, never from request dict order.
        self.features: list[str] = list(meta["features"])
        if self.booster.feature_name() != self.features:
            raise ValueError("metadata.json feature order does not match model.txt")
        self.threshold = float(meta["threshold"])
        self.version = str(meta["model_version"])

    def matrix(self, rows: Sequence[Mapping[str, float]]) -> FloatMatrix:
        return np.array([[row[f] for f in self.features] for row in rows], dtype=np.float64)

    def score(self, rows: Sequence[Mapping[str, float]]) -> list[Score]:
        x = self.matrix(rows)
        proba = np.asarray(self.booster.predict(x), dtype=np.float64)
        # pred_contrib returns one column per feature plus a trailing bias column: drop it.
        contrib = np.asarray(self.booster.predict(x, pred_contrib=True), dtype=np.float64)[:, :-1]
        return [
            Score(
                probability=float(p),
                decision="block" if p >= self.threshold else "approve",
                reasons=top_reasons(c, v, self.features),
            )
            for p, c, v in zip(proba, contrib, x, strict=True)
        ]
