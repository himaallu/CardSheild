"""Pydantic request/response models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH = 1000


class Transaction(BaseModel):
    """One card payment: all 30 fields required, unknown fields rejected."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Time: float = Field(ge=0, description="Seconds since the first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float = Field(description="SHAP contribution in log-odds; > 0 pushes to fraud")


class ScoreResult(BaseModel):
    fraud_probability: float
    decision: Literal["block", "approve"]
    reasons: list[Reason]


class ScoreResponse(ScoreResult):
    threshold: float
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: list[ScoreResult]
    threshold: float
    model_version: str
    latency_ms: float


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str
