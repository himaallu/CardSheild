"""Cost-based threshold search.

Cost of a policy = sum of Amount over missed frauds + review_cost x false alarms.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cardshield.config import REVIEW_COST

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CostResult:
    threshold: float  # flag when score >= threshold; +inf means flag nothing
    cost: float
    missed_fraud_amount: float
    false_alarms: int
    caught: int
    missed: int


def total_cost(
    y: IntArray, amount: FloatArray, flagged: npt.NDArray[np.bool_], review_cost: float
) -> float:
    missed = (y == 1) & ~flagged
    false_alarms = (y == 0) & flagged
    return float(amount[missed].sum() + review_cost * false_alarms.sum())


def best_threshold(
    y: IntArray, score: FloatArray, amount: FloatArray, review_cost: float = REVIEW_COST
) -> CostResult:
    """Cheapest cutoff over every distinct score, evaluated in one vectorised pass.

    Candidates are "flag nothing" plus each distinct score s (flag when score >= s).
    """
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    amount = np.asarray(amount, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    s, yy, amt = score[order], y[order], amount[order]
    # Flagging the top k rows: caught fraud amount and false alarms as k grows.
    caught_amt = np.cumsum(np.where(yy == 1, amt, 0.0))
    fp = np.cumsum(yy == 0)
    tp = np.cumsum(yy == 1)
    # Only cut between distinct scores: keep the last index of each tie group.
    last_of_group = np.r_[s[1:] != s[:-1], True]
    total_fraud_amt = float(amt[yy == 1].sum())
    n_fraud = int((yy == 1).sum())
    costs = total_fraud_amt - caught_amt[last_of_group] + review_cost * fp[last_of_group]
    best = int(np.argmin(costs)) if len(costs) else -1
    if best == -1 or costs[best] >= total_fraud_amt:
        return CostResult(float("inf"), total_fraud_amt, total_fraud_amt, 0, 0, n_fraud)
    k = np.flatnonzero(last_of_group)[best]
    return CostResult(
        threshold=float(s[k]),
        cost=float(costs[best]),
        missed_fraud_amount=float(total_fraud_amt - caught_amt[k]),
        false_alarms=int(fp[k]),
        caught=int(tp[k]),
        missed=n_fraud - int(tp[k]),
    )


def evaluate_threshold(
    y: IntArray,
    score: FloatArray,
    amount: FloatArray,
    threshold: float,
    review_cost: float = REVIEW_COST,
) -> CostResult:
    """Cost of a fixed cutoff (e.g. the validation threshold applied to test)."""
    y = np.asarray(y, dtype=np.int64)
    amount = np.asarray(amount, dtype=np.float64)
    flagged = np.asarray(score, dtype=np.float64) >= threshold
    missed = (y == 1) & ~flagged
    return CostResult(
        threshold=threshold,
        cost=total_cost(y, amount, flagged, review_cost),
        missed_fraud_amount=float(amount[missed].sum()),
        false_alarms=int(((y == 0) & flagged).sum()),
        caught=int(((y == 1) & flagged).sum()),
        missed=int(missed.sum()),
    )


def best_amount_rule(
    y: IntArray, amount: FloatArray, review_cost: float = REVIEW_COST
) -> CostResult:
    """Best "flag if Amount >= X" rule; may be "flag nothing" (threshold = inf)."""
    return best_threshold(y, amount, amount, review_cost)
