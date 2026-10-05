import itertools
import math

import numpy as np
import pytest

from cardshield.threshold import (
    best_amount_rule,
    best_threshold,
    evaluate_threshold,
    total_cost,
)


def test_total_cost_adds_missed_amounts_and_review_cost() -> None:
    y = np.array([1, 1, 0, 0])
    amount = np.array([100.0, 30.0, 10.0, 10.0])
    flagged = np.array([True, False, True, False])
    assert total_cost(y, amount, flagged, review_cost=5.0) == 30.0 + 5.0


def test_best_threshold_picks_known_minimum() -> None:
    # Scores descending: fraud 100, legit, legit, fraud 8, legit.
    y = np.array([1, 0, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    amount = np.array([100.0, 1.0, 1.0, 8.0, 1.0])
    # Cutoffs: none=108, >=0.9 -> 8, >=0.8 -> 13, >=0.7 -> 18, >=0.6 -> 10, >=0.1 -> 15.
    r = best_threshold(y, score, amount, review_cost=5.0)
    assert r.threshold == 0.9
    assert r.cost == 8.0
    assert (r.caught, r.missed, r.false_alarms) == (1, 1, 0)


def test_best_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.1).astype(np.int64)
    score = np.round(rng.random(200) + 0.3 * y, 2)  # rounding creates ties
    amount = rng.exponential(50.0, 200)
    candidates = [math.inf, *np.unique(score)]
    brute = min(total_cost(y, amount, score >= t, 5.0) for t in candidates)
    assert best_threshold(y, score, amount, 5.0).cost == pytest.approx(brute)


def test_ties_are_flagged_together() -> None:
    y = np.array([1, 0, 0])
    score = np.array([0.5, 0.5, 0.1])
    amount = np.array([100.0, 1.0, 1.0])
    r = best_threshold(y, score, amount, review_cost=5.0)
    assert r.threshold == 0.5
    assert r.false_alarms == 1


def test_flag_nothing_when_review_cost_exceeds_fraud() -> None:
    y = np.array([1, 0, 0])
    score = np.array([0.9, 0.95, 0.1])
    amount = np.array([2.0, 1.0, 1.0])
    r = best_threshold(y, score, amount, review_cost=5.0)
    assert math.isinf(r.threshold)
    assert r.cost == 2.0


def test_amount_rule_can_choose_flag_nothing() -> None:
    y = np.array([0, 0, 1, 0])
    amount = np.array([500.0, 300.0, 3.0, 1.0])
    assert math.isinf(best_amount_rule(y, amount, review_cost=5.0).threshold)


def test_evaluate_threshold_agrees_with_best() -> None:
    y = np.array([1, 0, 1, 0, 0, 1])
    score = np.array([0.9, 0.2, 0.7, 0.6, 0.3, 0.1])
    amount = np.array([50.0, 5.0, 20.0, 5.0, 5.0, 2.0])
    best = best_threshold(y, score, amount)
    again = evaluate_threshold(y, score, amount, best.threshold)
    assert again == best
    for t in itertools.chain([math.inf], score):
        assert evaluate_threshold(y, score, amount, float(t)).cost >= best.cost
