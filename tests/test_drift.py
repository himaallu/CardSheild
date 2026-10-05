from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardshield.config import AMOUNT_COL, INPUT_COLS
from cardshield.monitoring import drift
from cardshield.monitoring.drift import (
    DRIFT_COLS,
    DriftResult,
    apply_drift,
    build_report,
    drift_shifts,
    parse_column_drift,
    psi,
    run_evidently,
    run_psi,
    summarize,
)
from cardshield.serve.model import Reason, Score
from cardshield.serve.store import PredictionStore
from tests.conftest import make_synthetic


def _shifted(frame: pd.DataFrame) -> pd.DataFrame:
    """Whole-frame drift on Amount and V14 (V17 untouched)."""
    out = frame.copy()
    out[AMOUNT_COL] *= 3
    out["V14"] += 2.0
    return out


@pytest.fixture
def reference() -> pd.DataFrame:
    return make_synthetic(n_rows=2_000, seed=10)


@pytest.fixture
def same_dist() -> pd.DataFrame:
    return make_synthetic(n_rows=2_000, seed=11)


def test_apply_drift_changes_only_second_half_and_target_columns(
    synthetic_raw: pd.DataFrame,
) -> None:
    shifts = {"V14": 1.5, "V17": -0.5}
    out = apply_drift(synthetic_raw, shifts, amount_factor=3.0)
    half = len(synthetic_raw) // 2
    first, second = slice(0, half), slice(half, None)
    pd.testing.assert_frame_equal(out.iloc[first], synthetic_raw.iloc[first])
    np.testing.assert_allclose(
        out[AMOUNT_COL].iloc[second], synthetic_raw[AMOUNT_COL].iloc[second] * 3
    )
    np.testing.assert_allclose(out["V14"].iloc[second], synthetic_raw["V14"].iloc[second] + 1.5)
    np.testing.assert_allclose(out["V17"].iloc[second], synthetic_raw["V17"].iloc[second] - 0.5)
    untouched = [c for c in synthetic_raw.columns if c not in (AMOUNT_COL, "V14", "V17")]
    pd.testing.assert_frame_equal(out[untouched], synthetic_raw[untouched])
    assert synthetic_raw is not out  # input not mutated


def test_drift_shifts_are_k_standard_deviations(synthetic_raw: pd.DataFrame) -> None:
    shifts = drift_shifts(synthetic_raw, k=2.0)
    assert set(shifts) == {"V14", "V17"}
    assert shifts["V14"] == pytest.approx(2.0 * synthetic_raw["V14"].std())


def test_psi_near_zero_for_same_distribution_and_large_for_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5_000)
    assert psi(ref, rng.normal(size=5_000)) < 0.02
    assert psi(ref, rng.normal(loc=1.0, size=5_000)) > 0.2


def test_parse_column_drift_handles_distance_and_p_value_tests() -> None:
    wass = "ValueDrift(column=V14,method=Wasserstein distance (normed),threshold=0.1)"
    ks = "ValueDrift(column=Amount,method=K-S p_value,threshold=0.05)"
    assert parse_column_drift(wass, 0.3) == ("V14", True)
    assert parse_column_drift(wass, 0.05) == ("V14", False)
    assert parse_column_drift(ks, 0.01) == ("Amount", True)  # small p-value = drift
    assert parse_column_drift(ks, 0.5) == ("Amount", False)
    assert parse_column_drift("DriftedColumnsCount(drift_share=0.5)", 0.1) is None


def test_psi_report_flags_shifted_columns_only(
    reference: pd.DataFrame, same_dist: pd.DataFrame, tmp_path: Path
) -> None:
    assert run_psi(reference, same_dist, tmp_path / "a.html").drifted == []
    result = run_psi(reference, _shifted(same_dist), tmp_path / "b.html")
    assert set(result.drifted) == {AMOUNT_COL, "V14"}
    assert result.n_features == len(DRIFT_COLS)
    assert "PSI" in (tmp_path / "b.html").read_text()


def test_evidently_report_flags_shifted_columns_only(
    reference: pd.DataFrame, same_dist: pd.DataFrame, tmp_path: Path
) -> None:
    result = run_evidently(reference, _shifted(same_dist), tmp_path / "drift.html")
    assert result.method == "evidently"
    assert set(result.drifted) == {AMOUNT_COL, "V14"}
    assert (tmp_path / "drift.html").stat().st_size > 0


def test_build_report_falls_back_to_psi_when_evidently_fails(
    reference: pd.DataFrame,
    same_dist: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_: object) -> DriftResult:
        raise ImportError("evidently API changed")

    monkeypatch.setattr(drift, "run_evidently", broken)
    result = build_report(reference, _shifted(same_dist), tmp_path)
    assert result.method == "psi"
    assert set(result.drifted) == {AMOUNT_COL, "V14"}
    assert (tmp_path / "drift.html").exists()
    with pytest.raises(ImportError):
        build_report(reference, same_dist, tmp_path, method="evidently")


def test_summary_line_format() -> None:
    result = DriftResult("evidently", ["V17", "Amount", "V14"], 29)
    assert summarize(result) == "Drift [evidently]: 3/29 features drifted (10%): Amount, V14, V17"
    assert summarize(DriftResult("psi", [], 29)).endswith("0/29 features drifted (0%): none")


def test_store_recent_returns_latest_rows_oldest_first(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "log.db")
    frame = make_synthetic(n_rows=10, seed=4)
    rows = [{c: float(r[c]) for c in INPUT_COLS} for _, r in frame.iterrows()]
    scores = [Score(0.1, "approve", [Reason("V1", 0.0, 0.0)]) for _ in rows]
    store.log(rows, scores, "test")
    recent = store.recent(3)
    store.close()
    assert len(recent) == 3
    assert recent["Amount"].tolist() == [r["Amount"] for r in rows[-3:]]
    assert set(DRIFT_COLS) <= set(recent.columns)
