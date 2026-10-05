"""Data-drift report: validation-period reference vs recently logged traffic.

Evidently (new API, 0.7.x) builds the HTML report; if it fails, a hand-written PSI per
feature produces the same one-line summary. `Time` is excluded: it only ever increases.
"""

import argparse
import html
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from cardshield.config import AMOUNT_COL, RAW_CSV, REPORTS_DIR, ROOT
from cardshield.data import FEATURE_COLS, load_raw, time_split

DRIFT_COLS: Final[tuple[str, ...]] = FEATURE_COLS
REFERENCE_ROWS: Final = 5000
WINDOW: Final = 2000  # rows of logged traffic compared; matches one default replay
PSI_THRESHOLD: Final = 0.2  # conventional "significant shift"
DRIFT_FEATURES: Final = ("V14", "V17")
AMOUNT_FACTOR: Final = 3.0
SHIFT_STDS: Final = 2.0
SEED: Final = 0
DEFAULT_DB: Final = ROOT / "predictions.db"


@dataclass(frozen=True)
class DriftResult:
    method: str
    drifted: list[str]
    n_features: int

    @property
    def share(self) -> float:
        return len(self.drifted) / self.n_features if self.n_features else 0.0


# ---- simulation (used by scripts/replay.py) -------------------------------------------


def drift_shifts(reference: pd.DataFrame, k: float = SHIFT_STDS) -> dict[str, float]:
    """Additive shift per drifted feature: k standard deviations of the reference."""
    return {c: float(k * reference[c].std()) for c in DRIFT_FEATURES}


def apply_drift(
    frame: pd.DataFrame, shifts: Mapping[str, float], amount_factor: float = AMOUNT_FACTOR
) -> pd.DataFrame:
    """Simulated drift on the second half of `frame`: Amount x factor, features shifted."""
    out = frame.copy()
    half = out.index[len(out) // 2 :]
    out.loc[half, AMOUNT_COL] = out.loc[half, AMOUNT_COL] * amount_factor
    for col, delta in shifts.items():
        out.loc[half, col] = out.loc[half, col] + delta
    return out


# ---- reference and current data -------------------------------------------------------


def reference_sample(raw: pd.DataFrame, n: int = REFERENCE_ROWS) -> pd.DataFrame:
    """Sample of the validation period: the most recent window the model was validated on.
    (A training-period reference flags normal test traffic: the V features vary by time
    of day and training covers different hours.)"""
    val = time_split(raw).val
    return val.sample(min(n, len(val)), random_state=SEED).reset_index(drop=True)


def load_current(db_path: Path, n: int = WINDOW) -> pd.DataFrame:
    from cardshield.serve.store import PredictionStore

    if not db_path.exists():
        raise FileNotFoundError(f"no prediction log at {db_path}: run the API and a replay first")
    store = PredictionStore(db_path)
    try:
        return store.recent(n)
    finally:
        store.close()


# ---- PSI (fallback) -------------------------------------------------------------------


def psi(reference: npt.ArrayLike, current: npt.ArrayLike, bins: int = 10) -> float:
    """Population Stability Index over reference-quantile bins."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_scores(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    return {c: psi(reference[c], current[c]) for c in DRIFT_COLS}


def run_psi(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path) -> DriftResult:
    scores = psi_scores(reference, current)
    drifted = [c for c, v in scores.items() if v > PSI_THRESHOLD]
    rows = "".join(
        f"<tr><td>{html.escape(c)}</td><td>{v:.3f}</td>"
        f"<td>{'yes' if c in drifted else ''}</td></tr>"
        for c, v in scores.items()
    )
    html_path.write_text(
        "<html><body><h1>CardShield drift (PSI fallback)</h1>"
        f"<p>Drifted if PSI &gt; {PSI_THRESHOLD}</p><table border=1>"
        f"<tr><th>feature</th><th>PSI</th><th>drifted</th></tr>{rows}</table></body></html>"
    )
    return DriftResult("psi", drifted, len(DRIFT_COLS))


# ---- Evidently ------------------------------------------------------------------------

_VALUE_DRIFT = re.compile(
    r"ValueDrift\(column=(?P<col>[^,]+),method=(?P<method>[^,]+),threshold=(?P<thr>[0-9.eE-]+)\)"
)


def parse_column_drift(metric_name: str, value: float) -> tuple[str, bool] | None:
    """(column, drifted) from an Evidently ValueDrift metric. p-value tests drift when the
    value is below the threshold; distance tests (e.g. Wasserstein) when at or above it."""
    m = _VALUE_DRIFT.fullmatch(metric_name)
    if m is None:
        return None
    thr = float(m["thr"])
    drifted = value < thr if "p_value" in m["method"] else value >= thr
    return m["col"], drifted


def run_evidently(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path) -> DriftResult:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    cols = list(DRIFT_COLS)
    snapshot = Report([DataDriftPreset()]).run(
        current[cols].reset_index(drop=True), reference[cols].reset_index(drop=True)
    )
    snapshot.save_html(str(html_path))
    parsed = [
        parse_column_drift(m["metric_name"], float(m["value"]))
        for m in snapshot.dict()["metrics"]
        if isinstance(m["value"], int | float)  # skips the DriftedColumnsCount summary dict
    ]
    columns = [p for p in parsed if p is not None]
    if len(columns) != len(cols):
        raise RuntimeError(f"expected {len(cols)} column drift metrics, got {len(columns)}")
    return DriftResult("evidently", [c for c, d in columns if d], len(cols))


# ---- entry point ----------------------------------------------------------------------


def summarize(result: DriftResult) -> str:
    names = ", ".join(sorted(result.drifted)) or "none"
    return (
        f"Drift [{result.method}]: {len(result.drifted)}/{result.n_features} features "
        f"drifted ({result.share:.0%}): {names}"
    )


def build_report(
    reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path, method: str = "auto"
) -> DriftResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "drift.html"
    if method == "psi":
        return run_psi(reference, current, html_path)
    try:
        return run_evidently(reference, current, html_path)
    except Exception as err:  # noqa: BLE001 - PRD M3: any Evidently failure -> PSI fallback
        if method == "evidently":
            raise
        print(f"Evidently failed ({err!r}); using PSI fallback", file=sys.stderr)
        return run_psi(reference, current, html_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=Path(os.environ.get("CARDSHIELD_DB_PATH", DEFAULT_DB))
    )
    parser.add_argument("--last", type=int, default=WINDOW, help="logged rows to compare")
    parser.add_argument("--method", choices=["auto", "evidently", "psi"], default="auto")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    current = load_current(args.db, args.last)
    reference = reference_sample(load_raw(RAW_CSV))
    result = build_report(reference, current, args.out, args.method)
    print(f"Compared {len(current)} logged rows with {len(reference)} validation rows")
    print(summarize(result))
    print(f"Report: {args.out / 'drift.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
