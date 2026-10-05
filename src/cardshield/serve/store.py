"""SQLite prediction log: every scored request, for drift monitoring."""

import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from cardshield.config import INPUT_COLS
from cardshield.serve.model import Score

_COLS = ("ts", "model_version", "probability", "decision", *INPUT_COLS)


class PredictionStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        feature_cols = ", ".join(f'"{c}" REAL NOT NULL' for c in INPUT_COLS)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
            "model_version TEXT NOT NULL, probability REAL NOT NULL, "
            f"decision TEXT NOT NULL, {feature_cols})"
        )
        self._conn.commit()

    def log(
        self, rows: Sequence[Mapping[str, float]], scores: Sequence[Score], model_version: str
    ) -> None:
        ts = datetime.now(UTC).isoformat()
        records = [
            (ts, model_version, s.probability, s.decision, *(row[c] for c in INPUT_COLS))
            for row, s in zip(rows, scores, strict=True)
        ]
        cols = ", ".join(f'"{c}"' for c in _COLS)
        marks = ", ".join("?" * len(_COLS))
        with self._lock, self._conn:
            self._conn.executemany(f"INSERT INTO predictions ({cols}) VALUES ({marks})", records)

    def count(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return int(n)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
