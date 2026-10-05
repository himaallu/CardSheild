import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from cardshield.serve.app import create_app
from cardshield.serve.schemas import MAX_BATCH


def test_score_returns_decision_with_three_reasons(
    client: TestClient, fraud_tx: dict[str, float]
) -> None:
    r = client.post("/v1/score", json=fraud_tx)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "fraud_probability",
        "decision",
        "threshold",
        "model_version",
        "reasons",
        "latency_ms",
    }
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"block", "approve"}
    assert body["model_version"] == "test"
    assert len(body["reasons"]) == 3
    assert set(body["reasons"][0]) == {"feature", "value", "contribution"}
    assert body["latency_ms"] > 0


def test_missing_field_is_422(client: TestClient, legit_tx: dict[str, float]) -> None:
    del legit_tx["V14"]
    assert client.post("/v1/score", json=legit_tx).status_code == 422


def test_negative_amount_is_422(client: TestClient, legit_tx: dict[str, float]) -> None:
    legit_tx["Amount"] = -1.0
    assert client.post("/v1/score", json=legit_tx).status_code == 422


def test_unknown_field_is_422(client: TestClient, legit_tx: dict[str, float]) -> None:
    assert client.post("/v1/score", json={**legit_tx, "Class": 1}).status_code == 422


def test_non_numeric_field_is_422(client: TestClient, legit_tx: dict[str, float]) -> None:
    assert client.post("/v1/score", json={**legit_tx, "V1": "abc"}).status_code == 422


def test_batch_scores_each_transaction(
    client: TestClient, fraud_tx: dict[str, float], legit_tx: dict[str, float]
) -> None:
    r = client.post("/v1/score/batch", json={"transactions": [fraud_tx, legit_tx]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert all(len(x["reasons"]) == 3 for x in results)
    assert results[0]["fraud_probability"] > results[1]["fraud_probability"]


def test_batch_over_limit_is_422(client: TestClient, legit_tx: dict[str, float]) -> None:
    payload = {"transactions": [legit_tx] * (MAX_BATCH + 1)}
    assert client.post("/v1/score/batch", json=payload).status_code == 422


def test_empty_batch_is_422(client: TestClient) -> None:
    assert client.post("/v1/score/batch", json={"transactions": []}).status_code == 422


def test_every_scored_transaction_is_logged(
    model_dir: Path, tmp_path: Path, fraud_tx: dict[str, float], legit_tx: dict[str, float]
) -> None:
    db = tmp_path / "log.db"
    with TestClient(create_app(model_dir, db)) as c:
        c.post("/v1/score", json=fraud_tx)
        c.post("/v1/score/batch", json={"transactions": [fraud_tx, legit_tx]})
        c.post("/v1/score", json={**legit_tx, "Amount": -5})  # rejected: not logged
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            'SELECT model_version, decision, probability, "Amount", "Time" FROM predictions'
        ).fetchall()
    assert len(rows) == 3
    assert rows[0][0] == "test"
    assert rows[0][3] == fraud_tx["Amount"]
    assert rows[0][4] == fraud_tx["Time"]


def test_healthz_reports_model_version(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_metrics_count_requests_and_decisions(
    client: TestClient, fraud_tx: dict[str, float]
) -> None:
    client.post("/v1/score", json=fraud_tx)
    text = client.get("/metrics").text
    assert 'cardshield_requests_total{endpoint="score"} 1.0' in text
    assert "cardshield_request_latency_seconds_bucket" in text
    assert "cardshield_decisions_total" in text
