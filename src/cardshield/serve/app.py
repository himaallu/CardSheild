"""FastAPI app, routes, startup model load."""

import os
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from cardshield.config import MODEL_DIR, ROOT
from cardshield.serve.model import FraudModel, Score
from cardshield.serve.schemas import (
    BatchRequest,
    BatchResponse,
    Health,
    Reason,
    ScoreResponse,
    ScoreResult,
    Transaction,
)
from cardshield.serve.store import PredictionStore

DEFAULT_DB = ROOT / "predictions.db"
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "cardshield_requests_total", "Scored requests", ["endpoint"], registry=self.registry
        )
        self.latency = Histogram(
            "cardshield_request_latency_seconds",
            "Scoring latency (model + log)",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            "cardshield_decisions_total", "Decisions by class", ["decision"], registry=self.registry
        )


def _result(score: Score) -> ScoreResult:
    return ScoreResult(
        fraud_probability=score.probability,
        decision=score.decision,
        reasons=[
            Reason(feature=r.feature, value=r.value, contribution=r.contribution)
            for r in score.reasons
        ],
    )


def create_app(model_dir: Path | None = None, db_path: Path | None = None) -> FastAPI:
    model_dir = model_dir or Path(os.environ.get("CARDSHIELD_MODEL_DIR", MODEL_DIR))
    db_path = db_path or Path(os.environ.get("CARDSHIELD_DB_PATH", DEFAULT_DB))
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = FraudModel(model_dir)  # loaded once, at startup
        app.state.store = PredictionStore(db_path)
        yield
        app.state.store.close()

    app = FastAPI(title="CardShield", version="0.1.0", lifespan=lifespan)

    def score_and_log(
        request: Request, txs: Sequence[Transaction], endpoint: str
    ) -> tuple[list[Score], float]:
        model: FraudModel = request.app.state.model
        store: PredictionStore = request.app.state.store
        start = time.perf_counter()
        rows = [tx.model_dump() for tx in txs]
        scores = model.score(rows)
        store.log(rows, scores, model.version)
        elapsed = time.perf_counter() - start
        metrics.requests.labels(endpoint).inc()
        metrics.latency.labels(endpoint).observe(elapsed)
        for s in scores:
            metrics.decisions.labels(s.decision).inc()
        return scores, elapsed * 1000

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(tx: Transaction, request: Request) -> ScoreResponse:
        model: FraudModel = request.app.state.model
        scores, latency_ms = score_and_log(request, [tx], "score")
        return ScoreResponse(
            **_result(scores[0]).model_dump(),
            threshold=model.threshold,
            model_version=model.version,
            latency_ms=latency_ms,
        )

    @app.post("/v1/score/batch", response_model=BatchResponse)
    def score_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        model: FraudModel = request.app.state.model
        scores, latency_ms = score_and_log(request, batch.transactions, "batch")
        return BatchResponse(
            results=[_result(s) for s in scores],
            threshold=model.threshold,
            model_version=model.version,
            latency_ms=latency_ms,
        )

    @app.get("/healthz", response_model=Health)
    def healthz(request: Request) -> Health:
        model: FraudModel | None = getattr(request.app.state, "model", None)
        return Health(
            status="ok",
            model_loaded=model is not None,
            model_version=model.version if model else "",
        )

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
