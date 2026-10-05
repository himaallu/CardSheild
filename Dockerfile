# CardShield scoring API. Serves the exported champion baked in from artifacts/model/.

FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
# LightGBM needs the OpenMP runtime, which slim images do not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Core dependencies only (pinned in pyproject.toml): no MLflow, sklearn or Evidently.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

RUN useradd --system --uid 10001 --create-home cardshield \
    && mkdir -p /data && chown cardshield /data
WORKDIR /app
COPY --chown=cardshield artifacts/model/ /app/artifacts/model/

ENV CARDSHIELD_MODEL_DIR=/app/artifacts/model \
    CARDSHIELD_DB_PATH=/data/predictions.db \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
USER cardshield
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["uvicorn", "cardshield.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
