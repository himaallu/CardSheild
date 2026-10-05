# CardShield — Implementation Plan (2 hours)

Six sprints. Each sprint: plan mode → approve → build → run the checks → commit → `/clear`.
Times are hard limits: if a sprint runs over, cut its P1 items and move on.

| Sprint | Minutes | Goal |
| --- | --- | --- |
| 0 | 10 | Repo, tooling, data |
| 1 | 30 | Training + MLflow + threshold + export |
| 2 | 25 | FastAPI serving + explanations + logging |
| 3 | 20 | Docker, compose, CI |
| 4 | 20 | Replay, drift report, latency bench |
| 5 | 15 | README with real numbers, final push |

---

## Sprint 0 — Setup (10 min)

- [x] `pyproject.toml` with pinned deps: pandas, numpy, scikit-learn, lightgbm, mlflow, fastapi, uvicorn, pydantic, prometheus-client, evidently, httpx, pytest, ruff, mypy, matplotlib
- [x] Package skeleton matching the repo layout in `docs/PRD.md`
- [x] `.gitignore` (data/, mlruns/, mlflow.db, reports/, .venv/, __pycache__/)
- [x] `Makefile` with all targets from the PRD (stubs are fine for later sprints)
- [x] `scripts/download_data.py`: original CSV (TensorFlow mirror, SHA-256 pinned; OpenML 1597 drops `Time`), save to `data/creditcard.csv`, assert 284,807 rows and 492 frauds

**Check:** `make data` prints row and fraud counts that match. First commit.

## Sprint 1 — Training pipeline (30 min)

- [x] `data.py`: load, validate columns, time-based 60/20/20 split by `Time`
- [x] `threshold.py`: cost-based threshold search on validation (missed fraud = `Amount`, false alarm = `REVIEW_COST`)
- [x] `train.py`: logistic regression baseline + LightGBM (`scale_pos_weight`); log params, PR-AUC, ROC-AUC, precision/recall/F1 at threshold, PR curve PNG, confusion matrix, data hash to MLflow (`sqlite:///mlflow.db`)
- [x] Business-impact table on test: total cost for approve-all, best amount-threshold rule (tuned on validation), and the model; log to MLflow
- [x] Register best LightGBM as `cardshield`, set alias `champion`
- [x] `export.py`: write `artifacts/model/model.txt` + `metadata.json`
- [x] Tests: split has no time overlap; threshold function picks the cost minimum on a toy example; export writes both files

**Check:** `make train` prints a metrics table and the cost table (model cheaper than both baselines); `mlflow ui` shows 2 runs and the champion alias; `make test` green. Commit.

## Sprint 2 — Serving (25 min)

- [ ] `schemas.py`: `Transaction` (Time, Amount ≥ 0, V1–V28, `extra="forbid"`), `ScoreResponse`, batch models
- [ ] `model.py`: load artifact once; `predict_proba`; top-3 reasons from `pred_contrib=True` (feature, value, contribution)
- [ ] `store.py`: SQLite prediction log
- [ ] `app.py`: `/v1/score`, `/v1/score/batch` (≤ 1,000), `/healthz`, `/metrics` (Prometheus counter + latency histogram)
- [ ] Tests with FastAPI TestClient: valid request → 200 with 3 reasons; missing field → 422; negative Amount → 422; batch over limit → 422; log row written

**Check:** `make serve`, open `/docs`, score one real fraud row from the test set and see a high probability with reasons. `make test` green. Commit.

## Sprint 3 — Docker and CI (20 min)

- [ ] `Dockerfile`: python slim, non-root user, install deps, copy `src/` and `artifacts/model/`, run uvicorn, `HEALTHCHECK` on `/healthz`
- [ ] `docker-compose.yml`: `api` (port 8000, volume for the SQLite log) and `mlflow` UI (port 5000)
- [ ] `.github/workflows/ci.yml`: install, ruff, pytest (synthetic fixture only, no dataset download), docker build
- [ ] `tests/conftest.py` synthetic fixture with the real schema; a tiny model trained on it for API tests in CI

**Check:** `make up` → `curl localhost:8000/healthz` returns the model version. Push and confirm CI is green. Commit.

## Sprint 4 — Monitoring and performance (20 min)

- [ ] `scripts/replay.py`: send test rows to the API at a set rate; `--drift` multiplies `Amount` ×3 and shifts V14 and V17 for the second half
- [ ] `monitoring/drift.py`: reference = sample of training data; current = prediction log; Evidently data-drift report to `reports/drift.html` + one-line summary. If Evidently errors after one quick fix attempt, use the PSI fallback
- [ ] `scripts/bench.py`: async httpx load test, prints p50/p95/p99 and req/s for single and batch requests

**Check:** normal replay → drift summary shows few drifted features; drifted replay → `Amount` flagged. Bench p95 < 50 ms. Save a screenshot of the drift report. Commit.

## Sprint 5 — README and ship (15 min)

- [ ] README: problem and objective (from the PRD), cost-saved table, Mermaid architecture diagram (from the PRD), real metrics table, latency table, drift screenshot, "Run it" commands, design decisions, limitations (PCA features)
- [ ] Fill the resume bullet in `docs/PRD.md` with real numbers
- [ ] Final `make lint && make test`, push, CI green

**Check:** a stranger could clone and run it from the README alone.

---

## If time runs out, cut in this order

1. Batch endpoint
2. Logistic regression baseline (keep LightGBM only)
3. MLflow UI in compose (keep tracking locally)
4. Evidently (use the PSI fallback)

Never cut: time-based split, cost-based threshold, tests, Docker, README numbers.

## P1 (only after Sprint 5)

ONNX export + latency comparison · Locust file · deploy to Render/Fly.io · live flagged-transactions page · champion/challenger promotion script.

---

## Prompts to paste into Claude Code

**Start of every sprint (in plan mode, Shift+Tab):**

```
We're starting Sprint N in docs/IMPLEMENTATION_PLAN.md. Read the sprint, the
relevant parts of docs/PRD.md and the current code. Tell me your approach and
the files you'll create or change, then wait for my OK.
```

**After approving (leave plan mode):**

```
Go. Run make test and make lint after each task, tick the checkboxes in
docs/IMPLEMENTATION_PLAN.md, and commit when the sprint's Check passes.
Stop and tell me if anything would take more than the sprint's time box.
```

**End of every sprint:**

```
Summarise what you built, what you skipped, and what I should verify by hand.
Then explain the two most important decisions in this sprint as if I'll be
asked about them in an interview.
```

Then type `/clear` before the next sprint.
