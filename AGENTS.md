# CardShield

Objective: block fraudulent card payments in real time while minimising the total cost of
fraud (missed-fraud losses + review cost of false alarms), explain every decision, and detect
when the model goes stale.

How: an MLOps loop with LightGBM trained with a
time-based split and cost-based threshold, MLflow tracking and registry, FastAPI serving
with SHAP reasons, SQLite prediction log, Evidently drift monitoring, Docker and CI.

- Spec and system design: docs/PRD.md
- Sprint plan and progress: docs/IMPLEMENTATION_PLAN.md

## Stack

Python 3.12 · pandas · scikit-learn · LightGBM · MLflow (tracking URI `sqlite:///mlflow.db`) ·
FastAPI + Pydantic v2 · prometheus-client · Evidently · pytest · ruff · mypy · Docker ·
GitHub Actions.

## Commands

- `make data` download and verify the dataset into data/
- `make train` train, log to MLflow, register champion, export to artifacts/model/
- `make serve` run the API on :8000
- `make test` / `make lint`
- `make replay` / `make replay-drift` / `make drift` / `make bench`
- `make up` docker compose (API :8000, MLflow UI :5000)

## Layout

Code lives in `src/cardshield/` (data, train, threshold, export, serve/, monitoring/).
Scripts in `scripts/`. Tests in `tests/`. See docs/PRD.md for the full tree.

## ML rules (non-negotiable)

- Split by `Time`: 60% train, 20% validation, 20% test. Never shuffle across the boundary.
- `Class` is the label only; it never appears in features.
- The threshold is chosen on validation by cost. Test data is used once, for final reporting.
- Report PR-AUC as the primary metric. Never report accuracy as a headline number.
- Every training run logs params, metrics, artifacts and a data hash to MLflow.
- The API serves the exported artifact in artifacts/model/, never a live MLflow lookup.
- Feature order at serve time comes from metadata.json, never from dict ordering.

## Engineering rules

- Type hints everywhere; ruff and mypy must pass.
- Tests must run without the real dataset: use the synthetic fixture in tests/conftest.py.
- Never delete or weaken a test to make it pass. If a test looks wrong, stop and say so.
- Never commit data/, mlruns/, mlflow.db or reports/.
- Pin dependency versions in pyproject.toml.
- When a library API differs from what you expect, check the installed version first
  (`pip show <pkg>`), then use that version's API. Don't guess between old and new APIs.
- Keep functions small and pure where possible; I/O at the edges.
