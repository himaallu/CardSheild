@AGENTS.md

## Claude Code

### How we work

- Work one sprint at a time from docs/IMPLEMENTATION_PLAN.md. Don't start the next sprint
  until I say so.
- At the start of a sprint, propose the approach and the files you'll touch, then wait for my OK.
- After every task: run `make test` and `make lint`, fix failures, then tick the checkbox.
- Commit at the end of each task with a clear message (`feat:`, `test:`, `fix:`, `docs:`).
- Each sprint has a time box. If something will clearly blow it (a library fight, a flaky
  install), stop and tell me with the fastest workaround instead of pushing on.

### Known traps in this project

- Evidently's API changed between versions (old `evidently.report` vs new `evidently.Report`
  with `evidently.presets`). Check the installed version and use its API; after one failed
  attempt, switch to the PSI fallback in monitoring/drift.py.
- MLflow model aliases need the registry: use `MlflowClient().set_registered_model_alias`.
- LightGBM SHAP: `predict(X, pred_contrib=True)` returns one extra column (the bias term)
  at the end; drop it before ranking reasons.
- Don't download the dataset in CI.

### When you finish a sprint

Summarise what you built, what you skipped, and what I should check by hand, then explain the
key design decisions briefly so I can defend them in an interview.
