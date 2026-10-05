"""Train LR baseline + LightGBM, pick cost-based thresholds on validation, evaluate once on
test, log everything to MLflow, register the LightGBM champion and export it."""

import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from cardshield.config import (
    AMOUNT_COL,
    CHAMPION_ALIAS,
    LABEL_COL,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    RAW_CSV,
    REVIEW_COST,
)
from cardshield.data import FEATURE_COLS, Split, data_hash, features, load_raw, time_split
from cardshield.export import export_model
from cardshield.threshold import (
    CostResult,
    FloatArray,
    IntArray,
    best_amount_rule,
    best_threshold,
    evaluate_threshold,
)

EXPERIMENT = "cardshield"
SEED = 42
LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    # Chosen on validation PR-AUC. With scale_pos_weight ~470, lr 0.05 and short patience
    # stopped at 1-4 rounds with saturated scores; a lower lr, a capped leaf step
    # (max_delta_step) and L2 keep boosting stable.
    "learning_rate": 0.02,
    "max_delta_step": 1.0,
    "lambda_l2": 5.0,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "metric": "average_precision",
    "seed": SEED,
    "verbose": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 200


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, IntArray, FloatArray]:
    return (
        features(frame),
        frame[LABEL_COL].to_numpy(dtype=np.int64),
        frame[AMOUNT_COL].to_numpy(dtype=np.float64),
    )


def classification_metrics(y: IntArray, proba: FloatArray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(np.int64)
    return {
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def pr_curve_figure(y: IntArray, proba: FloatArray, title: str) -> Figure:
    precision, recall, _ = precision_recall_curve(y, proba)
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def confusion_figure(y: IntArray, proba: FloatArray, threshold: float, title: str) -> Figure:
    cm = confusion_matrix(y, (proba >= threshold).astype(np.int64), labels=[0, 1])
    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["legit", "fraud"],
        yticklabels=["legit", "fraud"],
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    fig.tight_layout()
    return fig


def fit_logreg(x: pd.DataFrame, y: IntArray) -> Pipeline:
    model = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    model.fit(x, y)
    return model


def predict_lgbm(booster: lgb.Booster, x: pd.DataFrame) -> FloatArray:
    return np.asarray(booster.predict(x), dtype=np.float64)


def fit_lgbm(x_tr: pd.DataFrame, y_tr: IntArray, x_va: pd.DataFrame, y_va: IntArray) -> lgb.Booster:
    params = {**LGBM_PARAMS, "scale_pos_weight": float((y_tr == 0).sum() / (y_tr == 1).sum())}
    train_set = lgb.Dataset(x_tr, y_tr)
    val_set = lgb.Dataset(x_va, y_va, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )


def cost_table(split: Split, val_proba: FloatArray, test_proba: FloatArray) -> pd.DataFrame:
    """Test-period cost of approve-all, best amount rule and model (both tuned on validation)."""
    _, y_va, amt_va = _xy(split.val)
    _, y_te, amt_te = _xy(split.test)
    rule_thr = best_amount_rule(y_va, amt_va).threshold
    model_thr = best_threshold(y_va, val_proba, amt_va).threshold
    rows: dict[str, CostResult] = {
        "approve_all": evaluate_threshold(y_te, test_proba, amt_te, float("inf")),
        "amount_rule": evaluate_threshold(y_te, amt_te, amt_te, rule_thr),
        "model": evaluate_threshold(y_te, test_proba, amt_te, model_thr),
    }
    table = pd.DataFrame({k: asdict(v) for k, v in rows.items()}).T
    for base in ("approve_all", "amount_rule"):
        base_cost = rows[base].cost
        table[f"saved_vs_{base}_pct"] = 100 * (1 - table["cost"].astype(float) / base_cost)
    return table


def run_model(
    name: str,
    split: Split,
    val_proba: FloatArray,
    test_proba: FloatArray,
    params: dict[str, Any],
    tags: dict[str, str],
) -> tuple[CostResult, dict[str, float]]:
    """Log one active MLflow run: threshold from validation, metrics on validation and test."""
    _, y_va, amt_va = _xy(split.val)
    _, y_te, amt_te = _xy(split.test)
    chosen = best_threshold(y_va, val_proba, amt_va)
    test_cost = evaluate_threshold(y_te, test_proba, amt_te, chosen.threshold)
    val_m = classification_metrics(y_va, val_proba, chosen.threshold)
    test_m = classification_metrics(y_te, test_proba, chosen.threshold)
    mlflow.set_tags(tags)
    mlflow.log_params({**params, "review_cost": REVIEW_COST, "features": ",".join(FEATURE_COLS)})
    mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
    mlflow.log_metrics(
        {
            "threshold": chosen.threshold,
            "val_cost": chosen.cost,
            "test_cost": test_cost.cost,
            "test_false_alarms": test_cost.false_alarms,
            "test_caught": test_cost.caught,
            "test_missed": test_cost.missed,
        }
    )
    mlflow.log_dict({"features": list(FEATURE_COLS)}, "features.json")
    mlflow.log_figure(pr_curve_figure(y_te, test_proba, f"{name} PR curve (test)"), "pr_curve.png")
    mlflow.log_figure(
        confusion_figure(y_te, test_proba, chosen.threshold, f"{name} @ {chosen.threshold:.3f}"),
        "confusion_matrix.png",
    )
    return chosen, test_m


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    split = time_split(load_raw(RAW_CSV))
    x_tr, y_tr, _ = _xy(split.train)
    x_va, y_va, _ = _xy(split.val)
    x_te, _, _ = _xy(split.test)
    tags = {
        "data_hash": data_hash(RAW_CSV),
        "train_rows": str(len(split.train)),
        "val_rows": str(len(split.val)),
        "test_rows": str(len(split.test)),
    }
    summary: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name="logreg"):
        lr = fit_logreg(x_tr, y_tr)
        val_p, test_p = lr.predict_proba(x_va)[:, 1], lr.predict_proba(x_te)[:, 1]
        chosen, test_m = run_model(
            "LogReg", split, val_p, test_p, {"model": "logreg", "class_weight": "balanced"}, tags
        )
        mlflow.sklearn.log_model(lr, name="model")
        summary["logreg"] = {**test_m, "threshold": chosen.threshold}

    with mlflow.start_run(run_name="lightgbm") as run:
        booster = fit_lgbm(x_tr, y_tr, x_va, y_va)
        val_p, test_p = predict_lgbm(booster, x_va), predict_lgbm(booster, x_te)
        params = {
            "model": "lightgbm",
            **LGBM_PARAMS,
            "best_iteration": booster.best_iteration,
            "scale_pos_weight": round(float((y_tr == 0).sum() / (y_tr == 1).sum()), 3),
        }
        chosen, test_m = run_model("LightGBM", split, val_p, test_p, params, tags)
        costs = cost_table(split, val_p, test_p)
        mlflow.log_table(costs.reset_index(names="policy"), "cost_table.json")
        mlflow.log_metrics({f"test_cost_{p}": float(c) for p, c in costs["cost"].items()})
        info = mlflow.lightgbm.log_model(booster, name="model", registered_model_name=MODEL_NAME)
        version = str(info.registered_model_version)
        MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
        summary["lightgbm"] = {**test_m, "threshold": chosen.threshold}

    export_model(
        booster,
        {
            "features": list(FEATURE_COLS),
            "threshold": chosen.threshold,
            "review_cost": REVIEW_COST,
            "model_name": MODEL_NAME,
            "model_version": version,
            "run_id": run.info.run_id,
            "data_hash": tags["data_hash"],
            "test_metrics": test_m,
            "test_cost_table": {p: float(c) for p, c in costs["cost"].items()},
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        MODEL_DIR,
    )

    pd.set_option("display.width", 120)
    print("\nTest metrics (threshold chosen on validation by cost):")
    print(pd.DataFrame(summary).T.round(4).to_string())
    print("\nTest-period cost by policy:")
    print(costs.round(2).to_string())
    print(f"\nRegistered {MODEL_NAME} v{version} as @{CHAMPION_ALIAS}; exported to {MODEL_DIR}")


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
