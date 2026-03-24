"""
Walk-Forward Model Trainer
==========================

Loads pre-split data from candle_dataset_samples, trains LightGBM binary
classifiers with Optuna hyperparameter search, and saves the best model.

Label mapping: barrier_label 1 (TP hit) → win=1, else → win=0.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger("trainer")

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    n_trials: int = 50
    seed: int = 42
    label_col: str = "barrier_label"
    model_dir: str = ""
    early_stopping_rounds: int = 50


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split_from_db(
    db_path: str,
    symbol: str,
    split_name: str,
    dataset_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load features/labels from candle_dataset_samples.

    Returns (X, y_binary, feature_names, timestamps).
    y_binary: 1 if barrier_label == 1 (TP), else 0.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row

    if dataset_id is not None:
        rows = conn.execute(
            "SELECT ts, features, labels FROM candle_dataset_samples "
            "WHERE dataset_id=? AND split_name=? AND symbol=? ORDER BY ts",
            (dataset_id, split_name, symbol),
        ).fetchall()
    else:
        # Use latest dataset for this symbol
        ds_row = conn.execute(
            "SELECT id FROM candle_datasets WHERE symbol=? "
            "ORDER BY created_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if not ds_row:
            conn.close()
            return np.array([]), np.array([]), [], np.array([])
        did = ds_row["id"]
        rows = conn.execute(
            "SELECT ts, features, labels FROM candle_dataset_samples "
            "WHERE dataset_id=? AND split_name=? AND symbol=? ORDER BY ts",
            (did, split_name, symbol),
        ).fetchall()

    conn.close()

    if not rows:
        return np.array([]), np.array([]), [], np.array([])

    # Parse JSON
    feature_dicts = [json.loads(r["features"]) for r in rows]
    label_dicts = [json.loads(r["labels"]) for r in rows]
    timestamps = np.array([r["ts"] for r in rows])

    feature_names = sorted(feature_dicts[0].keys())
    X = np.array([[fd[k] for k in feature_names] for fd in feature_dicts],
                 dtype=np.float64)

    # Binary label: 1 = win (barrier_label == 1.0), 0 = not win
    y = np.array(
        [1 if ld.get("barrier_label", 0.0) == 1.0 else 0 for ld in label_dicts],
        dtype=np.int32,
    )

    return X, y, feature_names, timestamps


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def _create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": seed,
            "verbosity": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                _lgbm_early_stopping(50),
                _lgbm_log_evaluation(-1),
            ],
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, y_pred_proba)

    return objective


def _lgbm_early_stopping(rounds):
    """Create early stopping callback compatible with LightGBM."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=rounds, verbose=False)


def _lgbm_log_evaluation(period):
    """Create log evaluation callback."""
    from lightgbm import log_evaluation
    return log_evaluation(period=period)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class WalkForwardTrainer:
    """Train LightGBM binary classifiers with Optuna HPO."""

    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()

    def train(
        self,
        db_path: str,
        symbol: str,
        dataset_id: int | None = None,
    ) -> dict:
        """Train a model for a single symbol. Returns result dict."""
        logger.info("Loading data for %s...", symbol)

        X_train, y_train, feat_names, ts_train = load_split_from_db(
            db_path, symbol, "train", dataset_id,
        )
        X_val, y_val, _, ts_val = load_split_from_db(
            db_path, symbol, "val", dataset_id,
        )
        X_test, y_test, _, ts_test = load_split_from_db(
            db_path, symbol, "test", dataset_id,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            logger.warning("No data for %s, skipping", symbol)
            return {"symbol": symbol, "status": "no_data"}

        logger.info(
            "%s: train=%d, val=%d, test=%d, features=%d",
            symbol, len(X_train), len(X_val), len(X_test), len(feat_names),
        )

        # Replace NaN/Inf
        for arr in [X_train, X_val, X_test]:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Optuna HPO ───────────────────────────────────────────
        logger.info("Running Optuna (%d trials)...", self.config.n_trials)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            _create_objective(X_train, y_train, X_val, y_val, self.config.seed),
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_params["random_state"] = self.config.seed
        best_params["verbosity"] = -1
        logger.info("Best params: %s (loss=%.5f)", best_params, study.best_value)

        # ── Retrain on train+val with best params ────────────────
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        best_model = LGBMClassifier(**best_params)
        best_model.fit(X_trainval, y_trainval)

        # ── Evaluate on test ─────────────────────────────────────
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        report_str = classification_report(
            y_test, y_pred, target_names=["not_win", "win"], zero_division=0,
        )
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        test_logloss = log_loss(y_test, y_proba)

        logger.info("Test accuracy=%.4f, AUC=%.4f, logloss=%.4f",
                     accuracy, auc, test_logloss)
        logger.info("\n%s", report_str)

        # ── Feature importance ───────────────────────────────────
        importances = best_model.feature_importances_
        feat_importance = sorted(
            zip(feat_names, importances.tolist()),
            key=lambda x: x[1], reverse=True,
        )

        # ── Calibration data ─────────────────────────────────────
        calibration = _compute_calibration(y_test, y_proba, n_bins=10)

        # ── Save model ───────────────────────────────────────────
        model_dir = Path(self.config.model_dir) if self.config.model_dir else (
            Path(__file__).resolve().parent.parent / "models"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine version
        existing = list(model_dir.glob(f"{symbol}_lgbm_v*.joblib"))
        version = len(existing) + 1
        model_path = model_dir / f"{symbol}_lgbm_v{version}.joblib"

        artifact = {
            "model": best_model,
            "feature_names": feat_names,
            "best_params": best_params,
            "symbol": symbol,
            "version": version,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "test_accuracy": accuracy,
            "test_auc": auc,
            "test_logloss": test_logloss,
            "created_at": time.time(),
        }
        joblib.dump(artifact, model_path)
        logger.info("Model saved: %s", model_path)

        return {
            "symbol": symbol,
            "status": "ok",
            "model_path": str(model_path),
            "version": version,
            "best_params": best_params,
            "best_val_loss": study.best_value,
            "test_accuracy": accuracy,
            "test_auc": auc,
            "test_logloss": test_logloss,
            "classification_report": report_str,
            "feature_importance": feat_importance[:20],
            "calibration": calibration,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------
def _compute_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10,
) -> list[dict]:
    """Compute calibration curve data (predicted vs actual probability)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    result = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi)
        count = int(mask.sum())
        if count > 0:
            avg_pred = float(y_proba[mask].mean())
            avg_true = float(y_true[mask].mean())
        else:
            avg_pred = (lo + hi) / 2
            avg_true = 0.0
        result.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "count": count,
            "avg_predicted": round(avg_pred, 4),
            "avg_actual": round(avg_true, 4),
        })
    return result
