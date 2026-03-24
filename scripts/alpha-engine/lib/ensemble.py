"""
Stacked Ensemble
================

Trains LightGBM + XGBoost base models, then a Logistic Regression
meta-learner on their out-of-fold predictions.

Saves the full ensemble as a single joblib artifact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

from lib.trainer import load_split_from_db

logger = logging.getLogger("ensemble")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class EnsembleConfig:
    seed: int = 42
    model_dir: str = ""
    lgbm_params: dict | None = None
    xgb_params: dict | None = None


# ---------------------------------------------------------------------------
# Ensemble model wrapper
# ---------------------------------------------------------------------------
class StackedEnsemble:
    """LightGBM + XGBoost → Logistic Regression meta-learner."""

    def __init__(self, lgbm: LGBMClassifier, xgb: XGBClassifier,
                 meta: LogisticRegression, feature_names: list[str]):
        self.lgbm = lgbm
        self.xgb = xgb
        self.meta = meta
        self.feature_names = feature_names

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(win) from the meta-learner."""
        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        meta_X = np.column_stack([p_lgbm, p_xgb])
        return self.meta.predict_proba(meta_X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class EnsembleBuilder:
    """Build stacked ensembles from the feature store."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()

    def build(
        self,
        db_path: str,
        symbol: str,
        dataset_id: int | None = None,
    ) -> dict:
        """Train ensemble for a symbol. Returns result dict."""
        logger.info("Building ensemble for %s...", symbol)

        X_train, y_train, feat_names, _ = load_split_from_db(
            db_path, symbol, "train", dataset_id,
        )
        X_val, y_val, _, _ = load_split_from_db(
            db_path, symbol, "val", dataset_id,
        )
        X_test, y_test, _, _ = load_split_from_db(
            db_path, symbol, "test", dataset_id,
        )

        if len(X_train) == 0 or len(X_val) == 0:
            logger.warning("No data for %s, skipping", symbol)
            return {"symbol": symbol, "status": "no_data"}

        # Replace NaN/Inf
        for arr in [X_train, X_val, X_test]:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        seed = self.config.seed

        # ── Base model 1: LightGBM ──────────────────────────────
        lgbm_params = self.config.lgbm_params or {
            "n_estimators": 500,
            "max_depth": 6,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": seed,
            "verbosity": -1,
        }
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train, y_train)
        logger.info("LightGBM trained")

        # ── Base model 2: XGBoost ────────────────────────────────
        xgb_params = self.config.xgb_params or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": seed,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        logger.info("XGBoost trained")

        # ── Meta-learner: fit on validation set predictions ──────
        p_lgbm_val = lgbm.predict_proba(X_val)[:, 1]
        p_xgb_val = xgb.predict_proba(X_val)[:, 1]
        meta_X_val = np.column_stack([p_lgbm_val, p_xgb_val])

        meta = LogisticRegression(random_state=seed, max_iter=1000)
        meta.fit(meta_X_val, y_val)
        logger.info("Meta-learner trained on validation predictions")

        ensemble = StackedEnsemble(lgbm, xgb, meta, feat_names)

        # ── Evaluate on test ─────────────────────────────────────
        y_proba = ensemble.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(np.int32)

        report_str = classification_report(
            y_test, y_pred, target_names=["not_win", "win"], zero_division=0,
        )
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        test_logloss = log_loss(y_test, y_proba)

        logger.info("Ensemble test: accuracy=%.4f, AUC=%.4f, logloss=%.4f",
                     accuracy, auc, test_logloss)

        # ── Save ─────────────────────────────────────────────────
        model_dir = Path(self.config.model_dir) if self.config.model_dir else (
            Path(__file__).resolve().parent.parent / "models"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        existing = list(model_dir.glob(f"{symbol}_ensemble_v*.joblib"))
        version = len(existing) + 1
        model_path = model_dir / f"{symbol}_ensemble_v{version}.joblib"

        artifact = {
            "ensemble": ensemble,
            "feature_names": feat_names,
            "symbol": symbol,
            "version": version,
            "lgbm_params": lgbm_params,
            "xgb_params": xgb_params,
            "test_accuracy": accuracy,
            "test_auc": auc,
            "test_logloss": test_logloss,
            "created_at": time.time(),
        }
        joblib.dump(artifact, model_path)
        logger.info("Ensemble saved: %s", model_path)

        return {
            "symbol": symbol,
            "status": "ok",
            "model_path": str(model_path),
            "version": version,
            "test_accuracy": accuracy,
            "test_auc": auc,
            "test_logloss": test_logloss,
            "classification_report": report_str,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }
