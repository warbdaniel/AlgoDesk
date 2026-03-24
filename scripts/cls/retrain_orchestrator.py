"""
Continuous Learning System - Retrain Orchestrator
==================================================

Coordinates automated model retraining in response to:
  - Performance degradation (from PerformanceMonitor)
  - Drift detection (from DriftDetector)
  - Regime changes
  - Scheduled periodic retraining

Integrates with the alpha-engine's training pipeline to produce
new model versions, then registers them as challengers in the
model registry.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import threading
from pathlib import Path

from cls_store import CLSStore
from config import RetrainConfig, ALPHA_ENGINE_DIR, MODELS_DIR
from model_registry import ModelRegistry

logger = logging.getLogger("cls.retrain")


class RetrainOrchestrator:
    """Orchestrates automated model retraining."""

    def __init__(
        self,
        store: CLSStore,
        registry: ModelRegistry,
        config: RetrainConfig | None = None,
    ):
        self._store = store
        self._registry = registry
        self._cfg = config or RetrainConfig()
        self._active_retrains: dict[str, int] = {}  # symbol -> retrain_id
        self._lock = threading.Lock()

    # ── Trigger evaluation ───────────────────────────────────────

    def should_retrain(
        self,
        symbol: str,
        performance_metrics: dict | None = None,
        drift_result: dict | None = None,
    ) -> tuple[bool, str]:
        """Evaluate whether a retrain should be triggered.

        Returns (should_retrain, reason).
        """
        # Check cooldown
        last_retrain = self._store.get_last_retrain(symbol)
        if last_retrain:
            elapsed = time.time() - last_retrain["started_at"]
            if elapsed < self._cfg.retrain_cooldown_seconds:
                return False, f"cooldown ({elapsed:.0f}s < {self._cfg.retrain_cooldown_seconds}s)"

        # Check if already retraining
        with self._lock:
            if symbol in self._active_retrains:
                return False, "retrain_already_running"

        # Check concurrent retrain limit
        with self._lock:
            if len(self._active_retrains) >= self._cfg.max_concurrent_retrains:
                return False, "max_concurrent_retrains_reached"

        # Performance degradation trigger
        if self._cfg.trigger_on_performance_degradation and performance_metrics:
            if performance_metrics.get("degraded"):
                alerts = performance_metrics.get("alerts", [])
                critical = [a for a in alerts if a.get("severity") == "critical"]
                if critical:
                    return True, "critical_performance_degradation"
                if len(alerts) >= 2:
                    return True, "multiple_performance_alerts"

        # Drift trigger
        if self._cfg.trigger_on_drift and drift_result:
            if drift_result.get("any_drift_detected"):
                severity = drift_result.get("overall_severity", "none")
                if severity in ("high", "critical"):
                    return True, f"drift_detected_{severity}"

        # Periodic retrain check
        if self._cfg.periodic_retrain_interval > 0:
            if last_retrain:
                elapsed = time.time() - last_retrain["started_at"]
                if elapsed >= self._cfg.periodic_retrain_interval:
                    return True, "periodic_schedule"
            else:
                # No retrain history - trigger first retrain
                return True, "initial_retrain"

        return False, "no_trigger"

    # ── Retrain execution ────────────────────────────────────────

    def trigger_retrain(
        self,
        symbol: str,
        reason: str,
        trigger_details: dict | None = None,
        async_mode: bool = True,
    ) -> dict:
        """Trigger a model retrain for a symbol.

        If async_mode=True, runs retraining in a background thread.
        """
        # Get current champion
        champion = self._registry.get_champion(symbol)
        old_model_id = champion["id"] if champion else None

        # Record retrain start
        retrain_id = self._store.start_retrain(
            symbol=symbol,
            trigger_reason=reason,
            old_model_id=old_model_id,
            trigger_details=trigger_details,
        )

        with self._lock:
            self._active_retrains[symbol] = retrain_id

        logger.info("Retrain triggered for %s (reason=%s, retrain_id=%d)",
                     symbol, reason, retrain_id)

        if async_mode:
            thread = threading.Thread(
                target=self._run_retrain,
                args=(symbol, retrain_id, old_model_id),
                daemon=True,
                name=f"retrain-{symbol}",
            )
            thread.start()
            return {
                "status": "started",
                "retrain_id": retrain_id,
                "symbol": symbol,
                "reason": reason,
                "async": True,
            }
        else:
            result = self._run_retrain(symbol, retrain_id, old_model_id)
            return result

    def _run_retrain(
        self,
        symbol: str,
        retrain_id: int,
        old_model_id: int | None,
    ) -> dict:
        """Execute the retraining pipeline."""
        try:
            logger.info("Starting retrain for %s (retrain_id=%d)", symbol, retrain_id)

            # Add alpha-engine to path for imports
            engine_dir = str(ALPHA_ENGINE_DIR)
            if engine_dir not in sys.path:
                sys.path.insert(0, engine_dir)

            from lib.trainer import WalkForwardTrainer, TrainerConfig

            # Configure trainer
            trainer_cfg = TrainerConfig(
                n_trials=self._cfg.retrain_n_trials,
                model_dir=str(MODELS_DIR),
            )
            trainer = WalkForwardTrainer(config=trainer_cfg)

            # Run training
            db_path = self._cfg.candle_ml_db
            train_result = trainer.train(db_path, symbol)

            if train_result.get("status") != "ok":
                self._store.complete_retrain(
                    retrain_id, "failed", result=train_result,
                )
                logger.error("Retrain failed for %s: %s", symbol, train_result)
                return {"status": "failed", "result": train_result}

            # Register the new model as a challenger
            new_model_id = self._store.register_model(
                symbol=symbol,
                model_type="lgbm",
                version=train_result["version"],
                model_path=train_result["model_path"],
                status="challenger",
                train_metrics={
                    "auc": train_result.get("test_auc", 0),
                    "accuracy": train_result.get("test_accuracy", 0),
                    "logloss": train_result.get("test_logloss", 0),
                    "train_size": train_result.get("train_size", 0),
                    "val_size": train_result.get("val_size", 0),
                    "test_size": train_result.get("test_size", 0),
                },
                feature_names=None,  # Stored in artifact
                hyperparameters=train_result.get("best_params"),
                metadata={
                    "retrain_id": retrain_id,
                    "trained_at": time.time(),
                },
            )

            # Auto-promote if configured and challenger is better
            promoted = False
            if self._cfg.auto_promote and old_model_id:
                old_model = self._store.get_model(old_model_id)
                if old_model:
                    eval_result = self._registry.evaluate_challenger(
                        champion_id=old_model_id,
                        challenger_id=new_model_id,
                        champion_metrics={
                            "auc": old_model["train_auc"],
                            "accuracy": old_model["train_accuracy"],
                        },
                        challenger_metrics={
                            "auc": train_result.get("test_auc", 0),
                            "accuracy": train_result.get("test_accuracy", 0),
                        },
                    )
                    if eval_result["should_promote"]:
                        self._registry.promote_challenger(new_model_id)
                        promoted = True
                        logger.info("Auto-promoted new model %d for %s",
                                     new_model_id, symbol)
            elif not old_model_id:
                # No previous champion - auto-promote
                self._registry.promote_challenger(new_model_id)
                promoted = True

            # Complete retrain record
            self._store.complete_retrain(
                retrain_id, "completed",
                new_model_id=new_model_id,
                result={
                    "train_result": {
                        k: v for k, v in train_result.items()
                        if k not in ("classification_report", "feature_importance", "calibration")
                    },
                    "new_model_id": new_model_id,
                    "promoted": promoted,
                },
            )

            logger.info(
                "Retrain completed for %s: new model %d (AUC=%.4f, promoted=%s)",
                symbol, new_model_id, train_result.get("test_auc", 0), promoted,
            )

            return {
                "status": "completed",
                "retrain_id": retrain_id,
                "new_model_id": new_model_id,
                "test_auc": train_result.get("test_auc", 0),
                "test_accuracy": train_result.get("test_accuracy", 0),
                "promoted": promoted,
            }

        except Exception as e:
            logger.error("Retrain error for %s: %s", symbol, e, exc_info=True)
            self._store.complete_retrain(
                retrain_id, "failed",
                result={"error": str(e)},
            )
            return {"status": "failed", "error": str(e)}

        finally:
            with self._lock:
                self._active_retrains.pop(symbol, None)

    # ── Status ───────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get orchestrator status."""
        with self._lock:
            active = dict(self._active_retrains)

        recent = self._store.get_retrain_history(limit=10)

        return {
            "active_retrains": active,
            "max_concurrent": self._cfg.max_concurrent_retrains,
            "cooldown_seconds": self._cfg.retrain_cooldown_seconds,
            "auto_promote": self._cfg.auto_promote,
            "periodic_interval": self._cfg.periodic_retrain_interval,
            "recent_history": [
                {
                    "id": r["id"],
                    "symbol": r["symbol"],
                    "trigger_reason": r["trigger_reason"],
                    "status": r["status"],
                    "started_at": r["started_at"],
                    "completed_at": r["completed_at"],
                }
                for r in recent
            ],
        }

    def cancel_retrain(self, symbol: str) -> dict:
        """Cancel an active retrain (best-effort, cannot interrupt training)."""
        with self._lock:
            retrain_id = self._active_retrains.pop(symbol, None)

        if retrain_id:
            self._store.complete_retrain(retrain_id, "cancelled")
            logger.info("Retrain cancelled for %s (retrain_id=%d)", symbol, retrain_id)
            return {"status": "cancelled", "retrain_id": retrain_id}

        return {"status": "no_active_retrain", "symbol": symbol}
