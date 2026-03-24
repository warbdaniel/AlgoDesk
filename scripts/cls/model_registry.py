"""
Continuous Learning System - Model Registry
============================================

Manages model lifecycle: registration, promotion (champion/challenger),
retirement, and artifact discovery. Integrates with the alpha-engine's
model storage format (joblib artifacts).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib

from cls_store import CLSStore
from config import RegistryConfig

logger = logging.getLogger("cls.registry")


class ModelRegistry:
    """Manages model versions with champion/challenger paradigm."""

    def __init__(self, store: CLSStore, config: RegistryConfig | None = None):
        self._store = store
        self._cfg = config or RegistryConfig()
        self._loaded_models: dict[int, object] = {}  # model_id -> loaded model

    # ── Discovery ────────────────────────────────────────────────

    def discover_models(self) -> list[dict]:
        """Scan the models directory and register any untracked models.

        Looks for alpha-engine model artifacts ({SYMBOL}_lgbm_v{N}.joblib
        and {SYMBOL}_ensemble_v{N}.joblib) and registers them if not
        already in the registry.
        """
        models_dir = Path(self._cfg.models_dir)
        if not models_dir.exists():
            logger.warning("Models directory not found: %s", models_dir)
            return []

        registered = []

        for pattern, model_type in [
            ("*_lgbm_v*.joblib", "lgbm"),
            ("*_ensemble_v*.joblib", "ensemble"),
        ]:
            for path in sorted(models_dir.glob(pattern)):
                try:
                    artifact = joblib.load(path)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)
                    continue

                symbol = artifact.get("symbol", "")
                version = artifact.get("version", 0)
                if not symbol or not version:
                    continue

                # Check if already registered
                existing = self._store.list_models(symbol=symbol)
                already_registered = any(
                    m["model_type"] == model_type and m["version"] == version
                    for m in existing
                )
                if already_registered:
                    continue

                # Register it
                train_metrics = {
                    "auc": artifact.get("test_auc", 0),
                    "accuracy": artifact.get("test_accuracy", 0),
                    "logloss": artifact.get("test_logloss", 0),
                    "train_size": artifact.get("train_size", 0),
                    "val_size": artifact.get("val_size", 0),
                    "test_size": artifact.get("test_size", 0),
                }
                feature_names = artifact.get("feature_names", [])
                hyperparams = artifact.get("best_params", {})

                model_id = self._store.register_model(
                    symbol=symbol,
                    model_type=model_type,
                    version=version,
                    model_path=str(path),
                    status="challenger",
                    train_metrics=train_metrics,
                    feature_names=feature_names,
                    hyperparameters=hyperparams,
                    metadata={"discovered": True, "created_at": artifact.get("created_at", 0)},
                )
                registered.append({
                    "model_id": model_id,
                    "symbol": symbol,
                    "model_type": model_type,
                    "version": version,
                    "path": str(path),
                })
                logger.info("Discovered and registered: %s/%s v%d",
                            symbol, model_type, version)

        return registered

    # ── Model loading ────────────────────────────────────────────

    def load_model(self, model_id: int) -> object | None:
        """Load a model artifact into memory. Returns the model object."""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        entry = self._store.get_model(model_id)
        if not entry:
            logger.warning("Model %d not found in registry", model_id)
            return None

        model_path = entry["model_path"]
        if not Path(model_path).exists():
            logger.error("Model file missing: %s", model_path)
            return None

        try:
            artifact = joblib.load(model_path)
            # Extract the actual model object
            if "ensemble" in artifact:
                model = artifact["ensemble"]
            elif "model" in artifact:
                model = artifact["model"]
            else:
                model = artifact

            self._loaded_models[model_id] = model
            logger.info("Loaded model %d from %s", model_id, model_path)
            return model
        except Exception as e:
            logger.error("Failed to load model %d: %s", model_id, e)
            return None

    def unload_model(self, model_id: int):
        """Remove a model from memory cache."""
        self._loaded_models.pop(model_id, None)

    # ── Champion management ──────────────────────────────────────

    def get_champion(self, symbol: str, model_type: str = "lgbm") -> dict | None:
        """Get the current champion model for a symbol."""
        return self._store.get_champion(symbol, model_type)

    def get_or_elect_champion(self, symbol: str, model_type: str = "lgbm") -> dict | None:
        """Get the champion, or elect the best challenger if none exists."""
        champion = self._store.get_champion(symbol, model_type)
        if champion:
            return champion

        # No champion - elect the best challenger
        challengers = self._store.get_challengers(symbol, model_type)
        if not challengers:
            return None

        # Pick challenger with best AUC
        best = max(challengers, key=lambda c: c.get("train_auc", 0))
        self._store.promote_model(best["id"])
        logger.info("Auto-elected champion for %s/%s: model %d (AUC=%.4f)",
                     symbol, model_type, best["id"], best.get("train_auc", 0))
        return self._store.get_model(best["id"])

    def evaluate_challenger(
        self,
        champion_id: int,
        challenger_id: int,
        champion_metrics: dict,
        challenger_metrics: dict,
    ) -> dict:
        """Compare champion vs challenger. Returns evaluation result."""
        c_auc = champion_metrics.get("auc", 0)
        ch_auc = challenger_metrics.get("auc", 0)
        c_acc = champion_metrics.get("accuracy", 0)
        ch_acc = challenger_metrics.get("accuracy", 0)

        auc_improvement = ch_auc - c_auc
        acc_improvement = ch_acc - c_acc

        should_promote = (
            auc_improvement >= self._cfg.promotion_auc_margin
            and acc_improvement >= -self._cfg.promotion_accuracy_margin
        )

        result = {
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "champion_auc": c_auc,
            "challenger_auc": ch_auc,
            "auc_improvement": round(auc_improvement, 6),
            "champion_accuracy": c_acc,
            "challenger_accuracy": ch_acc,
            "accuracy_improvement": round(acc_improvement, 6),
            "should_promote": should_promote,
            "reason": (
                f"Challenger AUC +{auc_improvement:.4f} "
                f"(threshold: {self._cfg.promotion_auc_margin})"
            ),
        }

        logger.info(
            "Challenger evaluation: %s (AUC %.4f vs %.4f, promote=%s)",
            "PROMOTE" if should_promote else "KEEP",
            ch_auc, c_auc, should_promote,
        )
        return result

    def promote_challenger(self, model_id: int) -> bool:
        """Promote a challenger to champion."""
        return self._store.promote_model(model_id)

    # ── Pruning ──────────────────────────────────────────────────

    def prune_old_models(self, symbol: str, model_type: str = "lgbm"):
        """Retire old models beyond the retention limit."""
        models = self._store.list_models(symbol=symbol, status="retired")
        models = [m for m in models if m["model_type"] == model_type]

        if len(models) <= self._cfg.max_versions_per_symbol:
            return

        # Sort by retired_at, keep most recent
        models.sort(key=lambda m: m.get("retired_at", 0))
        to_remove = models[:-self._cfg.max_versions_per_symbol]

        for m in to_remove:
            self.unload_model(m["id"])
            model_path = Path(m["model_path"])
            if model_path.exists():
                model_path.unlink()
                logger.info("Pruned model artifact: %s", model_path)

    # ── Status ───────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get registry status summary."""
        all_models = self._store.list_models()
        symbols = set(m["symbol"] for m in all_models)

        per_symbol = {}
        for sym in symbols:
            champion = self._store.get_champion(sym)
            challengers = self._store.get_challengers(sym)
            per_symbol[sym] = {
                "champion": {
                    "id": champion["id"],
                    "version": champion["version"],
                    "model_type": champion["model_type"],
                    "auc": champion["train_auc"],
                } if champion else None,
                "challengers": len(challengers),
            }

        return {
            "total_models": len(all_models),
            "symbols": per_symbol,
            "loaded_models": len(self._loaded_models),
        }
