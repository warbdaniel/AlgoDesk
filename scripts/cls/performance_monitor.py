"""
Continuous Learning System - Performance Monitor
=================================================

Tracks model predictions against actual outcomes in real-time,
computes rolling performance metrics, and raises alerts when
performance degrades below configured thresholds.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from cls_store import CLSStore
from config import PerformanceConfig

logger = logging.getLogger("cls.performance")


@dataclass
class PerformanceAlert:
    """An alert raised when model performance crosses a threshold."""

    symbol: str
    model_id: int
    metric: str
    current_value: float
    threshold: float
    direction: str  # "below" or "above"
    severity: str  # "warning" or "critical"
    message: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "model_id": self.model_id,
            "metric": self.metric,
            "current_value": round(self.current_value, 4),
            "threshold": round(self.threshold, 4),
            "direction": self.direction,
            "severity": self.severity,
            "message": self.message,
            "ts": self.ts,
        }


class PerformanceMonitor:
    """Monitors model performance using prediction outcomes."""

    def __init__(self, store: CLSStore, config: PerformanceConfig | None = None):
        self._store = store
        self._cfg = config or PerformanceConfig()
        self._last_eval: dict[str, float] = {}  # symbol -> last eval timestamp
        self._alerts: list[PerformanceAlert] = []

    def evaluate(self, symbol: str, model_id: int) -> dict:
        """Evaluate model performance on recent resolved predictions.

        Returns a metrics dict and raises alerts if thresholds are breached.
        """
        # Get resolved predictions within rolling window
        predictions = self._store.get_recent_predictions(
            symbol=symbol,
            model_id=model_id,
            limit=self._cfg.rolling_window,
            resolved_only=True,
        )

        if len(predictions) < self._cfg.min_predictions:
            return {
                "symbol": symbol,
                "model_id": model_id,
                "status": "insufficient_data",
                "predictions_available": len(predictions),
                "min_required": self._cfg.min_predictions,
            }

        # Extract arrays
        y_pred = np.array([p["prediction"] for p in predictions])
        y_actual = np.array([p["actual_outcome"] for p in predictions])
        pnl_pips = np.array([
            p["pnl_pips"] if p["pnl_pips"] is not None else 0.0
            for p in predictions
        ])

        # Compute classification metrics
        y_pred_binary = (y_pred >= 0.5).astype(int)
        y_actual_binary = (y_actual >= 0.5).astype(int)

        accuracy = float(np.mean(y_pred_binary == y_actual_binary))
        auc = self._compute_auc(y_actual_binary, y_pred)
        logloss = self._compute_logloss(y_actual_binary, y_pred)

        # Compute trading metrics
        traded_mask = y_pred >= 0.5
        trade_pnls = pnl_pips[traded_mask]
        total_trades = int(traded_mask.sum())

        win_rate = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        total_pnl = 0.0

        if total_trades > 0:
            wins = trade_pnls[trade_pnls > 0]
            losses = trade_pnls[trade_pnls <= 0]
            win_rate = len(wins) / total_trades
            total_pnl = float(trade_pnls.sum())

            gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
            gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            if len(trade_pnls) > 1:
                mean_pnl = trade_pnls.mean()
                std_pnl = trade_pnls.std()
                if std_pnl > 0:
                    sharpe = float((mean_pnl / std_pnl) * math.sqrt(252 * 57))

        metrics = {
            "symbol": symbol,
            "model_id": model_id,
            "status": "ok",
            "window_size": len(predictions),
            "accuracy": round(accuracy, 4),
            "auc": round(auc, 4),
            "logloss": round(logloss, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "total_pnl_pips": round(total_pnl, 2),
            "total_predictions": len(predictions),
            "total_trades": total_trades,
        }

        # Save performance snapshot
        self._store.save_performance_snapshot(symbol, model_id, metrics)

        # Check thresholds and raise alerts
        alerts = self._check_thresholds(symbol, model_id, metrics)
        metrics["alerts"] = [a.to_dict() for a in alerts]
        metrics["degraded"] = len(alerts) > 0

        self._last_eval[symbol] = time.time()

        logger.info(
            "%s model %d: acc=%.3f auc=%.3f wr=%.3f pf=%.2f pnl=%.1f %s",
            symbol, model_id, accuracy, auc, win_rate, profit_factor,
            total_pnl, "DEGRADED" if alerts else "OK",
        )

        return metrics

    def should_evaluate(self, symbol: str) -> bool:
        """Check if enough time has passed since last evaluation."""
        last = self._last_eval.get(symbol, 0)
        return (time.time() - last) >= self._cfg.eval_interval_seconds

    def get_alerts(self, symbol: str = "") -> list[dict]:
        """Get recent performance alerts."""
        alerts = self._alerts
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        return [a.to_dict() for a in alerts[-50:]]

    def clear_alerts(self, symbol: str = ""):
        if symbol:
            self._alerts = [a for a in self._alerts if a.symbol != symbol]
        else:
            self._alerts.clear()

    # ── Threshold checks ─────────────────────────────────────────

    def _check_thresholds(
        self, symbol: str, model_id: int, metrics: dict,
    ) -> list[PerformanceAlert]:
        alerts: list[PerformanceAlert] = []

        checks = [
            ("auc", metrics["auc"], self._cfg.min_auc, "below", "AUC"),
            ("accuracy", metrics["accuracy"], self._cfg.min_accuracy, "below", "Accuracy"),
            ("logloss", metrics["logloss"], self._cfg.max_logloss, "above", "Log Loss"),
            ("win_rate", metrics["win_rate"], self._cfg.min_win_rate, "below", "Win Rate"),
            ("profit_factor", metrics["profit_factor"], self._cfg.min_profit_factor,
             "below", "Profit Factor"),
        ]

        for metric_name, value, threshold, direction, label in checks:
            breached = (
                (direction == "below" and value < threshold)
                or (direction == "above" and value > threshold)
            )
            if breached:
                severity = "critical" if self._is_critical(metric_name, value, threshold) else "warning"
                alert = PerformanceAlert(
                    symbol=symbol,
                    model_id=model_id,
                    metric=metric_name,
                    current_value=value,
                    threshold=threshold,
                    direction=direction,
                    severity=severity,
                    message=(
                        f"{label} {direction} threshold: "
                        f"{value:.4f} vs {threshold:.4f} for {symbol}"
                    ),
                )
                alerts.append(alert)
                self._alerts.append(alert)
                logger.warning("Performance alert: %s", alert.message)

        return alerts

    def _is_critical(self, metric: str, value: float, threshold: float) -> bool:
        """Determine if a threshold breach is critical (vs warning)."""
        # Critical if more than 10% below threshold
        if metric in ("auc", "accuracy", "win_rate", "profit_factor"):
            return value < threshold * 0.9
        if metric == "logloss":
            return value > threshold * 1.1
        return False

    # ── Metric helpers ───────────────────────────────────────────

    @staticmethod
    def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute AUC-ROC. Returns 0.5 if degenerate."""
        if len(np.unique(y_true)) < 2:
            return 0.5

        # Simple trapezoidal AUC (avoids sklearn dependency at runtime)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5

        auc = 0.0
        for p in pos:
            auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        auc /= len(pos) * len(neg)
        return float(auc)

    @staticmethod
    def _compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary log loss."""
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -(
            y_true * np.log(y_pred_clipped)
            + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss.mean())

    # ── Trend analysis ───────────────────────────────────────────

    def get_performance_trend(
        self, symbol: str, model_id: int | None = None, lookback: int = 20,
    ) -> dict:
        """Analyse the trend in performance snapshots."""
        snapshots = self._store.get_performance_history(
            symbol, model_id=model_id, limit=lookback,
        )
        if len(snapshots) < 3:
            return {"symbol": symbol, "status": "insufficient_data", "snapshots": len(snapshots)}

        # Reverse to chronological order
        snapshots = list(reversed(snapshots))

        aucs = [s["auc"] for s in snapshots]
        accuracies = [s["accuracy"] for s in snapshots]
        win_rates = [s["win_rate"] for s in snapshots]

        def _trend(values: list[float]) -> str:
            if len(values) < 3:
                return "stable"
            recent = np.mean(values[-3:])
            earlier = np.mean(values[:3])
            diff = recent - earlier
            if diff > 0.02:
                return "improving"
            elif diff < -0.02:
                return "declining"
            return "stable"

        return {
            "symbol": symbol,
            "status": "ok",
            "snapshots": len(snapshots),
            "auc_trend": _trend(aucs),
            "accuracy_trend": _trend(accuracies),
            "win_rate_trend": _trend(win_rates),
            "latest_auc": aucs[-1] if aucs else 0,
            "latest_accuracy": accuracies[-1] if accuracies else 0,
            "latest_win_rate": win_rates[-1] if win_rates else 0,
        }
