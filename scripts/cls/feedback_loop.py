"""
Continuous Learning System - Feedback Loop
==========================================

Closes the loop between model predictions and real trade outcomes.
Polls the analytics dashboard for completed trades, matches them to
predictions, and updates the prediction log with actual outcomes.

This resolved data feeds into the PerformanceMonitor and DriftDetector.
"""

from __future__ import annotations

import logging
import time

import requests

from cls_store import CLSStore
from config import FeedbackConfig

logger = logging.getLogger("cls.feedback")


class FeedbackLoop:
    """Matches trade outcomes to model predictions."""

    def __init__(self, store: CLSStore, config: FeedbackConfig | None = None):
        self._store = store
        self._cfg = config or FeedbackConfig()
        self._last_poll: float = 0.0
        self._processed_trade_ids: set[str] = set()

    # ── Trade outcome polling ────────────────────────────────────

    def poll_trade_outcomes(self) -> dict:
        """Poll the dashboard for recent trade outcomes and match to predictions.

        Returns summary of processed trades.
        """
        try:
            resp = requests.get(
                f"{self._cfg.dashboard_url}/api/trades",
                params={"limit": self._cfg.batch_size},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Failed to poll dashboard: %s", e)
            return {"status": "error", "error": str(e)}

        trades = data if isinstance(data, list) else data.get("trades", [])
        if not trades:
            return {"status": "ok", "processed": 0, "matched": 0}

        processed = 0
        matched = 0

        for trade in trades:
            trade_id = trade.get("id", "")
            if not trade_id or trade_id in self._processed_trade_ids:
                continue

            symbol = trade.get("symbol", "")
            pnl_pips = trade.get("pnl_pips", 0)
            direction = trade.get("direction", "")
            regime = trade.get("regime", "")
            entry_time = trade.get("entry_time", "")

            # Try to match this trade to an unresolved prediction
            match = self._match_prediction(symbol, entry_time, direction)
            if match:
                # Resolve the prediction
                actual_outcome = 1.0 if pnl_pips > 0 else 0.0
                self._store.resolve_prediction(
                    prediction_id=match["id"],
                    actual_outcome=actual_outcome,
                    pnl_pips=pnl_pips,
                )

                # Save feedback record
                self._store.save_feedback(
                    prediction_id=match["id"],
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=direction,
                    entry_price=trade.get("entry_price", 0),
                    exit_price=trade.get("exit_price", 0),
                    pnl=trade.get("pnl", 0),
                    pnl_pips=pnl_pips,
                    regime=regime,
                )
                matched += 1

            self._processed_trade_ids.add(trade_id)
            processed += 1

        self._last_poll = time.time()

        logger.info("Feedback poll: %d trades processed, %d matched to predictions",
                     processed, matched)

        return {
            "status": "ok",
            "processed": processed,
            "matched": matched,
        }

    def should_poll(self) -> bool:
        """Check if enough time has passed since last poll."""
        return (time.time() - self._last_poll) >= self._cfg.poll_interval_seconds

    # ── Manual outcome recording ─────────────────────────────────

    def record_outcome(
        self,
        prediction_id: int,
        actual_outcome: float,
        pnl_pips: float | None = None,
        trade_id: str = "",
    ) -> dict:
        """Manually record an outcome for a prediction."""
        pred = self._store.get_recent_predictions(
            symbol="", limit=1,
        )
        # Verify prediction exists by trying to resolve it
        self._store.resolve_prediction(
            prediction_id=prediction_id,
            actual_outcome=actual_outcome,
            pnl_pips=pnl_pips,
        )

        if trade_id:
            self._processed_trade_ids.add(trade_id)

        logger.info("Recorded outcome for prediction %d: outcome=%.1f, pnl=%.1f",
                     prediction_id, actual_outcome, pnl_pips or 0)

        return {
            "status": "ok",
            "prediction_id": prediction_id,
            "actual_outcome": actual_outcome,
            "pnl_pips": pnl_pips,
        }

    # ── Batch outcome resolution ─────────────────────────────────

    def resolve_expired_predictions(
        self,
        max_age_seconds: float = 7200,  # 2 hours
    ) -> dict:
        """Mark old unresolved predictions as losses (timed out).

        Predictions that remain unresolved beyond max_age are assumed
        to have resulted in no trade (FLAT / no action).
        """
        unresolved = self._store.get_unresolved_predictions(limit=500)
        now = time.time()
        expired = 0

        for pred in unresolved:
            age = now - pred["ts"]
            if age > max_age_seconds:
                # Mark as no-outcome (0.0 = not a win)
                self._store.resolve_prediction(
                    prediction_id=pred["id"],
                    actual_outcome=0.0,
                    pnl_pips=0.0,
                )
                expired += 1

        if expired > 0:
            logger.info("Expired %d unresolved predictions (age > %ds)",
                         expired, max_age_seconds)

        return {"status": "ok", "expired": expired}

    # ── Feedback analysis ────────────────────────────────────────

    def get_feedback_summary(self, symbol: str = "") -> dict:
        """Summarise feedback data for analysis."""
        unprocessed = self._store.get_unprocessed_feedback(limit=1000)
        if symbol:
            unprocessed = [f for f in unprocessed if f["symbol"] == symbol]

        if not unprocessed:
            return {"status": "ok", "total": 0}

        total_pnl = sum(f["pnl_pips"] for f in unprocessed)
        wins = sum(1 for f in unprocessed if f["pnl_pips"] > 0)
        losses = sum(1 for f in unprocessed if f["pnl_pips"] <= 0)
        total = len(unprocessed)

        by_regime: dict[str, dict] = {}
        for f in unprocessed:
            regime = f.get("regime", "UNKNOWN") or "UNKNOWN"
            if regime not in by_regime:
                by_regime[regime] = {"count": 0, "pnl": 0, "wins": 0}
            by_regime[regime]["count"] += 1
            by_regime[regime]["pnl"] += f["pnl_pips"]
            if f["pnl_pips"] > 0:
                by_regime[regime]["wins"] += 1

        return {
            "status": "ok",
            "total": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total, 4) if total > 0 else 0,
            "total_pnl_pips": round(total_pnl, 2),
            "by_regime": by_regime,
        }

    # ── Internal helpers ─────────────────────────────────────────

    def _match_prediction(
        self,
        symbol: str,
        entry_time: str,
        direction: str,
    ) -> dict | None:
        """Find an unresolved prediction matching a trade.

        Matches by symbol and approximate timestamp (within 5 minutes).
        """
        unresolved = self._store.get_unresolved_predictions(symbol=symbol, limit=100)
        if not unresolved:
            return None

        # Parse entry_time to epoch
        entry_ts = self._parse_time(entry_time)
        if entry_ts <= 0:
            return None

        # Find closest prediction within 5 minutes
        best_match = None
        best_diff = float("inf")
        window = 300  # 5 minutes

        for pred in unresolved:
            diff = abs(pred["ts"] - entry_ts)
            if diff < window and diff < best_diff:
                best_match = pred
                best_diff = diff

        return best_match

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Parse a time string to epoch. Supports ISO format."""
        if not time_str:
            return 0.0
        try:
            from datetime import datetime, timezone
            # Try ISO format
            if "T" in time_str:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            else:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            return 0.0
