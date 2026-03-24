"""
Continuous Learning System - Persistence Layer
===============================================

SQLite WAL-mode database for CLS state: model registry entries,
prediction logs, drift snapshots, retrain history, and feedback records.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("cls.store")


class CLSStore:
    """Thread-safe SQLite store for the Continuous Learning System."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ── Connection management ────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Schema ───────────────────────────────────────────────────

    def _init_schema(self):
        conn = self._conn()
        conn.executescript("""
            -- Model registry: tracks all model versions and their status
            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL DEFAULT 'lgbm',
                version INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'challenger'
                    CHECK(status IN ('champion', 'challenger', 'retired', 'failed')),
                train_auc REAL DEFAULT 0,
                train_accuracy REAL DEFAULT 0,
                train_logloss REAL DEFAULT 0,
                train_size INTEGER DEFAULT 0,
                val_size INTEGER DEFAULT 0,
                test_size INTEGER DEFAULT 0,
                feature_names TEXT DEFAULT '[]',
                hyperparameters TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                promoted_at REAL DEFAULT 0,
                retired_at REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_registry_symbol
                ON model_registry(symbol, status);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_registry_symbol_version
                ON model_registry(symbol, model_type, version);

            -- Prediction log: records every prediction for performance tracking
            CREATE TABLE IF NOT EXISTS prediction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                prediction REAL NOT NULL,
                confidence REAL DEFAULT 0,
                features TEXT DEFAULT '{}',
                actual_outcome REAL DEFAULT NULL,
                outcome_ts REAL DEFAULT NULL,
                pnl_pips REAL DEFAULT NULL,
                regime TEXT DEFAULT '',
                FOREIGN KEY (model_id) REFERENCES model_registry(id)
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts
                ON prediction_log(symbol, ts);
            CREATE INDEX IF NOT EXISTS idx_predictions_model
                ON prediction_log(model_id, ts);
            CREATE INDEX IF NOT EXISTS idx_predictions_unresolved
                ON prediction_log(actual_outcome) WHERE actual_outcome IS NULL;

            -- Drift snapshots: periodic drift detection results
            CREATE TABLE IF NOT EXISTS drift_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                drift_type TEXT NOT NULL
                    CHECK(drift_type IN ('feature', 'concept', 'performance')),
                drift_detected INTEGER NOT NULL DEFAULT 0,
                severity TEXT DEFAULT 'none'
                    CHECK(severity IN ('none', 'low', 'moderate', 'high', 'critical')),
                details TEXT DEFAULT '{}',
                FOREIGN KEY (model_id) REFERENCES model_registry(id)
            );
            CREATE INDEX IF NOT EXISTS idx_drift_symbol_ts
                ON drift_snapshots(symbol, ts);

            -- Performance snapshots: periodic model performance metrics
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                window_size INTEGER NOT NULL,
                accuracy REAL DEFAULT 0,
                auc REAL DEFAULT 0,
                logloss REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                total_pnl_pips REAL DEFAULT 0,
                total_predictions INTEGER DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                details TEXT DEFAULT '{}',
                FOREIGN KEY (model_id) REFERENCES model_registry(id)
            );
            CREATE INDEX IF NOT EXISTS idx_perf_symbol_ts
                ON performance_snapshots(symbol, ts);

            -- Retrain history: tracks all retraining events
            CREATE TABLE IF NOT EXISTS retrain_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trigger_reason TEXT NOT NULL,
                trigger_details TEXT DEFAULT '{}',
                started_at REAL NOT NULL,
                completed_at REAL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'running'
                    CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
                old_model_id INTEGER DEFAULT NULL,
                new_model_id INTEGER DEFAULT NULL,
                result TEXT DEFAULT '{}',
                FOREIGN KEY (old_model_id) REFERENCES model_registry(id),
                FOREIGN KEY (new_model_id) REFERENCES model_registry(id)
            );
            CREATE INDEX IF NOT EXISTS idx_retrain_symbol
                ON retrain_history(symbol, started_at);

            -- Feedback records: trade outcomes linked back to predictions
            CREATE TABLE IF NOT EXISTS feedback_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pips REAL DEFAULT 0,
                regime TEXT DEFAULT '',
                ts REAL NOT NULL,
                processed INTEGER DEFAULT 0,
                FOREIGN KEY (prediction_id) REFERENCES prediction_log(id)
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_symbol
                ON feedback_records(symbol, ts);
            CREATE INDEX IF NOT EXISTS idx_feedback_unprocessed
                ON feedback_records(processed) WHERE processed = 0;
        """)
        conn.commit()
        logger.info("CLS store initialised: %s", self._db_path)

    # ── Model registry operations ────────────────────────────────

    def register_model(
        self,
        symbol: str,
        model_type: str,
        version: int,
        model_path: str,
        status: str = "challenger",
        train_metrics: dict | None = None,
        feature_names: list[str] | None = None,
        hyperparameters: dict | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Register a new model version. Returns the model ID."""
        m = train_metrics or {}
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO model_registry
               (symbol, model_type, version, model_path, status,
                train_auc, train_accuracy, train_logloss,
                train_size, val_size, test_size,
                feature_names, hyperparameters, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol, model_type, version, model_path, status,
                m.get("auc", 0), m.get("accuracy", 0), m.get("logloss", 0),
                m.get("train_size", 0), m.get("val_size", 0), m.get("test_size", 0),
                json.dumps(feature_names or []),
                json.dumps(hyperparameters or {}),
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        model_id = cur.lastrowid
        logger.info("Registered model %s/%s v%d (id=%d, status=%s)",
                     symbol, model_type, version, model_id, status)
        return model_id

    def get_champion(self, symbol: str, model_type: str = "lgbm") -> dict | None:
        """Get the current champion model for a symbol."""
        row = self._conn().execute(
            "SELECT * FROM model_registry WHERE symbol=? AND model_type=? "
            "AND status='champion' ORDER BY promoted_at DESC LIMIT 1",
            (symbol, model_type),
        ).fetchone()
        return dict(row) if row else None

    def get_challengers(self, symbol: str, model_type: str = "lgbm") -> list[dict]:
        """Get all challenger models for a symbol."""
        rows = self._conn().execute(
            "SELECT * FROM model_registry WHERE symbol=? AND model_type=? "
            "AND status='challenger' ORDER BY created_at DESC",
            (symbol, model_type),
        ).fetchall()
        return [dict(r) for r in rows]

    def promote_model(self, model_id: int) -> bool:
        """Promote a challenger to champion, retiring the old champion."""
        conn = self._conn()
        row = conn.execute(
            "SELECT symbol, model_type FROM model_registry WHERE id=?",
            (model_id,),
        ).fetchone()
        if not row:
            return False

        now = time.time()
        # Retire current champion
        conn.execute(
            "UPDATE model_registry SET status='retired', retired_at=? "
            "WHERE symbol=? AND model_type=? AND status='champion'",
            (now, row["symbol"], row["model_type"]),
        )
        # Promote new champion
        conn.execute(
            "UPDATE model_registry SET status='champion', promoted_at=? WHERE id=?",
            (now, model_id),
        )
        conn.commit()
        logger.info("Promoted model %d to champion for %s/%s",
                     model_id, row["symbol"], row["model_type"])
        return True

    def retire_model(self, model_id: int):
        """Retire a model."""
        conn = self._conn()
        conn.execute(
            "UPDATE model_registry SET status='retired', retired_at=? WHERE id=?",
            (time.time(), model_id),
        )
        conn.commit()

    def get_model(self, model_id: int) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM model_registry WHERE id=?", (model_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_models(self, symbol: str = "", status: str = "") -> list[dict]:
        """List models with optional symbol/status filter."""
        sql = "SELECT * FROM model_registry WHERE 1=1"
        params: list = []
        if symbol:
            sql += " AND symbol=?"
            params.append(symbol)
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Prediction log operations ────────────────────────────────

    def log_prediction(
        self,
        symbol: str,
        model_id: int,
        prediction: float,
        confidence: float = 0.0,
        features: dict | None = None,
        regime: str = "",
    ) -> int:
        """Log a model prediction. Returns prediction ID."""
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO prediction_log
               (symbol, model_id, ts, prediction, confidence, features, regime)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, model_id, time.time(), prediction, confidence,
             json.dumps(features or {}), regime),
        )
        conn.commit()
        return cur.lastrowid

    def resolve_prediction(
        self,
        prediction_id: int,
        actual_outcome: float,
        pnl_pips: float | None = None,
    ):
        """Update a prediction with the actual outcome."""
        conn = self._conn()
        conn.execute(
            "UPDATE prediction_log SET actual_outcome=?, outcome_ts=?, pnl_pips=? "
            "WHERE id=?",
            (actual_outcome, time.time(), pnl_pips, prediction_id),
        )
        conn.commit()

    def get_recent_predictions(
        self,
        symbol: str,
        model_id: int | None = None,
        limit: int = 200,
        resolved_only: bool = False,
    ) -> list[dict]:
        """Get recent predictions for a symbol."""
        sql = "SELECT * FROM prediction_log WHERE symbol=?"
        params: list = [symbol]
        if model_id is not None:
            sql += " AND model_id=?"
            params.append(model_id)
        if resolved_only:
            sql += " AND actual_outcome IS NOT NULL"
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_unresolved_predictions(
        self, symbol: str = "", limit: int = 500,
    ) -> list[dict]:
        """Get predictions that haven't been resolved with outcomes."""
        sql = "SELECT * FROM prediction_log WHERE actual_outcome IS NULL"
        params: list = []
        if symbol:
            sql += " AND symbol=?"
            params.append(symbol)
        sql += " ORDER BY ts ASC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Drift snapshot operations ────────────────────────────────

    def save_drift_snapshot(
        self,
        symbol: str,
        model_id: int,
        drift_type: str,
        drift_detected: bool,
        severity: str = "none",
        details: dict | None = None,
    ) -> int:
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO drift_snapshots
               (symbol, model_id, ts, drift_type, drift_detected, severity, details)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, model_id, time.time(), drift_type,
             1 if drift_detected else 0, severity, json.dumps(details or {})),
        )
        conn.commit()
        return cur.lastrowid

    def get_drift_history(
        self, symbol: str, drift_type: str = "", limit: int = 50,
    ) -> list[dict]:
        sql = "SELECT * FROM drift_snapshots WHERE symbol=?"
        params: list = [symbol]
        if drift_type:
            sql += " AND drift_type=?"
            params.append(drift_type)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Performance snapshot operations ──────────────────────────

    def save_performance_snapshot(
        self,
        symbol: str,
        model_id: int,
        metrics: dict,
    ) -> int:
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO performance_snapshots
               (symbol, model_id, ts, window_size,
                accuracy, auc, logloss, win_rate, profit_factor,
                sharpe_ratio, total_pnl_pips,
                total_predictions, total_trades, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol, model_id, time.time(),
                metrics.get("window_size", 0),
                metrics.get("accuracy", 0),
                metrics.get("auc", 0),
                metrics.get("logloss", 0),
                metrics.get("win_rate", 0),
                metrics.get("profit_factor", 0),
                metrics.get("sharpe_ratio", 0),
                metrics.get("total_pnl_pips", 0),
                metrics.get("total_predictions", 0),
                metrics.get("total_trades", 0),
                json.dumps(metrics.get("details", {})),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_performance_history(
        self, symbol: str, model_id: int | None = None, limit: int = 50,
    ) -> list[dict]:
        sql = "SELECT * FROM performance_snapshots WHERE symbol=?"
        params: list = [symbol]
        if model_id is not None:
            sql += " AND model_id=?"
            params.append(model_id)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Retrain history operations ───────────────────────────────

    def start_retrain(
        self,
        symbol: str,
        trigger_reason: str,
        old_model_id: int | None = None,
        trigger_details: dict | None = None,
    ) -> int:
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO retrain_history
               (symbol, trigger_reason, trigger_details, started_at,
                status, old_model_id)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (symbol, trigger_reason, json.dumps(trigger_details or {}),
             time.time(), old_model_id),
        )
        conn.commit()
        return cur.lastrowid

    def complete_retrain(
        self,
        retrain_id: int,
        status: str,
        new_model_id: int | None = None,
        result: dict | None = None,
    ):
        conn = self._conn()
        conn.execute(
            "UPDATE retrain_history SET status=?, completed_at=?, "
            "new_model_id=?, result=? WHERE id=?",
            (status, time.time(), new_model_id,
             json.dumps(result or {}), retrain_id),
        )
        conn.commit()

    def get_retrain_history(
        self, symbol: str = "", limit: int = 20,
    ) -> list[dict]:
        sql = "SELECT * FROM retrain_history WHERE 1=1"
        params: list = []
        if symbol:
            sql += " AND symbol=?"
            params.append(symbol)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_last_retrain(self, symbol: str) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM retrain_history WHERE symbol=? "
            "ORDER BY started_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return dict(row) if row else None

    # ── Feedback operations ──────────────────────────────────────

    def save_feedback(
        self,
        prediction_id: int,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pips: float = 0.0,
        regime: str = "",
    ) -> int:
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO feedback_records
               (prediction_id, trade_id, symbol, direction,
                entry_price, exit_price, pnl, pnl_pips, regime, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, trade_id, symbol, direction,
             entry_price, exit_price, pnl, pnl_pips, regime, time.time()),
        )
        conn.commit()
        return cur.lastrowid

    def get_unprocessed_feedback(self, limit: int = 100) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM feedback_records WHERE processed=0 "
            "ORDER BY ts ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_feedback_processed(self, feedback_ids: list[int]):
        if not feedback_ids:
            return
        conn = self._conn()
        placeholders = ",".join("?" * len(feedback_ids))
        conn.execute(
            f"UPDATE feedback_records SET processed=1 WHERE id IN ({placeholders})",
            feedback_ids,
        )
        conn.commit()

    # ── Statistics ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        conn = self._conn()
        return {
            "db_path": self._db_path,
            "total_models": conn.execute(
                "SELECT COUNT(*) FROM model_registry"
            ).fetchone()[0],
            "champion_models": conn.execute(
                "SELECT COUNT(*) FROM model_registry WHERE status='champion'"
            ).fetchone()[0],
            "challenger_models": conn.execute(
                "SELECT COUNT(*) FROM model_registry WHERE status='challenger'"
            ).fetchone()[0],
            "total_predictions": conn.execute(
                "SELECT COUNT(*) FROM prediction_log"
            ).fetchone()[0],
            "unresolved_predictions": conn.execute(
                "SELECT COUNT(*) FROM prediction_log WHERE actual_outcome IS NULL"
            ).fetchone()[0],
            "total_drift_snapshots": conn.execute(
                "SELECT COUNT(*) FROM drift_snapshots"
            ).fetchone()[0],
            "drift_detected_count": conn.execute(
                "SELECT COUNT(*) FROM drift_snapshots WHERE drift_detected=1"
            ).fetchone()[0],
            "total_retrains": conn.execute(
                "SELECT COUNT(*) FROM retrain_history"
            ).fetchone()[0],
            "total_feedback": conn.execute(
                "SELECT COUNT(*) FROM feedback_records"
            ).fetchone()[0],
        }
