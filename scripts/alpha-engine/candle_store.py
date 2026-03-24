"""
Historical 5M Candle ML Pipeline - Persistence Layer
=====================================================

SQLite-backed store for candle ML artifacts: computed features, labels,
normalisation stats, and complete datasets.  Follows the same WAL-mode,
thread-safe pattern as the data-pipeline MarketDataStore and the
tick-level AlphaStore.

Schema is separate from the tick-level alpha.db to keep the two
pipelines decoupled.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from candle_config import CANDLE_DATA_DIR
from candle_features import CandleFeatureVector
from candle_labels import CandleLabel
from candle_dataset import (
    CandleDataset,
    CandleNormStats,
    CandleSample,
    CandleDatasetSplit,
)


class CandleMLStore:
    """Persistent storage for the 5M candle ML pipeline."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or str(CANDLE_DATA_DIR / "candle_ml.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    # ── connection management ─────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _write(self):
        with self._write_lock:
            conn = self._conn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── schema ────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._write() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS candle_features (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    ts          REAL NOT NULL,
                    interval    TEXT NOT NULL DEFAULT '5m',
                    data        TEXT NOT NULL,
                    created_at  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cfeatures_sym_ts
                    ON candle_features(symbol, ts);

                CREATE TABLE IF NOT EXISTS candle_labels (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    ts          REAL NOT NULL,
                    interval    TEXT NOT NULL DEFAULT '5m',
                    data        TEXT NOT NULL,
                    created_at  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_clabels_sym_ts
                    ON candle_labels(symbol, ts);

                CREATE TABLE IF NOT EXISTS candle_datasets (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    interval    TEXT NOT NULL DEFAULT '5m',
                    created_at  REAL NOT NULL,
                    config      TEXT NOT NULL,
                    norm_stats  TEXT NOT NULL,
                    summary     TEXT NOT NULL,
                    train_size  INTEGER NOT NULL DEFAULT 0,
                    val_size    INTEGER NOT NULL DEFAULT 0,
                    test_size   INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS candle_dataset_samples (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id  INTEGER NOT NULL,
                    split_name  TEXT NOT NULL,
                    ts          REAL NOT NULL,
                    symbol      TEXT NOT NULL,
                    features    TEXT NOT NULL,
                    labels      TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES candle_datasets(id)
                );
                CREATE INDEX IF NOT EXISTS idx_cds_samples_dataset
                    ON candle_dataset_samples(dataset_id, split_name);
            """)

    # ── features ──────────────────────────────────────────────

    def save_features(
        self,
        features: list[CandleFeatureVector],
        interval: str = "5m",
    ) -> int:
        """Persist computed candle feature vectors. Returns count saved."""
        now = time.time()
        rows = [
            (fv.symbol, fv.ts, interval, json.dumps(fv.to_dict()), now)
            for fv in features
        ]
        with self._write() as conn:
            conn.executemany(
                "INSERT INTO candle_features "
                "(symbol, ts, interval, data, created_at) VALUES (?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def load_features(
        self,
        symbol: str,
        start_ts: float = 0,
        end_ts: float = 0,
        interval: str = "5m",
        limit: int = 50000,
    ) -> list[dict]:
        conn = self._conn()
        if end_ts <= 0:
            end_ts = time.time()
        rows = conn.execute(
            "SELECT ts, data FROM candle_features "
            "WHERE symbol=? AND interval=? AND ts>=? AND ts<=? "
            "ORDER BY ts LIMIT ?",
            (symbol, interval, start_ts, end_ts, limit),
        ).fetchall()
        results = []
        for r in rows:
            d = json.loads(r["data"])
            d["ts"] = r["ts"]
            d["symbol"] = symbol
            results.append(d)
        return results

    def count_features(self, symbol: str, interval: str = "5m") -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM candle_features "
            "WHERE symbol=? AND interval=?",
            (symbol, interval),
        ).fetchone()
        return row["cnt"] if row else 0

    # ── labels ────────────────────────────────────────────────

    def save_labels(
        self,
        labels: list[CandleLabel],
        interval: str = "5m",
    ) -> int:
        """Persist computed candle labels. Returns count saved."""
        now = time.time()
        rows = [
            (lbl.symbol, lbl.ts, interval, json.dumps(lbl.to_dict()), now)
            for lbl in labels
        ]
        with self._write() as conn:
            conn.executemany(
                "INSERT INTO candle_labels "
                "(symbol, ts, interval, data, created_at) VALUES (?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def load_labels(
        self,
        symbol: str,
        start_ts: float = 0,
        end_ts: float = 0,
        interval: str = "5m",
        limit: int = 50000,
    ) -> list[dict]:
        conn = self._conn()
        if end_ts <= 0:
            end_ts = time.time()
        rows = conn.execute(
            "SELECT ts, data FROM candle_labels "
            "WHERE symbol=? AND interval=? AND ts>=? AND ts<=? "
            "ORDER BY ts LIMIT ?",
            (symbol, interval, start_ts, end_ts, limit),
        ).fetchall()
        results = []
        for r in rows:
            d = json.loads(r["data"])
            d["ts"] = r["ts"]
            d["symbol"] = symbol
            results.append(d)
        return results

    def count_labels(self, symbol: str, interval: str = "5m") -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM candle_labels "
            "WHERE symbol=? AND interval=?",
            (symbol, interval),
        ).fetchone()
        return row["cnt"] if row else 0

    # ── datasets ──────────────────────────────────────────────

    def save_dataset(self, ds: CandleDataset) -> int:
        """Persist a full dataset (metadata + all samples). Returns dataset ID."""
        config_json = json.dumps({
            "train_ratio": ds.config.train_ratio,
            "val_ratio": ds.config.val_ratio,
            "test_ratio": ds.config.test_ratio,
            "purge_gap_candles": ds.config.purge_gap_candles,
            "normalize": ds.config.normalize,
            "normalize_method": ds.config.normalize_method,
            "clip_std": ds.config.clip_std,
            "seed": ds.config.seed,
        })
        norm_json = json.dumps(ds.norm_stats.to_dict())
        summary_json = json.dumps(ds.summary())

        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO candle_datasets "
                "(symbol, interval, created_at, config, norm_stats, summary, "
                " train_size, val_size, test_size) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ds.symbol, ds.interval, ds.created_at,
                    config_json, norm_json, summary_json,
                    ds.train.size, ds.val.size, ds.test.size,
                ),
            )
            ds_id = cursor.lastrowid

            for split in [ds.train, ds.val, ds.test]:
                if not split.samples:
                    continue
                sample_rows = [
                    (
                        ds_id, split.name, s.ts, s.symbol,
                        json.dumps(s.features), json.dumps(s.labels),
                    )
                    for s in split.samples
                ]
                conn.executemany(
                    "INSERT INTO candle_dataset_samples "
                    "(dataset_id, split_name, ts, symbol, features, labels) "
                    "VALUES (?,?,?,?,?,?)",
                    sample_rows,
                )

        return ds_id

    def load_dataset_meta(self, dataset_id: int) -> dict | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM candle_datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "symbol": row["symbol"],
            "interval": row["interval"],
            "created_at": row["created_at"],
            "config": json.loads(row["config"]),
            "norm_stats": json.loads(row["norm_stats"]),
            "summary": json.loads(row["summary"]),
            "train_size": row["train_size"],
            "val_size": row["val_size"],
            "test_size": row["test_size"],
        }

    def load_dataset_samples(
        self, dataset_id: int, split_name: str, limit: int = 0,
    ) -> list[dict]:
        conn = self._conn()
        query = (
            "SELECT ts, symbol, features, labels FROM candle_dataset_samples "
            "WHERE dataset_id=? AND split_name=? ORDER BY ts"
        )
        params: list = [dataset_id, split_name]
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "ts": r["ts"],
                "symbol": r["symbol"],
                "features": json.loads(r["features"]),
                "labels": json.loads(r["labels"]),
            }
            for r in rows
        ]

    def list_datasets(self, symbol: str = "") -> list[dict]:
        conn = self._conn()
        if symbol:
            rows = conn.execute(
                "SELECT id, symbol, interval, created_at, "
                "train_size, val_size, test_size, summary "
                "FROM candle_datasets WHERE symbol=? ORDER BY created_at DESC",
                (symbol,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, symbol, interval, created_at, "
                "train_size, val_size, test_size, summary "
                "FROM candle_datasets ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "symbol": r["symbol"],
                "interval": r["interval"],
                "created_at": r["created_at"],
                "train_size": r["train_size"],
                "val_size": r["val_size"],
                "test_size": r["test_size"],
                "summary": json.loads(r["summary"]),
            }
            for r in rows
        ]

    def load_norm_stats(self, dataset_id: int) -> CandleNormStats | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT norm_stats FROM candle_datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if not row:
            return None
        return CandleNormStats.from_dict(json.loads(row["norm_stats"]))

    # ── maintenance ───────────────────────────────────────────

    def purge_features(self, symbol: str, older_than_ts: float) -> int:
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM candle_features WHERE symbol=? AND ts<?",
                (symbol, older_than_ts),
            )
            return cursor.rowcount

    def purge_labels(self, symbol: str, older_than_ts: float) -> int:
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM candle_labels WHERE symbol=? AND ts<?",
                (symbol, older_than_ts),
            )
            return cursor.rowcount

    def get_stats(self) -> dict:
        conn = self._conn()
        feat_count = conn.execute(
            "SELECT COUNT(*) as c FROM candle_features"
        ).fetchone()["c"]
        label_count = conn.execute(
            "SELECT COUNT(*) as c FROM candle_labels"
        ).fetchone()["c"]
        ds_count = conn.execute(
            "SELECT COUNT(*) as c FROM candle_datasets"
        ).fetchone()["c"]
        sample_count = conn.execute(
            "SELECT COUNT(*) as c FROM candle_dataset_samples"
        ).fetchone()["c"]
        return {
            "total_features": feat_count,
            "total_labels": label_count,
            "total_datasets": ds_count,
            "total_dataset_samples": sample_count,
            "db_path": self._db_path,
        }
