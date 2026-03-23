"""
Project Alpha - Persistent Storage Layer

SQLite-backed store for computed features, labels, normalisation stats,
and dataset metadata.  Follows the same WAL-mode, thread-safe pattern
used by the data-pipeline MarketDataStore.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from config import AlphaConfig, DATA_DIR
from scalping_features import ScalpingFeatureVector
from label_engine import ScalpingLabel
from dataset_builder import AlphaDataset, NormStats, Sample


# ---------------------------------------------------------------------------
# Alpha Store
# ---------------------------------------------------------------------------
class AlphaStore:
    """Persistent storage for Project Alpha ML pipeline artefacts."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or str(DATA_DIR / "alpha.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    # ---- connection -------------------------------------------------------

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

    # ---- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        with self._write() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS features (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    ts          REAL NOT NULL,
                    data        TEXT NOT NULL,
                    created_at  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_features_sym_ts
                    ON features(symbol, ts);

                CREATE TABLE IF NOT EXISTS labels (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    entry_ts    REAL NOT NULL,
                    data        TEXT NOT NULL,
                    created_at  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_labels_sym_ts
                    ON labels(symbol, entry_ts);

                CREATE TABLE IF NOT EXISTS datasets (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT NOT NULL,
                    created_at  REAL NOT NULL,
                    config      TEXT NOT NULL,
                    norm_stats  TEXT NOT NULL,
                    summary     TEXT NOT NULL,
                    train_size  INTEGER NOT NULL DEFAULT 0,
                    val_size    INTEGER NOT NULL DEFAULT 0,
                    test_size   INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS dataset_samples (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id  INTEGER NOT NULL,
                    split_name  TEXT NOT NULL,
                    ts          REAL NOT NULL,
                    symbol      TEXT NOT NULL,
                    features    TEXT NOT NULL,
                    labels      TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
                );
                CREATE INDEX IF NOT EXISTS idx_ds_samples_dataset
                    ON dataset_samples(dataset_id, split_name);
            """)

    # ---- features ---------------------------------------------------------

    def save_features(self, features: list[ScalpingFeatureVector]) -> int:
        """Persist computed feature vectors. Returns count saved."""
        now = time.time()
        rows = [
            (fv.symbol, fv.ts, json.dumps(fv.to_dict()), now)
            for fv in features
        ]
        with self._write() as conn:
            conn.executemany(
                "INSERT INTO features (symbol, ts, data, created_at) VALUES (?,?,?,?)",
                rows,
            )
        return len(rows)

    def load_features(
        self, symbol: str, start_ts: float = 0, end_ts: float = 0, limit: int = 10000,
    ) -> list[dict]:
        """Load stored features as dicts."""
        conn = self._conn()
        if end_ts <= 0:
            end_ts = time.time()
        rows = conn.execute(
            "SELECT ts, data FROM features WHERE symbol=? AND ts>=? AND ts<=? "
            "ORDER BY ts LIMIT ?",
            (symbol, start_ts, end_ts, limit),
        ).fetchall()
        results = []
        for r in rows:
            d = json.loads(r["data"])
            d["ts"] = r["ts"]
            d["symbol"] = symbol
            results.append(d)
        return results

    def count_features(self, symbol: str) -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM features WHERE symbol=?", (symbol,)
        ).fetchone()
        return row["cnt"] if row else 0

    # ---- labels -----------------------------------------------------------

    def save_labels(self, labels: list[ScalpingLabel]) -> int:
        """Persist computed labels. Returns count saved."""
        now = time.time()
        rows = [
            (lbl.symbol, lbl.entry_ts, json.dumps(lbl.to_dict()), now)
            for lbl in labels
        ]
        with self._write() as conn:
            conn.executemany(
                "INSERT INTO labels (symbol, entry_ts, data, created_at) VALUES (?,?,?,?)",
                rows,
            )
        return len(rows)

    def load_labels(
        self, symbol: str, start_ts: float = 0, end_ts: float = 0, limit: int = 10000,
    ) -> list[dict]:
        conn = self._conn()
        if end_ts <= 0:
            end_ts = time.time()
        rows = conn.execute(
            "SELECT entry_ts, data FROM labels WHERE symbol=? AND entry_ts>=? AND entry_ts<=? "
            "ORDER BY entry_ts LIMIT ?",
            (symbol, start_ts, end_ts, limit),
        ).fetchall()
        results = []
        for r in rows:
            d = json.loads(r["data"])
            results.append(d)
        return results

    def count_labels(self, symbol: str) -> int:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM labels WHERE symbol=?", (symbol,)
        ).fetchone()
        return row["cnt"] if row else 0

    # ---- datasets ---------------------------------------------------------

    def save_dataset(self, ds: AlphaDataset) -> int:
        """Persist a full dataset (metadata + all samples). Returns dataset ID."""
        config_json = json.dumps({
            "train_ratio": ds.config.train_ratio,
            "val_ratio": ds.config.val_ratio,
            "test_ratio": ds.config.test_ratio,
            "purge_gap_sec": ds.config.purge_gap_sec,
            "normalize": ds.config.normalize,
            "normalize_method": ds.config.normalize_method,
            "clip_std": ds.config.clip_std,
            "seed": ds.config.seed,
        })
        norm_json = json.dumps(ds.norm_stats.to_dict())
        summary_json = json.dumps(ds.summary())

        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO datasets "
                "(symbol, created_at, config, norm_stats, summary, train_size, val_size, test_size) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    ds.symbol, ds.created_at, config_json, norm_json, summary_json,
                    ds.train.size, ds.val.size, ds.test.size,
                ),
            )
            ds_id = cursor.lastrowid

            # Save samples per split
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
                    "INSERT INTO dataset_samples "
                    "(dataset_id, split_name, ts, symbol, features, labels) "
                    "VALUES (?,?,?,?,?,?)",
                    sample_rows,
                )

        return ds_id

    def load_dataset_meta(self, dataset_id: int) -> dict | None:
        """Load dataset metadata (without samples)."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "symbol": row["symbol"],
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
        """Load samples from a specific split of a dataset."""
        conn = self._conn()
        query = (
            "SELECT ts, symbol, features, labels FROM dataset_samples "
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
        """List all datasets, optionally filtered by symbol."""
        conn = self._conn()
        if symbol:
            rows = conn.execute(
                "SELECT id, symbol, created_at, train_size, val_size, test_size, summary "
                "FROM datasets WHERE symbol=? ORDER BY created_at DESC",
                (symbol,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, symbol, created_at, train_size, val_size, test_size, summary "
                "FROM datasets ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "symbol": r["symbol"],
                "created_at": r["created_at"],
                "train_size": r["train_size"],
                "val_size": r["val_size"],
                "test_size": r["test_size"],
                "summary": json.loads(r["summary"]),
            }
            for r in rows
        ]

    def load_norm_stats(self, dataset_id: int) -> NormStats | None:
        """Load normalisation statistics for a dataset."""
        conn = self._conn()
        row = conn.execute(
            "SELECT norm_stats FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if not row:
            return None
        return NormStats.from_dict(json.loads(row["norm_stats"]))

    # ---- maintenance ------------------------------------------------------

    def purge_features(self, symbol: str, older_than_ts: float) -> int:
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM features WHERE symbol=? AND ts<?",
                (symbol, older_than_ts),
            )
            return cursor.rowcount

    def purge_labels(self, symbol: str, older_than_ts: float) -> int:
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM labels WHERE symbol=? AND entry_ts<?",
                (symbol, older_than_ts),
            )
            return cursor.rowcount

    def get_stats(self) -> dict:
        conn = self._conn()
        feat_count = conn.execute("SELECT COUNT(*) as c FROM features").fetchone()["c"]
        label_count = conn.execute("SELECT COUNT(*) as c FROM labels").fetchone()["c"]
        ds_count = conn.execute("SELECT COUNT(*) as c FROM datasets").fetchone()["c"]
        sample_count = conn.execute("SELECT COUNT(*) as c FROM dataset_samples").fetchone()["c"]
        return {
            "total_features": feat_count,
            "total_labels": label_count,
            "total_datasets": ds_count,
            "total_dataset_samples": sample_count,
            "db_path": self._db_path,
        }
