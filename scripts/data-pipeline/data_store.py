"""
Persistent market data store backed by SQLite.

Stores ticks and multi-timeframe OHLCV candles with efficient
time-range queries. Thread-safe for concurrent read/write from
the FIX price feed and API consumers.
"""

import sqlite3
import threading
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("data_store")

DEFAULT_DB_PATH = Path(__file__).parent / "market_data.db"

# Supported candle intervals in seconds
INTERVALS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: float  # unix epoch


@dataclass
class Candle:
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: int  # tick count
    open_time: float
    close_time: float


class MarketDataStore:
    """SQLite-backed store for ticks and candles with WAL mode."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self._db_path = str(db_path)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_db()

    # ── Connection management ────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8 MB
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _write(self):
        """Serialize writes through a lock."""
        with self._write_lock:
            conn = self._get_conn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _init_db(self):
        with self._write() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    bid         REAL    NOT NULL,
                    ask         REAL    NOT NULL,
                    bid_size    REAL    NOT NULL DEFAULT 0,
                    ask_size    REAL    NOT NULL DEFAULT 0,
                    ts          REAL    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts
                    ON ticks (symbol, ts);

                CREATE TABLE IF NOT EXISTS candles (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    interval    TEXT    NOT NULL,
                    open        REAL    NOT NULL,
                    high        REAL    NOT NULL,
                    low         REAL    NOT NULL,
                    close       REAL    NOT NULL,
                    volume      INTEGER NOT NULL DEFAULT 0,
                    open_time   REAL    NOT NULL,
                    close_time  REAL    NOT NULL,
                    UNIQUE(symbol, interval, open_time)
                );

                CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_time
                    ON candles (symbol, interval, open_time);

                CREATE TABLE IF NOT EXISTS symbols (
                    symbol      TEXT PRIMARY KEY,
                    description TEXT    NOT NULL DEFAULT '',
                    pip_size    REAL    NOT NULL DEFAULT 0.0001,
                    digits      INTEGER NOT NULL DEFAULT 5,
                    added_at    REAL    NOT NULL
                );
            """)
        logger.info("Market data store initialized at %s", self._db_path)

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Symbol registry ──────────────────────────────────────────

    def register_symbol(self, symbol: str, description: str = "",
                        pip_size: float = 0.0001, digits: int = 5):
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO symbols (symbol, description, pip_size, digits, added_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol, description, pip_size, digits, time.time()),
            )

    def get_symbols(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM symbols ORDER BY symbol").fetchall()
        return [dict(r) for r in rows]

    # ── Tick storage ─────────────────────────────────────────────

    def insert_tick(self, tick: Tick):
        with self._write() as conn:
            conn.execute(
                "INSERT INTO ticks (symbol, bid, ask, bid_size, ask_size, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tick.symbol, tick.bid, tick.ask, tick.bid_size, tick.ask_size, tick.timestamp),
            )

    def insert_ticks_batch(self, ticks: list[Tick]):
        if not ticks:
            return
        with self._write() as conn:
            conn.executemany(
                "INSERT INTO ticks (symbol, bid, ask, bid_size, ask_size, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(t.symbol, t.bid, t.ask, t.bid_size, t.ask_size, t.timestamp) for t in ticks],
            )

    def get_ticks(self, symbol: str, start_ts: float, end_ts: float,
                  limit: int = 10000) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT symbol, bid, ask, bid_size, ask_size, ts "
            "FROM ticks WHERE symbol = ? AND ts >= ? AND ts <= ? "
            "ORDER BY ts LIMIT ?",
            (symbol, start_ts, end_ts, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_tick(self, symbol: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT symbol, bid, ask, bid_size, ask_size, ts "
            "FROM ticks WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return dict(row) if row else None

    def count_ticks(self, symbol: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM ticks WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row["cnt"]

    def purge_ticks(self, older_than_ts: float) -> int:
        """Delete ticks older than timestamp. Returns count deleted."""
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM ticks WHERE ts < ?", (older_than_ts,)
            )
            deleted = cursor.rowcount
        if deleted:
            logger.info("Purged %d old ticks", deleted)
        return deleted

    # ── Candle storage ───────────────────────────────────────────

    def upsert_candle(self, candle: Candle):
        with self._write() as conn:
            conn.execute(
                "INSERT INTO candles (symbol, interval, open, high, low, close, volume, open_time, close_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, interval, open_time) DO UPDATE SET "
                "high = MAX(excluded.high, candles.high), "
                "low = MIN(excluded.low, candles.low), "
                "close = excluded.close, "
                "volume = excluded.volume, "
                "close_time = excluded.close_time",
                (candle.symbol, candle.interval, candle.open, candle.high,
                 candle.low, candle.close, candle.volume,
                 candle.open_time, candle.close_time),
            )

    def upsert_candles_batch(self, candles: list[Candle]):
        if not candles:
            return
        with self._write() as conn:
            conn.executemany(
                "INSERT INTO candles (symbol, interval, open, high, low, close, volume, open_time, close_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, interval, open_time) DO UPDATE SET "
                "high = MAX(excluded.high, candles.high), "
                "low = MIN(excluded.low, candles.low), "
                "close = excluded.close, "
                "volume = excluded.volume, "
                "close_time = excluded.close_time",
                [(c.symbol, c.interval, c.open, c.high, c.low, c.close,
                  c.volume, c.open_time, c.close_time) for c in candles],
            )

    def get_candles(self, symbol: str, interval: str, start_ts: float = 0,
                    end_ts: float = 0, limit: int = 1000) -> list[dict]:
        conn = self._get_conn()
        if end_ts <= 0:
            end_ts = time.time() + 86400
        if start_ts <= 0:
            # Default: fetch last N candles
            rows = conn.execute(
                "SELECT symbol, interval, open, high, low, close, volume, open_time, close_time "
                "FROM candles WHERE symbol = ? AND interval = ? AND open_time <= ? "
                "ORDER BY open_time DESC LIMIT ?",
                (symbol, interval, end_ts, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

        rows = conn.execute(
            "SELECT symbol, interval, open, high, low, close, volume, open_time, close_time "
            "FROM candles WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time <= ? "
            "ORDER BY open_time LIMIT ?",
            (symbol, interval, start_ts, end_ts, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_candle(self, symbol: str, interval: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT symbol, interval, open, high, low, close, volume, open_time, close_time "
            "FROM candles WHERE symbol = ? AND interval = ? ORDER BY open_time DESC LIMIT 1",
            (symbol, interval),
        ).fetchone()
        return dict(row) if row else None

    def count_candles(self, symbol: str, interval: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM candles WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()
        return row["cnt"]

    # ── Aggregation: ticks → candles ─────────────────────────────

    def aggregate_ticks_to_candles(self, symbol: str, interval: str,
                                   start_ts: float = 0, end_ts: float = 0) -> int:
        """Build candles from stored ticks. Returns count of candles created/updated."""
        interval_sec = INTERVALS.get(interval)
        if not interval_sec:
            raise ValueError(f"Unknown interval: {interval}")

        if end_ts <= 0:
            end_ts = time.time()
        if start_ts <= 0:
            # Start from latest candle or beginning of tick data
            latest = self.get_latest_candle(symbol, interval)
            start_ts = latest["open_time"] if latest else 0

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT bid, ask, ts FROM ticks "
            "WHERE symbol = ? AND ts >= ? AND ts <= ? ORDER BY ts",
            (symbol, start_ts, end_ts),
        ).fetchall()

        if not rows:
            return 0

        # Group ticks into buckets
        buckets: dict[float, list] = {}
        for row in rows:
            mid = (row["bid"] + row["ask"]) / 2.0
            bucket_start = int(row["ts"] // interval_sec) * interval_sec
            if bucket_start not in buckets:
                buckets[bucket_start] = []
            buckets[bucket_start].append(mid)

        candles = []
        for bucket_start, mids in sorted(buckets.items()):
            candles.append(Candle(
                symbol=symbol,
                interval=interval,
                open=mids[0],
                high=max(mids),
                low=min(mids),
                close=mids[-1],
                volume=len(mids),
                open_time=bucket_start,
                close_time=bucket_start + interval_sec,
            ))

        self.upsert_candles_batch(candles)
        return len(candles)

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        conn = self._get_conn()
        tick_count = conn.execute("SELECT COUNT(*) as cnt FROM ticks").fetchone()["cnt"]
        candle_count = conn.execute("SELECT COUNT(*) as cnt FROM candles").fetchone()["cnt"]
        symbol_count = conn.execute("SELECT COUNT(*) as cnt FROM symbols").fetchone()["cnt"]

        # Per-symbol tick counts
        tick_breakdown = {}
        rows = conn.execute(
            "SELECT symbol, COUNT(*) as cnt, MIN(ts) as first_ts, MAX(ts) as last_ts "
            "FROM ticks GROUP BY symbol"
        ).fetchall()
        for r in rows:
            tick_breakdown[r["symbol"]] = {
                "count": r["cnt"],
                "first": datetime.fromtimestamp(r["first_ts"], tz=timezone.utc).isoformat() if r["first_ts"] else None,
                "last": datetime.fromtimestamp(r["last_ts"], tz=timezone.utc).isoformat() if r["last_ts"] else None,
            }

        # Per-symbol/interval candle counts
        candle_breakdown = {}
        rows = conn.execute(
            "SELECT symbol, interval, COUNT(*) as cnt FROM candles GROUP BY symbol, interval"
        ).fetchall()
        for r in rows:
            key = f"{r['symbol']}_{r['interval']}"
            candle_breakdown[key] = r["cnt"]

        return {
            "total_ticks": tick_count,
            "total_candles": candle_count,
            "registered_symbols": symbol_count,
            "ticks_by_symbol": tick_breakdown,
            "candles_by_symbol_interval": candle_breakdown,
            "db_path": self._db_path,
        }
