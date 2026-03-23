"""
Historical market data manager.

Downloads OHLC bars from cTrader via the Open API SDK, persists them
as candles in the data store, and provides gap-fill / backfill logic.
Also supports CSV import/export for offline analysis.
"""

import os
import csv
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from data_store import MarketDataStore, Candle, INTERVALS

logger = logging.getLogger("historical_data")

HISTORICAL_DIR = Path(__file__).parent.parent.parent / "data" / "historical"

# cTrader Open API timeframe mapping (matches regime_detector.py)
CTRADER_TIMEFRAMES = {
    "1m": "m1",
    "5m": "m5",
    "15m": "m15",
    "30m": "m30",
    "1h": "h1",
    "4h": "h4",
    "1d": "d1",
}


@dataclass
class BackfillJob:
    symbol: str
    interval: str
    start_ts: float
    end_ts: float
    status: str = "pending"  # pending, running, completed, failed
    candles_fetched: int = 0
    error: str = ""


class HistoricalDataManager:
    """Manages historical OHLC data: download, backfill, import/export."""

    def __init__(self, store: MarketDataStore, broker_client=None):
        self._store = store
        self._broker = broker_client
        self._jobs: dict[str, BackfillJob] = {}
        self._job_lock = threading.Lock()
        self._job_counter = 0

        HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Backfill from cTrader ────────────────────────────────────

    def backfill(self, symbol: str, interval: str, days: int = 30) -> str:
        """Start a background backfill job. Returns job ID."""
        if interval not in INTERVALS:
            raise ValueError(f"Unknown interval: {interval}")
        if not self._broker:
            raise RuntimeError("No broker client configured for live data download")

        end_ts = time.time()
        start_ts = end_ts - (days * 86400)

        self._job_counter += 1
        job_id = f"backfill_{self._job_counter}"
        job = BackfillJob(
            symbol=symbol, interval=interval,
            start_ts=start_ts, end_ts=end_ts,
        )
        with self._job_lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_backfill, args=(job_id, job),
            daemon=True, name=f"backfill-{job_id}",
        )
        thread.start()
        return job_id

    def _run_backfill(self, job_id: str, job: BackfillJob):
        job.status = "running"
        logger.info("Backfill started: %s %s %d days", job.symbol, job.interval,
                     int((job.end_ts - job.start_ts) / 86400))
        try:
            ct_tf = CTRADER_TIMEFRAMES.get(job.interval)
            if not ct_tf:
                raise ValueError(f"No cTrader mapping for interval {job.interval}")

            # Fetch in chunks to avoid API limits (max ~5000 bars per request)
            interval_sec = INTERVALS[job.interval]
            chunk_bars = 5000
            chunk_duration = chunk_bars * interval_sec

            cursor = job.start_ts
            total_candles = 0

            while cursor < job.end_ts:
                chunk_end = min(cursor + chunk_duration, job.end_ts)
                bars = self._broker.get_bars(
                    symbol=job.symbol,
                    timeframe=ct_tf,
                    from_ts=int(cursor * 1000),  # cTrader uses milliseconds
                    to_ts=int(chunk_end * 1000),
                )

                candles = []
                for bar in bars:
                    candles.append(Candle(
                        symbol=job.symbol,
                        interval=job.interval,
                        open=bar.get("open", 0),
                        high=bar.get("high", 0),
                        low=bar.get("low", 0),
                        close=bar.get("close", 0),
                        volume=bar.get("volume", bar.get("tick_count", 0)),
                        open_time=bar.get("timestamp", cursor) / 1000.0,
                        close_time=bar.get("timestamp", cursor) / 1000.0 + interval_sec,
                    ))

                if candles:
                    self._store.upsert_candles_batch(candles)
                    total_candles += len(candles)

                cursor = chunk_end
                time.sleep(0.5)  # Rate limit

            job.candles_fetched = total_candles
            job.status = "completed"
            logger.info("Backfill completed: %s %s → %d candles",
                         job.symbol, job.interval, total_candles)

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error("Backfill failed: %s", e)

    def get_job_status(self, job_id: str) -> dict | None:
        with self._job_lock:
            job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job_id,
            "symbol": job.symbol,
            "interval": job.interval,
            "status": job.status,
            "candles_fetched": job.candles_fetched,
            "error": job.error,
        }

    def list_jobs(self) -> list[dict]:
        with self._job_lock:
            return [
                self.get_job_status(jid) for jid in self._jobs
            ]

    # ── Gap detection ────────────────────────────────────────────

    def find_gaps(self, symbol: str, interval: str,
                  start_ts: float = 0, end_ts: float = 0) -> list[dict]:
        """Find gaps in candle data where bars are missing."""
        if end_ts <= 0:
            end_ts = time.time()
        if start_ts <= 0:
            start_ts = end_ts - 30 * 86400  # last 30 days

        candles = self._store.get_candles(symbol, interval, start_ts, end_ts, limit=50000)
        if len(candles) < 2:
            return []

        interval_sec = INTERVALS.get(interval, 60)
        # Allow 1.5x interval tolerance for gaps (markets close weekends, etc.)
        tolerance = interval_sec * 1.5

        gaps = []
        for i in range(1, len(candles)):
            prev_close = candles[i - 1]["open_time"] + interval_sec
            curr_open = candles[i]["open_time"]
            gap_duration = curr_open - prev_close

            if gap_duration > tolerance:
                gaps.append({
                    "start": datetime.fromtimestamp(prev_close, tz=timezone.utc).isoformat(),
                    "end": datetime.fromtimestamp(curr_open, tz=timezone.utc).isoformat(),
                    "missing_bars": int(gap_duration / interval_sec),
                    "start_ts": prev_close,
                    "end_ts": curr_open,
                })

        return gaps

    # ── CSV import / export ──────────────────────────────────────

    def export_csv(self, symbol: str, interval: str,
                   start_ts: float = 0, end_ts: float = 0) -> Path:
        """Export candles to CSV file. Returns file path."""
        candles = self._store.get_candles(symbol, interval, start_ts, end_ts, limit=100000)

        filename = f"{symbol}_{interval}_{int(time.time())}.csv"
        filepath = HISTORICAL_DIR / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "symbol", "interval", "open", "high", "low", "close",
                "volume", "open_time", "close_time",
            ])
            writer.writeheader()
            writer.writerows(candles)

        logger.info("Exported %d candles to %s", len(candles), filepath)
        return filepath

    def import_csv(self, filepath: str | Path, symbol: str = "",
                   interval: str = "") -> int:
        """Import candles from CSV. Returns count imported."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV not found: {filepath}")

        candles = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append(Candle(
                    symbol=row.get("symbol", symbol),
                    interval=row.get("interval", interval),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row.get("volume", 0)),
                    open_time=float(row["open_time"]),
                    close_time=float(row["close_time"]),
                ))

        self._store.upsert_candles_batch(candles)
        logger.info("Imported %d candles from %s", len(candles), filepath)
        return len(candles)

    # ── Data availability summary ────────────────────────────────

    def get_coverage(self, symbol: str) -> dict:
        """Return data coverage summary for a symbol across all intervals."""
        coverage = {}
        for interval in INTERVALS:
            count = self._store.count_candles(symbol, interval)
            if count == 0:
                continue
            latest = self._store.get_latest_candle(symbol, interval)
            candles = self._store.get_candles(symbol, interval, limit=1)
            first = candles[0] if candles else None
            coverage[interval] = {
                "count": count,
                "first": datetime.fromtimestamp(first["open_time"], tz=timezone.utc).isoformat() if first else None,
                "last": datetime.fromtimestamp(latest["open_time"], tz=timezone.utc).isoformat() if latest else None,
            }

        tick_count = self._store.count_ticks(symbol)
        latest_tick = self._store.get_latest_tick(symbol)

        return {
            "symbol": symbol,
            "tick_count": tick_count,
            "latest_tick": datetime.fromtimestamp(latest_tick["ts"], tz=timezone.utc).isoformat() if latest_tick else None,
            "candles": coverage,
        }
