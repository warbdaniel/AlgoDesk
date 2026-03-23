"""
Project Alpha - High-Performance Tick Ring Buffer

Lock-free-style ring buffer for real-time tick ingestion with O(1) append
and O(1) random access.  Maintains per-symbol rolling statistics used
downstream by the scalping feature engine.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from config import TICK_BUFFER_SIZE


# ---------------------------------------------------------------------------
# Tick data structure  (lightweight, no overhead of dict)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AlphaTick:
    """Single market tick optimised for scalping analytics."""
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    ts: float = 0.0          # unix epoch (set on ingest if 0)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) * 0.5

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "mid": self.mid,
            "spread": self.spread,
            "ts": self.ts,
        }


# ---------------------------------------------------------------------------
# Ring buffer for a single symbol
# ---------------------------------------------------------------------------
class TickRing:
    """Fixed-size ring buffer storing ticks for one symbol.

    Provides O(1) append, O(1) indexed access, and rolling statistics
    (count, spread EMA, tick-rate) without allocating new memory.
    """

    def __init__(self, capacity: int = TICK_BUFFER_SIZE, spread_ema_halflife: int = 20):
        self._cap = capacity
        self._buf: list[AlphaTick | None] = [None] * capacity
        self._head = 0          # next write index
        self._count = 0         # total ticks ingested (monotonic)
        self._lock = threading.Lock()

        # Rolling stats
        self._spread_ema: float = 0.0
        self._spread_alpha: float = 1.0 - math.exp(-math.log(2) / max(spread_ema_halflife, 1))
        self._last_ts: float = 0.0
        self._first_ts: float = 0.0

    # ---- write path -------------------------------------------------------

    def append(self, tick: AlphaTick) -> None:
        """Append a tick.  Overwrites oldest when full."""
        if tick.ts == 0.0:
            tick.ts = time.time()

        with self._lock:
            self._buf[self._head] = tick
            self._head = (self._head + 1) % self._cap
            self._count += 1

            # update rolling spread EMA
            if self._count == 1:
                self._spread_ema = tick.spread
                self._first_ts = tick.ts
            else:
                self._spread_ema += self._spread_alpha * (tick.spread - self._spread_ema)
            self._last_ts = tick.ts

    # ---- read path --------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of ticks currently held (up to capacity)."""
        return min(self._count, self._cap)

    @property
    def total_count(self) -> int:
        """Total ticks ever ingested (monotonically increasing)."""
        return self._count

    def get_latest(self, n: int = 1) -> list[AlphaTick]:
        """Return the *n* most recent ticks (newest last)."""
        with self._lock:
            sz = self.size
            n = min(n, sz)
            if n == 0:
                return []
            result: list[AlphaTick] = []
            start = (self._head - n) % self._cap
            for i in range(n):
                idx = (start + i) % self._cap
                result.append(self._buf[idx])  # type: ignore[arg-type]
            return result

    def get_slice(self, start_ts: float, end_ts: float) -> list[AlphaTick]:
        """Return ticks in [start_ts, end_ts].  Linear scan of buffer."""
        with self._lock:
            sz = self.size
            if sz == 0:
                return []
            result: list[AlphaTick] = []
            oldest_idx = (self._head - sz) % self._cap
            for i in range(sz):
                idx = (oldest_idx + i) % self._cap
                t = self._buf[idx]
                if t is not None and start_ts <= t.ts <= end_ts:
                    result.append(t)
            return result

    # ---- rolling statistics -----------------------------------------------

    @property
    def spread_ema(self) -> float:
        return self._spread_ema

    @property
    def tick_rate(self) -> float:
        """Ticks per second over the buffered window."""
        if self._count < 2 or self._last_ts <= self._first_ts:
            return 0.0
        elapsed = self._last_ts - self._first_ts
        return min(self._count, self._cap) / elapsed if elapsed > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "total_ingested": self._count,
            "spread_ema": round(self._spread_ema, 8),
            "tick_rate": round(self.tick_rate, 2),
            "first_ts": self._first_ts,
            "last_ts": self._last_ts,
        }


# ---------------------------------------------------------------------------
# Multi-symbol tick buffer manager
# ---------------------------------------------------------------------------
class TickBufferManager:
    """Manages per-symbol TickRing instances and dispatches callbacks."""

    def __init__(self, capacity: int = TICK_BUFFER_SIZE, spread_ema_halflife: int = 20):
        self._capacity = capacity
        self._spread_hl = spread_ema_halflife
        self._rings: dict[str, TickRing] = {}
        self._callbacks: list[Callable[[AlphaTick], None]] = []
        self._lock = threading.Lock()

    def _get_ring(self, symbol: str) -> TickRing:
        if symbol not in self._rings:
            with self._lock:
                if symbol not in self._rings:
                    self._rings[symbol] = TickRing(self._capacity, self._spread_hl)
        return self._rings[symbol]

    # ---- ingest -----------------------------------------------------------

    def ingest(self, tick: AlphaTick) -> None:
        """Ingest a single tick: store + notify callbacks."""
        ring = self._get_ring(tick.symbol)
        ring.append(tick)
        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception:
                pass  # never let a bad callback break ingestion

    def ingest_raw(
        self, symbol: str, bid: float, ask: float,
        bid_size: float = 0.0, ask_size: float = 0.0, ts: float = 0.0,
    ) -> AlphaTick:
        """Convenience: build AlphaTick and ingest, return the tick."""
        tick = AlphaTick(
            symbol=symbol, bid=bid, ask=ask,
            bid_size=bid_size, ask_size=ask_size,
            ts=ts or time.time(),
        )
        self.ingest(tick)
        return tick

    # ---- subscriptions ----------------------------------------------------

    def on_tick(self, callback: Callable[[AlphaTick], None]) -> None:
        self._callbacks.append(callback)

    # ---- queries ----------------------------------------------------------

    def latest(self, symbol: str, n: int = 1) -> list[AlphaTick]:
        ring = self._rings.get(symbol)
        return ring.get_latest(n) if ring else []

    def slice(self, symbol: str, start_ts: float, end_ts: float) -> list[AlphaTick]:
        ring = self._rings.get(symbol)
        return ring.get_slice(start_ts, end_ts) if ring else []

    def ring(self, symbol: str) -> TickRing | None:
        return self._rings.get(symbol)

    @property
    def symbols(self) -> list[str]:
        return list(self._rings.keys())

    def stats(self) -> dict:
        return {sym: ring.stats() for sym, ring in self._rings.items()}
