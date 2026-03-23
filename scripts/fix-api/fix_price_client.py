"""
FIX 4.4 Price (Quote) client for cTrader.

Handles market data requests (35=V), full snapshot (35=W),
incremental refresh (35=X), bid/ask storage, and tick-to-candle aggregation.
"""

import time
import threading
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import simplefix

from fix_connector import FIXConnector, _get_field, get_all_fields

logger = logging.getLogger("fix_price")


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    timestamp: float


@dataclass
class Candle:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    tick_count: int
    start_time: float
    end_time: float


@dataclass
class PriceBook:
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_update: float = 0.0


class CandleAggregator:
    """Aggregates ticks into 1-minute candles."""

    def __init__(self, symbol: str, interval_sec: int = 60):
        self.symbol = symbol
        self.interval = interval_sec
        self._lock = threading.Lock()
        self._current: Candle | None = None
        self._completed: list[Candle] = []
        self._max_history = 1440  # 24h of 1m candles

    def add_tick(self, mid: float, ts: float):
        with self._lock:
            bucket = int(ts // self.interval) * self.interval
            if self._current is None or self._current.start_time != bucket:
                if self._current is not None:
                    self._completed.append(self._current)
                    if len(self._completed) > self._max_history:
                        self._completed = self._completed[-self._max_history:]
                self._current = Candle(
                    symbol=self.symbol,
                    open=mid, high=mid, low=mid, close=mid,
                    tick_count=1,
                    start_time=bucket,
                    end_time=bucket + self.interval,
                )
            else:
                self._current.high = max(self._current.high, mid)
                self._current.low = min(self._current.low, mid)
                self._current.close = mid
                self._current.tick_count += 1

    def get_candles(self, count: int = 100) -> list[dict]:
        with self._lock:
            candles = list(self._completed[-count:])
            if self._current:
                candles.append(self._current)
        return [
            {
                "symbol": c.symbol,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "tick_count": c.tick_count,
                "start_time": datetime.fromtimestamp(c.start_time, tz=timezone.utc).isoformat(),
                "end_time": datetime.fromtimestamp(c.end_time, tz=timezone.utc).isoformat(),
            }
            for c in candles
        ]


class FIXPriceClient:
    """Market data client over FIX 4.4 price connection."""

    def __init__(self, connector: FIXConnector):
        self.connector = connector
        self.connector.on_message = self._on_message
        self._prices: dict[str, PriceBook] = {}
        self._aggregators: dict[str, CandleAggregator] = {}
        self._subscriptions: dict[str, str] = {}  # symbol -> mdReqID
        self._lock = threading.Lock()
        self._req_counter = 0
        self._callbacks: list = []

    def on_tick(self, callback):
        """Register a tick callback: callback(symbol, bid, ask)."""
        self._callbacks.append(callback)

    # ── Market data requests ─────────────────────────────────────

    def subscribe(self, symbol: str):
        """Send a Market Data Request (35=V) for symbol."""
        if not self.connector.is_logged_in:
            logger.warning("Cannot subscribe, not logged in")
            return
        self._req_counter += 1
        req_id = f"MD_{self._req_counter}"

        msg = self.connector.build_message(b"V")
        msg.append_pair(262, req_id.encode())          # MDReqID
        msg.append_pair(263, b"1")                      # SubscriptionRequestType = Snapshot+Updates
        msg.append_pair(264, b"1")                      # MarketDepth = Top of book
        msg.append_pair(267, b"2")                      # NoMDEntryTypes
        msg.append_pair(269, b"0")                      # MDEntryType = Bid
        msg.append_pair(269, b"1")                      # MDEntryType = Offer
        msg.append_pair(146, b"1")                      # NoRelatedSym
        msg.append_pair(55, symbol.encode())            # Symbol
        self.connector.send_message(msg)

        with self._lock:
            self._subscriptions[symbol] = req_id
            if symbol not in self._prices:
                self._prices[symbol] = PriceBook(symbol=symbol)
            if symbol not in self._aggregators:
                self._aggregators[symbol] = CandleAggregator(symbol)

        logger.info("Subscribed to %s (reqID=%s)", symbol, req_id)

    def unsubscribe(self, symbol: str):
        """Cancel a market data subscription."""
        with self._lock:
            req_id = self._subscriptions.pop(symbol, None)
        if not req_id:
            return

        msg = self.connector.build_message(b"V")
        msg.append_pair(262, req_id.encode())
        msg.append_pair(263, b"2")  # Unsubscribe
        msg.append_pair(264, b"1")
        msg.append_pair(267, b"2")
        msg.append_pair(269, b"0")
        msg.append_pair(269, b"1")
        msg.append_pair(146, b"1")
        msg.append_pair(55, symbol.encode())
        self.connector.send_message(msg)
        logger.info("Unsubscribed from %s", symbol)

    # ── Message handling ─────────────────────────────────────────

    def _on_message(self, msg: simplefix.FixMessage):
        msg_type = _get_field(msg, 35)
        if msg_type == "W":
            self._handle_snapshot(msg)
        elif msg_type == "X":
            self._handle_incremental(msg)

    def _handle_snapshot(self, msg: simplefix.FixMessage):
        """Parse Market Data Snapshot / Full Refresh (35=W)."""
        symbol = _get_field(msg, 55)
        if not symbol:
            return

        bid, ask, bid_size, ask_size = self._extract_entries(msg)
        self._update_price(symbol, bid, ask, bid_size, ask_size)

    def _handle_incremental(self, msg: simplefix.FixMessage):
        """Parse Market Data Incremental Refresh (35=X)."""
        # Incremental may contain multiple entries with repeating groups
        symbol = _get_field(msg, 55)
        if not symbol:
            # Try to find symbol from entries
            symbols = get_all_fields(msg, 55)
            if symbols:
                symbol = symbols[0]
            else:
                return

        bid, ask, bid_size, ask_size = self._extract_entries(msg)
        with self._lock:
            book = self._prices.get(symbol)
        if book:
            if bid > 0:
                self._update_price(symbol, bid, ask or book.ask, bid_size, ask_size)
            elif ask > 0:
                self._update_price(symbol, book.bid, ask, bid_size, ask_size)

    def _extract_entries(self, msg: simplefix.FixMessage):
        """Walk repeating group to extract bid/ask prices and sizes."""
        bid = 0.0
        ask = 0.0
        bid_size = 0.0
        ask_size = 0.0

        entry_types = get_all_fields(msg, 269)
        prices = get_all_fields(msg, 270)
        sizes = get_all_fields(msg, 271)

        for i, et in enumerate(entry_types):
            price = float(prices[i]) if i < len(prices) else 0.0
            size = float(sizes[i]) if i < len(sizes) else 0.0
            if et == "0":  # Bid
                bid = price
                bid_size = size
            elif et == "1":  # Offer
                ask = price
                ask_size = size

        return bid, ask, bid_size, ask_size

    def _update_price(self, symbol: str, bid: float, ask: float, bid_size: float, ask_size: float):
        now = time.time()
        with self._lock:
            if symbol not in self._prices:
                self._prices[symbol] = PriceBook(symbol=symbol)
            book = self._prices[symbol]
            if bid > 0:
                book.bid = bid
                book.bid_size = bid_size
            if ask > 0:
                book.ask = ask
                book.ask_size = ask_size
            book.last_update = now

            agg = self._aggregators.get(symbol)
        if agg and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            agg.add_tick(mid, now)

        for cb in self._callbacks:
            try:
                cb(symbol, bid, ask)
            except Exception:
                logger.exception("Tick callback error")

    # ── Public API ───────────────────────────────────────────────

    def get_price(self, symbol: str) -> dict | None:
        with self._lock:
            book = self._prices.get(symbol)
        if not book or book.last_update == 0:
            return None
        return {
            "symbol": book.symbol,
            "bid": book.bid,
            "ask": book.ask,
            "bid_size": book.bid_size,
            "ask_size": book.ask_size,
            "spread": round(book.ask - book.bid, 6) if book.bid and book.ask else 0,
            "mid": round((book.bid + book.ask) / 2, 6) if book.bid and book.ask else 0,
            "last_update": datetime.fromtimestamp(book.last_update, tz=timezone.utc).isoformat(),
        }

    def get_all_prices(self) -> dict:
        with self._lock:
            symbols = list(self._prices.keys())
        return {s: self.get_price(s) for s in symbols if self.get_price(s)}

    def get_candles(self, symbol: str, count: int = 100) -> list[dict]:
        with self._lock:
            agg = self._aggregators.get(symbol)
        if not agg:
            return []
        return agg.get_candles(count)
