"""
Tick poller for the data pipeline.

Polls the FIX API for price updates and delivers them as ticks.
Also periodically triggers candle aggregation from stored ticks.
"""

import time
import logging
import threading
from typing import Callable

import requests

logger = logging.getLogger("tick_poller")


class TickPoller:
    """Polls FIX API prices and triggers candle aggregation."""

    def __init__(
        self,
        fix_api_url: str = "http://localhost:5200",
        poll_interval: float = 1.0,
        tick_callback: Callable | None = None,
        symbol_ids: list[str] | None = None,
        candle_intervals: list[str] | None = None,
        store=None,
    ):
        self._fix_api_url = fix_api_url.rstrip("/")
        self._poll_interval = poll_interval
        self._tick_callback = tick_callback
        self._symbol_ids = symbol_ids or []
        self._candle_intervals = candle_intervals or []
        self._store = store
        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._agg_thread: threading.Thread | None = None
        self._last_prices: dict[str, tuple[float, float]] = {}  # symbol → (bid, ask)

    def start(self):
        """Start the polling and aggregation threads."""
        self._running = True

        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="tick-poller"
        )
        self._poll_thread.start()

        if self._candle_intervals and self._store:
            self._agg_thread = threading.Thread(
                target=self._aggregation_loop, daemon=True, name="candle-aggregator"
            )
            self._agg_thread.start()

        logger.info(
            "Tick poller started (url=%s, interval=%.1fs, symbols=%d, candle_intervals=%s)",
            self._fix_api_url, self._poll_interval,
            len(self._symbol_ids), self._candle_intervals,
        )

    def stop(self):
        """Stop the polling threads."""
        self._running = False
        logger.info("Tick poller stopped")

    def _poll_loop(self):
        """Poll FIX API for price changes and deliver ticks."""
        while self._running:
            try:
                resp = requests.get(
                    f"{self._fix_api_url}/prices",
                    timeout=5,
                )
                if resp.status_code == 200:
                    prices = resp.json()
                    self._process_prices(prices)
                else:
                    logger.warning("FIX API returned %d", resp.status_code)
            except requests.ConnectionError:
                logger.warning("FIX API unreachable at %s", self._fix_api_url)
            except Exception:
                logger.exception("Tick poll error")

            time.sleep(self._poll_interval)

    def _process_prices(self, prices: dict):
        """Compare prices to last poll and emit ticks for changes."""
        for symbol, data in prices.items():
            if self._symbol_ids and symbol not in self._symbol_ids:
                continue

            bid = data.get("bid", 0)
            ask = data.get("ask", 0)
            if bid <= 0 or ask <= 0:
                continue

            last = self._last_prices.get(symbol)
            if last and last[0] == bid and last[1] == ask:
                continue  # No change

            self._last_prices[symbol] = (bid, ask)

            if self._tick_callback:
                self._tick_callback(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    bid_size=data.get("bid_size", 0),
                    ask_size=data.get("ask_size", 0),
                )

    def _aggregation_loop(self):
        """Periodically aggregate ticks into candles."""
        # Wait a bit before first aggregation to let ticks accumulate
        time.sleep(self._poll_interval * 10)

        while self._running:
            for symbol in self._symbol_ids:
                for interval in self._candle_intervals:
                    try:
                        self._store.aggregate_ticks_to_candles(symbol, interval)
                    except Exception:
                        logger.exception(
                            "Candle aggregation failed for %s/%s", symbol, interval
                        )

            # Aggregate every 60 seconds
            for _ in range(60):
                if not self._running:
                    return
                time.sleep(1)
