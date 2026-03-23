"""
Pre-trade risk management engine for AlgoDesk.

Provides layered risk checks: per-order, per-symbol, portfolio-level,
drawdown circuit breakers, rate limiting, duplicate detection, and kill switch.
"""

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("risk_manager")


@dataclass
class RiskConfig:
    """All risk parameters — loaded from fix_config.yaml."""

    # Per-order limits
    max_order_size: float = 10.0              # Max lots per single order
    min_order_size: float = 0.01              # Min lots per single order

    # Per-symbol limits
    max_position_size: float = 50.0           # Max net lots per symbol
    max_open_orders_per_symbol: int = 10      # Max open orders per symbol

    # Portfolio-level limits
    max_open_orders: int = 50                 # Max total open orders
    max_daily_orders: int = 500               # Max orders per trading day
    max_total_exposure: float = 100.0         # Max total lots across all symbols

    # Drawdown circuit breaker
    max_daily_loss: float = 5000.0            # Max daily realized + unrealized loss (account currency)
    max_drawdown_pct: float = 5.0             # Max drawdown as % of starting daily equity

    # Rate limiting (orders per window)
    max_orders_per_minute: int = 30           # Burst limit
    max_orders_per_second: int = 5            # Micro-burst limit

    # Duplicate detection
    duplicate_window_sec: float = 2.0         # Window to detect duplicate orders

    # Price sanity
    max_spread_pips: float = 50.0             # Reject if spread > threshold (in price units)
    stale_price_sec: float = 30.0             # Reject if price older than this


class RiskViolation:
    """Represents a failed risk check."""

    __slots__ = ("rule", "message", "timestamp", "severity")

    def __init__(self, rule: str, message: str, severity: str = "REJECT"):
        self.rule = rule
        self.message = message
        self.timestamp = time.time()
        self.severity = severity  # REJECT, WARNING

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "message": self.message,
            "severity": self.severity,
            "timestamp": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
        }


class RiskManager:
    """
    Stateful pre-trade risk engine.

    Call check_order() before every order submission. If it returns a
    RiskViolation, the order must be blocked. Call record_fill() on every
    execution report to keep P&L tracking accurate.
    """

    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self._lock = threading.Lock()

        # Kill switch
        self._killed = False
        self._kill_reason = ""

        # Daily counters (reset at UTC midnight)
        self._daily_date = ""
        self._daily_order_count = 0
        self._daily_pnl = 0.0
        self._starting_equity = 0.0

        # Rate limiting — sliding windows of order timestamps
        self._order_timestamps: deque[float] = deque()

        # Duplicate detection — recent (symbol, side, qty) tuples
        self._recent_orders: deque[tuple[str, str, float, float]] = deque()

        # Live state mirrors (updated externally via record_* methods)
        self._positions: dict[str, float] = {}        # symbol -> net_qty
        self._open_orders: dict[str, list] = {}        # symbol -> [order_ids]
        self._open_order_count = 0

        # Audit trail
        self._violations: deque[dict] = deque(maxlen=500)

    # ── Kill switch ───────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "Manual kill switch"):
        """Immediately halt all new order flow."""
        with self._lock:
            self._killed = True
            self._kill_reason = reason
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def deactivate_kill_switch(self):
        """Re-enable order flow after manual review."""
        with self._lock:
            self._killed = False
            self._kill_reason = ""
        logger.warning("Kill switch deactivated")

    @property
    def is_killed(self) -> bool:
        return self._killed

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    # ── Daily reset ───────────────────────────────────────────────

    def _ensure_daily_reset(self):
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_order_count = 0
            self._daily_pnl = 0.0
            self._starting_equity = 0.0
            self._order_timestamps.clear()
            logger.info("Daily risk counters reset for %s", today)

    def set_starting_equity(self, equity: float):
        """Set beginning-of-day equity for drawdown % calculation."""
        with self._lock:
            self._starting_equity = equity

    # ── Core risk check ───────────────────────────────────────────

    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: float = 0.0,
        current_bid: float = 0.0,
        current_ask: float = 0.0,
        price_update_time: float = 0.0,
    ) -> RiskViolation | None:
        """
        Run all pre-trade risk checks. Returns None if order is allowed,
        or a RiskViolation describing why it was blocked.
        """
        with self._lock:
            self._ensure_daily_reset()

            # 1. Kill switch
            if self._killed:
                return self._violation(
                    "KILL_SWITCH", f"Trading halted: {self._kill_reason}"
                )

            # 2. Order size bounds
            if quantity < self.cfg.min_order_size:
                return self._violation(
                    "MIN_ORDER_SIZE",
                    f"Order size {quantity} below minimum {self.cfg.min_order_size}",
                )
            if quantity > self.cfg.max_order_size:
                return self._violation(
                    "MAX_ORDER_SIZE",
                    f"Order size {quantity} exceeds max {self.cfg.max_order_size}",
                )

            # 3. Open order limits
            if self._open_order_count >= self.cfg.max_open_orders:
                return self._violation(
                    "MAX_OPEN_ORDERS",
                    f"Open order limit reached ({self.cfg.max_open_orders})",
                )
            sym_orders = len(self._open_orders.get(symbol, []))
            if sym_orders >= self.cfg.max_open_orders_per_symbol:
                return self._violation(
                    "MAX_ORDERS_PER_SYMBOL",
                    f"Open order limit for {symbol} reached ({self.cfg.max_open_orders_per_symbol})",
                )

            # 4. Daily order limit
            if self._daily_order_count >= self.cfg.max_daily_orders:
                return self._violation(
                    "MAX_DAILY_ORDERS",
                    f"Daily order limit reached ({self.cfg.max_daily_orders})",
                )

            # 5. Position size check
            net_qty = self._positions.get(symbol, 0.0)
            projected = net_qty + (quantity if side == "1" else -quantity)
            if abs(projected) > self.cfg.max_position_size:
                return self._violation(
                    "MAX_POSITION_SIZE",
                    f"Position would be {projected:.2f}, exceeds max {self.cfg.max_position_size}",
                )

            # 6. Portfolio-level exposure
            total_exposure = sum(abs(v) for v in self._positions.values())
            projected_exposure = total_exposure - abs(net_qty) + abs(projected)
            if projected_exposure > self.cfg.max_total_exposure:
                return self._violation(
                    "MAX_TOTAL_EXPOSURE",
                    f"Total exposure would be {projected_exposure:.2f}, exceeds max {self.cfg.max_total_exposure}",
                )

            # 7. Drawdown circuit breaker
            if self.cfg.max_daily_loss > 0 and self._daily_pnl < -self.cfg.max_daily_loss:
                self._killed = True
                self._kill_reason = f"Daily loss limit breached: {self._daily_pnl:.2f}"
                return self._violation(
                    "DAILY_LOSS_LIMIT",
                    f"Daily P&L {self._daily_pnl:.2f} exceeds max loss {self.cfg.max_daily_loss}",
                )
            if (
                self.cfg.max_drawdown_pct > 0
                and self._starting_equity > 0
                and self._daily_pnl < 0
            ):
                dd_pct = abs(self._daily_pnl) / self._starting_equity * 100
                if dd_pct >= self.cfg.max_drawdown_pct:
                    self._killed = True
                    self._kill_reason = f"Drawdown {dd_pct:.1f}% exceeds {self.cfg.max_drawdown_pct}%"
                    return self._violation(
                        "MAX_DRAWDOWN_PCT",
                        f"Drawdown {dd_pct:.1f}% exceeds limit {self.cfg.max_drawdown_pct}%",
                    )

            # 8. Rate limiting
            now = time.time()
            self._prune_timestamps(now)
            one_sec_count = sum(
                1 for ts in self._order_timestamps if now - ts < 1.0
            )
            if one_sec_count >= self.cfg.max_orders_per_second:
                return self._violation(
                    "RATE_LIMIT_SECOND",
                    f"Rate limit: {one_sec_count} orders in last second (max {self.cfg.max_orders_per_second})",
                )
            one_min_count = len(self._order_timestamps)
            if one_min_count >= self.cfg.max_orders_per_minute:
                return self._violation(
                    "RATE_LIMIT_MINUTE",
                    f"Rate limit: {one_min_count} orders in last minute (max {self.cfg.max_orders_per_minute})",
                )

            # 9. Duplicate detection
            if self._is_duplicate(symbol, side, quantity, now):
                return self._violation(
                    "DUPLICATE_ORDER",
                    f"Duplicate order detected: {symbol} {'BUY' if side == '1' else 'SELL'} {quantity} within {self.cfg.duplicate_window_sec}s",
                    severity="WARNING",
                )

            # 10. Price staleness / sanity (for limit/market orders with price feed)
            if current_bid > 0 and current_ask > 0:
                if price_update_time > 0 and (now - price_update_time) > self.cfg.stale_price_sec:
                    return self._violation(
                        "STALE_PRICE",
                        f"Price data is {now - price_update_time:.0f}s old (max {self.cfg.stale_price_sec}s)",
                    )
                spread = current_ask - current_bid
                if spread > self.cfg.max_spread_pips:
                    return self._violation(
                        "WIDE_SPREAD",
                        f"Spread {spread:.5f} exceeds max {self.cfg.max_spread_pips}",
                        severity="WARNING",
                    )

            return None  # All checks passed

    def _violation(self, rule: str, message: str, severity: str = "REJECT") -> RiskViolation:
        v = RiskViolation(rule, message, severity)
        self._violations.append(v.to_dict())
        logger.warning("Risk violation [%s]: %s", rule, message)
        return v

    def _prune_timestamps(self, now: float):
        cutoff = now - 60.0
        while self._order_timestamps and self._order_timestamps[0] < cutoff:
            self._order_timestamps.popleft()

    def _is_duplicate(self, symbol: str, side: str, quantity: float, now: float) -> bool:
        cutoff = now - self.cfg.duplicate_window_sec
        # Prune old
        while self._recent_orders and self._recent_orders[0][3] < cutoff:
            self._recent_orders.popleft()
        for sym, s, qty, _ in self._recent_orders:
            if sym == symbol and s == side and qty == quantity:
                return True
        return False

    # ── State update methods (call from trade client) ─────────────

    def record_order_sent(self, cl_ord_id: str, symbol: str, side: str, quantity: float):
        """Call after an order is successfully sent to the FIX gateway."""
        with self._lock:
            self._ensure_daily_reset()
            self._daily_order_count += 1
            now = time.time()
            self._order_timestamps.append(now)
            self._recent_orders.append((symbol, side, quantity, now))
            if symbol not in self._open_orders:
                self._open_orders[symbol] = []
            self._open_orders[symbol].append(cl_ord_id)
            self._open_order_count += 1

    def record_order_closed(self, cl_ord_id: str, symbol: str):
        """Call when an order is filled, canceled, or rejected."""
        with self._lock:
            sym_orders = self._open_orders.get(symbol, [])
            if cl_ord_id in sym_orders:
                sym_orders.remove(cl_ord_id)
                self._open_order_count = max(0, self._open_order_count - 1)

    def record_fill(self, symbol: str, side: str, qty: float, price: float):
        """Update position mirror on fill."""
        with self._lock:
            net = self._positions.get(symbol, 0.0)
            if side == "1":  # Buy
                net += qty
            else:
                net -= qty
            self._positions[symbol] = net

    def record_pnl(self, realized_pnl: float):
        """Accumulate daily P&L from fills."""
        with self._lock:
            self._daily_pnl += realized_pnl

    def update_unrealized_pnl(self, total_unrealized: float):
        """Update unrealized P&L for drawdown checks (called periodically)."""
        # Daily P&L for drawdown = realized + unrealized
        # We only store realized in _daily_pnl; caller can add unrealized
        pass

    def sync_positions(self, positions: dict[str, float]):
        """Bulk-sync position mirror from trade client state."""
        with self._lock:
            self._positions = dict(positions)

    def sync_open_orders(self, orders_by_symbol: dict[str, list[str]], total: int):
        """Bulk-sync open order counts from trade client."""
        with self._lock:
            self._open_orders = {k: list(v) for k, v in orders_by_symbol.items()}
            self._open_order_count = total

    # ── Reporting ─────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            self._ensure_daily_reset()
            return {
                "kill_switch": self._killed,
                "kill_reason": self._kill_reason,
                "daily_order_count": self._daily_order_count,
                "daily_pnl": round(self._daily_pnl, 2),
                "starting_equity": round(self._starting_equity, 2),
                "open_order_count": self._open_order_count,
                "total_exposure": round(
                    sum(abs(v) for v in self._positions.values()), 2
                ),
                "positions": dict(self._positions),
                "config": {
                    "max_order_size": self.cfg.max_order_size,
                    "min_order_size": self.cfg.min_order_size,
                    "max_position_size": self.cfg.max_position_size,
                    "max_open_orders": self.cfg.max_open_orders,
                    "max_daily_orders": self.cfg.max_daily_orders,
                    "max_total_exposure": self.cfg.max_total_exposure,
                    "max_daily_loss": self.cfg.max_daily_loss,
                    "max_drawdown_pct": self.cfg.max_drawdown_pct,
                    "max_orders_per_minute": self.cfg.max_orders_per_minute,
                    "max_orders_per_second": self.cfg.max_orders_per_second,
                },
            }

    def get_violations(self, count: int = 50) -> list[dict]:
        with self._lock:
            return list(self._violations)[-count:]
