"""
FIX 4.4 Trade client for cTrader.

Handles new orders (35=D), cancel requests (35=F), execution reports (35=8),
position/order tracking, and pre-trade risk checks via RiskManager.
"""

import time
import uuid
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import simplefix

from fix_connector import FIXConnector, _get_field, msg_to_dict
from risk_manager import RiskManager, RiskConfig

logger = logging.getLogger("fix_trade")


class OrderSide(str, Enum):
    BUY = "1"
    SELL = "2"


class OrderType(str, Enum):
    MARKET = "1"
    LIMIT = "2"
    STOP = "3"


class TimeInForce(str, Enum):
    GTC = "1"    # Good Till Cancel
    IOC = "3"    # Immediate or Cancel
    FOK = "4"    # Fill or Kill


class OrdStatus(str, Enum):
    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    CANCELED = "4"
    REJECTED = "8"
    PENDING_NEW = "A"
    PENDING_CANCEL = "6"


@dataclass
class Order:
    cl_ord_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float = 0.0
    stop_price: float = 0.0
    status: str = "PENDING"
    order_id: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    text: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cl_ord_id": self.cl_ord_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": "BUY" if self.side == "1" else "SELL",
            "type": {"1": "MARKET", "2": "LIMIT", "3": "STOP"}.get(self.order_type, self.order_type),
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "text": self.text,
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "updated_at": datetime.fromtimestamp(self.updated_at, tz=timezone.utc).isoformat(),
        }


@dataclass
class Position:
    symbol: str
    long_qty: float = 0.0
    short_qty: float = 0.0
    net_qty: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "long_qty": self.long_qty,
            "short_qty": self.short_qty,
            "net_qty": self.net_qty,
            "avg_entry_price": self.avg_entry_price,
            "unrealized_pnl": self.unrealized_pnl,
        }


# ── Risk limits ──────────────────────────────────────────────────

@dataclass
class RiskLimits:
    max_order_size: float = 10.0          # Max lots per order
    max_position_size: float = 50.0       # Max net lots per symbol
    max_open_orders: int = 50             # Max concurrent open orders
    max_daily_orders: int = 500           # Max orders per day


class FIXTradeClient:
    """Trade client over FIX 4.4 trade connection with integrated risk management."""

    def __init__(
        self,
        connector: FIXConnector,
        risk_limits: RiskLimits | None = None,
        risk_manager: RiskManager | None = None,
    ):
        self.connector = connector
        self.connector.on_message = self._on_message
        self.risk = risk_limits or RiskLimits()

        # Use advanced risk manager if provided, otherwise create one from legacy limits
        if risk_manager:
            self.risk_manager = risk_manager
        else:
            self.risk_manager = RiskManager(RiskConfig(
                max_order_size=self.risk.max_order_size,
                max_position_size=self.risk.max_position_size,
                max_open_orders=self.risk.max_open_orders,
                max_daily_orders=self.risk.max_daily_orders,
            ))

        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._lock = threading.Lock()
        self._daily_order_count = 0
        self._daily_reset_date = ""
        self._callbacks: list = []
        self._price_feed = None  # Set externally for price-aware risk checks

    def set_price_feed(self, price_client):
        """Set reference to price client for spread/staleness checks."""
        self._price_feed = price_client

    def on_execution(self, callback):
        """Register execution callback: callback(order_dict)."""
        self._callbacks.append(callback)

    # ── Order placement ──────────────────────────────────────────

    def new_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = 0.0,
        stop_price: float = 0.0,
        time_in_force: str = "1",
    ) -> dict:
        """Send a New Order Single (35=D). Returns order dict or error."""
        if not self.connector.is_logged_in:
            return {"error": "Not logged in to trade connection"}

        # Gather price context for risk checks
        current_bid, current_ask, price_update_time = 0.0, 0.0, 0.0
        if self._price_feed:
            price_data = self._price_feed.get_price(symbol)
            if price_data:
                current_bid = price_data.get("bid", 0.0)
                current_ask = price_data.get("ask", 0.0)
                last_update = price_data.get("last_update", "")
                if last_update:
                    try:
                        dt = datetime.fromisoformat(last_update)
                        price_update_time = dt.timestamp()
                    except (ValueError, TypeError):
                        pass

        # Run pre-trade risk checks via RiskManager
        violation = self.risk_manager.check_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            current_bid=current_bid,
            current_ask=current_ask,
            price_update_time=price_update_time,
        )
        if violation:
            return {"error": violation.message, "risk_rule": violation.rule}

        cl_ord_id = f"ORD_{uuid.uuid4().hex[:12]}"
        msg = self.connector.build_message(b"D")
        msg.append_pair(11, cl_ord_id.encode())         # ClOrdID
        msg.append_pair(55, symbol.encode())             # Symbol
        msg.append_pair(54, side.encode())               # Side
        msg.append_pair(40, order_type.encode())         # OrdType
        msg.append_pair(38, str(quantity).encode())      # OrderQty (lots)
        msg.append_pair(59, time_in_force.encode())      # TimeInForce

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
        msg.append_pair(60, ts.encode())                 # TransactTime

        if order_type == OrderType.LIMIT and price > 0:
            msg.append_pair(44, str(price).encode())     # Price
        elif order_type == OrderType.STOP and stop_price > 0:
            msg.append_pair(99, str(stop_price).encode())  # StopPx

        self.connector.send_message(msg)

        # Record with risk manager
        self.risk_manager.record_order_sent(cl_ord_id, symbol, side, quantity)
        self._daily_order_count += 1

        order = Order(
            cl_ord_id=cl_ord_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
        )
        with self._lock:
            self._orders[cl_ord_id] = order

        logger.info("Order sent: %s %s %s qty=%.2f", cl_ord_id, symbol,
                     "BUY" if side == "1" else "SELL", quantity)
        return order.to_dict()

    def cancel_order(self, cl_ord_id: str) -> dict:
        """Send an Order Cancel Request (35=F)."""
        if not self.connector.is_logged_in:
            return {"error": "Not logged in to trade connection"}

        with self._lock:
            order = self._orders.get(cl_ord_id)
        if not order:
            return {"error": f"Order {cl_ord_id} not found"}
        if order.status not in ("PENDING", "NEW", "PARTIALLY_FILLED"):
            return {"error": f"Order {cl_ord_id} cannot be canceled (status={order.status})"}

        cancel_id = f"CXL_{uuid.uuid4().hex[:12]}"
        msg = self.connector.build_message(b"F")
        msg.append_pair(41, cl_ord_id.encode())          # OrigClOrdID
        msg.append_pair(11, cancel_id.encode())          # ClOrdID (cancel req id)
        msg.append_pair(55, order.symbol.encode())       # Symbol
        msg.append_pair(54, order.side.encode())         # Side
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
        msg.append_pair(60, ts.encode())                 # TransactTime

        self.connector.send_message(msg)
        order.status = "PENDING_CANCEL"
        order.updated_at = time.time()

        logger.info("Cancel request sent for %s", cl_ord_id)
        return order.to_dict()

    # ── Execution report parsing ─────────────────────────────────

    def _on_message(self, msg: simplefix.FixMessage):
        msg_type = _get_field(msg, 35)
        if msg_type == "8":
            self._handle_execution_report(msg)
        elif msg_type == "9":
            self._handle_cancel_reject(msg)

    def _handle_execution_report(self, msg: simplefix.FixMessage):
        """Parse Execution Report (35=8)."""
        cl_ord_id = _get_field(msg, 11)
        order_id = _get_field(msg, 37)
        exec_type = _get_field(msg, 150)
        ord_status = _get_field(msg, 39)
        symbol = _get_field(msg, 55)
        side = _get_field(msg, 54)

        cum_qty = float(_get_field(msg, 14) or 0)
        avg_px = float(_get_field(msg, 6) or 0)
        last_qty = float(_get_field(msg, 32) or 0)
        last_px = float(_get_field(msg, 31) or 0)
        text = _get_field(msg, 58) or ""

        status_map = {
            "0": "NEW", "1": "PARTIALLY_FILLED", "2": "FILLED",
            "4": "CANCELED", "8": "REJECTED", "A": "PENDING_NEW", "6": "PENDING_CANCEL",
        }

        with self._lock:
            order = self._orders.get(cl_ord_id)
            if order is None:
                # Could be an order from a previous session
                orig_id = _get_field(msg, 41)
                if orig_id:
                    order = self._orders.get(orig_id)

        if order:
            order.order_id = order_id or order.order_id
            order.status = status_map.get(ord_status, ord_status or order.status)
            order.filled_qty = cum_qty
            order.avg_price = avg_px
            order.text = text
            order.updated_at = time.time()

            # Update position on fills
            if exec_type in ("F", "1", "2") and last_qty > 0:
                self._update_position(order.symbol, order.side, last_qty, last_px)
                self.risk_manager.record_fill(order.symbol, order.side, last_qty, last_px)

            # Notify risk manager when order is terminal
            if order.status in ("FILLED", "CANCELED", "REJECTED"):
                self.risk_manager.record_order_closed(order.cl_ord_id, order.symbol)

            logger.info(
                "ExecReport: %s status=%s filled=%.2f@%.5f %s",
                cl_ord_id, order.status, cum_qty, avg_px, text,
            )

            for cb in self._callbacks:
                try:
                    cb(order.to_dict())
                except Exception:
                    logger.exception("Execution callback error")
        else:
            logger.warning("ExecReport for unknown order: clOrdID=%s orderID=%s", cl_ord_id, order_id)

    def _handle_cancel_reject(self, msg: simplefix.FixMessage):
        """Parse Order Cancel Reject (35=9)."""
        cl_ord_id = _get_field(msg, 11)
        orig_id = _get_field(msg, 41)
        text = _get_field(msg, 58) or ""
        logger.warning("Cancel rejected: orig=%s reason=%s", orig_id, text)

        with self._lock:
            order = self._orders.get(orig_id)
        if order and order.status == "PENDING_CANCEL":
            order.status = "NEW"
            order.text = f"Cancel rejected: {text}"
            order.updated_at = time.time()

    def _update_position(self, symbol: str, side: str, qty: float, price: float):
        """Track position from fills."""
        with self._lock:
            if symbol not in self._positions:
                self._positions[symbol] = Position(symbol=symbol)
            pos = self._positions[symbol]

            if side == "1":  # Buy
                pos.long_qty += qty
            else:  # Sell
                pos.short_qty += qty

            pos.net_qty = pos.long_qty - pos.short_qty

            # Weighted average entry
            if pos.net_qty != 0:
                total_value = pos.avg_entry_price * (abs(pos.net_qty) - qty) + price * qty
                pos.avg_entry_price = total_value / abs(pos.net_qty) if pos.net_qty != 0 else 0

    # ── Public API ───────────────────────────────────────────────

    def get_orders(self, active_only: bool = False) -> list[dict]:
        with self._lock:
            orders = list(self._orders.values())
        if active_only:
            orders = [o for o in orders if o.status in ("PENDING", "NEW", "PARTIALLY_FILLED", "PENDING_CANCEL")]
        return [o.to_dict() for o in orders]

    def get_order(self, cl_ord_id: str) -> dict | None:
        with self._lock:
            order = self._orders.get(cl_ord_id)
        return order.to_dict() if order else None

    def get_positions(self) -> list[dict]:
        with self._lock:
            return [p.to_dict() for p in self._positions.values() if p.net_qty != 0]

    def get_account_summary(self) -> dict:
        with self._lock:
            positions = list(self._positions.values())
            orders = list(self._orders.values())
        return {
            "open_positions": len([p for p in positions if p.net_qty != 0]),
            "active_orders": len([o for o in orders if o.status in ("NEW", "PARTIALLY_FILLED")]),
            "total_orders_today": self._daily_order_count,
            "risk_limits": {
                "max_order_size": self.risk.max_order_size,
                "max_position_size": self.risk.max_position_size,
                "max_open_orders": self.risk.max_open_orders,
                "max_daily_orders": self.risk.max_daily_orders,
            },
            "risk_status": self.risk_manager.get_status(),
        }
