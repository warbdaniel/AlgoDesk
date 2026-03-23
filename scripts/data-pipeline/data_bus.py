"""
Lightweight event bus for inter-service communication.

Provides pub/sub within the trading desk: tick events, regime changes,
trade signals, and execution reports flow between services without
tight coupling. Supports both in-process callbacks and HTTP webhook
delivery for cross-service events.
"""

import time
import logging
import threading
import queue
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger("data_bus")


class EventType(str, Enum):
    TICK = "tick"
    CANDLE_CLOSE = "candle_close"
    REGIME_CHANGE = "regime_change"
    SIGNAL = "signal"
    ORDER_FILL = "order_fill"
    RISK_ALERT = "risk_alert"
    SYSTEM = "system"


@dataclass
class Event:
    event_type: EventType
    source: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.event_type.value}_{int(self.timestamp * 1000)}"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }


@dataclass
class WebhookSubscription:
    url: str
    event_types: list[EventType]
    name: str = ""
    active: bool = True
    failures: int = 0
    max_failures: int = 5


class DataBus:
    """Thread-safe pub/sub event bus with webhook delivery."""

    def __init__(self, max_history: int = 1000):
        self._callbacks: dict[EventType, list] = defaultdict(list)
        self._webhooks: dict[str, WebhookSubscription] = {}
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._max_history = max_history
        self._event_count = 0

        # Async webhook delivery queue
        self._webhook_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._webhook_thread = threading.Thread(
            target=self._webhook_delivery_loop, daemon=True, name="webhook-delivery"
        )
        self._webhook_thread.start()
        self._running = True

    # ── In-process subscriptions ─────────────────────────────────

    def subscribe(self, event_type: EventType, callback):
        """Register callback for an event type. callback(event: Event)."""
        with self._lock:
            self._callbacks[event_type].append(callback)
        logger.info("Subscribed callback to %s", event_type.value)

    def unsubscribe(self, event_type: EventType, callback):
        with self._lock:
            try:
                self._callbacks[event_type].remove(callback)
            except ValueError:
                pass

    # ── Webhook subscriptions ────────────────────────────────────

    def add_webhook(self, name: str, url: str,
                    event_types: list[str]) -> str:
        """Register a webhook URL for event delivery."""
        types = [EventType(et) for et in event_types]
        sub = WebhookSubscription(url=url, event_types=types, name=name)
        with self._lock:
            self._webhooks[name] = sub
        logger.info("Webhook registered: %s → %s for %s",
                     name, url, [t.value for t in types])
        return name

    def remove_webhook(self, name: str):
        with self._lock:
            self._webhooks.pop(name, None)
        logger.info("Webhook removed: %s", name)

    def list_webhooks(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": w.name,
                    "url": w.url,
                    "event_types": [t.value for t in w.event_types],
                    "active": w.active,
                    "failures": w.failures,
                }
                for w in self._webhooks.values()
            ]

    # ── Publishing ───────────────────────────────────────────────

    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        self._event_count += 1

        # Store in history (skip high-frequency tick events to save memory)
        if event.event_type != EventType.TICK:
            with self._lock:
                self._history.append(event)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        # Dispatch to in-process callbacks
        with self._lock:
            callbacks = list(self._callbacks.get(event.event_type, []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("Event callback error for %s", event.event_type.value)

        # Queue webhook delivery
        with self._lock:
            for sub in self._webhooks.values():
                if sub.active and event.event_type in sub.event_types:
                    try:
                        self._webhook_queue.put_nowait((sub, event))
                    except queue.Full:
                        logger.warning("Webhook queue full, dropping event")

    def emit_tick(self, symbol: str, bid: float, ask: float,
                  bid_size: float = 0, ask_size: float = 0):
        """Convenience: emit a tick event."""
        self.publish(Event(
            event_type=EventType.TICK,
            source="price_feed",
            data={
                "symbol": symbol, "bid": bid, "ask": ask,
                "bid_size": bid_size, "ask_size": ask_size,
                "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else 0,
                "spread": ask - bid if bid > 0 and ask > 0 else 0,
            },
        ))

    def emit_candle_close(self, symbol: str, interval: str, candle: dict):
        """Convenience: emit candle close event."""
        self.publish(Event(
            event_type=EventType.CANDLE_CLOSE,
            source="data_pipeline",
            data={"symbol": symbol, "interval": interval, **candle},
        ))

    def emit_regime_change(self, symbol: str, old_regime: str,
                           new_regime: str, metadata: dict = None):
        """Convenience: emit regime change event."""
        self.publish(Event(
            event_type=EventType.REGIME_CHANGE,
            source="regime_detector",
            data={
                "symbol": symbol,
                "old_regime": old_regime,
                "new_regime": new_regime,
                **(metadata or {}),
            },
        ))

    def emit_signal(self, symbol: str, direction: str, strength: float,
                    source: str = "strategy", metadata: dict = None):
        """Convenience: emit a trade signal."""
        self.publish(Event(
            event_type=EventType.SIGNAL,
            source=source,
            data={
                "symbol": symbol,
                "direction": direction,
                "strength": strength,
                **(metadata or {}),
            },
        ))

    # ── Webhook delivery loop ────────────────────────────────────

    def _webhook_delivery_loop(self):
        while self._running:
            try:
                sub, event = self._webhook_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                resp = requests.post(
                    sub.url,
                    json=event.to_dict(),
                    timeout=5,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code >= 400:
                    sub.failures += 1
                    logger.warning("Webhook %s returned %d (failures=%d)",
                                   sub.name, resp.status_code, sub.failures)
                else:
                    sub.failures = 0  # reset on success
            except Exception as e:
                sub.failures += 1
                logger.warning("Webhook %s failed: %s (failures=%d)",
                               sub.name, e, sub.failures)

            # Disable after too many failures
            if sub.failures >= sub.max_failures:
                sub.active = False
                logger.error("Webhook %s disabled after %d failures",
                             sub.name, sub.failures)

    # ── History / stats ──────────────────────────────────────────

    def get_history(self, event_type: str = "", limit: int = 50) -> list[dict]:
        with self._lock:
            events = list(self._history)
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        return [e.to_dict() for e in events[-limit:]]

    def get_stats(self) -> dict:
        with self._lock:
            webhook_count = len(self._webhooks)
            callback_count = sum(len(cbs) for cbs in self._callbacks.values())
        return {
            "total_events_published": self._event_count,
            "history_size": len(self._history),
            "registered_callbacks": callback_count,
            "registered_webhooks": webhook_count,
            "webhook_queue_size": self._webhook_queue.qsize(),
        }

    def shutdown(self):
        self._running = False
