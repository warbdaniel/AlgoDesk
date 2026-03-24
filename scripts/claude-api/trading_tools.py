"""
HTTP client wrappers for AlgoDesk internal services.

Each function calls one of the existing microservices (regime-detector,
fix-api, data-pipeline, dashboard) and returns a plain-text summary
suitable for inclusion in a Claude tool_result.
"""

import json
import logging
from typing import Any

import requests

logger = logging.getLogger("trading_tools")

_TIMEOUT = 10  # seconds


# ── helpers ──────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> dict | list | str:
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def _post(url: str, payload: dict | None = None) -> dict | list | str:
    try:
        r = requests.post(url, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def _delete(url: str) -> dict | str:
    try:
        r = requests.delete(url, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def _json_summary(data: Any) -> str:
    """Return compact JSON text (for tool results)."""
    return json.dumps(data, indent=2, default=str)


# ── Regime Detector (port 5000) ──────────────────────────────────

def detect_regime(base_url: str, symbol: str, timeframe: str = "15m") -> str:
    """Classify the current market regime for a symbol."""
    data = _post(f"{base_url}/regime", {"symbol": symbol, "timeframe": timeframe})
    return _json_summary(data)


def regime_health(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/health"))


# ── FIX API (port 5200) ─────────────────────────────────────────

def get_prices(base_url: str, symbol: str | None = None) -> str:
    """Get latest bid/ask prices."""
    url = f"{base_url}/prices/{symbol}" if symbol else f"{base_url}/prices"
    return _json_summary(_get(url))


def get_candles(base_url: str, symbol: str, count: int = 50) -> str:
    """Get recent 1-minute candles."""
    return _json_summary(_get(f"{base_url}/candles/{symbol}", {"count": count}))


def subscribe_symbol(base_url: str, symbol: str) -> str:
    return _json_summary(_post(f"{base_url}/subscribe", {"symbol": symbol}))


def unsubscribe_symbol(base_url: str, symbol: str) -> str:
    return _json_summary(_delete(f"{base_url}/subscribe/{symbol}"))


def place_order(
    base_url: str,
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "MARKET",
    price: float = 0.0,
    stop_price: float = 0.0,
    time_in_force: str = "GTC",
) -> str:
    """Submit an order through the FIX trade connection."""
    payload = {
        "symbol": symbol,
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": quantity,
        "price": price,
        "stop_price": stop_price,
        "time_in_force": time_in_force,
    }
    return _json_summary(_post(f"{base_url}/orders", payload))


def cancel_order(base_url: str, cl_ord_id: str) -> str:
    return _json_summary(_delete(f"{base_url}/orders/{cl_ord_id}"))


def get_orders(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/orders"))


def get_positions(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/positions"))


def get_account(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/account"))


def get_risk_status(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/risk/status"))


def get_risk_violations(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/risk/violations"))


def activate_kill_switch(base_url: str) -> str:
    return _json_summary(_post(f"{base_url}/risk/kill"))


def deactivate_kill_switch(base_url: str) -> str:
    return _json_summary(_delete(f"{base_url}/risk/kill"))


def fix_health(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/health"))


def connection_health(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/risk/connection-health"))


# ── Data Pipeline (port 5300) ────────────────────────────────────

def get_features(base_url: str, symbol: str, interval: str = "15m") -> str:
    """Compute technical indicator feature vector."""
    return _json_summary(_get(f"{base_url}/features/{symbol}/{interval}"))


def get_feature_names(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/features/names"))


def get_pipeline_candles(
    base_url: str, symbol: str, interval: str = "15m", limit: int = 50
) -> str:
    """Get stored candles from the data pipeline."""
    return _json_summary(
        _get(f"{base_url}/candles/{symbol}/{interval}", {"limit": limit})
    )


def get_ticks(base_url: str, symbol: str, limit: int = 100) -> str:
    return _json_summary(_get(f"{base_url}/ticks/{symbol}", {"limit": limit}))


def get_latest_tick(base_url: str, symbol: str) -> str:
    return _json_summary(_get(f"{base_url}/ticks/{symbol}/latest"))


def get_symbols(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/symbols"))


def get_pipeline_stats(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/stats"))


def get_events(base_url: str, event_type: str | None = None, limit: int = 50) -> str:
    params: dict[str, Any] = {"limit": limit}
    if event_type:
        params["event_type"] = event_type
    return _json_summary(_get(f"{base_url}/events", params))


def pipeline_health(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/health"))


# ── Dashboard (port 5100) ────────────────────────────────────────

def get_trades(
    base_url: str,
    symbol: str | None = None,
    regime: str | None = None,
    limit: int = 50,
) -> str:
    """Query trade history with optional filters."""
    params: dict[str, Any] = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    if regime:
        params["regime"] = regime
    return _json_summary(_get(f"{base_url}/api/trades", params))


def get_analytics(base_url: str) -> str:
    """Retrieve computed metrics: Sharpe, PF, win rate, drawdown."""
    return _json_summary(_get(f"{base_url}/api/analytics"))


def get_filters(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/api/filters"))


def dashboard_health(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/health"))


# ── Continuous Learning System (port 5500) ─────────────────────

def cls_health(base_url: str) -> str:
    return _json_summary(_get(f"{base_url}/health"))


def cls_status(base_url: str) -> str:
    """Get full CLS status: registry, retrain, alerts, loop state."""
    return _json_summary(_get(f"{base_url}/status"))


def cls_list_models(base_url: str, symbol: str | None = None) -> str:
    """List registered models, optionally filtered by symbol."""
    params = {"symbol": symbol} if symbol else {}
    return _json_summary(_get(f"{base_url}/models", params))


def cls_get_champion(base_url: str, symbol: str) -> str:
    """Get the current champion model for a symbol."""
    return _json_summary(_get(f"{base_url}/models/{symbol}/champion"))


def cls_evaluate_performance(base_url: str, symbol: str) -> str:
    """Evaluate current model performance for a symbol."""
    return _json_summary(_get(f"{base_url}/performance/{symbol}"))


def cls_performance_trend(base_url: str, symbol: str) -> str:
    """Get performance trend analysis for a symbol."""
    return _json_summary(_get(f"{base_url}/performance/{symbol}/trend"))


def cls_check_drift(base_url: str, symbol: str) -> str:
    """Run feature and concept drift detection for a symbol."""
    return _json_summary(_get(f"{base_url}/drift/{symbol}"))


def cls_get_alerts(base_url: str) -> str:
    """Get active performance alerts."""
    return _json_summary(_get(f"{base_url}/performance/alerts"))


def cls_trigger_retrain(base_url: str, symbol: str, reason: str = "manual") -> str:
    """Trigger model retraining for a symbol."""
    return _json_summary(
        _post(f"{base_url}/retrain/{symbol}", {"reason": reason})
    )


def cls_retrain_status(base_url: str) -> str:
    """Get retrain orchestrator status and history."""
    return _json_summary(_get(f"{base_url}/retrain/status"))


def cls_loop_status(base_url: str) -> str:
    """Get continuous learning loop status."""
    return _json_summary(_get(f"{base_url}/loop/status"))


def cls_start_loop(base_url: str, interval_seconds: int = 300) -> str:
    """Start the continuous learning loop."""
    return _json_summary(
        _post(f"{base_url}/loop/start", {"interval_seconds": interval_seconds})
    )


def cls_stop_loop(base_url: str) -> str:
    """Stop the continuous learning loop."""
    return _json_summary(_post(f"{base_url}/loop/stop"))
