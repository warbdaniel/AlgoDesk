"""
FastAPI REST server wrapping the FIX price and trade connections.
Runs on port 5200.
"""

import os
import sys
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('/opt/trading-desk/.env')

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from fix_connector import FIXConnector
from fix_price_client import FIXPriceClient
from fix_trade_client import FIXTradeClient, RiskLimits
from risk_manager import RiskManager, RiskConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("fix_api_server")

# ── Config ───────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "fix_config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


cfg = load_config()
price_cfg = cfg.get("price_connection", {})
trade_cfg = cfg.get("trade_connection", {})

PASSWORD = os.environ.get("FIX_PASSWORD", "")
HOST = price_cfg.get("host", "live-uk-eqx-01.p.c-trader.com")
SENDER_COMP_ID = price_cfg.get("sender_comp_id", "live4.icmarkets.6003600")
TARGET_COMP_ID = price_cfg.get("target_comp_id", "cServer")
PRICE_PORT = price_cfg.get("port", 5211)
TRADE_PORT = trade_cfg.get("port", 5212)
DEFAULT_SYMBOLS = cfg.get("default_symbols", ["1", "2"])  # cTrader symbol IDs

risk_cfg = cfg.get("risk_limits", {})
risk_limits = RiskLimits(
    max_order_size=risk_cfg.get("max_order_size", 10.0),
    max_position_size=risk_cfg.get("max_position_size", 50.0),
    max_open_orders=risk_cfg.get("max_open_orders", 50),
    max_daily_orders=risk_cfg.get("max_daily_orders", 500),
)

# Advanced risk configuration
risk_config = RiskConfig(
    max_order_size=risk_cfg.get("max_order_size", 10.0),
    min_order_size=risk_cfg.get("min_order_size", 0.01),
    max_position_size=risk_cfg.get("max_position_size", 50.0),
    max_open_orders_per_symbol=risk_cfg.get("max_open_orders_per_symbol", 10),
    max_open_orders=risk_cfg.get("max_open_orders", 50),
    max_daily_orders=risk_cfg.get("max_daily_orders", 500),
    max_total_exposure=risk_cfg.get("max_total_exposure", 100.0),
    max_daily_loss=risk_cfg.get("max_daily_loss", 5000.0),
    max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 5.0),
    max_orders_per_minute=risk_cfg.get("max_orders_per_minute", 30),
    max_orders_per_second=risk_cfg.get("max_orders_per_second", 5),
    duplicate_window_sec=risk_cfg.get("duplicate_window_sec", 2.0),
    max_spread_pips=risk_cfg.get("max_spread_pips", 50.0),
    stale_price_sec=risk_cfg.get("stale_price_sec", 30.0),
)
risk_manager = RiskManager(risk_config)

# ── Global clients ───────────────────────────────────────────────

price_connector: FIXConnector | None = None
trade_connector: FIXConnector | None = None
price_client: FIXPriceClient | None = None
trade_client: FIXTradeClient | None = None
start_time: float = 0


def start_connections():
    global price_connector, trade_connector, price_client, trade_client, start_time

    if not PASSWORD:
        logger.error("FIX_PASSWORD environment variable not set!")
        return

    start_time = time.time()

    price_connector = FIXConnector(
        host=HOST,
        port=PRICE_PORT,
        sender_comp_id=SENDER_COMP_ID,
        target_comp_id=TARGET_COMP_ID,
        sender_sub_id="QUOTE",
        password=PASSWORD,
    )
    price_client = FIXPriceClient(price_connector)

    trade_connector = FIXConnector(
        host=HOST,
        port=TRADE_PORT,
        sender_comp_id=SENDER_COMP_ID,
        target_comp_id=TARGET_COMP_ID,
        sender_sub_id="TRADE",
        password=PASSWORD,
    )
    trade_client = FIXTradeClient(trade_connector, risk_limits, risk_manager)
    trade_client.set_price_feed(price_client)

    price_connector.connect()
    trade_connector.connect()

    # Wait for logon then subscribe to default symbols
    def _subscribe_defaults():
        for _ in range(30):
            if price_client.connector.is_logged_in:
                break
            time.sleep(1)
        for sym in DEFAULT_SYMBOLS:
            price_client.subscribe(sym)

    threading.Thread(target=_subscribe_defaults, daemon=True).start()


def stop_connections():
    if price_connector:
        price_connector.disconnect()
    if trade_connector:
        trade_connector.disconnect()


# ── FastAPI app ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_connections()
    yield
    stop_connections()


app = FastAPI(
    title="AlgoDesk FIX API",
    description="REST wrapper for cTrader FIX 4.4 price and trade connections",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ────────────────────────────────────

class NewOrderRequest(BaseModel):
    symbol: str
    side: str = Field(..., description="BUY or SELL")
    type: str = Field("MARKET", description="MARKET, LIMIT, or STOP")
    quantity: float = Field(..., gt=0)
    price: float = Field(0.0, ge=0)
    stop_price: float = Field(0.0, ge=0)
    time_in_force: str = Field("GTC", description="GTC, IOC, or FOK")


class SubscribeRequest(BaseModel):
    symbol: str


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "price_connected": price_connector.is_connected if price_connector else False,
        "price_logged_in": price_connector.is_logged_in if price_connector else False,
        "trade_connected": trade_connector.is_connected if trade_connector else False,
        "trade_logged_in": trade_connector.is_logged_in if trade_connector else False,
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else 0,
        "kill_switch_active": risk_manager.is_killed,
        "price_connection_health": price_connector.health.to_dict() if price_connector else None,
        "trade_connection_health": trade_connector.health.to_dict() if trade_connector else None,
    }


@app.get("/prices/{symbol}")
def get_price(symbol: str):
    if not price_client:
        raise HTTPException(503, "Price client not initialized")
    data = price_client.get_price(symbol)
    if not data:
        raise HTTPException(404, f"No price data for {symbol}")
    return data


@app.get("/prices")
def get_all_prices():
    if not price_client:
        raise HTTPException(503, "Price client not initialized")
    return price_client.get_all_prices()


@app.post("/subscribe")
def subscribe(req: SubscribeRequest):
    if not price_client:
        raise HTTPException(503, "Price client not initialized")
    price_client.subscribe(req.symbol)
    return {"status": "subscribed", "symbol": req.symbol}


@app.delete("/subscribe/{symbol}")
def unsubscribe(symbol: str):
    if not price_client:
        raise HTTPException(503, "Price client not initialized")
    price_client.unsubscribe(symbol)
    return {"status": "unsubscribed", "symbol": symbol}


@app.get("/candles/{symbol}")
def get_candles(symbol: str, count: int = 100):
    if not price_client:
        raise HTTPException(503, "Price client not initialized")
    candles = price_client.get_candles(symbol, count)
    return {"symbol": symbol, "count": len(candles), "candles": candles}


@app.post("/orders")
def create_order(req: NewOrderRequest):
    if not trade_client:
        raise HTTPException(503, "Trade client not initialized")

    side_map = {"BUY": "1", "SELL": "2"}
    type_map = {"MARKET": "1", "LIMIT": "2", "STOP": "3"}
    tif_map = {"GTC": "1", "IOC": "3", "FOK": "4"}

    side = side_map.get(req.side.upper())
    if not side:
        raise HTTPException(400, f"Invalid side: {req.side}")
    ord_type = type_map.get(req.type.upper())
    if not ord_type:
        raise HTTPException(400, f"Invalid order type: {req.type}")
    tif = tif_map.get(req.time_in_force.upper(), "1")

    result = trade_client.new_order(
        symbol=req.symbol,
        side=side,
        order_type=ord_type,
        quantity=req.quantity,
        price=req.price,
        stop_price=req.stop_price,
        time_in_force=tif,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.delete("/orders/{cl_ord_id}")
def cancel_order(cl_ord_id: str):
    if not trade_client:
        raise HTTPException(503, "Trade client not initialized")
    result = trade_client.cancel_order(cl_ord_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/orders")
def get_orders(active_only: bool = False):
    if not trade_client:
        raise HTTPException(503, "Trade client not initialized")
    return {"orders": trade_client.get_orders(active_only)}


@app.get("/positions")
def get_positions():
    if not trade_client:
        raise HTTPException(503, "Trade client not initialized")
    return {"positions": trade_client.get_positions()}


@app.get("/account")
def get_account():
    if not trade_client:
        raise HTTPException(503, "Trade client not initialized")
    return trade_client.get_account_summary()


# ── Risk management endpoints ────────────────────────────────────

@app.get("/risk/status")
def get_risk_status():
    """Current risk manager state: limits, counters, positions, kill switch."""
    return risk_manager.get_status()


@app.get("/risk/violations")
def get_risk_violations(count: int = 50):
    """Recent risk check violations (audit trail)."""
    return {"violations": risk_manager.get_violations(count)}


class KillSwitchRequest(BaseModel):
    reason: str = Field("Manual kill switch", description="Reason for activating kill switch")


@app.post("/risk/kill")
def activate_kill_switch(req: KillSwitchRequest):
    """Emergency: halt all new order flow immediately."""
    risk_manager.activate_kill_switch(req.reason)
    return {"status": "killed", "reason": req.reason}


@app.delete("/risk/kill")
def deactivate_kill_switch():
    """Re-enable order flow after manual review."""
    risk_manager.deactivate_kill_switch()
    return {"status": "active"}


class EquityRequest(BaseModel):
    equity: float = Field(..., gt=0, description="Starting daily equity for drawdown calculation")


@app.post("/risk/equity")
def set_starting_equity(req: EquityRequest):
    """Set beginning-of-day equity for drawdown % calculations."""
    risk_manager.set_starting_equity(req.equity)
    return {"status": "ok", "starting_equity": req.equity}


@app.get("/risk/connection-health")
def get_connection_health():
    """Detailed connection health metrics for both FIX sessions."""
    return {
        "price": price_connector.health.to_dict() if price_connector else None,
        "trade": trade_connector.health.to_dict() if trade_connector else None,
    }


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "fix_api_server:app",
        host="0.0.0.0",
        port=5200,
        log_level="info",
    )
