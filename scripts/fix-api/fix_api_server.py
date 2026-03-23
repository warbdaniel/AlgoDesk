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

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from fix_connector import FIXConnector
from fix_price_client import FIXPriceClient
from fix_trade_client import FIXTradeClient, RiskLimits

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
    trade_client = FIXTradeClient(trade_connector, risk_limits)

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


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "fix_api_server:app",
        host="0.0.0.0",
        port=5200,
        log_level="info",
    )
