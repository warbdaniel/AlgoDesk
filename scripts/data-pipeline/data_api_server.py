"""
FastAPI REST server for the data pipeline.

Exposes market data storage, feature computation, historical data
management, and the event bus over HTTP. Integrates with the FIX
price feed for live tick ingestion.

Runs on port 5300.
"""

import os
import sys
import time
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

from data_store import MarketDataStore, Tick, Candle, INTERVALS
from historical_data import HistoricalDataManager
from feature_engine import FeatureEngine
from data_bus import DataBus, EventType
from tick_poller import TickPoller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("data_api_server")

# ── Config ───────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


cfg = load_config()
db_cfg = cfg.get("database", {})
pipeline_cfg = cfg.get("pipeline", {})
api_cfg = cfg.get("api", {})

# ── Global services ──────────────────────────────────────────────

store: MarketDataStore | None = None
hist_manager: HistoricalDataManager | None = None
feature_engine: FeatureEngine | None = None
bus: DataBus | None = None
tick_poller: TickPoller | None = None
start_time: float = 0

# Tick buffer for batched writes
_tick_buffer: list[Tick] = []
_tick_lock = threading.Lock()
_TICK_BATCH_SIZE = 100
_TICK_FLUSH_INTERVAL = 5.0  # seconds


def _flush_ticks():
    """Periodically flush buffered ticks to the database."""
    global _tick_buffer
    while True:
        time.sleep(_TICK_FLUSH_INTERVAL)
        with _tick_lock:
            if not _tick_buffer:
                continue
            batch = _tick_buffer[:]
            _tick_buffer = []
        if store and batch:
            try:
                store.insert_ticks_batch(batch)
            except Exception:
                logger.exception("Failed to flush tick batch")


def on_tick_received(symbol: str, bid: float, ask: float,
                     bid_size: float = 0, ask_size: float = 0):
    """Callback from FIX price client → buffer tick and emit event."""
    tick = Tick(
        symbol=symbol, bid=bid, ask=ask,
        bid_size=bid_size, ask_size=ask_size, timestamp=time.time(),
    )
    batch = None
    with _tick_lock:
        _tick_buffer.append(tick)
        if len(_tick_buffer) >= _TICK_BATCH_SIZE:
            batch = _tick_buffer[:]
            _tick_buffer.clear()

    # Write outside the lock to avoid blocking tick ingestion
    if batch and store:
        try:
            store.insert_ticks_batch(batch)
        except Exception:
            logger.exception("Failed to flush tick batch")

    if bus:
        bus.emit_tick(symbol, bid, ask, bid_size, ask_size)


def start_services():
    global store, hist_manager, feature_engine, bus, tick_poller, start_time
    start_time = time.time()

    db_path = db_cfg.get("path", str(Path(__file__).parent / "market_data.db"))
    store = MarketDataStore(db_path)
    hist_manager = HistoricalDataManager(store)
    feature_engine = FeatureEngine()
    bus = DataBus(max_history=pipeline_cfg.get("event_history_size", 1000))

    # Register default symbols
    symbol_ids = []
    for sym_cfg in cfg.get("symbols", []):
        store.register_symbol(
            symbol=sym_cfg["id"],
            description=sym_cfg.get("description", ""),
            pip_size=sym_cfg.get("pip_size", 0.0001),
            digits=sym_cfg.get("digits", 5),
        )
        symbol_ids.append(sym_cfg["id"])

    # Start tick flush thread
    threading.Thread(target=_flush_ticks, daemon=True, name="tick-flusher").start()

    # Start tick purge scheduler (daily)
    retention_days = db_cfg.get("tick_retention_days", 7)
    def _purge_loop():
        while True:
            time.sleep(3600)  # check every hour
            cutoff = time.time() - (retention_days * 86400)
            if store:
                store.purge_ticks(cutoff)
    threading.Thread(target=_purge_loop, daemon=True, name="tick-purger").start()

    # Start tick poller if configured
    fix_cfg = cfg.get("fix_api", {})
    agg_cfg = cfg.get("auto_aggregate", {})
    poll_interval = fix_cfg.get("poll_interval", 0)
    agg_enabled = agg_cfg.get("enabled", False)
    candle_intervals = agg_cfg.get("intervals", []) if agg_enabled else []

    if poll_interval > 0:
        tick_poller = TickPoller(
            fix_api_url=fix_cfg.get("url", "http://localhost:5200"),
            poll_interval=poll_interval,
            tick_callback=on_tick_received,
            symbol_ids=symbol_ids,
            candle_intervals=candle_intervals,
            store=store,
        )
        tick_poller.start()

    logger.info("Data pipeline services started (db=%s)", db_path)


def stop_services():
    # Stop tick poller
    if tick_poller:
        tick_poller.stop()
    # Flush remaining ticks
    with _tick_lock:
        if store and _tick_buffer:
            store.insert_ticks_batch(_tick_buffer)
    if store:
        store.close()
    if bus:
        bus.shutdown()
    logger.info("Data pipeline services stopped")


# ── FastAPI app ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_services()
    yield
    stop_services()


app = FastAPI(
    title="AlgoDesk Data Pipeline",
    description="Market data storage, feature engineering, and event bus",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ────────────────────────────────────

class TickInput(BaseModel):
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0


class CandleInput(BaseModel):
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    open_time: float
    close_time: float


class BackfillRequest(BaseModel):
    symbol: str
    interval: str = "1m"
    days: int = Field(30, ge=1, le=365)


class AggregateRequest(BaseModel):
    symbol: str
    interval: str
    start_ts: float = 0
    end_ts: float = 0


class WebhookRequest(BaseModel):
    name: str
    url: str
    event_types: list[str]


class SymbolInput(BaseModel):
    symbol: str
    description: str = ""
    pip_size: float = 0.0001
    digits: int = 5


class CsvImportRequest(BaseModel):
    filepath: str
    symbol: str = ""
    interval: str = ""


# ── Health ───────────────────────────────────────────────────────

@app.get("/health")
def health():
    stats = store.get_stats() if store else {}
    bus_stats = bus.get_stats() if bus else {}
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else 0,
        "data_store": stats,
        "event_bus": bus_stats,
    }


# ── Symbols ──────────────────────────────────────────────────────

@app.get("/symbols")
def list_symbols():
    if not store:
        raise HTTPException(503, "Store not initialized")
    return {"symbols": store.get_symbols()}


@app.post("/symbols")
def register_symbol(req: SymbolInput):
    if not store:
        raise HTTPException(503, "Store not initialized")
    store.register_symbol(req.symbol, req.description, req.pip_size, req.digits)
    return {"status": "registered", "symbol": req.symbol}


# ── Ticks ────────────────────────────────────────────────────────

@app.post("/ticks")
def ingest_tick(req: TickInput):
    on_tick_received(req.symbol, req.bid, req.ask, req.bid_size, req.ask_size)
    return {"status": "buffered"}


@app.get("/ticks/{symbol}")
def get_ticks(symbol: str, start: float = 0, end: float = 0,
              limit: int = Query(1000, le=50000)):
    if not store:
        raise HTTPException(503, "Store not initialized")
    if end <= 0:
        end = time.time()
    if start <= 0:
        start = end - 3600  # last hour
    ticks = store.get_ticks(symbol, start, end, limit)
    return {"symbol": symbol, "count": len(ticks), "ticks": ticks}


@app.get("/ticks/{symbol}/latest")
def get_latest_tick(symbol: str):
    if not store:
        raise HTTPException(503, "Store not initialized")
    tick = store.get_latest_tick(symbol)

    # Also check the in-memory buffer for more recent ticks
    with _tick_lock:
        buffered = [t for t in _tick_buffer if t.symbol == symbol]
    if buffered:
        latest_buffered = buffered[-1]
        if not tick or latest_buffered.timestamp > tick.get("ts", 0):
            tick = {
                "symbol": latest_buffered.symbol,
                "bid": latest_buffered.bid,
                "ask": latest_buffered.ask,
                "bid_size": latest_buffered.bid_size,
                "ask_size": latest_buffered.ask_size,
                "ts": latest_buffered.timestamp,
            }

    if not tick:
        raise HTTPException(404, f"No ticks for {symbol}")
    return tick


# ── Candles ──────────────────────────────────────────────────────

@app.get("/candles/{symbol}/{interval}")
def get_candles(symbol: str, interval: str, start: float = 0,
                end: float = 0, limit: int = Query(500, le=10000)):
    if not store:
        raise HTTPException(503, "Store not initialized")
    if interval not in INTERVALS:
        raise HTTPException(400, f"Invalid interval. Valid: {list(INTERVALS.keys())}")
    candles = store.get_candles(symbol, interval, start, end, limit)
    return {"symbol": symbol, "interval": interval, "count": len(candles), "candles": candles}


@app.post("/candles")
def ingest_candle(req: CandleInput):
    if not store:
        raise HTTPException(503, "Store not initialized")
    if req.interval not in INTERVALS:
        raise HTTPException(400, f"Invalid interval. Valid: {list(INTERVALS.keys())}")
    store.upsert_candle(Candle(
        symbol=req.symbol, interval=req.interval,
        open=req.open, high=req.high, low=req.low, close=req.close,
        volume=req.volume, open_time=req.open_time, close_time=req.close_time,
    ))
    return {"status": "stored"}


@app.post("/candles/aggregate")
def aggregate_ticks(req: AggregateRequest):
    if not store:
        raise HTTPException(503, "Store not initialized")
    if req.interval not in INTERVALS:
        raise HTTPException(400, f"Invalid interval. Valid: {list(INTERVALS.keys())}")
    count = store.aggregate_ticks_to_candles(req.symbol, req.interval, req.start_ts, req.end_ts)
    return {"status": "aggregated", "candles_created": count}


# ── Features ─────────────────────────────────────────────────────

@app.get("/features/{symbol}/{interval}")
def get_features(symbol: str, interval: str, limit: int = Query(200, le=5000)):
    if not store or not feature_engine:
        raise HTTPException(503, "Services not initialized")
    if interval not in INTERVALS:
        raise HTTPException(400, f"Invalid interval. Valid: {list(INTERVALS.keys())}")

    candles = store.get_candles(symbol, interval, limit=limit + 50)
    if len(candles) < 50:
        raise HTTPException(404, f"Not enough candle data for {symbol}/{interval} (need 50+, have {len(candles)})")

    features = feature_engine.compute(candles, symbol)
    return {
        "symbol": symbol,
        "interval": interval,
        "count": len(features),
        "feature_names": features[0].feature_names() if features else [],
        "features": [f.to_dict() for f in features[-limit:]],
    }


@app.get("/features/{symbol}/{interval}/latest")
def get_latest_features(symbol: str, interval: str):
    if not store or not feature_engine:
        raise HTTPException(503, "Services not initialized")
    if interval not in INTERVALS:
        raise HTTPException(400, f"Invalid interval. Valid: {list(INTERVALS.keys())}")

    candles = store.get_candles(symbol, interval, limit=200)
    if len(candles) < 50:
        raise HTTPException(404, f"Not enough candle data for {symbol}/{interval}")

    fv = feature_engine.compute_latest(candles, symbol)
    if not fv:
        raise HTTPException(404, "Could not compute features")
    return fv.to_dict()


@app.get("/features/names")
def get_feature_names():
    from feature_engine import FeatureVector
    return {"feature_names": FeatureVector.feature_names()}


# ── Historical data ──────────────────────────────────────────────

@app.post("/historical/backfill")
def start_backfill(req: BackfillRequest):
    if not hist_manager:
        raise HTTPException(503, "Historical data manager not initialized")
    try:
        job_id = hist_manager.backfill(req.symbol, req.interval, req.days)
        return {"status": "started", "job_id": job_id}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/historical/jobs")
def list_backfill_jobs():
    if not hist_manager:
        raise HTTPException(503, "Historical data manager not initialized")
    return {"jobs": hist_manager.list_jobs()}


@app.get("/historical/jobs/{job_id}")
def get_backfill_job(job_id: str):
    if not hist_manager:
        raise HTTPException(503, "Historical data manager not initialized")
    status = hist_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(404, f"Job {job_id} not found")
    return status


@app.get("/historical/gaps/{symbol}/{interval}")
def find_data_gaps(symbol: str, interval: str, start: float = 0, end: float = 0):
    if not hist_manager:
        raise HTTPException(503, "Historical data manager not initialized")
    gaps = hist_manager.find_gaps(symbol, interval, start, end)
    return {"symbol": symbol, "interval": interval, "gaps": gaps, "gap_count": len(gaps)}


@app.get("/historical/coverage/{symbol}")
def get_data_coverage(symbol: str):
    if not hist_manager:
        raise HTTPException(503, "Historical data manager not initialized")
    return hist_manager.get_coverage(symbol)


@app.post("/historical/export")
def export_csv(symbol: str, interval: str, start: float = 0, end: float = 0):
    if not hist_manager:
        raise HTTPException(503, "Historical data manager not initialized")
    filepath = hist_manager.export_csv(symbol, interval, start, end)
    return {"status": "exported", "filepath": str(filepath)}


@app.post("/historical/import")
def import_csv(req: CsvImportRequest):
    if not hist_manager:
        raise HTTPException(503, "Historical data manager not initialized")
    try:
        count = hist_manager.import_csv(req.filepath, req.symbol, req.interval)
        return {"status": "imported", "candles_imported": count}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


# ── Event bus ────────────────────────────────────────────────────

@app.get("/events")
def get_event_history(event_type: str = "", limit: int = Query(50, le=500)):
    if not bus:
        raise HTTPException(503, "Event bus not initialized")
    return {"events": bus.get_history(event_type, limit)}


@app.get("/events/stats")
def get_event_stats():
    if not bus:
        raise HTTPException(503, "Event bus not initialized")
    return bus.get_stats()


@app.post("/webhooks")
def register_webhook(req: WebhookRequest):
    if not bus:
        raise HTTPException(503, "Event bus not initialized")
    valid_types = [e.value for e in EventType]
    for et in req.event_types:
        if et not in valid_types:
            raise HTTPException(400, f"Invalid event type '{et}'. Valid: {valid_types}")
    name = bus.add_webhook(req.name, req.url, req.event_types)
    return {"status": "registered", "name": name}


@app.delete("/webhooks/{name}")
def remove_webhook(name: str):
    if not bus:
        raise HTTPException(503, "Event bus not initialized")
    bus.remove_webhook(name)
    return {"status": "removed", "name": name}


@app.get("/webhooks")
def list_webhooks():
    if not bus:
        raise HTTPException(503, "Event bus not initialized")
    return {"webhooks": bus.list_webhooks()}


# ── Data stats ───────────────────────────────────────────────────

@app.get("/stats")
def get_data_stats():
    if not store:
        raise HTTPException(503, "Store not initialized")
    return store.get_stats()


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    api_host = api_cfg.get("host", "0.0.0.0")
    api_port = api_cfg.get("port", 5300)
    uvicorn.run(
        "data_api_server:app",
        host=api_host,
        port=api_port,
        log_level="info",
    )
