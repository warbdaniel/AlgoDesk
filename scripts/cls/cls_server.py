#!/usr/bin/env python3
"""
Continuous Learning System - FastAPI Server
============================================

REST API for the CLS microservice (port 5500).

Endpoints:
  Health & Status:
    GET  /health                          - Service health check
    GET  /status                          - Full system status

  Model Registry:
    GET  /models                          - List all registered models
    GET  /models/{symbol}                 - Models for a specific symbol
    GET  /models/{symbol}/champion        - Current champion model
    POST /models/discover                 - Scan & register new model artifacts
    POST /models/{model_id}/promote       - Promote challenger to champion
    POST /models/{model_id}/retire        - Retire a model

  Predictions:
    POST /predictions                     - Log a model prediction
    POST /predictions/{id}/resolve        - Record actual outcome
    GET  /predictions/{symbol}            - Recent predictions for symbol

  Performance:
    GET  /performance/{symbol}            - Evaluate current performance
    GET  /performance/{symbol}/trend      - Performance trend analysis
    GET  /performance/{symbol}/history    - Historical performance snapshots
    GET  /performance/alerts              - Active performance alerts

  Drift Detection:
    GET  /drift/{symbol}                  - Run drift detection
    GET  /drift/{symbol}/feature          - Feature drift only
    GET  /drift/{symbol}/concept          - Concept drift only
    GET  /drift/{symbol}/history          - Drift detection history

  Retraining:
    POST /retrain/{symbol}               - Trigger a retrain
    GET  /retrain/status                  - Retrain orchestrator status
    GET  /retrain/history                 - Retrain history
    POST /retrain/{symbol}/cancel        - Cancel active retrain

  Feedback:
    POST /feedback/poll                   - Poll dashboard for trade outcomes
    POST /feedback/record                 - Manually record an outcome
    GET  /feedback/summary                - Feedback data summary
    POST /feedback/expire                 - Expire old unresolved predictions

  Learning Loop:
    POST /loop/tick                       - Run one full learning loop cycle
    POST /loop/start                      - Start the continuous learning loop
    POST /loop/stop                       - Stop the learning loop
    GET  /loop/status                     - Learning loop status
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLSConfig, CLS_DATA_DIR
from cls_store import CLSStore
from model_registry import ModelRegistry
from performance_monitor import PerformanceMonitor
from drift_detector import DriftDetector
from feedback_loop import FeedbackLoop
from retrain_orchestrator import RetrainOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cls.server")

# =====================================================================
#  Global state
# =====================================================================
cfg = CLSConfig()
store: CLSStore | None = None
registry: ModelRegistry | None = None
perf_monitor: PerformanceMonitor | None = None
drift_detector: DriftDetector | None = None
feedback_loop: FeedbackLoop | None = None
retrain_orch: RetrainOrchestrator | None = None

# Learning loop state
_loop_running = False
_loop_thread: threading.Thread | None = None
_loop_interval = 300  # 5 minutes default
_loop_stats = {"cycles": 0, "last_cycle_ts": 0, "started_at": 0}


# =====================================================================
#  Lifespan
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, registry, perf_monitor, drift_detector, feedback_loop, retrain_orch

    CLS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    store = CLSStore(cfg.db_path)
    registry = ModelRegistry(store, cfg.registry)
    perf_monitor = PerformanceMonitor(store, cfg.performance)
    drift_detector = DriftDetector(store, cfg.drift)
    feedback_loop = FeedbackLoop(store, cfg.feedback)
    retrain_orch = RetrainOrchestrator(store, registry, cfg.retrain)

    # Discover existing models on startup
    discovered = registry.discover_models()
    if discovered:
        logger.info("Discovered %d model artifacts on startup", len(discovered))

    # Auto-elect champions for symbols without one
    all_models = store.list_models()
    symbols = set(m["symbol"] for m in all_models)
    for sym in symbols:
        registry.get_or_elect_champion(sym)

    logger.info("CLS server started (port %d, db=%s)", cfg.port, cfg.db_path)

    yield

    # Shutdown
    global _loop_running
    _loop_running = False
    store.close()
    logger.info("CLS server stopped")


app = FastAPI(
    title="AlgoDesk Continuous Learning System",
    version="1.0.0",
    description="Model performance monitoring, drift detection, and automated retraining.",
    lifespan=lifespan,
)


# =====================================================================
#  Pydantic models
# =====================================================================
class PredictionRequest(BaseModel):
    symbol: str
    model_id: int
    prediction: float
    confidence: float = 0.0
    features: Optional[dict] = None
    regime: str = ""


class ResolveRequest(BaseModel):
    actual_outcome: float
    pnl_pips: Optional[float] = None


class OutcomeRequest(BaseModel):
    prediction_id: int
    actual_outcome: float
    pnl_pips: Optional[float] = None
    trade_id: str = ""


class RetrainRequest(BaseModel):
    reason: str = "manual"
    details: Optional[dict] = None
    async_mode: bool = True


# =====================================================================
#  Health & Status
# =====================================================================
@app.get("/health")
async def health():
    return {
        "service": "cls",
        "status": "ok",
        "version": "1.0.0",
        "port": cfg.port,
        "ts": time.time(),
    }


@app.get("/status")
async def status():
    return {
        "service": "cls",
        "status": "ok",
        "store": store.get_stats(),
        "registry": registry.get_status(),
        "retrain": retrain_orch.get_status(),
        "learning_loop": {
            "running": _loop_running,
            "interval_seconds": _loop_interval,
            **_loop_stats,
        },
        "alerts": perf_monitor.get_alerts(),
    }


# =====================================================================
#  Model Registry
# =====================================================================
@app.get("/models")
async def list_models(symbol: str = "", status_filter: str = ""):
    models = store.list_models(symbol=symbol, status=status_filter)
    return {"models": models, "total": len(models)}


@app.get("/models/{symbol}")
async def get_symbol_models(symbol: str):
    models = store.list_models(symbol=symbol)
    champion = store.get_champion(symbol)
    challengers = store.get_challengers(symbol)
    return {
        "symbol": symbol,
        "champion": champion,
        "challengers": challengers,
        "total": len(models),
    }


@app.get("/models/{symbol}/champion")
async def get_champion(symbol: str, model_type: str = "lgbm"):
    champion = registry.get_or_elect_champion(symbol, model_type)
    if not champion:
        raise HTTPException(404, f"No champion model for {symbol}/{model_type}")
    return champion


@app.post("/models/discover")
async def discover_models():
    discovered = registry.discover_models()
    return {"discovered": discovered, "total": len(discovered)}


@app.post("/models/{model_id}/promote")
async def promote_model(model_id: int):
    success = registry.promote_challenger(model_id)
    if not success:
        raise HTTPException(404, f"Model {model_id} not found or already champion")
    return {"status": "promoted", "model_id": model_id}


@app.post("/models/{model_id}/retire")
async def retire_model(model_id: int):
    store.retire_model(model_id)
    return {"status": "retired", "model_id": model_id}


# =====================================================================
#  Predictions
# =====================================================================
@app.post("/predictions")
async def log_prediction(req: PredictionRequest):
    pred_id = store.log_prediction(
        symbol=req.symbol,
        model_id=req.model_id,
        prediction=req.prediction,
        confidence=req.confidence,
        features=req.features,
        regime=req.regime,
    )
    return {"prediction_id": pred_id, "symbol": req.symbol}


@app.post("/predictions/{prediction_id}/resolve")
async def resolve_prediction(prediction_id: int, req: ResolveRequest):
    store.resolve_prediction(
        prediction_id=prediction_id,
        actual_outcome=req.actual_outcome,
        pnl_pips=req.pnl_pips,
    )
    return {"status": "resolved", "prediction_id": prediction_id}


@app.get("/predictions/{symbol}")
async def get_predictions(
    symbol: str,
    model_id: Optional[int] = None,
    limit: int = 100,
    resolved_only: bool = False,
):
    predictions = store.get_recent_predictions(
        symbol=symbol,
        model_id=model_id,
        limit=limit,
        resolved_only=resolved_only,
    )
    return {"predictions": predictions, "total": len(predictions)}


# =====================================================================
#  Performance
# =====================================================================
@app.get("/performance/{symbol}")
async def evaluate_performance(symbol: str, model_type: str = "lgbm"):
    champion = registry.get_champion(symbol, model_type)
    if not champion:
        raise HTTPException(404, f"No champion model for {symbol}")
    metrics = perf_monitor.evaluate(symbol, champion["id"])
    return metrics


@app.get("/performance/{symbol}/trend")
async def performance_trend(symbol: str, lookback: int = 20):
    champion = registry.get_champion(symbol)
    model_id = champion["id"] if champion else None
    return perf_monitor.get_performance_trend(symbol, model_id, lookback)


@app.get("/performance/{symbol}/history")
async def performance_history(symbol: str, limit: int = 50):
    champion = registry.get_champion(symbol)
    model_id = champion["id"] if champion else None
    snapshots = store.get_performance_history(symbol, model_id=model_id, limit=limit)
    return {"symbol": symbol, "snapshots": snapshots, "total": len(snapshots)}


@app.get("/performance/alerts")
async def get_alerts(symbol: str = ""):
    return {"alerts": perf_monitor.get_alerts(symbol)}


@app.post("/performance/alerts/clear")
async def clear_alerts(symbol: str = ""):
    perf_monitor.clear_alerts(symbol)
    return {"status": "cleared"}


# =====================================================================
#  Drift Detection
# =====================================================================
@app.get("/drift/{symbol}")
async def check_drift(symbol: str, model_type: str = "lgbm"):
    champion = registry.get_champion(symbol, model_type)
    if not champion:
        raise HTTPException(404, f"No champion model for {symbol}")
    return drift_detector.check_all_drift(symbol, champion["id"])


@app.get("/drift/{symbol}/feature")
async def check_feature_drift(symbol: str, model_type: str = "lgbm"):
    champion = registry.get_champion(symbol, model_type)
    if not champion:
        raise HTTPException(404, f"No champion model for {symbol}")
    result = drift_detector.detect_feature_drift(symbol, champion["id"])
    return result.to_dict()


@app.get("/drift/{symbol}/concept")
async def check_concept_drift(symbol: str, model_type: str = "lgbm"):
    champion = registry.get_champion(symbol, model_type)
    if not champion:
        raise HTTPException(404, f"No champion model for {symbol}")
    result = drift_detector.detect_concept_drift(symbol, champion["id"])
    return result.to_dict()


@app.get("/drift/{symbol}/history")
async def drift_history(symbol: str, drift_type: str = "", limit: int = 50):
    history = store.get_drift_history(symbol, drift_type=drift_type, limit=limit)
    return {"symbol": symbol, "history": history, "total": len(history)}


# =====================================================================
#  Retraining
# =====================================================================
@app.post("/retrain/{symbol}")
async def trigger_retrain(symbol: str, req: RetrainRequest):
    result = retrain_orch.trigger_retrain(
        symbol=symbol,
        reason=req.reason,
        trigger_details=req.details,
        async_mode=req.async_mode,
    )
    return result


@app.get("/retrain/status")
async def retrain_status():
    return retrain_orch.get_status()


@app.get("/retrain/history")
async def retrain_history(symbol: str = "", limit: int = 20):
    history = store.get_retrain_history(symbol=symbol, limit=limit)
    return {"history": history, "total": len(history)}


@app.post("/retrain/{symbol}/cancel")
async def cancel_retrain(symbol: str):
    return retrain_orch.cancel_retrain(symbol)


# =====================================================================
#  Feedback Loop
# =====================================================================
@app.post("/feedback/poll")
async def poll_feedback():
    return feedback_loop.poll_trade_outcomes()


@app.post("/feedback/record")
async def record_outcome(req: OutcomeRequest):
    return feedback_loop.record_outcome(
        prediction_id=req.prediction_id,
        actual_outcome=req.actual_outcome,
        pnl_pips=req.pnl_pips,
        trade_id=req.trade_id,
    )


@app.get("/feedback/summary")
async def feedback_summary(symbol: str = ""):
    return feedback_loop.get_feedback_summary(symbol)


@app.post("/feedback/expire")
async def expire_predictions(max_age_seconds: float = 7200):
    return feedback_loop.resolve_expired_predictions(max_age_seconds)


# =====================================================================
#  Learning Loop (continuous monitoring cycle)
# =====================================================================
def _learning_loop():
    """Background thread that runs the continuous learning cycle."""
    global _loop_running, _loop_stats

    logger.info("Learning loop started (interval=%ds)", _loop_interval)
    _loop_stats["started_at"] = time.time()

    while _loop_running:
        try:
            _run_one_cycle()
            _loop_stats["cycles"] += 1
            _loop_stats["last_cycle_ts"] = time.time()
        except Exception as e:
            logger.error("Learning loop cycle error: %s", e, exc_info=True)

        # Sleep in small increments so we can stop quickly
        for _ in range(int(_loop_interval)):
            if not _loop_running:
                break
            time.sleep(1)

    logger.info("Learning loop stopped after %d cycles", _loop_stats["cycles"])


def _run_one_cycle():
    """Execute one full learning loop cycle."""
    # 1. Poll for trade outcomes (feedback)
    if feedback_loop.should_poll():
        feedback_loop.poll_trade_outcomes()

    # 2. Expire old unresolved predictions
    feedback_loop.resolve_expired_predictions()

    # 3. For each monitored symbol: evaluate performance + drift
    all_models = store.list_models(status="champion")
    for model in all_models:
        symbol = model["symbol"]
        model_id = model["id"]

        # Performance evaluation
        perf_metrics = None
        if perf_monitor.should_evaluate(symbol):
            perf_metrics = perf_monitor.evaluate(symbol, model_id)

        # Drift detection
        drift_result = None
        if drift_detector.should_check(symbol):
            drift_result = drift_detector.check_all_drift(symbol, model_id)

        # Retrain decision
        should, reason = retrain_orch.should_retrain(
            symbol, perf_metrics, drift_result,
        )
        if should:
            retrain_orch.trigger_retrain(
                symbol=symbol,
                reason=reason,
                trigger_details={
                    "performance": perf_metrics,
                    "drift": drift_result,
                },
            )


@app.post("/loop/tick")
async def loop_tick():
    """Run one learning loop cycle manually."""
    try:
        _run_one_cycle()
        return {"status": "ok", "message": "One cycle completed"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/loop/start")
async def loop_start(interval_seconds: int = 300):
    global _loop_running, _loop_thread, _loop_interval

    if _loop_running:
        return {"status": "already_running"}

    _loop_interval = max(interval_seconds, 60)
    _loop_running = True
    _loop_thread = threading.Thread(
        target=_learning_loop, daemon=True, name="learning-loop",
    )
    _loop_thread.start()
    return {"status": "started", "interval_seconds": _loop_interval}


@app.post("/loop/stop")
async def loop_stop():
    global _loop_running

    if not _loop_running:
        return {"status": "not_running"}

    _loop_running = False
    return {"status": "stopping"}


@app.get("/loop/status")
async def loop_status():
    return {
        "running": _loop_running,
        "interval_seconds": _loop_interval,
        **_loop_stats,
    }


# =====================================================================
#  CLI entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AlgoDesk Continuous Learning System Server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5500, help="Bind port")
    parser.add_argument("--db", type=str, default="", help="CLS database path")
    parser.add_argument(
        "--auto-loop", action="store_true",
        help="Auto-start the learning loop on boot",
    )
    parser.add_argument(
        "--loop-interval", type=int, default=300,
        help="Learning loop interval in seconds (default: 300)",
    )
    args = parser.parse_args()

    global cfg, _loop_interval
    cfg.host = args.host
    cfg.port = args.port
    if args.db:
        cfg.db_path = args.db
    _loop_interval = args.loop_interval

    import uvicorn

    if args.auto_loop:
        # Start loop after uvicorn starts
        import asyncio

        @app.on_event("startup")
        async def _auto_start_loop():
            global _loop_running, _loop_thread
            _loop_running = True
            _loop_thread = threading.Thread(
                target=_learning_loop, daemon=True, name="learning-loop",
            )
            _loop_thread.start()

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
