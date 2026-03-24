"""
FastAPI REST server for the Claude API autonomous trading agent.
Runs on port 5400.

Provides endpoints to send natural-language queries to an autonomous
Claude-powered agent that can inspect market data, detect regimes,
manage orders, and analyse performance across the full AlgoDesk stack.
"""

import os
import sys
import json
import time
import hmac
import hashlib
import secrets
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from claude_agent import TradingAgent, AgentConfig, AgentResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("claude_api_server")

# ── Config ───────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


cfg = load_config()
server_cfg = cfg.get("server", {})
claude_cfg = cfg.get("claude", {})
services_cfg = cfg.get("services", {})
agent_cfg = cfg.get("agent", {})

SERVICE_URLS = {
    "regime_detector": services_cfg.get("regime_detector", "http://localhost:5000"),
    "dashboard": services_cfg.get("dashboard", "http://localhost:5100"),
    "fix_api": services_cfg.get("fix_api", "http://localhost:5200"),
    "data_pipeline": services_cfg.get("data_pipeline", "http://localhost:5300"),
    "cls": services_cfg.get("cls", "http://localhost:5500"),
}

# ── Auth ─────────────────────────────────────────────────────────

UI_PASSWORD = os.environ.get("CLAUDE_UI_PASSWORD", "")
SESSION_SECRET = secrets.token_hex(32)  # random per server start
SESSION_TTL = 24 * 60 * 60  # 24 hours
SESSION_COOKIE = "algodesk_session"

# Rate limiting: {ip: [timestamps]}
_login_attempts: dict[str, list[float]] = defaultdict(list)
LOGIN_RATE_LIMIT = 5       # max attempts
LOGIN_RATE_WINDOW = 60     # per 60 seconds


def _sign_session(issued_at: int) -> str:
    """Create an HMAC-signed session token: timestamp.signature"""
    msg = str(issued_at).encode()
    sig = hmac.new(SESSION_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    return f"{issued_at}.{sig}"


def _verify_session(token: str) -> bool:
    """Verify session token signature and TTL."""
    if not token or "." not in token:
        return False
    try:
        issued_str, sig = token.split(".", 1)
        issued_at = int(issued_str)
    except (ValueError, TypeError):
        return False
    expected = hmac.new(
        SESSION_SECRET.encode(), issued_str.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    return (time.time() - issued_at) < SESSION_TTL


def _check_rate_limit(ip: str) -> bool:
    """Return True if the IP is within rate limits."""
    now = time.time()
    attempts = _login_attempts[ip]
    # Prune old entries
    _login_attempts[ip] = [t for t in attempts if now - t < LOGIN_RATE_WINDOW]
    return len(_login_attempts[ip]) < LOGIN_RATE_LIMIT


def _is_auth_enabled() -> bool:
    return bool(UI_PASSWORD)


# ── Auth middleware ──────────────────────────────────────────────

PUBLIC_PATHS = {"/", "/health", "/services/health", "/auth/login", "/auth/check"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _is_auth_enabled():
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        token = request.cookies.get(SESSION_COOKIE, "")
        if not _verify_session(token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        return await call_next(request)


# ── Global agent ─────────────────────────────────────────────────

agent: TradingAgent | None = None
start_time: float = 0
query_count: int = 0


def _load_system_prompt() -> str:
    """Load custom system prompt from file if configured."""
    prompt_file = agent_cfg.get("system_prompt_file")
    if prompt_file:
        p = Path(prompt_file)
        if p.exists():
            return p.read_text()
        logger.warning("system_prompt_file %s not found, using default", prompt_file)
    from claude_agent import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def start_agent():
    global agent, start_time

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set!")
        logger.error("The agent will not be able to make API calls.")

    config = AgentConfig(
        model=claude_cfg.get("model", "claude-opus-4-6"),
        max_tokens=claude_cfg.get("max_tokens", 16000),
        max_tool_rounds=claude_cfg.get("max_tool_rounds", 20),
        read_only=agent_cfg.get("read_only", False),
        system_prompt=_load_system_prompt(),
    )
    agent = TradingAgent(config, SERVICE_URLS)
    start_time = time.time()
    logger.info(
        "Claude trading agent started (model=%s, read_only=%s, tools=%d)",
        config.model,
        config.read_only,
        len(agent._specs_by_name),
    )


def stop_agent():
    logger.info("Claude trading agent stopped (queries=%d)", query_count)


# ── FastAPI app ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_agent()
    yield
    stop_agent()


app = FastAPI(
    title="AlgoDesk Claude Agent",
    description=(
        "Autonomous trading agent powered by the Claude API. "
        "Send natural-language queries and the agent will interact with "
        "regime-detector, FIX API, data pipeline, and dashboard services."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── UI template ──────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).parent / "templates"


# ── Request / Response models ────────────────────────────────────

class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Natural-language query for the agent")
    conversation: list[dict] | None = Field(
        None,
        description="Optional prior conversation messages for multi-turn context",
    )


class QueryResponse(BaseModel):
    response: str
    tool_calls: list[dict]
    rounds: int
    input_tokens: int
    output_tokens: int
    elapsed_sec: float


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    model: str
    read_only: bool
    tools_available: int
    total_queries: int
    services: dict[str, str]


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the trading agent web UI."""
    html_path = TEMPLATE_DIR / "agent.html"
    return HTMLResponse(html_path.read_text())


class LoginRequest(BaseModel):
    password: str


@app.post("/auth/login")
async def auth_login(req: LoginRequest, request: Request, response: Response):
    """Authenticate with password and receive a session cookie."""
    if not _is_auth_enabled():
        return {"status": "ok", "message": "Auth disabled"}

    client_ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again later.",
        )

    _login_attempts[client_ip].append(time.time())

    if not hmac.compare_digest(req.password, UI_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = _sign_session(int(time.time()))
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="strict",
        max_age=SESSION_TTL,
    )
    return {"status": "ok"}


@app.get("/auth/check")
async def auth_check(request: Request):
    """Check if the current session is valid."""
    if not _is_auth_enabled():
        return {"authenticated": True, "auth_required": False}

    token = request.cookies.get(SESSION_COOKIE, "")
    if _verify_session(token):
        return {"authenticated": True, "auth_required": True}
    return JSONResponse(
        status_code=401,
        content={"authenticated": False, "auth_required": True},
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if agent else "not_initialized",
        uptime_sec=round(time.time() - start_time, 1) if start_time else 0,
        model=claude_cfg.get("model", "claude-opus-4-6"),
        read_only=agent_cfg.get("read_only", False),
        tools_available=len(agent._specs_by_name) if agent else 0,
        total_queries=query_count,
        services=SERVICE_URLS,
    )


@app.get("/services/health")
async def services_health():
    """Proxy health checks for all AlgoDesk microservices.

    Returns a consolidated view so the frontend only needs to hit this
    single endpoint instead of reaching each service directly.
    """
    import concurrent.futures

    service_map = {
        "regime_detector": SERVICE_URLS["regime_detector"],
        "dashboard": SERVICE_URLS["dashboard"],
        "fix_api": SERVICE_URLS["fix_api"],
        "data_pipeline": SERVICE_URLS["data_pipeline"],
        "cls": SERVICE_URLS["cls"],
    }

    def _check(name: str, base_url: str) -> tuple[str, dict]:
        try:
            r = _requests.get(f"{base_url}/health", timeout=5)
            if r.ok:
                data = r.json()
                return name, {"status": "ok", "url": base_url, "details": data}
            return name, {"status": "error", "url": base_url, "http_status": r.status_code}
        except _requests.RequestException as exc:
            return name, {"status": "unreachable", "url": base_url, "error": str(exc)}

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_check, name, url): name for name, url in service_map.items()}
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            results[name] = result

    healthy = sum(1 for v in results.values() if v["status"] == "ok")
    total = len(results)

    return {
        "status": "ok" if healthy == total else "degraded" if healthy > 0 else "down",
        "healthy": healthy,
        "total": total,
        "services": results,
        "ts": time.time(),
    }


@app.post("/query", response_model=QueryResponse)
async def query_agent(req: QueryRequest):
    """Send a query to the autonomous trading agent."""
    global query_count
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    logger.info("Query received: %s", req.message[:100])
    try:
        result: AgentResult = agent.query(req.message, req.conversation)
    except Exception as exc:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    query_count += 1
    logger.info(
        "Query completed: rounds=%d, tools=%d, tokens=%d/%d, %.1fs",
        result.rounds,
        len(result.tool_calls_made),
        result.input_tokens,
        result.output_tokens,
        result.elapsed_sec,
    )
    return QueryResponse(
        response=result.response_text,
        tool_calls=result.tool_calls_made,
        rounds=result.rounds,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        elapsed_sec=round(result.elapsed_sec, 2),
    )


@app.post("/query/stream")
async def query_agent_stream(req: QueryRequest):
    """
    Stream the agent's response as newline-delimited JSON events.

    Event types:
      {"type": "text",        "data": "..."}
      {"type": "tool_call",   "data": {"tool": "...", "input": {...}}}
      {"type": "tool_result", "data": {"tool": "...", "preview": "..."}}
      {"type": "done",        "data": {<QueryResponse fields>}}
    """
    global query_count
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    def event_stream():
        nonlocal req
        for event_type, data in agent.query_streaming(req.message, req.conversation):
            if event_type == "done":
                result: AgentResult = data
                payload = {
                    "type": "done",
                    "data": {
                        "response": result.response_text,
                        "tool_calls": result.tool_calls_made,
                        "rounds": result.rounds,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "elapsed_sec": round(result.elapsed_sec, 2),
                    },
                }
            else:
                payload = {"type": event_type, "data": data}
            yield json.dumps(payload, default=str) + "\n"

    query_count += 1
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/tools")
async def list_tools():
    """List all tools available to the agent."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    return [
        {
            "name": spec.name,
            "description": spec.description,
            "read_only": spec.read_only,
            "parameters": spec.properties,
            "required": spec.required,
        }
        for spec in agent._specs_by_name.values()
    ]


@app.get("/config")
async def get_config():
    """Return the current agent configuration (safe subset)."""
    return {
        "model": claude_cfg.get("model", "claude-opus-4-6"),
        "max_tokens": claude_cfg.get("max_tokens", 16000),
        "max_tool_rounds": claude_cfg.get("max_tool_rounds", 20),
        "read_only": agent_cfg.get("read_only", False),
        "services": SERVICE_URLS,
    }


# ── CLS proxy endpoints ──────────────────────────────────────────
# These allow the web UI and external clients to interact with the
# Continuous Learning System through the claude-api gateway.

import requests as _requests

_CLS_TIMEOUT = 15


def _cls_proxy_get(path: str, params: dict | None = None):
    """Forward a GET request to the CLS service."""
    try:
        r = _requests.get(
            f"{SERVICE_URLS['cls']}{path}",
            params=params,
            timeout=_CLS_TIMEOUT,
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except _requests.RequestException as exc:
        return JSONResponse(
            content={"error": str(exc), "service": "cls"},
            status_code=502,
        )


def _cls_proxy_post(path: str, payload: dict | None = None):
    """Forward a POST request to the CLS service."""
    try:
        r = _requests.post(
            f"{SERVICE_URLS['cls']}{path}",
            json=payload,
            timeout=_CLS_TIMEOUT,
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except _requests.RequestException as exc:
        return JSONResponse(
            content={"error": str(exc), "service": "cls"},
            status_code=502,
        )


@app.get("/cls/health")
async def cls_health():
    """CLS service health check."""
    return _cls_proxy_get("/health")


@app.get("/cls/status")
async def cls_status():
    """Full CLS system status."""
    return _cls_proxy_get("/status")


@app.get("/cls/models")
async def cls_models(symbol: str = ""):
    """List registered models."""
    params = {"symbol": symbol} if symbol else {}
    return _cls_proxy_get("/models", params)


@app.get("/cls/models/{symbol}/champion")
async def cls_champion(symbol: str):
    """Get champion model for a symbol."""
    return _cls_proxy_get(f"/models/{symbol}/champion")


@app.get("/cls/performance/{symbol}")
async def cls_performance(symbol: str):
    """Evaluate model performance for a symbol."""
    return _cls_proxy_get(f"/performance/{symbol}")


@app.get("/cls/performance/{symbol}/trend")
async def cls_performance_trend(symbol: str):
    """Get performance trend for a symbol."""
    return _cls_proxy_get(f"/performance/{symbol}/trend")


@app.get("/cls/performance/alerts")
async def cls_alerts():
    """Get active performance alerts."""
    return _cls_proxy_get("/performance/alerts")


@app.get("/cls/drift/{symbol}")
async def cls_drift(symbol: str):
    """Run drift detection for a symbol."""
    return _cls_proxy_get(f"/drift/{symbol}")


@app.get("/cls/drift/{symbol}/history")
async def cls_drift_history(symbol: str):
    """Get drift detection history."""
    return _cls_proxy_get(f"/drift/{symbol}/history")


@app.post("/cls/retrain/{symbol}")
async def cls_retrain(symbol: str, request: Request):
    """Trigger model retraining."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    return _cls_proxy_post(f"/retrain/{symbol}", body)


@app.get("/cls/retrain/status")
async def cls_retrain_status():
    """Get retrain orchestrator status."""
    return _cls_proxy_get("/retrain/status")


@app.get("/cls/retrain/history")
async def cls_retrain_history():
    """Get retraining history."""
    return _cls_proxy_get("/retrain/history")


@app.get("/cls/loop/status")
async def cls_loop_status():
    """Get learning loop status."""
    return _cls_proxy_get("/loop/status")


@app.post("/cls/loop/start")
async def cls_loop_start(request: Request):
    """Start the learning loop."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    return _cls_proxy_post("/loop/start", body)


@app.post("/cls/loop/stop")
async def cls_loop_stop():
    """Stop the learning loop."""
    return _cls_proxy_post("/loop/stop")


@app.post("/cls/loop/tick")
async def cls_loop_tick():
    """Run one learning loop cycle."""
    return _cls_proxy_post("/loop/tick")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = server_cfg.get("port", 5400)
    host = server_cfg.get("host", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
