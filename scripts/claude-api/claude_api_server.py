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
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
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
}

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


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = server_cfg.get("port", 5400)
    host = server_cfg.get("host", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
